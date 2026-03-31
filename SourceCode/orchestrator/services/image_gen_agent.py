from __future__ import annotations

import json
import os
import random
import re
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

from infra.tools import ToolRegistry
from shared_tools.comfyui_client import ComfyUIClient
from shared_tools.model_routing import lane_model_config
from .agent_contracts import AgentCapability, AgentTask, BaseAgentExecutor
from .result_types import WorkerResult


def _load_workflow_template(repo_root: Path, workflow_name: str) -> dict[str, Any]:
    path = repo_root / "SourceCode" / "configs" / "comfyui_workflows" / f"{workflow_name}.json"
    if not path.exists():
        raise FileNotFoundError(f"ComfyUI workflow template not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw.pop("_comment", None)
    return raw


def _build_workflow(
    template: dict[str, Any],
    *,
    positive_prompt: str,
    negative_prompt: str,
    seed: int,
    steps: int,
    width: int,
    height: int,
    unet_name: str,
    clip_l_name: str,
    t5_name: str,
    vae_name: str,
) -> dict[str, Any]:
    """Substitute all __PLACEHOLDER__ values in the workflow template."""
    raw = json.dumps(template)
    substitutions = {
        "__POSITIVE_PROMPT__": positive_prompt,
        "__NEGATIVE_PROMPT__": negative_prompt,
        "__SEED__": seed,
        "__STEPS__": steps,
        "__WIDTH__": width,
        "__HEIGHT__": height,
        "__UNET_NAME__": unet_name,
        "__CLIP_L_NAME__": clip_l_name,
        "__T5_NAME__": t5_name,
        "__VAE_NAME__": vae_name,
    }
    for placeholder, value in substitutions.items():
        if isinstance(value, str):
            raw = raw.replace(f'"{placeholder}"', json.dumps(value))
        else:
            raw = raw.replace(f'"{placeholder}"', str(value))
    return json.loads(raw)


def _release_ollama_vram(ollama_base_url: str = "http://127.0.0.1:11434") -> None:
    """Ask Ollama to unload models so VRAM is free for ComfyUI."""
    import urllib.request
    import urllib.error
    try:
        payload = json.dumps({"model": "", "keep_alive": 0}).encode("utf-8")
        req = urllib.request.Request(
            f"{ollama_base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=10)
    except Exception:
        pass


def _restart_comfyui(repo_root: Path) -> bool:
    """Best-effort ComfyUI restart when a stuck run cannot be interrupted."""
    try:
        proc = subprocess.run(
            ["pgrep", "-f", "/home/sc/ComfyUI/main.py"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        pids: list[int] = []
        for line in str(proc.stdout or "").splitlines():
            line = line.strip()
            if not line.isdigit():
                continue
            pid = int(line)
            if pid != os.getpid():
                pids.append(pid)
        for pid in pids:
            try:
                os.kill(pid, signal.SIGTERM)
            except Exception:
                pass
        time.sleep(1.5)
        log_dir = repo_root / "Runtime" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        cmd = f"cd {repo_root} && nohup ./start_comfyui.sh > Runtime/logs/comfyui.log 2>&1 &"
        subprocess.Popen(["bash", "-lc", cmd], cwd=str(repo_root))
        return True
    except Exception:
        return False


class ImageGenAgent(BaseAgentExecutor):
    capability = AgentCapability(
        lane="image_gen",
        supports_progress=True,
        supports_cancellation=False,
        description="Generates images locally via ComfyUI + FLUX.1 Schnell GGUF.",
    )

    def run(self, task: AgentTask, tools: ToolRegistry) -> WorkerResult:
        cfg = lane_model_config(task.repo_root, "image_generation")
        base_url: str = str(cfg.get("base_url", "http://127.0.0.1:8188"))
        workflow_name: str = str(cfg.get("workflow", "flux_schnell_gguf"))
        steps: int = int(cfg.get("steps", 4))
        width: int = int(cfg.get("width", 512))
        height: int = int(cfg.get("height", 512))
        timeout: int = int(cfg.get("timeout_sec", 60))
        unet_name: str = str(cfg.get("unet_name", "flux1-schnell-q5_k_s.gguf"))
        clip_l_name: str = str(cfg.get("clip_l_name", "clip_l.safetensors"))
        t5_name: str = str(cfg.get("t5_name", "t5xxl_fp8_e4m3fn.safetensors"))
        vae_name: str = str(cfg.get("vae_name", "ae.safetensors"))

        positive_prompt: str = str(task.context.get("positive_prompt") or task.prompt).strip()
        negative_prompt: str = str(task.context.get("negative_prompt", "")).strip()
        seed: int = int(task.context.get("seed") or random.randint(0, 2**32 - 1))

        if task.progress_callback:
            task.progress_callback("image_gen_started", {"note": "Preparing ComfyUI for image generation."})

        client = ComfyUIClient(base_url)
        if not client.is_available():
            return WorkerResult.from_legacy("image_gen", {
                "ok": False,
                "error": f"ComfyUI is not running at {base_url}. Start it with: python main.py --listen",
                "message": f"ComfyUI is not available at {base_url}.",
            })
        try:
            q = client.queue_info(timeout=5)
            running = q.get("queue_running", []) if isinstance(q, dict) else []
            if isinstance(running, list) and running:
                running_id = str(running[0][1]) if isinstance(running[0], list) and len(running[0]) > 1 else "unknown"
                return WorkerResult.from_legacy("image_gen", {
                    "ok": False,
                    "error": f"ComfyUI has a running job: {running_id}",
                    "message": (
                        f"Image generation is currently blocked because ComfyUI is still running job {running_id}. "
                        "This usually means ComfyUI got stuck. Restart ComfyUI and try again."
                    ),
                })
        except Exception:
            pass

        # Release Ollama VRAM before loading image model
        _release_ollama_vram()
        time.sleep(1.0)

        if task.progress_callback:
            task.progress_callback("image_gen_generating", {"note": f"Generating image ({steps} steps)…"})

        try:
            template = _load_workflow_template(task.repo_root, workflow_name)
            workflow = _build_workflow(
                template,
                positive_prompt=positive_prompt,
                negative_prompt=negative_prompt,
                seed=seed,
                steps=steps,
                width=width,
                height=height,
                unet_name=unet_name,
                clip_l_name=clip_l_name,
                t5_name=t5_name,
                vae_name=vae_name,
            )
            png_bytes = client.generate(workflow, timeout=timeout)
        except Exception as exc:
            extra = ""
            low = str(exc).lower()
            if "timed out" in low:
                try:
                    q = client.queue_info(timeout=5)
                    running = q.get("queue_running", []) if isinstance(q, dict) else []
                    if isinstance(running, list) and running:
                        running_id = str(running[0][1]) if isinstance(running[0], list) and len(running[0]) > 1 else "unknown"
                        restarted = _restart_comfyui(task.repo_root)
                        restart_note = " Auto-restart was attempted." if restarted else ""
                        extra = (
                            f" ComfyUI still reports a running job ({running_id}), which indicates it is stuck. "
                            f"Restart ComfyUI and try again.{restart_note}"
                        )
                except Exception:
                    pass
            return WorkerResult.from_legacy("image_gen", {
                "ok": False,
                "error": str(exc),
                "message": f"Image generation failed: {exc}{extra}",
            })

        # Save PNG to the conversation's attachment directory
        save_dir: Path | None = None
        raw_save_dir = task.context.get("attach_dir")
        if raw_save_dir:
            save_dir = Path(str(raw_save_dir))

        if save_dir is None:
            save_dir = task.repo_root / "Runtime" / "images" / "generated"

        save_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{ts}_gen_{seed % 100000:05d}.png"
        save_path = save_dir / filename
        save_path.write_bytes(png_bytes)

        conversation_id: str = str(task.context.get("conversation_id", "")).strip()
        url = f"/api/conversations/{conversation_id}/attachments/{filename}" if conversation_id else ""

        if task.progress_callback:
            task.progress_callback("image_gen_done", {"note": "Image saved."})

        return WorkerResult.from_legacy("image_gen", {
            "ok": True,
            "filename": filename,
            "save_path": str(save_path),
            "url": url,
            "seed": seed,
            "steps": steps,
            "message": "Image generated.",
        })
