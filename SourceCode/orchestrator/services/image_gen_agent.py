from __future__ import annotations

import base64
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


_MODEL_CONFIG_KEYS: dict[str, str] = {
    "checkpoint_name": "__CHECKPOINT_NAME__",
    "unet_name": "__UNET_NAME__",
    "clip_l_name": "__CLIP_L_NAME__",
    "clip_g_name": "__CLIP_G_NAME__",
    "t5_name": "__T5_NAME__",
    "vae_name": "__VAE_NAME__",
}


def _normalize_lora_selection(raw: Any) -> list[str]:
    values = raw if isinstance(raw, list) else []
    seen: set[str] = set()
    out: list[str] = []
    for item in values:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text[:220])
        if len(out) >= 32:
            break
    return out


def _inject_lora_chain(
    workflow: dict[str, Any],
    *,
    checkpoint_node_id: str = "1",
    selected_loras: list[str],
    strength_model: float,
    strength_clip: float,
) -> None:
    if not selected_loras:
        return
    if checkpoint_node_id not in workflow:
        return

    base_node_ids = list(workflow.keys())
    max_id = 0
    for node_id in workflow.keys():
        try:
            max_id = max(max_id, int(str(node_id)))
        except Exception:
            continue

    prev_model_ref: list[Any] = [checkpoint_node_id, 0]
    prev_clip_ref: list[Any] = [checkpoint_node_id, 1]
    for lora_name in selected_loras:
        max_id += 1
        lora_node_id = str(max_id)
        workflow[lora_node_id] = {
            "class_type": "LoraLoader",
            "inputs": {
                "model": prev_model_ref,
                "clip": prev_clip_ref,
                "lora_name": lora_name,
                "strength_model": strength_model,
                "strength_clip": strength_clip,
            },
        }
        prev_model_ref = [lora_node_id, 0]
        prev_clip_ref = [lora_node_id, 1]

    for node_id in base_node_ids:
        node = workflow.get(node_id)
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        for input_key, input_value in list(inputs.items()):
            if not isinstance(input_value, list) or len(input_value) != 2:
                continue
            ref_node = str(input_value[0])
            ref_output = int(input_value[1]) if str(input_value[1]).isdigit() else input_value[1]
            if ref_node != checkpoint_node_id:
                continue
            if ref_output == 0:
                inputs[input_key] = prev_model_ref
            elif ref_output == 1:
                inputs[input_key] = prev_clip_ref


def _build_workflow(
    template: dict[str, Any],
    *,
    substitutions: dict[str, Any],
) -> dict[str, Any]:
    """Substitute all __PLACEHOLDER__ values in the workflow template."""
    raw = json.dumps(template)
    for placeholder, value in substitutions.items():
        if isinstance(value, str):
            raw = raw.replace(f'"{placeholder}"', json.dumps(value))
        else:
            raw = raw.replace(f'"{placeholder}"', str(value))
    return json.loads(raw)


def _safe_int(raw: Any, default: int) -> int:
    try:
        return int(raw)
    except (TypeError, ValueError):
        return int(default)


def _safe_float(raw: Any, default: float) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


def _safe_dimension(raw: Any, default: int) -> int:
    value = _safe_int(raw, default)
    value = max(256, min(2048, value))
    value = int(round(value / 8) * 8)
    return max(256, value)


def _apply_ksampler_overrides(
    workflow: dict[str, Any],
    *,
    sampler_name: str = "",
    scheduler: str = "",
) -> None:
    sampler_text = str(sampler_name or "").strip()
    scheduler_text = str(scheduler or "").strip()
    if not sampler_text and not scheduler_text:
        return
    for node in workflow.values():
        if not isinstance(node, dict):
            continue
        if str(node.get("class_type", "")).strip() != "KSampler":
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        if sampler_text:
            inputs["sampler_name"] = sampler_text
        if scheduler_text:
            inputs["scheduler"] = scheduler_text


def _release_ollama_vram(ollama_base_url: str = "http://127.0.0.1:11434") -> None:
    """Ask Ollama to unload all loaded models so VRAM is free for ComfyUI."""
    import urllib.request
    try:
        req = urllib.request.Request(f"{ollama_base_url}/api/ps", method="GET")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        models = [m["name"] for m in data.get("models", []) if m.get("name")]
    except Exception:
        models = []
    for model_name in models:
        try:
            payload = json.dumps({"model": model_name, "keep_alive": 0}).encode("utf-8")
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


_RECREATE_RE = re.compile(r"^recreate\b[^a-z]*$", re.IGNORECASE)
_VISION_MODEL_KEYWORDS = ("moondream", "llava", "llama3.2-vision", "bakllava", "minicpm", "cogvlm", "internvl", "vision")


def _ollama_caption_image(
    image_path: str,
    ollama_base_url: str = "http://127.0.0.1:11434",
) -> str:
    """Try to caption an image using whatever Ollama vision model is available.

    Returns the caption string, or empty string if no vision model is available
    or captioning fails for any reason.
    """
    import urllib.request as _urlreq
    try:
        req = _urlreq.Request(f"{ollama_base_url}/api/tags", method="GET")
        with _urlreq.urlopen(req, timeout=8) as resp:
            tags_data = json.loads(resp.read().decode("utf-8"))
        models = [m.get("name", "") for m in tags_data.get("models", []) if m.get("name")]
    except Exception:
        return ""

    vision_model = next(
        (m for m in models if any(kw in m.lower() for kw in _VISION_MODEL_KEYWORDS)),
        None,
    )
    if not vision_model:
        return ""

    try:
        img_bytes = Path(image_path).read_bytes()
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
    except Exception:
        return ""

    try:
        payload = json.dumps({
            "model": vision_model,
            "prompt": (
                "Describe this image as a Stable Diffusion prompt. "
                "Be specific about subjects, setting, style, colors, mood, and composition. "
                "Use comma-separated descriptive tags. Do not write sentences."
            ),
            "images": [img_b64],
            "stream": False,
        }).encode("utf-8")
        req = _urlreq.Request(
            f"{ollama_base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with _urlreq.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        caption = str(data.get("response", "")).strip()
        return caption
    except Exception:
        return ""


class ImageGenAgent(BaseAgentExecutor):
    capability = AgentCapability(
        lane="image_gen",
        supports_progress=True,
        supports_cancellation=False,
        description="Generates images locally via ComfyUI + SD3.5 Medium GGUF.",
    )

    def run(self, task: AgentTask, tools: ToolRegistry) -> WorkerResult:
        cfg_realistic = lane_model_config(task.repo_root, "image_generation")
        cfg_sd15 = lane_model_config(task.repo_root, "image_generation_sd15")
        cfg_xl = lane_model_config(task.repo_root, "image_generation_xl")
        cfg_xl_standard = lane_model_config(task.repo_root, "image_generation_xl_standard")
        cfg_compose = lane_model_config(task.repo_root, "image_generation_compose")

        positive_prompt: str = str(task.context.get("positive_prompt") or task.prompt).strip()
        negative_prompt: str = str(task.context.get("negative_prompt", "")).strip()
        seed: int = int(task.context.get("seed") or random.randint(0, 2**32 - 1))
        image_style: str = str(task.context.get("image_style", "realistic")).strip().lower() or "realistic"
        selected_loras: list[str] = _normalize_lora_selection(task.context.get("selected_loras", []))
        use_lora_pipeline = bool(selected_loras) or image_style == "lora"

        checkpoint_name_override: str = str(task.context.get("checkpoint_name_override", "")).strip()
        workflow_override: str = str(task.context.get("workflow_override", "")).strip()
        vae_name_override: str = str(task.context.get("vae_name_override", "")).strip()
        model_family_override = str(task.context.get("model_family_override", "")).strip().lower()
        if model_family_override == "sdxl":
            model_family_override = "xl"
        if not model_family_override and workflow_override.strip().lower().startswith("sdxl"):
            model_family_override = "xl"
        use_xl = bool(use_lora_pipeline and model_family_override in {"xl", "xl_standard"})

        if use_lora_pipeline and use_xl:
            if model_family_override == "xl_standard":
                cfg = dict(cfg_xl_standard or cfg_xl or {})
            else:
                cfg = dict(cfg_xl or {})
            if not cfg:
                cfg = {
                    "base_url": cfg_sd15.get("base_url", cfg_realistic.get("base_url", "http://127.0.0.1:8188")),
                    "workflow": "sdxl_standard",
                    "checkpoint_name": "sdxl_base_1.0.safetensors",
                    "steps": 28,
                    "cfg": 5.5,
                    "width": 768,
                    "height": 768,
                    "timeout_sec": 420,
                    "lora_strength_model": 0.8,
                    "lora_strength_clip": 0.8,
                }
        elif use_lora_pipeline:
            cfg = dict(cfg_sd15 or {})
            if not cfg:
                cfg = {
                    "base_url": cfg_compose.get("base_url", cfg_realistic.get("base_url", "http://127.0.0.1:8188")),
                    "workflow": "sd15_standard",
                    "checkpoint_name": cfg_compose.get("checkpoint_name", "v1-5-pruned-emaonly.safetensors"),
                    "steps": cfg_compose.get("steps", 30),
                    "cfg": cfg_compose.get("cfg", 7.0),
                    "width": cfg_compose.get("width", 512),
                    "height": cfg_compose.get("height", 512),
                    "timeout_sec": cfg_compose.get("timeout_sec", 180),
                    "lora_strength_model": 0.8,
                    "lora_strength_clip": 0.8,
                }
        else:
            cfg = dict(cfg_realistic or {})

        base_url: str = str(cfg.get("base_url", "http://127.0.0.1:8188"))
        default_workflow = "sd35_medium_gguf"
        if use_lora_pipeline:
            default_workflow = "sdxl_standard" if use_xl else "sd15_standard"
        workflow_name: str = str(cfg.get("workflow", default_workflow))

        default_steps_value = 28
        default_cfg_value = 4.5
        default_width_value = 768
        default_height_value = 768
        default_timeout_value = 1200
        if use_lora_pipeline and use_xl:
            default_steps_value = 28
            default_cfg_value = 5.5
            default_width_value = 768
            default_height_value = 768
            default_timeout_value = 420
        elif use_lora_pipeline:
            default_steps_value = 30
            default_cfg_value = 7.0
            default_width_value = 512
            default_height_value = 512
            default_timeout_value = 180

        default_steps = _safe_int(cfg.get("steps", default_steps_value), default_steps_value)
        steps: int = _safe_int(task.context.get("steps"), default_steps)
        steps = max(4, min(80, steps))
        cfg_default = _safe_float(cfg.get("cfg", default_cfg_value), default_cfg_value)
        cfg_scale: float = _safe_float(task.context.get("cfg"), cfg_default)
        width: int = _safe_dimension(
            task.context.get("width"),
            _safe_int(cfg.get("width", default_width_value), default_width_value),
        )
        height: int = _safe_dimension(
            task.context.get("height"),
            _safe_int(cfg.get("height", default_height_value), default_height_value),
        )
        timeout: int = int(cfg.get("timeout_sec", default_timeout_value))
        sampler_name: str = str(task.context.get("sampler_name") or cfg.get("sampler_name", "")).strip()
        scheduler: str = str(task.context.get("scheduler") or cfg.get("scheduler", "")).strip()
        lora_strength_model: float = float(task.context.get("lora_strength_model") or cfg.get("lora_strength_model", cfg.get("lora_strength", 0.8)))
        lora_strength_clip: float = float(task.context.get("lora_strength_clip") or cfg.get("lora_strength_clip", cfg.get("lora_strength", 0.8)))

        if checkpoint_name_override:
            cfg["checkpoint_name"] = checkpoint_name_override
        if workflow_override:
            cfg["workflow"] = workflow_override
            workflow_name = workflow_override
        if vae_name_override:
            cfg["vae_name"] = vae_name_override

        # Upgrade to ADetailer workflow for character subjects on SD1.5
        scene_subject: str = str(task.context.get("scene_subject", "")).strip().lower()
        if use_lora_pipeline and not use_xl and scene_subject == "character":
            _adetailer_map = {
                "sd15_standard": "sd15_standard_adetailer",
                "sd15_clipskip2": "sd15_clipskip2_adetailer",
            }
            ad_workflow = _adetailer_map.get(workflow_name)
            if ad_workflow:
                ad_path = task.repo_root / "SourceCode" / "configs" / "comfyui_workflows" / f"{ad_workflow}.json"
                if ad_path.exists():
                    workflow_name = ad_workflow
                    cfg["workflow"] = ad_workflow

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
        time.sleep(3.0)

        if task.progress_callback:
            task.progress_callback("image_gen_generating", {"note": f"Generating image ({steps} steps)…"})

        try:
            template = _load_workflow_template(task.repo_root, workflow_name)
            subs: dict[str, Any] = {
                "__POSITIVE_PROMPT__": positive_prompt,
                "__NEGATIVE_PROMPT__": negative_prompt,
                "__SEED__": seed,
                "__STEPS__": steps,
                "__CFG__": cfg_scale,
                "__WIDTH__": width,
                "__HEIGHT__": height,
            }
            for cfg_key, placeholder in _MODEL_CONFIG_KEYS.items():
                if cfg_key in cfg:
                    subs[placeholder] = str(cfg[cfg_key])
            workflow = _build_workflow(template, substitutions=subs)
            _apply_ksampler_overrides(
                workflow,
                sampler_name=sampler_name,
                scheduler=scheduler,
            )
            if use_lora_pipeline and selected_loras:
                _inject_lora_chain(
                    workflow,
                    checkpoint_node_id="1",
                    selected_loras=selected_loras,
                    strength_model=lora_strength_model,
                    strength_clip=lora_strength_clip,
                )
            bundled_loras: list[dict[str, Any]] = list(task.context.get("bundled_loras") or [])
            for entry in bundled_loras:
                name = str(entry.get("name", "")).strip()
                if not name:
                    continue
                sm = float(entry.get("strength_model", 0.5))
                sc = float(entry.get("strength_clip", 0.5))
                _inject_lora_chain(workflow, checkpoint_node_id="1", selected_loras=[name], strength_model=sm, strength_clip=sc)
            png_bytes = client.generate(workflow, timeout=timeout)
            client.unload_models()
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
            "pipeline": ("sdxl_lora" if (use_lora_pipeline and use_xl) else ("sd15_lora" if use_lora_pipeline else "sd35_realistic")),
            "selected_loras": selected_loras if use_lora_pipeline else [],
            "message": "Image generated.",
        })


class ImageComposeAgent(BaseAgentExecutor):
    capability = AgentCapability(
        lane="image_gen_compose",
        supports_progress=True,
        supports_cancellation=False,
        description="Multi-image composition via ComfyUI + SD1.5 + IP-Adapter Plus.",
    )

    def run(self, task: AgentTask, tools: ToolRegistry) -> WorkerResult:
        model_family_override = str(task.context.get("model_family_override", "")).strip().lower()
        if model_family_override == "sdxl":
            model_family_override = "xl"
        use_xl = model_family_override == "xl"
        cfg = lane_model_config(
            task.repo_root,
            "image_generation_xl_compose" if use_xl else "image_generation_compose",
        )
        default_workflow = "sdxl_ipadapter_compose" if use_xl else "sd15_ipadapter_compose"
        default_width = 768 if use_xl else 512
        default_height = 768 if use_xl else 512
        default_timeout = 900 if use_xl else 180
        base_url: str = str(cfg.get("base_url", "http://127.0.0.1:8188"))
        workflow_name: str = str(cfg.get("workflow", default_workflow))
        default_steps = _safe_int(cfg.get("steps", 30), 30)
        steps: int = _safe_int(task.context.get("steps"), default_steps)
        steps = max(4, min(80, steps))
        cfg_scale: float = _safe_float(task.context.get("cfg"), _safe_float(cfg.get("cfg", 7.0), 7.0))
        ipadapter_weight: float = float(cfg.get("ipadapter_weight", 0.8))
        width: int = _safe_dimension(task.context.get("width"), _safe_int(cfg.get("width", default_width), default_width))
        height: int = _safe_dimension(task.context.get("height"), _safe_int(cfg.get("height", default_height), default_height))
        timeout: int = int(cfg.get("timeout_sec", default_timeout))
        sampler_name: str = str(task.context.get("sampler_name") or cfg.get("sampler_name", "")).strip()
        scheduler: str = str(task.context.get("scheduler") or cfg.get("scheduler", "")).strip()

        positive_prompt: str = str(task.context.get("positive_prompt") or task.prompt).strip()
        negative_prompt: str = str(task.context.get("negative_prompt", "blurry, low quality, deformed")).strip()
        seed: int = int(task.context.get("seed") or random.randint(0, 2**32 - 1))
        selected_loras: list[str] = _normalize_lora_selection(task.context.get("selected_loras", []))
        lora_strength_model: float = float(task.context.get("lora_strength_model") or cfg.get("lora_strength_model", cfg.get("lora_strength", 0.8)))
        lora_strength_clip: float = float(task.context.get("lora_strength_clip") or cfg.get("lora_strength_clip", cfg.get("lora_strength", 0.8)))
        ref_image_paths: list[str] = list(task.context.get("ref_image_paths") or [])

        if not ref_image_paths:
            return WorkerResult.from_legacy("image_gen_compose", {
                "ok": False,
                "error": "No reference images provided.",
                "message": "Please attach 1-3 reference images to use for composition.",
            })

        # Pad to exactly 3 refs (repeat last if fewer than 3)
        while len(ref_image_paths) < 3:
            ref_image_paths.append(ref_image_paths[-1])
        ref_image_paths = ref_image_paths[:3]

        # Auto-caption when prompt is blank or just "recreate"
        is_recreate = not positive_prompt or bool(_RECREATE_RE.match(positive_prompt))
        if is_recreate:
            if task.progress_callback:
                task.progress_callback("image_gen_captioning", {"note": "Analyzing reference image…"})
            caption = _ollama_caption_image(ref_image_paths[0])
            positive_prompt = caption if caption else "recreate this image faithfully, preserve all details, composition, colors, and style"

        if task.progress_callback:
            task.progress_callback("image_gen_started", {"note": f"Composing image from {len(list(task.context.get('ref_image_paths') or []))} reference image(s)."})

        client = ComfyUIClient(base_url)
        if not client.is_available():
            return WorkerResult.from_legacy("image_gen_compose", {
                "ok": False,
                "error": f"ComfyUI is not running at {base_url}.",
                "message": f"ComfyUI is not available at {base_url}.",
            })
        try:
            q = client.queue_info(timeout=5)
            running = q.get("queue_running", []) if isinstance(q, dict) else []
            if isinstance(running, list) and running:
                running_id = str(running[0][1]) if isinstance(running[0], list) and len(running[0]) > 1 else "unknown"
                return WorkerResult.from_legacy("image_gen_compose", {
                    "ok": False,
                    "error": f"ComfyUI has a running job: {running_id}",
                    "message": f"ComfyUI is busy with job {running_id}. Try again in a moment.",
                })
        except Exception:
            pass

        _release_ollama_vram()
        time.sleep(3.0)

        # Upload reference images to ComfyUI's input directory (they live on this
        # machine; ComfyUI may be on a remote machine and can't access local paths)
        try:
            uploaded_refs = [client.upload_image(p) for p in ref_image_paths]
        except Exception as exc:
            return WorkerResult.from_legacy("image_gen_compose", {
                "ok": False,
                "error": str(exc),
                "message": f"Failed to upload reference images to ComfyUI: {exc}",
            })

        if task.progress_callback:
            task.progress_callback("image_gen_generating", {"note": f"Composing image ({steps} steps)…"})

        try:
            template = _load_workflow_template(task.repo_root, workflow_name)
            subs: dict[str, Any] = {
                "__POSITIVE_PROMPT__": positive_prompt,
                "__NEGATIVE_PROMPT__": negative_prompt,
                "__SEED__": seed,
                "__STEPS__": steps,
                "__CFG__": cfg_scale,
                "__IPADAPTER_WEIGHT__": ipadapter_weight,
                "__WIDTH__": width,
                "__HEIGHT__": height,
                "__REF_IMAGE_1__": uploaded_refs[0],
                "__REF_IMAGE_2__": uploaded_refs[1],
                "__REF_IMAGE_3__": uploaded_refs[2],
            }
            for cfg_key, placeholder in _MODEL_CONFIG_KEYS.items():
                if cfg_key in cfg:
                    subs[placeholder] = str(cfg[cfg_key])
            workflow = _build_workflow(template, substitutions=subs)
            _apply_ksampler_overrides(
                workflow,
                sampler_name=sampler_name,
                scheduler=scheduler,
            )
            if selected_loras:
                _inject_lora_chain(
                    workflow,
                    checkpoint_node_id="1",
                    selected_loras=selected_loras,
                    strength_model=lora_strength_model,
                    strength_clip=lora_strength_clip,
                )
            png_bytes = client.generate(workflow, timeout=timeout)
            client.unload_models()
        except Exception as exc:
            return WorkerResult.from_legacy("image_gen_compose", {
                "ok": False,
                "error": str(exc),
                "message": f"Image composition failed: {exc}",
            })

        save_dir: Path | None = None
        raw_save_dir = task.context.get("attach_dir")
        if raw_save_dir:
            save_dir = Path(str(raw_save_dir))
        if save_dir is None:
            save_dir = task.repo_root / "Runtime" / "images" / "generated"

        save_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{ts}_compose_{seed % 100000:05d}.png"
        save_path = save_dir / filename
        save_path.write_bytes(png_bytes)

        conversation_id: str = str(task.context.get("conversation_id", "")).strip()
        url = f"/api/conversations/{conversation_id}/attachments/{filename}" if conversation_id else ""

        if task.progress_callback:
            task.progress_callback("image_gen_done", {"note": "Composition saved."})

        return WorkerResult.from_legacy("image_gen_compose", {
            "ok": True,
            "filename": filename,
            "save_path": str(save_path),
            "url": url,
            "seed": seed,
            "steps": steps,
            "selected_loras": selected_loras,
            "message": "Composition generated.",
        })
