from __future__ import annotations

import json
import random
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


def _build_workflow(template: dict[str, Any], *, substitutions: dict[str, Any]) -> dict[str, Any]:
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


def _release_ollama_vram(ollama_base_url: str = "http://127.0.0.1:11434") -> None:
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


class ImageEnhanceAgent(BaseAgentExecutor):
    """Post-process an existing image with a detail-enhancing LoRA via SDXL img2img."""

    capability = AgentCapability(
        lane="image_enhance",
        supports_progress=True,
        supports_cancellation=False,
        description="Enhances background detail in an existing XL image via img2img + zy_Detailed_Backgrounds LoRA.",
    )

    def run(self, task: AgentTask, tools: ToolRegistry) -> WorkerResult:
        cfg = lane_model_config(task.repo_root, "image_enhancement_xl")
        if not cfg:
            return WorkerResult.from_legacy("image_enhance", {
                "ok": False,
                "error": "image_enhancement_xl lane not configured",
                "message": "image_enhancement_xl lane not configured in model_routing.json.",
            })

        base_url: str = str(cfg.get("base_url", "http://127.0.0.1:8188"))
        workflow_name: str = str(cfg.get("workflow", "sdxl_img2img_lora"))
        checkpoint_name: str = str(cfg.get("checkpoint_name", ""))
        vae_name: str = str(cfg.get("vae_name", ""))
        lora_name: str = str(cfg.get("lora_name", ""))
        lora_strength: float = _safe_float(cfg.get("lora_strength", 0.7), 0.7)
        steps: int = _safe_int(cfg.get("steps", 20), 20)
        cfg_scale: float = _safe_float(cfg.get("cfg", 5.0), 5.0)
        denoise: float = _safe_float(cfg.get("denoise", 0.38), 0.38)
        timeout: int = _safe_int(cfg.get("timeout_sec", 360), 360)

        ref_image_path: str = str(task.context.get("ref_image_path", "")).strip()
        seed: int = int(task.context.get("seed") or random.randint(0, 2**32 - 1))
        conversation_id: str = str(task.context.get("conversation_id", "")).strip()

        if not ref_image_path or not Path(ref_image_path).exists():
            return WorkerResult.from_legacy("image_enhance", {
                "ok": False,
                "error": f"Source image not found: {ref_image_path}",
                "message": "Source image file not found.",
            })

        if task.progress_callback:
            task.progress_callback("enhance_init", {"note": "Uploading image for enhancement."})

        _release_ollama_vram()
        client = ComfyUIClient(base_url)

        try:
            uploaded_ref = client.upload_image(ref_image_path)
        except Exception as exc:
            return WorkerResult.from_legacy("image_enhance", {
                "ok": False,
                "error": str(exc),
                "message": f"Failed to upload image: {exc}",
            })

        try:
            template = _load_workflow_template(task.repo_root, workflow_name)
        except Exception as exc:
            return WorkerResult.from_legacy("image_enhance", {
                "ok": False,
                "error": str(exc),
                "message": f"Failed to load workflow template: {exc}",
            })

        # Minimal positive prompt — the LoRA drives the enhancement, not the text
        positive_prompt = "score_9, score_8_up, score_7_up, detailed background, intricate scenery"
        negative_prompt = "score_4, score_5, score_6, blurry, low quality, flat, empty background"

        workflow = _build_workflow(
            template,
            substitutions={
                "__CHECKPOINT_NAME__": checkpoint_name,
                "__VAE_NAME__": vae_name,
                "__LORA_NAME__": lora_name,
                "__LORA_STRENGTH__": lora_strength,
                "__REF_IMAGE__": uploaded_ref,
                "__POSITIVE_PROMPT__": positive_prompt,
                "__NEGATIVE_PROMPT__": negative_prompt,
                "__SEED__": seed,
                "__STEPS__": steps,
                "__CFG__": cfg_scale,
                "__DENOISE__": denoise,
            },
        )

        if task.progress_callback:
            task.progress_callback("enhance_started", {"note": "BG+ enhancement running."})

        try:
            png_bytes = client.generate(workflow, timeout=timeout)
        except Exception as exc:
            return WorkerResult.from_legacy("image_enhance", {
                "ok": False,
                "error": str(exc),
                "message": f"Enhancement failed: {exc}",
            })

        save_dir: Path | None = None
        raw_save_dir = task.context.get("attach_dir")
        if raw_save_dir:
            save_dir = Path(str(raw_save_dir))
        if save_dir is None:
            save_dir = task.repo_root / "Runtime" / "images" / "enhanced"

        save_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{ts}_enhanced_{seed % 100000:05d}.png"
        save_path = save_dir / filename
        save_path.write_bytes(png_bytes)

        url = f"/api/conversations/{conversation_id}/attachments/{filename}" if conversation_id else ""

        if task.progress_callback:
            task.progress_callback("enhance_done", {"note": "Enhanced image saved."})

        return WorkerResult.from_legacy("image_enhance", {
            "ok": True,
            "filename": filename,
            "save_path": str(save_path),
            "url": url,
            "seed": seed,
        })
