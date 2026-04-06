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


def _build_workflow(
    template: dict[str, Any],
    *,
    substitutions: dict[str, Any],
) -> dict[str, Any]:
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


def _motion_bucket_from_selection(num_frames: int) -> int:
    """Map the UI intensity selection (repurposed num_frames field) to SVD motion_bucket_id.

    The video tool modal sends:
      49  → Subtle  (gentle motion)
      81  → Normal  (default)
      121 → Dynamic (strong motion)
    """
    if num_frames <= 60:
        return 64   # subtle
    if num_frames <= 100:
        return 127  # normal
    return 200      # dynamic


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


class StableVideoAgent(BaseAgentExecutor):
    """Animates a reference image using Stable Video Diffusion XT 1.1.

    SVD is text-free: the motion_prompt is mapped to a motion_bucket_id
    (subtle / normal / dynamic) via the UI selection passed as num_frames.
    Outputs 25 frames at the configured fps (~3 seconds).
    """

    capability = AgentCapability(
        lane="video_gen",
        supports_progress=True,
        supports_cancellation=False,
        description="Generates video from a reference image via ComfyUI + Stable Video Diffusion XT 1.1.",
    )

    def run(self, task: AgentTask, tools: ToolRegistry) -> WorkerResult:
        cfg = lane_model_config(task.repo_root, "video_generation_svd")
        if not cfg:
            return WorkerResult.from_legacy("video_gen", {
                "ok": False,
                "error": "video_generation_svd lane not configured",
                "message": "video_generation_svd lane not configured in model_routing.json.",
            })

        base_url: str = str(cfg.get("base_url", "http://127.0.0.1:8188"))
        workflow_name: str = str(cfg.get("workflow", "svd_xt_i2v"))
        checkpoint_name: str = str(cfg.get("checkpoint_name", "svd_xt_1_1.safetensors"))
        default_steps: int = _safe_int(cfg.get("steps", 20), 20)
        default_cfg: float = _safe_float(cfg.get("cfg", 2.5), 2.5)
        default_width: int = _safe_int(cfg.get("width", 1024), 1024)
        default_height: int = _safe_int(cfg.get("height", 576), 576)
        default_fps: int = _safe_int(cfg.get("fps", 8), 8)
        default_motion_bucket: int = _safe_int(cfg.get("motion_bucket_id", 127), 127)
        timeout: int = _safe_int(cfg.get("timeout_sec", 600), 600)

        ref_image_path: str = str(task.context.get("ref_image_path", "")).strip()
        seed: int = int(task.context.get("seed") or random.randint(0, 2**32 - 1))
        steps: int = max(4, min(50, _safe_int(task.context.get("steps"), default_steps)))
        cfg_scale: float = _safe_float(task.context.get("cfg"), default_cfg)
        width: int = _safe_int(task.context.get("width") or default_width, default_width)
        height: int = _safe_int(task.context.get("height") or default_height, default_height)
        conversation_id: str = str(task.context.get("conversation_id", "")).strip()

        # num_frames repurposed as intensity selector: 49=subtle, 81=normal, 121=dynamic
        num_frames_raw: int = _safe_int(task.context.get("num_frames"), 81)
        motion_bucket_id: int = _motion_bucket_from_selection(num_frames_raw)

        if not ref_image_path:
            return WorkerResult.from_legacy("video_gen", {
                "ok": False,
                "error": "ref_image_path is required",
                "message": "No reference image path provided.",
            })

        if not Path(ref_image_path).exists():
            return WorkerResult.from_legacy("video_gen", {
                "ok": False,
                "error": f"Reference image not found: {ref_image_path}",
                "message": f"Reference image file not found: {ref_image_path}",
            })

        if task.progress_callback:
            task.progress_callback("video_gen_init", {"note": "Releasing VRAM and uploading reference image."})

        _release_ollama_vram()

        client = ComfyUIClient(base_url)

        try:
            uploaded_ref = client.upload_image(ref_image_path)
        except Exception as exc:
            return WorkerResult.from_legacy("video_gen", {
                "ok": False,
                "error": str(exc),
                "message": f"Failed to upload reference image: {exc}",
            })

        try:
            template = _load_workflow_template(task.repo_root, workflow_name)
        except Exception as exc:
            return WorkerResult.from_legacy("video_gen", {
                "ok": False,
                "error": str(exc),
                "message": f"Failed to load workflow template: {exc}",
            })

        workflow = _build_workflow(
            template,
            substitutions={
                "__CHECKPOINT_NAME__": checkpoint_name,
                "__REF_IMAGE__": uploaded_ref,
                "__WIDTH__": width,
                "__HEIGHT__": height,
                "__MOTION_BUCKET_ID__": motion_bucket_id,
                "__FPS__": default_fps,
                "__SEED__": seed,
                "__STEPS__": steps,
                "__CFG__": cfg_scale,
            },
        )

        if task.progress_callback:
            task.progress_callback("video_gen_started", {
                "note": f"Queued in ComfyUI. SVD XT — motion bucket {motion_bucket_id}, {steps} steps.",
            })

        try:
            video_bytes = client.generate_video(workflow, timeout=timeout)
        except Exception as exc:
            return WorkerResult.from_legacy("video_gen", {
                "ok": False,
                "error": str(exc),
                "message": f"Video generation failed: {exc}",
            })

        save_dir: Path | None = None
        raw_save_dir = task.context.get("attach_dir")
        if raw_save_dir:
            save_dir = Path(str(raw_save_dir))
        if save_dir is None:
            save_dir = task.repo_root / "Runtime" / "videos" / "generated"

        save_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{ts}_video_{seed % 100000:05d}.mp4"
        save_path = save_dir / filename
        save_path.write_bytes(video_bytes)

        url = f"/api/conversations/{conversation_id}/attachments/{filename}" if conversation_id else ""

        if task.progress_callback:
            task.progress_callback("video_gen_done", {"note": "Video saved."})

        return WorkerResult.from_legacy("video_gen", {
            "ok": True,
            "filename": filename,
            "save_path": str(save_path),
            "url": url,
            "seed": seed,
            "steps": steps,
            "num_frames": 25,
            "motion_bucket_id": motion_bucket_id,
        })
