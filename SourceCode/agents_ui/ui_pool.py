from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from agents_ui.flask_builder import draft_flask_backend
from agents_ui.ux_reviewer import ux_review
from agents_ui.vanilla_js_builder import draft_vanilla_js_frontend
from shared_tools.feedback_learning import FeedbackLearningEngine
from shared_tools.file_store import ProjectStore
from shared_tools.model_routing import lane_model_config
from shared_tools.ollama_client import OllamaClient


UI_PERSONAS = [
    ("backend_architect", "Design Flask architecture, routes, validation, and deployment shape."),
    ("frontend_architect", "Design high-end vanilla JS UI structure, interactions, and state flow."),
    ("ux_reviewer", "Critique UX and produce acceptance criteria and test checklist."),
]


def _run_ui_agent(
    client: OllamaClient,
    model_cfg: dict[str, Any],
    task: str,
    persona: str,
    directive: str,
    learned_guidance: str,
) -> tuple[str, str]:
    model = model_cfg.get("model", "")
    if not model:
        return persona, ""

    system_prompt = (
        "You are a specialized web implementation agent in a multi-agent team. "
        f"Role: {persona}. {directive} "
        "Output markdown only. Be implementation-focused."
    )
    if learned_guidance:
        system_prompt = f"{system_prompt}\n\n{learned_guidance}"
    user_prompt = (
        f"Task:\n{task}\n\n"
        "Target stack: Python Flask backend + vanilla JavaScript frontend.\n"
        "Produce concrete technical output."
    )
    try:
        content = client.chat(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=float(model_cfg.get("temperature", 0.3)),
            num_ctx=int(model_cfg.get("num_ctx", 12288)),
            think=bool(model_cfg.get("think", False)),
            timeout=int(model_cfg.get("timeout_sec", 300)),
        )
    except Exception as exc:
        content = f"Model call failed for {persona}: {exc}"
    return persona, content


def run_ui_pool(task: str, repo_root: Path, project_slug: str, bus) -> dict:
    bus.emit("ui_pool", "start", {"task": task, "project": project_slug})

    model_cfg = lane_model_config(repo_root, "ui_pool")
    orchestrator_cfg = lane_model_config(repo_root, "orchestrator_reasoning")
    client = OllamaClient()
    learning = FeedbackLearningEngine(repo_root, client=client, model_cfg=orchestrator_cfg)
    learned_guidance = learning.guidance_for_lane("ui", limit=5)
    worker_count = max(1, min(int(model_cfg.get("parallel_agents", 3)), len(UI_PERSONAS)))
    personas = UI_PERSONAS[:worker_count]

    outputs: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(_run_ui_agent, client, model_cfg, task, persona, directive, learned_guidance)
            for persona, directive in personas
        ]
        for future in futures:
            persona, content = future.result()
            outputs[persona] = content

    backend = outputs.get("backend_architect") or draft_flask_backend(task)
    frontend = outputs.get("frontend_architect") or draft_vanilla_js_frontend(task)
    review = outputs.get("ux_reviewer") or ux_review(task)

    spec = "\n\n".join(["# UI Pool Output", backend, frontend, review]) + "\n"

    store = ProjectStore(repo_root)
    name = store.timestamped_name("ui_spec")
    path = store.write_project_file(project_slug, "implementation", name, spec)

    bus.emit(
        "ui_pool",
        "completed",
        {
            "project": project_slug,
            "spec_path": str(path),
            "model": model_cfg.get("model", ""),
            "workers": worker_count,
        },
    )
    return {"message": "UI pool produced a Flask + vanilla JS implementation spec.", "path": str(path)}
