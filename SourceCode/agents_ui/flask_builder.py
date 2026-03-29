def draft_flask_backend(task: str) -> str:
    return "\n".join(
        [
            "## Flask Backend Draft",
            f"- Objective: {task}",
            "- App factory pattern with blueprints.",
            "- API endpoints for core workflow states.",
            "- Server-side validation and logging hooks.",
        ]
    )
