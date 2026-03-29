def draft_vanilla_js_frontend(task: str) -> str:
    return "\n".join(
        [
            "## Vanilla JS Frontend Draft",
            f"- Objective: {task}",
            "- Progressive enhancement architecture.",
            "- Component-like modules without framework lock-in.",
            "- Accessibility-first interactions and keyboard support.",
        ]
    )
