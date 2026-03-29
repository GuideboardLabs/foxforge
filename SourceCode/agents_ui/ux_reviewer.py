def ux_review(task: str) -> str:
    return "\n".join(
        [
            "## UX Review",
            f"- Focus area: {task}",
            "- Ensure high signal visual hierarchy.",
            "- Confirm mobile-first flow and responsive breakpoints.",
            "- Add acceptance criteria for usability and clarity.",
        ]
    )
