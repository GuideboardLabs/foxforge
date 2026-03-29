import re
from datetime import datetime
from pathlib import Path


class ProjectStore:
    def __init__(self, repo_root: Path, user_scope: str = "", root_folder: str = "Projects") -> None:
        self.repo_root = repo_root
        self.user_scope = str(user_scope or "").strip().lower()
        self.root_folder = str(root_folder or "Projects").strip() or "Projects"

    def ensure_project_dirs(self, project_slug: str) -> Path:
        target_slug = project_slug
        if self.user_scope and not str(project_slug).startswith(f"{self.user_scope}__"):
            target_slug = f"{self.user_scope}__{project_slug}"
        root = self.repo_root / self.root_folder / target_slug
        for folder in [
            "brief",
            "research_raw",
            "research_summaries",
            "plan",
            "implementation",
            "qa",
            "deliverables",
        ]:
            (root / folder).mkdir(parents=True, exist_ok=True)
        return root

    @staticmethod
    def _normalize_markdown(content: str) -> str:
        raw = str(content or "")
        # Unwrap full-document markdown fences like ```markdown ... ``` emitted by some models.
        normalized = raw.replace("\r\n", "\n")
        stripped = normalized.strip()
        match = re.match(r"^```([A-Za-z0-9_-]*)\s*\n([\s\S]*?)\n```\s*$", stripped)
        if not match:
            return raw
        lang = (match.group(1) or "").strip().lower()
        if lang and lang not in {"markdown", "md"}:
            return raw
        body = match.group(2).strip("\n")
        return f"{body}\n" if body else ""

    def write_project_file(self, project_slug: str, subfolder: str, filename: str, content: str) -> Path:
        project_root = self.ensure_project_dirs(project_slug)
        path = project_root / subfolder / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() == ".md":
            content = self._normalize_markdown(content)
        path.write_text(content, encoding="utf-8")
        return path

    def timestamped_name(self, prefix: str, ext: str = "md") -> str:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{stamp}_{prefix}.{ext}"
