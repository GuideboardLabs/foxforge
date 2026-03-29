from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ActionExecutor:
    """Executes approved action proposals. Sandboxed to safe write paths."""

    SAFE_WRITE_PREFIXES = ("Projects/", "Runtime/notes/")

    def execute(
        self,
        action_type: str,
        payload: dict[str, Any],
        repo_root: Path,
    ) -> dict[str, Any]:
        """Execute one approved action. Returns {"ok": bool, "message": str}."""
        kind = str(action_type or "").strip().lower()
        try:
            if kind == "write_file":
                return self._write_file(payload, repo_root)
            if kind == "apply_patch":
                return self._apply_patch(payload, repo_root)
            if kind == "apply_patch_batch":
                return self._apply_patch_batch(payload, repo_root)
            if kind == "append_note":
                return self._append_note(payload, repo_root)
            return {"ok": False, "message": f"Unknown action type: {kind}"}
        except Exception as exc:
            return {"ok": False, "message": f"Action failed: {exc}"}

    # ------------------------------------------------------------------
    # Action implementations
    # ------------------------------------------------------------------

    def _write_file(self, payload: dict, repo_root: Path) -> dict[str, Any]:
        raw_path = str(payload.get("path", "")).strip()
        content = str(payload.get("content", ""))
        if not raw_path:
            return {"ok": False, "message": "write_file: path is required."}
        if not any(raw_path.startswith(p) for p in self.SAFE_WRITE_PREFIXES):
            return {
                "ok": False,
                "message": (
                    f"write_file: path '{raw_path}' is outside safe write prefixes "
                    f"({', '.join(self.SAFE_WRITE_PREFIXES)})."
                ),
            }
        # Prevent path traversal
        target = (repo_root / raw_path).resolve()
        if not str(target).startswith(str(repo_root.resolve())):
            return {"ok": False, "message": "write_file: path traversal blocked."}
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return {"ok": True, "message": f"File written: {target}"}


    def _apply_patch(self, payload: dict, repo_root: Path) -> dict[str, Any]:
        import hashlib

        raw_path = str(payload.get("path", "")).strip()
        new_content = str(payload.get("new_content", ""))
        expected_hash = str(payload.get("original_sha256", "")).strip()
        if not raw_path:
            return {"ok": False, "message": "apply_patch: path is required."}
        if not any(raw_path.startswith(p) for p in self.SAFE_WRITE_PREFIXES):
            return {"ok": False, "message": f"apply_patch: path '{raw_path}' is outside safe write prefixes."}
        target = (repo_root / raw_path).resolve()
        if not str(target).startswith(str(repo_root.resolve())):
            return {"ok": False, "message": "apply_patch: path traversal blocked."}
        old_text = ""
        if target.exists():
            try:
                old_text = target.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                old_text = target.read_text(encoding="utf-8-sig")
            current_hash = hashlib.sha256(old_text.encode("utf-8")).hexdigest()
            if expected_hash and current_hash != expected_hash:
                return {
                    "ok": False,
                    "message": "apply_patch: file changed since proposal was created; regenerate the patch.",
                }
        elif expected_hash:
            return {"ok": False, "message": "apply_patch: target file is missing; regenerate the patch."}
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(new_content, encoding="utf-8")
        summary = str(payload.get("summary", "")).strip()
        return {"ok": True, "message": f"Patch applied: {target.name}{' | ' + summary if summary else ''}"}

    def _apply_patch_batch(self, payload: dict, repo_root: Path) -> dict[str, Any]:
        files = payload.get("files", []) if isinstance(payload.get("files"), list) else []
        if not files:
            return {"ok": False, "message": "apply_patch_batch: files are required."}
        applied: list[str] = []
        for item in files:
            if not isinstance(item, dict):
                return {"ok": False, "message": "apply_patch_batch: invalid file payload."}
            result = self._apply_patch(item, repo_root)
            if not result.get("ok"):
                return {"ok": False, "message": f"apply_patch_batch halted: {result.get('message', 'unknown error')}"}
            path_text = str(item.get("path", "")).strip()
            if path_text:
                applied.append(path_text)
        summary = str(payload.get("summary", "")).strip()
        suffix = f" | {summary}" if summary else ""
        return {"ok": True, "message": f"Batch patch applied: {len(applied)} files{suffix}"}

    def _append_note(self, payload: dict, repo_root: Path) -> dict[str, Any]:
        content = str(payload.get("content", "")).strip()
        if not content:
            return {"ok": False, "message": "append_note: content is required."}
        date_str = str(payload.get("date", "")).strip()
        if not date_str:
            date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        notes_dir = repo_root / "Runtime" / "notes"
        notes_dir.mkdir(parents=True, exist_ok=True)
        note_path = notes_dir / f"{date_str}.md"
        ts = datetime.now(timezone.utc).strftime("%H:%M UTC")
        entry = f"\n## {ts}\n\n{content}\n"
        with note_path.open("a", encoding="utf-8") as fh:
            fh.write(entry)
        return {"ok": True, "message": f"Note appended to {note_path.name}"}
