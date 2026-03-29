from __future__ import annotations

import difflib
import fnmatch
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from shared_tools.db import connect, transaction
from shared_tools.migrations import initialize_database


LOGGER = logging.getLogger(__name__)


class WorkspaceTools:
    """Safe, project-scoped workspace helpers for repo reading and patch proposals."""

    SAFE_REPO_PREFIXES = ("Projects/",)
    SKIP_DIR_NAMES = {
        ".git",
        "__pycache__",
        ".venv",
        "venv",
        "node_modules",
        "dist",
        "build",
        ".pytest_cache",
        ".mypy_cache",
    }
    TEXT_SUFFIX_ALLOWLIST = {
        ".py", ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg",
        ".js", ".ts", ".tsx", ".jsx", ".html", ".css", ".scss", ".sql", ".sh", ".ps1",
        ".java", ".kt", ".go", ".rs", ".cpp", ".c", ".h", ".hpp", ".cs", ".php", ".rb",
        ".xml", ".csv",
    }

    def __init__(self, repo_root: Path, *, client: Any | None = None, model_cfg: dict[str, Any] | None = None) -> None:
        self.repo_root = Path(repo_root)
        self.client = client
        self.model_cfg = dict(model_cfg or {})
        initialize_database(self.repo_root)

    def workspace_root(self, project_slug: str) -> Path:
        slug = str(project_slug or "").strip().replace(" ", "_") or "general"
        root = (self.repo_root / "Projects" / slug).resolve()
        root.mkdir(parents=True, exist_ok=True)
        return root

    def repo_relative_path(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.repo_root.resolve()).as_posix()
        except Exception as exc:
            raise ValueError(f"Path is outside repository root: {path}") from exc

    def resolve_path(self, project_slug: str, rel_path: str) -> Path:
        root = self.workspace_root(project_slug)
        raw = str(rel_path or ".").strip() or "."
        target = (root / raw).resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Path escapes the active project workspace: {rel_path}") from exc
        return target

    def tree_text(self, project_slug: str, rel_path: str = ".", *, max_depth: int = 2, max_entries: int = 200) -> str:
        root = self.resolve_path(project_slug, rel_path)
        if not root.exists():
            return f"Workspace path not found: {rel_path}"
        lines = [f"Workspace tree for {self.repo_relative_path(root)}:"]
        entries_seen = 0

        def walk(path: Path, depth: int) -> None:
            nonlocal entries_seen
            if entries_seen >= max_entries:
                return
            try:
                children = sorted(path.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower()))
            except OSError:
                return
            for child in children:
                if entries_seen >= max_entries:
                    return
                if child.name in self.SKIP_DIR_NAMES:
                    continue
                indent = "  " * depth
                suffix = "/" if child.is_dir() else ""
                lines.append(f"{indent}- {child.name}{suffix}")
                entries_seen += 1
                if child.is_dir() and depth < max_depth:
                    walk(child, depth + 1)

        if root.is_file():
            return f"Workspace path is a file: {self.repo_relative_path(root)}"
        walk(root, 0)
        if entries_seen >= max_entries:
            lines.append("... truncated ...")
        self._log_action(project_slug=project_slug, action_kind="tree", path_text=self.repo_relative_path(root), status="ok")
        return "\n".join(lines)

    def read_text(self, project_slug: str, rel_path: str, *, max_chars: int = 12000) -> str:
        path = self.resolve_path(project_slug, rel_path)
        if not path.exists():
            return f"Workspace file not found: {rel_path}"
        if path.is_dir():
            return f"Workspace path is a directory, not a file: {rel_path}"
        existing = self._read_existing_text(path)
        if existing is None:
            return f"Could not read text from file: {rel_path}"
        preview = existing[: max(500, min(max_chars, 30000))]
        self._log_action(project_slug=project_slug, action_kind="read", path_text=self.repo_relative_path(path), status="ok")
        return f"File: {self.repo_relative_path(path)}\n\n{preview}"

    def search_text(self, project_slug: str, query: str, *, rel_glob: str = "*", limit: int = 20, max_file_bytes: int = 200_000) -> str:
        root = self.workspace_root(project_slug)
        needle = str(query or "").strip()
        if not needle:
            return "Search query is required."

        result_limit = max(1, min(limit, 100))
        needle_lower = needle.lower()
        results: list[str] = []
        scanned_files = 0
        skipped_large_files = 0

        for path in self._iter_candidate_files(root, rel_glob=rel_glob):
            if len(results) >= result_limit:
                break
            try:
                if path.stat().st_size > max_file_bytes:
                    skipped_large_files += 1
                    continue
            except OSError as exc:
                LOGGER.warning("Skipping workspace file with unreadable metadata %s: %s", path, exc)
                continue

            scanned_files += 1
            try:
                for line_no, excerpt in self._iter_line_matches(path, needle_lower):
                    results.append(f"- {self.repo_relative_path(path)}:{line_no} | {excerpt}")
                    if len(results) >= result_limit:
                        break
            except OSError as exc:
                LOGGER.warning("Skipping unreadable workspace file %s: %s", path, exc)
            except UnicodeDecodeError:
                LOGGER.warning("Skipping non-text workspace file during search: %s", path)

        self._log_action(project_slug=project_slug, action_kind="search", path_text=rel_glob or "*", status="ok", detail=needle)
        if not results:
            details = f" Scanned {scanned_files} file(s)."
            if skipped_large_files:
                details += f" Skipped {skipped_large_files} oversized file(s)."
            return f"No matches found in workspace for: {needle}.{details}"

        header = f"Workspace search for '{needle}' ({len(results)} matches; scanned {scanned_files} file(s)):"
        if skipped_large_files:
            header += f"\nSkipped {skipped_large_files} oversized file(s) above {max_file_bytes} bytes."
        return "\n".join([header, *results])

    def _iter_candidate_files(self, root: Path, *, rel_glob: str = "*"):
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if any(part in self.SKIP_DIR_NAMES for part in path.parts):
                continue
            if rel_glob not in {"", "*"} and not fnmatch.fnmatch(path.name, rel_glob):
                continue
            if path.suffix.lower() not in self.TEXT_SUFFIX_ALLOWLIST:
                continue
            yield path

    def propose_patch(self, project_slug: str, rel_path: str, instruction: str, *, approval_gate: Any, source: str = "workspace_tools") -> dict[str, Any]:
        target = self.resolve_path(project_slug, rel_path)
        if target.exists() and target.is_dir():
            return {"ok": False, "message": f"Cannot patch a directory: {rel_path}"}
        if not self.client:
            return {"ok": False, "message": "No local model client configured for patch generation."}

        existing = self._read_existing_text(target) if target.exists() else ""
        if target.exists() and existing is None:
            return {"ok": False, "message": "Could not read target file."}
        prompt = self._build_patch_prompt(
            project_slug=project_slug,
            rel_path=rel_path,
            existing=existing or "",
            instruction=instruction,
            ext=target.suffix.lower() or "(none)",
        )
        raw = self._chat(prompt)
        if raw is None:
            return {"ok": False, "message": "Patch generation failed."}
        parsed = self._parse_patch_response(raw)
        new_content = str(parsed.get("new_content", ""))
        summary = str(parsed.get("summary", "")).strip() or f"Update {rel_path}"
        if new_content == (existing or ""):
            return {"ok": False, "message": "Model returned no file changes."}

        diff_text = self._unified_diff(existing or "", new_content, rel_path)
        repo_rel = self.repo_relative_path(target)
        original_hash = hashlib.sha256((existing or "").encode("utf-8")).hexdigest() if target.exists() else ""
        proposal_id = approval_gate.create_action_proposal(
            action_type="apply_patch",
            action_payload={
                "path": repo_rel,
                "project_slug": project_slug,
                "original_sha256": original_hash,
                "new_content": new_content,
                "diff_text": diff_text,
                "summary": summary,
            },
            source=source,
            project_slug=project_slug,
            title=f"Patch {rel_path}: {summary[:80]}",
        )
        self._log_action(project_slug=project_slug, action_kind="patch_proposed", path_text=repo_rel, status="pending", detail=summary)
        return {
            "ok": True,
            "proposal_id": proposal_id,
            "path": repo_rel,
            "summary": summary,
            "diff_text": diff_text,
            "message": f"Patch proposal queued: {proposal_id}\nPath: {repo_rel}\nSummary: {summary}\n\n{diff_text[:8000]}",
        }

    def propose_patch_batch(self, project_slug: str, rel_paths: list[str], instruction: str, *, approval_gate: Any, source: str = "workspace_tools") -> dict[str, Any]:
        rel_paths = [str(item or "").strip() for item in rel_paths if str(item or "").strip()]
        deduped: list[str] = []
        for item in rel_paths:
            if item not in deduped:
                deduped.append(item)
        if not deduped:
            return {"ok": False, "message": "At least one workspace path is required."}
        if len(deduped) > 8:
            return {"ok": False, "message": "Batch patch is limited to 8 files at a time."}
        if not self.client:
            return {"ok": False, "message": "No local model client configured for patch generation."}

        file_specs: list[dict[str, Any]] = []
        for rel_path in deduped:
            target = self.resolve_path(project_slug, rel_path)
            if target.exists() and target.is_dir():
                return {"ok": False, "message": f"Cannot patch a directory in batch: {rel_path}"}
            existing = self._read_existing_text(target) if target.exists() else ""
            if target.exists() and existing is None:
                return {"ok": False, "message": f"Could not read target file: {rel_path}"}
            file_specs.append({
                "rel_path": rel_path,
                "repo_rel": self.repo_relative_path(target),
                "existing": existing or "",
                "ext": target.suffix.lower() or "(none)",
                "exists": target.exists(),
            })

        prompt = self._build_batch_patch_prompt(project_slug=project_slug, file_specs=file_specs, instruction=instruction)
        raw = self._chat(prompt)
        if raw is None:
            return {"ok": False, "message": "Batch patch generation failed."}
        parsed = self._parse_batch_patch_response(raw)
        files_data = parsed.get("files") if isinstance(parsed, dict) else None
        if not isinstance(files_data, list) or not files_data:
            return {"ok": False, "message": "Model did not return any file updates."}

        by_rel = {spec["rel_path"]: spec for spec in file_specs}
        batch_files: list[dict[str, Any]] = []
        preview_lines = ["Proposed batch changes:"]
        for item in files_data:
            if not isinstance(item, dict):
                continue
            rel_path = str(item.get("path", "")).strip()
            new_content = str(item.get("new_content", ""))
            if not rel_path or rel_path not in by_rel:
                continue
            spec = by_rel[rel_path]
            old_content = str(spec["existing"])
            if new_content == old_content:
                continue
            summary = str(item.get("summary", "")).strip() or f"Update {rel_path}"
            diff_text = self._unified_diff(old_content, new_content, rel_path)
            original_hash = hashlib.sha256(old_content.encode("utf-8")).hexdigest() if spec["exists"] else ""
            batch_files.append({
                "path": spec["repo_rel"],
                "project_slug": project_slug,
                "original_sha256": original_hash,
                "new_content": new_content,
                "diff_text": diff_text,
                "summary": summary,
            })
            preview_lines.append(f"- {spec['repo_rel']}: {summary}")
        if not batch_files:
            return {"ok": False, "message": "Model returned no effective file changes for the batch."}

        batch_summary = str(parsed.get("summary", "")).strip() or f"Batch update for {len(batch_files)} files"
        proposal_id = approval_gate.create_action_proposal(
            action_type="apply_patch_batch",
            action_payload={
                "project_slug": project_slug,
                "summary": batch_summary,
                "files": batch_files,
            },
            source=source,
            project_slug=project_slug,
            title=f"Batch patch ({len(batch_files)} files): {batch_summary[:70]}",
        )
        self._log_action(
            project_slug=project_slug,
            action_kind="patch_batch_proposed",
            path_text=",".join(item["path"] for item in batch_files),
            status="pending",
            detail=batch_summary,
        )
        preview = "\n".join(preview_lines[:20])
        return {
            "ok": True,
            "proposal_id": proposal_id,
            "summary": batch_summary,
            "file_count": len(batch_files),
            "message": f"Batch patch proposal queued: {proposal_id}\nSummary: {batch_summary}\nFiles: {len(batch_files)}\n\n{preview}",
        }

    def list_patch_proposals_text(self, approval_gate: Any, *, limit: int = 20) -> str:
        rows = approval_gate.list_action_proposals(limit=limit)
        patch_rows = [row for row in rows if str(row.get("action_type", "")).strip().lower() in {"apply_patch", "apply_patch_batch"}]
        if not patch_rows:
            return "No pending patch proposals."
        lines = [f"Pending patch proposals ({len(patch_rows)}):"]
        for row in patch_rows:
            payload = row.get("action_payload", {}) if isinstance(row.get("action_payload"), dict) else {}
            action_type = str(row.get("action_type", "")).strip().lower()
            if action_type == "apply_patch_batch":
                files = payload.get("files", []) if isinstance(payload.get("files"), list) else []
                file_names = ", ".join(str(item.get("path", "")).strip() for item in files[:3] if isinstance(item, dict))
                extra = f" (+{len(files) - 3} more)" if len(files) > 3 else ""
                lines.append(
                    f"- {row.get('id','')} | project={row.get('project','')} | files={len(files)} | {file_names}{extra} | title={row.get('title','')}"
                )
            else:
                lines.append(
                    f"- {row.get('id','')} | project={row.get('project','')} | path={payload.get('path','')} | title={row.get('title','')}"
                )
        return "\n".join(lines)

    def _patch_system_prompt(self) -> str:
        return (
            "You are a precise local coding assistant. Produce exactly one JSON object. "
            "For single-file mode return keys summary and new_content. For multi-file mode return keys summary and files, "
            "where files is an array of objects with path, summary, and new_content. "
            "Do not wrap JSON in markdown fences. Do not include commentary. Preserve unrelated code."
        )

    def _build_patch_prompt(self, *, project_slug: str, rel_path: str, existing: str, instruction: str, ext: str) -> str:
        return (
            f"Project: {project_slug}\n"
            f"Target file: {rel_path}\n"
            f"Extension: {ext}\n"
            f"Instruction:\n{instruction.strip()}\n\n"
            "Return JSON with:\n"
            "- summary: one short sentence\n"
            "- new_content: full replacement content for the file\n\n"
            "Current file content follows between markers. If the file does not exist, create an appropriate new file.\n"
            "<<<FILE>>>\n"
            f"{existing}\n"
            "<<<END_FILE>>>\n"
        )

    def _build_batch_patch_prompt(self, *, project_slug: str, file_specs: list[dict[str, Any]], instruction: str) -> str:
        lines = [
            f"Project: {project_slug}",
            f"Instruction:\n{instruction.strip()}",
            "",
            "Return one JSON object with:",
            "- summary: one short sentence for the whole batch",
            "- files: an array of objects with keys path, summary, and new_content",
            "Only include the listed files. Omit files that do not need changes.",
            "",
        ]
        for spec in file_specs:
            lines.extend([
                f"<<<FILE {spec['rel_path']}>>>",
                str(spec['existing']),
                f"<<<END_FILE {spec['rel_path']}>>>",
                "",
            ])
        return "\n".join(lines)

    def _chat(self, prompt: str) -> str | None:
        try:
            return self.client.chat(
                model=self.model_cfg.get("model", ""),
                system_prompt=self._patch_system_prompt(),
                user_prompt=prompt,
                temperature=0.1,
                num_ctx=int(self.model_cfg.get("num_ctx", 16384) or 16384),
                timeout=int(self.model_cfg.get("timeout", 240) or 240),
            )
        except Exception:
            LOGGER.exception("Workspace model chat failed.")
            return None

    def _parse_patch_response(self, raw: str) -> dict[str, Any]:
        text = str(raw or "").strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:].strip()
        candidates = [text]
        if "{" in text and "}" in text:
            candidates.append(text[text.find("{") : text.rfind("}") + 1])
        for candidate in candidates:
            try:
                data = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                return data
        return {"summary": "Generated file update", "new_content": text}

    def _parse_batch_patch_response(self, raw: str) -> dict[str, Any]:
        data = self._parse_patch_response(raw)
        if isinstance(data.get("files"), list):
            return data
        path = str(data.get("path", "")).strip()
        if path:
            return {"summary": str(data.get("summary", "")).strip(), "files": [data]}
        return data

    def _read_existing_text(self, path: Path) -> str | None:
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                return path.read_text(encoding="utf-8-sig")
            except Exception:
                return None
        except Exception:
            LOGGER.exception("Workspace model chat failed.")
            return None

    def _unified_diff(self, old: str, new: str, rel_path: str) -> str:
        diff = difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=f"a/{rel_path}",
            tofile=f"b/{rel_path}",
            lineterm="",
        )
        text = "\n".join(diff).strip()
        return text or f"(No textual diff generated for {rel_path})"

    def _log_action(self, *, project_slug: str, action_kind: str, path_text: str, status: str, detail: str = "") -> None:
        now = datetime.now(timezone.utc).isoformat()
        with connect(self.repo_root) as conn, transaction(conn, immediate=True):
            conn.execute(
                """
                INSERT INTO workspace_actions (
                    project, action_kind, path, status, detail, created_at
                ) VALUES (?, ?, ?, ?, ?, ?);
                """.strip(),
                (project_slug, action_kind, path_text, status, detail[:4000], now),
            )


__all__ = ["WorkspaceTools"]
