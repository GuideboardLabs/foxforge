from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from shared_tools.db import row_to_dict, transaction
from infra.persistence.sqlite_db import connect_db, ensure_state_db


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobStore:
    def __init__(self, repo_root: Path) -> None:
        self.repo_root = Path(repo_root)
        ensure_state_db(self.repo_root)

    def _normalize_job(self, row: dict[str, Any] | None) -> dict[str, Any] | None:
        if not row:
            return None
        return {
            "request_id": str(row.get("request_id", "")).strip(),
            "profile_id": str(row.get("profile_id", "")).strip(),
            "conversation_id": str(row.get("conversation_id", "")).strip(),
            "mode": str(row.get("mode", "command")).strip() or "command",
            "project": str(row.get("project", "")).strip(),
            "lane": str(row.get("lane", "")).strip(),
            "status": str(row.get("status", "running")).strip() or "running",
            "stage": str(row.get("stage", "queued")).strip() or "queued",
            "started_at": str(row.get("started_at", "")).strip(),
            "updated_at": str(row.get("updated_at", "")).strip(),
            "cancel_requested": bool(int(row.get("cancel_requested", 0) or 0)),
            "cancel_requested_at": str(row.get("cancel_requested_at", "")).strip(),
            "summary_path": str(row.get("summary_path", "")).strip(),
            "raw_path": str(row.get("raw_path", "")).strip(),
            "web_stack": self._load_json(row.get("web_stack_json"), {}),
            "agent_tracker": self._load_json(row.get("agent_tracker_json"), {}),
            "user_text_preview": str(row.get("user_text_preview", "")).strip(),
        }

    @staticmethod
    def _load_json(value: Any, default: Any) -> Any:
        try:
            raw = json.loads(str(value or ""))
        except (TypeError, ValueError, json.JSONDecodeError):
            return default
        return raw if isinstance(raw, type(default)) else default

    def start_job(self, *, profile_id: str, request_id: str, conversation_id: str, mode: str, user_text_preview: str) -> dict[str, Any]:
        now = _now_iso()
        with connect_db(self.repo_root) as conn, transaction(conn, immediate=True):
            conn.execute(
                """
                INSERT INTO job_runs(profile_id, request_id, conversation_id, mode, status, stage, started_at, updated_at, user_text_preview)
                VALUES (?, ?, ?, ?, 'running', 'queued', ?, ?, ?)
                ON CONFLICT(profile_id, request_id) DO UPDATE SET
                    conversation_id = excluded.conversation_id,
                    mode = excluded.mode,
                    status = 'running',
                    stage = 'queued',
                    updated_at = excluded.updated_at,
                    cancel_requested = 0,
                    cancel_requested_at = '',
                    user_text_preview = excluded.user_text_preview;
                """.strip(),
                (profile_id, request_id, conversation_id, mode, now, now, user_text_preview[:280]),
            )
        return self.get_job(profile_id=profile_id, request_id=request_id) or {}

    def attach_context(self, *, profile_id: str, request_id: str, conversation_id: str = '', project: str = '', lane: str = '') -> None:
        now = _now_iso()
        with connect_db(self.repo_root) as conn, transaction(conn, immediate=True):
            conn.execute(
                """
                UPDATE job_runs
                   SET conversation_id = CASE WHEN ? <> '' THEN ? ELSE conversation_id END,
                       project = CASE WHEN ? <> '' THEN ? ELSE project END,
                       lane = CASE WHEN ? <> '' THEN ? ELSE lane END,
                       updated_at = ?
                 WHERE profile_id = ? AND request_id = ?;
                """.strip(),
                (conversation_id, conversation_id, project, project, lane, lane, now, profile_id, request_id),
            )

    def update_job(self, *, profile_id: str, request_id: str, stage: str, detail: str = '', summary_path: str = '', raw_path: str = '', web_stack: dict[str, Any] | None = None, agent_tracker: dict[str, Any] | None = None) -> dict[str, Any] | None:
        now = _now_iso()
        with connect_db(self.repo_root) as conn, transaction(conn, immediate=True):
            current = row_to_dict(conn.execute("SELECT * FROM job_runs WHERE profile_id = ? AND request_id = ?;", (profile_id, request_id)).fetchone())
            if not current:
                return None
            web_stack_json = current.get('web_stack_json', '{}')
            if web_stack is not None:
                web_stack_json = json.dumps(web_stack, ensure_ascii=True)
            agent_tracker_json = current.get('agent_tracker_json', '{}')
            if agent_tracker is not None:
                agent_tracker_json = json.dumps(agent_tracker, ensure_ascii=True)
            conn.execute(
                """
                UPDATE job_runs
                   SET stage = ?,
                       updated_at = ?,
                       summary_path = CASE WHEN ? <> '' THEN ? ELSE summary_path END,
                       raw_path = CASE WHEN ? <> '' THEN ? ELSE raw_path END,
                       web_stack_json = ?,
                       agent_tracker_json = ?
                 WHERE profile_id = ? AND request_id = ?;
                """.strip(),
                (stage, now, summary_path, summary_path, raw_path, raw_path, web_stack_json, agent_tracker_json, profile_id, request_id),
            )
            conn.execute(
                "INSERT INTO job_events(profile_id, request_id, ts, stage, detail) VALUES (?, ?, ?, ?, ?);",
                (profile_id, request_id, now, stage, detail[:400]),
            )
        return self.get_job(profile_id=profile_id, request_id=request_id)

    def request_cancel(self, *, profile_id: str, request_id: str) -> bool:
        now = _now_iso()
        with connect_db(self.repo_root) as conn, transaction(conn, immediate=True):
            row = row_to_dict(conn.execute("SELECT * FROM job_runs WHERE profile_id = ? AND request_id = ?;", (profile_id, request_id)).fetchone())
            if not row or str(row.get('status', '')) != 'running':
                return False
            conn.execute(
                """
                UPDATE job_runs
                   SET cancel_requested = 1, cancel_requested_at = ?, updated_at = ?, stage = 'cancel_requested'
                 WHERE profile_id = ? AND request_id = ?;
                """.strip(),
                (now, now, profile_id, request_id),
            )
            conn.execute(
                "INSERT INTO job_events(profile_id, request_id, ts, stage, detail) VALUES (?, ?, ?, 'cancel_requested', 'User pressed cancel.');",
                (profile_id, request_id, now),
            )
        return True

    def finish_job(self, *, profile_id: str, request_id: str, status: str, detail: str = '', summary_path: str = '', raw_path: str = '') -> dict[str, Any] | None:
        now = _now_iso()
        final_stage = str(status or 'completed').strip().lower() or 'completed'
        with connect_db(self.repo_root) as conn, transaction(conn, immediate=True):
            row = row_to_dict(conn.execute("SELECT * FROM job_runs WHERE profile_id = ? AND request_id = ?;", (profile_id, request_id)).fetchone())
            if not row:
                return None
            conn.execute(
                """
                UPDATE job_runs
                   SET status = ?, stage = ?, updated_at = ?,
                       summary_path = CASE WHEN ? <> '' THEN ? ELSE summary_path END,
                       raw_path = CASE WHEN ? <> '' THEN ? ELSE raw_path END
                 WHERE profile_id = ? AND request_id = ?;
                """.strip(),
                (final_stage, final_stage, now, summary_path, summary_path, raw_path, raw_path, profile_id, request_id),
            )
            conn.execute(
                "INSERT INTO job_events(profile_id, request_id, ts, stage, detail) VALUES (?, ?, ?, ?, ?);",
                (profile_id, request_id, now, final_stage, detail[:400]),
            )
        return self.get_job(profile_id=profile_id, request_id=request_id)

    def get_job(self, *, profile_id: str, request_id: str, event_limit: int = 24) -> dict[str, Any] | None:
        with connect_db(self.repo_root) as conn:
            row = row_to_dict(conn.execute("SELECT * FROM job_runs WHERE profile_id = ? AND request_id = ?;", (profile_id, request_id)).fetchone())
            job = self._normalize_job(row)
            if job is None:
                return None
            events = conn.execute(
                "SELECT ts, stage, detail FROM job_events WHERE profile_id = ? AND request_id = ? ORDER BY id DESC LIMIT ?;",
                (profile_id, request_id, max(1, int(event_limit))),
            ).fetchall()
        job['events'] = [row_to_dict(item) or {} for item in reversed(events)]
        return job

    def list_events(self, *, profile_id: str, request_id: str, limit: int = 50) -> list[dict[str, Any]]:
        with connect_db(self.repo_root) as conn:
            rows = conn.execute(
                "SELECT ts, stage, detail FROM job_events WHERE profile_id = ? AND request_id = ? ORDER BY id DESC LIMIT ?;",
                (profile_id, request_id, max(1, int(limit))),
            ).fetchall()
        return [row_to_dict(item) or {} for item in reversed(rows)]

    def list_recent_for_profile(self, *, profile_id: str, limit: int = 50, mode: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM job_runs WHERE profile_id = ?"
        params: list[Any] = [profile_id]
        if mode:
            sql += " AND mode = ?"
            params.append(mode)
        sql += " ORDER BY updated_at DESC LIMIT ?;"
        params.append(max(1, int(limit)))
        with connect_db(self.repo_root) as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
        return [self._normalize_job(row_to_dict(row)) or {} for row in rows]
