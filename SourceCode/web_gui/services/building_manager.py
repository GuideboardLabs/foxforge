"""Building state management — tracks active Make/Build jobs and pause state.

Mirror of ForagingManager for the Build lane.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from web_gui.services.job_manager import JobManager


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class BuildingManager:
    """Thread-safe building state manager.

    Tracks active Build jobs, paused state, and make_type per job.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state: dict[str, Any] = {
            "active_jobs": {},
            "last_successful_by_profile": {},
            "updated_at": _now_iso(),
            "paused": False,
        }

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    def snapshot(self, profile_id: str | None = None) -> dict[str, Any]:
        with self._lock:
            active_jobs = dict(self._state.get("active_jobs", {}))
            completions = dict(self._state.get("last_successful_by_profile", {}))
            paused = bool(self._state.get("paused", False))
            updated_at = str(self._state.get("updated_at", "")).strip()
        completion_row: dict[str, Any] = {}
        if profile_id:
            pid = str(profile_id).strip()
            active_jobs = {
                key: value
                for key, value in active_jobs.items()
                if isinstance(value, dict) and str(value.get("profile_id", "")).strip() == pid
            }
            candidate = completions.get(pid)
            if isinstance(candidate, dict):
                completion_row = dict(candidate)
        return {
            "paused": paused,
            "active_jobs": len(active_jobs),
            "updated_at": updated_at or _now_iso(),
            "completion_unread": bool(completion_row.get("completion_unread", False)),
            "last_completed_id": str(completion_row.get("id", "")).strip(),
            "last_completed_at": str(completion_row.get("finished_at", "")).strip(),
        }

    def rows_for_profile(
        self, profile: dict[str, Any], job_manager: "JobManager", limit: int = 50
    ) -> list[dict[str, Any]]:
        pid = str(profile.get("id", "")).strip()
        rows: list[dict[str, Any]] = []
        with self._lock:
            active_jobs = dict(self._state.get("active_jobs", {}))
            completions = dict(self._state.get("last_successful_by_profile", {}))
        for item in active_jobs.values():
            if not isinstance(item, dict):
                continue
            if str(item.get("profile_id", "")).strip() != pid:
                continue
            request_id = str(item.get("request_id", "")).strip()
            if not request_id:
                continue
            job_key = job_manager.key(profile, request_id)
            job_row = job_manager.snapshot(job_key)
            events = job_row.get("events", []) if isinstance(job_row, dict) else []
            last_event = events[-1] if isinstance(events, list) and events else {}
            rows.append({
                "id": request_id,
                "conversation_id": str(item.get("conversation_id", "")).strip(),
                "project": str(item.get("project", "")).strip(),
                "make_type": str(item.get("make_type", "")).strip(),
                "lane": str(item.get("lane", "")).strip() or "make_longform",
                "topic_type": str(item.get("topic_type", "")).strip(),
                "extends_request_id": str(item.get("extends_request_id", "")).strip(),
                "started_at": str(job_row.get("started_at", "") or item.get("started_at", "")).strip(),
                "updated_at": str(job_row.get("updated_at", "") or item.get("started_at", "")).strip(),
                "stage": str(job_row.get("stage", "")).strip() or "running",
                "status": str(job_row.get("status", "")).strip() or "running",
                "summary_path": str(job_row.get("summary_path", "")).strip(),
                "last_detail": str(last_event.get("detail", "")).strip() if isinstance(last_event, dict) else "",
                "agent_tracker": job_row.get("agent_tracker", {}) if isinstance(job_row.get("agent_tracker"), dict) else {},
            })
        last_success = completions.get(pid)
        if isinstance(last_success, dict):
            completed_request_id = str(last_success.get("id", "")).strip()
            if completed_request_id and not any(str(row.get("id", "")).strip() == completed_request_id for row in rows):
                rows.append(dict(last_success))
        rows.sort(key=lambda row: str(row.get("started_at", "")), reverse=True)
        return rows[:max(1, int(limit))]

    def active_count(self) -> int:
        with self._lock:
            active_jobs = self._state.get("active_jobs", {})
            if not isinstance(active_jobs, dict):
                return 0
            return len(active_jobs)

    # ------------------------------------------------------------------
    # Job registration
    # ------------------------------------------------------------------

    def register_job(
        self,
        *,
        profile: dict[str, Any],
        conversation_id: str,
        request_id: str,
        project: str,
        make_type: str,
        lane: str,
        job_key: str,
        topic_type: str = "",
        extends_request_id: str = "",
    ) -> None:
        with self._lock:
            active_jobs = self._state.get("active_jobs", {})
            if not isinstance(active_jobs, dict):
                active_jobs = {}
                self._state["active_jobs"] = active_jobs
            active_jobs[job_key] = {
                "request_id": str(request_id).strip(),
                "profile_id": str(profile.get("id", "")).strip(),
                "conversation_id": str(conversation_id).strip(),
                "project": str(project).strip(),
                "make_type": str(make_type).strip(),
                "lane": str(lane).strip(),
                "topic_type": str(topic_type).strip().lower(),
                "extends_request_id": str(extends_request_id).strip(),
                "started_at": _now_iso(),
            }
            self._state["updated_at"] = _now_iso()

    def unregister_job(self, job_key: str) -> None:
        with self._lock:
            active_jobs = self._state.get("active_jobs", {})
            if isinstance(active_jobs, dict):
                active_jobs.pop(job_key, None)
            self._state["updated_at"] = _now_iso()

    def record_completion(
        self,
        *,
        profile: dict[str, Any],
        conversation_id: str,
        request_id: str,
        project: str,
        make_type: str,
        lane: str,
        topic_type: str = "",
        job_row: dict[str, Any] | None = None,
        status: str = "completed",
    ) -> None:
        status_key = str(status or "").strip().lower()
        if status_key not in {"completed", "completed_with_warnings"}:
            return
        row = job_row if isinstance(job_row, dict) else {}
        events = row.get("events", [])
        last_event = events[-1] if isinstance(events, list) and events else {}
        started_at = str(row.get("started_at", "")).strip() or _now_iso()
        finished_at = str(row.get("updated_at", "")).strip() or _now_iso()
        snapshot = {
            "id": str(request_id).strip(),
            "conversation_id": str(conversation_id).strip(),
            "project": str(project).strip(),
            "make_type": str(make_type).strip(),
            "lane": str(lane).strip() or "make_longform",
            "topic_type": str(topic_type).strip().lower(),
            "started_at": started_at,
            "updated_at": finished_at,
            "finished_at": finished_at,
            "stage": str(row.get("stage", "")).strip() or status_key,
            "status": status_key,
            "summary_path": str(row.get("summary_path", "")).strip(),
            "last_detail": str(last_event.get("detail", "")).strip() if isinstance(last_event, dict) else "",
            "agent_tracker": row.get("agent_tracker", {}) if isinstance(row.get("agent_tracker"), dict) else {},
            "is_last_successful": True,
            "completion_unread": True,
        }
        pid = str(profile.get("id", "")).strip()
        if not pid:
            return
        with self._lock:
            completions = self._state.get("last_successful_by_profile", {})
            if not isinstance(completions, dict):
                completions = {}
                self._state["last_successful_by_profile"] = completions
            completions[pid] = snapshot
            self._state["updated_at"] = _now_iso()

    def mark_completion_read(self, profile_id: str | None = None) -> None:
        pid = str(profile_id or "").strip()
        if not pid:
            return
        with self._lock:
            completions = self._state.get("last_successful_by_profile", {})
            if not isinstance(completions, dict):
                return
            row = completions.get(pid)
            if not isinstance(row, dict):
                return
            if bool(row.get("completion_unread", False)):
                row["completion_unread"] = False
                row["updated_at"] = _now_iso()
                self._state["updated_at"] = row["updated_at"]

    # ------------------------------------------------------------------
    # Pause control
    # ------------------------------------------------------------------

    def set_paused(self, paused: bool) -> bool:
        value = bool(paused)
        with self._lock:
            self._state["paused"] = value
            self._state["updated_at"] = _now_iso()
        return value

    def is_paused(self) -> bool:
        with self._lock:
            return bool(self._state.get("paused", False))
