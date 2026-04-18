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
            "updated_at": _now_iso(),
            "paused": False,
        }

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    def snapshot(self, profile_id: str | None = None) -> dict[str, Any]:
        with self._lock:
            active_jobs = dict(self._state.get("active_jobs", {}))
            paused = bool(self._state.get("paused", False))
            updated_at = str(self._state.get("updated_at", "")).strip()
        if profile_id:
            pid = str(profile_id).strip()
            active_jobs = {
                key: value
                for key, value in active_jobs.items()
                if isinstance(value, dict) and str(value.get("profile_id", "")).strip() == pid
            }
        return {
            "paused": paused,
            "active_jobs": len(active_jobs),
            "updated_at": updated_at or _now_iso(),
        }

    def rows_for_profile(
        self, profile: dict[str, Any], job_manager: "JobManager", limit: int = 50
    ) -> list[dict[str, Any]]:
        pid = str(profile.get("id", "")).strip()
        rows: list[dict[str, Any]] = []
        with self._lock:
            active_jobs = dict(self._state.get("active_jobs", {}))
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
                "started_at": str(job_row.get("started_at", "") or item.get("started_at", "")).strip(),
                "updated_at": str(job_row.get("updated_at", "") or item.get("started_at", "")).strip(),
                "stage": str(job_row.get("stage", "")).strip() or "running",
                "status": str(job_row.get("status", "")).strip() or "running",
                "summary_path": str(job_row.get("summary_path", "")).strip(),
                "last_detail": str(last_event.get("detail", "")).strip() if isinstance(last_event, dict) else "",
                "agent_tracker": job_row.get("agent_tracker", {}) if isinstance(job_row.get("agent_tracker"), dict) else {},
            })
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
                "started_at": _now_iso(),
            }
            self._state["updated_at"] = _now_iso()

    def unregister_job(self, job_key: str) -> None:
        with self._lock:
            active_jobs = self._state.get("active_jobs", {})
            if isinstance(active_jobs, dict):
                active_jobs.pop(job_key, None)
            self._state["updated_at"] = _now_iso()

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
