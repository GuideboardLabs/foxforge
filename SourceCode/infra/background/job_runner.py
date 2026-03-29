from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from infra.background.job_store import JobStore
from shared_tools.activity_bus import ActivityBus

LOGGER = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobRunner:
    def __init__(self, repo_root: Path, *, activity_bus: ActivityBus | None = None) -> None:
        self.repo_root = Path(repo_root)
        self.store = JobStore(self.repo_root)
        self.activity_bus = activity_bus or ActivityBus(self.repo_root)
        self._lock = Lock()
        self._active: dict[str, dict[str, Any]] = {}

    @staticmethod
    def make_key(profile_id: str, request_id: str) -> str:
        return f"{str(profile_id).strip() or 'owner'}:{str(request_id).strip()}"

    def start_job(self, *, profile_id: str, conversation_id: str, request_id: str, mode: str, user_text: str) -> str:
        self.store.start_job(
            profile_id=profile_id,
            request_id=request_id,
            conversation_id=conversation_id,
            mode=mode,
            user_text_preview=user_text,
        )
        return request_id

    def register_active(self, *, profile_id: str, request_id: str, conversation_id: str, project: str, lane: str) -> None:
        key = self.make_key(profile_id, request_id)
        with self._lock:
            self._active[key] = {
                'request_id': request_id,
                'profile_id': profile_id,
                'conversation_id': conversation_id,
                'project': project,
                'lane': lane or 'project',
                'started_at': _now_iso(),
            }
        self.store.attach_context(profile_id=profile_id, request_id=request_id, conversation_id=conversation_id, project=project, lane=lane)

    def unregister_active(self, *, profile_id: str, request_id: str) -> None:
        key = self.make_key(profile_id, request_id)
        with self._lock:
            self._active.pop(key, None)

    def active_count(self, profile_id: str | None = None) -> int:
        with self._lock:
            if profile_id:
                pid = str(profile_id).strip()
                return sum(1 for row in self._active.values() if str(row.get('profile_id', '')).strip() == pid)
            return len(self._active)

    def list_active_jobs_for_profile(self, *, profile_id: str, limit: int = 50) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with self._lock:
            active = list(self._active.values())
        for item in active:
            if str(item.get('profile_id', '')).strip() != str(profile_id).strip():
                continue
            request_id = str(item.get('request_id', '')).strip()
            if not request_id:
                continue
            job = self.store.get_job(profile_id=profile_id, request_id=request_id, event_limit=8) or {}
            if not job:
                continue
            job.setdefault('project', str(item.get('project', '')).strip())
            job.setdefault('lane', str(item.get('lane', '')).strip() or 'project')
            job['started_at'] = str(job.get('started_at', '')).strip() or str(item.get('started_at', '')).strip()
            rows.append(job)
        rows.sort(key=lambda row: str(row.get('started_at', '')), reverse=True)
        return rows[:max(1, int(limit))]

    def update_job(self, *, profile_id: str, request_id: str, stage: str, detail: str = '', summary_path: str = '', raw_path: str = '', web_stack: dict[str, Any] | None = None, agent_event: dict[str, Any] | None = None) -> None:
        current = self.store.get_job(profile_id=profile_id, request_id=request_id, event_limit=8) or {}
        tracker = current.get('agent_tracker', {}) if isinstance(current.get('agent_tracker'), dict) else {}
        if agent_event:
            tracker = self._apply_agent_event(tracker, agent_event)
        job = self.store.update_job(
            profile_id=profile_id,
            request_id=request_id,
            stage=stage,
            detail=detail,
            summary_path=summary_path,
            raw_path=raw_path,
            web_stack=web_stack,
            agent_tracker=tracker if isinstance(tracker, dict) else None,
        )
        if job:
            self.activity_bus.emit('job_runner', 'job_progress', {
                'request_id': request_id,
                'profile_id': profile_id,
                'stage': stage,
                'detail': detail,
                'project': str(job.get('project', '')).strip(),
                'lane': str(job.get('lane', '')).strip(),
            })

    def request_cancel(self, *, profile_id: str, request_id: str) -> tuple[bool, dict[str, Any] | None]:
        ok = self.store.request_cancel(profile_id=profile_id, request_id=request_id)
        return ok, self.store.get_job(profile_id=profile_id, request_id=request_id, event_limit=8)

    def is_cancel_requested(self, *, profile_id: str, request_id: str) -> bool:
        job = self.store.get_job(profile_id=profile_id, request_id=request_id, event_limit=1)
        return bool(job and job.get('cancel_requested', False))

    def finish_job(self, *, profile_id: str, request_id: str, status: str, detail: str = '', summary_path: str = '', raw_path: str = '') -> dict[str, Any] | None:
        job = self.store.finish_job(
            profile_id=profile_id,
            request_id=request_id,
            status=status,
            detail=detail,
            summary_path=summary_path,
            raw_path=raw_path,
        )
        self.unregister_active(profile_id=profile_id, request_id=request_id)
        if job:
            self.activity_bus.emit('job_runner', 'job_finished', {
                'request_id': request_id,
                'profile_id': profile_id,
                'status': status,
                'project': str(job.get('project', '')).strip(),
                'lane': str(job.get('lane', '')).strip(),
            })
        return job

    def get_job(self, *, profile_id: str, request_id: str, event_limit: int = 24) -> dict[str, Any] | None:
        return self.store.get_job(profile_id=profile_id, request_id=request_id, event_limit=event_limit)

    def list_events(self, *, profile_id: str, request_id: str, limit: int = 50) -> list[dict[str, Any]]:
        return self.store.list_events(profile_id=profile_id, request_id=request_id, limit=limit)

    @staticmethod
    def _apply_agent_event(tracker: dict[str, Any], agent_event: dict[str, Any]) -> dict[str, Any]:
        tracker = dict(tracker or {})
        tracker.setdefault('total', 0)
        tracker.setdefault('profile', '')
        tracker.setdefault('topic_type', '')
        tracker.setdefault('workers', 1)
        tracker.setdefault('all_agents', [])
        tracker.setdefault('active', [])
        tracker.setdefault('done', [])
        ae_stage = str(agent_event.get('stage', '')).strip()
        if ae_stage == 'research_pool_started':
            tracker['total'] = int(agent_event.get('agents_total', 0))
            tracker['profile'] = str(agent_event.get('analysis_profile', '')).strip().replace('_', ' ')
            tracker['topic_type'] = str(agent_event.get('topic_type', '')).strip()
            tracker['workers'] = int(agent_event.get('workers', 1))
            tracker['all_agents'] = list(agent_event.get('agents', []))
            tracker['active'] = []
            tracker['done'] = []
        elif ae_stage == 'research_agent_started':
            persona = str(agent_event.get('agent', '')).strip()
            if persona and persona not in [a.get('persona') for a in tracker['active'] if isinstance(a, dict)]:
                tracker['active'].append({
                    'persona': persona,
                    'directive': str(agent_event.get('directive', '')).strip(),
                    'role': str(agent_event.get('role', 'primary')).strip(),
                    'model': str(agent_event.get('model', '')).strip(),
                })
        elif ae_stage == 'research_agent_completed':
            persona = str(agent_event.get('agent', '')).strip()
            tracker['active'] = [a for a in tracker['active'] if (a.get('persona') if isinstance(a, dict) else a) != persona]
            if persona:
                tracker['done'].append({
                    'persona': persona,
                    'failed': bool(agent_event.get('failed', False)),
                    'role': str(agent_event.get('role', 'primary')).strip(),
                    'finding_preview': str(agent_event.get('finding_preview', '')).strip()[:400],
                    'confidence': int(agent_event.get('confidence', 0)),
                })
        return tracker
