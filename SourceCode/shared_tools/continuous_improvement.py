from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock
from typing import Any


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _now_iso() -> str:
    return _now().isoformat()


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _parse_iso(value: str | None) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


class ContinuousImprovementEngine:
    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root
        self.path = repo_root / "Runtime" / "learning" / "continuous_improvement.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = Lock()
        self.default_config = {
            "enabled": True,
            "facts_refresh_interval_minutes": 20,
            "facts_refresh_min_user_messages": 6,
            "auto_reinforce_up_threshold": 0.72,
            "auto_reinforce_down_threshold": 0.2,
            "auto_trigger_reflection_threshold": 0.55,
            "auto_reinforce_up_min_consecutive": 3,
            "max_recent_events": 200,
        }
        if not self.path.exists():
            self._save(self._empty_state())

    def _empty_state(self) -> dict[str, Any]:
        return {
            "config": dict(self.default_config),
            "stats": {
                "total_turns": 0,
                "auto_fact_refreshes": 0,
                "auto_reinforce_up": 0,
                "auto_reinforce_down": 0,
                "avg_context_quality": 0.0,
            },
            "projects": {},
            "recent_events": [],
            "updated_at": _now_iso(),
        }

    def _load(self) -> dict[str, Any]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            data = self._empty_state()
        if not isinstance(data, dict):
            data = self._empty_state()
        cfg = data.get("config")
        if not isinstance(cfg, dict):
            cfg = {}
        normalized = dict(self.default_config)
        for key in normalized.keys():
            if key in cfg:
                normalized[key] = cfg[key]
        data["config"] = normalized
        stats = data.get("stats")
        if not isinstance(stats, dict):
            stats = {}
        data["stats"] = {
            "total_turns": int(stats.get("total_turns", 0)),
            "auto_fact_refreshes": int(stats.get("auto_fact_refreshes", 0)),
            "auto_reinforce_up": int(stats.get("auto_reinforce_up", 0)),
            "auto_reinforce_down": int(stats.get("auto_reinforce_down", 0)),
            "avg_context_quality": float(stats.get("avg_context_quality", 0.0) or 0.0),
        }
        projects = data.get("projects")
        if not isinstance(projects, dict):
            data["projects"] = {}
        events = data.get("recent_events")
        if not isinstance(events, list):
            data["recent_events"] = []
        return data

    def _save(self, data: dict[str, Any]) -> None:
        data["updated_at"] = _now_iso()
        self.path.write_text(json.dumps(data, indent=2, ensure_ascii=True), encoding="utf-8")

    def _project_key(self, project: str) -> str:
        text = "_".join(str(project or "").strip().split()).lower()
        return text or "general"

    def _project_bucket(self, data: dict[str, Any], project: str) -> dict[str, Any]:
        key = self._project_key(project)
        projects = data.setdefault("projects", {})
        bucket = projects.get(key)
        if not isinstance(bucket, dict):
            bucket = {}
        bucket.setdefault("turns", 0)
        bucket.setdefault("avg_quality", 0.0)
        bucket.setdefault("avg_context_quality", 0.0)
        bucket.setdefault("last_turn_at", "")
        bucket.setdefault("last_fact_refresh_at", "")
        projects[key] = bucket
        return bucket

    def _append_event(self, data: dict[str, Any], event: dict[str, Any]) -> None:
        rows = data.setdefault("recent_events", [])
        if not isinstance(rows, list):
            rows = []
            data["recent_events"] = rows
        rows.append(event)
        max_rows = int(data.get("config", {}).get("max_recent_events", 200))
        if max_rows < 20:
            max_rows = 20
        if len(rows) > max_rows:
            del rows[:-max_rows]

    def is_enabled(self) -> bool:
        with self.lock:
            data = self._load()
            return bool(data.get("config", {}).get("enabled", True))

    def should_refresh_facts(
        self,
        *,
        project: str,
        history_user_count: int,
        facts_count: int,
        facts_updated_at: str | None,
    ) -> tuple[bool, str]:
        with self.lock:
            data = self._load()
            cfg = data.get("config", {})
            if not bool(cfg.get("enabled", True)):
                return False, "disabled"
            min_msgs = max(1, int(cfg.get("facts_refresh_min_user_messages", 6)))
            if int(history_user_count) < min_msgs:
                return False, "history_too_short"
            interval_mins = max(1, int(cfg.get("facts_refresh_interval_minutes", 20)))
            interval = timedelta(minutes=interval_mins)
            bucket = self._project_bucket(data, project)
            last_refresh = _parse_iso(str(bucket.get("last_fact_refresh_at", "")))
            now = _now()
            if last_refresh and (now - last_refresh) < interval:
                return False, "recently_refreshed"
            if int(facts_count) <= 0:
                return True, "no_facts"
            facts_ts = _parse_iso(facts_updated_at)
            if not facts_ts:
                return True, "facts_missing_timestamp"
            if (now - facts_ts) >= (interval * 2):
                return True, "facts_stale"
            return False, "fresh_enough"

    def note_fact_refresh(self, *, project: str, reason: str, refresh_result: dict[str, Any]) -> None:
        with self.lock:
            data = self._load()
            bucket = self._project_bucket(data, project)
            bucket["last_fact_refresh_at"] = _now_iso()
            stats = data.setdefault("stats", {})
            stats["auto_fact_refreshes"] = int(stats.get("auto_fact_refreshes", 0)) + 1
            self._append_event(
                data,
                {
                    "ts": _now_iso(),
                    "type": "facts_refresh",
                    "project": self._project_key(project),
                    "reason": reason,
                    "scanned_user_messages": int(refresh_result.get("scanned_user_messages", 0)),
                    "updated_fields": int(refresh_result.get("updated_fields", 0)),
                    "facts_count": int(refresh_result.get("facts_count", 0)),
                },
            )
            self._save(data)

    def evaluate_turn(
        self,
        *,
        user_text: str,
        assistant_text: str,
        lane: str,
        worker_result: dict[str, Any] | None,
        context_feedback: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        user = str(user_text or "").strip()
        reply = str(assistant_text or "").strip()
        low = reply.lower()
        score = 0.6
        notes: list[str] = []
        context_score = 0.5

        if len(user) >= 40:
            score += 0.03
            notes.append("substantial_user_input")

        if "completed" in low:
            score += 0.08
            notes.append("reply_marks_completion")
        if "next action" in low:
            score += 0.05
            notes.append("reply_has_next_action")
        if "self-reflection check" in low:
            score += 0.02
            notes.append("reflection_attached")

        failure_hits = 0
        for token in (
            "http 500",
            "error:",
            "failed",
            "could not",
            "not detect",
            "no attached image",
            "no stored project facts yet",
        ):
            if token in low:
                failure_hits += 1
        if failure_hits:
            score -= min(0.34, 0.12 * float(failure_hits))
            notes.append(f"failure_markers={failure_hits}")

        if isinstance(worker_result, dict):
            artifact_paths = [
                str(worker_result.get("summary_path", "")).strip(),
                str(worker_result.get("path", "")).strip(),
            ]
            if any(path for path in artifact_paths):
                score += 0.14
                notes.append("worker_artifact_written")
            message = str(worker_result.get("message", "")).lower().strip()
            if "completed" in message or "success" in message:
                score += 0.05
                notes.append("worker_success_message")

        if isinstance(context_feedback, dict):
            context_score = _clamp01(float(context_feedback.get("score", 0.5) or 0.5))
            score = (score * 0.78) + (context_score * 0.22)
            fb_notes = context_feedback.get("notes", [])
            if isinstance(fb_notes, list):
                for note in fb_notes[:8]:
                    text = str(note or "").strip()
                    if text:
                        notes.append(f"context:{text}")

        score = _clamp01(score)
        if score >= 0.75:
            outcome = "good"
        elif score <= 0.35:
            outcome = "poor"
        else:
            outcome = "mixed"
        return {
            "score": score,
            "outcome": outcome,
            "notes": notes,
            "lane": str(lane or "project").strip().lower(),
            "context_score": context_score,
        }

    def decide_reinforcement_direction(self, score: float) -> str:
        with self.lock:
            data = self._load()
            cfg = data.get("config", {})
            if not bool(cfg.get("enabled", True)):
                return ""
            up = float(cfg.get("auto_reinforce_up_threshold", 0.72))
            down = float(cfg.get("auto_reinforce_down_threshold", 0.2))
        if score >= up:
            return "up"
        if score <= down:
            return "down"
        return ""

    def should_trigger_reflection(self, score: float) -> bool:
        """Return True when a turn score is low enough to warrant a reflection pass."""
        with self.lock:
            data = self._load()
            cfg = data.get("config", {})
            if not bool(cfg.get("enabled", True)):
                return False
            threshold = float(cfg.get("auto_trigger_reflection_threshold", 0.55))
        return score < threshold

    def note_turn(
        self,
        *,
        project: str,
        lane: str,
        quality_score: float,
        outcome: str,
        notes: list[str] | None = None,
        context_score: float = 0.5,
        reinforcement_direction: str = "",
        reinforcement_lesson_id: str = "",
    ) -> None:
        with self.lock:
            data = self._load()
            bucket = self._project_bucket(data, project)
            turns = int(bucket.get("turns", 0)) + 1
            prior_avg = float(bucket.get("avg_quality", 0.0))
            avg = ((prior_avg * float(turns - 1)) + float(quality_score)) / float(turns)
            prior_context_avg = float(bucket.get("avg_context_quality", 0.0))
            context_avg = ((prior_context_avg * float(turns - 1)) + float(context_score)) / float(turns)
            bucket["turns"] = turns
            bucket["avg_quality"] = round(_clamp01(avg), 4)
            bucket["avg_context_quality"] = round(_clamp01(context_avg), 4)
            bucket["last_turn_at"] = _now_iso()
            stats = data.setdefault("stats", {})
            stats["total_turns"] = int(stats.get("total_turns", 0)) + 1
            total_turns = int(stats.get("total_turns", 0))
            prior_global_context = float(stats.get("avg_context_quality", 0.0))
            global_context_avg = ((prior_global_context * float(max(0, total_turns - 1))) + float(context_score)) / float(max(1, total_turns))
            stats["avg_context_quality"] = round(_clamp01(global_context_avg), 4)
            direction = str(reinforcement_direction or "").strip().lower()
            if direction == "up":
                stats["auto_reinforce_up"] = int(stats.get("auto_reinforce_up", 0)) + 1
            elif direction == "down":
                stats["auto_reinforce_down"] = int(stats.get("auto_reinforce_down", 0)) + 1
            self._append_event(
                data,
                {
                    "ts": _now_iso(),
                    "type": "turn_quality",
                    "project": self._project_key(project),
                    "lane": str(lane or "project").strip().lower(),
                    "score": round(_clamp01(quality_score), 4),
                    "context_score": round(_clamp01(context_score), 4),
                    "outcome": str(outcome or "mixed"),
                    "notes": list(notes or []),
                    "reinforcement_direction": direction,
                    "reinforcement_lesson_id": str(reinforcement_lesson_id or "").strip(),
                },
            )
            self._save(data)

    def status_snapshot(self, project: str) -> dict[str, Any]:
        with self.lock:
            data = self._load()
        key = self._project_key(project)
        project_row = data.get("projects", {}).get(key, {})
        recent = list(data.get("recent_events", []))
        recent_project_events = [row for row in recent if str(row.get("project", "")) == key][-8:]
        return {
            "enabled": bool(data.get("config", {}).get("enabled", True)),
            "config": dict(data.get("config", {})),
            "stats": dict(data.get("stats", {})),
            "project": {
                "slug": key,
                "turns": int(project_row.get("turns", 0)) if isinstance(project_row, dict) else 0,
                "avg_quality": float(project_row.get("avg_quality", 0.0)) if isinstance(project_row, dict) else 0.0,
                "avg_context_quality": float(project_row.get("avg_context_quality", 0.0)) if isinstance(project_row, dict) else 0.0,
                "last_turn_at": str(project_row.get("last_turn_at", "")) if isinstance(project_row, dict) else "",
                "last_fact_refresh_at": (
                    str(project_row.get("last_fact_refresh_at", "")) if isinstance(project_row, dict) else ""
                ),
            },
            "recent_project_events": recent_project_events,
        }

    def status_text(self, project: str) -> str:
        snap = self.status_snapshot(project)
        stats = snap.get("stats", {})
        proj = snap.get("project", {})
        lines = [
            "Continuous improvement status:",
            f"- enabled: {bool(snap.get('enabled', True))}",
            f"- active_project: {proj.get('slug', '')}",
            f"- project_turns_seen: {int(proj.get('turns', 0))}",
            f"- project_avg_quality: {float(proj.get('avg_quality', 0.0)):.2f}",
            f"- project_avg_context_quality: {float(proj.get('avg_context_quality', 0.0)):.2f}",
            f"- last_turn_at: {proj.get('last_turn_at', '') or 'n/a'}",
            f"- last_fact_refresh_at: {proj.get('last_fact_refresh_at', '') or 'n/a'}",
            f"- total_turns_seen: {int(stats.get('total_turns', 0))}",
            f"- global_avg_context_quality: {float(stats.get('avg_context_quality', 0.0)):.2f}",
            f"- auto_fact_refreshes: {int(stats.get('auto_fact_refreshes', 0))}",
            f"- auto_reinforce_up: {int(stats.get('auto_reinforce_up', 0))}",
            f"- auto_reinforce_down: {int(stats.get('auto_reinforce_down', 0))}",
        ]
        return "\n".join(lines)
