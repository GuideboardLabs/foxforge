from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Callable

LOGGER = logging.getLogger(__name__)
ActivityListener = Callable[[dict[str, Any]], None]


class ActivityBus:
    _shared_listeners: dict[str, list[ActivityListener]] = {}
    _shared_lock = Lock()

    def __init__(self, repo_root: Path) -> None:
        self.events_path = repo_root / "Runtime" / "activity" / "events.jsonl"
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        self.events_path.touch(exist_ok=True)
        self._key = str(self.events_path.resolve())
        self._lock = Lock()
        with self._shared_lock:
            self._shared_listeners.setdefault(self._key, [])

    def emit(self, actor: str, event: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
        row = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "actor": actor,
            "event": event,
            "details": details or {},
        }
        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
        self._notify(row)
        return row

    def subscribe(self, listener: ActivityListener) -> Callable[[], None]:
        with self._shared_lock:
            self._shared_listeners.setdefault(self._key, []).append(listener)

        def _unsubscribe() -> None:
            with self._shared_lock:
                listeners = self._shared_listeners.get(self._key, [])
                self._shared_listeners[self._key] = [item for item in listeners if item is not listener]

        return _unsubscribe

    def _notify(self, row: dict[str, Any]) -> None:
        with self._shared_lock:
            listeners = list(self._shared_listeners.get(self._key, []))
        for listener in listeners:
            try:
                listener(dict(row))
            except Exception:
                LOGGER.exception("Activity listener failed.")
