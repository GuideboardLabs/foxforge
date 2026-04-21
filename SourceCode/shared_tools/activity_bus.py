from __future__ import annotations

import gzip
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Callable

LOGGER = logging.getLogger(__name__)
ActivityListener = Callable[[dict[str, Any]], None]
_TELEMETRY_LOCK = Lock()


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


def _utc_today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _date_from_name(path: Path, stem: str) -> datetime | None:
    name = path.name
    prefix = f"{stem}."
    if not name.startswith(prefix):
        return None
    suffix = name[len(prefix):]
    if suffix.endswith(".jsonl"):
        suffix = suffix[:-6]
    elif suffix.endswith(".jsonl.gz"):
        suffix = suffix[:-9]
    try:
        return datetime.strptime(suffix, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _gzip_file(path: Path) -> Path:
    gz_path = path.with_suffix(path.suffix + ".gz")
    with path.open("rb") as src, gzip.open(gz_path, "wb") as dst:
        while True:
            chunk = src.read(1024 * 1024)
            if not chunk:
                break
            dst.write(chunk)
    path.unlink(missing_ok=True)
    return gz_path


def _rotate_daily(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        return
    mtime_day = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).strftime("%Y-%m-%d")
    today = _utc_today()
    if mtime_day >= today:
        return
    rolled = path.with_name(f"{path.stem}.{mtime_day}.jsonl")
    if rolled.exists():
        rolled.unlink(missing_ok=True)
    path.rename(rolled)
    _gzip_file(rolled)


def _prune_telemetry(path: Path, *, retention_days: int = 14) -> None:
    cutoff = datetime.now(timezone.utc) - timedelta(days=max(1, int(retention_days)))
    stem = path.stem
    for candidate in path.parent.glob(f"{stem}.*.jsonl*"):
        dt = _date_from_name(candidate, stem)
        if dt is None:
            continue
        if dt < cutoff:
            candidate.unlink(missing_ok=True)


def telemetry_emit(
    repo_root: Path,
    filename: str,
    payload: dict[str, Any],
    *,
    retention_days: int = 14,
) -> Path:
    """Append one telemetry JSON line with daily rotation + gzip retention."""
    telemetry_dir = repo_root / "Runtime" / "telemetry"
    telemetry_dir.mkdir(parents=True, exist_ok=True)
    safe_name = str(filename or "").strip() or "events.jsonl"
    if not safe_name.endswith(".jsonl"):
        safe_name = f"{safe_name}.jsonl"
    path = telemetry_dir / safe_name
    row = dict(payload)
    row.setdefault("ts", datetime.now(timezone.utc).isoformat())
    with _TELEMETRY_LOCK:
        try:
            _rotate_daily(path)
            _prune_telemetry(path, retention_days=retention_days)
        except Exception:
            LOGGER.exception("Telemetry rotation failed for %s", path)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=True) + "\n")
    return path
