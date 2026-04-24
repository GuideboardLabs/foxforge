from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from shared_tools.activity_bus import telemetry_emit
from shared_tools.model_routing import load_model_routing

try:  # pragma: no cover - windows fallback path
    import fcntl  # type: ignore
except Exception:  # pragma: no cover
    fcntl = None  # type: ignore


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class PremiumLease:
    model: str
    wait_ms: float
    acquired_at: str
    swapped_from: str = ""
    managed: bool = True


class PremiumModelLock:
    """Process-wide + filesystem-sentinel mutex for premium model residency."""

    _PROCESS_LOCK = threading.RLock()
    _STALE_TTL_SEC = 600.0

    def __init__(
        self,
        repo_root: Path,
        *,
        client: Any | None = None,
        premium_models: list[str] | None = None,
    ) -> None:
        self.repo_root = Path(repo_root)
        self.client = client
        self._state_path = self.repo_root / "var" / "state" / "premium_lock.json"
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        self._owner_pid = os.getpid()
        self._owner_token = f"{self._owner_pid}:{id(self)}"
        if premium_models is None:
            routing = load_model_routing(self.repo_root)
            raw = routing.get("premium_models", []) if isinstance(routing, dict) else []
            premium_models = [str(x).strip() for x in raw if str(x).strip()] if isinstance(raw, list) else []
        self.premium_models = set(str(x).strip() for x in (premium_models or []) if str(x).strip())

    def _is_premium_model(self, model: str) -> bool:
        name = str(model or "").strip()
        if not name:
            return False
        if name in self.premium_models:
            return True
        low = name.lower()
        return ":14b" in low or ":32b" in low

    def is_premium_model(self, model: str) -> bool:
        return self._is_premium_model(model)

    def _with_state_lock(self, fn):
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._state_path, "a+", encoding="utf-8") as fh:
            if fcntl is not None:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            try:
                fh.seek(0)
                raw = fh.read()
                state = {}
                if raw.strip():
                    try:
                        parsed = json.loads(raw)
                        if isinstance(parsed, dict):
                            state = dict(parsed)
                    except Exception:
                        state = {}
                result, new_state = fn(state)
                if isinstance(new_state, dict):
                    fh.seek(0)
                    fh.truncate(0)
                    fh.write(json.dumps(new_state, ensure_ascii=True, indent=2))
                    fh.flush()
                    os.fsync(fh.fileno())
                return result
            finally:
                if fcntl is not None:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _state_model(state: dict[str, Any]) -> str:
        return str(state.get("model", "")).strip()

    @staticmethod
    def _state_updated_ts(state: dict[str, Any]) -> float:
        try:
            return float(state.get("updated_ts", 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def _emit(self, action: str, *, model: str, wait_ms: float, prev_model: str = "") -> None:
        try:
            telemetry_emit(
                self.repo_root,
                "premium_lock.jsonl",
                {
                    "action": str(action or "").strip(),
                    "model": str(model or "").strip(),
                    "wait_ms": round(float(wait_ms), 3),
                    "prev_model": str(prev_model or "").strip(),
                    "owner_pid": self._owner_pid,
                },
                retention_days=30,
            )
        except Exception:
            pass

    def _release_model(self, model: str) -> None:
        name = str(model or "").strip()
        if not name:
            return
        try:
            if self.client is not None and hasattr(self.client, "release_model"):
                self.client.release_model(name)
                return
            if self.client is not None and hasattr(self.client, "_ollama"):
                inner = getattr(self.client, "_ollama", None)
                if inner is not None and hasattr(inner, "release_model"):
                    inner.release_model(name)
        except Exception:
            pass

    def acquire(self, model: str, timeout_sec: float = 180.0) -> PremiumLease:
        name = str(model or "").strip()
        if not name:
            raise ValueError("model is required for premium lock acquisition")
        if not self._is_premium_model(name):
            return PremiumLease(model=name, wait_ms=0.0, acquired_at=_now_iso(), swapped_from="", managed=False)

        started = time.monotonic()
        swapped_from = ""
        while True:
            elapsed = time.monotonic() - started
            if elapsed > max(1.0, float(timeout_sec)):
                raise TimeoutError(f"Timed out waiting for premium model lock: model={name}")

            with self._PROCESS_LOCK:
                now_epoch = time.time()

                def _mutate(state: dict[str, Any]):
                    nonlocal swapped_from
                    current_model = self._state_model(state)
                    current_owner = str(state.get("owner_token", "")).strip()
                    stale = bool(current_model) and ((now_epoch - self._state_updated_ts(state)) > self._STALE_TTL_SEC)
                    if stale:
                        current_model = ""
                        current_owner = ""
                        state = {}
                    if current_model and current_owner and current_owner != self._owner_token:
                        return False, state
                    swapped_from = current_model if (current_model and current_model != name) else ""
                    new_state = {
                        "model": name,
                        "owner_pid": self._owner_pid,
                        "owner_token": self._owner_token,
                        "updated_ts": now_epoch,
                        "updated_at": _now_iso(),
                    }
                    return True, new_state

                acquired = bool(self._with_state_lock(_mutate))
            if acquired:
                wait_ms = max(0.0, (time.monotonic() - started) * 1000.0)
                if swapped_from:
                    self._release_model(swapped_from)
                    time.sleep(3.0)
                    self._emit("swap", model=name, wait_ms=wait_ms, prev_model=swapped_from)
                else:
                    self._emit("acquire", model=name, wait_ms=wait_ms, prev_model="")
                return PremiumLease(
                    model=name,
                    wait_ms=wait_ms,
                    acquired_at=_now_iso(),
                    swapped_from=swapped_from,
                    managed=True,
                )
            time.sleep(0.1)

    def release(self, lease: PremiumLease, *, force_unload: bool = True) -> None:
        if not isinstance(lease, PremiumLease):
            return
        if not bool(getattr(lease, "managed", True)):
            return
        model = str(lease.model or "").strip()
        if not model:
            return
        with self._PROCESS_LOCK:
            now_epoch = time.time()

            def _mutate(state: dict[str, Any]):
                current_model = self._state_model(state)
                current_owner = str(state.get("owner_token", "")).strip()
                if current_model == model and current_owner == self._owner_token:
                    return True, {
                        "model": "",
                        "owner_pid": 0,
                        "owner_token": "",
                        "updated_ts": now_epoch,
                        "updated_at": _now_iso(),
                    }
                return False, state

            released = bool(self._with_state_lock(_mutate))
        if force_unload:
            self._release_model(model)
        self._emit("release", model=model, wait_ms=float(lease.wait_ms or 0.0), prev_model=lease.swapped_from if released else "")
