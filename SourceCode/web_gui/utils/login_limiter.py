from __future__ import annotations

import threading
from typing import Any


class LoginRateLimiter:
    """In-memory rate limiter for login attempts. Tracks failures per (IP, username) pair."""

    MAX_ATTEMPTS = 10
    WINDOW_SECONDS = 300   # 5 minutes
    LOCKOUT_SECONDS = 900  # 15 minutes

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._attempts: dict[str, list[float]] = {}
        self._locked_until: dict[str, float] = {}

    def _key(self, ip: str, username: str) -> str:
        return f"{ip}|{username.lower()}"

    def is_locked(self, ip: str, username: str) -> bool:
        import time
        key = self._key(ip, username)
        with self._lock:
            until = self._locked_until.get(key, 0)
            if until:
                if time.time() < until:
                    return True
                del self._locked_until[key]
        return False

    def record_failure(self, ip: str, username: str) -> None:
        import time
        key = self._key(ip, username)
        now = time.time()
        with self._lock:
            window_start = now - self.WINDOW_SECONDS
            attempts = [t for t in self._attempts.get(key, []) if t > window_start]
            attempts.append(now)
            if len(attempts) >= self.MAX_ATTEMPTS:
                self._locked_until[key] = now + self.LOCKOUT_SECONDS
                self._attempts.pop(key, None)
            else:
                self._attempts[key] = attempts

    def record_success(self, ip: str, username: str) -> None:
        key = self._key(ip, username)
        with self._lock:
            self._attempts.pop(key, None)
            self._locked_until.pop(key, None)
