from __future__ import annotations

import unittest
from unittest.mock import patch

from tests.common import ROOT  # noqa: F401
from web_gui.utils.login_limiter import LoginRateLimiter


class LoginRateLimiterTests(unittest.TestCase):
    def test_lockout_triggers_after_max_attempts_and_expires(self) -> None:
        limiter = LoginRateLimiter()
        with patch("time.time", return_value=1000.0):
            for _ in range(limiter.MAX_ATTEMPTS):
                limiter.record_failure("127.0.0.1", "owner")
            self.assertTrue(limiter.is_locked("127.0.0.1", "owner"))

        with patch("time.time", return_value=1000.0 + limiter.LOCKOUT_SECONDS + 1):
            self.assertFalse(limiter.is_locked("127.0.0.1", "owner"))

    def test_success_clears_attempt_history(self) -> None:
        limiter = LoginRateLimiter()
        with patch("time.time", return_value=1000.0):
            limiter.record_failure("127.0.0.1", "owner")
            limiter.record_success("127.0.0.1", "owner")
            self.assertFalse(limiter.is_locked("127.0.0.1", "owner"))


if __name__ == "__main__":
    unittest.main()
