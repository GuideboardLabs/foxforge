from __future__ import annotations

import unittest

from tests.common import ROOT  # noqa: F401  # ensure SourceCode on sys.path
from shared_tools.loop_controller import should_escalate


class EscalationPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = {
            "enabled": True,
            "importance_min": "high",
            "severity_min": 3,
            "max_premium_passes": 1,
            "max_revise_loops": 2,
            "require_prior_default_pass": True,
            "cooloff_sec": 0,
        }

    def test_matrix_high_importance_high_severity_escalates(self) -> None:
        ok, reason = should_escalate(
            importance="high",
            severity=4,
            policy=self.policy,
            has_premium_tier=True,
            default_passes_done=1,
            lane_key="synthesis",
        )
        self.assertTrue(ok)
        self.assertEqual(reason, "policy_threshold_met")

    def test_low_importance_does_not_escalate(self) -> None:
        ok, reason = should_escalate(
            importance="medium",
            severity=5,
            policy=self.policy,
            has_premium_tier=True,
            default_passes_done=1,
            lane_key="synthesis",
        )
        self.assertFalse(ok)
        self.assertIn("importance", reason)

    def test_severity_five_high_importance_forces_escalation(self) -> None:
        ok, reason = should_escalate(
            importance="critical",
            severity=5,
            policy=self.policy,
            has_premium_tier=True,
            default_passes_done=1,
            lane_key="synthesis",
        )
        self.assertTrue(ok)
        self.assertEqual(reason, "severity_5_high_importance")

    def test_requires_prior_default_pass_when_configured(self) -> None:
        ok, reason = should_escalate(
            importance="high",
            severity=4,
            policy=self.policy,
            has_premium_tier=True,
            default_passes_done=0,
            lane_key="synthesis",
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "prior_default_pass_required")


if __name__ == "__main__":
    unittest.main()
