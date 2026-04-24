from __future__ import annotations

import unittest

from tests.common import ROOT  # noqa: F401  # ensure SourceCode on sys.path
from shared_tools.model_routing import resolved_tier_config


class ModelRoutingTierResolutionTests(unittest.TestCase):
    def test_synthesizes_default_tier_from_legacy_keys(self) -> None:
        lane_cfg = {
            "model": "qwen3:8b",
            "num_ctx": 12288,
            "temperature": 0.2,
            "fallback_models": ["deepseek-r1:8b"],
            "timeout_sec": 480,
        }
        resolved = resolved_tier_config(lane_cfg, "default")
        self.assertIsInstance(resolved, dict)
        self.assertEqual(resolved.get("model"), "qwen3:8b")
        self.assertEqual(resolved.get("num_ctx"), 12288)
        self.assertEqual(resolved.get("temperature"), 0.2)
        self.assertEqual(resolved.get("fallback_models"), ["deepseek-r1:8b"])
        self.assertEqual(resolved.get("timeout_sec"), 480)

    def test_returns_explicit_default_tier_when_present(self) -> None:
        lane_cfg = {
            "model": "legacy",
            "tier_default": {"model": "qwen3:8b", "num_ctx": 16384},
        }
        resolved = resolved_tier_config(lane_cfg, "default")
        self.assertEqual(resolved, {"model": "qwen3:8b", "num_ctx": 16384})

    def test_missing_premium_tier_returns_none(self) -> None:
        lane_cfg = {"model": "qwen3:8b"}
        resolved = resolved_tier_config(lane_cfg, "premium")
        self.assertIsNone(resolved)


if __name__ == "__main__":
    unittest.main()
