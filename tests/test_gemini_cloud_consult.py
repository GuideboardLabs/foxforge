from __future__ import annotations

import json
import os
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.common import ROOT, ensure_runtime
from shared_tools.cloud_consult import CloudConsultEngine


class GeminiCloudConsultTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runtime_tmp = Path(ROOT) / "Runtime" / "test_gemini_cloud_tmp"
        if self.runtime_tmp.exists():
            shutil.rmtree(self.runtime_tmp, ignore_errors=True)
        self.repo_root = self.runtime_tmp / "repo"
        self.repo_root.mkdir(parents=True, exist_ok=True)
        ensure_runtime(self.repo_root)

    def tearDown(self) -> None:
        shutil.rmtree(self.runtime_tmp, ignore_errors=True)

    def test_critique_settings_round_trip_counts_stored_and_env_keys(self) -> None:
        engine = CloudConsultEngine(self.repo_root)
        with patch.dict(os.environ, {"GEMINI_API_KEY": "env-key"}, clear=False):
            settings = engine.set_critique_settings(enabled=True, api_keys=["stored-key"])
            self.assertTrue(settings["enabled"])
            self.assertEqual(settings["api_keys_count"], 2)
            self.assertEqual(settings["model"], "gemini-2.0-flash")

        raw = json.loads((self.repo_root / "Runtime" / "cloud" / "settings.json").read_text(encoding="utf-8"))
        self.assertTrue(raw["gemini_critique_enabled"])
        self.assertEqual(raw["gemini_api_keys"], ["stored-key"])

    def test_claim_check_research_summary_records_successful_gemini_result(self) -> None:
        engine = CloudConsultEngine(self.repo_root)
        engine.set_critique_settings(enabled=True, api_keys=["stored-key"])
        fake_response = json.dumps(
            [
                {
                    "claim": "Dogs need regular exercise.",
                    "verdict": "supported",
                    "reason": "The excerpt explicitly states dogs benefit from daily exercise.",
                    "evidence_refs": [1],
                }
            ]
        )
        sources = [
            {
                "title": "Vet Guidance",
                "source_domain": "example.org",
                "snippet": "Dogs benefit from daily exercise and regular movement for health.",
            }
        ]

        with patch.object(engine, "_call_provider_with_retry", return_value=(fake_response, "gemini-2.0-flash")):
            result = engine.claim_check_research_summary(
                project="family",
                query="How much exercise does a dog need?",
                sources=sources,
                source_path="Projects/family/research.md",
                claims=["Dogs need regular exercise."],
            )

        self.assertTrue(result["ok"])
        self.assertEqual(len(result["claim_checks"]), 1)
        self.assertEqual(result["claim_checks"][0]["verdict"], "supported")

        runs = engine.recent_runs_for_project("family", limit=5)
        self.assertTrue(runs)
        self.assertEqual(runs[0]["provider"], "gemini")
        self.assertEqual(runs[0]["status"], "completed")

    def test_claim_check_research_summary_returns_failure_when_provider_raises(self) -> None:
        engine = CloudConsultEngine(self.repo_root)
        engine.set_critique_settings(enabled=True, api_keys=["stored-key"])
        sources = [
            {
                "title": "Vet Guidance",
                "source_domain": "example.org",
                "snippet": "Cats can hide pain, so subtle behavior changes matter.",
            }
        ]

        with patch.object(engine, "_call_provider_with_retry", side_effect=RuntimeError("HTTP 429")):
            result = engine.claim_check_research_summary(
                project="family",
                query="How can I tell if my cat is in pain?",
                sources=sources,
                source_path="Projects/family/cats.md",
                claims=["Cats hide pain."],
            )

        self.assertFalse(result["ok"])
        self.assertIn("Gemini claim-check failed", result["error"])


if __name__ == "__main__":
    unittest.main()
