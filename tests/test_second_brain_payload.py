from __future__ import annotations

import shutil
import unittest
from pathlib import Path

from tests.common import ROOT
from shared_tools.personal_memory import PersonalMemory


class SecondBrainPayloadTests(unittest.TestCase):
    def test_second_brain_payload_includes_memory_overview(self) -> None:
        import tempfile
        runtime_tmp = Path(tempfile.mkdtemp(prefix="foxforge_test_second_brain_"))
        repo_root = runtime_tmp / "repo"
        repo_root.mkdir(parents=True, exist_ok=True)

        pm = PersonalMemory(repo_root)
        pm.upsert_record(
            category="family",
            subject="Mia",
            field="birthday",
            value="2016-06-12",
            status="pinned",
            confidence=1.0,
            source_type="manual",
            source_label="life_admin",
            evidence="manual family entry",
        )
        pm.upsert_record(
            category="pet",
            subject="Scout",
            field="vet",
            value="Pine Street Vet",
            status="confirmed",
            confidence=0.92,
            source_type="manual",
            source_label="life_admin",
            evidence="manual pet entry",
        )
        payload = pm.second_brain_payload()

        self.assertEqual(payload["overview"]["active_records"], 2)
        self.assertEqual(payload["overview"]["pinned"], 1)
        self.assertIn("manual", payload["source_breakdown"])
        self.assertTrue(payload["briefing_lines"])
        self.assertTrue(payload["timeline"])
        shutil.rmtree(runtime_tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
