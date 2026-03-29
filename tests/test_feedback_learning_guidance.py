from __future__ import annotations

import shutil
import unittest
from datetime import datetime, timezone
from pathlib import Path

from tests.common import ROOT, ensure_runtime
from shared_tools.db import connect, transaction
from shared_tools.feedback_learning import FeedbackLearningEngine


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class FeedbackLearningGuidanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runtime_tmp = Path(ROOT) / "Runtime" / "test_feedback_guidance_tmp"
        if self.runtime_tmp.exists():
            shutil.rmtree(self.runtime_tmp, ignore_errors=True)
        self.repo_root = self.runtime_tmp / "repo"
        self.repo_root.mkdir(parents=True, exist_ok=True)
        ensure_runtime(self.repo_root)
        self.engine = FeedbackLearningEngine(self.repo_root)

    def tearDown(self) -> None:
        shutil.rmtree(self.runtime_tmp, ignore_errors=True)

    def _insert_lesson(
        self,
        *,
        lesson_id: str,
        lane: str,
        summary: str,
        guidance: str,
        origin_type: str,
        status: str,
        confidence: float,
        active: int,
    ) -> None:
        ts = _now_iso()
        with connect(self.repo_root) as conn, transaction(conn, immediate=True):
            conn.execute(
                """
                INSERT INTO lessons(
                    id, lane, project, summary, guidance, origin_type, source, status,
                    confidence, active, approved_by, created_at, updated_at, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """.strip(),
                (
                    lesson_id,
                    lane,
                    "general",
                    summary,
                    guidance,
                    origin_type,
                    "test",
                    status,
                    confidence,
                    active,
                    "owner" if status == "approved" else None,
                    ts,
                    ts,
                    None,
                ),
            )

    def test_guidance_uses_high_confidence_candidate_fallback_when_no_approved(self) -> None:
        self._insert_lesson(
            lesson_id="lsn_candidate_1",
            lane="research",
            summary="Use event-timeline framing for post synthesis.",
            guidance="Trigger: when drafting updates\nDo: include timeline\nAvoid: vague recap",
            origin_type="reflection",
            status="candidate",
            confidence=0.91,
            active=0,
        )
        guidance = self.engine.guidance_for_lane("research", limit=5)
        self.assertIn("high-confidence candidate feedback", guidance)
        self.assertIn("lsn_candidate_1", guidance)
        self.assertIn("timeline", guidance.lower())

    def test_guidance_prefers_approved_rows_when_available(self) -> None:
        self._insert_lesson(
            lesson_id="lsn_candidate_2",
            lane="research",
            summary="Candidate summary",
            guidance="Candidate guidance",
            origin_type="reflection",
            status="candidate",
            confidence=0.95,
            active=0,
        )
        self._insert_lesson(
            lesson_id="lsn_approved_1",
            lane="research",
            summary="Approved summary",
            guidance="Approved guidance",
            origin_type="manual_feedback",
            status="approved",
            confidence=0.7,
            active=1,
        )
        guidance = self.engine.guidance_for_lane("research", limit=5)
        self.assertIn("approved feedback", guidance)
        self.assertIn("lsn_approved_1", guidance)
        self.assertNotIn("high-confidence candidate feedback", guidance)


if __name__ == "__main__":
    unittest.main()
