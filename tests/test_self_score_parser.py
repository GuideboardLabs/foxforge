from __future__ import annotations

import unittest

from tests.common import ROOT  # noqa: F401
from agents_research.deep_researcher import _extract_self_score


class SelfScoreParserTests(unittest.TestCase):
    def test_accepts_heading_prefixed_self_score_line(self) -> None:
        finding = (
            "## Findings\n"
            "- Point A\n"
            "- Point B\n\n"
            "# SELF_SCORE: confidence=0.85; coverage=0.75; notes=good coverage."
        )
        clean, score, err = _extract_self_score(finding)
        self.assertEqual(err, "")
        self.assertIsInstance(score, dict)
        self.assertAlmostEqual(float(score.get("confidence", 0.0)), 0.85, places=2)
        self.assertAlmostEqual(float(score.get("coverage", 0.0)), 0.75, places=2)
        self.assertNotIn("SELF_SCORE", clean)

    def test_accepts_conf_alias_and_comma_separators(self) -> None:
        finding = (
            "Observation text\n"
            "## Self Score: conf=0.8, coverage=0.7, notes=usable signal;"
        )
        clean, score, err = _extract_self_score(finding)
        self.assertEqual(err, "")
        self.assertIsInstance(score, dict)
        self.assertAlmostEqual(float(score.get("confidence", 0.0)), 0.8, places=2)
        self.assertAlmostEqual(float(score.get("coverage", 0.0)), 0.7, places=2)
        self.assertNotIn("Self Score", clean)

    def test_rejects_out_of_range_values(self) -> None:
        finding = "SELF_SCORE: confidence=1.2; coverage=0.8; notes=bad"
        _clean, score, err = _extract_self_score(finding)
        self.assertIsNone(score)
        self.assertIn("out of range", err)


if __name__ == "__main__":
    unittest.main()
