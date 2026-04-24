from __future__ import annotations

import unittest

from tests.common import ROOT  # noqa: F401  # ensure SourceCode on sys.path
from agents_research.synthesizer import run_skeptic_pass_with_severity


class _StaticClient:
    def __init__(self, response: str) -> None:
        self.response = response

    def chat(self, *args, **kwargs):  # noqa: ANN002, ANN003
        _ = args, kwargs
        return self.response


class SkepticPassSeverityContractTests(unittest.TestCase):
    def test_three_block_xml_parses_severity(self) -> None:
        body = (
            "<REVISED_SUMMARY># Revised\n\nhello</REVISED_SUMMARY>\n"
            "<CRITIQUE_LOG>- tightened claims</CRITIQUE_LOG>\n"
            "<SEVERITY>{\"severity\":4,\"issues\":{\"fabricated_specifics\":1,\"unsupported_claims\":2,"
            "\"contradictions\":0,\"weak_evidence_caveats_missing\":1,\"missing_perspectives\":1,"
            "\"authority_misattribution\":0},\"conclusion_vulnerability\":\"high\","
            "\"recommended_action\":\"escalate_premium\",\"revise_focus\":[\"reduce certainty\"]}</SEVERITY>"
        )
        revised, critique, severity = run_skeptic_pass_with_severity(
            "Question?",
            "Base synthesis",
            client=_StaticClient(body),
            model_cfg={"model": "qwen3:8b"},
            findings=None,
        )
        self.assertIn("# Revised", revised)
        self.assertIn("tightened", critique)
        self.assertEqual(severity.get("severity"), 4)
        self.assertEqual(severity.get("recommended_action"), "escalate_premium")
        self.assertEqual(severity.get("conclusion_vulnerability"), "high")
        self.assertTrue(isinstance(severity.get("issues"), dict))

    def test_malformed_severity_json_falls_back_safely(self) -> None:
        body = (
            "<REVISED_SUMMARY># Revised\n\nhello</REVISED_SUMMARY>\n"
            "<CRITIQUE_LOG>- log</CRITIQUE_LOG>\n"
            "<SEVERITY>{not-json}</SEVERITY>"
        )
        _revised, _critique, severity = run_skeptic_pass_with_severity(
            "Question?",
            "Base synthesis",
            client=_StaticClient(body),
            model_cfg={"model": "qwen3:8b"},
            findings=None,
        )
        self.assertEqual(severity.get("severity"), 2)
        self.assertEqual(severity.get("recommended_action"), "revise_default")


if __name__ == "__main__":
    unittest.main()
