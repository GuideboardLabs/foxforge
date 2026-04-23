from __future__ import annotations

import unittest

from tests.common import ROOT  # noqa: F401
from agents_research.synthesizer import run_skeptic_pass


class _FakeClient:
    def __init__(self, body: str) -> None:
        self._body = body

    def chat(self, **_kwargs):
        return self._body


class _CaptureClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def chat(self, **kwargs):
        self.calls.append(dict(kwargs))
        return "<REVISED_SUMMARY>Revised summary</REVISED_SUMMARY><CRITIQUE_LOG>Audit notes</CRITIQUE_LOG>"


class SkepticPassContractTests(unittest.TestCase):
    def test_returns_revised_summary_and_critique_when_delimiter_present(self) -> None:
        revised, critique = run_skeptic_pass(
            question="q",
            synthesis="Base summary",
            client=_FakeClient("Revised summary\n---CRITIQUE---\nAudit notes"),
            model_cfg={"model": "dummy-model"},
            findings=[],
        )
        self.assertEqual(revised, "Revised summary")
        self.assertEqual(critique, "Audit notes")

    def test_fallback_edit_applied_when_delimiter_missing(self) -> None:
        # When the model returns no structured delimiter, the skeptic pass
        # runs a second "apply edits" call. The FakeClient returns the same
        # body both times, so the revised summary equals the client body.
        revised, critique = run_skeptic_pass(
            question="q",
            synthesis="Base summary",
            client=_FakeClient("Plain critique text only"),
            model_cfg={"model": "dummy-model"},
            findings=[],
        )
        self.assertEqual(revised, "Plain critique text only")
        self.assertEqual(critique, "Plain critique text only")

    def test_prompt_includes_raw_evidence_guard_and_extended_finding_context(self) -> None:
        long_finding = ("A" * 1200) + " [source: https://example.com/proof]"
        client = _CaptureClient()
        revised, critique = run_skeptic_pass(
            question="q",
            synthesis="Base summary",
            client=client,
            model_cfg={"model": "dummy-model"},
            findings=[{"agent": "critical_analyst", "finding": long_finding}],
        )
        self.assertIn("Revised summary", revised)
        self.assertIn("Source Anchors", revised)
        self.assertIn("(https://example.com/proof)", revised)
        self.assertEqual(critique, "Audit notes")
        self.assertGreaterEqual(len(client.calls), 1)
        first = client.calls[0]
        system_prompt = str(first.get("system_prompt", ""))
        user_prompt = str(first.get("user_prompt", ""))
        self.assertIn("Do NOT demote [E] just because the synthesis sentence itself dropped the link", system_prompt)
        self.assertIn("Only demote [E] -> [I]", system_prompt)
        self.assertIn("first 2000 chars per agent", user_prompt)
        self.assertIn("[source: https://example.com/proof]", user_prompt)


if __name__ == "__main__":
    unittest.main()
