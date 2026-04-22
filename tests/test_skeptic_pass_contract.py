from __future__ import annotations

import unittest

from tests.common import ROOT  # noqa: F401
from agents_research.synthesizer import run_skeptic_pass


class _FakeClient:
    def __init__(self, body: str) -> None:
        self._body = body

    def chat(self, **_kwargs):
        return self._body


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

    def test_preserves_summary_when_delimiter_missing(self) -> None:
        revised, critique = run_skeptic_pass(
            question="q",
            synthesis="Base summary",
            client=_FakeClient("Plain critique text only"),
            model_cfg={"model": "dummy-model"},
            findings=[],
        )
        self.assertEqual(revised, "Base summary")
        self.assertEqual(critique, "Plain critique text only")


if __name__ == "__main__":
    unittest.main()
