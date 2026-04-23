from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import patch

from tests.common import ROOT  # noqa: F401  # ensure SourceCode on sys.path
from orchestrator.services import chat_routing_gate


class _SemanticNoWeb:
    def classify(self, _text: str) -> dict[str, object]:
        return {"route": "no_web", "confidence": 0.88, "below_threshold": False}


class ChatRoutingGateRecencyNudgesTests(unittest.TestCase):
    def _route(self, text: str) -> str:
        with patch.object(chat_routing_gate, "_semantic_gate", return_value=_SemanticNoWeb()), patch.object(
            chat_routing_gate,
            "_emit_gate_decision",
            return_value=None,
        ):
            result = chat_routing_gate.check_web_routing(
                text,
                prior_messages=[],
                repo_root=Path(ROOT),
            )
        return str(result.get("route", "")).strip()

    def test_recency_nudges_and_dev_context_guards(self) -> None:
        cases = [
            ("What is currently happening in AI?", "web"),
            ("As of 2026, what changed in this market?", "web"),
            ("What happened this week in robotics?", "web"),
            ("Please update the current implementation of make_catalog.py", "no_web"),
            ("Show me the current code for this endpoint", "no_web"),
            ("hello", "no_web"),
            ("what's the latest news", "web"),
        ]
        for text, expected in cases:
            with self.subTest(text=text):
                self.assertEqual(self._route(text), expected)


if __name__ == "__main__":
    unittest.main()
