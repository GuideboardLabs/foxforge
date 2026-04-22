from __future__ import annotations

import unittest

from tests.common import ROOT  # noqa: F401
from orchestrator.services.make_catalog import stack_summary
from orchestrator.services.research_service import _is_stack_decided_question


class StackDecisionAndCatalogTests(unittest.TestCase):
    def test_stack_summary_includes_fixed_make_types(self) -> None:
        text = stack_summary()
        self.assertIn("tool", text)
        self.assertIn("web_app", text)
        self.assertIn("desktop_app", text)
        self.assertIn("Flask 3.x", text)
        self.assertIn("Vue 3.5", text)
        self.assertIn(".NET 8 LTS", text)
        self.assertIn("system-fixed", text.lower())

    def test_stack_decision_guard_triggers_outside_technical(self) -> None:
        self.assertTrue(_is_stack_decided_question("Should I use SQLite or Postgres for my app?", "general"))
        self.assertTrue(_is_stack_decided_question("Flask or FastAPI for this tracker?", "project"))
        self.assertTrue(_is_stack_decided_question("React vs Vue for this web app?", "general"))

    def test_stack_decision_guard_allows_technical_topic(self) -> None:
        self.assertFalse(_is_stack_decided_question("Should I use SQLite or Postgres for my app?", "technical"))
        self.assertFalse(_is_stack_decided_question("Flask or FastAPI for this tracker?", "technical"))


if __name__ == "__main__":
    unittest.main()
