from __future__ import annotations

import unittest

from tests.common import ROOT  # noqa: F401
from agents_tool.tool_pool import _PYTHON_CLI_PATTERNS, _PYTHON_GOTCHAS, _TOOL_AGENTS
from agents_make.app_pool import (
    _FLASK_GOTCHAS,
    _FLASK_PATTERNS,
    _SQLITE_GOTCHAS,
    _SQLITE_PATTERNS,
    _VUE3_GOTCHAS,
    _VUE3_PATTERNS,
)
from agents_make.desktop_pool import (
    _AVALONIA_GOTCHAS,
    _AVALONIA_PATTERNS,
    _DOTNET_GOTCHAS,
    _DOTNET_PATTERNS,
    _REACTIVEUI_GOTCHAS,
    _REACTIVEUI_PATTERNS,
)


class GenerationStackPatternTests(unittest.TestCase):
    def test_tool_pool_patterns_and_gotchas_present(self) -> None:
        self.assertIn("Python 3.12+", _PYTHON_CLI_PATTERNS)
        self.assertIn("shell=True", _PYTHON_GOTCHAS)
        self.assertEqual(len(_TOOL_AGENTS), 2)
        for agent in _TOOL_AGENTS:
            directive = str(agent.get("directive", ""))
            self.assertIn("Python CLI patterns", directive)
            self.assertIn("Python gotchas", directive)

    def test_web_app_pool_patterns_are_pinned_and_modern(self) -> None:
        self.assertIn("Flask 3.0+", _FLASK_PATTERNS)
        self.assertIn("from flask_cors import CORS", _FLASK_PATTERNS)
        self.assertIn("vue@3.5/dist/vue.global.prod.js", _VUE3_PATTERNS)
        self.assertIn("sqlite3 stdlib on Python 3.12+", _SQLITE_PATTERNS)
        self.assertIn("PRAGMA journal_mode = WAL", _SQLITE_PATTERNS)
        self.assertIn("before_first_request", _FLASK_GOTCHAS)
        self.assertIn("v-for without :key", _VUE3_GOTCHAS)
        self.assertIn("foreign_keys = ON", _SQLITE_GOTCHAS)

    def test_desktop_pool_has_dotnet_avalonia_reactiveui_guidance(self) -> None:
        self.assertIn(".NET 8", _DOTNET_PATTERNS)
        self.assertIn("net8.0", _DOTNET_PATTERNS)
        self.assertIn("Avalonia 11", _AVALONIA_PATTERNS)
        self.assertIn("ReactiveUI", _REACTIVEUI_PATTERNS)
        self.assertIn("async void", _DOTNET_GOTCHAS)
        self.assertIn("x:DataType", _AVALONIA_GOTCHAS)
        self.assertIn("ThrownExceptions", _REACTIVEUI_GOTCHAS)


if __name__ == "__main__":
    unittest.main()
