from __future__ import annotations

import logging
import unittest

from tests.common import ROOT  # noqa: F401
from agents_research import deep_researcher


class DeepResearcherLoggingTests(unittest.TestCase):
    def test_module_exposes_logger_for_error_paths(self) -> None:
        self.assertTrue(hasattr(deep_researcher, "LOGGER"))
        self.assertIsInstance(deep_researcher.LOGGER, logging.Logger)


if __name__ == "__main__":
    unittest.main()
