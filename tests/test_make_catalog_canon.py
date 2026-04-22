from __future__ import annotations

import unittest

from tests.common import ROOT  # noqa: F401
from orchestrator.services.make_catalog import MAKE_CATALOG


class MakeCatalogCanonTests(unittest.TestCase):
    def test_web_app_catalog_entry_has_canon_scaffold_path(self) -> None:
        entry = MAKE_CATALOG["web_app"]
        self.assertEqual(entry.get("scaffold_path"), "canon/web_app_v1")


if __name__ == "__main__":
    unittest.main()
