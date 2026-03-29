from __future__ import annotations

import shutil
import sqlite3
import unittest
from pathlib import Path

from tests.common import ROOT, ensure_runtime
from shared_tools.db import connect


class DbConnectionContextTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runtime_tmp = Path(ROOT) / "Runtime" / "test_db_context_tmp"
        if self.runtime_tmp.exists():
            shutil.rmtree(self.runtime_tmp, ignore_errors=True)
        self.repo_root = self.runtime_tmp / "repo"
        self.repo_root.mkdir(parents=True, exist_ok=True)
        ensure_runtime(self.repo_root)

    def tearDown(self) -> None:
        shutil.rmtree(self.runtime_tmp, ignore_errors=True)

    def test_connect_context_manager_closes_connection(self) -> None:
        with connect(self.repo_root) as conn:
            conn.execute("SELECT 1;").fetchone()

        with self.assertRaises(sqlite3.ProgrammingError):
            conn.execute("SELECT 1;")


if __name__ == "__main__":
    unittest.main()
