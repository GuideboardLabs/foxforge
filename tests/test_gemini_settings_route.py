from __future__ import annotations

import os
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.common import ROOT, ensure_runtime


class GeminiSettingsRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ.setdefault("FOXFORGE_OWNER_PASSWORD", "test-password")
        os.environ.setdefault("FOXFORGE_AUTH_ENABLED", "0")
        from web_gui import app as appmod

        cls.appmod = appmod

    def setUp(self) -> None:
        self.runtime_tmp = Path(ROOT) / "Runtime" / "test_gemini_route_tmp"
        if self.runtime_tmp.exists():
            shutil.rmtree(self.runtime_tmp, ignore_errors=True)
        self.repo_root = self.runtime_tmp / "repo"
        self.repo_root.mkdir(parents=True, exist_ok=True)
        ensure_runtime(self.repo_root)

        self.original_root = self.appmod.ROOT
        self.original_background = self.appmod._ensure_background_services_started
        self.appmod.ROOT = self.repo_root
        self.appmod._ensure_background_services_started = lambda _app=None: None
        self.app = self.appmod.create_app()

    def tearDown(self) -> None:
        self.appmod.ROOT = self.original_root
        self.appmod._ensure_background_services_started = self.original_background
        shutil.rmtree(self.runtime_tmp, ignore_errors=True)

    def test_gemini_critique_route_round_trip(self) -> None:
        with patch.dict(os.environ, {"GEMINI_API_KEY": ""}, clear=False), self.app.test_client() as client:
            before = client.get("/api/settings/gemini-critique")
            self.assertEqual(before.status_code, 200)
            self.assertTrue(before.get_json()["ok"])
            self.assertFalse(before.get_json()["enabled"])

            response = client.post(
                "/api/settings/gemini-critique",
                json={"enabled": True, "api_keys": ["route-key"]},
            )
            self.assertEqual(response.status_code, 200)
            payload = response.get_json()
            self.assertTrue(payload["ok"])
            self.assertTrue(payload["enabled"])
            self.assertEqual(payload["api_keys_count"], 1)

            after = client.get("/api/settings/gemini-critique").get_json()
            self.assertTrue(after["enabled"])
            self.assertEqual(after["api_keys_count"], 1)


if __name__ == "__main__":
    unittest.main()
