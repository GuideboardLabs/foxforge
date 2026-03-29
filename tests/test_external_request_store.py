from __future__ import annotations

import shutil
import unittest
from pathlib import Path

from tests.common import ROOT, ensure_runtime
from shared_tools.external_requests import (
    ExternalRequestStore,
    ExternalRequestTransitionError,
    ExternalRequestValidationError,
    ExternalToolsSettings,
)


class ExternalRequestStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runtime_tmp = Path(ROOT) / "Runtime" / "test_external_requests_tmp"
        if self.runtime_tmp.exists():
            shutil.rmtree(self.runtime_tmp, ignore_errors=True)
        self.repo_root = self.runtime_tmp / "repo"
        self.repo_root.mkdir(parents=True, exist_ok=True)
        ensure_runtime(self.repo_root)
        self.store = ExternalRequestStore(self.repo_root)
        self.settings = ExternalToolsSettings(self.repo_root)

    def tearDown(self) -> None:
        shutil.rmtree(self.runtime_tmp, ignore_errors=True)

    def test_settings_default_off_and_round_trip(self) -> None:
        self.assertEqual(self.settings.get_mode(), "off")
        self.assertEqual(self.settings.set_mode("ask"), "ask")
        self.assertEqual(self.settings.get_mode(), "ask")

    def test_create_validates_required_fields(self) -> None:
        with self.assertRaises(ExternalRequestValidationError):
            self.store.create({"provider": "openclaw", "intent": "send_email"})

    def test_create_and_get_round_trip(self) -> None:
        row = self.store.create(
            {
                "provider": "openclaw",
                "intent": "send_email",
                "project": "general",
                "lane": "project",
                "summary": "Send school update email",
                "payload_json": {"to": "ops@example.com"},
                "policy_json": {"requires_user_approval": True},
            }
        )
        loaded = self.store.get(str(row.get("id", "")))
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["provider"], "openclaw")
        self.assertEqual(loaded["intent"], "send_email")
        self.assertEqual(loaded["status"], "queued")

    def test_idempotency_on_id_and_external_ref(self) -> None:
        first = self.store.create(
            {
                "id": "ext_fixed_1",
                "provider": "openclaw",
                "intent": "send_email",
                "project": "general",
                "lane": "project",
                "summary": "Email 1",
            }
        )
        second = self.store.create(
            {
                "id": "ext_fixed_1",
                "provider": "openclaw",
                "intent": "send_email",
                "project": "general",
                "lane": "project",
                "summary": "Email 1 changed",
            }
        )
        self.assertEqual(first["id"], second["id"])

        with_ref_1 = self.store.create(
            {
                "provider": "openclaw",
                "intent": "send_email",
                "project": "general",
                "lane": "project",
                "summary": "Email with ref",
                "external_ref": "provider-123",
            }
        )
        with_ref_2 = self.store.create(
            {
                "provider": "openclaw",
                "intent": "send_email",
                "project": "general",
                "lane": "project",
                "summary": "Email with ref duplicate",
                "external_ref": "provider-123",
            }
        )
        self.assertEqual(with_ref_1["id"], with_ref_2["id"])

    def test_transition_state_machine_and_invalid_transition(self) -> None:
        row = self.store.create(
            {
                "provider": "openclaw",
                "intent": "task_suggestion",
                "project": "family",
                "lane": "project",
                "summary": "Suggest pickup task",
            }
        )
        rid = str(row["id"])
        row = self.store.transition_status(rid, "dispatched", note="queued for adapter")
        self.assertEqual(row["status"], "dispatched")
        row = self.store.transition_status(rid, "acknowledged")
        self.assertEqual(row["status"], "acknowledged")
        row = self.store.transition_status(rid, "working")
        self.assertEqual(row["status"], "working")
        row = self.store.mark_terminal(rid, "completed", {"summary": "done"})
        self.assertEqual(row["status"], "completed")
        self.assertTrue(str(row.get("completed_at", "")).strip())

        with self.assertRaises(ExternalRequestTransitionError):
            self.store.transition_status(rid, "working")

    def test_append_result_and_suggestions_count(self) -> None:
        row = self.store.create(
            {
                "provider": "openclaw",
                "intent": "event_suggestion",
                "project": "family",
                "lane": "project",
                "summary": "Suggest family event",
            }
        )
        rid = str(row["id"])
        updated = self.store.append_result(
            rid,
            {
                "summary": "Suggestion ready",
                "suggestions": [
                    {"type": "task", "title": "Book venue"},
                    {"type": "event", "title": "Family dinner"},
                ],
            },
        )
        self.assertEqual(updated["suggestions_count"], 2)
        open_rows = self.store.list_open(limit=20)
        hit = next((x for x in open_rows if x.get("id") == rid), None)
        self.assertIsNotNone(hit)
        self.assertEqual(int(hit.get("suggestions_count", 0)), 2)


if __name__ == "__main__":
    unittest.main()
