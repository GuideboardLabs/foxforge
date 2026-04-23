from __future__ import annotations

import unittest

from web_gui.services.building_manager import BuildingManager
from web_gui.services.foraging_manager import ForagingManager
from web_gui.services.job_manager import JobManager


class LaneRunStateManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.profile = {"id": "owner"}
        self.job_manager = JobManager()

    def test_foraging_manager_keeps_last_successful_run_with_unread_state(self) -> None:
        manager = ForagingManager()
        request_id = self.job_manager.start(
            profile=self.profile,
            conversation_id="c1",
            request_id="job_forage_1",
            mode="command",
            user_text="Find latest topic guidance",
        )
        job_key = self.job_manager.key(self.profile, request_id)
        manager.register_job(
            profile=self.profile,
            conversation_id="c1",
            request_id=request_id,
            project="pets_project",
            lane="research",
            topic_type="animal_care",
            job_key=job_key,
        )

        active_rows = manager.rows_for_profile(self.profile, self.job_manager, limit=20)
        self.assertEqual(len(active_rows), 1)
        self.assertEqual(active_rows[0]["status"], "running")
        self.assertEqual(active_rows[0]["topic_type"], "animal_care")

        self.job_manager.finish(self.profile, request_id, status="completed", detail="ok")
        manager.unregister_job(job_key)
        manager.record_completion(
            profile=self.profile,
            conversation_id="c1",
            request_id=request_id,
            project="pets_project",
            lane="research",
            topic_type="animal_care",
            job_row=self.job_manager.get(self.profile, request_id),
            status="completed",
        )

        snapshot = manager.snapshot(profile_id="owner")
        self.assertTrue(snapshot["completion_unread"])
        self.assertEqual(snapshot["last_completed_id"], request_id)

        rows = manager.rows_for_profile(self.profile, self.job_manager, limit=20)
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0].get("is_last_successful"))
        self.assertEqual(rows[0]["status"], "completed")

        manager.mark_completion_read("owner")
        updated_snapshot = manager.snapshot(profile_id="owner")
        self.assertFalse(updated_snapshot["completion_unread"])

    def test_building_manager_ignores_non_success_completion_status(self) -> None:
        manager = BuildingManager()
        request_id = self.job_manager.start(
            profile=self.profile,
            conversation_id="c2",
            request_id="job_build_1",
            mode="command",
            user_text="Build release notes",
        )
        job_key = self.job_manager.key(self.profile, request_id)
        manager.register_job(
            profile=self.profile,
            conversation_id="c2",
            request_id=request_id,
            project="notes_project",
            make_type="brief",
            lane="build:brief",
            topic_type="general",
            job_key=job_key,
        )
        manager.unregister_job(job_key)
        self.job_manager.finish(self.profile, request_id, status="canceled", detail="cancelled")
        manager.record_completion(
            profile=self.profile,
            conversation_id="c2",
            request_id=request_id,
            project="notes_project",
            make_type="brief",
            lane="build:brief",
            topic_type="general",
            job_row=self.job_manager.get(self.profile, request_id),
            status="canceled",
        )

        snapshot = manager.snapshot(profile_id="owner")
        self.assertFalse(snapshot["completion_unread"])
        self.assertEqual(snapshot["last_completed_id"], "")


if __name__ == "__main__":
    unittest.main()
