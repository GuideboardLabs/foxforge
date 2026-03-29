from __future__ import annotations

from typing import TYPE_CHECKING

from flask import Blueprint, request

if TYPE_CHECKING:
    from web_gui.app_context import AppContext


def register_panel_routes(bp: Blueprint, ctx: AppContext) -> None:
    @bp.get("/api/panel/reflections")
    def panel_reflections() -> tuple[dict, int]:
        from web_gui.utils.request_utils import parse_optional_int

        profile = ctx.require_profile()
        limit = parse_optional_int(request.args.get("limit"), default=20, minimum=1, maximum=200)
        orch = ctx.new_orch(profile)
        rows = orch.reflection_engine.list_open(limit=limit)
        return {"reflections": rows}, 200

    @bp.get("/api/panel/reflections-history")
    def panel_reflections_history() -> tuple[dict, int]:
        from web_gui.utils.request_utils import parse_optional_int

        profile = ctx.require_profile()
        limit = parse_optional_int(request.args.get("limit"), default=80, minimum=1, maximum=300)
        orch = ctx.new_orch(profile)
        rows = orch.reflection_engine.list_history(limit=limit)
        return {"reflections": rows}, 200

    @bp.get("/api/panel/lessons")
    def panel_lessons() -> tuple[dict, int]:
        from web_gui.utils.request_utils import parse_optional_int

        profile = ctx.require_profile()
        lane = str(request.args.get("lane", "")).strip() or None
        limit = parse_optional_int(request.args.get("limit"), default=25, minimum=1, maximum=200)
        sort_by = str(request.args.get("sort", "newest")).strip().lower() or "newest"
        orch = ctx.new_orch(profile)
        rows = orch.learning_engine.list_lessons(lane=lane, limit=limit, sort_by=sort_by)
        return {"lessons": rows}, 200

    @bp.post("/api/lessons/<lesson_id>/approve")
    def lesson_approve(lesson_id: str) -> tuple[dict, int]:
        profile = ctx.require_profile()
        orch = ctx.new_orch(profile)
        approved_by = str(profile.get("username", "")).strip() or "owner"
        row = orch.learning_engine.approve_lesson(lesson_id=lesson_id, approved_by=approved_by)
        if row is None:
            return {"ok": False, "message": "Lesson not found."}, 404
        if bool(row.get("policy_blocked", False)):
            return {"ok": False, "message": str(row.get("policy_message", "Lesson cannot be approved.")), "lesson": row}, 400
        return {"ok": True, "message": "Lesson approved.", "lesson": row}, 200

    @bp.post("/api/lessons/<lesson_id>/reject")
    def lesson_reject(lesson_id: str) -> tuple[dict, int]:
        profile = ctx.require_profile()
        orch = ctx.new_orch(profile)
        rejected_by = str(profile.get("username", "")).strip() or "owner"
        row = orch.learning_engine.reject_lesson(lesson_id=lesson_id, rejected_by=rejected_by)
        if row is None:
            return {"ok": False, "message": "Lesson not found."}, 404
        return {"ok": True, "message": "Lesson rejected.", "lesson": row}, 200

    @bp.post("/api/lessons/<lesson_id>/expire")
    def lesson_expire(lesson_id: str) -> tuple[dict, int]:
        profile = ctx.require_profile()
        orch = ctx.new_orch(profile)
        row = orch.learning_engine.expire_lesson(lesson_id=lesson_id)
        if row is None:
            return {"ok": False, "message": "Lesson not found."}, 404
        return {"ok": True, "message": "Lesson expired.", "lesson": row}, 200

    @bp.get("/api/panel/handoffs")
    def panel_handoffs() -> tuple[dict, int]:
        from web_gui.utils.request_utils import parse_optional_int

        profile = ctx.require_profile()
        limit = parse_optional_int(request.args.get("limit"), default=20, minimum=1, maximum=200)
        orch = ctx.new_orch(profile)
        rows = orch.handoff_queue.monitor_threads(limit=limit)
        return {"handoffs": rows}, 200

    @bp.get("/api/panel/outbox")
    def panel_outbox() -> tuple[dict, int]:
        from web_gui.utils.request_utils import parse_optional_int

        profile = ctx.require_profile()
        limit = parse_optional_int(request.args.get("limit"), default=40, minimum=1, maximum=300)
        orch = ctx.new_orch(profile)
        rows = orch.handoff_queue.monitor_threads(limit=500)
        outbox_rows = [row for row in rows if str(row.get("outbox_path", "")).strip()]
        outbox_rows.sort(key=lambda row: str(row.get("created_at", "")), reverse=True)
        return {"outbox": outbox_rows[:limit]}, 200

    @bp.get("/api/panel/projects")
    def panel_projects() -> tuple[dict, int]:
        from web_gui.routes.projects import _project_panel_rows
        from web_gui.utils.request_utils import parse_optional_int

        profile = ctx.require_profile()
        cache_scope = str(profile.get("id", ""))
        limit = parse_optional_int(request.args.get("limit"), default=40, minimum=1, maximum=200)
        orch = ctx.new_orch(profile)
        rows = ctx.cache_get(
            cache_scope,
            f"panel_projects:{limit}",
            ttl_sec=2.0,
            build_fn=lambda: _project_panel_rows(ctx, orch, limit=limit),
        )
        return {"projects": rows}, 200

    @bp.get("/api/panel/foraging")
    def panel_foraging() -> tuple[dict, int]:
        from web_gui.utils.request_utils import parse_optional_int

        profile = ctx.require_profile()
        limit = parse_optional_int(request.args.get("limit"), default=60, minimum=1, maximum=200)
        snapshot = ctx.foraging_manager.snapshot(profile_id=str(profile.get("id", "")))
        jobs = ctx.foraging_manager.rows_for_profile(profile, ctx.job_manager, limit=limit)
        return {"foraging": snapshot, "jobs": jobs}, 200
