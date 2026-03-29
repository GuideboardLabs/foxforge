from __future__ import annotations

from typing import TYPE_CHECKING

from flask import Blueprint, request

from web_gui.utils.request_utils import parse_optional_int

if TYPE_CHECKING:
    from web_gui.app_context import AppContext


def create_watchtower_blueprint(ctx: AppContext) -> Blueprint:
    bp = Blueprint('watchtower_routes', __name__)

    @bp.route('/api/watchtower/watches', methods=['GET'])
    def watchtower_list_watches() -> tuple[dict, int]:
        ctx.require_profile()
        wt = ctx.get_watchtower()
        return {"watches": wt.list_watches()}, 200

    @bp.route('/api/watchtower/watches', methods=['POST'])
    def watchtower_add_watch() -> tuple[dict, int]:
        ctx.require_profile()
        payload = request.get_json(silent=True) or {}
        wt = ctx.get_watchtower()
        try:
            watch = wt.add_watch(
                topic=str(payload.get("topic", "")).strip(),
                profile=str(payload.get("profile", "general")).strip(),
                schedule=str(payload.get("schedule", "daily")).strip(),
                schedule_hour=int(payload.get("schedule_hour", 7)),
            )
        except ValueError as exc:
            return {"ok": False, "message": str(exc)}, 400
        return {"ok": True, "watch": watch}, 200

    @bp.route('/api/watchtower/watches/<watch_id>', methods=['PUT'])
    def watchtower_update_watch(watch_id: str) -> tuple[dict, int]:
        ctx.require_profile()
        payload = request.get_json(silent=True) or {}
        wt = ctx.get_watchtower()
        updated = wt.update_watch(watch_id, **{k: v for k, v in payload.items()})
        if updated is None:
            return {"ok": False, "message": "Watch not found."}, 404
        return {"ok": True, "watch": updated}, 200

    @bp.route('/api/watchtower/watches/<watch_id>', methods=['DELETE'])
    def watchtower_delete_watch(watch_id: str) -> tuple[dict, int]:
        ctx.require_profile()
        wt = ctx.get_watchtower()
        ok = wt.delete_watch(watch_id)
        if not ok:
            return {"ok": False, "message": "Watch not found."}, 404
        return {"ok": True}, 200

    @bp.route('/api/watchtower/watches/<watch_id>/trigger', methods=['POST'])
    def watchtower_trigger_watch(watch_id: str) -> tuple[dict, int]:
        ctx.require_profile()
        wt = ctx.get_watchtower()
        try:
            result = wt.trigger_watch(watch_id)
        except ValueError as exc:
            return {"ok": False, "message": str(exc)}, 404
        return {"ok": True, "watch": result}, 200

    @bp.route('/api/panel/briefings', methods=['GET'])
    def panel_briefings() -> tuple[dict, int]:
        ctx.require_profile()
        limit = parse_optional_int(request.args.get("limit"), default=50, minimum=1, maximum=200)
        wt = ctx.get_watchtower()
        briefings = wt.list_briefings(limit=limit)
        return {"briefings": briefings, "unread_count": wt.unread_count()}, 200

    @bp.route('/api/briefings/<briefing_id>', methods=['GET'])
    def briefing_get(briefing_id: str) -> tuple[dict, int]:
        ctx.require_profile()
        wt = ctx.get_watchtower()
        row = wt.get_briefing(briefing_id)
        if row is None:
            return {"ok": False, "message": "Briefing not found."}, 404
        return {"ok": True, "briefing": row}, 200

    @bp.route('/api/briefings/<briefing_id>/read', methods=['POST'])
    def mark_briefing_read(briefing_id: str) -> tuple[dict, int]:
        ctx.require_profile()
        wt = ctx.get_watchtower()
        ok = wt.mark_read(briefing_id)
        return {"ok": ok}, 200

    @bp.route('/api/briefings/<briefing_id>/unread', methods=['POST'])
    def mark_briefing_unread(briefing_id: str) -> tuple[dict, int]:
        ctx.require_profile()
        wt = ctx.get_watchtower()
        ok = wt.mark_unread(briefing_id)
        return {"ok": ok}, 200

    return bp
