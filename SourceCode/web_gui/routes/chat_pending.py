from __future__ import annotations

from typing import TYPE_CHECKING

from flask import Blueprint, request

from web_gui.utils.request_utils import parse_optional_int

if TYPE_CHECKING:
    from web_gui.app_context import AppContext


def register_pending_routes(bp: Blueprint, ctx: AppContext) -> None:
    @bp.route("/api/pending-actions", methods=["GET"])
    def pending_actions() -> tuple[dict, int]:
        profile = ctx.require_profile()
        limit = parse_optional_int(request.args.get("limit"), default=20, minimum=1, maximum=200)
        orch = ctx.new_orch(profile)
        actions = orch.pending_actions_data(limit=limit)
        return {"pending_actions": actions}, 200

    @bp.route("/api/pending-actions/<action_id>/ignore", methods=["POST"])
    def pending_action_ignore(action_id: str) -> tuple[dict, int]:
        profile = ctx.require_profile()
        payload = request.get_json(silent=True) or {}
        reason = str(payload.get("reason", "")).strip()
        orch = ctx.new_orch(profile)
        result = orch.ignore_pending_action(action_id=action_id, reason=reason)
        ctx.cache_clear(str(profile.get("id", "")))
        ok = result.lower().startswith("pending action ignored")
        return {"ok": ok, "message": result}, (200 if ok else 400)

    @bp.route("/api/pending-actions/<action_id>/answer", methods=["POST"])
    def pending_action_answer(action_id: str) -> tuple[dict, int]:
        profile = ctx.require_profile()
        payload = request.get_json(silent=True) or {}
        answer = str(payload.get("answer", "")).strip()
        if not answer:
            return {"ok": False, "message": "Answer text is required."}, 400
        orch = ctx.new_orch(profile)
        result = orch.answer_pending_action(action_id=action_id, answer=answer)
        ctx.cache_clear(str(profile.get("id", "")))
        low = result.lower()
        ok = (
            low.startswith("reflection answered and closed")
            or low.startswith("web research completed")
            or low.startswith("pending action ignored")
        )
        return {"ok": ok, "message": result}, (200 if ok else 400)

    @bp.route("/api/pending-actions/<action_id>/codex", methods=["POST"])
    def pending_action_codex(action_id: str) -> tuple[dict, int]:
        profile = ctx.require_profile()
        payload = request.get_json(silent=True) or {}
        note = str(payload.get("note", "")).strip()
        orch = ctx.new_orch(profile)
        result = orch.send_pending_action_to_codex(action_id=action_id, note=note)
        ctx.cache_clear(str(profile.get("id", "")))
        ok = result.lower().startswith("pending action routed to codex inbox")
        return {"ok": ok, "message": result}, (200 if ok else 400)

    @bp.route("/api/outbox/<target>/<thread_id>/ingest", methods=["POST"])
    def outbox_ingest(target: str, thread_id: str) -> tuple[dict, int]:
        profile = ctx.require_profile()
        payload = request.get_json(silent=True) or {}
        lane_hint = str(payload.get("lane", "")).strip() or None
        orch = ctx.new_orch(profile)
        result = orch.learn_outbox_one(target=target, thread_id=thread_id, lane_hint=lane_hint)
        ctx.cache_clear(str(profile.get("id", "")))
        ok = bool(result.get("ok", False))
        return result, (200 if ok else 400)
