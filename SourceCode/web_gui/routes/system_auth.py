from __future__ import annotations

from typing import TYPE_CHECKING

from flask import Blueprint, request, session

if TYPE_CHECKING:
    from web_gui.app_context import AppContext


def register_auth_routes(bp: Blueprint, ctx: AppContext) -> None:
    @bp.route("/api/auth/status", methods=["GET"])
    def auth_status() -> tuple[dict, int]:
        profile = ctx.session_profile()
        return {
            "enabled": ctx.auth_enabled,
            "authenticated": profile is not None,
            "profile": ctx.public_profile(profile),
        }, 200

    @bp.route("/api/auth/login", methods=["POST"])
    def auth_login() -> tuple[dict, int]:
        if not ctx.auth_enabled:
            session.permanent = True
            session["authenticated"] = True
            session["user_id"] = ctx.owner_id
            return {
                "ok": True,
                "enabled": False,
                "authenticated": True,
                "profile": ctx.public_profile(ctx.owner_profile),
            }, 200
        payload = request.get_json(silent=True) or {}
        username = str(payload.get("username", "")).strip()
        password = str(payload.get("password", ""))
        client_ip = str(request.remote_addr or "")
        if ctx.login_limiter.is_locked(client_ip, username):
            return {"ok": False, "error": "Too many failed attempts. Please try again later."}, 429
        profile = ctx.auth_store.verify_login(username=username, password=password)
        if profile is None:
            ctx.login_limiter.record_failure(client_ip, username)
            return {"ok": False, "error": "Invalid username or PIN"}, 401
        ctx.login_limiter.record_success(client_ip, username)
        session.permanent = True
        session["authenticated"] = True
        session["user_id"] = str(profile.get("id", "")).strip()
        return {
            "ok": True,
            "enabled": True,
            "authenticated": True,
            "profile": ctx.public_profile(profile),
        }, 200

    @bp.route("/api/auth/logout", methods=["POST"])
    def auth_logout() -> tuple[dict, int]:
        session.clear()
        return {"ok": True}, 200
