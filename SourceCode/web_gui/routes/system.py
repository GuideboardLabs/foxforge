from __future__ import annotations

from typing import TYPE_CHECKING

from flask import Blueprint

from web_gui.routes.system_auth import register_auth_routes
from web_gui.routes.system_health import register_health_routes
from web_gui.routes.system_owner import register_owner_routes
from web_gui.routes.system_panel import register_panel_routes
from web_gui.routes.system_settings import register_settings_routes

if TYPE_CHECKING:
    from web_gui.app_context import AppContext


def create_system_blueprint(ctx: AppContext) -> Blueprint:
    bp = Blueprint("system_routes", __name__)
    register_health_routes(bp, ctx)
    register_auth_routes(bp, ctx)
    register_settings_routes(bp, ctx)
    register_owner_routes(bp, ctx)
    register_panel_routes(bp, ctx)
    return bp
