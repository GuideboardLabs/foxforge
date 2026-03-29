from __future__ import annotations

from typing import TYPE_CHECKING

from flask import Blueprint

from web_gui.routes.chat_conversations import register_conversation_routes
from web_gui.routes.chat_messages import register_message_routes
from web_gui.routes.chat_pending import register_pending_routes

if TYPE_CHECKING:
    from web_gui.app_context import AppContext


def create_chat_blueprint(ctx: AppContext) -> Blueprint:
    bp = Blueprint("chat_routes", __name__)
    register_conversation_routes(bp, ctx)
    register_pending_routes(bp, ctx)
    register_message_routes(bp, ctx)
    return bp
