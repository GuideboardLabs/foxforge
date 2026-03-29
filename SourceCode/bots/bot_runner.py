from __future__ import annotations

import logging
import threading
from pathlib import Path

from bots.bot_config import load_bot_config

LOGGER = logging.getLogger(__name__)

_running_bots: list[threading.Thread] = []
_bots_lock = threading.Lock()


def start_bots(repo_root: Path) -> None:
    cfg = load_bot_config(repo_root)

    tg_cfg = cfg.get("telegram", {})
    if tg_cfg.get("enabled") and tg_cfg.get("bot_token"):
        try:
            from bots.telegram_bot import TelegramBot
            bot = TelegramBot(repo_root, tg_cfg["bot_token"])
            bot.start()
            with _bots_lock:
                _running_bots.append(bot)
            LOGGER.info("Telegram bot thread started.")
        except Exception:
            LOGGER.exception("Failed to start Telegram bot.")

    dc_cfg = cfg.get("discord", {})
    if dc_cfg.get("enabled") and dc_cfg.get("bot_token"):
        try:
            from bots.discord_bot import DiscordBot
            bot = DiscordBot(repo_root, dc_cfg["bot_token"])
            bot.start()
            with _bots_lock:
                _running_bots.append(bot)
            LOGGER.info("Discord bot thread started.")
        except Exception:
            LOGGER.exception("Failed to start Discord bot.")
