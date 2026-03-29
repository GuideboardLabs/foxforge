"""Request classification filters — determine if a message is a reminder-only or event-only request."""

from __future__ import annotations

import re
from typing import Callable


def _make_contains(low: str) -> Callable[[str], bool]:
    def _contains(token: str) -> bool:
        key = str(token or "").strip().lower()
        if not key:
            return False
        if " " in key or "-" in key:
            return key in low
        return bool(re.search(rf"\b{re.escape(key)}\b", low))
    return _contains


def is_reminder_only_request(text: str, extract_reminder_fn: Callable[[str], object | None]) -> bool:
    """Return True if the message is purely a reminder/scheduling request.

    Args:
        text: The user message text.
        extract_reminder_fn: Callable that takes raw text and returns a reminder object/dict if
            a reminder is found, or None/falsy if not. Typically orchestrator._extract_reminder_from_text.
    """
    raw = str(text or "").strip()
    if not raw:
        return False
    if not extract_reminder_fn(raw):
        return False
    low = raw.lower()
    _contains = _make_contains(low)

    block_tokens = (
        "research",
        "analyze",
        "analysis",
        "compare",
        "study",
        "sources",
        "citation",
        "citations",
        "build",
        "design",
        "spec",
        "code",
        "ui",
        "web app",
        "foraging",
    )
    if any(_contains(tok) for tok in block_tokens):
        return False

    control_tokens = (
        "remind me",
        "set reminder",
        "set a reminder",
        "set me a reminder",
        "appointment",
        "schedule",
        "tomorrow",
        "today",
    )
    return any(_contains(tok) for tok in control_tokens)


def is_event_only_request(text: str) -> bool:
    """Return True if the message is purely a calendar event request."""
    raw = str(text or "").strip()
    if not raw:
        return False
    low = raw.lower()
    _contains = _make_contains(low)

    intent_tokens = (
        "calendar",
        "event",
        "appointment",
        "schedule",
        "add it to the calendar",
        "put it on the calendar",
        "recurring",
        "weekly",
        "monthly",
        "every",
    )
    if not any(_contains(tok) for tok in intent_tokens):
        return False

    block_tokens = (
        "research",
        "analyze",
        "analysis",
        "compare",
        "citation",
        "citations",
        "sources",
        "build",
        "design",
        "code",
        "foraging",
    )
    if any(_contains(tok) for tok in block_tokens):
        return False
    return True
