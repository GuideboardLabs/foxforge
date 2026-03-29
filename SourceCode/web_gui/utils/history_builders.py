"""Conversation history builders — format raw message dicts into LLM history lists."""

from __future__ import annotations

from typing import Any


def extract_talk_text(raw: str) -> str | None:
    """Return the text after '/talk' if the message is a talk-mode message, else None."""
    text = str(raw or "").strip()
    low = text.lower()
    if low == "/talk":
        return ""
    if low.startswith("/talk "):
        return text[6:].strip()
    return None


def build_talk_history(messages: list[dict[str, Any]], limit_turns: int = 16) -> list[dict[str, str]]:
    """Build an LLM history list containing only talk-mode exchanges."""
    rows = messages[-max(20, limit_turns * 4):]
    out: list[dict[str, str]] = []
    expecting_assistant = False
    for row in rows:
        role = str(row.get("role", "")).strip().lower()
        content = str(row.get("content", "")).strip()
        mode = str(row.get("mode", "")).strip().lower()

        if role == "user":
            talk_text = extract_talk_text(content)
            if mode == "talk":
                talk_text = content
            if talk_text is None:
                expecting_assistant = False
                continue
            if talk_text:
                out.append({"role": "user", "content": talk_text})
                expecting_assistant = True
            else:
                expecting_assistant = False
            continue

        if role == "assistant" and expecting_assistant:
            if content:
                out.append({"role": "assistant", "content": content})
            expecting_assistant = False

    max_messages = max(2, limit_turns * 2)
    return out[-max_messages:]


def build_command_history(messages: list[dict[str, Any]], limit_turns: int = 16) -> list[dict[str, str]]:
    """Build an LLM history list for command/research context (excludes talk, control commands)."""
    rows = messages[-max(24, limit_turns * 4):]
    out: list[dict[str, str]] = []
    for row in rows:
        role = str(row.get("role", "")).strip().lower()
        content = str(row.get("content", "")).strip()
        mode = str(row.get("mode", "")).strip().lower()
        if role not in {"user", "assistant"} or not content:
            continue

        if role == "user":
            if mode and mode != "command":
                continue
            low = content.lower()
            if low.startswith("/talk"):
                continue
            if low.startswith("/") and low not in {"/status", "/models", "/local-models"}:
                continue
        out.append({"role": role, "content": content})

    max_messages = max(4, limit_turns * 2)
    return out[-max_messages:]


def build_fact_history(messages: list[dict[str, Any]], limit_turns: int = 120) -> list[dict[str, str]]:
    """Build a history list of user messages only for project fact extraction."""
    rows = messages[-max(80, limit_turns * 4):]
    out: list[dict[str, str]] = []
    for row in rows:
        role = str(row.get("role", "")).strip().lower()
        content = str(row.get("content", "")).strip()
        if role != "user" or not content:
            continue

        talk_text = extract_talk_text(content)
        normalized = talk_text if talk_text is not None else content
        normalized = str(normalized or "").strip()
        if not normalized:
            continue

        low = normalized.lower()
        if low.startswith("/") and low not in {"/status", "/models", "/local-models"}:
            continue
        out.append({"role": "user", "content": normalized})

    max_messages = max(10, limit_turns * 2)
    return out[-max_messages:]
