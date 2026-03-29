"""Reminder extraction from natural language text."""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone


def extract_reminder_from_text(text: str) -> dict[str, str] | None:
    """Parse a user message and return a reminder dict, or None if no reminder intent found.

    Returns:
        dict with keys: 'title', 'due_date' (ISO date string or ''), 'time' (HH:MM or '').
    """
    raw = str(text or "").strip()
    low = raw.lower()
    if not raw:
        return None

    title_source = ""
    has_explicit_reminder_intent = False
    match_direct = re.search(r"\bremind me to\b(.+)$", raw, flags=re.IGNORECASE)
    if match_direct:
        has_explicit_reminder_intent = True
        title_source = str(match_direct.group(1) or "").strip()
    else:
        match_set = re.search(r"\bset(?: me)?(?: a)? reminder\b(.+)$", raw, flags=re.IGNORECASE)
        if match_set:
            has_explicit_reminder_intent = True
            remainder = str(match_set.group(1) or "").strip()
            if remainder.lower().startswith("for "):
                remainder = remainder[4:].strip()
            to_match = re.search(r"\bto\b(.+)$", remainder, flags=re.IGNORECASE)
            title_source = str(to_match.group(1) if to_match else remainder).strip()

    due_date = ""
    if re.search(r"\btomorrow\b", raw, flags=re.IGNORECASE):
        due_date = (datetime.now(timezone.utc).date() + timedelta(days=1)).isoformat()
    else:
        date_match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", raw)
        if date_match:
            due_date = str(date_match.group(1) or "").strip()

    time_text = ""
    time_match = re.search(r"\bat\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", raw, flags=re.IGNORECASE)
    if time_match:
        hour = int(time_match.group(1))
        minute = int(time_match.group(2) or "0")
        mer = str(time_match.group(3) or "").strip().lower()
        if 0 <= minute <= 59:
            if mer in {"am", "pm"}:
                if 1 <= hour <= 12:
                    if mer == "am":
                        hour = 0 if hour == 12 else hour
                    else:
                        hour = 12 if hour == 12 else hour + 12
                    time_text = f"{hour:02d}:{minute:02d}"
            elif 0 <= hour <= 23:
                time_text = f"{hour:02d}:{minute:02d}"

    if not title_source:
        title_source = raw

    has_schedule_cue = bool(due_date or time_text)
    if not has_schedule_cue and re.search(r"\btomorrow\b|\btoday\b", raw, flags=re.IGNORECASE):
        has_schedule_cue = True
    if not has_schedule_cue and re.search(r"\bon\s+\d{4}-\d{2}-\d{2}\b", raw, flags=re.IGNORECASE):
        has_schedule_cue = True

    if not has_explicit_reminder_intent and not has_schedule_cue:
        return None

    if not has_explicit_reminder_intent:
        if not re.search(
            r"\b(i need to|i have to|need to|have to|call|text|email|pay|take|pick up|go|buy|book|schedule|renew|submit|prepare|remember to|don't let me forget)\b",
            low,
            flags=re.IGNORECASE,
        ):
            return None

    title = re.sub(r"\b(tomorrow|today)\b", "", title_source, flags=re.IGNORECASE)
    title = re.sub(r"\bon\s+\d{4}-\d{2}-\d{2}\b", "", title, flags=re.IGNORECASE)
    title = re.sub(r"\bat\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?\b", "", title, flags=re.IGNORECASE)
    title = re.sub(
        r"^(?:can you|could you|please|hey|hi|i need to|i have to|need to|have to|remember to|don't let me forget to)\s+",
        "",
        title,
        flags=re.IGNORECASE,
    )
    title = re.sub(r"\s+", " ", title).strip(" .,!?:;")
    if not title:
        return None
    return {"title": title[:96], "due_date": due_date, "time": time_text}
