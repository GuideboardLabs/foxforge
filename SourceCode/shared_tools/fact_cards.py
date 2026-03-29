from __future__ import annotations

import re
from collections import Counter
from typing import Any

from shared_tools.answer_composer import format_source_label
from shared_tools.fact_policy import detect_topic_type


def is_combat_sports_topic(query: str, topic_type: str = "general") -> bool:
    return detect_topic_type(query, topic_type) == "combat_sports"


def _top_value(values: list[tuple[str, str]]) -> tuple[str, str] | None:
    if not values:
        return None
    counts = Counter(v for v, _ in values if v)
    if not counts:
        return None
    best = counts.most_common(1)[0][0]
    for value, source in values:
        if value == best:
            return value, source
    return None


def build_event_fact_card(query: str, sources: list[dict[str, Any]]) -> dict[str, Any]:
    event_dates: list[tuple[str, str]] = []
    odds: list[tuple[str, str]] = []
    broadcasts: list[tuple[str, str]] = []
    venues: list[tuple[str, str]] = []
    for row in sources:
        text = f"{row.get('title', '')} {row.get('snippet', '')}"
        label = format_source_label(row)
        for match in re.findall(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+(?:19|20)\d{2}\b", text, flags=re.IGNORECASE):
            event_dates.append((match.strip(), label))
        for match in re.findall(r"(?:[+-]\d{3,4}|\b\d+(?:\.\d+)?\s*(?:favorite|underdog)\b)", text, flags=re.IGNORECASE):
            odds.append((match.strip(), label))
        if re.search(r"paramount\+|espn\+|ppv|fight pass", text, flags=re.IGNORECASE):
            bmatch = re.search(r"(Paramount\+|ESPN\+|PPV|Fight Pass)", text, flags=re.IGNORECASE)
            if bmatch:
                broadcasts.append((bmatch.group(1), label))
        vmatch = re.search(r"\b(?:at|from)\s+([A-Z][A-Za-z'\-.]+(?:\s+[A-Z][A-Za-z'\-.]+){0,4})", text)
        if vmatch:
            venues.append((vmatch.group(1).strip(), label))
    return {
        "event_date": _top_value(event_dates),
        "odds_snapshot": _top_value(odds),
        "broadcast": _top_value(broadcasts),
        "venue": _top_value(venues),
        "source_count": len(sources),
    }


def build_fighter_fact_card(query: str, sources: list[dict[str, Any]]) -> dict[str, Any]:
    title_history: list[tuple[str, str]] = []
    injuries: list[tuple[str, str]] = []
    records: list[tuple[str, str]] = []
    for row in sources:
        text = f"{row.get('title', '')} {row.get('snippet', '')}"
        label = format_source_label(row)
        for match in re.findall(r"\b\d{1,2}-\d{1,2}(?:-\d{1,2})?\b", text):
            records.append((match, label))
        if re.search(r"title|champion|belt", text, flags=re.IGNORECASE):
            sent = re.search(r"([^.!?]{0,80}(?:title|champion|belt)[^.!?]{0,80})", text, flags=re.IGNORECASE)
            if sent:
                title_history.append((sent.group(1).strip(), label))
        if re.search(r"shoulder|knee|ankle|injur", text, flags=re.IGNORECASE):
            sent = re.search(r"([^.!?]{0,80}(?:shoulder|knee|ankle|injur)[^.!?]{0,80})", text, flags=re.IGNORECASE)
            if sent:
                injuries.append((sent.group(1).strip(), label))
    return {
        "record": _top_value(records),
        "title_history": _top_value(title_history),
        "notable_injury": _top_value(injuries),
    }


def render_fact_card_markdown(query: str, sources: list[dict[str, Any]], topic_type: str = "general") -> str:
    if not is_combat_sports_topic(query, topic_type):
        return ""
    event = build_event_fact_card(query, sources)
    fighter = build_fighter_fact_card(query, sources)
    rows: list[str] = []
    if event.get("event_date"):
        value, source = event["event_date"]
        rows.append(f"- Event date snapshot: **{value}** (source: {source})")
    if event.get("broadcast"):
        value, source = event["broadcast"]
        rows.append(f"- Broadcast mention: **{value}** (source: {source})")
    if event.get("odds_snapshot"):
        value, source = event["odds_snapshot"]
        rows.append(f"- Odds snapshot: **{value}** (source: {source})")
    if fighter.get("record"):
        value, source = fighter["record"]
        rows.append(f"- Record signal found: **{value}** (source: {source})")
    if fighter.get("title_history"):
        value, source = fighter["title_history"]
        rows.append(f"- Title-history note: {value} (source: {source})")
    if fighter.get("notable_injury"):
        value, source = fighter["notable_injury"]
        rows.append(f"- Injury note: {value} (source: {source})")
    return "\n".join(rows)
