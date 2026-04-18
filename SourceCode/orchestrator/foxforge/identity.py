"""Foxforge alias detection and identity query classification."""

from __future__ import annotations

import re

FOXFORGE_ALIASES: tuple[str, ...] = (
    "reynard",
    "foxforge",
)

FOXFORGE_ADDRESS_NEXT_WORDS: frozenset[str] = frozenset({
    "can", "could", "would", "will", "please",
    "set", "add", "show", "tell", "help",
    "what", "who", "how", "why", "when", "where",
})

FOXFORGE_IDENTITY_CUES: tuple[str, ...] = (
    "who are you",
    "what are you",
    "what is foxforge",
    "what's foxforge",
    "who is reynard",
    "what is reynard",
    "what's reynard",
    "about foxforge",
    "about yourself",
    "what do you do",
    "what is your purpose",
    "your purpose",
    "tech stack",
    "technology stack",
    "architecture",
    "how are you built",
    "how do you work",
    "origin story",
    "backstory",
    "where did you come from",
    "what's your name",
    "what is your name",
)


def mentions_foxforge_alias(text: str) -> bool:
    low = str(text or "").strip().lower()
    if not low:
        return False
    for alias in FOXFORGE_ALIASES:
        if re.search(rf"\b{re.escape(alias)}\b", low):
            return True
    return False


def strip_foxforge_vocative_prefix(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    pattern = re.compile(
        r"^\s*(?:(?:hey|hi|yo|ok|okay)\s+)?"
        r"(?P<alias>reynard|foxforge)\b"
        r"(?P<sep>\s*[,:\-!]\s*|\s+)"
        r"(?P<rest>.+)$",
        flags=re.IGNORECASE,
    )
    match = pattern.match(raw)
    if not match:
        return raw
    rest = str(match.group("rest") or "").strip()
    if not rest:
        return raw
    sep = str(match.group("sep") or "")
    if any(ch in sep for ch in ",:-!"):
        return rest
    first_word = re.split(r"\s+", rest, maxsplit=1)[0].strip().lower()
    if first_word in FOXFORGE_ADDRESS_NEXT_WORDS:
        return rest
    return raw


def is_foxforge_self_query(text: str) -> bool:
    low = str(text or "").strip().lower()
    if not low:
        return False
    mentions_identity_target = mentions_foxforge_alias(low) or bool(
        re.search(r"\b(you|your|yourself)\b", low)
    )
    if not mentions_identity_target:
        return False
    return any(cue in low for cue in FOXFORGE_IDENTITY_CUES)
