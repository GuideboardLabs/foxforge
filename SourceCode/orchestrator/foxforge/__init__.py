"""Foxforge identity, persona, and manifesto handling."""

from .identity import (
    FOXFORGE_ALIASES,
    FOXFORGE_ADDRESS_NEXT_WORDS,
    FOXFORGE_IDENTITY_CUES,
    mentions_foxforge_alias,
    strip_foxforge_vocative_prefix,
    is_foxforge_self_query,
)
from .manifesto import (
    load_manifesto_text,
    manifesto_principles_block,
    foxforge_persona_block,
    foxforge_identity_reply,
)

__all__ = [
    "FOXFORGE_ALIASES",
    "FOXFORGE_ADDRESS_NEXT_WORDS",
    "FOXFORGE_IDENTITY_CUES",
    "mentions_foxforge_alias",
    "strip_foxforge_vocative_prefix",
    "is_foxforge_self_query",
    "load_manifesto_text",
    "manifesto_principles_block",
    "foxforge_persona_block",
    "foxforge_identity_reply",
]
