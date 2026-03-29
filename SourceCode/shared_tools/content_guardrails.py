"""
Hard-coded content guardrails.

These checks fire BEFORE any model call and are not LLM-based — they
cannot be bypassed by an abliterated or unfiltered model.

Usage:
    from shared_tools.content_guardrails import check_content

    result = check_content(user_text)
    if result.blocked:
        return result.reason  # refusal message, safe to show the user
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# CSAM — absolute block
# ---------------------------------------------------------------------------
_CSAM_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bchild\s*(porn|pornography|sex|sexual|nude|nudes|naked|exploit)\b", re.I),
    re.compile(r"\b(minor|underage|preteen|pre-teen|toddler|infant)\b.{0,60}\b(sex|nude|naked|explicit|porn)\b", re.I | re.S),
    re.compile(r"\b(sex|sexual|nude|naked|explicit|porn)\b.{0,60}\b(minor|child|kid|underage|preteen)\b", re.I | re.S),
    re.compile(r"\b(lolita|loli)\b.{0,40}\b(sex|nude|image|video|photo|explicit|porn)\b", re.I | re.S),
    re.compile(r"\bpedo(phile|philia|file)?\b.{0,60}\b(image|video|photo|material|content|collect)\b", re.I | re.S),
    re.compile(r"\bcsam\b", re.I),
    re.compile(r"\bchild\s*porn\b", re.I),
]

_REFUSAL_CSAM = (
    "This request cannot be processed. Content involving the sexual exploitation "
    "of minors is absolutely prohibited."
)

# ---------------------------------------------------------------------------
# WMD construction — absolute block
# Patterns are (action verb) + (prohibited subject) within a bounded window.
# Academic discussion (no construction verb) does not match.
# ---------------------------------------------------------------------------
_WEAPONS_PATTERNS: list[re.Pattern[str]] = [
    # Conventional explosives / IEDs
    re.compile(
        r"\b(make|build|construct|assemble|fabricate|wire|detonate|arm)\b.{0,80}"
        r"\b(bomb|explosive|ied|pipe.?bomb|nail.?bomb|car.?bomb|vest.?bomb|suicide.?vest)\b",
        re.I | re.S,
    ),
    re.compile(
        r"\b(instructions?|recipe|steps?|how.?to|guide|tutorial|walk.?through)\b.{0,80}"
        r"\b(bomb|explosive|ied|pipe.?bomb|improvised.?explosive)\b",
        re.I | re.S,
    ),
    # Explosive precursor chemistry
    re.compile(
        r"\b(anfo|tatp|hmtd|rdx|hmx|petn|semtex)\b.{0,60}"
        r"\b(make|mix|synth|prepare|detonate|trigger|create)\b",
        re.I | re.S,
    ),
    re.compile(
        r"\b(make|mix|synth|prepare|create)\b.{0,60}"
        r"\b(anfo|tatp|hmtd|rdx|hmx|petn|semtex)\b",
        re.I | re.S,
    ),
    re.compile(
        r"\bammonium.?nitrate\b.{0,60}\b(bomb|explosive|detonate|mix|make|anfo)\b",
        re.I | re.S,
    ),
    # Chemical weapons
    re.compile(
        r"\b(sarin|vx\b|novichok|tabun|soman|mustard.?gas|phosgene|chlorine.?gas)\b.{0,80}"
        r"\b(make|synth|produce|weaponize|deploy|release|disperse)\b",
        re.I | re.S,
    ),
    re.compile(
        r"\b(nerve.?agent|chemical.?weapon|chem.?weapon)\b.{0,80}"
        r"\b(make|synth|produce|weaponize|deploy)\b",
        re.I | re.S,
    ),
    # Biological weapons
    re.compile(
        r"\b(ricin|botulinum|anthrax|plague|smallpox|ebola)\b.{0,80}"
        r"\b(make|synth|produce|extract|weaponize|aerosolize|culture|grow)\b",
        re.I | re.S,
    ),
    re.compile(
        r"\b(bio(logical)?.?weapon|weaponized.?pathogen|weaponized.?bacteria|weaponized.?virus)\b",
        re.I,
    ),
    # Radiological / nuclear
    re.compile(
        r"\b(dirty.?bomb|radiological.?weapon|rad.?dispersal)\b.{0,80}"
        r"\b(make|build|assemble|construct|deploy)\b",
        re.I | re.S,
    ),
    re.compile(
        r"\bnuclear.?(device|weapon|warhead)\b.{0,80}"
        r"\b(make|build|construct|assemble|detonate|trigger)\b",
        re.I | re.S,
    ),
]

_REFUSAL_WEAPONS = (
    "This request cannot be processed. Instructions for constructing bombs, "
    "explosive devices, or chemical, biological, or radiological weapons are "
    "prohibited regardless of mode."
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class GuardrailResult:
    blocked: bool
    reason: str = field(default="")


def check_content(text: str) -> GuardrailResult:
    """
    Pre-flight content check. Returns GuardrailResult with blocked=True
    if text matches any hard-block pattern.

    Safe to call on every request regardless of topic_type — topic type
    does NOT bypass these checks, including Underground mode.
    """
    t = str(text or "")
    if not t:
        return GuardrailResult(blocked=False)

    for pat in _CSAM_PATTERNS:
        if pat.search(t):
            return GuardrailResult(blocked=True, reason=_REFUSAL_CSAM)

    for pat in _WEAPONS_PATTERNS:
        if pat.search(t):
            return GuardrailResult(blocked=True, reason=_REFUSAL_WEAPONS)

    return GuardrailResult(blocked=False)
