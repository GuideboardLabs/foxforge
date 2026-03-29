from __future__ import annotations

import re
from typing import Any

_TOKEN_RE = re.compile(r"[a-z0-9]{2,}")
_STOP_WORDS = {
    "the", "and", "for", "with", "from", "that", "this", "have", "your", "about", "they", "them",
    "will", "would", "could", "should", "into", "their", "there", "what", "when", "where", "which",
}
_FAMILY_HINTS = {
    "family", "wife", "husband", "son", "daughter", "child", "children", "kid", "kids", "mom", "dad",
    "mother", "father", "parent", "parents", "brother", "sister", "spouse",
}
_PET_HINTS = {
    "pet", "pets", "dog", "cat", "puppy", "kitten", "fish", "snake", "hamster", "turtle", "vet", "walk",
}
_PROFILE_HINTS = {
    "name", "birthday", "gender", "job", "location", "live", "age",
}
_PREFERENCE_HINTS = {
    "prefer", "preference", "favorite", "favourite", "like", "likes", "dislike", "dislikes", "love", "hate",
    "gift", "gifts", "food", "meal", "restaurant", "cook", "buy", "shop", "present",
}
_ROUTINE_HINTS = {
    "routine", "routines", "habit", "habits", "usually", "often", "normally", "every", "weekly", "monthly",
}
_MEMORY_PHRASES = (
    "what do you remember",
    "what do you know about",
    "why do you know",
    "remember this",
    "forget this",
    "forget that",
    "pin this",
)
_INTRUSIVE_REPLY_PHRASES = (
    "i remembered",
    "i remember that",
    "from my memory",
    "from your memory",
    "because i remember",
)


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _tokens(text: str) -> set[str]:
    return {token for token in _TOKEN_RE.findall(_clean_text(text).lower()) if token not in _STOP_WORDS}


def analyze_query_context(query: str) -> dict[str, Any]:
    text = _clean_text(query)
    low = text.lower()
    terms = _tokens(text)
    explicit_memory_query = any(phrase in low for phrase in _MEMORY_PHRASES)
    family_query = bool(terms & _FAMILY_HINTS)
    pet_query = bool(terms & _PET_HINTS)
    profile_query = bool(terms & _PROFILE_HINTS)
    preference_query = bool(terms & _PREFERENCE_HINTS)
    routine_query = bool(terms & _ROUTINE_HINTS)
    recommendation_query = any(token in low for token in ("should i", "what should", "help me choose", "recommend"))
    direct_answer_query = any(token in low for token in ("write", "draft", "rewrite", "summarize", "translate"))

    allow_personal = bool(
        explicit_memory_query
        or family_query
        or pet_query
        or profile_query
        or preference_query
        or routine_query
        or (recommendation_query and (family_query or pet_query or preference_query))
    )
    personal_priority = "high" if explicit_memory_query or family_query or pet_query else ("medium" if allow_personal else "low")

    return {
        "query": text,
        "terms": sorted(terms),
        "explicit_memory_query": explicit_memory_query,
        "family_query": family_query,
        "pet_query": pet_query,
        "profile_query": profile_query,
        "preference_query": preference_query,
        "routine_query": routine_query,
        "recommendation_query": recommendation_query,
        "direct_answer_query": direct_answer_query,
        "allow_personal": allow_personal,
        "personal_priority": personal_priority,
    }


def build_context_usage_guidance(
    analysis: dict[str, Any],
    *,
    personal_available: bool,
) -> str:
    if not isinstance(analysis, dict):
        analysis = analyze_query_context("")
    explicit_memory_query = bool(analysis.get("explicit_memory_query", False))
    lines = [
        "Context policy:",
        "- Use retrieved personal context only when it materially improves the answer.",
        "- Prefer quiet relevance over clever callbacks.",
    ]
    if explicit_memory_query:
        lines.append("- The user explicitly asked about memory/context. It is fine to explain what you know and where it came from.")
    else:
        lines.append("- Do not say you remembered something or mention memory systems unless the user asks.")
    if personal_available and bool(analysis.get("allow_personal", False)):
        lines.append("- Personal context is relevant here. Use only the smallest detail that changes the advice or answer.")
    elif personal_available:
        lines.append("- Personal context is probably not relevant. Do not force personalization.")
    if bool(analysis.get("recommendation_query", False)):
        lines.append("- If context changes the recommendation, apply it directly instead of narrating it.")
    if bool(analysis.get("direct_answer_query", False)):
        lines.append("- Prioritize completing the requested output. Do not derail into context recaps.")
    return "\n".join(lines)


def evaluate_context_use(
    user_text: str,
    assistant_text: str,
    *,
    personal_context_available: bool = False,
    personal_context_injected: bool = False,
) -> dict[str, Any]:
    analysis = analyze_query_context(user_text)
    reply = _clean_text(assistant_text).lower()
    score = 0.72
    notes: list[str] = []

    intrusive_hits = sum(1 for phrase in _INTRUSIVE_REPLY_PHRASES if phrase in reply)
    if intrusive_hits and not bool(analysis.get("explicit_memory_query", False)):
        score -= min(0.34, 0.14 * intrusive_hits)
        notes.append(f"intrusive_context_mention={intrusive_hits}")

    if personal_context_injected and not bool(analysis.get("allow_personal", False)):
        score -= 0.18
        notes.append("personal_context_injected_without_clear_need")
    elif personal_context_injected:
        score += 0.06
        notes.append("personal_context_injected_with_clear_need")
    elif personal_context_available and bool(analysis.get("allow_personal", False)) and str(analysis.get("personal_priority", "low")) == "high":
        score -= 0.12
        notes.append("missed_high_value_personal_context")

    if bool(analysis.get("explicit_memory_query", False)):
        if any(token in reply for token in ("memory", "remember", "context", "know that")):
            score += 0.04
            notes.append("handled_explicit_memory_query")
        else:
            score -= 0.08
            notes.append("explicit_memory_query_not_addressed")

    score = max(0.0, min(1.0, score))
    if score >= 0.78:
        outcome = "good"
    elif score <= 0.42:
        outcome = "poor"
    else:
        outcome = "mixed"
    return {
        "score": round(score, 4),
        "outcome": outcome,
        "notes": notes,
        "analysis": analysis,
    }
