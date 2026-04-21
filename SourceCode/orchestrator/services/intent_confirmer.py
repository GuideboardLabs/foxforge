"""Intent Confirmer — cheap LLM gate between prompt digestion and Make lane routing.

Prevents casual phrases ("make me some tea") from firing expensive multi-agent
Make pools. Uses gemma3:4b for fast (<2s) inference.

Rules:
- If UI mode == "make" AND make_type is explicitly set → skip (user was deliberate).
- If UI mode == "make" but no type → confirms and suggests type.
- If UI mode == "talk" but build-intent regex fired → gates the upgrade.
  Defaults to "chat" on confidence < 0.7.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from pathlib import Path
from typing import Any

from shared_tools.activity_bus import telemetry_emit

LOGGER = logging.getLogger(__name__)

_MODEL = "gemma3:4b"
_TEMPERATURE = 0.1
_NUM_CTX = 4096
_TIMEOUT = 30

_BUILD_INTENT_TERMS = frozenset({
    "build", "create", "make", "generate", "draft", "design", "redesign",
    "implement", "code", "develop", "scaffold", "spec", "prototype",
    "produce", "assemble", "ship", "write the", "launch",
})

_AMBIGUOUS_MAKE_PHRASES = frozenset({
    "make me", "make it", "make sure", "make sense", "make do",
    "make up", "make out", "make time", "make room", "make way",
    "make believe", "make peace", "make friends", "make money",
    "make dinner", "make lunch", "make breakfast", "make food",
    "make tea", "make coffee", "make a move", "make a deal",
    "create an account", "create a profile", "create an event",
})


def _input_hash(text: str) -> str:
    norm = " ".join(str(text or "").strip().lower().split())
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]


def _emit_decision(
    repo_root: Path,
    *,
    text: str,
    route: str,
    confidence: float,
    skipped: bool,
    fast_path_hit: bool,
    latency_ms: float,
) -> None:
    telemetry_emit(
        repo_root,
        "gate_decisions.jsonl",
        {
            "gate": "intent_confirmer",
            "input_hash": _input_hash(text),
            "fast_path_hit": bool(fast_path_hit),
            "route": str(route),
            "confidence": round(float(confidence), 4),
            "skipped": bool(skipped),
            "latency_ms": round(float(latency_ms), 2),
        },
        retention_days=14,
    )


def has_build_intent(text: str) -> bool:
    """Quick regex-based build intent check (same logic as orchestrator/main.py)."""
    low = text.lower()
    for term in _BUILD_INTENT_TERMS:
        if " " in term or "-" in term:
            if term in low:
                return True
        elif re.search(rf"\b{re.escape(term)}\b", low):
            return True
    return False


def is_obviously_ambiguous(text: str) -> bool:
    """Return True if the text matches a known ambiguous make-phrase."""
    low = text.lower().strip()
    for phrase in _AMBIGUOUS_MAKE_PHRASES:
        if low.startswith(phrase) or f" {phrase} " in f" {low} ":
            return True
    return False


def confirm_make_intent(
    text: str,
    repo_root: Path,
    *,
    ui_mode: str = "talk",
    make_type: str = "",
) -> dict[str, Any]:
    """Confirm whether the user prompt is a genuine Make-lane build request.

    Returns dict with:
      - intent: "make" | "chat" | "forage"
      - confidence: float 0..1
      - suggested_type: str (best Make type_id guess, if intent=="make")
      - reason: str
      - skipped: bool (True when confirmation was skipped — fast path)
    """
    started = time.perf_counter()
    ui_mode = str(ui_mode or "talk").strip().lower()
    make_type = str(make_type or "").strip().lower()

    # Fast path: user explicitly chose mode=make AND selected a type from the modal
    if ui_mode == "make" and make_type:
        result = {
            "intent": "make",
            "confidence": 1.0,
            "suggested_type": make_type,
            "reason": "User explicitly selected Make mode and type from the UI.",
            "skipped": True,
        }
        _emit_decision(
            repo_root,
            text=text,
            route=str(result.get("intent", "chat")),
            confidence=float(result.get("confidence", 0.0)),
            skipped=True,
            fast_path_hit=True,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )
        return result

    # If obviously ambiguous — skip LLM call, return chat
    if is_obviously_ambiguous(text):
        result = {
            "intent": "chat",
            "confidence": 0.95,
            "suggested_type": "",
            "reason": "Phrase matches known ambiguous non-build expression.",
            "skipped": True,
        }
        _emit_decision(
            repo_root,
            text=text,
            route=str(result.get("intent", "chat")),
            confidence=float(result.get("confidence", 0.0)),
            skipped=True,
            fast_path_hit=True,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )
        return result

    # If mode is "talk" and no build intent regex — skip LLM call
    if ui_mode == "talk" and not has_build_intent(text):
        result = {
            "intent": "chat",
            "confidence": 0.99,
            "suggested_type": "",
            "reason": "No build-intent keywords found in talk mode.",
            "skipped": True,
        }
        _emit_decision(
            repo_root,
            text=text,
            route=str(result.get("intent", "chat")),
            confidence=float(result.get("confidence", 0.0)),
            skipped=True,
            fast_path_hit=True,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )
        return result

    # LLM call for ambiguous cases
    try:
        from shared_tools.ollama_client import OllamaClient

        system_prompt = (
            "You are an intent classifier. Determine whether the user's message is a "
            "genuine request to BUILD or CREATE a deliverable artifact (code, document, script, "
            "video script, essay, app, etc.) using an AI Make pipeline — or whether it is "
            "casual conversation, a question, or a non-build request.\n\n"
            "Respond with ONLY valid JSON in this exact format:\n"
            '{"intent": "make"|"chat"|"forage", "confidence": 0.0-1.0, '
            '"suggested_type": "<type_id or empty string>", "reason": "<one sentence>"}\n\n'
            "Valid type_ids: tool, web_app, desktop_app, social_post, email, blog, "
            "essay_short, essay_long, guide, tutorial, video_script, newsletter, press_release, "
            "novel_chapter, memoir_chapter, book_chapter, screenplay, "
            "medical, finance, sports, history, game_design_doc\n\n"
            "Rules:\n"
            "- intent='make' only if the user wants an artifact PRODUCED (a file, a document, "
            "  code, a script). A question about how to do something is 'chat'.\n"
            "- intent='forage' only if the user wants research/investigation without building.\n"
            "- intent='chat' for everything else.\n"
            "- confidence < 0.7 means you're not sure — default to 'chat'.\n"
            "- Return ONLY the JSON object. No markdown, no explanation."
        )
        user_prompt = (
            f"UI mode declared by user: {ui_mode}\n"
            f"Make type pre-selected: {make_type or '(none)'}\n\n"
            f"User message:\n{text[:800]}"
        )

        client = OllamaClient()
        raw = client.chat(
            model=_MODEL,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=_TEMPERATURE,
            num_ctx=_NUM_CTX,
            think=False,
            timeout=_TIMEOUT,
            retry_attempts=2,
            retry_backoff_sec=1.0,
        )
        raw = str(raw or "").strip()

        # Extract JSON from response
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
            intent = str(data.get("intent", "chat")).strip().lower()
            if intent not in ("make", "chat", "forage"):
                intent = "chat"
            confidence = float(data.get("confidence", 0.0))
            # Enforce the confidence floor: < 0.7 → chat
            if intent == "make" and confidence < 0.7:
                intent = "chat"
            result = {
                "intent": intent,
                "confidence": confidence,
                "suggested_type": str(data.get("suggested_type", "")).strip().lower(),
                "reason": str(data.get("reason", "")).strip()[:200],
                "skipped": False,
            }
            _emit_decision(
                repo_root,
                text=text,
                route=str(result.get("intent", "chat")),
                confidence=float(result.get("confidence", 0.0)),
                skipped=False,
                fast_path_hit=False,
                latency_ms=(time.perf_counter() - started) * 1000.0,
            )
            return result
    except Exception as exc:
        LOGGER.warning("IntentConfirmer LLM call failed: %s — defaulting to declared mode", exc)

    # Fallback: trust declared UI mode
    fallback_intent = ui_mode if ui_mode in ("make", "forage") else "chat"
    result = {
        "intent": fallback_intent,
        "confidence": 0.5,
        "suggested_type": make_type,
        "reason": "LLM confirmation unavailable; falling back to declared UI mode.",
        "skipped": False,
    }
    _emit_decision(
        repo_root,
        text=text,
        route=str(result.get("intent", "chat")),
        confidence=float(result.get("confidence", 0.0)),
        skipped=False,
        fast_path_hit=False,
        latency_ms=(time.perf_counter() - started) * 1000.0,
    )
    return result
