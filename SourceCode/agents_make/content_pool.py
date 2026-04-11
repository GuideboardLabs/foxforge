"""Content pool — blogs, social posts, emails.

Pipeline:
    1. Drafter       — llama3.1-abliterated:8b (temp 0.8) produces full content
                        in a single pass.
    2. Tone Reviewer — qwen3:14b (temp 0.2) checks tone appropriateness, CTA
                        clarity, format compliance.
    3. Final Polish  — llama3.1-abliterated:8b applies tone notes if any.

Fast, lightweight pipeline optimized for short-form professional content.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from shared_tools.ollama_client import OllamaClient


_MODEL_DRAFTER  = "llama3.1-abliterated:8b"
_MODEL_REVIEWER = "qwen3:14b"


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _trim(text: str, max_chars: int) -> str:
    body = str(text or "").strip()
    if len(body) <= max_chars:
        return body
    cut = body[:max_chars].rsplit("\n", 1)[0].strip()
    return cut or body[:max_chars]


_KIND_SPECS: dict[str, dict[str, Any]] = {
    "blog": {
        "word_range": "600-800 words",
        "structure": (
            "Structure:\n"
            "1. Hook & Headline — attention-grabbing title + 1-2 sentence hook\n"
            "2. Context & Why Now — brief background, conversational not academic\n"
            "3. Core Content — key insights with subheadings, short paragraphs, concrete examples\n"
            "4. Takeaway & CTA — clear takeaway and call to action"
        ),
        "voice": "Conversational, authoritative, accessible. Write like you're explaining to a smart friend.",
    },
    "social_post": {
        "word_range": "100-200 words total",
        "structure": (
            "Structure:\n"
            "1. Hook — one punchy sentence (under 40 words) that stops the scroll\n"
            "2. Body — 2-3 concise sentences expanding the hook\n"
            "3. Call to Action — one sentence: what should the reader do next?"
        ),
        "voice": "Punchy, direct, conversational. No jargon. Every word earns its place.",
    },
    "email": {
        "word_range": "200-400 words",
        "structure": (
            "Structure:\n"
            "1. Subject Line — clear, specific, under 60 characters\n"
            "2. Greeting — appropriate formality for the context\n"
            "3. Body — context + ask in 2-3 short paragraphs, most important point first\n"
            "4. Sign-off — professional closing with clear next step"
        ),
        "voice": "Professional, clear, respectful of the reader's time. Front-load the key message.",
    },
}


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def _run_drafter(
    client: OllamaClient,
    question: str,
    kind: str,
    research_context: str,
    project_context: str,
) -> str:
    spec = _KIND_SPECS.get(kind, _KIND_SPECS["blog"])
    system_prompt = (
        f"Today: {_today()}. "
        f"You are a professional content writer. Write a {kind} in one pass.\n\n"
        f"Target length: {spec['word_range']}.\n"
        f"{spec['structure']}\n\n"
        f"Voice: {spec['voice']}\n\n"
        "Output the final content in markdown. No meta-commentary or notes to self."
    )
    user_prompt = (
        f"Request: {question}\n\n"
        f"Research context:\n{_trim(research_context, 8000)}\n\n"
        f"Project context:\n{_trim(project_context, 3000)}"
    )
    try:
        result = client.chat(
            model=_MODEL_DRAFTER,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.8,
            num_ctx=12288,
            think=False,
            timeout=240,
            retry_attempts=3,
            retry_backoff_sec=1.2,
        )
        return str(result or "").strip()
    except Exception as exc:
        return f"[Draft generation failed: {exc}]"


def _run_tone_reviewer(
    client: OllamaClient,
    draft: str,
    kind: str,
    question: str,
) -> str:
    spec = _KIND_SPECS.get(kind, _KIND_SPECS["blog"])
    system_prompt = (
        f"Today: {_today()}. "
        f"You are an editorial reviewer for a {kind}. Check the draft for:\n"
        "1. Tone appropriateness — does it match the expected voice?\n"
        "2. Structure compliance — does it follow the required format?\n"
        "3. CTA clarity — is the call to action specific and actionable?\n"
        "4. Length compliance — is it within the target range?\n"
        "5. Engagement — will it hold the reader's attention?\n\n"
        f"Expected voice: {spec['voice']}\n"
        f"Expected length: {spec['word_range']}\n\n"
        "For each issue: give a one-sentence fix instruction. "
        "If the draft is strong, say 'Approved.' and stop."
    )
    user_prompt = (
        f"Kind: {kind} | Request: {question}\n\n"
        f"Draft to review:\n{_trim(draft, 6000)}"
    )
    try:
        result = client.chat(
            model=_MODEL_REVIEWER,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.2,
            num_ctx=12288,
            think=False,
            timeout=180,
            retry_attempts=3,
            retry_backoff_sec=1.2,
        )
        return str(result or "").strip()
    except Exception:
        return "Approved."


def _run_polish(
    client: OllamaClient,
    draft: str,
    review_notes: str,
    kind: str,
) -> str:
    system_prompt = (
        f"You are a {kind} editor. Apply the reviewer's notes to polish this content. "
        "Preserve the voice and approximate length. Return the complete polished content only."
    )
    user_prompt = (
        f"Reviewer notes:\n{_trim(review_notes, 1500)}\n\n"
        f"Draft to polish:\n{draft}"
    )
    try:
        result = client.chat(
            model=_MODEL_DRAFTER,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.6,
            num_ctx=12288,
            think=False,
            timeout=180,
            retry_attempts=3,
            retry_backoff_sec=1.2,
        )
        polished = str(result or "").strip()
        return polished if polished and len(polished) >= len(draft) * 0.5 else draft
    except Exception:
        return draft


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_content_pool(
    question: str,
    repo_root: Path,
    project_slug: str,
    bus: Any,
    target: str = "blog",
    research_context: str = "",
    project_context: str = "",
    cancel_checker: Callable[[], bool] | None = None,
    progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run the content pipeline and return the final content."""

    def _progress(stage: str, detail: dict[str, Any] | None = None) -> None:
        if callable(progress_callback):
            try:
                progress_callback(stage, detail or {})
            except Exception:
                pass

    def _cancelled() -> bool:
        if callable(cancel_checker):
            try:
                return bool(cancel_checker())
            except Exception:
                return False
        return False

    kind = str(target).strip().lower() or "blog"
    bus.emit("content_pool", "start", {"question": question, "target": kind})

    client = OllamaClient()

    # Step 1: Draft
    if _cancelled():
        return {"ok": False, "message": "Cancelled before drafting.", "body": ""}

    _progress("content_draft_started", {"kind": kind})
    draft = _run_drafter(client, question, kind, research_context, project_context)
    _progress("content_draft_completed", {"preview": draft[:300], "chars": len(draft)})

    # Step 2: Tone review
    if _cancelled():
        return {"ok": False, "message": "Cancelled before review.", "body": draft}

    _progress("content_review_started", {})
    review_notes = _run_tone_reviewer(client, draft, kind, question)
    _progress("content_review_completed", {"preview": review_notes[:200]})

    # Step 3: Polish (only if reviewer flagged issues)
    final_body = draft
    if review_notes and "approved" not in review_notes.lower() and not _cancelled():
        _progress("content_polish_started", {})
        final_body = _run_polish(client, draft, review_notes, kind)
        _progress("content_polish_completed", {"chars": len(final_body)})

    bus.emit("content_pool", "completed", {
        "project": project_slug, "target": kind, "chars": len(final_body),
    })

    return {
        "ok": True,
        "body": final_body,
        "review_notes": review_notes,
        "message": f"{kind.replace('_', ' ').title()} complete — {len(final_body):,} chars.",
    }
