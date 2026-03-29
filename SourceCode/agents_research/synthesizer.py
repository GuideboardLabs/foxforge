from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any


def extract_action_proposals(synthesis_text: str) -> list[dict[str, str]]:
    """
    Parse the 'Actionable Next Steps' section of a synthesis and return
    up to 5 create_task proposals. Pure text parsing — no LLM call.
    Each result: {"action_type": "create_task", "title": str, "notes": str}
    """
    body = str(synthesis_text or "").strip()
    if not body:
        return []

    # Find the Actionable Next Steps section
    match = re.search(
        r"(?:##\s*Actionable Next Steps|##\s*Next Steps)(.*?)(?=\n##|\Z)",
        body,
        re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return []

    section = match.group(1).strip()
    proposals: list[dict[str, str]] = []

    for line in section.splitlines():
        stripped = line.strip()
        # Match bullet points: - item or * item or 1. item
        m = re.match(r"^[-*•]\s+(.+)$|^\d+[.)]\s+(.+)$", stripped)
        if not m:
            continue
        text = (m.group(1) or m.group(2) or "").strip()
        # Strip inline evidence labels and markdown bold/italic
        text = re.sub(r"\[E\]|\[I\]|\[S\]", "", text).strip()
        text = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", text).strip()
        if len(text) < 8:
            continue
        title = text[:120].rstrip(".,;:")
        proposals.append({
            "action_type": "create_task",
            "title": title,
            "notes": f"Extracted from research synthesis actionable steps.",
        })
        if len(proposals) >= 5:
            break

    return proposals


def _fallback_synthesis(question: str, findings: list[dict]) -> str:
    lines = [
        "# Research Synthesis",
        "",
        f"Question: {question}",
        "",
        "## Executive Summary",
        "Fallback synthesis — LLM unavailable. Raw findings listed below.",
        "Evidence Confidence: Low — no synthesis model was available.",
        "",
        "## Key Findings",
    ]
    for item in findings:
        lines.append(f"- {item['agent']}: {item['finding']}")
    lines.extend(
        [
            "",
            "## Uncertainties & Risks",
            "- Validate assumptions with primary sources.",
            "- Identify time-sensitive risks before execution.",
            "",
            "## Next Steps",
            "- Convert this synthesis into an actionable plan at current resource scale.",
        ]
    )
    return "\n".join(lines)


def _is_valid_synthesis(text: str) -> bool:
    body = str(text or "").strip()
    if len(body) < 380:
        return False
    low = body.lower()
    if any(token in low for token in ("model call failed", "ollama chat failed", "traceback")):
        return False
    expected_sections = [
        "executive summary",
        "key findings",
        "uncertainties",
        "risks",
        "next steps",
    ]
    hits = 0
    for section in expected_sections:
        if section in low:
            hits += 1
    return hits >= 3


def synthesize(
    question: str,
    findings: list[dict],
    *,
    client: Any | None = None,
    model_cfg: dict | None = None,
    project_context: str = "",
    prior_messages: list[dict[str, str]] | None = None,
    conflict_report: str = "",
) -> str:
    if client is None or not model_cfg:
        return _fallback_synthesis(question, findings)

    model = str(model_cfg.get("synthesis_model") or model_cfg.get("model", "")).strip()
    if not model:
        return _fallback_synthesis(question, findings)

    def _conf_label(item: dict) -> str:
        score = item.get("confidence", 0)
        try:
            score = int(score)
        except (TypeError, ValueError):
            score = 0
        if score >= 4:
            return "HIGH"
        if score >= 2:
            return "MED"
        return "LOW"

    primary = [item for item in findings if str(item.get("role", "primary")).strip().lower() != "advisory"]
    advisory = [item for item in findings if str(item.get("role", "primary")).strip().lower() == "advisory"]
    primary_blob = "\n\n".join([
        f"[{item['agent']} | confidence:{_conf_label(item)}]\n{item['finding']}"
        for item in primary
    ])
    advisory_blob = "\n\n".join([
        f"[{item['agent']} | confidence:{_conf_label(item)}]\n{item['finding']}"
        for item in advisory
    ])
    findings_blob = primary_blob
    if advisory_blob:
        findings_blob = (
            f"{primary_blob}\n\n"
            "---\n"
            "ADVISORY CONTEXT (supplementary — do not treat as equal-weight primary research; "
            "use only to add caveats, flag compliance notes, or note statistical uncertainty):\n\n"
            f"{advisory_blob}"
        )
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    system_prompt = (
        f"Today's date: {today_str}. "
        "You are a research synthesizer for an orchestrator. "
        "Produce concise, high-signal markdown with sections: Executive Summary, "
        "Key Findings, Uncertainties & Risks, Next Steps. Avoid fluff. "
        "When known project facts are provided, treat them as answered context. "
        "Do not wrap the response in triple backticks or fenced code blocks.\n\n"
        "SYNTHESIS DISCIPLINE: Do NOT summarize each agent sequentially "
        "('Agent X found... Agent Y found...'). Extract the highest-signal claims "
        "across ALL agents and write a unified narrative. The reader should not be "
        "able to tell which claim came from which agent.\n\n"
        "CONFIDENCE WEIGHTING: Each agent finding is labelled with confidence:HIGH/MED/LOW "
        "(self-assessed by the agent on a 1-5 scale). Weight your conclusions toward HIGH-confidence "
        "findings. If your summary relies heavily on MED or LOW findings, explicitly flag this "
        "in the Evidence Confidence line.\n\n"
        "EVIDENCE DISCIPLINE: Agent findings include [E]/[I]/[S] labels.\n"
        "- [E]: state confidently, include source domain or URL.\n"
        "- [I]: frame as inference — 'this suggests...'\n"
        "- [S]: frame as hypothesis — 'one possibility is...'\n"
        "Never launder [I] or [S] into presented facts. "
        "End Executive Summary with: 'Evidence Confidence: [High/Mixed/Low] — [one-line reason].' "
        "For time-sensitive topics, state whether events are upcoming, ongoing, or past relative to today."
    )
    history_lines: list[str] = []
    if isinstance(prior_messages, list):
        for row in prior_messages[-8:]:
            if not isinstance(row, dict):
                continue
            role = str(row.get("role", "")).strip().lower()
            content = str(row.get("content", "")).strip()
            if role not in {"user", "assistant"} or not content:
                continue
            history_lines.append(f"{role.upper()}: {content}")
    history_block = "\n".join(history_lines).strip()
    _conflict_section = ""
    if conflict_report and conflict_report.strip():
        _conflict_section = (
            f"\n\nCROSS-AGENT DISPUTES — reconcile these explicitly in your synthesis "
            f"(state which position has stronger evidence or note genuine uncertainty):\n"
            f"{conflict_report.strip()}"
        )
    user_prompt = (
        f"Question:\n{question}\n\n"
        f"Known project facts (if any):\n{project_context.strip() or '(none)'}\n\n"
        f"Recent command-thread history (if any):\n{history_block or '(none)'}\n\n"
        f"Research outputs:\n{findings_blob}"
        f"{_conflict_section}\n\n"
        "Return markdown only, not inside ``` fences."
    )
    validation_cycles = max(1, int(model_cfg.get("synthesis_validation_cycles", 3)))
    retry_attempts = max(1, int(model_cfg.get("synthesis_retry_attempts", 6)))
    retry_backoff_sec = max(0.0, float(model_cfg.get("synthesis_retry_backoff_sec", 1.5)))
    timeout = int(model_cfg.get("synthesis_timeout_sec", model_cfg.get("timeout_sec", 0)))
    fallback_models_raw = model_cfg.get("synthesis_fallback_models", [])
    fallback_models: list[str] = []
    if isinstance(fallback_models_raw, list):
        for item in fallback_models_raw:
            name = str(item or "").strip()
            if name:
                fallback_models.append(name)

    last_text = ""
    for cycle in range(validation_cycles):
        prompt = user_prompt
        if cycle > 0:
            prompt = (
                f"{user_prompt}\n\n"
                "Regenerate with stricter quality control. Ensure all required sections appear with clear headers."
            )
        try:
            candidate = client.chat(
                model=model,
                fallback_models=fallback_models,
                system_prompt=system_prompt,
                user_prompt=prompt,
                temperature=float(model_cfg.get("temperature", 0.2)),
                num_ctx=int(model_cfg.get("num_ctx", 16384)),
                think=bool(model_cfg.get("think", False)),
                timeout=timeout,
                retry_attempts=retry_attempts,
                retry_backoff_sec=retry_backoff_sec,
            )
            last_text = candidate
            if _is_valid_synthesis(candidate):
                return candidate
        except Exception:
            continue

    if _is_valid_synthesis(last_text):
        return last_text
    if last_text.strip():
        return (
            f"{last_text}\n\n"
            "_Reliability note: synthesis did not pass full section validation after retries; "
            "review before treating as final._"
        )
    return _fallback_synthesis(question, findings)


def run_skeptic_pass(
    question: str,
    synthesis: str,
    *,
    client: Any | None = None,
    model_cfg: dict | None = None,
) -> str:
    """
    Adversarial second pass on the completed synthesis.

    Runs the same model with a hostile system prompt that instructs it to
    challenge every claim, find unsupported conclusions, identify missing
    perspectives, and assess how easily the findings could be overturned.

    Returns skeptic critique markdown, or empty string if unavailable.
    The caller is responsible for appending this to the synthesis document.
    """
    if client is None or not model_cfg or not synthesis.strip():
        return ""
    model = str(model_cfg.get("model", "")).strip()
    if not model:
        return ""

    fallback_models_raw = model_cfg.get("synthesis_fallback_models", [])
    fallback_models: list[str] = (
        [str(m) for m in fallback_models_raw if str(m or "").strip()]
        if isinstance(fallback_models_raw, list)
        else []
    )

    system_prompt = (
        "You are the Skeptic Engine — an internal adversary whose only job is to stress-test "
        "research conclusions before they reach the user. You are not trying to be balanced or "
        "reassuring. You are trying to find every crack.\n\n"
        "For the synthesis provided, produce a structured critique covering:\n"
        "  1. Unsupported claims — assertions presented as fact with no [E] backing\n"
        "  2. Weak evidence — claims resting on a single source, low-tier source, or wire-laundered reporting\n"
        "  3. Missing perspectives — what expert voices, data types, or opposing viewpoints are absent\n"
        "  4. Conclusion vulnerabilities — what single piece of contradicting evidence would overturn the main findings\n"
        "  5. Confidence adjustment — one direct sentence on whether the reader's confidence should be "
        "higher, lower, or unchanged, and exactly why\n\n"
        "Format strictly as markdown. Be direct and specific. "
        "Do not summarise the synthesis back. Do not hedge. Attack the reasoning."
    )
    user_prompt = (
        f"Research question: {question}\n\n"
        f"Synthesis to challenge:\n{synthesis}\n\n"
        "Return your critique as markdown only."
    )

    try:
        result = client.chat(
            model=model,
            fallback_models=fallback_models,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.6,  # hardcoded above synthesis temp — skeptic needs adversarial latitude
            num_ctx=int(model_cfg.get("num_ctx", 16384)),
            think=False,
            timeout=int(model_cfg.get("synthesis_timeout_sec", model_cfg.get("timeout_sec", 0))),
            retry_attempts=2,
            retry_backoff_sec=1.5,
        )
        return str(result or "").strip()
    except Exception:
        return ""
