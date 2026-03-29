from __future__ import annotations

import re
from collections import OrderedDict
from typing import Any
from urllib.parse import urlparse

_DOMAIN_LABELS = {
    "ufcstats.com": "UFC Stats",
    "ufc.com": "UFC.com",
    "mmajunkie.usatoday.com": "MMA Junkie",
    "draftkings.com": "DraftKings Sportsbook",
    "cbssports.com": "CBS Sports",
    "espn.com": "ESPN",
    "wikipedia.org": "Wikipedia",
    "tapology.com": "Tapology",
    "sherdog.com": "Sherdog",
    "mmafighting.com": "MMA Fighting",
}

_SECTION_ALIASES = {
    "event overview": "Event Overview",
    "overview": "Event Overview",
    "summary": "Event Overview",
    "findings": "Event Overview",
    "fighter comparison": "Fighter Comparison",
    "matchup": "Fighter Comparison",
    "fighters": "Fighter Comparison",
    "evidence signals": "Evidence Signals",
    "evidence": "Evidence Signals",
    "key risks": "Key Risks / Open Questions",
    "open questions": "Key Risks / Open Questions",
    "uncertainties": "Key Risks / Open Questions",
    "remaining questions": "Key Risks / Open Questions",
    "unknowns": "Key Risks / Open Questions",
    "odds": "Odds / Market Snapshot",
    "market": "Odds / Market Snapshot",
    "betting": "Odds / Market Snapshot",
    "broadcast": "Broadcast / Timing",
    "timing": "Broadcast / Timing",
    "watch info": "Broadcast / Timing",
    "bottom line": "Bottom Line",
    "final take": "Bottom Line",
    "conclusion": "Bottom Line",
}

_SCHEMAS = {
    "sports_event": [
        "Event Overview",
        "Fact Card",
        "Fighter Comparison",
        "Evidence Signals",
        "Odds / Market Snapshot",
        "Broadcast / Timing",
        "Key Risks / Open Questions",
        "Bottom Line",
    ],
    "generic_research": [
        "Event Overview",
        "Evidence Signals",
        "Key Risks / Open Questions",
        "Bottom Line",
    ],
}


def _hostname(value: str) -> str:
    try:
        host = urlparse(value).hostname or ""
    except Exception:
        host = ""
    return host.lower().strip()


def format_source_label(source_obj: dict[str, Any]) -> str:
    host = _hostname(str(source_obj.get("url", ""))) or str(source_obj.get("source_domain", "")).strip().lower()
    if not host:
        return "Web source"
    for known, label in _DOMAIN_LABELS.items():
        if host == known or host.endswith("." + known):
            return label
    pieces = [p for p in host.split(".") if p and p not in {"www", "m", "amp"}]
    if not pieces:
        return host
    label = pieces[-2] if len(pieces) >= 2 else pieces[0]
    return label.replace("-", " ").title()


def source_labels(sources: list[dict[str, Any]], limit: int = 3) -> list[str]:
    labels: list[str] = []
    seen: set[str] = set()
    for row in sources:
        label = format_source_label(row)
        if label in seen:
            continue
        seen.add(label)
        labels.append(label)
        if len(labels) >= limit:
            break
    return labels


def normalize_heading(text: str) -> str | None:
    clean = re.sub(r"^[#\-*\s]+", "", str(text or "")).strip().lower()
    clean = re.sub(r"\s+", " ", clean)
    if not clean:
        return None
    if clean in _SECTION_ALIASES:
        return _SECTION_ALIASES[clean]
    for key, value in _SECTION_ALIASES.items():
        if key in clean:
            return value
    return clean.title()


def parse_markdown_sections(text: str) -> list[tuple[str, str]]:
    lines = str(text or "").splitlines()
    sections: list[tuple[str, str]] = []
    current_heading = "Event Overview"
    buffer: list[str] = []
    for line in lines:
        heading_match = re.match(r"^\s{0,3}(#{1,6})\s+(.+?)\s*$", line)
        if heading_match:
            body = "\n".join(buffer).strip()
            if body:
                sections.append((current_heading, body))
            current_heading = normalize_heading(heading_match.group(2)) or current_heading
            buffer = []
            continue
        buffer.append(line)
    body = "\n".join(buffer).strip()
    if body:
        sections.append((current_heading, body))
    return sections


def _dedupe_paragraphs(text: str) -> str:
    paras = [p.strip() for p in re.split(r"\n{2,}", str(text or "")) if p.strip()]
    kept: list[str] = []
    seen: set[str] = set()
    for para in paras:
        key = re.sub(r"\W+", "", para.lower())[:180]
        if key in seen:
            continue
        seen.add(key)
        kept.append(para)
    return "\n\n".join(kept)


def merge_duplicate_sections(sections: list[tuple[str, str]]) -> "OrderedDict[str, str]":
    merged: "OrderedDict[str, str]" = OrderedDict()
    for heading, body in sections:
        canonical = normalize_heading(heading) or "Event Overview"
        existing = merged.get(canonical, "")
        body_clean = _dedupe_paragraphs(body)
        if existing:
            merged[canonical] = _dedupe_paragraphs(existing + "\n\n" + body_clean)
        else:
            merged[canonical] = body_clean
    return merged


def _choose_schema(topic_type: str, question: str) -> list[str]:
    low = f"{topic_type} {question}".lower()
    if any(token in low for token in ["ufc", "mma", "boxing", "fight", "main card", "weigh-in"]):
        return _SCHEMAS["sports_event"]
    return _SCHEMAS["generic_research"]


def compose_research_summary(
    summary_text: str,
    *,
    sources: list[dict[str, Any]] | None = None,
    topic_type: str = "general",
    question: str = "",
    fact_card_md: str = "",
) -> str:
    raw = str(summary_text or "").strip()
    if not raw:
        return raw
    sections = merge_duplicate_sections(parse_markdown_sections(raw))
    labels = source_labels(sources or [], limit=4)
    schema = _choose_schema(topic_type, question)

    if fact_card_md.strip():
        existing = sections.get("Fact Card", "")
        sections["Fact Card"] = fact_card_md.strip() + ("\n\n" + existing if existing else "")

    if labels:
        existing = sections.get("Event Overview", "")
        source_note = "Sources emphasized: " + ", ".join(labels) + "."
        if source_note not in existing:
            sections["Event Overview"] = (source_note + "\n\n" + existing).strip()

    ordered: list[str] = []
    used: set[str] = set()
    title_match = re.match(r"^#\s+.+$", raw.splitlines()[0].strip()) if raw.splitlines() else None
    if title_match:
        ordered.append(raw.splitlines()[0].strip())
        ordered.append("")
    else:
        ordered.append("# Research Synthesis")
        ordered.append("")

    for heading in schema:
        body = sections.get(heading, "").strip()
        if not body:
            continue
        used.add(heading)
        ordered.append(f"## {heading}")
        ordered.append("")
        ordered.append(body)
        ordered.append("")

    for heading, body in sections.items():
        if heading in used or not body.strip():
            continue
        ordered.append(f"## {heading}")
        ordered.append("")
        ordered.append(body.strip())
        ordered.append("")

    return "\n".join(ordered).strip() + "\n"



def evaluate_answer_confidence(
    *,
    sources: list[dict[str, Any]] | None = None,
    conflict_summary: dict[str, Any] | None = None,
    question: str = '',
) -> dict[str, Any]:
    rows = sources or []
    unique_domains = {str(r.get('source_domain', '')).strip().lower() for r in rows if str(r.get('source_domain', '')).strip()}
    tier1 = sum(1 for r in rows if str(r.get('source_tier', '')).strip() == 'tier1')
    fresh = [float(r.get('freshness_score', 0.0) or 0.0) for r in rows]
    avg_fresh = (sum(fresh) / len(fresh)) if fresh else 0.0
    conflict_count = int((conflict_summary or {}).get('conflict_count', 0))
    score = 0.0
    score += min(0.35, len(rows) * 0.06)
    score += min(0.2, len(unique_domains) * 0.05)
    score += 0.15 if tier1 else 0.0
    score += min(0.2, avg_fresh * 0.2)
    score -= min(0.25, conflict_count * 0.08)
    score = max(0.0, min(1.0, round(score, 3)))
    mode = 'high' if score >= 0.72 else 'medium' if score >= 0.48 else 'low'
    notes: list[str] = []
    if tier1 == 0 and rows:
        notes.append('No tier-1 anchor source was found.')
    if len(unique_domains) < 3 and rows:
        notes.append('Source diversity is limited.')
    if conflict_count:
        notes.append(f'{conflict_count} source conflict flag(s) detected.')
    if avg_fresh < 0.4 and rows:
        notes.append('Freshness is weak for this query.')
    return {
        'score': score,
        'mode': mode,
        'notes': notes,
        'unique_domains': len(unique_domains),
        'tier1_count': tier1,
    }
