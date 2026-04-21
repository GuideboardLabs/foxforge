from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
import re
import time
from urllib.parse import urlsplit
from typing import Any, Callable

from agents_research.synthesizer import synthesize, run_skeptic_pass
from shared_tools.embedding_memory import _vec_cosine
from shared_tools.file_store import ProjectStore
from shared_tools.feedback_learning import FeedbackLearningEngine
from shared_tools.model_routing import lane_model_config
from shared_tools.inference_router import InferenceRouter
from shared_tools.ollama_client import OllamaClient


_URL_PATTERN = re.compile(
    r"https?://\S+"
    r"|(?<!\w)(?:[a-zA-Z0-9-]+\.(?:com|org|gov|edu|io|net|co|uk|de|fr|ca|au))(?:/\S*)?",
    re.IGNORECASE,
)
_WEB_CONTEXT_SOURCE_RE = re.compile(r"^\-\s.*\|\s+(https?://\S+)\s*$", re.IGNORECASE)


def _domain_from_url(url: str) -> str:
    try:
        return str(urlsplit(str(url)).hostname or "").lower().removeprefix("www.")
    except Exception:
        return ""


def _extract_web_source_evidence(web_context: str) -> list[dict[str, str]]:
    evidence: list[dict[str, str]] = []
    if not str(web_context or "").strip():
        return evidence
    current: dict[str, str] | None = None
    for raw_line in str(web_context).splitlines():
        line = str(raw_line or "").rstrip()
        match = _WEB_CONTEXT_SOURCE_RE.match(line.strip())
        if match:
            if current and current.get("url"):
                evidence.append(current)
            url = str(match.group(1)).strip()
            current = {"url": url, "domain": _domain_from_url(url), "snippet": ""}
            continue
        if current is not None and line.strip().startswith("snippet:"):
            current["snippet"] = line.split("snippet:", 1)[1].strip()
    if current and current.get("url"):
        evidence.append(current)
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in evidence:
        url = str(row.get("url", "")).strip()
        if not url or url in seen:
            continue
        seen.add(url)
        out.append(row)
    return out


def _audit_evidence_labels(findings: list[dict]) -> list[dict]:
    """Downgrade [E] labels unless they align to known source evidence."""
    emb_client: OllamaClient | None = None
    line_vec_cache: dict[str, list[float]] = {}
    snippet_vec_cache: dict[str, list[float]] = {}

    def _client() -> OllamaClient | None:
        nonlocal emb_client
        if emb_client is not None:
            return emb_client
        try:
            emb_client = OllamaClient()
            return emb_client
        except Exception:
            return None

    def _alignment(line_text: str, source_evidence: list[dict[str, str]]) -> float:
        snippets = [str(row.get("snippet", "")).strip() for row in source_evidence if str(row.get("snippet", "")).strip()]
        if not snippets:
            return 0.0
        client = _client()
        if client is None:
            # Fallback: token overlap when embedding path is unavailable.
            line_words = set(re.findall(r"[a-z0-9]{4,}", line_text.lower()))
            if not line_words:
                return 0.0
            best = 0.0
            for snippet in snippets:
                snippet_words = set(re.findall(r"[a-z0-9]{4,}", snippet.lower()))
                if not snippet_words:
                    continue
                overlap = len(line_words & snippet_words) / max(1, len(line_words))
                if overlap > best:
                    best = overlap
            return float(best)
        try:
            line_key = line_text[:600]
            if line_key not in line_vec_cache:
                line_vec_cache[line_key] = client.embed("qwen3-embedding:4b", line_key, timeout=20)
            line_vec = line_vec_cache[line_key]
            best = 0.0
            for snippet in snippets:
                snippet_key = snippet[:1200]
                if snippet_key not in snippet_vec_cache:
                    snippet_vec_cache[snippet_key] = client.embed("qwen3-embedding:4b", snippet_key, timeout=20)
                score = _vec_cosine(line_vec, snippet_vec_cache[snippet_key])
                if score > best:
                    best = score
            return float(best)
        except Exception:
            return 0.0

    result: list[dict] = []
    for item in findings:
        text = str(item.get("finding", ""))
        if "[E]" not in text:
            result.append(item)
            continue
        source_evidence = [dict(x) for x in item.get("source_evidence", []) if isinstance(x, dict)]
        source_domains = {
            str(x.get("domain", "")).strip().lower()
            for x in source_evidence
            if str(x.get("domain", "")).strip()
        }
        source_urls = [str(x.get("url", "")).strip() for x in source_evidence if str(x.get("url", "")).strip()]
        alignment_scores: list[float] = []
        lines = text.split("\n")
        new_lines: list[str] = []
        for i, line in enumerate(lines):
            if "[E]" not in line:
                new_lines.append(line)
                continue
            window = line + (" " + lines[i + 1] if i + 1 < len(lines) else "")
            if _URL_PATTERN.search(window):
                new_lines.append(line)
                continue
            low_window = window.lower()
            if any(domain and domain in low_window for domain in source_domains):
                new_lines.append(line)
                continue
            score = _alignment(line, source_evidence)
            alignment_scores.append(score)
            if score >= 0.55 and source_urls:
                new_lines.append(line)
                continue
            new_lines.append(line.replace("[E]", "[I]"))
        new_item = dict(item)
        new_item["finding"] = "\n".join(new_lines)
        if source_urls:
            new_item["source_urls"] = source_urls
        if alignment_scores:
            new_item["evidence_alignment_max"] = round(max(alignment_scores), 3)
        result.append(new_item)
    return result


def _self_check(client: OllamaClient, model_cfg: dict, question: str, finding: str) -> int:
    """Ask the agent to rate its own finding quality. Returns 1-5 or 0 on failure."""
    model = str(model_cfg.get("model", "")).strip()
    if not model or not client or not finding:
        return 0
    try:
        result = client.chat(
            model=model,
            system_prompt=(
                "Rate the quality and relevance of this research finding on a scale of 1-5.\n"
                "1=poor/off-topic or contains specific numbers, dates, names, or quotes with no cited source URL.\n"
                "2=weak — relevant but mostly unsourced or vague.\n"
                "3=adequate — answers the question with mostly sourced [E] claims.\n"
                "4=good — well-sourced, directly relevant, clear [E]/[I]/[S] discipline.\n"
                "5=excellent — directly answers the question, all specific claims cited, no apparent fabrication.\n"
                "Deduct at least 2 points if ANY [E] claim lacks an immediately following source URL or domain. "
                "[E] always requires a citation regardless of whether the claim is a statistic, name, or general observation. "
                "General knowledge presented as [E] without a source is a fabrication error.\n"
                "Reply with ONLY a single digit 1-5."
            ),
            user_prompt=f"Question: {question[:200]}\n\nFinding: {finding[:600]}",
            temperature=0.0,
            num_ctx=512,
            think=False,
            timeout=20,
            retry_attempts=1,
            retry_backoff_sec=0.5,
        )
        _match = re.search(r"[1-5]", str(result or "").strip())
        digit = _match.group(0) if _match else ""
        if digit in {"1", "2", "3", "4", "5"}:
            return int(digit)
    except Exception:
        pass
    return 0


def _gap_assess(client: Any, model_cfg: dict, question: str, summary_md: str) -> list[str]:
    """Identify 1-3 research gaps in the synthesis using a fast LLM call.

    Returns a list of specific gap questions, or [] on failure/timeout (loop is skipped).
    Uses a 15s timeout with no retries — if the model is busy, skip gap fill entirely.
    """
    model = str(model_cfg.get("model", "")).strip()
    if not model or not client or not summary_md.strip():
        return []
    try:
        result = client.chat(
            model=model,
            system_prompt=(
                "You are a research gap analyst. Your job is to identify the most important claims "
                "in a research synthesis that lack direct supporting evidence or remain unresolved. "
                "Output ONLY the gap questions, one per line. No preamble, no numbering, no explanations. "
                "Maximum 3 questions. If the synthesis is comprehensive, output just 1."
            ),
            user_prompt=(
                f"Research question: {question[:300]}\n\n"
                f"Synthesis:\n{summary_md[:3000]}\n\n"
                "List the 2-3 most important gaps as specific research questions, one per line:"
            ),
            temperature=0.2,
            num_ctx=1024,
            think=False,
            timeout=15,
            retry_attempts=1,
            retry_backoff_sec=0.5,
        )
        raw = str(result or "").strip()
        gaps = [
            line.strip().lstrip("0123456789.-) \t").strip()
            for line in raw.splitlines()
            if line.strip()
        ]
        gaps = [g for g in gaps if len(g) >= 15][:3]
        return gaps
    except Exception:
        return []


RESEARCH_PERSONAS = [
    ("market_analyst", "Focus on market dynamics, alternatives, and strategic positioning."),
    ("technical_researcher", "Focus on technical feasibility, architecture tradeoffs, and bottlenecks."),
    ("risk_researcher", "Focus on risks, failure modes, constraints, and mitigation plans."),
    ("execution_planner", "Focus on practical sequencing, milestones, and resource-fit execution."),
]
DEFAULT_DIRECTIVES = {persona: directive for persona, directive in RESEARCH_PERSONAS}
ANALYSIS_PROFILE_TECHNICAL      = "technical_analysis"
ANALYSIS_PROFILE_GENERAL        = "general_analysis"
ANALYSIS_PROFILE_MEDICAL        = "medical_analysis"
ANALYSIS_PROFILE_PARENTING      = "parenting_analysis"
ANALYSIS_PROFILE_FINANCE        = "finance_analysis"
ANALYSIS_PROFILE_SPORTS         = "sports_analysis"
ANALYSIS_PROFILE_HISTORY        = "history_analysis"
ANALYSIS_PROFILE_SCIENCE        = "science_analysis"
ANALYSIS_PROFILE_MATH           = "math_analysis"
ANALYSIS_PROFILE_POLITICS       = "politics_analysis"
ANALYSIS_PROFILE_CURRENT_EVENTS = "current_events_analysis"
ANALYSIS_PROFILE_UNDERGROUND    = "underground_analysis"
STATISTICAL_ANALYSIS_PERSONA = "statistical_analysis"
STATISTICAL_ANALYSIS_DIRECTIVE = (
    "Focus on statistical patterns, trend quality, uncertainty bounds, and bias checks. "
    "Prioritize quantified signal over speculation."
)
LEGAL_ANALYSIS_PERSONA = "legal_analysis"
LEGAL_ANALYSIS_DIRECTIVE = (
    "Focus on legal and compliance constraints, jurisdiction caveats, and explicit risk language. "
    "Flag where professional legal counsel is required."
)
STATISTICAL_ANALYSIS_MODEL = "qwen3:8b"
LEGAL_ANALYSIS_MODEL = "qwen3:8b"

TOPIC_TYPE_TO_PROFILE: dict[str, str] = {
    "sports":         ANALYSIS_PROFILE_SPORTS,
    "technical":      ANALYSIS_PROFILE_TECHNICAL,
    "medical":        ANALYSIS_PROFILE_MEDICAL,
    "animal_care":    ANALYSIS_PROFILE_MEDICAL,
    "finance":        ANALYSIS_PROFILE_FINANCE,
    "history":        ANALYSIS_PROFILE_HISTORY,
    "science":        ANALYSIS_PROFILE_SCIENCE,
    "math":           ANALYSIS_PROFILE_MATH,
    "politics":       ANALYSIS_PROFILE_POLITICS,
    "current_events": ANALYSIS_PROFILE_CURRENT_EVENTS,
    "general":        ANALYSIS_PROFILE_GENERAL,
    "underground":    ANALYSIS_PROFILE_UNDERGROUND,
    "business":       ANALYSIS_PROFILE_FINANCE,
    "law":            ANALYSIS_PROFILE_POLITICS,
    "education":      ANALYSIS_PROFILE_GENERAL,
    "travel":         ANALYSIS_PROFILE_GENERAL,
    "food":           ANALYSIS_PROFILE_GENERAL,
    "gaming":         ANALYSIS_PROFILE_TECHNICAL,
    "books":          ANALYSIS_PROFILE_GENERAL,
    "real_estate":    ANALYSIS_PROFILE_FINANCE,
    "automotive":     ANALYSIS_PROFILE_TECHNICAL,
    "parenting":      ANALYSIS_PROFILE_PARENTING,
    "tv_shows":       ANALYSIS_PROFILE_CURRENT_EVENTS,
    "movies":         ANALYSIS_PROFILE_CURRENT_EVENTS,
    "music":          ANALYSIS_PROFILE_CURRENT_EVENTS,
    "art":            ANALYSIS_PROFILE_CURRENT_EVENTS,
}


def _analysis_profile_for_type(topic_type: str) -> str:
    return TOPIC_TYPE_TO_PROFILE.get(str(topic_type).strip().lower(), ANALYSIS_PROFILE_GENERAL)


def _sanitize_model_list(raw_models: Any) -> list[str]:
    models: list[str] = []
    if isinstance(raw_models, list):
        for entry in raw_models:
            name = str(entry or "").strip()
            if not name:
                continue
            # Fully retire this model from research usage.
            if name.lower() == "qwen3:4b":
                continue
            if name not in models:
                models.append(name)
    return models


def _profile_agent_templates(profile: str) -> list[dict[str, Any]]:
    if profile == ANALYSIS_PROFILE_SPORTS:
        return [
            {
                "persona": "sports_context_researcher",
                "model": "qwen3:8b",
                "directive": (
                    "Focus on current schedules, rosters, recent form, rankings, and event context. "
                    "For combat sports: confirm weight class, title type (divisional vs symbolic belt such as BMF), "
                    "event date relative to today, and flag card changes or injury substitutions."
                ),
            },
            {
                "persona": "sports_stats_and_history_researcher",
                "model": "deepseek-r1:8b",
                "directive": (
                    "Focus on head-to-head records, statistical trends, historical performance trajectory. "
                    "Cite specific figures with dates."
                ),
            },
            {
                "persona": "sports_risk_analyst",
                "model": "qwen3:8b",
                "directive": (
                    "Focus on injury reports, availability uncertainty, current momentum, venue/officiating factors, "
                    "and what could shift the expected outcome."
                ),
            },
        ]
    if profile == ANALYSIS_PROFILE_TECHNICAL:
        return [
            {
                "persona": "technical_architecture_researcher",
                "model": "deepseek-r1:8b",
                "directive": (
                    "Focus on system design patterns, architectural tradeoffs, scalability, and technology choices. "
                    "Compare competing approaches with evidence."
                ),
            },
            {
                "persona": "technical_implementation_researcher",
                "model": "qwen2.5-coder:7b",
                "directive": (
                    "Focus on concrete implementation patterns, library/framework comparisons, code-level feasibility, "
                    "API shapes, version specifics, and known gotchas."
                ),
            },
            {
                "persona": "technical_risk_researcher",
                "model": "deepseek-r1:8b",
                "directive": (
                    "Focus on security vulnerabilities, failure modes, performance bottlenecks, "
                    "maintenance burden, and technical debt."
                ),
            },
            {
                "persona": "technical_market_analyst",
                "model": "qwen3:8b",
                "directive": (
                    "Focus on ecosystem maturity, adoption trends, community support, and competitive alternatives."
                ),
                "role": "advisory",
            },
        ]
    if profile == ANALYSIS_PROFILE_MEDICAL:
        return [
            {
                "persona": "clinical_evidence_researcher",
                "model": "deepseek-r1:8b",
                "directive": (
                    "Focus on peer-reviewed evidence, trial data, systematic reviews. Note study quality, sample sizes, recency. "
                    "Tag by evidence tier: RCT > observational > case study > expert opinion. "
                    "For every statistic or prevalence figure, include the publication year and flag if data is older than 3 years."
                ),
            },
            {
                "persona": "guideline_verifier",
                "model": "qwen3:8b",
                "directive": (
                    "Cross-check against current clinical guidelines (WHO, CDC, NIH, specialty societies). "
                    "Explicitly state the guideline version year (e.g., 'CDC 2023'). Flag when the most recent guideline is more than 3 years old. "
                    "Note evidence-guideline divergences and any guidelines under active revision."
                ),
            },
            {
                "persona": "safety_risk_researcher",
                "model": "deepseek-r1:8b",
                "directive": (
                    "Focus on contraindications, adverse event profiles, drug interactions, population-specific risks. "
                    "Flag black box warnings and active regulatory advisories."
                ),
            },
            {
                "persona": STATISTICAL_ANALYSIS_PERSONA,
                "model": STATISTICAL_ANALYSIS_MODEL,
                "directive": STATISTICAL_ANALYSIS_DIRECTIVE,
                "role": "advisory",
            },
            {
                "persona": LEGAL_ANALYSIS_PERSONA,
                "model": LEGAL_ANALYSIS_MODEL,
                "directive": LEGAL_ANALYSIS_DIRECTIVE,
                "role": "advisory",
            },
        ]
    if profile == ANALYSIS_PROFILE_PARENTING:
        return [
            {
                "persona": "developmental_evidence_researcher",
                "model": "deepseek-r1:8b",
                "directive": (
                    "Focus on peer-reviewed developmental psychology, pediatric research, and educational studies. "
                    "Tag evidence tier (RCT > observational > case study > expert opinion) and include publication years. "
                    "Flag any prevalence statistics with their source year — developmental norms shift over time. "
                    "For neurodiverse populations (autism, ADHD, sensory processing differences), note when study samples "
                    "are representative vs. skewed (e.g., predominantly male samples, clinical vs. community populations)."
                ),
            },
            {
                "persona": "clinical_guideline_verifier",
                "model": "qwen3:8b",
                "directive": (
                    "Cross-check against current pediatric and developmental guidelines (AAP, CDC, AOTA, ASHA, DSM-5-TR). "
                    "Explicitly state guideline version years. Flag guidelines older than 3 years. "
                    "Note where guidelines are being revised or where evidence and current practice diverge."
                ),
            },
            {
                "persona": "neurodiversity_perspective_researcher",
                "model": "qwen3:8b",
                "directive": (
                    "Actively seek neurodiversity-affirming frameworks, perspectives, and research. "
                    "This means: (1) Look for research and guidance written from a strengths-based or identity-affirming lens, not deficit-only. "
                    "(2) Identify where the primary literature reflects a predominantly neurotypical or pathology framing and flag it. "
                    "(3) Seek out autistic self-advocate perspectives, disability justice viewpoints, and culturally responsive approaches. "
                    "(4) Flag where interventions have been critiqued by the autistic community vs. endorsed. "
                    "(5) Look for intersectional considerations: how do gender, race, culture, and socioeconomic status affect diagnosis rates, "
                    "access to support, and outcomes for neurodiverse children?"
                ),
            },
            {
                "persona": "practical_family_advisor",
                "model": "qwen3:8b",
                "directive": (
                    "Focus on actionable, practical strategies families can use. Prioritize approaches that have real-world parent/caregiver evidence. "
                    "Identify what school systems, therapists, and pediatricians can be asked for specifically. "
                    "Flag cost, accessibility, and availability barriers. "
                    "Note where online communities (e.g., autistic-led spaces, parent support groups) offer supplementary lived-experience knowledge "
                    "beyond what appears in clinical literature."
                ),
            },
        ]
    if profile == ANALYSIS_PROFILE_FINANCE:
        return [
            {
                "persona": "macro_market_researcher",
                "model": "deepseek-r1:8b",
                "directive": (
                    "Focus on macroeconomic indicators, market trends, sector dynamics, monetary/fiscal policy. "
                    "Cite data points with sources and dates."
                ),
            },
            {
                "persona": "fundamentals_researcher",
                "model": "qwen3:8b",
                "directive": (
                    "Focus on valuation multiples, earnings/revenue trends, balance sheet health, competitive positioning."
                ),
            },
            {
                "persona": "risk_stress_researcher",
                "model": "deepseek-r1:8b",
                "directive": (
                    "Focus on downside scenarios, tail risks, liquidity constraints, regulatory headwinds. "
                    "What breaks this thesis first?"
                ),
            },
            {
                "persona": LEGAL_ANALYSIS_PERSONA,
                "model": LEGAL_ANALYSIS_MODEL,
                "directive": LEGAL_ANALYSIS_DIRECTIVE,
                "role": "advisory",
            },
            {
                "persona": STATISTICAL_ANALYSIS_PERSONA,
                "model": STATISTICAL_ANALYSIS_MODEL,
                "directive": STATISTICAL_ANALYSIS_DIRECTIVE,
                "role": "advisory",
            },
        ]
    if profile == ANALYSIS_PROFILE_HISTORY:
        return [
            {
                "persona": "history_timeline_researcher",
                "model": "qwen3:8b",
                "directive": (
                    "Focus on chronology, causal chains, periodization with explicit date anchors. "
                    "Identify pivotal turning points and distinguish immediate causes from structural forces."
                ),
            },
            {
                "persona": "history_source_critic",
                "model": "deepseek-r1:8b",
                "directive": (
                    "Focus on source quality, authorial bias, historiographical disputes, and missing/contested evidence. "
                    "Actively challenge the dominant narrative."
                ),
            },
            {
                "persona": "history_comparative_analyst",
                "model": "qwen3:8b",
                "directive": (
                    "Focus on parallels with other periods or regions. What does this resemble? "
                    "What's different? What precedents exist and how reliable are they?"
                ),
            },
        ]
    if profile == ANALYSIS_PROFILE_SCIENCE:
        return [
            {
                "persona": "scientific_evidence_researcher",
                "model": "deepseek-r1:8b",
                "directive": (
                    "Focus on peer-reviewed research, experimental findings, and current scientific consensus. "
                    "Note methodology quality, replication status. Distinguish established consensus from active frontier debate."
                ),
            },
            {
                "persona": "frontier_science_analyst",
                "model": "qwen3:8b",
                "directive": (
                    "Focus on cutting-edge preprints, recent papers, emerging findings, and where the field is actively moving. "
                    "Flag contested vs widely accepted claims."
                ),
            },
            {
                "persona": "science_application_researcher",
                "model": "qwen3:8b",
                "directive": (
                    "Focus on real-world applications, technology readiness level, practical implications, "
                    "and how this science connects to existing technologies or societal challenges."
                ),
            },
            {
                "persona": STATISTICAL_ANALYSIS_PERSONA,
                "model": STATISTICAL_ANALYSIS_MODEL,
                "directive": STATISTICAL_ANALYSIS_DIRECTIVE,
                "role": "advisory",
            },
        ]
    if profile == ANALYSIS_PROFILE_MATH:
        return [
            {
                "persona": "formal_reasoning_researcher",
                "model": "deepseek-r1:8b",
                "directive": (
                    "Focus on rigorous mathematical foundations, proof structures, axioms and assumptions, logical validity. "
                    "Identify where informal reasoning substitutes for proof."
                ),
            },
            {
                "persona": "computational_methods_researcher",
                "model": "qwen2.5-coder:7b",
                "directive": (
                    "Focus on algorithms, numerical methods, computational complexity, and implementation approaches. "
                    "Compare efficiency and accuracy tradeoffs with examples."
                ),
            },
            {
                "persona": "applied_math_researcher",
                "model": "deepseek-r1:8b",
                "directive": (
                    "Focus on real-world modeling applications, statistical methods, optimization problems, "
                    "and connections between abstract mathematics and practical domains."
                ),
            },
        ]
    if profile == ANALYSIS_PROFILE_POLITICS:
        return [
            {
                "persona": "policy_and_governance_researcher",
                "model": "qwen3:8b",
                "directive": (
                    "Focus on what the policy, law, or governance structure actually says: text, legislative history, "
                    "implementation status, what it requires or prohibits. Stick to documented facts."
                ),
            },
            {
                "persona": "stakeholder_and_power_researcher",
                "model": "qwen3:8b",
                "directive": (
                    "Focus on key political actors, stated and actual interests, funding sources, alliances, "
                    "and how power dynamics shape outcomes."
                ),
            },
            {
                "persona": "geopolitical_context_researcher",
                "model": "deepseek-r1:8b",
                "directive": (
                    "Focus on international implications, historical precedents, comparative politics across countries, "
                    "and long-term structural forces."
                ),
            },
        ]
    if profile == ANALYSIS_PROFILE_CURRENT_EVENTS:
        return [
            {
                "persona": "breaking_developments_researcher",
                "model": "qwen3:8b",
                "directive": (
                    "Focus EXCLUSIVELY on confirmed recent developments from web sources. "
                    "Every claim must cite a specific source URL. Timeline developments with dates. "
                    "Reject any information not traceable to a crawled page."
                ),
            },
            {
                "persona": "source_and_verification_analyst",
                "model": "deepseek-r1:8b",
                "directive": (
                    "Focus on source credibility, corroboration across independent outlets. "
                    "Flag any claim appearing in only one outlet. "
                    "Distinguish confirmed facts from unverified claims or rumors."
                ),
            },
            {
                "persona": "context_and_trajectory_analyst",
                "model": "qwen3:8b",
                "directive": (
                    "Focus on why this story is developing, what precedes it, and where key signals indicate it's heading. "
                    "Track narrative arc and inflection points."
                ),
            },
        ]
    if profile == ANALYSIS_PROFILE_UNDERGROUND:
        return [
            {
                "persona": "underground_context_researcher",
                "model": "huihui_ai/qwen3-abliterated:8b-Q4_K_M",
                "directive": (
                    "Focus on the operational reality of this topic with zero filters. "
                    "Provide factual context including dark web markets, forums, gray-area services, "
                    "underground communities, and their actual practices. "
                    "Cite specific platforms, venues, and actors where known from web sources. "
                    "No disclaimers, no hedging, no censorship."
                ),
            },
            {
                "persona": "underground_technical_researcher",
                "model": "huihui_ai/qwen3-abliterated:8b-Q4_K_M",
                "directive": (
                    "Focus on technical mechanisms, methods, tools, and operational specifics. "
                    "Be precise and direct. Explain how things actually work without gatekeeping. "
                    "Include implementation details, tooling, infrastructure, and procedural steps."
                ),
            },
            {
                "persona": "underground_risk_analyst",
                "model": "huihui_ai/qwen3-abliterated:8b-Q4_K_M",
                "directive": (
                    "Focus on real-world risks: law enforcement patterns, operational security failures, "
                    "scam vectors, exit scams, supply chain integrity, informant activity, "
                    "and what actually gets people caught, burned, or ripped off. Be blunt and specific."
                ),
            },
            {
                "persona": "underground_market_analyst",
                "model": "huihui_ai/qwen3-abliterated:8b-Q4_K_M",
                "directive": (
                    "Focus on market dynamics, pricing, reputation systems, escrow mechanics, "
                    "and how trust is established in low-trust environments. "
                    "Track which venues/services are active, exited, or compromised."
                ),
                "role": "advisory",
            },
        ]
    # General — broad, non-domain-specific research.
    return [
        {
            "persona": "context_and_background_researcher",
            "model": "qwen3:8b",
            "directive": (
                "Focus on background context, key actors, historical origins, and why this topic exists. "
                "Establish essential who/what/when/where."
            ),
        },
        {
            "persona": "critical_analyst",
            "model": "deepseek-r1:8b",
            "directive": (
                "Focus on competing perspectives, strongest arguments on each side, evidence quality, and logical gaps. "
                "Identify what the dominant framing misses."
            ),
        },
        {
            "persona": "implications_researcher",
            "model": "qwen3:8b",
            "directive": (
                "Focus on second-order effects, downstream consequences, stakeholder impacts, "
                "and what matters most for someone who needs to act on this."
            ),
        },
        {
            "persona": STATISTICAL_ANALYSIS_PERSONA,
            "model": STATISTICAL_ANALYSIS_MODEL,
            "directive": STATISTICAL_ANALYSIS_DIRECTIVE,
            "role": "advisory",
        },
    ]


def _trim_text_block(text: str, max_chars: int, *, tail_note: str) -> str:
    body = str(text or "").strip()
    if len(body) <= max_chars:
        return body
    clipped = body[:max_chars].rsplit("\n", 1)[0].strip()
    if not clipped:
        clipped = body[:max_chars].strip()
    removed = max(0, len(body) - len(clipped))
    return f"{clipped}\n\n[{tail_note}; trimmed {removed} chars]"


def _is_failure_text(text: str) -> bool:
    low = str(text or "").strip().lower()
    if not low:
        return True
    markers = [
        "model call failed",
        "fallback failed",
        "ollama chat failed",
        "no model configured",
        "could not connect to ollama",
        "ollama http 5",
        "traceback",
    ]
    return any(token in low for token in markers)


def _looks_like_research_note(text: str) -> bool:
    body = str(text or "").strip()
    if len(body) < 220:
        return False
    if _is_failure_text(body):
        return False
    low = body.lower()
    section_hits = 0
    for token in ("findings", "evidence", "open questions", "open question", "risks", "next steps"):
        if token in low:
            section_hits += 1
    return section_hits >= 2


def _agent_prompt(question: str, persona: str, directive: str, learned_guidance: str, web_context: str, max_web_chars: int = 9000) -> tuple[str, str]:
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    guidance_block = f"\n\n{learned_guidance}" if learned_guidance else ""
    web_block = ""
    if web_context.strip():
        web_context_trimmed = _trim_text_block(
            web_context,
            max_chars=max_web_chars,
            tail_note="web source cache truncated for reliability",
        )
        web_block = (
            "\n\nUse the web source context below selectively and cite source URLs in your notes. "
            "Do not discuss how sources were obtained."
            f"\n\nWeb source context:\n{web_context_trimmed}"
        )
    system_prompt = (
        f"Today's date: {today_str}. "
        "You are a Foraging sub-agent in a multi-agent council. "
        f"Your role is {persona}. {directive} "
        "Be concrete and avoid vague statements. "
        "Format output as markdown with sections: Findings, Evidence Signals, Open Questions.\n\n"
        "CLAIM LABELING — tag every substantive claim with one of:\n"
        "  [E] directly supported by a cited source or explicit data point\n"
        "  [I] logically inferred from evidence — reasonable but not directly stated\n"
        "  [S] speculative or hypothetical — plausible but no direct source backing\n"
        "Cite the source URL or domain after every [E] claim. "
        "Never present [I] or [S] claims as established facts.\n\n"
        "FABRICATION PROHIBITED: Do not state specific numbers, statistics, dates, names, "
        "product versions, prices, or direct quotes without a cited source URL. "
        "If you cannot find a source for a specific detail, omit it or write "
        "'[source not found]' — do not guess or approximate. "
        "Stating 'the available sources do not cover this' is correct and preferred over filling gaps. "
        "Use [S] only as a last resort for genuine hypotheses, never to launder missing facts. "
        "Uncertainty is not a failure — fabrication is."
        f"{guidance_block}{web_block}"
    )
    user_prompt = (
        f"Research request:\n{question}\n\n"
        "Return high-signal research notes that can be merged by a synthesizer."
    )
    return system_prompt, user_prompt


def _history_block(prior_messages: list[dict[str, str]] | None, limit_turns: int = 10) -> str:
    if not isinstance(prior_messages, list):
        return ""
    rows: list[str] = []
    for row in prior_messages[-max(6, limit_turns * 2) :]:
        if not isinstance(row, dict):
            continue
        role = str(row.get("role", "")).strip().lower()
        content = _trim_text_block(
            str(row.get("content", "")).strip(),
            max_chars=520,
            tail_note="message truncated",
        )
        if role not in {"user", "assistant"} or not content:
            continue
        tag = "USER" if role == "user" else "ASSISTANT"
        rows.append(f"{tag}: {content}")
    if not rows:
        return ""
    return "Recent command-thread context:\n" + "\n".join(rows)


_MULTI_PASS_BATCH_SIZE = 6   # sources per LLM pass (doubled from 3 — models have 24K ctx)
_MULTI_PASS_THRESHOLD = 4   # only batch when there are more than this many source blocks


def _split_web_sources(web_context: str) -> tuple[str, list[str]]:
    """Split web_context into (header_line, [source_block, ...]).

    Each source block starts with a "- " line (tier/depth prefix) as written by
    WebResearchEngine.web_context_for_project().
    """
    lines = web_context.strip().split("\n")
    if len(lines) <= 1:
        return web_context, []
    header = lines[0]
    source_blocks: list[str] = []
    current: list[str] = []
    for line in lines[1:]:
        if line.startswith("- "):
            if current:
                source_blocks.append("\n".join(current))
            current = [line]
        elif current:
            current.append(line)
    if current:
        source_blocks.append("\n".join(current))
    return header, source_blocks


def _run_one_agent(
    client: OllamaClient,
    model_cfg: dict[str, Any],
    agent_cfg: dict[str, Any],
    question: str,
    learned_guidance: str,
    web_context: str,
    source_evidence: list[dict[str, str]] | None,
    project_context: str,
    prior_messages: list[dict[str, str]] | None,
    cancel_checker: Callable[[], bool] | None = None,
    pause_checker: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    persona = str(agent_cfg.get("persona", "")).strip() or "research_agent"
    directive = str(agent_cfg.get("directive", "")).strip() or DEFAULT_DIRECTIVES.get(
        persona,
        "Focus on evidence quality, contradictions, and practical implications.",
    )
    base_model = str(model_cfg.get("model", "")).strip()
    requested_model = str(agent_cfg.get("model", "")).strip() or base_model
    if not requested_model:
        return {
            "agent": persona,
            "model": "",
            "requested_model": "",
            "finding": "No model configured for research_pool.",
            "source_urls": [],
            "source_evidence": [],
        }

    _max_web = 30000 if persona.startswith("breaking_") else (24000 if persona.startswith("sports_") else 20000)
    system_prompt, user_prompt = _agent_prompt(question, persona, directive, learned_guidance, web_context, max_web_chars=_max_web)
    context_blocks: list[str] = []
    if project_context.strip():
        context_blocks.append(project_context.strip())
    history = _history_block(prior_messages, limit_turns=12)
    if history:
        context_blocks.append(history)
    if context_blocks:
        user_prompt = f"{user_prompt}\n\n" + "\n\n".join(context_blocks)
    temperature = float(agent_cfg.get("temperature", model_cfg.get("temperature", 0.3)))
    num_ctx = int(agent_cfg.get("num_ctx", model_cfg.get("num_ctx", 16384)))
    think = bool(agent_cfg.get("think", model_cfg.get("think", False)))
    timeout = int(agent_cfg.get("timeout_sec", model_cfg.get("timeout_sec", 0)))
    retry_attempts = int(agent_cfg.get("retry_attempts", model_cfg.get("retry_attempts", 6)))
    retry_backoff_sec = float(agent_cfg.get("retry_backoff_sec", model_cfg.get("retry_backoff_sec", 1.5)))
    validation_cycles = int(agent_cfg.get("validation_cycles", model_cfg.get("validation_cycles", 3)))

    fallback_models_raw = agent_cfg.get("fallback_models", model_cfg.get("fallback_models", []))
    fallback_models: list[str] = []
    if isinstance(fallback_models_raw, list):
        for item in fallback_models_raw:
            name = str(item or "").strip()
            if name:
                fallback_models.append(name)
    if base_model and requested_model != base_model:
        fallback_models.append(base_model)

    used_model = requested_model
    finding = ""
    failure_notes: list[str] = []
    for cycle in range(max(1, validation_cycles)):
        if callable(pause_checker):
            while True:
                try:
                    paused = bool(pause_checker())
                except Exception:
                    paused = False
                if not paused:
                    break
                if callable(cancel_checker):
                    try:
                        if bool(cancel_checker()):
                            finding = f"Cancelled by user before {persona} could complete."
                            break
                    except Exception:
                        pass
                time.sleep(0.4)
            if finding.lower().startswith("cancelled by user"):
                break
        if callable(cancel_checker):
            try:
                if bool(cancel_checker()):
                    finding = f"Cancelled by user before {persona} could complete."
                    break
            except Exception:
                pass
        cycle_prompt = user_prompt
        if cycle > 0:
            cycle_prompt = (
                f"{user_prompt}\n\n"
                "Regenerate with stricter rigor. Include clear sections for Findings, Evidence Signals, and Open Questions."
            )
        try:
            finding = client.chat(
                model=requested_model,
                fallback_models=fallback_models,
                system_prompt=system_prompt,
                user_prompt=cycle_prompt,
                temperature=temperature,
                num_ctx=num_ctx,
                think=think,
                timeout=timeout,
                retry_attempts=max(1, retry_attempts),
                retry_backoff_sec=max(0.0, retry_backoff_sec),
            )
            if _looks_like_research_note(finding):
                break
            failure_notes.append(f"validation cycle {cycle + 1}: weak structure/content")
            if cycle == (max(1, validation_cycles) - 1):
                finding = (
                    f"{finding}\n\n"
                    "_Reliability note: transport retries succeeded, but the response missed structure quality checks._"
                )
        except Exception as exc:
            failure_notes.append(str(exc))
            if cycle == (max(1, validation_cycles) - 1):
                finding = f"Model call failed for {persona} after retries and fallbacks: {exc}"

    if _is_failure_text(finding) and failure_notes:
        finding = f"{finding}\n\nReliability diagnostics: {' | '.join(failure_notes[-4:])}"

    role = str(agent_cfg.get("role", "primary")).strip() or "primary"
    source_rows = [dict(x) for x in (source_evidence or []) if isinstance(x, dict)]
    source_urls = [str(x.get("url", "")).strip() for x in source_rows if str(x.get("url", "")).strip()]
    return {
        "agent": persona,
        "model": used_model,
        "requested_model": requested_model,
        "finding": finding,
        "role": role,
        "source_urls": source_urls,
        "source_evidence": source_rows,
    }


def _agent_specs(model_cfg: dict[str, Any], topic_type: str = "general") -> list[dict[str, Any]]:
    profile = _analysis_profile_for_type(topic_type)
    templates = _profile_agent_templates(profile)
    default_validation_cycles = int(model_cfg.get("validation_cycles", 3))
    if profile in {ANALYSIS_PROFILE_MEDICAL, ANALYSIS_PROFILE_FINANCE, ANALYSIS_PROFILE_UNDERGROUND}:
        default_validation_cycles = max(4, default_validation_cycles)

    base_fallbacks = _sanitize_model_list(model_cfg.get("fallback_models", []))
    out: list[dict[str, Any]] = []
    for item in templates:
        row = dict(item)
        model_name = str(row.get("model", "")).strip()
        fallback = _sanitize_model_list(list(base_fallbacks) + [model_name])
        if model_name and model_name in fallback:
            fallback = [model_name] + [x for x in fallback if x != model_name]
        row["fallback_models"] = fallback
        row.setdefault("validation_cycles", default_validation_cycles)
        # deepseek-r1 has built-in chain-of-thought reasoning activated by think=True.
        # Enable it automatically for primary deepseek-r1 agents unless explicitly overridden.
        # Advisory agents do NOT get think=True — their findings are supplementary and
        # chain-of-thought overhead isn't justified for that role.
        _role = str(row.get("role", "primary")).strip()
        if str(row.get("model", "")).startswith("deepseek-r1") and "think" not in row and _role != "advisory":
            row["think"] = True
        out.append(row)
    return out


_POSITIVE_SIGNALS = re.compile(
    r"\b(increase[sd]?|rise[sd]?|rose|rises|improve[sd]?|gain[sd]?|grow[sth]?|grew|"
    r"strengthen[sed]?|accelerate[sd]?|surge[sd]?|win[sd]?|won|higher|more|outperform[sed]?)\b",
    re.I,
)
_NEGATIVE_SIGNALS = re.compile(
    r"\b(decrease[sd]?|decline[sd]?|fall[sd]?|fell|reduce[sd]?|lower|shrink[s]?|shrunk|"
    r"weaken[sed]?|worsen[sed]?|lose[sd]?|lost|fail[sed]?|risk[s]?|harm[sed]?|threaten[sed]?)\b",
    re.I,
)


def _cross_agent_conflict_report(findings: list[dict]) -> str:
    """Heuristic cross-agent conflict detection — no LLM call.

    Compares primary-role agent findings pairwise. For each pair, splits sentences
    and looks for any sentence containing the same root noun (3+ chars, alpha) where
    one sentence has positive directional signals and the other has negative ones.
    Returns a markdown block of conflicts, or empty string if none found.
    """
    primary = [f for f in findings if str(f.get("role", "primary")).lower() != "advisory"]
    if len(primary) < 2:
        return ""

    # Extract (agent, sentence, has_pos, has_neg) rows from each finding.
    rows: list[tuple[str, str, bool, bool]] = []
    for item in primary:
        agent = str(item.get("agent", "agent"))
        text = str(item.get("finding", ""))
        for sent in re.split(r"(?<=[.!?])\s+", text):
            sent = sent.strip()
            if len(sent) < 20:
                continue
            has_pos = bool(_POSITIVE_SIGNALS.search(sent))
            has_neg = bool(_NEGATIVE_SIGNALS.search(sent))
            if has_pos or has_neg:
                rows.append((agent, sent, has_pos, has_neg))

    conflicts: list[str] = []
    seen: set[tuple[str, str]] = set()
    for i, (ag_a, sent_a, pos_a, neg_a) in enumerate(rows):
        # Extract noun tokens (4+ char alpha words, excluding common stop words).
        nouns_a = {
            w.lower() for w in re.findall(r"\b[a-zA-Z]{4,}\b", sent_a)
            if w.lower() not in {"this", "that", "with", "from", "into", "than", "their", "they", "have", "will", "been", "when", "where", "what", "which", "about", "more", "less", "also", "both", "each"}
        }
        for j, (ag_b, sent_b, pos_b, neg_b) in enumerate(rows):
            if j <= i or ag_a == ag_b:
                continue
            # Conflict: one positive, one negative, sharing a content noun.
            if not ((pos_a and neg_b) or (neg_a and pos_b)):
                continue
            nouns_b = {
                w.lower() for w in re.findall(r"\b[a-zA-Z]{4,}\b", sent_b)
                if w.lower() not in {"this", "that", "with", "from", "into", "than", "their", "they", "have", "will", "been", "when", "where", "what", "which", "about", "more", "less", "also", "both", "each"}
            }
            shared = nouns_a & nouns_b
            if len(shared) < 2:
                continue
            key = (ag_a, ag_b, tuple(sorted(shared))[:3])
            if key in seen:
                continue
            seen.add(key)
            shared_str = ", ".join(sorted(shared)[:3])
            snippet_a = sent_a[:120].rstrip()
            snippet_b = sent_b[:120].rstrip()
            conflicts.append(
                f"- **{ag_a}** (positive signals on: {shared_str}): \"{snippet_a}...\"\n"
                f"  **{ag_b}** (negative signals on: {shared_str}): \"{snippet_b}...\""
            )
            if len(conflicts) >= 5:
                break
        if len(conflicts) >= 5:
            break

    if not conflicts:
        return ""
    return "## Disputed Claims Across Agents\n" + "\n".join(conflicts)


def _reliability_summary(findings: list[dict[str, str]]) -> dict[str, int]:
    total = len(findings)
    failed = 0
    weak = 0
    for row in findings:
        text = str(row.get("finding", ""))
        if _is_failure_text(text):
            failed += 1
            continue
        if not _looks_like_research_note(text):
            weak += 1
    good = max(0, total - failed - weak)
    return {
        "agents_total": total,
        "good": good,
        "weak": weak,
        "failed": failed,
    }


def _run_fill_agents(
    *,
    client: Any,
    model_cfg: dict[str, Any],
    question: str,
    gap_queries: list[str],
    web_context: str,
    project_context: str = "",
    prior_messages: list[dict[str, str]] | None = None,
    findings: list[dict[str, Any]] | None = None,
    source_evidence: list[dict[str, str]] | None = None,
    cancel_checker: Callable[[], bool] | None = None,
    pause_checker: Callable[[], bool] | None = None,
) -> list[dict[str, Any]]:
    """Run exactly 2 targeted fill agents against identified gaps in parallel.

    Selects the two lowest-confidence primary agents from the first pass
    (falls back to technical_researcher + risk_researcher if unavailable).
    Each fill agent gets its standard directive augmented with the gap questions.
    Returns advisory-role findings, or [] if fill agents produce nothing useful.
    """
    if not gap_queries:
        return []
    findings = list(findings or [])

    gap_text = "\n".join(f"- {q}" for q in gap_queries)

    # Select the 2 lowest-confidence primary agent personas from the first pass.
    primary_findings = [f for f in findings if str(f.get("role", "primary")).strip().lower() != "advisory"]
    scored_primary = sorted(
        [f for f in primary_findings if isinstance(f.get("confidence"), (int, float))],
        key=lambda x: int(x.get("confidence", 0)),
    )

    fill_personas: list[str] = []
    for f in scored_primary[:2]:
        persona = str(f.get("agent", "")).strip()
        if persona and persona not in fill_personas:
            fill_personas.append(persona)

    # Fill to 2 with defaults if needed.
    for default in ("technical_researcher", "risk_researcher"):
        if len(fill_personas) >= 2:
            break
        if default not in fill_personas:
            fill_personas.append(default)

    fill_cfg_list: list[dict[str, Any]] = []
    for persona in fill_personas[:2]:
        base_directive = DEFAULT_DIRECTIVES.get(
            persona,
            "Focus on evidence quality, contradictions, and practical implications.",
        )
        fill_cfg_list.append({
            "persona": f"{persona}_gap_fill",
            "directive": f"{base_directive}\n\nFocus specifically on these gaps:\n{gap_text}",
            "model": str(model_cfg.get("model", "qwen3:8b")).strip(),
            "temperature": float(model_cfg.get("temperature", 0.3)),
            "num_ctx": int(model_cfg.get("num_ctx", 12288)),
            "think": False,
            "timeout_sec": 90,
            "retry_attempts": 3,
            "retry_backoff_sec": float(model_cfg.get("retry_backoff_sec", 1.5)),
            "validation_cycles": 1,
            "fallback_models": list(model_cfg.get("fallback_models") or []),
            "role": "advisory",
        })

    if not fill_cfg_list:
        return []

    fill_findings: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(fill_cfg_list)) as executor:
        futures = {
            executor.submit(
                _run_one_agent,
                client,
                model_cfg,
                cfg,
                question,
                "",  # no learned_guidance for fill pass
                web_context,
                source_evidence,
                project_context,
                prior_messages,
                cancel_checker,
                pause_checker,
            ): cfg
            for cfg in fill_cfg_list
        }
        for future in futures:
            try:
                result = future.result()
                finding_text = str(result.get("finding", "")).strip()
                if finding_text and not _is_failure_text(finding_text):
                    score = _self_check(client, model_cfg, question, finding_text)
                    result["confidence"] = score
                    result["role"] = "advisory"
                    fill_findings.append(result)
            except Exception:
                pass

    return fill_findings


def _recover_failed_findings(
    *,
    client: OllamaClient,
    findings: list[dict[str, Any]],
    model_cfg: dict[str, Any],
    question: str,
    learned_guidance: str,
    web_context: str,
    source_evidence: list[dict[str, str]] | None,
    project_context: str,
    prior_messages: list[dict[str, str]] | None,
    cancel_checker: Callable[[], bool] | None = None,
    pause_checker: Callable[[], bool] | None = None,
) -> list[dict[str, Any]]:
    recovered: list[dict[str, Any]] = []
    for row in findings:
        if callable(cancel_checker):
            try:
                if bool(cancel_checker()):
                    recovered.append(
                        {
                            "agent": "recovery",
                            "model": "",
                            "requested_model": "",
                            "finding": "Cancelled by user during recovery pass.",
                        }
                    )
                    break
            except Exception:
                pass
        text = str(row.get("finding", ""))
        if not _is_failure_text(text):
            recovered.append(row)
            continue

        persona = str(row.get("agent", "research_recovery")).strip() or "research_recovery"
        directive = DEFAULT_DIRECTIVES.get(persona, "Focus on evidence quality, contradictions, and practical implications.")
        emergency_cfg = {
            "persona": persona,
            "directive": directive,
            "model": str(model_cfg.get("model", "")).strip() or str(row.get("requested_model", "")).strip(),
            "temperature": float(model_cfg.get("temperature", 0.3)),
            "num_ctx": int(model_cfg.get("num_ctx", 16384)),
            "think": bool(model_cfg.get("think", False)),
            "timeout_sec": int(model_cfg.get("timeout_sec", 0)),
            "retry_attempts": int(model_cfg.get("retry_attempts", 6)) + 2,
            "retry_backoff_sec": float(model_cfg.get("retry_backoff_sec", 1.5)),
            "validation_cycles": int(model_cfg.get("validation_cycles", 3)),
            "fallback_models": model_cfg.get("fallback_models", []),
        }
        repaired = _run_one_agent(
            client,
            model_cfg,
            emergency_cfg,
            question,
            learned_guidance,
            web_context,
            source_evidence,
            project_context,
            prior_messages,
            cancel_checker,
            pause_checker,
        )
        recovered.append(repaired)
    return recovered


def _run_multipass_agent(
    client: OllamaClient,
    model_cfg: dict[str, Any],
    agent_cfg: dict[str, Any],
    question: str,
    learned_guidance: str,
    web_context: str,
    source_evidence: list[dict[str, str]] | None,
    project_context: str,
    prior_messages: list[dict[str, str]] | None = None,
    cancel_checker: Any = None,
    pause_checker: Any = None,
) -> dict[str, Any]:
    """Wrapper around _run_one_agent that processes sources in batches.

    When web_context contains more than _MULTI_PASS_THRESHOLD source blocks,
    splits them into batches of _MULTI_PASS_BATCH_SIZE and runs a separate
    LLM call per batch.  All partial findings are concatenated into one result.
    Falls back to a single _run_one_agent call when there are few sources.
    """
    header, source_blocks = _split_web_sources(web_context)
    if len(source_blocks) <= _MULTI_PASS_THRESHOLD:
        return _run_one_agent(
            client, model_cfg, agent_cfg, question, learned_guidance,
            web_context, source_evidence, project_context, prior_messages, cancel_checker, pause_checker,
        )

    batches = [
        source_blocks[i: i + _MULTI_PASS_BATCH_SIZE]
        for i in range(0, len(source_blocks), _MULTI_PASS_BATCH_SIZE)
    ]
    batches = batches[:4]  # cap at 4 passes to bound LLM calls

    partial_findings: list[str] = []
    last_result: dict[str, Any] = {}
    original_directive = str(agent_cfg.get("directive", "")).strip()

    for idx, batch in enumerate(batches):
        batch_context = header + "\n" + "\n".join(batch)
        batch_cfg = dict(agent_cfg)
        batch_cfg["directive"] = (
            f"{original_directive}\n"
            f"[Source scan {idx + 1} of {len(batches)}: analyse only the sources in this batch.]"
        )
        result = _run_one_agent(
            client, model_cfg, batch_cfg, question, learned_guidance,
            batch_context, source_evidence, project_context, prior_messages, cancel_checker, pause_checker,
        )
        last_result = result
        finding = str(result.get("finding", "")).strip()
        if finding and not finding.startswith("[FAILED]") and not finding.startswith("[No model"):
            partial_findings.append(f"[Scan {idx + 1}/{len(batches)}]\n{finding}")

    if not partial_findings:
        return last_result

    last_result = dict(last_result)
    last_result["finding"] = "\n\n---\n\n".join(partial_findings)
    return last_result


def run_research_pool(
    question: str,
    repo_root: Path,
    project_slug: str,
    bus,
    web_context: str = "",
    project_context: str = "",
    prior_messages: list[dict[str, str]] | None = None,
    cancel_checker: Callable[[], bool] | None = None,
    pause_checker: Callable[[], bool] | None = None,
    yield_checker: Callable[[], bool] | None = None,
    progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
    topic_type: str = "general",
) -> dict:
    bus.emit("research_pool", "start", {"question": question, "project": project_slug})

    def _is_cancelled() -> bool:
        if callable(cancel_checker):
            try:
                return bool(cancel_checker())
            except Exception:
                return False
        return False

    def _progress(stage: str, detail: dict[str, Any] | None = None) -> None:
        if not callable(progress_callback):
            return
        try:
            progress_callback(stage, detail or {})
        except Exception:
            pass

    def _is_paused() -> bool:
        if callable(pause_checker):
            try:
                return bool(pause_checker())
            except Exception:
                return False
        return False

    def _should_yield() -> bool:
        if callable(yield_checker):
            try:
                return bool(yield_checker())
            except Exception:
                return False
        return False

    model_cfg = lane_model_config(repo_root, "research_pool")
    orchestrator_cfg = lane_model_config(repo_root, "orchestrator_reasoning")
    client = InferenceRouter(repo_root)
    learning = FeedbackLearningEngine(repo_root, client=client, model_cfg=orchestrator_cfg)
    learned_guidance = learning.guidance_for_lane("research", limit=5)
    resolved_type = str(topic_type or "general").strip().lower() or "general"
    profile_name = _analysis_profile_for_type(resolved_type)
    agents = _agent_specs(model_cfg, topic_type=resolved_type)
    source_evidence = _extract_web_source_evidence(web_context)
    worker_count = max(1, min(int(model_cfg.get("parallel_agents", 4)), len(agents)))
    agent_roster = [
        {
            "persona": str(a.get("persona", "")).strip(),
            "directive": str(a.get("directive", "")).strip()[:120],
            "role": str(a.get("role", "primary")).strip(),
        }
        for a in agents
    ]
    _progress(
        "research_pool_started",
        {
            "agents_total": len(agents),
            "agents": agent_roster,
            "workers": worker_count,
            "project": project_slug,
            "topic_type": resolved_type,
            "analysis_profile": profile_name,
        },
    )

    if _is_cancelled():
        summary_path = ""
        if question.strip():
            store = ProjectStore(repo_root)
            summary_name = store.timestamped_name("research_summary")
            summary_md = (
                "# Research Synthesis (Cancelled)\n\n"
                f"Question: {question}\n\n"
                "Request was cancelled before worker execution.\n"
            )
            summary_path = str(store.write_project_file(project_slug, "research_summaries", summary_name, summary_md))
        cancel_summary = (
            "Request cancelled before Foraging worker execution started.\n"
            + (f"Summary written to:\n{summary_path}" if summary_path else "No summary file was written.")
        )
        return {
            "message": "Research cancelled before execution.",
            "summary_path": summary_path,
            "web_context_used": bool(web_context.strip()),
            "reliability": {"agents_total": len(agents), "good": 0, "weak": 0, "failed": 0},
            "canceled": True,
            "cancel_summary": cancel_summary,
        }

    findings: list[dict[str, Any]] = []
    canceled = False
    executor = ThreadPoolExecutor(max_workers=worker_count)
    pending: set[Any] = set()
    future_agent: dict[Any, str] = {}
    # Sort agents by model name so same-model agents run consecutively.
    # This keeps each model warm in VRAM across back-to-back calls,
    # reducing Ollama load/evict churn within the pool.
    queue = sorted(agents, key=lambda a: str(a.get("model", "")))
    try:
        while queue or pending:
            if _is_cancelled():
                canceled = True
                _progress(
                    "research_cancel_requested",
                    {"completed": len(findings), "total": len(agents)},
                )
                break
            if _is_paused():
                _progress(
                    "foraging_paused",
                    {"completed": len(findings), "total": len(agents), "active_workers": len(pending)},
                )
                time.sleep(0.5)
                continue

            desired_workers = 1 if _should_yield() else worker_count
            while queue and len(pending) < desired_workers:
                agent_cfg = queue.pop(0)
                future = executor.submit(
                    _run_multipass_agent,
                    client,
                    model_cfg,
                    agent_cfg,
                    question,
                    learned_guidance,
                    web_context,
                    source_evidence,
                    project_context,
                    prior_messages,
                    cancel_checker,
                    pause_checker,
                )
                pending.add(future)
                persona = str(agent_cfg.get("persona", "research_agent")).strip() or "research_agent"
                future_agent[future] = persona
                _progress(
                    "research_agent_started",
                    {
                        "agent": persona,
                        "directive": str(agent_cfg.get("directive", "")).strip()[:120],
                        "role": str(agent_cfg.get("role", "primary")).strip(),
                        "model": str(agent_cfg.get("model", "")).strip(),
                        "completed": len(findings),
                        "total": len(agents),
                        "active_workers": len(pending),
                        "yield_mode": bool(_should_yield()),
                    },
                )

            if not pending:
                time.sleep(0.15)
                continue

            done, pending = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)
            if not done:
                continue
            for future in done:
                persona = future_agent.pop(future, "research_agent")
                try:
                    result = future.result()
                except Exception as exc:  # pragma: no cover - defensive
                    result = {
                        "agent": persona,
                        "model": "",
                        "requested_model": "",
                        "finding": f"Model call failed for {persona}: {exc}",
                    }
                findings.append(result)
                _finding_text = str(result.get("finding", "")).strip()
                _finding_failed = _is_failure_text(_finding_text)
                _confidence = 0 if _finding_failed else _self_check(client, model_cfg, question, _finding_text)
                result["confidence"] = _confidence
                _progress(
                    "research_agent_completed",
                    {
                        "completed": len(findings),
                        "total": len(agents),
                        "agent": str(result.get("agent", "")),
                        "role": str(result.get("role", "primary")),
                        "failed": _finding_failed,
                        "finding_preview": _finding_text[:400] if not _finding_failed else "",
                        "confidence": _confidence,
                    },
                )
    finally:
        if canceled:
            for future in pending:
                future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
        else:
            executor.shutdown(wait=True, cancel_futures=False)

    pre_reliability = _reliability_summary(findings)
    if not canceled and pre_reliability.get("failed", 0) > 0:
        findings = _recover_failed_findings(
            client=client,
            findings=findings,
            model_cfg=model_cfg,
            question=question,
            learned_guidance=learned_guidance,
            web_context=web_context,
            source_evidence=source_evidence,
            project_context=project_context,
            prior_messages=prior_messages,
            cancel_checker=cancel_checker,
            pause_checker=pause_checker,
        )
    reliability = _reliability_summary(findings)
    findings = _audit_evidence_labels(findings)

    store = ProjectStore(repo_root)

    raw_name = store.timestamped_name("research_raw")
    raw_sections: list[str] = []
    for item in findings:
        persona = str(item.get("agent", "")).strip() or "research_agent"
        used_model = str(item.get("model", "")).strip()
        requested = str(item.get("requested_model", "")).strip()
        if used_model and requested and used_model != requested:
            title = f"## {persona} (model: {used_model}; requested: {requested})"
        elif used_model:
            title = f"## {persona} (model: {used_model})"
        else:
            title = f"## {persona}"
        raw_sections.append(f"{title}\n{item.get('finding', '')}")
    raw_body = "\n\n".join(raw_sections)
    raw_path = store.write_project_file(project_slug, "research_raw", raw_name, f"# Raw Research Notes\n\n{raw_body}\n")
    _progress(
        "research_raw_written",
        {"raw_path": str(raw_path), "findings_collected": len(findings), "canceled": canceled},
    )

    if canceled:
        summary_name = store.timestamped_name("research_summary")
        partial = synthesize(question, findings, client=None, model_cfg=None)
        cancel_md = (
            "# Research Synthesis (Cancelled)\n\n"
            f"Question: {question}\n\n"
            "The request was cancelled by the user. This is a partial synthesis from completed workers only.\n\n"
            f"Completed worker findings: {len(findings)} / {len(agents)}\n\n"
            f"{partial}\n"
        )
        summary_path = store.write_project_file(project_slug, "research_summaries", summary_name, cancel_md)
        cancel_summary = (
            "Request cancelled during Foraging.\n"
            f"- completed_workers: {len(findings)} / {len(agents)}\n"
            f"- partial_raw_notes: {raw_path}\n"
            f"- partial_summary: {summary_path}"
        )
        bus.emit(
            "research_pool",
            "cancelled",
            {
                "project": project_slug,
                "raw_path": str(raw_path),
                "summary_path": str(summary_path),
                "completed_workers": len(findings),
                "agents_total": len(agents),
            },
        )
        _progress(
            "research_cancelled",
            {
                "summary_path": str(summary_path),
                "raw_path": str(raw_path),
                "completed_workers": len(findings),
                "agents_total": len(agents),
            },
        )
        return {
            "message": "Research cancelled and partial synthesis written for review.",
            "summary_path": str(summary_path),
            "raw_path": str(raw_path),
            "web_context_used": bool(web_context.strip()),
            "reliability": reliability,
            "canceled": True,
            "cancel_summary": cancel_summary,
        }

    _synthesis_lane = lane_model_config(repo_root, "synthesis") or {}
    synth_cfg = dict(_synthesis_lane or orchestrator_cfg or {})
    synth_cfg.setdefault("synthesis_timeout_sec", int(_synthesis_lane.get("timeout_sec", int(model_cfg.get("timeout_sec", 0)))))
    synth_cfg.setdefault("synthesis_retry_attempts", int(model_cfg.get("retry_attempts", 6)))
    synth_cfg.setdefault("synthesis_retry_backoff_sec", float(model_cfg.get("retry_backoff_sec", 1.5)))
    synth_cfg.setdefault("synthesis_validation_cycles", int(model_cfg.get("validation_cycles", 3)))
    fb = list(model_cfg.get("fallback_models", [])) if isinstance(model_cfg.get("fallback_models", []), list) else []
    main_model = str(model_cfg.get("model", "")).strip()
    if main_model:
        fb.append(main_model)
    synth_cfg.setdefault("synthesis_fallback_models", fb)

    # Underground topics: force abliterated model for synthesis — no filtered models in the pipeline.
    if str(topic_type).strip().lower() == "underground":
        synth_cfg["model"] = "huihui_ai/qwen3-abliterated:8b-Q4_K_M"
        synth_cfg["synthesis_fallback_models"] = ["huihui_ai/qwen3-abliterated:8b-Q4_K_M"]

    summary_name = store.timestamped_name("research_summary")
    conflict_report = _cross_agent_conflict_report(findings)
    summary_md = synthesize(
        question,
        findings,
        client=client,
        model_cfg=synth_cfg,
        project_context=project_context,
        prior_messages=prior_messages,
        conflict_report=conflict_report,
    )

    skeptic_md = run_skeptic_pass(question, summary_md, client=client, model_cfg=synth_cfg, findings=findings)
    if skeptic_md:
        summary_md = f"{summary_md}\n\n---\n\n{skeptic_md}"

    _all_scores = [f.get("confidence") for f in findings]
    _scored = [int(s) for s in _all_scores if isinstance(s, (int, float)) and int(s) > 0]
    _unscored_count = sum(1 for s in _all_scores if not isinstance(s, (int, float)) or int(s) == 0)
    if _scored:
        _avg_conf = sum(_scored) / len(_scored)
        _conf_labels = {1: "very low", 2: "low", 3: "medium", 4: "high", 5: "very high"}
        _agent_conf_lines = "\n".join(
            f"- {str(f.get('agent', 'agent'))}: {int(f.get('confidence', 0))}/5"
            + (" (unscored)" if int(f.get("confidence", 0)) == 0 else "")
            for f in findings
        )
        _unscored_note = f" | {_unscored_count} agent(s) unscored" if _unscored_count else ""
        summary_md = (
            f"{summary_md}\n\n---\n\n"
            f"**Source Quality** — avg confidence {_avg_conf:.1f}/5 "
            f"({_conf_labels.get(round(_avg_conf), 'unknown')})"
            f" | scored: {len(_scored)}/{len(findings)}{_unscored_note}\n\n"
            f"{_agent_conf_lines}"
        )
    elif _unscored_count:
        summary_md = (
            f"{summary_md}\n\n---\n\n"
            f"**Source Quality** — agent self-scoring unavailable "
            f"({_unscored_count} agent(s) did not return a score)"
        )

    summary_path = store.write_project_file(project_slug, "research_summaries", summary_name, summary_md)
    _progress(
        "research_summary_written",
        {"summary_path": str(summary_path), "findings_collected": len(findings)},
    )

    # Release Ollama-hosted models from VRAM now that the full pipeline is done.
    # Models routed through llama.cpp are managed by that server process and skipped.
    _release_models = sorted({
        str(f.get("model", "")).strip()
        for f in findings
        if str(f.get("model", "")).strip()
    } | {str(model_cfg.get("model", "")).strip(), str(synth_cfg.get("model", "")).strip()})
    client.release_models([m for m in _release_models if m])

    bus.emit(
        "research_pool",
        "completed",
        {
            "project": project_slug,
            "raw_path": str(raw_path),
                "summary_path": str(summary_path),
                "model": model_cfg.get("model", ""),
                "workers": worker_count,
                "agents_total": len(agents),
                "models_used": sorted({str(x.get("model", "")).strip() for x in findings if str(x.get("model", "")).strip()}),
                "web_context_used": bool(web_context.strip()),
                "reliability": reliability,
                "analysis_profile": profile_name,
                "topic_type": resolved_type,
            },
        )

    return {
        "message": (
            "Foraging council completed a synthesis for orchestrator review. "
            f"Reliability: good={reliability.get('good', 0)}, "
            f"weak={reliability.get('weak', 0)}, failed={reliability.get('failed', 0)}."
        ),
        "summary_path": str(summary_path),
        "raw_path": str(raw_path),
        "web_context_used": bool(web_context.strip()),
        "reliability": reliability,
        "analysis_profile": profile_name,
        "topic_type": resolved_type,
        "findings": findings,
    }
