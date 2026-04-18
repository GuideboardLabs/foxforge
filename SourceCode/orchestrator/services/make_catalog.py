"""Canonical Make type registry.

Single source of truth for all Make output types. Every type_id maps to
display metadata, process notes (fed into pool prompts as expert guardrails),
the routing lane, output destination folder, and the model_routing.json lane key.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Catalog definition
# ---------------------------------------------------------------------------
# fmt: off
MAKE_CATALOG: dict[str, dict[str, Any]] = {

    # ------------------------------------------------------------------ Code / Build
    "tool": {
        "label": "Tool / Script",
        "category": "code",
        "short_description": "Shell or single-file Python script with smoke test.",
        "process_note": (
            "Determine invocation shape → argparse/getopts skeleton → core logic → "
            "error modes and exit codes → print/exit discipline → smoke-test block "
            "under `if __name__ == '__main__':`. Keep it a single file; avoid external "
            "deps unless unavoidable. Shell scripts use set -euo pipefail."
        ),
        "lane": "make_tool",
        "destination": "tools",
        "model_lane": "make_tool",
    },
    "web_app": {
        "label": "Web App",
        "category": "code",
        "short_description": "Vue 3 + Flask + SQLite full-stack app in its own git-style project folder.",
        "process_note": (
            "Design the SQLite schema first → generate Flask routes with CRUD endpoints → "
            "build Vue 3 reactive frontend (CDN, no build step) → integration-check "
            "route/fetch mismatches → generate CSS from selectors → write README with "
            "setup + run instructions. Output goes in backend/ + frontend/ subfolders "
            "with a .gitignore and CHANGELOG.md."
        ),
        "lane": "make_app",
        "destination": "web_apps",
        "model_lane": "make_app",
    },
    "desktop_app": {
        "label": "Desktop App",
        "category": "code",
        "short_description": ".NET 8 + Avalonia UI desktop application (Windows-first, Linux-portable).",
        "process_note": (
            "Plan feature set and state model → scaffold `dotnet new avalonia.mvvm` "
            "project layout → implement ViewModels using ReactiveUI for non-trivial state → "
            "implement AXAML Views → add Services/data layer → validate with `dotnet build` → "
            "write README with Windows install steps and Linux port notes. "
            "MVVM: View ↔ ViewModel ↔ Model ↔ Service. No code-behind logic."
        ),
        "lane": "make_desktop_app",
        "destination": "desktop_apps",
        "model_lane": "make_desktop_app",
    },

    # ------------------------------------------------------------------ Writing — short-form
    "social_post": {
        "label": "Social Post",
        "category": "writing",
        "short_description": "Single viral LinkedIn/X/Bluesky post (80–220 words).",
        "process_note": (
            "Hook in the first line — the 'stop-scrolling' line. "
            "Context in ≤2 sentences. Payoff/insight. Optional CTA. "
            "No hashtag soup. Platform-specific rhythm: LinkedIn likes paragraph breaks, "
            "X rewards punchy single shots, Bluesky tolerates medium length."
        ),
        "lane": "make_content",
        "destination": "Content",
        "model_lane": "make_content",
    },
    "email": {
        "label": "Email",
        "category": "writing",
        "short_description": "Professional email (200–400 words).",
        "process_note": (
            "Subject line: clear, specific, under 60 chars. "
            "Front-load the key ask. Body in 2–3 short paragraphs, most important first. "
            "Professional sign-off with a clear next step."
        ),
        "lane": "make_content",
        "destination": "Content",
        "model_lane": "make_content",
    },
    "blog": {
        "label": "Blog Post",
        "category": "writing",
        "short_description": "Standard blog post (600–800 words), conversational and authoritative.",
        "process_note": (
            "Hook & headline → context/why now → core content with subheadings and "
            "concrete examples → takeaway and CTA. Conversational, authoritative, "
            "accessible. Write like you're explaining to a smart friend."
        ),
        "lane": "make_content",
        "destination": "Content",
        "model_lane": "make_content",
    },
    "essay_short": {
        "label": "Short-Form Essay",
        "category": "writing",
        "short_description": "Substack/X-style short essay or thread seed (400–900 words).",
        "process_note": (
            "One strong thesis, stated early. Two or three argument beats with concrete "
            "detail. Steelman one counterpoint. Strong close that earns the space. "
            "Voice is personal and direct. Designed to be read in under 4 minutes."
        ),
        "lane": "make_longform",
        "destination": "Essays-Scripts",
        "model_lane": "make_longform",
    },

    # ------------------------------------------------------------------ Writing — long-form
    "essay_long": {
        "label": "Long-Form Essay",
        "category": "writing",
        "short_description": "Substack-style long-form piece (1800–3500 words).",
        "process_note": (
            "Thesis → 3–5 argument pillars each with evidence → steelman counterargument → "
            "synthesis → so-what → memorable close. Uses narrative hook in lede. "
            "Each section earns the next. Cut anything that slows the argument."
        ),
        "lane": "make_longform",
        "destination": "Essays-Scripts",
        "model_lane": "make_longform",
    },
    "guide": {
        "label": "Guide",
        "category": "writing",
        "short_description": "Teaching/how-to MD guide that converts cleanly to Word.",
        "process_note": (
            "Prereqs callout → 'what you'll learn' → steps with exact commands or "
            "clear actions → common pitfalls block per step → verification step → "
            "next steps. Assume reader skims first: H2/H3 structure + callout blocks. "
            "Writes well in markdown; tables and callouts convert cleanly to Word via Pandoc."
        ),
        "lane": "make_longform",
        "destination": "Essays-Scripts",
        "model_lane": "make_longform",
    },
    "tutorial": {
        "label": "Tutorial",
        "category": "writing",
        "short_description": "Step-by-step technical walkthrough, code-heavy.",
        "process_note": (
            "Goal statement → prereqs → numbered steps with runnable code blocks → "
            "'you should see' verification after each major step → troubleshooting block → "
            "next steps. Code blocks are copy-paste ready. "
            "Distinct from Guide: teaches a specific task rather than a concept."
        ),
        "lane": "make_longform",
        "destination": "Essays-Scripts",
        "model_lane": "make_longform",
    },
    "video_script": {
        "label": "Video Script",
        "category": "writing",
        "short_description": "Read-aloud video essay script with [CUT]/[SEGMENT]/[B-ROLL] markers.",
        "process_note": (
            "Hook (0–15s) → premise (15–45s) → body in 3–5 beats with visual pacing notes → "
            "turn/reveal → close with CTA. Every sentence is read aloud — prefer short "
            "sentences, conversational rhythm, avoid words that trip on the tongue. "
            "Use [SEGMENT: title] markers for chapter cuts, [B-ROLL: description] for "
            "visual suggestions. No image generation — just structural suggestions."
        ),
        "lane": "make_longform",
        "destination": "Essays-Scripts",
        "model_lane": "make_longform",
    },
    "newsletter": {
        "label": "Newsletter",
        "category": "writing",
        "short_description": "Periodical digest with standing sections: This Week / Worth Your Time / One Idea / Dessert.",
        "process_note": (
            "Standing sections: This Week (news + takes) / Worth Your Time (curated links + why) / "
            "One Idea (the core insight this edition) / Dessert (lighter closer). "
            "Voice consistent across editions. Hook in the first two lines of the email preview. "
            "Each section self-contained — readers pick and choose."
        ),
        "lane": "make_longform",
        "destination": "Essays-Scripts",
        "model_lane": "make_longform",
    },
    "press_release": {
        "label": "Press Release",
        "category": "writing",
        "short_description": "Short marketing-style announcement or launch pitch document.",
        "process_note": (
            "Headline → dateline → lede (who/what/when/where/why in two sentences) → "
            "2–3 body paragraphs with quotes from stakeholders (use placeholders if real "
            "names not provided) → boilerplate about the org → contact info. "
            "Inverted pyramid: most important first."
        ),
        "lane": "make_longform",
        "destination": "Essays-Scripts",
        "model_lane": "make_longform",
    },

    # ------------------------------------------------------------------ Narrative
    "novel_chapter": {
        "label": "Novel Chapter",
        "category": "narrative",
        "short_description": "Literary fiction chapter with scene structure and dialogue.",
        "process_note": (
            "Scene headers → dialogue → interior monologue → chapter hook at close. "
            "Consistent POV and tense. Each scene has an entry point, a turn, and an exit."
        ),
        "lane": "make_creative",
        "destination": "Creative",
        "model_lane": "make_creative",
    },
    "memoir_chapter": {
        "label": "Memoir Chapter",
        "category": "narrative",
        "short_description": "First-person memoir with reflective passages and temporal anchoring.",
        "process_note": (
            "First-person voice. Reflective passages anchor the past to present insight. "
            "Specific sensory detail. Temporal grounding (age, year, place)."
        ),
        "lane": "make_creative",
        "destination": "Creative",
        "model_lane": "make_creative",
    },
    "book_chapter": {
        "label": "Book Chapter",
        "category": "narrative",
        "short_description": "Thesis-driven non-fiction chapter with evidence integration.",
        "process_note": (
            "Thesis-driven chapters. Evidence integrated (not dumped). "
            "Chapter opens with a claim, closes with synthesis. "
            "Transitions carry the argument forward."
        ),
        "lane": "make_creative",
        "destination": "Creative",
        "model_lane": "make_creative",
    },
    "screenplay": {
        "label": "Screenplay",
        "category": "narrative",
        "short_description": "Industry-standard screenplay format: INT./EXT. headings, action lines, character cues.",
        "process_note": (
            "INT./EXT. headings. Action lines in present tense, visual only — no internal states. "
            "Character cues centered. Dialogue punchy. Scene transitions where needed."
        ),
        "lane": "make_creative",
        "destination": "Creative",
        "model_lane": "make_creative",
    },

    # ------------------------------------------------------------------ Domain expert
    "medical": {
        "label": "Medical Report",
        "category": "domain",
        "short_description": "Evidence-graded clinical summary with required disclaimers.",
        "process_note": (
            "Clinical summary → evidence review (RCT > observational > case study > expert opinion) → "
            "risk/safety → clinical guidelines → patient considerations → limitations. "
            "Must include 'not medical advice' disclaimer. No diagnostic language."
        ),
        "lane": "make_specialist",
        "destination": "Medical-Writing",
        "model_lane": "make_specialist",
    },
    "finance": {
        "label": "Financial Analysis",
        "category": "domain",
        "short_description": "Executive-level financial analysis with explicit risk disclosures.",
        "process_note": (
            "Executive summary → market context → core findings → risk factors → "
            "thesis/recommendation → caveats/disclosures. "
            "Must include 'not financial advice' disclaimer. "
            "State assumptions explicitly. Note data dates."
        ),
        "lane": "make_specialist",
        "destination": "Financial-Analysis",
        "model_lane": "make_specialist",
    },
    "sports": {
        "label": "Sports Analysis",
        "category": "domain",
        "short_description": "Statistical sports analysis with recency and sample-size caveats.",
        "process_note": (
            "Intro/stakes → context/current form → statistical analysis → "
            "risk/uncertainty → analysis/outlook → conclusion. "
            "Specific statistics. Recency noted. Injury/roster freshness caveats. "
            "Sample size and context flagged."
        ),
        "lane": "make_specialist",
        "destination": "Sports-Analysis",
        "model_lane": "make_specialist",
    },
    "history": {
        "label": "Historical Essay",
        "category": "domain",
        "short_description": "Historically grounded essay with source quality and historiographical balance.",
        "process_note": (
            "Intro/thesis → historical background → key events/turning points → "
            "key actors → historiographical debate → conclusion/legacy. "
            "Source quality noted (primary/secondary). No anachronisms. "
            "Specific dates and actors."
        ),
        "lane": "make_specialist",
        "destination": "Historical-Writing",
        "model_lane": "make_specialist",
    },
    "game_design_doc": {
        "label": "Game Design Doc",
        "category": "domain",
        "short_description": "GDD with core loop, systems design, content/narrative, and production notes.",
        "process_note": (
            "Game overview → core loop (what the player does every 30 seconds) → "
            "systems design (how mechanics interlock) → content/narrative → "
            "technical constraints → production notes (MVP vs. full vision). "
            "Scope must be realistic."
        ),
        "lane": "make_specialist",
        "destination": "Game-Design",
        "model_lane": "make_specialist",
    },
}
# fmt: on

# ---------------------------------------------------------------------------
# Lane → type_id lookup sets (for fast routing in main.py)
# ---------------------------------------------------------------------------

def lane_for_type(type_id: str) -> str:
    """Return the routing lane for a Make type_id."""
    entry = MAKE_CATALOG.get(str(type_id or "").strip().lower())
    if entry:
        return str(entry["lane"])
    return "make_doc"


def destination_for_type(type_id: str) -> str:
    """Return the output folder name for a Make type_id."""
    entry = MAKE_CATALOG.get(str(type_id or "").strip().lower())
    if entry:
        return str(entry["destination"])
    return "Essays-Scripts"


def label_for_type(type_id: str) -> str:
    """Return the human-readable label for a Make type_id."""
    entry = MAKE_CATALOG.get(str(type_id or "").strip().lower())
    if entry:
        return str(entry["label"])
    return type_id.replace("_", " ").title()


def catalog_for_api() -> list[dict[str, Any]]:
    """Serialise the catalog to a list suitable for the /api/make/catalog endpoint."""
    rows: list[dict[str, Any]] = []
    for type_id, entry in MAKE_CATALOG.items():
        rows.append({
            "type_id": type_id,
            "label": entry["label"],
            "category": entry["category"],
            "short_description": entry["short_description"],
            "lane": entry["lane"],
            "destination": entry["destination"],
        })
    return rows


# Reverse maps from lane → frozenset of type_ids (for main.py routing compatibility)
_LANE_TYPES: dict[str, frozenset[str]] = {}
for _tid, _entry in MAKE_CATALOG.items():
    _lane = _entry["lane"]
    _LANE_TYPES.setdefault(_lane, set()).add(_tid)
LANE_TYPES: dict[str, frozenset[str]] = {k: frozenset(v) for k, v in _LANE_TYPES.items()}
