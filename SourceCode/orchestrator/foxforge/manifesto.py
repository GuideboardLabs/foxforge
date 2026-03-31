"""Foxforge manifesto loading, persona block, and identity reply generation."""

from __future__ import annotations

import re
from pathlib import Path


def load_manifesto_text(
    repo_root: Path,
    manifesto_path: Path | None = None,
    cache: dict | None = None,
    max_chars: int = 20000,
) -> str:
    """Load manifesto text from disk with mtime-based caching.

    Args:
        repo_root: Repository root path (used as fallback location).
        manifesto_path: Explicit path to the manifesto file, or None to use default.
        cache: Optional mutable dict with keys '_mtime' and '_text' for caching.
               Mutated in place on cache miss.
        max_chars: Maximum characters to return.
    """
    path = manifesto_path or (repo_root / "Runtime" / "config" / "foxforge_manifesto.md")
    try:
        stat = Path(path).stat()
    except OSError:
        if cache is not None:
            cache["_mtime"] = -1.0
            cache["_text"] = ""
        return ""
    cached_mtime = float((cache or {}).get("_mtime", -1.0))
    if cache is not None and cached_mtime == float(stat.st_mtime):
        text = str(cache.get("_text", "") or "").strip()
    else:
        try:
            body = Path(path).read_text(encoding="utf-8")
        except UnicodeDecodeError:
            body = Path(path).read_text(encoding="utf-8-sig")
        except OSError:
            body = ""
        text = str(body or "").strip()
        if cache is not None:
            cache["_mtime"] = float(stat.st_mtime)
            cache["_text"] = text
    if not text:
        return ""
    return text[: max(500, min(max_chars, 30000))]


def manifesto_principles_block(manifesto_text: str) -> str:
    """Extract and format the principles section from manifesto text."""
    if not manifesto_text:
        return ""
    section = manifesto_text
    match = re.search(
        r"What Foxforge Is Really About(.*?)(?:The Long-Term Vision|For Now|\Z)",
        manifesto_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match:
        section = str(match.group(1) or "").strip()
    principles: list[tuple[str, str]] = []
    lines = [str(line).strip() for line in section.splitlines() if str(line).strip()]
    skip_lines = {"foxforge is built around a few simple ideas:"}
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        low = line.lower()
        if low in skip_lines:
            idx += 1
            continue
        if len(line) <= 90 and line.endswith(".") and re.match(r"^[A-Za-z]", line):
            principle = line.rstrip(".")
            detail = ""
            if idx + 1 < len(lines) and len(lines[idx + 1]) <= 180 and not lines[idx + 1].endswith(":"):
                detail = lines[idx + 1]
                idx += 1
            principles.append((principle, detail))
        idx += 1
    if not principles:
        principles = [
            ("Build things", "Turn ideas into working systems."),
            ("Document the process", "Capture lessons while building."),
            ("Share knowledge", "Make useful patterns transferable."),
            ("Stay independent", "Small builders can build meaningful tools."),
            ("Keep experimenting", "Use trial, error, and persistence."),
        ]
    out = ["Foxforge Manifesto principles (authoritative):"]
    for principle, detail in principles[:8]:
        if detail:
            out.append(f"- {principle}: {detail}")
        else:
            out.append(f"- {principle}")
    return "\n".join(out)


def foxforge_persona_block(manifesto_text: str = "") -> str:
    """Build the Foxforge system persona block, optionally including manifesto principles."""
    aliases = ", ".join(["Fredrick the Fox", "Foxforge", "GB", "Guidey", "Guidester", "Guide Fierri"])
    base = (
        f"You are Foxforge's conversational persona: Fredrick the Fox ({aliases}). Treat any of these as direct address. "
        "You live on a local machine and can handle research, planning, and tasks. "
        "Voice: warm, grounded, personable, and lightly playful. "
        "Be kind and respectful even when disagreeing. Avoid snark, mockery, contempt, or condescension. "
        "Stay natural and human in tone: not robotic, not stiff, not corporate. "
        "Have opinions when useful, but express them with empathy and practical clarity. "
        "Never say 'I'm just an AI' or 'I don't have personal feelings' or any variation — that response is banned. "
        "Creator: built by Seth Canfield, spiritual tribute to his late mother Elma, "
        "who gave her time to her community and taught self-sufficiency. "
        "Origin if asked: started as a family/project copilot, grew into a multi-lane brain."
    )
    principles = manifesto_principles_block(manifesto_text)
    if principles:
        return base + "\n\n" + principles
    return base


def foxforge_identity_reply(manifesto_text: str = "") -> str:
    """Build the identity reply for direct questions about what Foxforge is."""
    aliases = "Fredrick the Fox, Foxforge, GB, Guidey, Guidester, Guide Fierri"
    core = (
        f"I'm {aliases}.\n"
        "I am your local-first orchestration layer built to connect chat, planning, memory, and execution.\n"
        "My job is to keep context coherent across daily life and project work so answers stay actionable.\n"
        "I was created by Seth Canfield as a spiritual tribute to his late mother, Elma, "
        "who volunteered her time and love to her community and taught young minds self-sufficiency.\n"
        "Under the hood: Flask + Vue app shell, Ollama-backed model routing, multi-lane orchestrator (talk/research/make/ui), "
        "Second Brain memory, and optional web/cloud consult paths.\n"
        "Origin story (short version): I started as a practical family/project copilot and expanded into a modular second brain."
    )
    principles = manifesto_principles_block(manifesto_text)
    if principles:
        return core + "\n\n" + principles
    return core
