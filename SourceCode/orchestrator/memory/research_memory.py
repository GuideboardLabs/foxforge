"""Research context readers — load summaries, raw notes, and sources from project directories."""

from __future__ import annotations

from pathlib import Path


def latest_research_summary_preview(
    repo_root: Path, project_slug: str, limit_chars: int = 7000
) -> tuple[str, str]:
    """Return (path_str, preview_text) for the newest research summary, or ('', '') if none."""
    root = repo_root / "Projects" / project_slug / "research_summaries"
    if not root.exists():
        return "", ""
    candidates = sorted(root.glob("*.md"))
    if not candidates:
        return "", ""
    newest = candidates[-1]
    try:
        preview = newest.read_text(encoding="utf-8").strip()[:limit_chars]
    except OSError:
        preview = ""
    return str(newest), preview


def read_research_context(
    repo_root: Path,
    project_slug: str,
    max_summaries: int = 3,
    chars_per_summary: int = 6000,
) -> str:
    """Read the last N research summaries and combine them into one context block."""
    root = repo_root / "Projects" / project_slug / "research_summaries"
    if not root.exists():
        return ""
    candidates = sorted(root.glob("*.md"))
    if not candidates:
        return ""
    selected = candidates[-max_summaries:]
    parts: list[str] = []
    for path in selected:
        try:
            text = path.read_text(encoding="utf-8").strip()
            if text:
                parts.append(f"--- [{path.name}] ---\n{text[:chars_per_summary]}")
        except OSError:
            continue
    return "\n\n".join(parts)


def read_raw_notes_context(
    repo_root: Path,
    project_slug: str,
    max_files: int = 2,
    chars_per_file: int = 4000,
) -> str:
    """Read the last N raw research note files (per-agent findings with [E]/[I]/[S] labels)."""
    root = repo_root / "Projects" / project_slug / "research_raw"
    if not root.exists():
        return ""
    candidates = sorted(root.glob("*.md"))
    if not candidates:
        return ""
    selected = candidates[-max_files:]
    parts: list[str] = []
    for path in selected:
        try:
            text = path.read_text(encoding="utf-8").strip()
            if text:
                parts.append(f"--- [raw: {path.name}] ---\n{text[:chars_per_file]}")
        except OSError:
            continue
    return "\n\n".join(parts)


def read_sources_context(web_engine, project_slug: str, limit: int = 14) -> str:
    """Read recent web sources (URLs, tier ratings, snippets) for the project."""
    try:
        return web_engine.web_context_for_project(project_slug, limit=limit)
    except Exception:
        return ""
