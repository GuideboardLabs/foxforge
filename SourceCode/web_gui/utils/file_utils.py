"""File and path security utilities for the web GUI."""

from __future__ import annotations

import mimetypes
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "SourceCode"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))


def safe_path_in_roots(
    raw_path: str,
    *,
    allowed_roots: list[Path],
    denied_roots: list[Path] | None = None,
    must_exist: bool = True,
) -> Path | None:
    """Resolve a path and verify it falls within allowed_roots.

    Returns the resolved path if safe, None if the path escapes allowed roots or is denied.
    """
    candidate = Path(raw_path.strip())
    if not candidate.is_absolute():
        candidate = ROOT / candidate

    try:
        resolved = candidate.resolve(strict=must_exist)
    except (FileNotFoundError, OSError):
        return None

    for denied in denied_roots or []:
        try:
            resolved.relative_to(denied.resolve())
            return None
        except (ValueError, OSError):
            continue

    for root in allowed_roots:
        try:
            resolved.relative_to(root.resolve())
            return resolved
        except (ValueError, OSError):
            continue
    return None


def safe_markdown_path(
    raw_path: str,
    *,
    allowed_roots: list[Path],
    denied_roots: list[Path] | None = None,
) -> Path | None:
    """Return a safe path only if it is a .md file within allowed roots."""
    resolved = safe_path_in_roots(
        raw_path, allowed_roots=allowed_roots, denied_roots=denied_roots, must_exist=True
    )
    if resolved is None or resolved.suffix.lower() != ".md":
        return None
    return resolved


def normalize_project_slug(raw: str | None, default: str = "general") -> str:
    text = str(raw or "").strip()
    cleaned = "_".join(text.split()).lower()
    return cleaned or default


def safe_upload_name(raw: str) -> str:
    text = str(raw or "").strip()
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._")
    if not cleaned:
        return "image"
    return cleaned[:120]


def guess_mime_from_ext(ext: str) -> str:
    low = str(ext or "").strip().lower()
    if low in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if low == ".png":
        return "image/png"
    if low == ".webp":
        return "image/webp"
    if low == ".gif":
        return "image/gif"
    if low == ".bmp":
        return "image/bmp"
    guessed, _ = mimetypes.guess_type(f"file{low}")
    return str(guessed or "application/octet-stream")


def read_text_file_preview(path: Path, *, max_bytes: int = 250_000) -> tuple[str, bool, bool]:
    """Read a text file preview.

    Returns:
        (text, truncated, is_binary)
    """
    try:
        raw = path.read_bytes()
    except OSError:
        return "", False, False
    truncated = len(raw) > max_bytes
    chunk = raw[:max_bytes]
    if b"\x00" in chunk:
        return "", truncated, True
    try:
        text = chunk.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = chunk.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = chunk.decode("utf-8", errors="replace")
    return text, truncated, False
