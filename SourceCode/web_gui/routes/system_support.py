from __future__ import annotations

from pathlib import Path


_STATIC_DIR = Path(__file__).parent.parent / "static"


def asset_versions() -> tuple[int, int]:
    try:
        css_v = int((_STATIC_DIR / "styles.css").stat().st_mtime)
        js_v = int((_STATIC_DIR / "app.js").stat().st_mtime)
        asset_v = max(css_v, js_v)
        vendor_v = int((_STATIC_DIR / "vendor" / "vue.global.prod.js").stat().st_mtime)
    except OSError:
        asset_v = 0
        vendor_v = 0
    return asset_v, vendor_v


def guess_mime_from_ext(ext: str) -> str:
    mime_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".svg": "image/svg+xml",
        ".avif": "image/avif",
        ".mp4": "video/mp4",
        ".webm": "video/webm",
        ".pdf": "application/pdf",
        ".json": "application/json",
        ".md": "text/markdown",
        ".txt": "text/plain",
        ".csv": "text/csv",
    }
    return mime_map.get(str(ext or "").lower(), "application/octet-stream")
