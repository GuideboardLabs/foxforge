"""Request parsing utilities for route handlers."""

from __future__ import annotations


def parse_optional_int(value: str | None, default: int, minimum: int = 1, maximum: int = 500) -> int:
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return max(minimum, min(maximum, parsed))
