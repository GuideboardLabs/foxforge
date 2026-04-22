"""Canonical SQLite helpers for Flask request-scoped connections."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from flask import current_app, g


def _database_path() -> Path:
    """Resolve the SQLite file path relative to the app root."""
    return Path(current_app.root_path) / "app.db"


def get_db() -> sqlite3.Connection:
    """Return the request-scoped SQLite connection."""
    if "db" not in g:
        db = sqlite3.connect(str(_database_path()), detect_types=sqlite3.PARSE_DECLTYPES)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys = ON")
        db.execute("PRAGMA journal_mode = WAL")
        db.execute("PRAGMA synchronous = NORMAL")
        g.db = db
    return g.db


def close_db(_error: BaseException | None = None) -> None:
    """Close and clear the request-scoped connection."""
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db(schema_path: Path | None = None) -> None:
    """Initialize database schema from schema.sql."""
    db = get_db()
    resolved_schema = schema_path or (Path(current_app.root_path) / "schema.sql")
    script = resolved_schema.read_text(encoding="utf-8")
    db.executescript(script)
    db.commit()


def row_to_dict(row: sqlite3.Row | None) -> dict[str, Any]:
    """Convert a sqlite3.Row to a plain dictionary."""
    if row is None:
        return {}
    return {str(key): row[key] for key in row.keys()}
