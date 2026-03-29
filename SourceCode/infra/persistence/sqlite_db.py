from __future__ import annotations

from pathlib import Path
import sqlite3

from shared_tools.db import connect, resolve_db_path
from shared_tools.migrations import initialize_database


def ensure_state_db(repo_root: Path | str | None = None) -> dict[str, object]:
    return initialize_database(repo_root)


def connect_db(repo_root: Path | str | None = None, *, timeout: float = 30.0) -> sqlite3.Connection:
    ensure_state_db(repo_root)
    return connect(repo_root, timeout=timeout)


def db_path(repo_root: Path | str | None = None) -> Path:
    ensure_state_db(repo_root)
    return resolve_db_path(repo_root)


__all__ = ["connect_db", "db_path", "ensure_state_db"]
