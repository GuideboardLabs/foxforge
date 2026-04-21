from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from shared_tools.db import connect as _db_connect_shared
from shared_tools.migrations import initialize_database


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso(raw: str) -> datetime | None:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def normalize_query(query: str) -> str:
    return " ".join(str(query or "").strip().lower().split())


def settings_digest(settings: dict[str, Any]) -> str:
    payload = json.dumps(settings, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def cache_key(*, project: str, query: str, topic_type: str, settings_hash: str) -> str:
    # Include project for predictable source-path behavior and per-project context.
    raw = "|".join([
        str(project or "general").strip().lower() or "general",
        normalize_query(query),
        str(topic_type or "general").strip().lower() or "general",
        str(settings_hash or "").strip(),
    ])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class WebQueryCache:
    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root
        self._lock = Lock()
        initialize_database(repo_root)
        self._ensure_schema()

    def _connect(self):
        return _db_connect_shared(self.repo_root)

    def _ensure_schema(self) -> None:
        conn = self._connect()
        with conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS web_query_cache (
                    key TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    ttl_sec INTEGER NOT NULL,
                    topic_type TEXT NOT NULL,
                    source TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_web_query_cache_created ON web_query_cache(created_at DESC)"
            )

    def get(self, key: str) -> dict[str, Any] | None:
        cache_key_value = str(key or "").strip()
        if not cache_key_value:
            return None
        with self._lock:
            with self._connect() as conn:
                row = conn.execute(
                    """
                    SELECT payload_json, created_at, ttl_sec, topic_type, source
                    FROM web_query_cache
                    WHERE key = ?
                    """,
                    (cache_key_value,),
                ).fetchone()
                if row is None:
                    return None
                created_at = _parse_iso(str(row["created_at"]))
                ttl_sec = int(row["ttl_sec"] or 0)
                if created_at is None or ttl_sec <= 0:
                    conn.execute("DELETE FROM web_query_cache WHERE key = ?", (cache_key_value,))
                    return None
                age_sec = (datetime.now(timezone.utc) - created_at).total_seconds()
                if age_sec >= ttl_sec:
                    conn.execute("DELETE FROM web_query_cache WHERE key = ?", (cache_key_value,))
                    return None
                try:
                    payload = json.loads(str(row["payload_json"]))
                except json.JSONDecodeError:
                    conn.execute("DELETE FROM web_query_cache WHERE key = ?", (cache_key_value,))
                    return None
                if not isinstance(payload, dict):
                    conn.execute("DELETE FROM web_query_cache WHERE key = ?", (cache_key_value,))
                    return None
                payload["_cache"] = {
                    "hit": True,
                    "age_sec": int(max(0, age_sec)),
                    "created_at": str(row["created_at"]),
                    "ttl_sec": ttl_sec,
                    "topic_type": str(row["topic_type"] or ""),
                    "source": str(row["source"] or ""),
                }
                return payload

    def put(
        self,
        key: str,
        payload: dict[str, Any],
        *,
        ttl_sec: int,
        topic_type: str,
        source: str,
    ) -> None:
        cache_key_value = str(key or "").strip()
        if not cache_key_value:
            return
        ttl = max(60, int(ttl_sec))
        row_payload = dict(payload)
        row_payload.pop("_cache", None)
        raw_json = json.dumps(row_payload, ensure_ascii=True)
        with self._lock:
            conn = self._connect()
            with conn:
                conn.execute(
                    """
                    INSERT INTO web_query_cache (key, payload_json, created_at, ttl_sec, topic_type, source)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(key) DO UPDATE SET
                        payload_json=excluded.payload_json,
                        created_at=excluded.created_at,
                        ttl_sec=excluded.ttl_sec,
                        topic_type=excluded.topic_type,
                        source=excluded.source
                    """,
                    (
                        cache_key_value,
                        raw_json,
                        _now_iso(),
                        ttl,
                        str(topic_type or "general"),
                        str(source or "web_research"),
                    ),
                )

    def purge_expired(self) -> int:
        now = datetime.now(timezone.utc)
        deleted = 0
        with self._lock:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT key, created_at, ttl_sec FROM web_query_cache"
                ).fetchall()
                stale_keys: list[str] = []
                for row in rows:
                    created_at = _parse_iso(str(row["created_at"]))
                    ttl = int(row["ttl_sec"] or 0)
                    if created_at is None or ttl <= 0:
                        stale_keys.append(str(row["key"]))
                        continue
                    age = (now - created_at).total_seconds()
                    if age >= ttl:
                        stale_keys.append(str(row["key"]))
                if stale_keys:
                    conn.executemany(
                        "DELETE FROM web_query_cache WHERE key = ?",
                        [(k,) for k in stale_keys],
                    )
                    deleted = len(stale_keys)
        return deleted
