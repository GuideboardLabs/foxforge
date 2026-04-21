from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .turn_graph import compile_chat_turn_graph


@dataclass(slots=True)
class TurnSummary:
    thread_id: str
    checkpoints: int
    last_checkpoint_id: str = ""
    last_ts: str = ""


@dataclass(slots=True)
class NodeDiff:
    key: str
    left: Any
    right: Any
    changed: bool


def _checkpoint_db_path(repo_root: Path) -> Path:
    return Path(repo_root) / "Runtime" / "state" / "turn_checkpoints.sqlite"


def _parse_iso(raw: str) -> datetime | None:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    try:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    except Exception:
        return set()
    cols: set[str] = set()
    for row in rows:
        try:
            cols.add(str(row[1]))
        except Exception:
            continue
    return cols


def _checkpoint_rows(repo_root: Path) -> list[dict[str, Any]]:
    db = _checkpoint_db_path(repo_root)
    if not db.exists():
        return []
    out: list[dict[str, Any]] = []
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    try:
        tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        names = [str(row[0]) for row in tables]
        target = "checkpoints" if "checkpoints" in names else ""
        if not target:
            return []
        rows = conn.execute(f"SELECT * FROM {target} ORDER BY rowid DESC").fetchall()
        for row in rows:
            payload = {k: row[k] for k in row.keys()}
            out.append(payload)
    finally:
        conn.close()
    return out


def list_turns(repo_root: Path, *, thread_id: str = "", since_ts: str = "") -> list[TurnSummary]:
    rows = _checkpoint_rows(repo_root)
    cutoff = _parse_iso(since_ts) if since_ts else None
    by_thread: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        tid = str(row.get("thread_id", "") or row.get("thread", "")).strip()
        if not tid:
            continue
        if thread_id and tid != str(thread_id).strip():
            continue
        meta = row.get("metadata")
        if isinstance(meta, (bytes, bytearray)):
            try:
                meta = meta.decode("utf-8", errors="ignore")
            except Exception:
                meta = ""
        last_ts = ""
        if isinstance(meta, str) and meta.strip():
            try:
                parsed = json.loads(meta)
            except Exception:
                parsed = {}
            if isinstance(parsed, dict):
                last_ts = str(parsed.get("ts", "") or parsed.get("created_at", "")).strip()
        if not last_ts:
            last_ts = str(row.get("ts", "") or row.get("created_at", "")).strip()
        if cutoff is not None:
            dt = _parse_iso(last_ts)
            if dt is not None and dt < cutoff:
                continue
        row["_ts"] = last_ts
        by_thread.setdefault(tid, []).append(row)

    out: list[TurnSummary] = []
    for tid, thread_rows in by_thread.items():
        out.append(
            TurnSummary(
                thread_id=tid,
                checkpoints=len(thread_rows),
                last_checkpoint_id=str(thread_rows[0].get("checkpoint_id", "") or thread_rows[0].get("id", "")),
                last_ts=str(thread_rows[0].get("_ts", "")),
            )
        )
    out.sort(key=lambda item: item.last_ts, reverse=True)
    return out


def get_turn_trace(
    repo_root: Path,
    *,
    thread_id: str,
    checkpoint_id: str = "",
    orchestrator: Any | None = None,
) -> list[dict[str, Any]]:
    tid = str(thread_id or "").strip()
    if not tid:
        return []

    graph_trace: list[dict[str, Any]] = []
    if orchestrator is not None:
        try:
            graph, _checkpoint, _version = compile_chat_turn_graph(orchestrator, with_checkpointer=True)
            if graph is not None and hasattr(graph, "get_state_history"):
                config: dict[str, Any] = {"configurable": {"thread_id": tid}}
                if checkpoint_id:
                    config["configurable"]["checkpoint_id"] = str(checkpoint_id).strip()
                history = graph.get_state_history(config)  # type: ignore[attr-defined]
                for item in history:
                    checkpoint = getattr(item, "config", {}) if item is not None else {}
                    values = getattr(item, "values", {}) if item is not None else {}
                    meta = getattr(item, "metadata", {}) if item is not None else {}
                    next_nodes = getattr(item, "next", []) if item is not None else []
                    graph_trace.append(
                        {
                            "checkpoint_id": str((checkpoint.get("configurable", {}) or {}).get("checkpoint_id", "")),
                            "thread_id": tid,
                            "state": values if isinstance(values, dict) else {},
                            "metadata": meta if isinstance(meta, dict) else {},
                            "next": list(next_nodes or []),
                        }
                    )
        except Exception:
            graph_trace = []
    if graph_trace:
        return graph_trace

    # Fallback: read persisted checkpoint rows directly.
    out: list[dict[str, Any]] = []
    for row in _checkpoint_rows(repo_root):
        row_tid = str(row.get("thread_id", "") or row.get("thread", "")).strip()
        if row_tid != tid:
            continue
        ckpt = str(row.get("checkpoint_id", "") or row.get("id", ""))
        if checkpoint_id and ckpt != checkpoint_id:
            continue
        metadata = row.get("metadata")
        if isinstance(metadata, (bytes, bytearray)):
            metadata = metadata.decode("utf-8", errors="ignore")
        meta_obj: dict[str, Any] = {}
        if isinstance(metadata, str) and metadata.strip():
            try:
                parsed = json.loads(metadata)
            except Exception:
                parsed = {}
            if isinstance(parsed, dict):
                meta_obj = parsed
        out.append(
            {
                "checkpoint_id": ckpt,
                "thread_id": row_tid,
                "state": {},
                "metadata": meta_obj,
                "next": [],
            }
        )
    return out


def _find_checkpoint_for_node(trace: list[dict[str, Any]], node_name: str) -> str:
    target = str(node_name or "").strip().lower()
    if not target:
        return ""
    for snap in trace:
        next_nodes = snap.get("next") if isinstance(snap.get("next"), list) else []
        if any(str(name).strip().lower() == target for name in next_nodes):
            return str(snap.get("checkpoint_id", "")).strip()
        metadata = snap.get("metadata", {}) if isinstance(snap.get("metadata", {}), dict) else {}
        node = str(metadata.get("source", "") or metadata.get("node", "")).strip().lower()
        if node == target:
            return str(snap.get("checkpoint_id", "")).strip()
    return ""


def replay_turn(
    orchestrator: Any,
    *,
    thread_id: str,
    from_node: str = "",
    mutate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    graph, checkpoint_path, version_hash = compile_chat_turn_graph(orchestrator, with_checkpointer=True)
    if graph is None:
        return {"ok": False, "error": "LangGraph is unavailable in this runtime."}

    tid = str(thread_id or "").strip()
    if not tid:
        return {"ok": False, "error": "thread_id is required."}
    config: dict[str, Any] = {"configurable": {"thread_id": tid}}

    trace = get_turn_trace(Path(orchestrator.repo_root), thread_id=tid, orchestrator=orchestrator)
    from_checkpoint_id = _find_checkpoint_for_node(trace, from_node) if from_node else ""
    if from_checkpoint_id:
        config["configurable"]["checkpoint_id"] = from_checkpoint_id

    branch_checkpoint_id = ""
    mutate_payload = dict(mutate or {})
    if mutate_payload:
        try:
            update_result = graph.update_state(  # type: ignore[attr-defined]
                config,
                mutate_payload,
                as_node=str(from_node or "compose"),
            )
            if isinstance(update_result, dict):
                cfg = update_result.get("configurable", {}) if isinstance(update_result.get("configurable", {}), dict) else {}
                branch_checkpoint_id = str(cfg.get("checkpoint_id", "")).strip()
                if branch_checkpoint_id:
                    config["configurable"]["checkpoint_id"] = branch_checkpoint_id
        except Exception as exc:
            return {"ok": False, "error": f"update_state failed: {exc}"}

    try:
        replay_state = graph.invoke(None, config=config)
    except Exception as exc:
        return {"ok": False, "error": f"graph replay failed: {exc}"}

    replay_id = f"replay_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{tid[:10]}"
    replay_dir = Path(orchestrator.repo_root) / "Runtime" / "state" / "replays" / replay_id
    replay_dir.mkdir(parents=True, exist_ok=True)

    lineage = {
        "thread_id": tid,
        "from_node": str(from_node or ""),
        "from_checkpoint_id": from_checkpoint_id,
        "branch_checkpoint_id": branch_checkpoint_id,
        "graph_version_hash": version_hash,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else "",
        "mutate": mutate_payload,
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    (replay_dir / "lineage.json").write_text(json.dumps(lineage, indent=2, ensure_ascii=True), encoding="utf-8")
    (replay_dir / "replay_state.json").write_text(json.dumps(replay_state, indent=2, ensure_ascii=True, default=str), encoding="utf-8")

    return {
        "ok": True,
        "thread_id": tid,
        "replay_dir": str(replay_dir),
        "lineage_path": str(replay_dir / "lineage.json"),
        "state_path": str(replay_dir / "replay_state.json"),
        "state": replay_state if isinstance(replay_state, dict) else {},
        "graph_version_hash": version_hash,
    }


def diff_turns(left_trace: list[dict[str, Any]], right_trace: list[dict[str, Any]]) -> list[NodeDiff]:
    left_state = left_trace[0].get("state", {}) if left_trace else {}
    right_state = right_trace[0].get("state", {}) if right_trace else {}
    if not isinstance(left_state, dict):
        left_state = {}
    if not isinstance(right_state, dict):
        right_state = {}
    keys = sorted(set(left_state.keys()) | set(right_state.keys()))
    out: list[NodeDiff] = []
    for key in keys:
        lval = left_state.get(key)
        rval = right_state.get(key)
        out.append(NodeDiff(key=key, left=lval, right=rval, changed=lval != rval))
    return out


def serialize_diffs(diffs: list[NodeDiff]) -> list[dict[str, Any]]:
    return [asdict(item) for item in diffs]

