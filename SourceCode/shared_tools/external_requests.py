from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Protocol, runtime_checkable

from shared_tools.db import connect, row_to_dict, transaction
from shared_tools.migrations import initialize_database

VALID_EXTERNAL_TOOL_MODES = {"off", "ask", "auto"}

VALID_EXTERNAL_REQUEST_STATUSES = (
    "queued",
    "dispatched",
    "acknowledged",
    "working",
    "completed",
    "failed",
    "cancelled",
    "ignored",
)

OPEN_EXTERNAL_REQUEST_STATUSES = frozenset(
    {"queued", "dispatched", "acknowledged", "working"}
)

TERMINAL_EXTERNAL_REQUEST_STATUSES = frozenset(
    {"completed", "failed", "cancelled", "ignored"}
)

EXTERNAL_STATUS_TRANSITIONS: dict[str, frozenset[str]] = {
    "queued": frozenset({"dispatched", "failed", "cancelled", "ignored"}),
    "dispatched": frozenset(
        {"acknowledged", "working", "completed", "failed", "cancelled", "ignored"}
    ),
    "acknowledged": frozenset({"working", "completed", "failed", "cancelled", "ignored"}),
    "working": frozenset({"completed", "failed", "cancelled", "ignored"}),
    "completed": frozenset(),
    "failed": frozenset(),
    "cancelled": frozenset(),
    "ignored": frozenset(),
}

EXTERNAL_AUDIT_EVENT_NAMES: dict[str, str] = {
    "queued": "external_request_queued",
    "dispatched": "external_request_dispatched",
    "acknowledged": "external_request_acknowledged",
    "working": "external_request_working",
    "completed": "external_request_completed",
    "failed": "external_request_failed",
    "cancelled": "external_request_cancelled",
    "ignored": "external_request_ignored",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_dump(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True)


def _json_load_dict(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _normalize_status(status: str) -> str:
    key = str(status or "").strip().lower()
    if key not in VALID_EXTERNAL_REQUEST_STATUSES:
        raise ExternalRequestValidationError(
            f"Invalid external request status '{status}'. "
            f"Use one of: {', '.join(VALID_EXTERNAL_REQUEST_STATUSES)}."
        )
    return key


def _normalize_object(value: Any, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    raise ExternalRequestValidationError(f"{field_name} must be an object.")


class ExternalRequestError(ValueError):
    pass


class ExternalRequestValidationError(ExternalRequestError):
    pass


class ExternalRequestTransitionError(ExternalRequestError):
    pass


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    message: str = ""


@dataclass(frozen=True)
class ProviderAck:
    external_ref: str
    status: str = "acknowledged"
    message: str = ""


@dataclass(frozen=True)
class ProviderStatus:
    status: str
    message: str = ""
    result: dict[str, Any] | None = None


@dataclass(frozen=True)
class ExternalResult:
    summary: str
    details: dict[str, Any]


@runtime_checkable
class ExternalProviderAdapter(Protocol):
    provider: str

    def validate_config(self, settings: dict[str, Any]) -> ValidationResult: ...

    def submit(self, request_row: dict[str, Any]) -> ProviderAck: ...

    def poll_status(self, request_row: dict[str, Any]) -> ProviderStatus: ...

    def normalize_result(self, raw_payload: dict[str, Any]) -> ExternalResult: ...

    def extract_suggestions(self, result: ExternalResult) -> list[dict[str, Any]]: ...


class ExternalToolsSettings:
    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root
        self.root = repo_root / "Runtime" / "external"
        self.path = self.root / "settings.json"
        self.lock = Lock()
        self.root.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text(
                json.dumps(
                    {
                        "mode": "off",
                        "providers": [],
                    },
                    indent=2,
                    ensure_ascii=True,
                ),
                encoding="utf-8",
            )

    def _load(self) -> dict[str, Any]:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        mode = str(payload.get("mode", "off")).strip().lower()
        if mode not in VALID_EXTERNAL_TOOL_MODES:
            mode = "off"
        providers = payload.get("providers", [])
        if not isinstance(providers, list):
            providers = []
        payload["mode"] = mode
        payload["providers"] = [str(x).strip().lower() for x in providers if str(x).strip()]
        return payload

    def _save(self, payload: dict[str, Any]) -> None:
        self.path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )

    def get_mode(self) -> str:
        with self.lock:
            return str(self._load().get("mode", "off"))

    def set_mode(self, mode: str) -> str:
        key = str(mode or "").strip().lower()
        if key not in VALID_EXTERNAL_TOOL_MODES:
            raise ValueError("Invalid external tools mode. Use: off, ask, auto.")
        with self.lock:
            payload = self._load()
            payload["mode"] = key
            self._save(payload)
        return key

    def mode_text(self) -> str:
        settings = self._load()
        providers = settings.get("providers", [])
        providers_text = ",".join([str(x) for x in providers]) if providers else "(none)"
        return (
            "External tools mode:\n"
            f"- mode: {settings.get('mode', 'off')}\n"
            f"- providers: {providers_text}"
        )


class ExternalRequestStore:
    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root
        initialize_database(self.repo_root)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with connect(self.repo_root) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS external_requests (
                    id TEXT PRIMARY KEY,
                    provider TEXT NOT NULL,
                    intent TEXT NOT NULL,
                    project TEXT NOT NULL,
                    lane TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    policy_json TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    external_ref TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    completed_at TEXT
                );
                """.strip()
            )
            conn.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_external_requests_provider_external_ref
                ON external_requests(provider, external_ref)
                WHERE external_ref <> '';
                """.strip()
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_external_requests_status_created ON external_requests(status, created_at);"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_external_requests_project_created ON external_requests(project, created_at);"
            )

    def _row_to_public(self, row: Any) -> dict[str, Any]:
        data = row_to_dict(row) if row is not None else {}
        payload = _json_load_dict(data.get("payload_json"))
        policy = _json_load_dict(data.get("policy_json"))
        result = _json_load_dict(data.get("result_json"))
        suggestions = result.get("suggestions", [])
        suggestions_count = len(suggestions) if isinstance(suggestions, list) else 0
        return {
            "id": str(data.get("id", "")),
            "provider": str(data.get("provider", "")),
            "intent": str(data.get("intent", "")),
            "project": str(data.get("project", "general") or "general"),
            "lane": str(data.get("lane", "project") or "project"),
            "summary": str(data.get("summary", "")),
            "payload_json": payload,
            "status": str(data.get("status", "")),
            "policy_json": policy,
            "result_json": result,
            "external_ref": str(data.get("external_ref", "")),
            "created_at": str(data.get("created_at", "")),
            "updated_at": str(data.get("updated_at", "")),
            "completed_at": str(data.get("completed_at", "")),
            "suggestions_count": suggestions_count,
        }

    def _get_row(self, conn, request_id: str):
        return conn.execute(
            "SELECT * FROM external_requests WHERE id = ?;",
            (request_id,),
        ).fetchone()

    def _validate_transition(self, current_status: str, next_status: str) -> None:
        current = _normalize_status(current_status)
        nxt = _normalize_status(next_status)
        if current == nxt:
            return
        if current in TERMINAL_EXTERNAL_REQUEST_STATUSES:
            raise ExternalRequestTransitionError(
                f"Cannot transition terminal status '{current}' to '{nxt}'."
            )
        allowed = EXTERNAL_STATUS_TRANSITIONS.get(current, frozenset())
        if nxt not in allowed:
            raise ExternalRequestTransitionError(
                f"Invalid transition '{current}' -> '{nxt}'."
            )

    def create(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ExternalRequestValidationError("External request payload must be an object.")

        request_id = str(payload.get("id", "")).strip() or f"ext_{uuid.uuid4().hex[:10]}"
        provider = str(payload.get("provider", "")).strip().lower()
        intent = str(payload.get("intent", "")).strip().lower()
        project = str(payload.get("project", "general")).strip() or "general"
        lane = str(payload.get("lane", "project")).strip() or "project"
        summary = str(payload.get("summary", "")).strip()
        if not provider:
            raise ExternalRequestValidationError("provider is required.")
        if not intent:
            raise ExternalRequestValidationError("intent is required.")
        if not summary:
            raise ExternalRequestValidationError("summary is required.")
        status = _normalize_status(str(payload.get("status", "queued")))
        body = _normalize_object(
            payload.get("payload_json", payload.get("payload", {})),
            "payload_json",
        )
        policy = _normalize_object(
            payload.get("policy_json", payload.get("policy", {})),
            "policy_json",
        )
        result = _normalize_object(
            payload.get("result_json", payload.get("result", {})),
            "result_json",
        )
        external_ref = str(payload.get("external_ref", "")).strip()
        now = _now_iso()
        completed_at = now if status in TERMINAL_EXTERNAL_REQUEST_STATUSES else None

        with connect(self.repo_root) as conn, transaction(conn, immediate=True):
            existing_by_id = self._get_row(conn, request_id)
            if existing_by_id is not None:
                return self._row_to_public(existing_by_id)

            if external_ref:
                existing_by_ref = conn.execute(
                    """
                    SELECT * FROM external_requests
                    WHERE provider = ? AND external_ref = ?;
                    """.strip(),
                    (provider, external_ref),
                ).fetchone()
                if existing_by_ref is not None:
                    return self._row_to_public(existing_by_ref)

            conn.execute(
                """
                INSERT INTO external_requests (
                    id, provider, intent, project, lane, summary,
                    payload_json, status, policy_json, result_json, external_ref,
                    created_at, updated_at, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """.strip(),
                (
                    request_id,
                    provider,
                    intent,
                    project,
                    lane,
                    summary,
                    _json_dump(body),
                    status,
                    _json_dump(policy),
                    _json_dump(result),
                    external_ref,
                    now,
                    now,
                    completed_at,
                ),
            )
            row = self._get_row(conn, request_id)
        return self._row_to_public(row)

    def get(self, request_id: str) -> dict[str, Any] | None:
        key = str(request_id).strip()
        if not key:
            return None
        with connect(self.repo_root) as conn:
            row = self._get_row(conn, key)
        return self._row_to_public(row) if row is not None else None

    def list_open(
        self,
        *,
        limit: int = 50,
        provider: str = "",
        project: str = "",
    ) -> list[dict[str, Any]]:
        cap = max(1, min(int(limit), 500))
        statuses = list(OPEN_EXTERNAL_REQUEST_STATUSES)
        where = [f"status IN ({','.join(['?'] * len(statuses))})"]
        params: list[Any] = statuses
        if str(provider).strip():
            where.append("provider = ?")
            params.append(str(provider).strip().lower())
        if str(project).strip():
            where.append("project = ?")
            params.append(str(project).strip())
        where_sql = " AND ".join(where)
        query = (
            "SELECT * FROM external_requests "
            f"WHERE {where_sql} "
            "ORDER BY datetime(created_at) DESC, id DESC LIMIT ?;"
        )
        params.append(cap)
        with connect(self.repo_root) as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
        return [self._row_to_public(row) for row in rows]

    def transition_status(
        self,
        request_id: str,
        to_status: str,
        *,
        note: str = "",
        external_ref: str = "",
    ) -> dict[str, Any]:
        key = str(request_id).strip()
        if not key:
            raise ExternalRequestValidationError("request_id is required.")
        target = _normalize_status(to_status)
        with connect(self.repo_root) as conn, transaction(conn, immediate=True):
            row = self._get_row(conn, key)
            if row is None:
                raise ExternalRequestValidationError(f"External request not found: {key}")
            public = self._row_to_public(row)
            current = str(public.get("status", "")).strip().lower()
            self._validate_transition(current, target)
            if current == target:
                return public

            result = dict(public.get("result_json", {}))
            if note.strip():
                transitions = result.get("transitions", [])
                if not isinstance(transitions, list):
                    transitions = []
                transitions.append(
                    {
                        "from": current,
                        "to": target,
                        "note": note.strip(),
                        "at": _now_iso(),
                    }
                )
                result["transitions"] = transitions[-50:]

            now = _now_iso()
            final_ref = str(public.get("external_ref", "")).strip()
            if not final_ref and str(external_ref).strip():
                final_ref = str(external_ref).strip()
            completed = now if target in TERMINAL_EXTERNAL_REQUEST_STATUSES else None
            conn.execute(
                """
                UPDATE external_requests
                SET status = ?, result_json = ?, external_ref = ?, updated_at = ?, completed_at = ?
                WHERE id = ?;
                """.strip(),
                (
                    target,
                    _json_dump(result),
                    final_ref,
                    now,
                    completed,
                    key,
                ),
            )
            updated = self._get_row(conn, key)
        return self._row_to_public(updated)

    def append_result(self, request_id: str, patch: dict[str, Any]) -> dict[str, Any]:
        key = str(request_id).strip()
        if not key:
            raise ExternalRequestValidationError("request_id is required.")
        if not isinstance(patch, dict):
            raise ExternalRequestValidationError("result patch must be an object.")
        with connect(self.repo_root) as conn, transaction(conn, immediate=True):
            row = self._get_row(conn, key)
            if row is None:
                raise ExternalRequestValidationError(f"External request not found: {key}")
            public = self._row_to_public(row)
            result = dict(public.get("result_json", {}))
            for k, v in patch.items():
                result[str(k)] = v
            now = _now_iso()
            conn.execute(
                "UPDATE external_requests SET result_json = ?, updated_at = ? WHERE id = ?;",
                (_json_dump(result), now, key),
            )
            updated = self._get_row(conn, key)
        return self._row_to_public(updated)

    def mark_terminal(
        self,
        request_id: str,
        status: str,
        result_patch: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        target = _normalize_status(status)
        if target not in TERMINAL_EXTERNAL_REQUEST_STATUSES:
            raise ExternalRequestTransitionError(
                f"mark_terminal requires terminal status. Got '{target}'."
            )
        row = self.get(request_id)
        if row is None:
            raise ExternalRequestValidationError(f"External request not found: {request_id}")
        current = str(row.get("status", "")).strip().lower()
        if current != target:
            row = self.transition_status(request_id, target)
        if isinstance(result_patch, dict) and result_patch:
            row = self.append_result(request_id, result_patch)
        return row

    def transition_event_name(self, status: str) -> str:
        key = _normalize_status(status)
        return EXTERNAL_AUDIT_EVENT_NAMES.get(key, "external_request_updated")
