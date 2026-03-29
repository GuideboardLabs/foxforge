from __future__ import annotations

import json
import os
import random
import time
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from shared_tools.db import connect, row_to_dict, transaction
from shared_tools.file_store import ProjectStore
from shared_tools.migrations import initialize_database


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class CloudConsultEngine:
    VALID_MODES = {"off", "ask", "auto"}
    VALID_PROVIDERS = ("gemini",)

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root
        self.store = ProjectStore(repo_root)
        self.root = repo_root / "Runtime" / "cloud"
        self.settings_path = self.root / "settings.json"
        self.lock = Lock()

        self.root.mkdir(parents=True, exist_ok=True)
        initialize_database(self.repo_root)
        if not self.settings_path.exists():
            self.settings_path.write_text(
                json.dumps(
                    {
                        "mode": "auto",
                        "providers": ["gemini"],
                        "gemini_model": "gemini-2.0-flash",
                        "max_output_chars": 12000,
                        "daily_limit": 250,
                        "gemini_critique_enabled": False,
                        "gemini_api_keys": [],
                    },
                    indent=2,
                    ensure_ascii=True,
                ),
                encoding="utf-8",
            )

    def _serialize_json(self, payload: dict[str, Any]) -> str:
        return json.dumps(payload or {}, ensure_ascii=True, sort_keys=True)

    def _deserialize_json(self, raw: Any) -> dict[str, Any]:
        if not isinstance(raw, str) or not raw.strip():
            return {}
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}

    def _cloud_row_to_public(self, row: Any) -> dict[str, Any]:
        data = row_to_dict(row) if row is not None else {}
        request_payload = self._deserialize_json(data.get("request_payload_json"))
        response_payload = self._deserialize_json(data.get("response_json"))
        public = {
            "id": str(data.get("id", "")),
            "type": "cloud_consult",
            "status": str(data.get("status", "")),
            "project": str(data.get("project", "general") or "general"),
            "lane": str(data.get("lane", "project") or "project"),
            "query": str(request_payload.get("query", "")),
            "reason": str(request_payload.get("reason", "")),
            "context_preview": str(request_payload.get("context_preview", "")),
            "question": str(request_payload.get("question", "Allow cloud consult for this request?")),
            "summary": str(request_payload.get("summary", "")),
            "created_at": str(data.get("created_at", "")),
            "updated_at": str(response_payload.get("updated_at", data.get("completed_at", "") or data.get("created_at", ""))),
            "resolved_at": str(data.get("completed_at", "")),
            "mode": str(data.get("mode", "ask")),
            "purpose": str(data.get("purpose", "advice")),
        }
        public.update({k: v for k, v in response_payload.items() if k not in {"request_payload_json", "response_json"}})
        return public

    def _today_count(self) -> int:
        today = datetime.now(timezone.utc).date().isoformat()
        with connect(self.repo_root) as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM cloud_requests WHERE substr(created_at, 1, 10) = ? AND status IN ('completed', 'failed', 'resolved');",
                (today,),
            ).fetchone()
        return int(row[0] if row is not None else 0)

    def _load_settings(self) -> dict[str, Any]:
        try:
            data = json.loads(self.settings_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = {}
        if not isinstance(data, dict):
            data = {}

        mode = str(data.get("mode", "auto")).strip().lower()
        if mode not in self.VALID_MODES:
            mode = "auto"
        data["mode"] = mode

        providers_raw = data.get("providers", ["gemini"])
        providers: list[str] = []
        if isinstance(providers_raw, list):
            for item in providers_raw:
                key = str(item).strip().lower()
                if key in self.VALID_PROVIDERS and key not in providers:
                    providers.append(key)
        if not providers:
            providers = ["gemini"]
        data["providers"] = providers

        data["gemini_model"] = str(data.get("gemini_model", "gemini-2.0-flash")).strip() or "gemini-2.0-flash"

        try:
            max_output_chars = int(data.get("max_output_chars", 12000))
        except (TypeError, ValueError):
            max_output_chars = 12000
        data["max_output_chars"] = max(1000, min(max_output_chars, 50000))

        try:
            daily_limit = int(data.get("daily_limit", 250))
        except (TypeError, ValueError):
            daily_limit = 250
        data["daily_limit"] = max(1, min(daily_limit, 500))

        try:
            retry_attempts = int(data.get("retry_attempts", 4))
        except (TypeError, ValueError):
            retry_attempts = 4
        data["retry_attempts"] = max(1, min(retry_attempts, 8))

        try:
            retry_base_delay_sec = float(data.get("retry_base_delay_sec", 1.0))
        except (TypeError, ValueError):
            retry_base_delay_sec = 1.0
        data["retry_base_delay_sec"] = max(0.1, min(retry_base_delay_sec, 10.0))

        try:
            retry_max_delay_sec = float(data.get("retry_max_delay_sec", 12.0))
        except (TypeError, ValueError):
            retry_max_delay_sec = 12.0
        data["retry_max_delay_sec"] = max(0.5, min(retry_max_delay_sec, 60.0))

        try:
            reserve_ratio = float(data.get("reserve_ratio", 0.2))
        except (TypeError, ValueError):
            reserve_ratio = 0.2
        data["reserve_ratio"] = max(0.0, min(reserve_ratio, 0.6))
        data["gemini_critique_enabled"] = bool(data.get("gemini_critique_enabled", False))
        keys_raw = data.get("gemini_api_keys", [])
        data["gemini_api_keys"] = [str(k).strip() for k in keys_raw if str(k).strip()] if isinstance(keys_raw, list) else []
        return data

    def _save_settings(self, settings: dict[str, Any]) -> None:
        self.settings_path.write_text(json.dumps(settings, indent=2, ensure_ascii=True), encoding="utf-8")

    def _resolve_gemini_keys(self, settings: dict[str, Any]) -> list[str]:
        keys: list[str] = list(settings.get("gemini_api_keys", []))
        env_key = str(os.getenv("GEMINI_API_KEY", "")).strip()
        if env_key and env_key not in keys:
            keys.append(env_key)
        return [k for k in keys if k]

    def _api_ready(self, provider: str) -> bool:
        key = provider.strip().lower()
        if key == "gemini":
            settings = self._load_settings()
            return bool(self._resolve_gemini_keys(settings))
        return False


    def get_mode(self) -> str:
        with self.lock:
            settings = self._load_settings()
            return str(settings.get("mode", "auto"))

    def set_mode(self, mode: str) -> str:
        key = mode.strip().lower()
        if key not in self.VALID_MODES:
            raise ValueError("Invalid cloud mode. Use: off, ask, auto.")
        with self.lock:
            settings = self._load_settings()
            settings["mode"] = key
            self._save_settings(settings)
        return key

    def mode_text(self) -> str:
        settings = self._load_settings()
        providers = settings.get("providers", [])
        providers_text = ",".join([str(x) for x in providers]) if isinstance(providers, list) else "gemini"
        return (
            "Cloud consult mode:\n"
            f"- mode: {settings.get('mode', 'auto')}\n"
            f"- providers: {providers_text}\n"
            f"- gemini_ready: {self._api_ready('gemini')}\n"
            f"- daily_limit: {settings.get('daily_limit', 250)}\n"
            f"- used_today: {self._today_count()}\n"
            f"- retry_attempts: {settings.get('retry_attempts', 4)}\n"
            f"- retry_base_delay_sec: {settings.get('retry_base_delay_sec', 1.0)}\n"
            f"- retry_max_delay_sec: {settings.get('retry_max_delay_sec', 12.0)}\n"
            f"- reserve_ratio: {settings.get('reserve_ratio', 0.2)}"
        )

    def usage_snapshot(self) -> dict[str, Any]:
        settings = self._load_settings()
        daily_limit = int(settings.get("daily_limit", 250))
        used_today = self._today_count()
        remaining = max(0, daily_limit - used_today)
        return {
            "used_today": used_today,
            "daily_limit": daily_limit,
            "remaining": remaining,
            "reserve_ratio": float(settings.get("reserve_ratio", 0.2)),
        }

    def create_pending(
        self,
        *,
        project: str,
        lane: str,
        query: str,
        reason: str,
        context: str = "",
    ) -> dict[str, Any]:
        query_text = query.strip()
        if not query_text:
            raise ValueError("Cloud pending query cannot be empty.")
        now = _now_iso()
        request_id = f"cloud_{uuid.uuid4().hex[:8]}"
        request_payload = {
            "query": query_text,
            "reason": reason.strip() or "Escalate to stronger cloud reasoning for quality.",
            "context_preview": context.strip()[:1200],
            "question": "Allow cloud consult for this request?",
            "summary": (
                f"Query: {query_text[:220]}"
                + ("" if len(query_text) <= 220 else "...")
                + f" | Reason: {(reason.strip() or 'Escalate to stronger cloud reasoning for quality.')[:180]}"
            ),
        }
        with connect(self.repo_root) as conn, transaction(conn, immediate=True):
            conn.execute(
                """
                INSERT INTO cloud_requests (
                    id, project, lane, mode, purpose,
                    request_payload_json, response_json, status,
                    created_at, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """.strip(),
                (
                    request_id,
                    project.strip() or "general",
                    lane.strip() or "project",
                    "ask",
                    "advice",
                    self._serialize_json(request_payload),
                    None,
                    "pending",
                    now,
                    None,
                ),
            )
            row = conn.execute("SELECT * FROM cloud_requests WHERE id = ?;", (request_id,)).fetchone()
        return self._cloud_row_to_public(row)

    def list_pending(self, limit: int = 50) -> list[dict[str, Any]]:
        limit = max(1, min(limit, 500))
        with connect(self.repo_root) as conn:
            rows = conn.execute(
                "SELECT * FROM cloud_requests WHERE status = 'pending' ORDER BY datetime(created_at) DESC, id DESC LIMIT ?;",
                (limit,),
            ).fetchall()
        return [self._cloud_row_to_public(row) for row in rows]

    def get_request(self, request_id: str) -> dict[str, Any] | None:
        with connect(self.repo_root) as conn:
            row = conn.execute("SELECT * FROM cloud_requests WHERE id = ?;", (request_id.strip(),)).fetchone()
        return self._cloud_row_to_public(row) if row is not None else None

    def ignore(self, request_id: str, reason: str = "") -> dict[str, Any] | None:
        key = request_id.strip()
        with connect(self.repo_root) as conn, transaction(conn, immediate=True):
            row = conn.execute("SELECT * FROM cloud_requests WHERE id = ? AND status = 'pending';", (key,)).fetchone()
            if row is None:
                return None
            response_payload = self._deserialize_json(row[6])
            response_payload.update({
                "ignore_reason": reason.strip() or "ignored by user",
                "updated_at": _now_iso(),
            })
            conn.execute(
                "UPDATE cloud_requests SET status = 'ignored', response_json = ?, completed_at = ? WHERE id = ?;",
                (self._serialize_json(response_payload), _now_iso(), key),
            )
            row = conn.execute("SELECT * FROM cloud_requests WHERE id = ?;", (key,)).fetchone()
        return self._cloud_row_to_public(row)

    def mark_routed(self, request_id: str, *, target: str, note: str = "", handoff_id: str = "") -> dict[str, Any] | None:
        key = request_id.strip()
        with connect(self.repo_root) as conn, transaction(conn, immediate=True):
            row = conn.execute("SELECT * FROM cloud_requests WHERE id = ? AND status = 'pending';", (key,)).fetchone()
            if row is None:
                return None
            response_payload = self._deserialize_json(row[6])
            response_payload.update({
                "routed_target": target.strip().lower(),
                "routed_note": note.strip(),
                "handoff_id": handoff_id.strip(),
                "updated_at": _now_iso(),
            })
            conn.execute(
                "UPDATE cloud_requests SET status = 'routed_external', response_json = ?, completed_at = ? WHERE id = ?;",
                (self._serialize_json(response_payload), _now_iso(), key),
            )
            row = conn.execute("SELECT * FROM cloud_requests WHERE id = ?;", (key,)).fetchone()
        return self._cloud_row_to_public(row)

    def _append_runs_log(self, payload: dict[str, Any]) -> None:
        return None


    def _record_cloud_request(
        self,
        *,
        request_id: str,
        project: str,
        lane: str,
        mode: str,
        purpose: str,
        request_payload: dict[str, Any],
        response_payload: dict[str, Any],
        status: str,
    ) -> None:
        now = _now_iso()
        with connect(self.repo_root) as conn, transaction(conn, immediate=True):
            conn.execute(
                """
                INSERT INTO cloud_requests (
                    id, project, lane, mode, purpose,
                    request_payload_json, response_json, status,
                    created_at, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    project = excluded.project,
                    lane = excluded.lane,
                    mode = excluded.mode,
                    purpose = excluded.purpose,
                    request_payload_json = excluded.request_payload_json,
                    response_json = excluded.response_json,
                    status = excluded.status,
                    completed_at = excluded.completed_at;
                """.strip(),
                (
                    request_id,
                    project.strip() or "general",
                    lane.strip() or "project",
                    mode.strip() or "auto",
                    purpose.strip() or "advice",
                    self._serialize_json(request_payload),
                    self._serialize_json(response_payload),
                    status.strip() or "completed",
                    now,
                    now,
                ),
            )

    def _extract_gemini_text(self, payload: dict[str, Any]) -> str:
        candidates = payload.get("candidates", [])
        if not isinstance(candidates, list):
            return ""
        texts: list[str] = []
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            content = candidate.get("content", {})
            if not isinstance(content, dict):
                continue
            parts = content.get("parts", [])
            if not isinstance(parts, list):
                continue
            for part in parts:
                if not isinstance(part, dict):
                    continue
                text = str(part.get("text", "")).strip()
                if text:
                    texts.append(text)
        return "\n".join(texts).strip()

    def _call_gemini(self, prompt: str, model: str, api_key: str = "") -> str:
        key = api_key.strip() or str(os.getenv("GEMINI_API_KEY", "")).strip()
        if not key:
            raise RuntimeError("GEMINI_API_KEY is not set.")
        safe_model = urllib.parse.quote(model.strip(), safe="")
        safe_key = urllib.parse.quote(key, safe="")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{safe_model}:generateContent?key={safe_key}"
        body = json.dumps(
            {
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.2},
            },
            ensure_ascii=True,
        ).encode("utf-8")
        req = urllib.request.Request(
            url=url,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "User-Agent": "Foxforge/1.0",
            },
        )
        with urllib.request.urlopen(req, timeout=90) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="ignore"))
        text = self._extract_gemini_text(data)
        if not text:
            raise RuntimeError("Gemini returned an empty response.")
        return text

    def _call_provider(
        self,
        provider: str,
        prompt: str,
        settings: dict[str, Any],
        api_key: str = "",
    ) -> tuple[str, str]:
        key = provider.strip().lower()
        if key == "gemini":
            model = str(settings.get("gemini_model", "gemini-2.0-flash"))
            return self._call_gemini(prompt, model, api_key=api_key), model
        raise RuntimeError(f"Unsupported cloud provider: {provider}")

    def _is_rate_limited_error(self, error_text: str) -> bool:
        low = str(error_text or "").lower()
        markers = ("429", "too many requests", "rate limit", "quota", "resource_exhausted")
        return any(token in low for token in markers)

    def _call_provider_with_retry(
        self,
        provider: str,
        prompt: str,
        settings: dict[str, Any],
        api_key: str = "",
    ) -> tuple[str, str]:
        attempts = int(settings.get("retry_attempts", 4))
        base_delay = float(settings.get("retry_base_delay_sec", 1.0))
        max_delay = float(settings.get("retry_max_delay_sec", 12.0))
        last_error = ""
        saw_rate_limit = False

        for idx in range(max(1, attempts)):
            try:
                return self._call_provider(provider, prompt, settings, api_key=api_key)
            except Exception as exc:  # pragma: no cover
                last_error = str(exc)
                if not self._is_rate_limited_error(last_error):
                    raise
                saw_rate_limit = True
                if idx >= attempts - 1:
                    break
                delay = min(max_delay, base_delay * (2**idx))
                # small jitter to avoid synchronized retries
                delay = delay * (1.0 + random.uniform(0.0, 0.2))
                time.sleep(delay)

        if saw_rate_limit:
            raise RuntimeError(
                f"HTTP 429/rate-limit after {attempts} attempt(s): {last_error or 'no detail'}"
            )
        raise RuntimeError(last_error or "Cloud provider call failed.")

    def _build_prompt(self, *, project: str, lane: str, query: str, reason: str, context: str) -> str:
        context_text = context.strip()[:12000]
        return (
            "You are a high-signal cloud advisor for Foxforge.\n"
            "Return concise, practical guidance with strong reasoning.\n"
            "If facts are uncertain, say so directly.\n"
            "Format sections: Assessment, Recommendations, Risks, Next Actions.\n\n"
            f"Project: {project}\n"
            f"Lane: {lane}\n"
            f"Reason for escalation: {reason}\n"
            f"User request: {query}\n\n"
            "Local context (may include local research output and web-source cache):\n"
            f"{context_text or '(none provided)'}\n"
        )

    def run_query(
        self,
        *,
        project: str,
        lane: str,
        query: str,
        reason: str,
        context: str = "",
        request_id: str = "",
        note: str = "",
    ) -> dict[str, Any]:
        settings = self._load_settings()
        mode = str(settings.get("mode", "auto"))
        if mode == "ask" and request_id.strip() not in {"", "auto"} and not request_id.strip().startswith("cloud_"):
            # allow explicit pending IDs or fully direct calls; deny legacy auto-try marker
            return {
                "ok": False,
                "project": project,
                "lane": lane,
                "query": query,
                "reason": reason,
                "request_id": request_id,
                "provider": "",
                "model": "",
                "response_text": "",
                "response_path": "",
                "message": "Ask mode requires an explicit approval request before cloud execution.",
            }
        daily_limit = int(settings.get("daily_limit", 250))
        used_today = self._today_count()
        request_key = request_id.strip() or f"cloud_run_{uuid.uuid4().hex[:8]}"
        request_payload = {
            "query": query,
            "reason": reason,
            "context_preview": context.strip()[:1200],
            "note": note.strip(),
        }
        if used_today >= daily_limit:
            result = {
                "ok": False,
                "project": project,
                "lane": lane,
                "query": query,
                "reason": reason,
                "request_id": request_key,
                "provider": "",
                "model": "",
                "response_text": "",
                "response_path": "",
                "message": f"Daily cloud limit reached ({used_today}/{daily_limit}).",
            }
            with connect(self.repo_root) as conn, transaction(conn, immediate=True):
                conn.execute(
                    """
                    INSERT INTO cloud_requests (id, project, lane, mode, purpose, request_payload_json, response_json, status, created_at, completed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET response_json = excluded.response_json, status = excluded.status, completed_at = excluded.completed_at;
                    """.strip(),
                    (request_key, project, lane, mode, "advice", self._serialize_json(request_payload), self._serialize_json(result), "failed", _now_iso(), _now_iso()),
                )
            return result

        prompt = self._build_prompt(project=project, lane=lane, query=query, reason=reason, context=context)
        providers = settings.get("providers", ["gemini"])
        provider_order = [str(x).strip().lower() for x in providers if str(x).strip().lower() in self.VALID_PROVIDERS]
        if not provider_order:
            provider_order = ["gemini"]

        errors: list[str] = []
        response_text = ""
        used_provider = ""
        used_model = ""
        rate_limited = False
        for provider in provider_order:
            if not self._api_ready(provider):
                errors.append(f"{provider}: api key missing")
                continue
            if provider == "gemini":
                keys = self._resolve_gemini_keys(settings)
                if not keys:
                    errors.append("gemini: api key missing")
                    continue
                for idx, api_key in enumerate(keys):
                    try:
                        response_text, used_model = self._call_provider_with_retry(
                            provider,
                            prompt,
                            settings,
                            api_key=api_key,
                        )
                        used_provider = provider
                        break
                    except Exception as exc:  # pragma: no cover
                        error_text = str(exc)
                        if self._is_rate_limited_error(error_text):
                            rate_limited = True
                        errors.append(f"{provider}[key_index={idx}]: {error_text}")
                if response_text.strip():
                    break
                continue
            try:
                response_text, used_model = self._call_provider_with_retry(provider, prompt, settings)
                used_provider = provider
                break
            except Exception as exc:  # pragma: no cover
                error_text = str(exc)
                if self._is_rate_limited_error(error_text):
                    rate_limited = True
                errors.append(f"{provider}: {error_text}")

        if not response_text.strip():
            result = {
                "ok": False,
                "project": project,
                "lane": lane,
                "query": query,
                "reason": reason,
                "request_id": request_key,
                "provider": "",
                "model": "",
                "response_text": "",
                "response_path": "",
                "rate_limited": rate_limited,
                "message": f"Cloud consult failed: {'; '.join(errors) if errors else 'No cloud provider available.'}",
            }
            with connect(self.repo_root) as conn, transaction(conn, immediate=True):
                conn.execute(
                    """
                    INSERT INTO cloud_requests (id, project, lane, mode, purpose, request_payload_json, response_json, status, created_at, completed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET response_json = excluded.response_json, status = excluded.status, completed_at = excluded.completed_at;
                    """.strip(),
                    (request_key, project, lane, mode, "advice", self._serialize_json(request_payload), self._serialize_json(result), "failed", _now_iso(), _now_iso()),
                )
            return result

        max_output_chars = int(settings.get("max_output_chars", 12000))
        text = response_text.strip()
        if len(text) > max_output_chars:
            text = text[:max_output_chars].rstrip() + "\n\n[Truncated by max_output_chars setting.]"
        lines = [
            "# Cloud Consult", "", f"- request_id: {request_key}", f"- project: {project}", f"- lane: {lane}",
            f"- provider: {used_provider}", f"- model: {used_model}", f"- query: {query}", f"- reason: {reason}",
            f"- note: {note.strip() or 'none'}", f"- captured_at: {_now_iso()}", "", "## Context", "", context.strip()[:12000] or "(none)", "", "## Response", "", text, "",
        ]
        filename = self.store.timestamped_name("cloud_consult")
        response_path = self.store.write_project_file(project, "research_cloud_advice", filename, "\n".join(lines))
        result = {
            "ok": True,
            "project": project,
            "lane": lane,
            "query": query,
            "reason": reason,
            "request_id": request_key,
            "provider": used_provider,
            "model": used_model,
            "response_text": text,
            "response_path": str(response_path),
            "rate_limited": False,
            "message": "Cloud consult completed.",
        }
        with connect(self.repo_root) as conn, transaction(conn, immediate=True):
            conn.execute(
                """
                INSERT INTO cloud_requests (id, project, lane, mode, purpose, request_payload_json, response_json, status, created_at, completed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET response_json = excluded.response_json, status = excluded.status, completed_at = excluded.completed_at;
                """.strip(),
                (request_key, project, lane, mode, "advice", self._serialize_json(request_payload), self._serialize_json(result), "completed", _now_iso(), _now_iso()),
            )
        return result

    def approve_and_run(self, request_id: str, note: str = "") -> dict[str, Any] | None:
        row = self.get_request(request_id)
        if not row or str(row.get("status", "")) != "pending":
            return None
        result = self.run_query(
            project=str(row.get("project", "general")),
            lane=str(row.get("lane", "project")),
            query=str(row.get("query", "")),
            reason=str(row.get("reason", "")),
            context=str(row.get("context_preview", "")),
            request_id=request_id.strip(),
            note=note,
        )
        return result

    def recent_runs_for_project(self, project: str, limit: int = 8) -> list[dict[str, Any]]:
        key = project.strip()
        limit = max(1, min(limit, 100))
        with connect(self.repo_root) as conn:
            rows = conn.execute(
                "SELECT * FROM cloud_requests WHERE project = ? AND status IN ('completed','resolved','failed') ORDER BY datetime(COALESCE(completed_at, created_at)) DESC, id DESC LIMIT ?;",
                (key, limit),
            ).fetchall()
        items: list[dict[str, Any]] = []
        for row in rows:
            public = self._cloud_row_to_public(row)
            response = self._deserialize_json(row_to_dict(row).get("response_json"))
            public.update({
                "ts": str(row_to_dict(row).get("completed_at") or row_to_dict(row).get("created_at") or ""),
                "provider": str(response.get("provider", public.get("provider", ""))),
                "model": str(response.get("model", public.get("model", ""))),
                "response_path": str(response.get("response_path", public.get("response_path", ""))),
                "response_preview": str(response.get("response_text", ""))[:240],
            })
            items.append(public)
        return items

    def consult_context_for_project(self, project: str, limit: int = 4) -> str:
        logs = self.recent_runs_for_project(project, limit=limit)
        if not logs:
            return ""
        lines = ["Recent cloud consult highlights (use only if relevant):"]
        for row in logs:
            provider = str(row.get("provider", "")).strip()
            model = str(row.get("model", "")).strip()
            preview = str(row.get("response_preview", "")).strip()
            path = str(row.get("response_path", "")).strip()
            if not preview:
                continue
            lines.append(f"- [{provider}:{model}] {preview}")
            if path:
                lines.append(f"  file: {path}")
        return "\n".join(lines)

    def runs_text(self, project: str, limit: int = 10) -> str:
        logs = self.recent_runs_for_project(project, limit=limit)
        if not logs:
            return f"No cloud consult runs yet for project '{project}'."
        lines = [f"Recent cloud consult runs for '{project}' ({len(logs)}):"]
        for row in logs:
            ts = str(row.get("ts", ""))
            provider = str(row.get("provider", ""))
            model = str(row.get("model", ""))
            path = str(row.get("response_path", ""))
            query = str(row.get("query", ""))
            lines.append(f"- [{ts}] provider={provider} model={model} query={query} file={path}")
        return "\n".join(lines)

    def _extract_json_array(self, raw: str) -> list[dict[str, Any]]:
        raw = str(raw or '').strip()
        if not raw:
            return []
        candidates = [raw]
        if '```' in raw:
            for fence in ('```json', '```'):
                if fence in raw:
                    parts = raw.split(fence)
                    for part in parts:
                        part = part.strip()
                        if part and part not in candidates:
                            candidates.append(part)
        left = raw.find('[')
        right = raw.rfind(']')
        if left != -1 and right > left:
            candidates.append(raw[left:right + 1])
        for candidate in candidates:
            try:
                value = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(value, list):
                rows: list[dict[str, Any]] = []
                for item in value:
                    if isinstance(item, dict):
                        rows.append(item)
                return rows
        return []

    def _build_claim_check_prompt(
        self,
        *,
        query: str,
        claims: list[str],
        source_packets: list[dict[str, str]],
        mode_label: str,
    ) -> str:
        claim_lines = "\n".join(f"- {claim[:240]}" for claim in claims[:8]) or "- (no claims supplied)"
        source_lines: list[str] = []
        for idx, packet in enumerate(source_packets[:12], start=1):
            title = str(packet.get("title", "")).strip() or f"Source {idx}"
            domain = str(packet.get("domain", "")).strip()
            excerpt = str(packet.get("excerpt", "")).strip()
            header = f"[{idx}] {title}"
            if domain:
                header += f" ({domain})"
            source_lines.append(header)
            source_lines.append(excerpt[:500] or "(no excerpt)")
        sources_text = "\n".join(source_lines) or "(no source excerpts available)"
        return (
            "You are a narrow claim-checker for Foxforge.\n"
            f"Task mode: {mode_label}.\n"
            "Review ONLY the listed factual claims against ONLY the provided source excerpts.\n"
            "Do not rewrite the answer. Do not critique style, completeness, or tone.\n"
            "For each claim, decide one verdict: supported, contradicted, or insufficient.\n"
            "Use insufficient when the excerpts do not establish the claim clearly.\n"
            "Return a JSON array only, with objects shaped like:\n"
            "[{\"claim\":\"...\",\"verdict\":\"supported|contradicted|insufficient\",\"reason\":\"...\",\"evidence_refs\":[1,2]}]\n"
            "Keep reasons short and cite only excerpt numbers that directly support the verdict.\n\n"
            f"Original request: {query[:500]}\n\n"
            f"Claims:\n{claim_lines}\n\n"
            f"Source excerpts:\n{sources_text}\n"
        )

    def _normalize_claim_checks(self, raw: str) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for item in self._extract_json_array(raw):
            claim = str(item.get('claim', '')).strip()
            verdict = str(item.get('verdict', '')).strip().lower()
            reason = str(item.get('reason', '')).strip()
            refs = item.get('evidence_refs', [])
            if verdict not in {'supported', 'contradicted', 'insufficient'}:
                continue
            ref_ids: list[int] = []
            if isinstance(refs, list):
                for ref in refs[:4]:
                    try:
                        ref_ids.append(int(ref))
                    except (TypeError, ValueError):
                        continue
            if not claim:
                continue
            results.append({
                'claim': claim,
                'verdict': verdict,
                'reason': reason[:300],
                'evidence_refs': ref_ids,
            })
        return results

    def claim_check_research_summary(
        self,
        *,
        project: str,
        query: str,
        sources: list[dict[str, Any]],
        source_path: str,
        claims: list[str] | None = None,
    ) -> dict[str, Any]:
        """Single Gemini claim-check over extracted research claims and source excerpts."""
        settings = self._load_settings()
        if not bool(settings.get('gemini_critique_enabled', False)):
            return {'ok': False, 'claim_checks': [], 'error': 'disabled'}

        keys = self._resolve_gemini_keys(settings)
        if not keys:
            return {'ok': False, 'claim_checks': [], 'error': 'No Gemini API keys configured.'}

        extracted_claims = [str(c).strip() for c in (claims or []) if str(c).strip()]
        if not extracted_claims:
            for s in sources[:8]:
                snippet = str(s.get('snippet', '')).strip()
                if snippet:
                    extracted_claims.append(snippet[:220])
        if not extracted_claims:
            return {'ok': False, 'claim_checks': [], 'error': 'No claims available for checking.'}

        source_packets: list[dict[str, str]] = []
        for s in sources[:12]:
            excerpt = str(s.get('snippet', '')).strip()
            if not excerpt:
                continue
            source_packets.append({
                'title': str(s.get('title', '')).strip(),
                'domain': str(s.get('source_domain', s.get('url', ''))).strip(),
                'excerpt': excerpt,
            })
        if not source_packets:
            return {'ok': False, 'claim_checks': [], 'error': 'No source excerpts available.'}

        model = str(settings.get('gemini_model', 'gemini-2.0-flash'))
        prompt = self._build_claim_check_prompt(
            query=query,
            claims=extracted_claims,
            source_packets=source_packets,
            mode_label='research_summary',
        )
        claim_check_settings = dict(settings)
        claim_check_settings["retry_attempts"] = 1

        last_error = ''
        for idx, key in enumerate(keys):
            try:
                raw, _used_model = self._call_provider_with_retry(
                    "gemini",
                    prompt,
                    claim_check_settings,
                    api_key=key,
                )
                claim_checks = self._normalize_claim_checks(raw)
                self._record_cloud_request(
                    request_id=f'claimcheck_{uuid.uuid4().hex[:12]}',
                    project=project,
                    lane='research',
                    mode='auto',
                    purpose='claim_check',
                    request_payload={
                        'query': query,
                        'source_path': source_path,
                        'claims': extracted_claims[:8],
                        'source_count': len(source_packets),
                    },
                    response_payload={
                        'status': 'completed',
                        'provider': 'gemini',
                        'model': model,
                        'key_index_used': idx,
                        'claim_checks': claim_checks,
                        'updated_at': _now_iso(),
                    },
                    status='completed',
                )
                return {'ok': True, 'claim_checks': claim_checks, 'error': ''}
            except Exception as exc:
                last_error = str(exc)
                continue

        self._record_cloud_request(
            request_id=f'claimcheck_{uuid.uuid4().hex[:12]}',
            project=project,
            lane='research',
            mode='auto',
            purpose='claim_check',
            request_payload={
                'query': query,
                'source_path': source_path,
                'claims': extracted_claims[:8],
                'source_count': len(source_packets),
            },
            response_payload={
                'status': 'failed',
                'error': last_error,
                'updated_at': _now_iso(),
            },
            status='failed',
        )
        return {'ok': False, 'claim_checks': [], 'error': f'Gemini claim-check failed: {last_error}'}

    def get_critique_settings(self) -> dict[str, Any]:
        settings = self._load_settings()
        keys = self._resolve_gemini_keys(settings)
        return {
            "enabled": bool(settings.get("gemini_critique_enabled", False)),
            "api_keys_count": len(keys),
            "model": settings.get("gemini_model", "gemini-2.0-flash"),
        }

    def set_critique_settings(
        self, *, enabled: bool, api_keys: list[str] | None = None
    ) -> dict[str, Any]:
        with self.lock:
            settings = self._load_settings()
            settings["gemini_critique_enabled"] = bool(enabled)
            if api_keys is not None:
                settings["gemini_api_keys"] = [str(k).strip() for k in api_keys if str(k).strip()]
            self._save_settings(settings)
        return self.get_critique_settings()
