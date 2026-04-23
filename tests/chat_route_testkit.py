from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class FakeConversationStore:
    def __init__(self, conversation: dict[str, Any]) -> None:
        self._conversation = conversation

    def get(self, conversation_id: str) -> dict[str, Any] | None:
        if str(self._conversation.get("id", "")).strip() != str(conversation_id).strip():
            return None
        return self._conversation

    def set_image_preferences(
        self,
        conversation_id: str,
        *,
        image_style: str | None = None,
        selected_loras: list[str] | None = None,
    ) -> dict[str, Any] | None:
        convo = self.get(conversation_id)
        if convo is None:
            return None
        if image_style is not None:
            convo["image_style"] = str(image_style)
        if selected_loras is not None:
            convo["selected_loras"] = list(selected_loras)
        return convo

    def set_project(self, conversation_id: str, project: str) -> dict[str, Any] | None:
        convo = self.get(conversation_id)
        if convo is None:
            return None
        convo["project"] = str(project)
        return convo

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        **kwargs: Any,
    ) -> dict[str, Any] | None:
        convo = self.get(conversation_id)
        if convo is None:
            return None
        messages = convo.setdefault("messages", [])
        if not isinstance(messages, list):
            return None
        row = {
            "id": f"m{len(messages) + 1}",
            "role": str(role),
            "content": str(content),
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        for key, value in kwargs.items():
            if value is not None:
                row[key] = value
        messages.append(row)
        convo["has_unread"] = False
        return row

    def get_summary(self, _conversation_id: str) -> str:
        return ""


class FakePipelineStore:
    def get(self, _project: str) -> dict[str, str]:
        return {"mode": "discovery", "target": "auto", "topic_type": "general"}

    def set(self, _project: str, **kwargs: Any) -> dict[str, Any]:
        row = {"mode": "discovery", "target": "auto", "topic_type": "general"}
        row.update(kwargs)
        return row


class FakeJobManager:
    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}
        self._next = 1

    def start(
        self,
        *,
        profile: dict[str, Any],
        conversation_id: str,
        request_id: str,
        mode: str,
        user_text: str,
    ) -> str:
        rid = str(request_id).strip() or f"req-{self._next}"
        self._next += 1
        self.rows[rid] = {
            "id": rid,
            "profile_id": str(profile.get("id", "")).strip(),
            "conversation_id": str(conversation_id).strip(),
            "mode": str(mode).strip(),
            "user_text": str(user_text).strip(),
            "events": [],
            "status": "running",
        }
        return rid

    def key(self, profile: dict[str, Any], request_id: str) -> str:
        return f"{str(profile.get('id', '')).strip()}:{str(request_id).strip()}"

    def update(self, profile: dict[str, Any], request_id: str, **kwargs: Any) -> None:
        row = self.get(profile, request_id)
        if row is None:
            return
        stage = str(kwargs.get("stage", "")).strip()
        detail = str(kwargs.get("detail", "")).strip()
        if stage or detail:
            row.setdefault("events", []).append({"stage": stage, "detail": detail})
        if "summary_path" in kwargs:
            row["summary_path"] = str(kwargs.get("summary_path") or "").strip()
        if "raw_path" in kwargs:
            row["raw_path"] = str(kwargs.get("raw_path") or "").strip()
        if isinstance(kwargs.get("web_stack"), dict):
            row["web_stack"] = dict(kwargs["web_stack"])

    def get(self, _profile: dict[str, Any], request_id: str) -> dict[str, Any]:
        return self.rows.setdefault(str(request_id).strip(), {"id": str(request_id).strip(), "events": []})

    def finish(self, profile: dict[str, Any], request_id: str, *, status: str, detail: str) -> None:
        row = self.get(profile, request_id)
        row["status"] = str(status).strip()
        row["detail"] = str(detail).strip()

    def is_cancel_requested(self, _profile: dict[str, Any], _request_id: str) -> bool:
        return False

    def append_live_source(self, _profile: dict[str, Any], _request_id: str, _live_source: dict[str, Any]) -> None:
        return None

    def progress_text(self, _row: dict[str, Any]) -> str:
        return "progress"


class FakeForagingManager:
    def __init__(self) -> None:
        self.register_calls = 0
        self.record_calls = 0

    def register_job(self, **_kwargs: Any) -> None:
        self.register_calls += 1

    def unregister_job(self, _job_key: str) -> None:
        return None

    def active_count(self) -> int:
        return 0

    def request_yield(self, *, seconds: float) -> None:
        _ = seconds

    def is_paused(self) -> bool:
        return False

    def should_yield(self) -> bool:
        return False

    def record_completion(self, **_kwargs: Any) -> None:
        self.record_calls += 1


class FakeBuildingManager:
    def __init__(self) -> None:
        self.register_calls = 0
        self.record_calls = 0

    def register_job(self, **_kwargs: Any) -> None:
        self.register_calls += 1

    def unregister_job(self, _job_key: str) -> None:
        return None

    def record_completion(self, **_kwargs: Any) -> None:
        self.record_calls += 1


class FakeOrch:
    def __init__(self) -> None:
        self.project_slug = "general"
        self.conversation_reply_calls = 0
        self.handle_message_calls = 0
        self.project_memory = type("ProjectMemory", (), {"get_facts": lambda _self, _project: {}})()

    def set_project(self, slug: str) -> None:
        self.project_slug = str(slug)

    def conversation_reply(self, *_args: Any, **_kwargs: Any) -> str:
        self.conversation_reply_calls += 1
        return "Talk-only response."

    def handle_message(self, *_args: Any, **_kwargs: Any) -> str:
        self.handle_message_calls += 1
        return "Unexpected handle_message path."


class FakeTopicEngine:
    def get_topic(self, _topic_id: str) -> None:
        return None


class FakeAppContext:
    def __init__(self, root: Path, store: FakeConversationStore) -> None:
        self.root = root
        self._store = store
        self._profile = {"id": "profile_1"}
        self._pipeline = FakePipelineStore()
        self._topic_engine = FakeTopicEngine()
        self._orch = FakeOrch()
        self.job_manager = FakeJobManager()
        self.foraging_manager = FakeForagingManager()
        self.building_manager = FakeBuildingManager()

    def require_profile(self) -> dict[str, Any]:
        return dict(self._profile)

    def conversation_store_for(self, _profile: dict[str, Any]) -> FakeConversationStore:
        return self._store

    def save_uploaded_images(self, _profile: dict[str, Any], _conversation_id: str) -> tuple[list[dict[str, Any]], list[str]]:
        return [], []

    def pipeline_for(self, _profile: dict[str, Any]) -> FakePipelineStore:
        return self._pipeline

    def get_topic_engine(self) -> FakeTopicEngine:
        return self._topic_engine

    def new_orch(self, _profile: dict[str, Any]) -> FakeOrch:
        return self._orch

    def attachment_dir_for(self, _profile: dict[str, Any], conversation_id: str) -> Path:
        path = self.root / "Runtime" / "attachments" / str(conversation_id)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def repo_root_for_profile(self, _profile: dict[str, Any]) -> Path:
        return self.root

    def describe_image_attachments(self, **_kwargs: Any) -> tuple[str, list[str]]:
        return "", []

    def cache_clear(self, _profile_id: str) -> None:
        return None

    def conversation_notification_payload(self, **_kwargs: Any) -> tuple[dict[str, Any], str]:
        return {}, ""

    def dispatch_web_push(self, _profile_id: str, _payload: dict[str, Any], *, event_key: str = "") -> None:
        _ = event_key

