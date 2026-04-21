from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from shared_tools.conversation_store import ConversationStore
from shared_tools.memory_types import TypedMemoryFacade

_RECALL_RE = re.compile(r"\[RECALL:\s*\"([^\"]{2,240})\"\s*\]", re.IGNORECASE)


def extract_recall_directives(text: str) -> list[str]:
    topics: list[str] = []
    for match in _RECALL_RE.finditer(str(text or "")):
        topic = str(match.group(1) or "").strip()
        if topic and topic not in topics:
            topics.append(topic)
    return topics


def strip_recall_directives(text: str) -> str:
    body = str(text or "")
    body = _RECALL_RE.sub("", body)
    return "\n".join(line.rstrip() for line in body.splitlines()).strip()


class MemoryPager:
    """MemGPT-style pager: working set + archival summary + typed recall."""

    def __init__(
        self,
        repo_root: Path,
        *,
        conversation_store: ConversationStore | None = None,
        typed_memory: TypedMemoryFacade | None = None,
    ) -> None:
        self.repo_root = Path(repo_root)
        self.store = conversation_store or ConversationStore(self.repo_root)
        self.typed_memory = typed_memory or TypedMemoryFacade(self.repo_root)

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        words = max(0, len(str(text or "").split()))
        return max(1, int(words * 1.35))

    def _split_working_set(self, messages: list[dict[str, Any]], max_tokens: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        budget = max(120, int(max_tokens or 0))
        selected: list[dict[str, Any]] = []
        used = 0
        for row in reversed(messages):
            content = str(row.get("content", "")).strip()
            if not content:
                continue
            cost = self._estimate_tokens(content)
            if selected and used + cost > budget:
                break
            selected.append(row)
            used += cost
        selected.reverse()
        selected_ids = {str(r.get("id", "")).strip() for r in selected}
        archived = [row for row in messages if str(row.get("id", "")).strip() not in selected_ids]
        return selected, archived

    def _archive_summary(self, rows: list[dict[str, Any]], max_chars: int = 900) -> str:
        if not rows:
            return ""
        user_chunks: list[str] = []
        assistant_chunks: list[str] = []
        for row in rows[-48:]:
            role = str(row.get("role", "")).strip().lower()
            content = str(row.get("content", "")).strip()
            if not content:
                continue
            compact = " ".join(content.split())[:180]
            if role == "user":
                user_chunks.append(compact)
            elif role == "assistant":
                assistant_chunks.append(compact)
        lines: list[str] = ["Archival context summary:"]
        if user_chunks:
            lines.append("- Earlier user focus: " + " | ".join(user_chunks[-4:]))
        if assistant_chunks:
            lines.append("- Earlier assistant outputs: " + " | ".join(assistant_chunks[-3:]))
        out = "\n".join(lines)
        if len(out) <= max_chars:
            return out
        return out[: max(220, max_chars)].rsplit(" ", 1)[0]

    def _working_context(self, rows: list[dict[str, Any]]) -> str:
        if not rows:
            return ""
        lines = ["Working-set recent turns:"]
        for row in rows[-14:]:
            role = str(row.get("role", "")).strip().upper() or "MSG"
            content = " ".join(str(row.get("content", "")).strip().split())
            if not content:
                continue
            lines.append(f"- {role}: {content[:260]}")
        return "\n".join(lines)

    def _typed_recall_block(self, query: str, *, project: str, conversation_id: str, max_items: int = 2) -> str:
        if not str(query or "").strip():
            return ""
        payload = self.typed_memory.recall(
            query,
            kinds=("semantic", "episodic"),
            k_per_kind=max(1, int(max_items or 1)),
            project=project,
            conversation_id=conversation_id,
        )
        results = payload.get("results", {}) if isinstance(payload.get("results", {}), dict) else {}
        semantic = [dict(x) for x in (results.get("semantic") or []) if isinstance(x, dict)]
        episodic = [dict(x) for x in (results.get("episodic") or []) if isinstance(x, dict)]
        lines: list[str] = []
        if semantic:
            lines.append("Typed memory recall (semantic):")
            for row in semantic[:max_items]:
                key = str(row.get("key", "")).strip().replace("_", " ")
                val = str(row.get("value", "")).strip()
                if val:
                    lines.append(f"- {key}: {val[:180]}")
        if episodic:
            if not lines:
                lines.append("Typed memory recall (episodic):")
            else:
                lines.append("Typed memory recall (episodic):")
            for row in episodic[:max_items]:
                key = str(row.get("key", "")).strip().replace("_", " ")
                val = str(row.get("value", "")).strip()
                if val:
                    lines.append(f"- {key}: {val[:180]}")
        return "\n".join(lines)

    def build_paged_context(
        self,
        conversation_id: str,
        *,
        query: str = "",
        project: str = "general",
        max_working_tokens: int = 1100,
        archive_chars: int = 900,
        recall_items: int = 2,
    ) -> dict[str, Any]:
        convo = self.store.get(conversation_id)
        if not isinstance(convo, dict):
            return {
                "conversation_summary": "",
                "working_turns": 0,
                "archived_turns": 0,
                "recall_directives": [],
            }
        messages = convo.get("messages") if isinstance(convo.get("messages"), list) else []
        working, archived = self._split_working_set(messages, max_working_tokens)
        working_block = self._working_context(working)
        archive_block = self._archive_summary(archived, max_chars=archive_chars)
        recall_topics = extract_recall_directives(query)
        recall_query = recall_topics[0] if recall_topics else query
        recall_block = self._typed_recall_block(
            recall_query,
            project=project,
            conversation_id=conversation_id,
            max_items=recall_items,
        )

        blocks = [b for b in [working_block, archive_block, recall_block] if b.strip()]
        summary = "\n\n".join(blocks).strip()
        if len(summary) > 2600:
            summary = summary[:2600].rsplit("\n", 1)[0].strip()

        return {
            "conversation_summary": summary,
            "working_turns": len(working),
            "archived_turns": len(archived),
            "recall_directives": recall_topics,
        }
