from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from infra.tools import ToolRegistry
from .result_types import WorkerResult


@dataclass(slots=True)
class AgentTask:
    lane: str
    prompt: str
    project_slug: str
    repo_root: Path
    history: list[dict[str, str]] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)
    cancel_checker: Callable[[], bool] | None = None
    pause_checker: Callable[[], bool] | None = None
    yield_checker: Callable[[], bool] | None = None
    progress_callback: Callable[[str, dict[str, Any]], None] | None = None


@dataclass(slots=True)
class AgentCapability:
    lane: str
    supports_progress: bool = False
    supports_cancellation: bool = False
    supports_history: bool = False
    produces_artifacts: bool = True
    description: str = ""


class AgentExecutor(Protocol):
    capability: AgentCapability

    def run(self, task: AgentTask, tools: ToolRegistry) -> WorkerResult: ...


class BaseAgentExecutor:
    capability = AgentCapability(lane="unknown")

    def run(self, task: AgentTask, tools: ToolRegistry) -> WorkerResult:
        raise NotImplementedError


__all__ = [
    "AgentCapability",
    "AgentExecutor",
    "AgentTask",
    "BaseAgentExecutor",
]
