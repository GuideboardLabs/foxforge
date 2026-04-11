from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from infra.tools import ToolRegistry
from .agent_contracts import AgentCapability, AgentTask, BaseAgentExecutor
from .result_types import WorkerResult

from agents_make.app_pool import run_app_pool
from agents_make.content_pool import run_content_pool
from agents_make.creative_pool import run_creative_pool
from agents_make.essay_pool import run_essay_pool
from agents_make.specialist_pool import run_specialist_pool
from agents_research.deep_researcher import run_research_pool
from agents_tool.tool_pool import run_tool_pool
from agents_ui.ui_pool import run_ui_pool
from .image_gen_agent import ImageComposeAgent, ImageGenAgent
from .image_to_video_agent import ImageToVideoAgent
from .image_enhance_agent import ImageEnhanceAgent
from .stable_video_agent import StableVideoAgent


class ResearchPoolAgent(BaseAgentExecutor):
    capability = AgentCapability(
        lane="research",
        supports_progress=True,
        supports_cancellation=True,
        supports_history=True,
        description="Runs the deep research pool against project and web context.",
    )

    def run(self, task: AgentTask, tools: ToolRegistry) -> WorkerResult:
        bus = tools.require("bus")
        result = run_research_pool(
            task.prompt,
            task.repo_root,
            task.project_slug,
            bus,
            web_context=str(task.context.get("web_context", "") or ""),
            project_context=str(task.context.get("project_context", "") or ""),
            prior_messages=task.history,
            cancel_checker=task.cancel_checker,
            pause_checker=task.pause_checker,
            yield_checker=task.yield_checker,
            progress_callback=task.progress_callback,
            topic_type=str(task.context.get("topic_type", "general") or "general"),
        )
        return WorkerResult.from_legacy("research", result)


class AppPoolAgent(BaseAgentExecutor):
    capability = AgentCapability(
        lane="make_app",
        supports_progress=True,
        supports_cancellation=True,
        description="Builds a standalone or web application deliverable.",
    )

    def run(self, task: AgentTask, tools: ToolRegistry) -> WorkerResult:
        bus = tools.require("bus")
        result = run_app_pool(
            question=task.prompt,
            repo_root=task.repo_root,
            project_slug=task.project_slug,
            bus=bus,
            project_context=str(task.context.get("project_context", "") or ""),
            research_context=str(task.context.get("research_context", "") or ""),
            cancel_checker=task.cancel_checker,
            progress_callback=task.progress_callback,
        )
        return WorkerResult.from_legacy("make_app", result)


class EssayPoolAgent(BaseAgentExecutor):
    capability = AgentCapability(
        lane="make_doc",
        supports_progress=True,
        supports_cancellation=True,
        description="Builds essay, brief, and report deliverables.",
    )

    def run(self, task: AgentTask, tools: ToolRegistry) -> WorkerResult:
        bus = tools.require("bus")
        result = run_essay_pool(
            question=task.prompt,
            repo_root=task.repo_root,
            project_slug=task.project_slug,
            bus=bus,
            topic_type=str(task.context.get("topic_type", "general") or "general"),
            target=str(task.context.get("target", "essay") or "essay"),
            research_context=str(task.context.get("research_context", "") or ""),
            raw_notes_context=str(task.context.get("raw_notes_context", "") or ""),
            sources_context=str(task.context.get("sources_context", "") or ""),
            project_context=str(task.context.get("project_context", "") or ""),
            cancel_checker=task.cancel_checker,
            progress_callback=task.progress_callback,
        )
        body = str(result.get("body", "") or "")
        data = dict(result)
        if body and not data.get("artifact_paths"):
            data["artifact_paths"] = []
        return WorkerResult.from_legacy("make_doc", data)


class UiPoolAgent(BaseAgentExecutor):
    capability = AgentCapability(lane="ui", description="Creates UI implementation specs.")

    def run(self, task: AgentTask, tools: ToolRegistry) -> WorkerResult:
        bus = tools.require("bus")
        result = run_ui_pool(task.prompt, task.repo_root, task.project_slug, bus)
        return WorkerResult.from_legacy("ui", result)


class ToolPoolAgent(BaseAgentExecutor):
    capability = AgentCapability(
        lane="make_tool",
        supports_progress=True,
        supports_cancellation=True,
        supports_history=True,
        description="Builds runnable tool scripts with fix loops.",
    )

    def run(self, task: AgentTask, tools: ToolRegistry) -> WorkerResult:
        bus = tools.require("bus")
        result = run_tool_pool(
            question=task.prompt,
            repo_root=task.repo_root,
            project_slug=task.project_slug,
            bus=bus,
            project_context=str(task.context.get("project_context", "") or ""),
            research_context=str(task.context.get("research_context", "") or ""),
            prior_messages=task.history,
            cancel_checker=task.cancel_checker,
            progress_callback=task.progress_callback,
        )
        return WorkerResult.from_legacy("make_tool", result)


class CreativePoolAgent(BaseAgentExecutor):
    capability = AgentCapability(
        lane="make_creative",
        supports_progress=True,
        supports_cancellation=True,
        description="Builds long-form creative writing: novels, memoirs, screenplays.",
    )

    def run(self, task: AgentTask, tools: ToolRegistry) -> WorkerResult:
        bus = tools.require("bus")
        result = run_creative_pool(
            question=task.prompt,
            repo_root=task.repo_root,
            project_slug=task.project_slug,
            bus=bus,
            target=str(task.context.get("target", "novel") or "novel"),
            research_context=str(task.context.get("research_context", "") or ""),
            project_context=str(task.context.get("project_context", "") or ""),
            cancel_checker=task.cancel_checker,
            progress_callback=task.progress_callback,
        )
        return WorkerResult.from_legacy("make_creative", result)


class ContentPoolAgent(BaseAgentExecutor):
    capability = AgentCapability(
        lane="make_content",
        supports_progress=True,
        supports_cancellation=True,
        description="Builds short-form content: blog posts, social media, emails.",
    )

    def run(self, task: AgentTask, tools: ToolRegistry) -> WorkerResult:
        bus = tools.require("bus")
        result = run_content_pool(
            question=task.prompt,
            repo_root=task.repo_root,
            project_slug=task.project_slug,
            bus=bus,
            target=str(task.context.get("target", "blog") or "blog"),
            research_context=str(task.context.get("research_context", "") or ""),
            project_context=str(task.context.get("project_context", "") or ""),
            cancel_checker=task.cancel_checker,
            progress_callback=task.progress_callback,
        )
        return WorkerResult.from_legacy("make_content", result)


class SpecialistPoolAgent(BaseAgentExecutor):
    capability = AgentCapability(
        lane="make_specialist",
        supports_progress=True,
        supports_cancellation=True,
        description="Builds domain-expert deliverables: medical, finance, sports, history, game design.",
    )

    def run(self, task: AgentTask, tools: ToolRegistry) -> WorkerResult:
        bus = tools.require("bus")
        result = run_specialist_pool(
            question=task.prompt,
            repo_root=task.repo_root,
            project_slug=task.project_slug,
            bus=bus,
            topic_type=str(task.context.get("topic_type", "general") or "general"),
            target=str(task.context.get("target", "document") or "document"),
            research_context=str(task.context.get("research_context", "") or ""),
            raw_notes_context=str(task.context.get("raw_notes_context", "") or ""),
            sources_context=str(task.context.get("sources_context", "") or ""),
            project_context=str(task.context.get("project_context", "") or ""),
            cancel_checker=task.cancel_checker,
            progress_callback=task.progress_callback,
        )
        return WorkerResult.from_legacy("make_specialist", result)


class AgentRegistry:
    def __init__(self) -> None:
        self._agents: dict[str, BaseAgentExecutor] = {}

    def register(self, lane: str, executor: BaseAgentExecutor) -> BaseAgentExecutor:
        key = str(lane or "").strip()
        if not key:
            raise ValueError("Agent lane must be non-empty.")
        self._agents[key] = executor
        return executor

    def get(self, lane: str) -> BaseAgentExecutor | None:
        return self._agents.get(str(lane or "").strip())

    def require(self, lane: str) -> BaseAgentExecutor:
        key = str(lane or "").strip()
        agent = self.get(key)
        if agent is None:
            raise KeyError(f"No agent registered for lane '{key}'.")
        return agent

    def lanes(self) -> list[str]:
        return sorted(self._agents.keys())

    def describe(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for lane in self.lanes():
            cap = self._agents[lane].capability
            rows.append(
                {
                    "lane": lane,
                    "supports_progress": cap.supports_progress,
                    "supports_cancellation": cap.supports_cancellation,
                    "supports_history": cap.supports_history,
                    "produces_artifacts": cap.produces_artifacts,
                    "description": cap.description,
                }
            )
        return rows


@dataclass(slots=True)
class OrchestratorRegistries:
    tools: ToolRegistry
    agents: AgentRegistry


def build_default_agent_registry() -> AgentRegistry:
    registry = AgentRegistry()
    registry.register("research", ResearchPoolAgent())
    registry.register("project", ResearchPoolAgent())
    registry.register("make_app", AppPoolAgent())
    registry.register("make_doc", EssayPoolAgent())
    registry.register("make_tool", ToolPoolAgent())
    registry.register("make_creative", CreativePoolAgent())
    registry.register("make_content", ContentPoolAgent())
    registry.register("make_specialist", SpecialistPoolAgent())
    registry.register("ui", UiPoolAgent())
    registry.register("image_gen", ImageGenAgent())
    registry.register("image_gen_compose", ImageComposeAgent())
    registry.register("image_enhance", ImageEnhanceAgent())
    registry.register("video_gen", StableVideoAgent())       # active: SVD XT
    registry.register("video_gen_wan", ImageToVideoAgent())  # dormant: Wan2.1 (requires 16GB+ VRAM)
    return registry


__all__ = [
    "AgentRegistry",
    "AppPoolAgent",
    "BaseAgentExecutor",
    "ContentPoolAgent",
    "CreativePoolAgent",
    "EssayPoolAgent",
    "ImageToVideoAgent",
    "OrchestratorRegistries",
    "ResearchPoolAgent",
    "SpecialistPoolAgent",
    "StableVideoAgent",
    "ToolPoolAgent",
    "UiPoolAgent",
    "build_default_agent_registry",
]
