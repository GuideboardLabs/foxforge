from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from shared_tools.web_research import build_web_progress_payload
from .agent_contracts import AgentTask


class ResearchService:
    """Encapsulates research-lane orchestration while preserving legacy behavior.

    The orchestrator still owns many helper methods and final formatting rules.
    This service extracts the lane-specific execution flow so `main.py` no longer
    carries the full research control path inline.
    """

    def __init__(self, repo_root: Path, research_pool_runner: Callable[..., dict[str, Any]] | None = None) -> None:
        self.repo_root = repo_root
        self._research_pool_runner = research_pool_runner

    def execute_research_lane(
        self,
        host: Any,
        *,
        text: str,
        history: list[dict[str, str]] | None,
        topic_type: str,
        project_context: str,
        turn_plan: Any,
        force_research: bool,
        cancel_checker=None,
        pause_checker=None,
        yield_checker=None,
        progress_callback=None,
        perf=None,
        reminder_note: str = "",
        event_note: str = "",
        lane: str = "research",
    ) -> str:
        if self._is_cancelled(cancel_checker):
            return "Request cancelled before research execution started."
        full_foraging = force_research or bool(getattr(turn_plan, "should_run_foraging", False))
        if not full_foraging:
            return self._execute_light_research(
                host,
                text=text,
                lane=lane,
                topic_type=topic_type,
                project_context=project_context,
                perf=perf,
                reminder_note=reminder_note,
                event_note=event_note,
            )
        return self._execute_full_research(
            host,
            text=text,
            lane=lane,
            history=history,
            topic_type=topic_type,
            project_context=project_context,
            cancel_checker=cancel_checker,
            pause_checker=pause_checker,
            yield_checker=yield_checker,
            progress_callback=progress_callback,
            reminder_note=reminder_note,
            event_note=event_note,
        )

    def execute_project_lane(
        self,
        host: Any,
        *,
        text: str,
        history: list[dict[str, str]] | None,
        topic_type: str,
        project_context: str,
        cancel_checker=None,
        pause_checker=None,
        yield_checker=None,
        progress_callback=None,
        reminder_note: str = "",
        event_note: str = "",
    ) -> str:
        if self._is_cancelled(cancel_checker):
            return "Request cancelled before project-research execution started."
        web_note, web_context, web_details = host._prepare_web_context(text=text, lane="project", topic_type=topic_type)
        self._emit_web_progress(progress_callback, web_details)
        out = self._run_research_pool(
            text=text,
            host=host,
            history=history,
            topic_type=topic_type,
            project_context=project_context,
            web_context=web_context,
            cancel_checker=cancel_checker,
            pause_checker=pause_checker,
            yield_checker=yield_checker,
            progress_callback=progress_callback,
        )
        if web_details:
            out["web_details"] = web_details
        host._postprocess_research_summary(question=text, worker_result=out, topic_type=topic_type)
        if bool(out.get("canceled", False)):
            return str(
                out.get("cancel_summary")
                or (
                    "Request cancelled during project research. "
                    f"Partial summary saved to {out.get('summary_path', '')}."
                )
            )
        fallback = (
            "I treated this as project strategy and asked the Foraging pool for a baseline synthesis. "
            f"Summary: {out['summary_path']}"
        )
        return self._finalize_research_reply(
            host,
            text=text,
            lane="project",
            out=out,
            fallback=fallback,
            web_note=web_note,
            reminder_note=reminder_note,
            event_note=event_note,
            queue_proposals=True,
        )

    def _execute_light_research(
        self,
        host: Any,
        *,
        text: str,
        lane: str,
        topic_type: str,
        project_context: str,
        perf=None,
        reminder_note: str = "",
        event_note: str = "",
    ) -> str:
        out = host._light_research_flow(
            question=text,
            lane=lane,
            topic_type=topic_type,
            project_context=project_context,
            trace=perf,
        )
        web_details = out.get("web_details", {}) if isinstance(out.get("web_details", {}), dict) else {}
        reply = f"{out['message']} Summary: {out['summary_path']}"
        sources = [dict(x) for x in (web_details.get("sources") or []) if isinstance(x, dict)]
        reply = host._apply_confidence_gate(
            reply,
            sources=sources,
            conflict_summary=web_details.get("conflict_summary", {}),
        )
        artifacts = host._format_research_artifacts_block(out)
        if perf is not None:
            perf.write()
        reply = f"{reply}\n\n{artifacts}"
        reply = host._append_daymarker_note(reply, event_note)
        reply = host._append_daymarker_note(reply, reminder_note)
        return host._complete_turn(user_text=text, lane=lane, reply_text=reply, worker_result=out)

    def _execute_full_research(
        self,
        host: Any,
        *,
        text: str,
        lane: str,
        history: list[dict[str, str]] | None,
        topic_type: str,
        project_context: str,
        cancel_checker=None,
        pause_checker=None,
        yield_checker=None,
        progress_callback=None,
        reminder_note: str = "",
        event_note: str = "",
    ) -> str:
        web_note, web_context, web_details = host._prepare_web_context(text=text, lane=lane, topic_type=topic_type)
        self._emit_web_progress(progress_callback, web_details)
        out = self._run_research_pool(
            text=text,
            host=host,
            history=history,
            topic_type=topic_type,
            project_context=project_context,
            web_context=web_context,
            cancel_checker=cancel_checker,
            pause_checker=pause_checker,
            yield_checker=yield_checker,
            progress_callback=progress_callback,
        )
        if web_details:
            out["web_details"] = web_details
        host._postprocess_research_summary(question=text, worker_result=out, topic_type=topic_type)
        if bool(out.get("canceled", False)):
            return str(
                out.get("cancel_summary")
                or (
                    "Request cancelled during Foraging. "
                    f"Partial summary saved to {out.get('summary_path', '')}."
                )
            )
        fallback = f"{out['message']} Summary: {out['summary_path']}"
        return self._finalize_research_reply(
            host,
            text=text,
            lane=lane,
            out=out,
            fallback=fallback,
            web_note=web_note,
            reminder_note=reminder_note,
            event_note=event_note,
            queue_proposals=True,
        )

    def _run_research_pool(
        self,
        *,
        text: str,
        host: Any,
        history: list[dict[str, str]] | None,
        topic_type: str,
        project_context: str,
        web_context: str,
        cancel_checker=None,
        pause_checker=None,
        yield_checker=None,
        progress_callback=None,
    ) -> dict[str, Any]:
        if hasattr(host, "_run_registered_agent") and hasattr(host, "_make_agent_task"):
            return host._run_registered_agent(
                "research",
                host._make_agent_task(
                    lane="research",
                    text=text,
                    history=history,
                    context={
                        "web_context": web_context,
                        "project_context": project_context,
                        "topic_type": topic_type,
                    },
                    cancel_checker=cancel_checker,
                    pause_checker=pause_checker,
                    yield_checker=yield_checker,
                    progress_callback=progress_callback,
                ),
            )
        if callable(self._research_pool_runner):
            return self._research_pool_runner(
                text,
                self.repo_root,
                host.project_slug,
                host.bus,
                web_context=web_context,
                project_context=project_context,
                prior_messages=history or [],
                cancel_checker=cancel_checker,
                pause_checker=pause_checker,
                yield_checker=yield_checker,
                progress_callback=progress_callback,
                topic_type=topic_type,
            )
        raise RuntimeError("No research pool executor is available.")

    def _finalize_research_reply(
        self,
        host: Any,
        *,
        text: str,
        lane: str,
        out: dict[str, Any],
        fallback: str,
        web_note: str,
        reminder_note: str,
        event_note: str,
        queue_proposals: bool,
    ) -> str:
        if web_note:
            fallback = f"{fallback}\n{web_note}"
        reply = host._orchestrator_finalize(text, lane, out, fallback)
        if web_note and web_note not in reply:
            reply = f"{reply}\n{web_note}"
        if queue_proposals:
            host._queue_action_proposals(reply)
        artifacts = host._format_research_artifacts_block(out)
        reply = f"{reply}\n\n{artifacts}"
        topic_reviews = int(out.get("topic_reviews_created", 0) or 0)
        if topic_reviews > 0:
            reply = f"{reply}\n\n_{topic_reviews} fact(s) queued for Postbag review._"
        reply = host._append_daymarker_note(reply, event_note)
        reply = host._append_daymarker_note(reply, reminder_note)
        return host._complete_turn(user_text=text, lane=lane, reply_text=reply, worker_result=out)

    def _emit_web_progress(self, progress_callback, web_details: dict[str, Any] | None) -> None:
        details = web_details if isinstance(web_details, dict) else {}
        if not details.get("requested") or not callable(progress_callback):
            return
        try:
            progress_callback("web_stack_ready", build_web_progress_payload(details))
        except Exception:
            pass

    @staticmethod
    def _is_cancelled(cancel_checker) -> bool:
        if callable(cancel_checker):
            try:
                return bool(cancel_checker())
            except Exception:
                return False
        return False
