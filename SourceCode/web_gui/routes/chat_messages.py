from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from flask import Blueprint, abort, jsonify, request

from shared_tools.content_guardrails import check_content
from web_gui.chat_helpers import bg_retitle, bg_summarize, handle_command
from web_gui.utils.file_utils import normalize_project_slug as _normalize_project_slug
from web_gui.utils.history_builders import (
    build_command_history as _build_command_history,
    build_fact_history as _build_fact_history,
    build_talk_history as _build_talk_history,
    extract_talk_text as _extract_talk_text,
)

if TYPE_CHECKING:
    from web_gui.app_context import AppContext


def register_message_routes(bp: Blueprint, ctx: AppContext) -> None:
    @bp.route("/api/conversations/<conversation_id>/messages", methods=["POST"])
    def add_message(conversation_id: str) -> tuple[dict, int]:
        profile = ctx.require_profile()
        store = ctx.conversation_store_for(profile)
        convo = store.get(conversation_id)
        if convo is None:
            abort(404, description="Conversation not found")

        requested_mode = ""
        raw_content = ""
        request_id = ""
        attachments: list[dict[str, Any]] = []
        upload_errors: list[str] = []
        reply_to_data: dict | None = None

        content_type = str(request.content_type or "").strip().lower()
        if content_type.startswith("multipart/form-data"):
            raw_content = str(request.form.get("content", "")).strip()
            requested_mode = str(request.form.get("mode", "")).strip().lower()
            request_id = str(request.form.get("request_id", "")).strip()
            attachments, upload_errors = ctx.save_uploaded_images(profile, conversation_id)
        else:
            payload = request.get_json(silent=True) or {}
            raw_content = str(payload.get("content", "")).strip()
            requested_mode = str(payload.get("mode", "")).strip().lower()
            request_id = str(payload.get("request_id", "")).strip()
            _rt = payload.get("reply_to")
            if isinstance(_rt, dict) and str(_rt.get("id", "")).strip():
                reply_to_data = {
                    "id": str(_rt.get("id", "")).strip(),
                    "role": str(_rt.get("role", "")).strip(),
                    "excerpt": str(_rt.get("excerpt", ""))[:300].strip(),
                }

        if not raw_content and not attachments:
            return {"error": "Message content or image attachment is required"}, 400

        talk_text = _extract_talk_text(raw_content)
        is_forage_request = requested_mode == "forage"
        is_talk_request = (requested_mode == "talk" or talk_text is not None) and not is_forage_request
        normalized_talk = (talk_text if talk_text is not None else raw_content).strip()
        if is_talk_request and not normalized_talk and attachments:
            normalized_talk = "Please analyze the attached file(s)."
        stored_user_content = normalized_talk if is_talk_request else raw_content
        if not stored_user_content and attachments:
            n_docs = sum(1 for a in attachments if str(a.get("type", "")) == "document")
            n_imgs = sum(1 for a in attachments if str(a.get("type", "")) == "image")
            parts = []
            if n_imgs:
                parts.append(f"{n_imgs} image(s)")
            if n_docs:
                parts.append(f"{n_docs} document(s)")
            stored_user_content = f"Uploaded {', '.join(parts)}."
        user_mode = "talk" if is_talk_request else "command"
        request_id = ctx.job_manager.start(
            profile=profile,
            conversation_id=conversation_id,
            request_id=request_id,
            mode=user_mode,
            user_text=stored_user_content,
        )

        convo_project = _normalize_project_slug(convo.get("project"))
        if not str(convo.get("project", "")).strip():
            store.set_project(conversation_id, convo_project)
        project_update = None
        pipeline_store = ctx.pipeline_for(profile)
        project_mode = pipeline_store.get(convo_project)

        guard = check_content(raw_content)
        if guard.blocked:
            reply_text = guard.reason
            store.add_message(conversation_id, "assistant", reply_text, mode=user_mode, request_id=request_id)
            ctx.job_manager.finish(profile, request_id, reply=reply_text)
            return jsonify({"reply": reply_text, "request_id": request_id}), 200

        orch = ctx.new_orch(profile)
        if orch.project_slug != convo_project:
            orch.set_project(convo_project)

        command_input_base = raw_content if raw_content else "Please analyze the attached image(s)."
        lane_guess = ""
        is_foraging_request = False
        if is_forage_request:
            lane_guess = "research"
            is_foraging_request = True
        elif not is_talk_request and not raw_content.startswith("/"):
            mode_value = str(project_mode.get("mode", "discovery")).strip().lower()
            target_value = str(project_mode.get("target", "auto")).strip().lower()
            if mode_value == "make":
                lane_guess = f"build:{target_value or 'auto'}"
                is_foraging_request = True
            else:
                try:
                    lane_guess = str(orch.router.route(command_input_base, project_slug=convo_project)).strip().lower()
                except Exception:
                    lane_guess = ""
                is_foraging_request = lane_guess in {"research", "project"}

        user_msg = store.add_message(
            conversation_id,
            "user",
            stored_user_content,
            mode=user_mode,
            attachments=attachments,
            foraging=is_foraging_request,
            request_id=request_id,
            reply_to=reply_to_data,
        )
        if user_msg is None:
            abort(404, description="Conversation not found")

        def _cancel_requested() -> bool:
            return ctx.job_manager.is_cancel_requested(profile, request_id)

        def _progress(stage: str, detail: str = "", *, summary_path: str = "", raw_path: str = "", web_stack: dict | None = None, agent_event: dict | None = None) -> None:
            ctx.job_manager.update(
                profile,
                request_id,
                stage=stage,
                detail=detail,
                summary_path=summary_path,
                raw_path=raw_path,
                web_stack=web_stack,
                agent_event=agent_event,
            )

        def _cancel_reply() -> str:
            row = ctx.job_manager.get(profile, request_id) or {}
            summary = ctx.job_manager.progress_text(row)
            return (
                "Request cancelled.\n"
                "I stopped this active job at the next safe checkpoint.\n\n"
                "Where I left off:\n"
                f"{summary}"
            )

        _progress("message_received", "Message accepted by API and queued for processing.")
        _progress("orchestrator_ready", f"Active project: {convo_project}")
        if is_foraging_request:
            ctx.foraging_manager.register_job(
                profile=profile,
                conversation_id=conversation_id,
                request_id=request_id,
                project=convo_project,
                lane=lane_guess or "project",
                job_key=ctx.job_manager.key(profile, request_id),
            )
            _progress("foraging_started", f"Foraging task started on lane '{lane_guess or 'project'}'.")
        elif ctx.foraging_manager.active_count() > 0:
            ctx.foraging_manager.request_yield(seconds=150.0)
            _progress("foraging_yield_requested", "Foreground chat/cmd requested temporary Foraging yield.")

        image_context = ""
        doc_context = ""
        image_analysis_failures: list[str] = []
        pipeline_error = ""
        try:
            image_attachments = [a for a in attachments if str(a.get("type", "")) == "image"]
            doc_attachments = [a for a in attachments if str(a.get("type", "")) == "document"]

            if image_attachments:
                _progress("attachment_analysis", f"Analyzing {len(image_attachments)} image attachment(s).")
                image_context, image_analysis_failures = ctx.describe_image_attachments(
                    profile=profile,
                    conversation_id=conversation_id,
                    orch=orch,
                    attachments=image_attachments,
                    user_text=normalized_talk if is_talk_request else raw_content,
                )
                if image_context.strip():
                    _progress("attachment_analysis_done", "Image context extracted for prompt assembly.")
                elif image_analysis_failures:
                    _progress("attachment_analysis_done", "Image analysis attempted with failures logged.")

            if doc_attachments:
                _progress("attachment_analysis", f"Extracting text from {len(doc_attachments)} document(s).")
                doc_parts: list[str] = []
                for doc_att in doc_attachments:
                    text = str(doc_att.get("extracted_text", "")).strip()
                    name = str(doc_att.get("name", "document"))
                    warning = str(doc_att.get("extraction_warning", "")).strip()
                    if text:
                        doc_parts.append(f"[Document: {name}]\n{text}")
                    elif warning:
                        doc_parts.append(f"[Document: {name} — {warning}]")
                    else:
                        doc_parts.append(f"[Document: {name} — text could not be extracted]")
                doc_context = "\n\n".join(doc_parts)
                if doc_context:
                    _progress("attachment_analysis_done", "Document text extracted for prompt assembly.")

            if _cancel_requested():
                reply_text = _cancel_reply()
                _progress("cancel_acknowledged", "Cancel request accepted before model execution.")
            else:
                reply_text = ""

            if not reply_text and is_talk_request:
                _progress("talk_mode", "Running conversation-layer reply.")
                talk_input = normalized_talk
                if image_context:
                    talk_input = f"{talk_input}\n\n{image_context}".strip()
                if doc_context:
                    talk_input = f"{talk_input}\n\n{doc_context}".strip()
                if not talk_input:
                    reply_text = "Talk mode message is empty. Send text to continue the conversation."
                else:
                    history = _build_talk_history(convo.get("messages", []), limit_turns=16)
                    capture_history = _build_fact_history(convo.get("messages", []), limit_turns=260)
                    reply_text = orch.conversation_reply(
                        talk_input,
                        history=history,
                        capture_history=capture_history,
                        project=convo_project,
                    )
                _progress("talk_mode_done", "Conversation-layer reply generated.")
            elif not reply_text and raw_content.startswith("/"):
                _progress("command_mode", f"Executing slash command: {raw_content.split(' ', 1)[0]}")
                command_history = _build_command_history(convo.get("messages", []), limit_turns=200)
                fact_history = _build_fact_history(convo.get("messages", []), limit_turns=220)
                history_for_command = fact_history if raw_content.strip().lower() == "/project-facts-refresh" else command_history
                if raw_content.strip().lower() == "/recap":
                    convs = store.list()[:5]
                    lines = ["## Recent Conversations\n"]
                    for row in convs:
                        preview = row.get("summary", "")[:160] or "(no summary yet)"
                        lines.append(f"**{row['title']}** — {row['updated_at'][:10]}\n{preview}\n")
                    reply_text = "\n".join(lines)
                else:
                    reply_text = handle_command(
                        orch,
                        raw_content,
                        command_history=history_for_command,
                        project_mode=project_mode,
                    )
                if raw_content.startswith("/project "):
                    requested = raw_content[len("/project "):].strip()
                    project_update = _normalize_project_slug(requested)
                _progress("command_mode_done", "Slash command execution completed.")
            elif not reply_text:
                _progress("foraging_run", "Running Foraging orchestration.")
                command_input = raw_content if raw_content else "Please analyze the attached file(s)."
                if image_context:
                    command_input = f"{command_input}\n\n{image_context}".strip()
                if doc_context:
                    command_input = f"{command_input}\n\n{doc_context}".strip()
                history = _build_command_history(convo.get("messages", []), limit_turns=18)
                if not orch.project_memory.get_facts(convo_project):
                    orch.refresh_project_facts(history=history, reset=False)
                conversation_summary = store.get_summary(conversation_id) if conversation_id else ""
                reply_text = orch.handle_message(
                    command_input,
                    history=history,
                    project_mode=project_mode,
                    cancel_checker=_cancel_requested,
                    pause_checker=ctx.foraging_manager.is_paused,
                    yield_checker=ctx.foraging_manager.should_yield,
                    conversation_summary=conversation_summary,
                    force_research=is_forage_request,
                    progress_callback=lambda stage, detail=None: _progress(
                        stage,
                        str(detail if not isinstance(detail, dict) else detail.get("note", "") or ""),
                        summary_path=(str(detail.get("summary_path", "")).strip() if isinstance(detail, dict) else ""),
                        raw_path=(str(detail.get("raw_path", "")).strip() if isinstance(detail, dict) else ""),
                        web_stack=(detail if isinstance(detail, dict) and stage == "web_stack_ready" else None),
                        agent_event=(dict(detail, stage=stage) if isinstance(detail, dict) and stage in {"research_pool_started", "research_agent_started", "research_agent_completed"} else None),
                    ),
                )
                _progress("foraging_run_done", "Foraging orchestrator returned final reply.")
        except Exception as exc:
            pipeline_error = str(exc).strip() or "unknown pipeline error"
            _progress("pipeline_error", pipeline_error)
            row = ctx.job_manager.get(profile, request_id) or {}
            progress_summary = ctx.job_manager.progress_text(row)
            if is_foraging_request:
                reply_text = (
                    "Foraging encountered a non-blocking pipeline error after partial progress.\n"
                    "I preserved checkpoints and output paths so you can continue without losing work.\n\n"
                    "Where I left off:\n"
                    f"{progress_summary}\n\n"
                    f"Internal error: {pipeline_error}"
                )
            else:
                reply_text = (
                    "I hit an internal pipeline error while processing this request.\n\n"
                    "Captured progress:\n"
                    f"{progress_summary}\n\n"
                    f"Internal error: {pipeline_error}"
                )
        finally:
            if is_foraging_request:
                ctx.foraging_manager.unregister_job(ctx.job_manager.key(profile, request_id))

        attachment_notes: list[str] = []
        if upload_errors:
            attachment_notes.extend(upload_errors)
        if image_analysis_failures:
            attachment_notes.extend([f"Vision note: {item}" for item in image_analysis_failures[:6]])
        if attachment_notes:
            notes_block = "\n".join([f"- {item}" for item in attachment_notes])
            reply_text = f"{reply_text}\n\nAttachment notes:\n{notes_block}"

        if project_update:
            store.set_project(conversation_id, project=project_update)
        ctx.cache_clear(str(profile.get("id", "")))

        job_row = ctx.job_manager.get(profile, request_id) or {}
        web_stack = job_row.get("web_stack") if isinstance(job_row.get("web_stack"), dict) else {}
        web_sources = [s for s in (web_stack.get("web_sources") or []) if isinstance(s, dict)]
        msg_meta: dict | None = {"web_sources": web_sources} if web_sources else None
        assistant_msg = store.add_message(
            conversation_id,
            "assistant",
            reply_text,
            mode=("talk" if is_talk_request else "command"),
            foraging=is_foraging_request,
            request_id=request_id,
            meta=msg_meta,
        )
        if assistant_msg is None:
            ctx.job_manager.finish(profile, request_id, status="failed", detail="Failed to persist assistant reply.")
            abort(500, description="Failed to persist assistant reply")

        if is_foraging_request and not pipeline_error:
            try:
                from infra.persistence.repositories import ForageCardRepository as _FCR
                import uuid as _uuid

                card_repo = _FCR(ctx.root)
                job_row = ctx.job_manager.get(profile, request_id) or {}
                summary_path = str(job_row.get("summary_path", "") or "").strip()
                raw_path = str(job_row.get("raw_path", "") or "").strip()
                if summary_path:
                    preview = ""
                    for line in reply_text.strip().splitlines():
                        line = line.strip()
                        if line:
                            preview = line[:300]
                            break
                    card_repo.save_card(
                        {
                            "id": f"fc_{request_id[:12]}_{_uuid.uuid4().hex[:4]}",
                            "title": raw_content[:120] if raw_content else "Forage Research",
                            "project": convo_project or "general",
                            "summary_path": summary_path,
                            "raw_path": raw_path,
                            "query": raw_content[:300] if raw_content else "",
                            "preview": preview,
                            "source_count": 0,
                            "is_pinned": 0,
                            "is_read": 0,
                            "created_at": datetime.now(timezone.utc).isoformat(),
                        }
                    )
            except Exception:
                pass

        updated_early = store.get(conversation_id)
        if updated_early:
            msg_count = len(updated_early.get("messages", []))
            root = ctx.root
            if msg_count >= 4 and msg_count % 4 == 0:
                threading.Thread(target=bg_summarize, args=(conversation_id, store, root), daemon=True).start()
            if msg_count == 4:
                first_user = next((m["content"] for m in updated_early["messages"] if m.get("role") == "user"), "")
                from shared_tools.conversation_store import _clean_title as _ct
                if first_user and updated_early.get("title", "") == _ct(first_user):
                    threading.Thread(target=bg_retitle, args=(conversation_id, store, root), daemon=True).start()

        updated = store.get(conversation_id)
        if updated is None:
            ctx.job_manager.finish(profile, request_id, status="failed", detail="Failed to load updated conversation.")
            abort(500, description="Failed to load updated conversation")

        if bool(updated.get("has_unread", False)):
            push_payload, push_event_key = ctx.conversation_notification_payload(
                profile=profile,
                conversation=updated,
                message=assistant_msg,
            )
            ctx.dispatch_web_push(str(profile.get("id", "")).strip(), push_payload, event_key=push_event_key)

        if _cancel_requested():
            job_status = "canceled"
            job_detail = "Message pipeline cancelled by user."
        elif pipeline_error:
            job_status = "completed_with_warnings"
            job_detail = "Message pipeline completed with non-blocking recovery after internal error."
        else:
            job_status = "completed"
            job_detail = "Message pipeline completed."
        ctx.job_manager.finish(profile, request_id, status=job_status, detail=job_detail)

        return {"conversation": updated, "assistant_message": assistant_msg, "request_id": request_id}, 200
