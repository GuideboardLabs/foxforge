from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from flask import Blueprint, request

from shared_tools.project_pipeline import ProjectPipelineStore
from web_gui.utils.file_utils import (
    normalize_project_slug as _normalize_project_slug,
    safe_path_in_roots as _safe_path_in_roots,
)
from web_gui.utils.request_utils import parse_optional_int

if TYPE_CHECKING:
    from web_gui.app_context import AppContext
    from orchestrator.main import FoxforgeOrchestrator


def _project_panel_rows(ctx: AppContext, orch: FoxforgeOrchestrator, limit: int = 40) -> list[dict]:
    projects_root = orch.repo_root / "Projects"
    project_map: dict[str, dict] = {}
    pipeline_rows = ProjectPipelineStore(orch.repo_root).list_all()
    catalog_rows = ctx.load_project_catalog(orch.repo_root)

    for path in sorted(projects_root.glob("*")):
        if not path.is_dir():
            continue
        slug = path.name
        summary_count = len(list((path / "research_summaries").glob("*.md")))
        implementation_count = len(list((path / "implementation").glob("*.md")))
        plan_count = len(list((path / "plan").glob("*.md")))
        latest_file: Path | None = None
        latest_mtime = 0.0
        for file_path in path.rglob("*.md"):
            try:
                mtime = file_path.stat().st_mtime
            except OSError:
                continue
            if mtime > latest_mtime:
                latest_mtime = mtime
                latest_file = file_path

        updated_at = ""
        if latest_mtime > 0:
            updated_at = datetime.fromtimestamp(latest_mtime, tz=timezone.utc).isoformat()

        project_map[slug] = {
            "project": slug, "source": "filesystem", "updated_at": updated_at,
            "latest_artifact": str(latest_file) if latest_file else "",
            "research_summaries": summary_count, "implementation_specs": implementation_count,
            "plan_docs": plan_count, "event_count": 0, "last_event": "", "last_event_at": "",
            "lane_counts": {}, "handoff_total": 0, "handoff_waiting_output": 0,
            "handoff_ready_for_ingest": 0, "handoff_processed": 0, "mode": "discovery",
            "topic_type": "general", "target": "auto",
            "description": str(catalog_rows.get(slug, {}).get("description", "")).strip(),
        }

    for row in orch.activity_store.rows():
        details = row.get("details") or {}
        project = str(details.get("project", "")).strip()
        if not project:
            continue
        event = str(row.get("event", "")).strip()
        ts = str(row.get("ts", "")).strip()
        lane = str(details.get("lane", "")).strip()

        data = project_map.setdefault(project, {
            "project": project, "source": "activity", "updated_at": ts,
            "latest_artifact": "", "research_summaries": 0, "implementation_specs": 0,
            "plan_docs": 0, "event_count": 0, "last_event": "", "last_event_at": "",
            "lane_counts": {}, "handoff_total": 0, "handoff_waiting_output": 0,
            "handoff_ready_for_ingest": 0, "handoff_processed": 0, "mode": "discovery",
            "topic_type": "general", "target": "auto",
            "description": str(catalog_rows.get(project, {}).get("description", "")).strip(),
        })

        data["event_count"] = int(data.get("event_count", 0)) + 1
        if ts and ts >= str(data.get("last_event_at", "")):
            data["last_event_at"] = ts
            data["last_event"] = event
        if ts and ts >= str(data.get("updated_at", "")):
            data["updated_at"] = ts
        if event == "routed" and lane:
            lc = data.setdefault("lane_counts", {})
            lc[lane] = int(lc.get(lane, 0)) + 1

    for row in orch.handoff_queue.monitor_threads(limit=500):
        project = str(row.get("project", "")).strip()
        if not project:
            continue
        status = str(row.get("status", "")).strip()
        created_at = str(row.get("created_at", "")).strip()
        data = project_map.setdefault(project, {
            "project": project, "source": "handoff", "updated_at": created_at,
            "latest_artifact": "", "research_summaries": 0, "implementation_specs": 0,
            "plan_docs": 0, "event_count": 0, "last_event": "", "last_event_at": "",
            "lane_counts": {}, "handoff_total": 0, "handoff_waiting_output": 0,
            "handoff_ready_for_ingest": 0, "handoff_processed": 0, "mode": "discovery",
            "topic_type": "general", "target": "auto",
            "description": str(catalog_rows.get(project, {}).get("description", "")).strip(),
        })
        data["handoff_total"] = int(data.get("handoff_total", 0)) + 1
        if status == "waiting_output":
            data["handoff_waiting_output"] = int(data.get("handoff_waiting_output", 0)) + 1
        elif status == "ready_for_ingest":
            data["handoff_ready_for_ingest"] = int(data.get("handoff_ready_for_ingest", 0)) + 1
        elif status == "processed":
            data["handoff_processed"] = int(data.get("handoff_processed", 0)) + 1
        if created_at and created_at >= str(data.get("updated_at", "")):
            data["updated_at"] = created_at

    for slug, data in project_map.items():
        mode_row = pipeline_rows.get(slug, {})
        data["mode"] = str(mode_row.get("mode", data.get("mode", "discovery"))).strip() or "discovery"
        data["topic_type"] = str(mode_row.get("topic_type", data.get("topic_type", "general"))).strip() or "general"
        data["target"] = str(mode_row.get("target", data.get("target", "auto"))).strip() or "auto"
        if not str(data.get("description", "")).strip():
            data["description"] = str(catalog_rows.get(slug, {}).get("description", "")).strip()

    for slug, row in catalog_rows.items():
        if slug in project_map:
            continue
        project_map[slug] = {
            "project": slug, "source": "catalog",
            "updated_at": str(row.get("updated_at", "")).strip(),
            "latest_artifact": "", "research_summaries": 0, "implementation_specs": 0,
            "plan_docs": 0, "event_count": 0, "last_event": "", "last_event_at": "",
            "lane_counts": {}, "handoff_total": 0, "handoff_waiting_output": 0,
            "handoff_ready_for_ingest": 0, "handoff_processed": 0, "mode": "discovery",
            "topic_type": "general", "target": "auto",
            "description": str(row.get("description", "")).strip(),
        }

    rows_list = list(project_map.values())
    rows_list.sort(key=lambda r: str(r.get("updated_at", "") or ""), reverse=True)
    return rows_list[: max(1, min(limit, 200))]


def _project_details(ctx: AppContext, orch: FoxforgeOrchestrator, slug: str, event_limit: int = 60, artifact_limit: int = 40) -> dict:
    project = _normalize_project_slug(slug)
    pipeline_store = ProjectPipelineStore(orch.repo_root)
    rows = _project_panel_rows(ctx, orch, limit=500)
    summary = next((x for x in rows if str(x.get("project", "")) == project), None)
    if summary is None:
        summary = {
            "project": project, "source": "none", "updated_at": "", "latest_artifact": "",
            "research_summaries": 0, "implementation_specs": 0, "plan_docs": 0,
            "event_count": 0, "last_event": "", "last_event_at": "", "lane_counts": {},
            "handoff_total": 0, "handoff_waiting_output": 0, "handoff_ready_for_ingest": 0,
            "handoff_processed": 0, "mode": "discovery", "topic_type": "general",
            "target": "auto", "description": "",
        }
    pipeline = pipeline_store.get(project)
    summary["mode"] = str(pipeline.get("mode", summary.get("mode", "discovery"))).strip() or "discovery"
    summary["topic_type"] = str(pipeline.get("topic_type", summary.get("topic_type", "general"))).strip() or "general"
    summary["target"] = str(pipeline.get("target", summary.get("target", "auto"))).strip() or "auto"

    events: list[dict] = []
    artifacts: list[str] = []
    artifact_keys = {"path", "summary_path", "raw_path", "spec_path"}
    for row in orch.activity_store.rows():
        details = row.get("details") or {}
        if str(details.get("project", "")).strip() != project:
            continue
        events.append({"ts": str(row.get("ts", "")), "actor": str(row.get("actor", "")),
                        "event": str(row.get("event", "")), "details": details})
        for key in artifact_keys:
            value = details.get(key)
            if isinstance(value, str) and value.strip():
                artifacts.append(value.strip())

    events = events[-max(1, min(event_limit, 200)):]
    unique_artifacts: list[str] = []
    seen: set[str] = set()
    for path in reversed(artifacts):
        if path in seen:
            continue
        seen.add(path)
        unique_artifacts.append(path)
    unique_artifacts = unique_artifacts[:max(1, min(artifact_limit, 200))]

    handoffs = sorted(
        [row for row in orch.handoff_queue.monitor_threads(limit=500) if str(row.get("project", "")).strip() == project],
        key=lambda item: str(item.get("created_at", "")), reverse=True,
    )
    return {"project": project, "summary": summary, "events": events,
            "artifacts": unique_artifacts, "handoffs": handoffs, "pipeline": pipeline}


def create_projects_blueprint(ctx: AppContext) -> Blueprint:
    bp = Blueprint('project_routes', __name__)

    @bp.route('/api/action-proposals', methods=['GET'])
    def list_action_proposals() -> tuple[dict, int]:
        profile = ctx.require_profile()
        limit = parse_optional_int(request.args.get("limit"), default=50, minimum=1, maximum=200)
        orch = ctx.new_orch(profile)
        proposals = orch.approval_gate.list_action_proposals(limit=limit)
        return {"proposals": proposals}, 200

    @bp.route('/api/action-proposals/<proposal_id>/approve', methods=['POST'])
    def approve_action_proposal(proposal_id: str) -> tuple[dict, int]:
        profile = ctx.require_profile()
        orch = ctx.new_orch(profile)
        result = orch.approval_gate.execute_proposal(proposal_id, ctx.root)
        status = 200 if result.get("ok") else 400
        return {"ok": result.get("ok", False), "message": result.get("message", "")}, status

    @bp.route('/api/action-proposals/<proposal_id>/reject', methods=['POST'])
    def reject_action_proposal(proposal_id: str) -> tuple[dict, int]:
        profile = ctx.require_profile()
        orch = ctx.new_orch(profile)
        ok = orch.approval_gate.decide(proposal_id, approved=False)
        return {"ok": ok}, 200

    @bp.route('/api/projects', methods=['GET'])
    def projects_index() -> tuple[dict, int]:
        profile = ctx.require_profile()
        limit = parse_optional_int(request.args.get("limit"), default=100, minimum=1, maximum=500)
        orch = ctx.new_orch(profile)
        rows = _project_panel_rows(ctx, orch, limit=limit)
        project_names = [str(row.get("project", "")) for row in rows if str(row.get("project", "")).strip()]
        return {"projects": project_names, "rows": rows}, 200

    @bp.route('/api/projects/catalog', methods=['POST'])
    def projects_catalog_upsert() -> tuple[dict, int]:
        profile = ctx.require_profile()
        payload = request.get_json(silent=True) or {}
        raw_project = str(payload.get("project", "")).strip()
        if not raw_project:
            return {"ok": False, "message": "Project name is required."}, 400
        project = _normalize_project_slug(raw_project)
        description = str(payload.get("description", "")).strip()
        project_dir = ctx.root / "Projects" / project
        project_dir.mkdir(parents=True, exist_ok=True)
        row = ctx.set_project_catalog_entry(ctx.root, project=project, description=description)
        ctx.cache_clear(str(profile.get("id", "")))
        return {"ok": True, "project": project, "description": str(row.get("description", "")).strip()}, 200

    @bp.route('/api/projects/<project_slug>/details', methods=['GET'])
    def project_details(project_slug: str) -> tuple[dict, int]:
        profile = ctx.require_profile()
        event_limit = parse_optional_int(request.args.get("events"), default=60, minimum=1, maximum=200)
        artifact_limit = parse_optional_int(request.args.get("artifacts"), default=40, minimum=1, maximum=200)
        orch = ctx.new_orch(profile)
        payload = _project_details(ctx, orch, project_slug, event_limit=event_limit, artifact_limit=artifact_limit)
        return payload, 200

    @bp.route('/api/projects/<project_slug>/content-tree', methods=['GET'])
    def project_content_tree(project_slug: str) -> tuple[dict, int]:
        profile = ctx.require_profile()
        project = _normalize_project_slug(project_slug)
        depth = parse_optional_int(request.args.get("depth"), default=5, minimum=1, maximum=8)
        node_limit = parse_optional_int(request.args.get("nodes"), default=1200, minimum=100, maximum=4000)
        repo_root = ctx.repo_root_for_profile(profile)
        projects_root = repo_root / "Projects"

        allowed_roots, denied_roots = ctx.file_roots_for(profile)
        safe_projects_root = _safe_path_in_roots(
            str(projects_root), allowed_roots=allowed_roots, denied_roots=denied_roots, must_exist=False,
        )
        if safe_projects_root is None:
            return {"ok": False, "error": "Invalid project root."}, 400

        project_root = projects_root / project
        candidates: list[tuple[str, Path]] = []
        if project_root.exists() and project_root.is_dir():
            candidates.append((project, project_root))
        if projects_root.exists() and projects_root.is_dir():
            try:
                bucket_dirs = sorted([r for r in projects_root.iterdir() if r.is_dir()], key=lambda r: r.name.lower())
            except OSError:
                bucket_dirs = []
            for bucket in bucket_dirs:
                if bucket.name.lower() == project.lower():
                    continue
                candidate = bucket / project
                if candidate.exists() and candidate.is_dir():
                    candidates.append((f"{bucket.name}/{project}", candidate))

        if not candidates:
            return {"ok": True, "project": project, "root": str(safe_projects_root),
                    "tree": [], "node_count": 0, "truncated": False}, 200

        node_count = 0
        truncated = False
        safe_projects_root_resolved = safe_projects_root.resolve()
        skip_names = {".git", "__pycache__", ".pytest_cache", ".mypy_cache"}

        def _walk_dir(dir_path: Path, root_resolved: Path, level: int) -> list[dict[str, Any]]:
            nonlocal node_count, truncated
            if level >= depth or truncated:
                return []
            try:
                entries = list(dir_path.iterdir())
            except OSError:
                return []
            entries.sort(key=lambda item: (item.is_file(), item.name.lower()))
            rows: list[dict[str, Any]] = []
            for entry in entries:
                if truncated or node_count >= node_limit:
                    truncated = True
                    break
                if entry.name in skip_names:
                    continue
                try:
                    resolved = entry.resolve()
                    rel_path = str(resolved.relative_to(root_resolved)).replace("\\", "/")
                except (OSError, ValueError):
                    continue
                if entry.is_dir():
                    node_count += 1
                    children = _walk_dir(entry, root_resolved, level + 1)
                    rows.append({"type": "dir", "name": entry.name, "path": str(resolved),
                                 "rel_path": rel_path, "children": children, "child_count": len(children)})
                elif entry.is_file():
                    node_count += 1
                    try:
                        size = int(entry.stat().st_size)
                    except OSError:
                        size = 0
                    rows.append({"type": "file", "name": entry.name, "path": str(resolved),
                                 "rel_path": rel_path, "ext": entry.suffix.lower(), "size": size})
            return rows

        tree: list[dict[str, Any]] = []
        for label, root_path in candidates:
            if truncated or node_count >= node_limit:
                truncated = True
                break
            try:
                root_resolved = root_path.resolve()
                root_resolved.relative_to(safe_projects_root_resolved)
            except (OSError, ValueError):
                continue
            node_count += 1
            children = _walk_dir(root_path, root_resolved, 0)
            tree.append({"type": "dir", "name": label, "path": str(root_resolved),
                         "rel_path": label.replace("\\", "/"), "children": children,
                         "child_count": len(children), "group_root": True})

        return {"ok": True, "project": project, "root": str(safe_projects_root_resolved),
                "depth": depth, "node_count": node_count, "truncated": truncated, "tree": tree}, 200

    @bp.route('/api/projects/<project_slug>/mode', methods=['GET'])
    def project_mode_get(project_slug: str) -> tuple[dict, int]:
        profile = ctx.require_profile()
        store = ctx.pipeline_for(profile)
        project = _normalize_project_slug(project_slug)
        return store.get(project), 200

    @bp.route('/api/projects/<project_slug>/mode', methods=['POST'])
    def project_mode_set(project_slug: str) -> tuple[dict, int]:
        profile = ctx.require_profile()
        payload = request.get_json(silent=True) or {}
        store = ctx.pipeline_for(profile)
        project = _normalize_project_slug(project_slug)
        row = store.set(project, mode=payload.get("mode"), target=payload.get("target"),
                        topic_type=payload.get("topic_type"))
        ctx.cache_clear(str(profile.get("id", "")))
        return row, 200

    @bp.route('/api/topics', methods=['GET'])
    def topics_list() -> tuple[dict, int]:
        ctx.require_profile()
        engine = ctx.get_topic_engine()
        parent_id = str(request.args.get("parent_id", "")).strip()
        topics = engine.list_topics(parent_id=parent_id)
        for topic in topics:
            slug = str(topic.get("slug", "")).strip()
            if slug:
                project_dir = ctx.root / "Projects" / slug
                summary_dir = project_dir / "research_summaries"
                impl_dir = project_dir / "implementation"
                topic["research_summaries"] = len(list(summary_dir.glob("*.md"))) if summary_dir.exists() else 0
                topic["implementation_specs"] = len(list(impl_dir.glob("*.md"))) if impl_dir.exists() else 0
                latest_mtime = 0.0
                if project_dir.exists():
                    for fp in project_dir.rglob("*.md"):
                        try:
                            mtime = fp.stat().st_mtime
                            if mtime > latest_mtime:
                                latest_mtime = mtime
                        except OSError:
                            pass
                topic["last_research"] = datetime.fromtimestamp(latest_mtime, tz=timezone.utc).isoformat() if latest_mtime > 0 else ""
            else:
                topic["research_summaries"] = 0
                topic["implementation_specs"] = 0
                topic["last_research"] = ""
        return {"topics": topics}, 200

    @bp.route('/api/topics', methods=['POST'])
    def topics_create() -> tuple[dict, int]:
        ctx.require_profile()
        engine = ctx.get_topic_engine()
        payload = request.get_json(silent=True) or {}
        try:
            topic = engine.create_topic(
                name=str(payload.get("name", "")).strip(), type=str(payload.get("type", "")).strip(),
                description=str(payload.get("description", "")).strip(),
                seed_question=str(payload.get("seed_question", "")).strip(),
                parent_id=str(payload.get("parent_id", "")).strip(),
            )
        except ValueError as exc:
            return {"error": str(exc)}, 400
        return topic, 201

    @bp.route('/api/topics/<topic_id>', methods=['GET'])
    def topics_get(topic_id: str) -> tuple[dict, int]:
        ctx.require_profile()
        engine = ctx.get_topic_engine()
        topic = engine.get_topic(topic_id)
        if topic is None:
            return {"error": "Topic not found."}, 404
        return topic, 200

    @bp.route('/api/topics/<topic_id>', methods=['PUT'])
    def topics_update(topic_id: str) -> tuple[dict, int]:
        ctx.require_profile()
        engine = ctx.get_topic_engine()
        payload = request.get_json(silent=True) or {}
        updated = engine.update_topic(topic_id, **{k: v for k, v in payload.items()})
        if updated is None:
            return {"error": "Topic not found."}, 404
        return updated, 200

    @bp.route('/api/topics/<topic_id>', methods=['DELETE'])
    def topics_delete(topic_id: str) -> tuple[dict, int]:
        ctx.require_profile()
        engine = ctx.get_topic_engine()
        ok = engine.delete_topic(topic_id)
        if not ok:
            return {"error": "Topic not found."}, 404
        return {"deleted": topic_id}, 200

    @bp.route('/api/topics/<topic_id>/detail', methods=['GET'])
    def topics_detail_full(topic_id: str) -> tuple[dict, int]:
        ctx.require_profile()
        engine = ctx.get_topic_engine()
        topic = engine.get_topic(topic_id)
        if topic is None:
            return {"error": "Topic not found."}, 404
        slug = str(topic.get("slug", "")).strip()
        artifacts: list[dict] = []
        if slug:
            project_dir = ctx.root / "Projects" / slug
            for sub in ["research_raw", "research_summaries", "implementation"]:
                subdir = project_dir / sub
                if subdir.exists():
                    for fp in sorted(subdir.glob("*.md"), key=lambda x: x.stat().st_mtime, reverse=True)[:20]:
                        artifacts.append({"path": str(fp), "folder": sub, "name": fp.name})
        result = dict(topic)
        result["artifacts"] = artifacts
        result["subtopics"] = engine.list_topics(parent_id=topic_id)
        return result, 200

    @bp.route('/api/topics/<topic_id>/subtopics', methods=['GET'])
    def topics_subtopics_list(topic_id: str) -> tuple[dict, int]:
        ctx.require_profile()
        engine = ctx.get_topic_engine()
        return {"topics": engine.list_topics(parent_id=topic_id)}, 200

    @bp.route('/api/topics/<topic_id>/subtopics', methods=['POST'])
    def topics_subtopics_create(topic_id: str) -> tuple[dict, int]:
        ctx.require_profile()
        engine = ctx.get_topic_engine()
        payload = request.get_json(silent=True) or {}
        try:
            topic = engine.create_topic(
                name=str(payload.get("name", "")).strip(), type=str(payload.get("type", "")).strip(),
                description=str(payload.get("description", "")).strip(),
                seed_question=str(payload.get("seed_question", "")).strip(),
                parent_id=topic_id,
            )
        except ValueError as exc:
            return {"error": str(exc)}, 400
        return topic, 201

    return bp
