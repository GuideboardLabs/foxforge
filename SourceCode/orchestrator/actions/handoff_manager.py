"""Handoff queue management — thin wrappers with bus emission."""

from __future__ import annotations


def create_handoff(handoff_queue, bus, project_slug: str, target: str, request_text: str) -> str:
    try:
        item = handoff_queue.create_pending(target=target, request_text=request_text, project_slug=project_slug)
    except ValueError as exc:
        return str(exc)
    bus.emit(
        "orchestrator",
        "handoff_pending_created",
        {"id": item["id"], "target": item["target"], "project": project_slug},
    )
    return (
        f"Handoff request queued as pending: {item['id']} -> {item['target']}. "
        "Use /handoff-pending, then /handoff-approve <id> [reason] or /handoff-deny <id> <reason>."
    )


def handoff_pending_text(handoff_queue) -> str:
    rows = handoff_queue.list_pending()
    if not rows:
        return "No pending handoff requests."
    lines = ["Pending handoff requests:"]
    for row in rows:
        lines.append(
            f"- {row.get('id', '')} | target={row.get('target', '')} | "
            f"project={row.get('project', '')} | text={row.get('request_text', '')}"
        )
    return "\n".join(lines)


def approve_handoff(handoff_queue, bus, request_id: str, reason: str = "") -> str:
    try:
        item = handoff_queue.approve(request_id=request_id, reason=reason, actor="orchestrator")
    except PermissionError as exc:
        return str(exc)
    if item is None:
        return f"Handoff id not found in pending: {request_id}"
    bus.emit(
        "orchestrator",
        "handoff_approved",
        {"id": request_id, "target": item.get("target", ""), "reason": reason},
    )
    out_path = item.get("outbox_path", "")
    return f"Handoff approved: {request_id} -> {item.get('target', '')}. Inbox file: {out_path}"


def deny_handoff(handoff_queue, bus, request_id: str, reason: str) -> str:
    if not reason.strip():
        return "Deny reason is required. Usage: /handoff-deny <id> <reason>"
    try:
        item = handoff_queue.deny(request_id=request_id, reason=reason, actor="orchestrator")
    except PermissionError as exc:
        return str(exc)
    if item is None:
        return f"Handoff id not found in pending: {request_id}"
    bus.emit("orchestrator", "handoff_denied", {"id": request_id, "reason": reason})
    denied_path = item.get("denied_path", "")
    return f"Handoff denied: {request_id}. Recorded at: {denied_path}"


def handoff_inbox_text(handoff_queue, target: str | None = None) -> str:
    handoff_queue.sync_outbox_placeholders()
    monitor_rows = handoff_queue.monitor_threads(limit=500)
    monitor_map = {f"{row.get('target', '')}::{row.get('id', '')}": row for row in monitor_rows}
    targets = [target.lower()] if target else ["codex"]
    lines: list[str] = []
    for key in targets:
        try:
            rows = handoff_queue.list_inbox(key)
        except ValueError as exc:
            return str(exc)
        lines.append(f"{key} inbox ({len(rows)}):")
        if not rows:
            lines.append("- empty")
            continue
        for row in rows[:20]:
            mk = f"{key}::{row.get('id', '')}"
            mrow = monitor_map.get(mk, {})
            mstatus = str(mrow.get("status", "unknown"))
            lines.append(
                f"- {row.get('id', '')} | project={row.get('project', '')} | "
                f"status={mstatus} | text={row.get('request_text', '')}"
            )
    return "\n".join(lines)


def handoff_sync(handoff_queue, bus) -> str:
    result = handoff_queue.sync_outbox_placeholders()
    bus.emit(
        "orchestrator",
        "handoff_placeholder_sync",
        {
            "scanned": result.get("scanned", 0),
            "created": result.get("created", 0),
            "removed_stale": result.get("removed_stale", 0),
        },
    )
    return (
        f"Handoff placeholder sync complete.\n"
        f"Threads scanned: {result.get('scanned', 0)}\n"
        f"New outbox placeholders created: {result.get('created', 0)}\n"
        f"Stale outbox placeholders removed: {result.get('removed_stale', 0)}"
    )


def handoff_monitor_text(handoff_queue, limit: int = 50) -> str:
    return handoff_queue.monitor_text(limit=limit)


def handoff_outbox_text(learning_engine, target: str | None = None, limit: int = 20) -> str:
    return learning_engine.outbox_text(source=target, limit=limit)
