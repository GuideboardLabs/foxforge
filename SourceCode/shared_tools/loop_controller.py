from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

from shared_tools.activity_bus import telemetry_emit
from shared_tools.inference_router import InferenceRouter
from shared_tools.model_routing import lane_model_config, resolved_tier_config
from shared_tools.premium_model_lock import PremiumModelLock

_IMPORTANCE_RANK = {"low": 0, "medium": 1, "high": 2, "critical": 3}
_LAST_PREMIUM_BY_LANE: dict[str, float] = {}
_LAST_PREMIUM_LOCK = threading.Lock()


@dataclass
class LoopResult:
    final_text: str
    loop_count: int
    tier_used_final: str
    severity_before: dict[str, Any]
    severity_after: dict[str, Any]
    premium_activated: bool
    escalation_reason: str
    critique_logs: list[str]
    warning_banner: str | None


def _default_severity() -> dict[str, Any]:
    return {
        "severity": 2,
        "issues": {},
        "conclusion_vulnerability": "medium",
        "recommended_action": "revise_default",
        "revise_focus": [],
    }


def _normalize_severity(raw: Any) -> dict[str, Any]:
    base = _default_severity()
    if not isinstance(raw, dict):
        return dict(base)
    out = dict(base)
    try:
        sev = int(raw.get("severity", 2))
    except (TypeError, ValueError):
        sev = 2
    out["severity"] = max(0, min(5, sev))
    issues = raw.get("issues")
    out["issues"] = dict(issues) if isinstance(issues, dict) else {}
    vuln = str(raw.get("conclusion_vulnerability", "medium")).strip().lower()
    out["conclusion_vulnerability"] = vuln if vuln in {"low", "medium", "high"} else "medium"
    action = str(raw.get("recommended_action", "revise_default")).strip().lower()
    out["recommended_action"] = action if action in {"accept", "revise_default", "escalate_premium", "reject"} else "revise_default"
    focus = raw.get("revise_focus")
    if isinstance(focus, list):
        out["revise_focus"] = [str(x).strip()[:240] for x in focus if str(x).strip()][:10]
    else:
        out["revise_focus"] = []
    return out


def _importance_meets(actual: str, minimum: str) -> bool:
    return _IMPORTANCE_RANK.get(str(actual or "").strip().lower(), 0) >= _IMPORTANCE_RANK.get(
        str(minimum or "").strip().lower(),
        0,
    )


def should_escalate(
    *,
    importance: str,
    severity: int,
    policy: dict[str, Any],
    has_premium_tier: bool,
    default_passes_done: int,
    lane_key: str,
) -> tuple[bool, str]:
    cfg = policy if isinstance(policy, dict) else {}
    if not bool(cfg.get("enabled", False)):
        return False, "policy_disabled"
    if not has_premium_tier:
        return False, "premium_tier_missing"
    if int(cfg.get("max_premium_passes", 1) or 1) <= 0:
        return False, "premium_passes_disabled"
    if bool(cfg.get("require_prior_default_pass", False)) and int(default_passes_done) <= 0:
        return False, "prior_default_pass_required"
    sev_val = max(0, min(5, int(severity or 0)))
    min_sev = max(0, min(5, int(cfg.get("severity_min", 3) or 3)))
    min_importance = str(cfg.get("importance_min", "high")).strip().lower() or "high"
    if sev_val >= 5 and _importance_meets(importance, "high"):
        return True, "severity_5_high_importance"
    if sev_val < min_sev:
        return False, "severity_below_threshold"
    if not _importance_meets(importance, min_importance):
        return False, "importance_below_threshold"
    cooloff_sec = max(0.0, float(cfg.get("cooloff_sec", 0.0) or 0.0))
    if cooloff_sec > 0:
        with _LAST_PREMIUM_LOCK:
            last = float(_LAST_PREMIUM_BY_LANE.get(lane_key, 0.0) or 0.0)
        if last > 0 and (time.time() - last) < cooloff_sec:
            return False, "cooloff_active"
    return True, "policy_threshold_met"


def _emit_critic_loop(
    repo_root: Path,
    *,
    lane: str,
    task_class: str,
    importance: str,
    tier_used: str,
    severity_before: dict[str, Any],
    severity_after: dict[str, Any],
    loop_count: int,
    premium_activated: bool,
    escalation_reason: str,
    lock_wait_ms: float,
    model_default: str,
    model_premium: str,
    total_elapsed_ms: float,
    phase: str,
) -> None:
    try:
        telemetry_emit(
            repo_root,
            "critic_loops.jsonl",
            {
                "lane": lane,
                "task_class": task_class,
                "importance": importance,
                "tier_used": tier_used,
                "severity_before": int(severity_before.get("severity", 2) or 2),
                "severity_after": int(severity_after.get("severity", 2) or 2),
                "loop_count": int(loop_count),
                "premium_activated": bool(premium_activated),
                "escalation_reason": str(escalation_reason or "").strip(),
                "lock_wait_ms": round(float(lock_wait_ms), 3),
                "model_default": model_default,
                "model_premium": model_premium,
                "total_elapsed_ms": round(float(total_elapsed_ms), 3),
                "phase": str(phase or "").strip(),
            },
            retention_days=30,
        )
    except Exception:
        pass


def _emit_escalation_decision(
    repo_root: Path,
    *,
    lane: str,
    task_class: str,
    importance: str,
    tier_used: str,
    severity_before: dict[str, Any],
    severity_after: dict[str, Any],
    loop_count: int,
    premium_activated: bool,
    escalation_reason: str,
    lock_wait_ms: float,
    model_default: str,
    model_premium: str,
    total_elapsed_ms: float,
) -> None:
    try:
        telemetry_emit(
            repo_root,
            "escalation_decisions.jsonl",
            {
                "lane": lane,
                "task_class": task_class,
                "importance": importance,
                "tier_used": tier_used,
                "severity_before": int(severity_before.get("severity", 2) or 2),
                "severity_after": int(severity_after.get("severity", 2) or 2),
                "loop_count": int(loop_count),
                "premium_activated": bool(premium_activated),
                "escalation_reason": str(escalation_reason or "").strip(),
                "lock_wait_ms": round(float(lock_wait_ms), 3),
                "model_default": model_default,
                "model_premium": model_premium,
                "total_elapsed_ms": round(float(total_elapsed_ms), 3),
            },
            retention_days=30,
        )
    except Exception:
        pass


def _revise_with_focus(
    *,
    repo_root: Path,
    client: InferenceRouter,
    tier_cfg: dict[str, Any],
    text: str,
    revise_focus: list[str],
    lane_key: str,
    importance: str,
    tier_name: str,
    task_class: str,
) -> str:
    model = str(tier_cfg.get("model", "")).strip()
    if not model:
        return text
    focus_lines = "\n".join(f"- {item}" for item in revise_focus if str(item).strip())
    if not focus_lines:
        return text
    try:
        lock = PremiumModelLock(repo_root, client=client)
        lease = None
        if lock.is_premium_model(model):
            lease = lock.acquire(model, timeout_sec=180.0)
        try:
            revised = client.chat(
                model=model,
                fallback_models=tier_cfg.get("fallback_models", []) if isinstance(tier_cfg.get("fallback_models", []), list) else [],
                system_prompt=(
                    "Apply the revision focus bullets to the draft. Preserve structure and useful content. "
                    "Do not add unsupported specifics. Return revised markdown only."
                ),
                user_prompt=(
                    f"Revision focus:\n{focus_lines}\n\n"
                    f"Draft:\n{text}"
                ),
                temperature=float(tier_cfg.get("temperature", 0.2)),
                num_ctx=int(tier_cfg.get("num_ctx", 12288)),
                think=bool(tier_cfg.get("think", False)),
                timeout=int(tier_cfg.get("timeout_sec", 300) or 300),
                retry_attempts=int(tier_cfg.get("retry_attempts", 2) or 2),
                retry_backoff_sec=float(tier_cfg.get("retry_backoff_sec", 1.2) or 1.2),
                keep_alive=str(tier_cfg.get("keep_alive", "10m")),
                task_class=task_class or lane_key,
                artifact_importance=importance,
                tier=tier_name,
            )
        finally:
            if lease is not None:
                try:
                    lock.release(lease, force_unload=True)
                except Exception:
                    pass
        revised_text = str(revised or "").strip()
        if revised_text:
            return revised_text
    except Exception:
        pass
    return text


def run_draft_critique_revise(
    *,
    repo_root: Path,
    lane_key: str,
    draft_fn: Callable[[dict], str],
    critique_fn: Callable[[str, dict], tuple[str, str, dict]],
    importance: Literal["low", "medium", "high", "critical"],
    client: InferenceRouter,
    telemetry_ctx: dict,
    cancel_checker: Callable[[], bool] | None = None,
) -> LoopResult:
    started = time.monotonic()
    lane_cfg = lane_model_config(repo_root, lane_key)
    policy = lane_cfg.get("escalation_policy", {}) if isinstance(lane_cfg.get("escalation_policy", {}), dict) else {}
    default_cfg = resolved_tier_config(lane_cfg, "default") or {}
    premium_cfg = resolved_tier_config(lane_cfg, "premium")
    raw_max_loops = policy.get("max_revise_loops", lane_cfg.get("max_revise_loops", 2))
    try:
        max_revise_loops = max(0, int(2 if raw_max_loops is None else raw_max_loops))
    except (TypeError, ValueError):
        max_revise_loops = 2
    raw_sev_min = policy.get("severity_min", 3)
    try:
        severity_threshold = max(0, min(5, int(3 if raw_sev_min is None else raw_sev_min)))
    except (TypeError, ValueError):
        severity_threshold = 3

    def _is_cancelled() -> bool:
        if callable(cancel_checker):
            try:
                return bool(cancel_checker())
            except Exception:
                return False
        return False

    task_class = str(telemetry_ctx.get("task_class", "")).strip() or lane_key
    model_default = str(default_cfg.get("model", "")).strip()
    model_premium = str((premium_cfg or {}).get("model", "")).strip()
    text = str(draft_fn(default_cfg) or "").strip()
    if _is_cancelled():
        return LoopResult(
            final_text=text,
            loop_count=0,
            tier_used_final="default",
            severity_before=_default_severity(),
            severity_after=_default_severity(),
            premium_activated=False,
            escalation_reason="cancelled",
            critique_logs=[],
            warning_banner=None,
        )

    critique_logs: list[str] = []
    severity_before = _default_severity()
    severity_after = _default_severity()
    max_seen_severity = int(severity_after.get("severity", 2))
    loops_done = 0
    premium_activated = False
    tier_used_final = "default"
    escalation_reason = "default_only"
    lock_wait_ms = 0.0

    for idx in range(max_revise_loops + 1):
        if _is_cancelled():
            escalation_reason = "cancelled"
            break
        default_model = str(default_cfg.get("model", "")).strip()
        default_lock = PremiumModelLock(repo_root, client=client)
        default_lease = None
        if default_lock.is_premium_model(default_model):
            default_lease = default_lock.acquire(default_model, timeout_sec=180.0)
        try:
            revised, critique_log, severity_payload = critique_fn(text, default_cfg)
        finally:
            if default_lease is not None:
                try:
                    default_lock.release(default_lease, force_unload=True)
                except Exception:
                    pass
        severity_payload = _normalize_severity(severity_payload)
        if idx == 0:
            severity_before = dict(severity_payload)
        severity_after = dict(severity_payload)
        max_seen_severity = max(max_seen_severity, int(severity_after.get("severity", 2) or 2))
        if str(critique_log or "").strip():
            critique_logs.append(str(critique_log).strip())
        revised_text = str(revised or "").strip()
        if revised_text:
            text = revised_text
        loops_done += 1

        _emit_critic_loop(
            repo_root,
            lane=lane_key,
            task_class=task_class,
            importance=importance,
            tier_used="default",
            severity_before=severity_before,
            severity_after=severity_after,
            loop_count=loops_done,
            premium_activated=False,
            escalation_reason="default_loop",
            lock_wait_ms=0.0,
            model_default=model_default,
            model_premium=model_premium,
            total_elapsed_ms=(time.monotonic() - started) * 1000.0,
            phase="default",
        )

        if int(severity_after.get("severity", 0) or 0) < severity_threshold:
            escalation_reason = "severity_below_threshold_after_default"
            break
        if idx >= max_revise_loops:
            escalation_reason = "default_loops_exhausted"
            break
        revise_focus = severity_after.get("revise_focus", [])
        if isinstance(revise_focus, list) and revise_focus:
            text = _revise_with_focus(
                repo_root=repo_root,
                client=client,
                tier_cfg=default_cfg,
                text=text,
                revise_focus=revise_focus,
                lane_key=lane_key,
                importance=importance,
                tier_name="default",
                task_class=task_class,
            )

    escalate, decision_reason = should_escalate(
        importance=importance,
        severity=int(severity_after.get("severity", 0) or 0),
        policy=policy,
        has_premium_tier=isinstance(premium_cfg, dict) and bool(str((premium_cfg or {}).get("model", "")).strip()),
        default_passes_done=loops_done,
        lane_key=lane_key,
    )

    if escalate and isinstance(premium_cfg, dict):
        lease = None
        lock = PremiumModelLock(repo_root, client=client)
        premium_model = str(premium_cfg.get("model", "")).strip()
        try:
            lease = lock.acquire(premium_model, timeout_sec=180.0)
            lock_wait_ms = float(getattr(lease, "wait_ms", 0.0) or 0.0)
            revised, critique_log, severity_payload = critique_fn(text, premium_cfg)
            severity_payload = _normalize_severity(severity_payload)
            severity_after = dict(severity_payload)
            max_seen_severity = max(max_seen_severity, int(severity_after.get("severity", 2) or 2))
            if str(critique_log or "").strip():
                critique_logs.append(str(critique_log).strip())
            revised_text = str(revised or "").strip()
            if revised_text:
                text = revised_text
            revise_focus = severity_after.get("revise_focus", [])
            if isinstance(revise_focus, list) and revise_focus:
                text = _revise_with_focus(
                    repo_root=repo_root,
                    client=client,
                    tier_cfg=premium_cfg,
                    text=text,
                    revise_focus=revise_focus,
                    lane_key=lane_key,
                    importance=importance,
                    tier_name="premium",
                    task_class=task_class,
                )
            premium_activated = True
            tier_used_final = "premium"
            escalation_reason = decision_reason
            loops_done += 1
            with _LAST_PREMIUM_LOCK:
                _LAST_PREMIUM_BY_LANE[lane_key] = time.time()
            _emit_critic_loop(
                repo_root,
                lane=lane_key,
                task_class=task_class,
                importance=importance,
                tier_used="premium",
                severity_before=severity_before,
                severity_after=severity_after,
                loop_count=loops_done,
                premium_activated=True,
                escalation_reason=decision_reason,
                lock_wait_ms=lock_wait_ms,
                model_default=model_default,
                model_premium=model_premium,
                total_elapsed_ms=(time.monotonic() - started) * 1000.0,
                phase="premium",
            )
        except Exception:
            escalation_reason = f"{decision_reason}_premium_fallback"
        finally:
            if lease is not None:
                try:
                    lock.release(lease, force_unload=True)
                except Exception:
                    pass
    elif not escalate:
        escalation_reason = decision_reason

    warning_banner = None
    if max_seen_severity >= 5 and _importance_meets(importance, "high"):
        warning_banner = (
            "Critic flagged unusable output; premium consolidation applied — review carefully."
        )

    _emit_escalation_decision(
        repo_root,
        lane=lane_key,
        task_class=task_class,
        importance=importance,
        tier_used=tier_used_final,
        severity_before=severity_before,
        severity_after=severity_after,
        loop_count=loops_done,
        premium_activated=premium_activated,
        escalation_reason=escalation_reason,
        lock_wait_ms=lock_wait_ms,
        model_default=model_default,
        model_premium=model_premium,
        total_elapsed_ms=(time.monotonic() - started) * 1000.0,
    )

    return LoopResult(
        final_text=text,
        loop_count=loops_done,
        tier_used_final=tier_used_final,
        severity_before=severity_before,
        severity_after=severity_after,
        premium_activated=premium_activated,
        escalation_reason=escalation_reason,
        critique_logs=critique_logs,
        warning_banner=warning_banner,
    )
