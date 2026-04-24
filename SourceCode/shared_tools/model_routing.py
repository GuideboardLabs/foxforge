import json
from pathlib import Path
from typing import Any, Literal


def load_model_routing(repo_root: Path) -> dict[str, Any]:
    primary = repo_root / "SourceCode" / "configs" / "model_routing.json"
    fallback = Path(__file__).resolve().parents[2] / "SourceCode" / "configs" / "model_routing.json"
    for path in (primary, fallback):
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def lane_model_config(repo_root: Path, lane_key: str) -> dict[str, Any]:
    routing = load_model_routing(repo_root)
    value = routing.get(lane_key, {})
    if isinstance(value, dict):
        return value
    return {}


def llama_cpp_server_config(repo_root: Path) -> dict[str, Any]:
    routing = load_model_routing(repo_root)
    servers = routing.get("llama_cpp_servers", {})
    if isinstance(servers, dict):
        return servers
    return {}


def resolved_tier_config(lane_cfg: dict, tier: Literal["default", "premium"]) -> dict[str, Any] | None:
    """Resolve a tier config for a lane with legacy-key fallback.

    - `tier="default"`:
      - uses explicit `tier_default` when present and dict-typed
      - otherwise synthesizes from legacy top-level keys
    - `tier="premium"`:
      - returns explicit `tier_premium` when present and dict-typed
      - returns `None` when absent
    """
    cfg = lane_cfg if isinstance(lane_cfg, dict) else {}
    if tier == "premium":
        premium = cfg.get("tier_premium")
        if isinstance(premium, dict):
            return dict(premium)
        return None

    default_cfg = cfg.get("tier_default")
    if isinstance(default_cfg, dict):
        return dict(default_cfg)

    legacy_keys = (
        "model",
        "num_ctx",
        "temperature",
        "fallback_models",
        "timeout_sec",
        "retry_attempts",
        "retry_backoff_sec",
        "think",
        "num_predict",
        "num_gpu",
        "keep_alive",
    )
    synthesized: dict[str, Any] = {}
    for key in legacy_keys:
        if key in cfg:
            synthesized[key] = cfg.get(key)
    return synthesized
