import json
from pathlib import Path
from typing import Any


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
