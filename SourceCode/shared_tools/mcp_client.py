from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Any


_LOCK = Lock()
_CACHE: dict[str, Any] = {}


def _config_path(repo_root: Path) -> Path:
    return Path(repo_root) / "SourceCode" / "configs" / "mcp_servers.json"


def load_mcp_servers(repo_root: Path) -> list[dict[str, Any]]:
    path = _config_path(repo_root)
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return []
    if not isinstance(payload, dict):
        return []
    rows = payload.get("servers") if isinstance(payload.get("servers"), list) else []
    out: list[dict[str, Any]] = []
    for row in rows:
        if isinstance(row, dict):
            out.append(dict(row))
    return out


class MCPClientPool:
    """Lazy MCP client registry.

    The full MCP client protocol is optional in this runtime. This class keeps
    config-parsing and enable/disable behavior centralized so callsites can
    safely attempt MCP usage and transparently fall back.
    """

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = Path(repo_root)
        self.servers = load_mcp_servers(self.repo_root)

    def fetch_server(self) -> dict[str, Any] | None:
        for row in self.servers:
            name = str(row.get("name", "")).strip().lower()
            if name == "fetch" and bool(row.get("enabled", False)):
                return row
        return None

    def fetch_url(self, url: str, *, timeout_sec: int = 20) -> bytes | None:
        _server = self.fetch_server()
        if _server is None:
            return None
        # Placeholder for full MCP transport integration.
        # Returning None means caller should use native HTTP fallback.
        _ = timeout_sec
        _ = url
        return None


def pool_for(repo_root: Path) -> MCPClientPool:
    key = str(Path(repo_root).resolve())
    with _LOCK:
        cached = _CACHE.get(key)
        if isinstance(cached, MCPClientPool):
            return cached
        client = MCPClientPool(Path(repo_root))
        _CACHE[key] = client
        return client


def mcp_fetch_url(repo_root: Path, url: str, *, timeout_sec: int = 20) -> bytes | None:
    client = pool_for(repo_root)
    return client.fetch_url(url, timeout_sec=timeout_sec)

