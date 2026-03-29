from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(slots=True)
class ToolSpec:
    name: str
    value: Any
    description: str = ""
    tags: tuple[str, ...] = ()


class ToolRegistry:
    """Simple runtime registry for shared tools and services.

    The registry provides a single dependency lookup surface so orchestrator
    services and agent adapters do not need to import or construct shared tools
    ad hoc. It intentionally stores already-built objects/callables.
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, name: str, value: Any, *, description: str = "", tags: tuple[str, ...] = ()) -> Any:
        key = str(name or "").strip()
        if not key:
            raise ValueError("Tool name must be non-empty.")
        self._tools[key] = ToolSpec(name=key, value=value, description=description, tags=tuple(tags))
        return value

    def get(self, name: str, default: Any | None = None) -> Any:
        spec = self._tools.get(str(name or "").strip())
        return spec.value if spec else default

    def require(self, name: str) -> Any:
        key = str(name or "").strip()
        spec = self._tools.get(key)
        if spec is None:
            raise KeyError(f"Tool '{key}' is not registered.")
        return spec.value

    def unregister(self, name: str) -> None:
        self._tools.pop(str(name or "").strip(), None)

    def has(self, name: str) -> bool:
        return str(name or "").strip() in self._tools

    def names(self) -> list[str]:
        return sorted(self._tools.keys())

    def describe(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for name in self.names():
            spec = self._tools[name]
            rows.append(
                {
                    "name": spec.name,
                    "description": spec.description,
                    "tags": list(spec.tags),
                    "type": type(spec.value).__name__,
                }
            )
        return rows

    def clone(self) -> "ToolRegistry":
        other = ToolRegistry()
        for name in self.names():
            spec = self._tools[name]
            other.register(spec.name, spec.value, description=spec.description, tags=spec.tags)
        return other


__all__ = ["ToolRegistry", "ToolSpec"]
