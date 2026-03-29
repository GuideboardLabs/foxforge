from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class TurnPlan:
    """A compact, serializable description of how a user turn should be handled."""

    project: str
    text: str
    lane: str = "research"
    query_mode: str = "chat"
    complexity: str = "simple"
    lane_override: str | None = None
    should_run_foraging: bool = False
    meta: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "project": self.project,
            "text": self.text,
            "lane": self.lane,
            "query_mode": self.query_mode,
            "complexity": self.complexity,
            "lane_override": self.lane_override,
            "should_run_foraging": self.should_run_foraging,
            "meta": dict(self.meta),
        }
