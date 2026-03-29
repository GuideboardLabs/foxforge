from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class PerfTrace:
    def __init__(self, repo_root: Path, category: str = 'orchestrator') -> None:
        self.repo_root = Path(repo_root)
        self.category = category
        self.starts: dict[str, float] = {}
        self.spans: dict[str, float] = {}
        self.meta: dict[str, Any] = {}
        self.root = self.repo_root / 'Runtime' / 'logs'
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / 'perf_trace.jsonl'

    def start(self, name: str) -> None:
        self.starts[name] = time.perf_counter()

    def end(self, name: str) -> float:
        start = self.starts.pop(name, None)
        if start is None:
            return 0.0
        value = round(time.perf_counter() - start, 4)
        self.spans[name] = value
        return value

    def mark(self, name: str, value: float) -> None:
        self.spans[name] = round(float(value), 4)

    def set_meta(self, **kwargs: Any) -> None:
        self.meta.update(kwargs)

    def snapshot(self) -> dict[str, Any]:
        return {
            'ts': _now_iso(),
            'category': self.category,
            'meta': dict(self.meta),
            'spans': dict(self.spans),
            'total_sec': round(sum(self.spans.values()), 4),
        }

    def write(self) -> dict[str, Any]:
        payload = self.snapshot()
        with self.path.open('a', encoding='utf-8') as fh:
            fh.write(json.dumps(payload, ensure_ascii=True) + '\n')
        return payload
