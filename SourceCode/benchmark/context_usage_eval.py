from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "SourceCode"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared_tools.context_policy import analyze_query_context


def _load_scenarios(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        return []
    return [row for row in data if isinstance(row, dict)]


def run_context_usage_eval(repo_root: Path) -> dict[str, Any]:
    data_path = repo_root / "SourceCode" / "benchmark" / "data" / "context_usage_scenarios.json"
    scenarios = _load_scenarios(data_path)
    results: list[dict[str, Any]] = []
    passed = 0

    for row in scenarios:
        scenario_id = str(row.get("id", "")).strip() or "scenario"
        query = str(row.get("query", "")).strip()
        expect = row.get("expect", {}) if isinstance(row.get("expect", {}), dict) else {}
        analysis = analyze_query_context(query)
        mismatches: list[str] = []
        for key, expected in expect.items():
            actual = analysis.get(key)
            if actual != expected:
                mismatches.append(f"{key}: expected {expected!r}, got {actual!r}")
        ok = not mismatches
        if ok:
            passed += 1
        results.append(
            {
                "id": scenario_id,
                "ok": ok,
                "query": query,
                "mismatches": mismatches,
            }
        )

    total = len(results)
    return {
        "ok": passed == total,
        "passed": passed,
        "total": total,
        "results": results,
    }


if __name__ == "__main__":
    payload = run_context_usage_eval(ROOT)
    print(json.dumps(payload, indent=2, ensure_ascii=True))
