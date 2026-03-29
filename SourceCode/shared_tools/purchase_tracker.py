from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

_HISTORY_REL = Path("Runtime") / "state" / "purchase_history.json"
_LOOKAHEAD_DAYS = 7
_MIN_COMPLETIONS = 2


def _load(root: Path) -> dict[str, Any]:
    path = root / _HISTORY_REL
    if not path.exists():
        return {"items": {}}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"items": {}}


def _save(root: Path, data: dict[str, Any]) -> None:
    path = root / _HISTORY_REL
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def record_completion(root: Path, title: str) -> None:
    title = title.strip()
    if not title:
        return
    key = title.lower()
    today = date.today().isoformat()
    data = _load(root)
    items = data.setdefault("items", {})
    entry = items.setdefault(key, {"title": title, "completions": []})
    if today not in entry["completions"]:
        entry["completions"].append(today)
        entry["completions"].sort()
    _save(root, data)


def get_recommendations(root: Path, lookahead_days: int = _LOOKAHEAD_DAYS) -> list[dict[str, Any]]:
    data = _load(root)
    today = date.today()
    results: list[dict[str, Any]] = []
    for _key, entry in data.get("items", {}).items():
        completions = entry.get("completions", [])
        if len(completions) < _MIN_COMPLETIONS:
            continue
        try:
            dates = sorted(date.fromisoformat(d) for d in completions)
        except ValueError:
            continue
        intervals = [(dates[i + 1] - dates[i]).days for i in range(len(dates) - 1)]
        avg_interval = sum(intervals) / len(intervals)
        predicted_next = dates[-1] + timedelta(days=round(avg_interval))
        if predicted_next <= today + timedelta(days=lookahead_days):
            results.append({
                "title": entry["title"],
                "predicted_date": predicted_next.isoformat(),
                "avg_interval_days": round(avg_interval),
                "last_purchased": dates[-1].isoformat(),
            })
    results.sort(key=lambda x: x["predicted_date"])
    return results
