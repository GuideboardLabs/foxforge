#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path


def _parse_iso(raw: str) -> datetime | None:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flag low-confidence make-type decisions for review.")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--hours", type=float, default=24.0, help="Lookback window in hours.")
    parser.add_argument("--min-confidence", type=float, default=0.4)
    parser.add_argument("--max-confidence", type=float, default=0.7)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    src = repo_root / "Runtime" / "telemetry" / "make_type_decisions.jsonl"
    out = repo_root / "Runtime" / "training" / "pending_review.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=max(1.0, float(args.hours)))
    min_conf = float(args.min_confidence)
    max_conf = float(args.max_confidence)

    selected: list[dict] = []
    if src.exists():
        for raw in src.read_text(encoding="utf-8").splitlines():
            line = str(raw or "").strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if not isinstance(row, dict):
                continue
            ts = _parse_iso(str(row.get("ts", "")))
            if ts is not None and ts < cutoff:
                continue
            try:
                conf = float(row.get("classifier_confidence", 0.0) or 0.0)
            except (TypeError, ValueError):
                conf = 0.0
            if conf < min_conf or conf > max_conf:
                continue
            selected.append({
                "ts": str(row.get("ts", "")),
                "text": str(row.get("text", "")),
                "classifier_type": str(row.get("classifier_type", "")),
                "classifier_confidence": conf,
                "llm_type": str(row.get("llm_type", "")),
                "used": str(row.get("used", "")),
            })

    content = "\n".join(json.dumps(row, ensure_ascii=True) for row in selected)
    if content:
        out.write_text(content + "\n", encoding="utf-8")
    else:
        out.write_text("", encoding="utf-8")
    print(f"pending_review={len(selected)} path={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

