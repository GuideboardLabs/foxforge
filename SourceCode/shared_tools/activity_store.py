import json
from collections import Counter
from pathlib import Path


class ActivityStore:
    def __init__(self, repo_root: Path) -> None:
        self.events_path = repo_root / "Runtime" / "activity" / "events.jsonl"

    def rows(self) -> list[dict]:
        if not self.events_path.exists():
            return []
        out: list[dict] = []
        for line in self.events_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

    def recent_text(self, limit: int = 20) -> str:
        rows = self.rows()
        if not rows:
            return "No activity yet."
        lines = [f"Recent activity (last {min(limit, len(rows))}):"]
        for row in rows[-limit:]:
            ts = row.get("ts", "")
            actor = row.get("actor", "unknown")
            event = row.get("event", "event")
            details = row.get("details", {})
            lines.append(f"- [{ts}] {actor}:{event} {details}")
        return "\n".join(lines)

    def lane_stats_text(self, window: int = 200) -> str:
        rows = self.rows()
        routed = [r for r in rows[-window:] if r.get("event") == "routed"]
        if not routed:
            return "No lane routing events yet."
        counts = Counter((r.get("details") or {}).get("lane", "unknown") for r in routed)
        lines = [f"Lane stats (last {len(routed)} routed events):"]
        for lane, count in counts.most_common():
            lines.append(f"- {lane}: {count}")
        return "\n".join(lines)

    def artifacts_text(self, limit: int = 20) -> str:
        rows = self.rows()
        artifacts: list[str] = []
        keys = {"path", "summary_path", "raw_path", "spec_path"}
        for row in rows:
            details = row.get("details") or {}
            for key in keys:
                value = details.get(key)
                if isinstance(value, str):
                    artifacts.append(value)
        if not artifacts:
            return "No artifacts recorded yet."
        lines = [f"Recent artifacts (last {min(limit, len(artifacts))}):"]
        for path in artifacts[-limit:]:
            lines.append(f"- {path}")
        return "\n".join(lines)

