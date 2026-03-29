#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

LONG_FILE_LIMIT = 1500
CHECKS = {
    "README.md": ROOT / "README.md",
    "CONTRIBUTING.md": ROOT / "CONTRIBUTING.md",
    "LICENSE": ROOT / "LICENSE",
    "smoke_test.py": ROOT / "smoke_test.py",
}


def scan_long_python_files() -> list[dict[str, int | str]]:
    findings: list[dict[str, int | str]] = []
    for path in sorted((ROOT / "SourceCode").rglob("*.py")):
        try:
            lines = sum(1 for _ in path.open("r", encoding="utf-8", errors="ignore"))
        except OSError:
            continue
        if lines > LONG_FILE_LIMIT:
            findings.append({"file": str(path.relative_to(ROOT)), "lines": lines})
    return findings


def main() -> int:
    report = {
        "required_files": {name: path.exists() for name, path in CHECKS.items()},
        "long_python_files": scan_long_python_files(),
    }
    print(json.dumps(report, indent=2))
    missing = [name for name, ok in report["required_files"].items() if not ok]
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
