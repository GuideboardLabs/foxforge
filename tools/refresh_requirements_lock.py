#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "requirements.lock"


def main() -> int:
    proc = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr or "pip freeze failed\n")
        return proc.returncode

    rows = sorted(
        line.strip()
        for line in proc.stdout.splitlines()
        if line.strip() and not line.startswith("#")
    )
    LOCK_PATH.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"Wrote {LOCK_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
