#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "SourceCode"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.common import ensure_runtime
from shared_tools.cloud_consult import CloudConsultEngine


def _load_live_settings() -> dict:
    settings_path = ROOT / "Runtime" / "cloud" / "settings.json"
    if not settings_path.exists():
        return {}
    try:
        raw = json.loads(settings_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return raw if isinstance(raw, dict) else {}


def main() -> int:
    temp_root = ROOT / "Runtime" / "test_gemini_live_check"
    if temp_root.exists():
        shutil.rmtree(temp_root, ignore_errors=True)
    temp_root.mkdir(parents=True, exist_ok=True)
    ensure_runtime(temp_root)

    live_settings = _load_live_settings()
    has_env_key = bool(str(os.getenv("GEMINI_API_KEY", "")).strip())
    stored_keys = live_settings.get("gemini_api_keys", []) if isinstance(live_settings.get("gemini_api_keys", []), list) else []
    if not stored_keys and not has_env_key:
        print("Gemini live check skipped: no stored or environment API key found.")
        return 2

    temp_settings = {
        "mode": "auto",
        "providers": ["gemini"],
        "gemini_model": str(live_settings.get("gemini_model", "gemini-2.0-flash")).strip() or "gemini-2.0-flash",
        "max_output_chars": 12000,
        "daily_limit": 250,
        "retry_attempts": 1,
        "retry_base_delay_sec": 0.5,
        "retry_max_delay_sec": 1.0,
        "reserve_ratio": 0.2,
        "gemini_critique_enabled": True,
        "gemini_api_keys": [str(key).strip() for key in stored_keys if str(key).strip()],
    }
    settings_path = temp_root / "Runtime" / "cloud" / "settings.json"
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(json.dumps(temp_settings, indent=2, ensure_ascii=True), encoding="utf-8")

    engine = CloudConsultEngine(temp_root)
    try:
        result = engine.claim_check_research_summary(
            project="live_check",
            query="Is a normal resting body temperature for dogs typically higher than for humans?",
            sources=[
                {
                    "title": "Veterinary example excerpt",
                    "source_domain": "example.org",
                    "snippet": "Dogs usually run warmer than humans, with normal temperatures often around 101 to 102.5 F.",
                },
                {
                    "title": "Human baseline excerpt",
                    "source_domain": "example.org",
                    "snippet": "Typical adult human body temperature is commonly cited near 98.6 F, though normal ranges vary.",
                },
            ],
            source_path="Runtime/live_check_sources.md",
            claims=["Dogs usually have a higher normal body temperature than humans."],
        )
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)

    if not result.get("ok"):
        error_text = str(result.get("error", "unknown error"))
        if "429" in error_text or "rate-limit" in error_text.lower() or "too many requests" in error_text.lower():
            print(f"Gemini live check rate-limited: {error_text}")
            return 3
        print(f"Gemini live check failed: {error_text}")
        return 1

    checks = result.get("claim_checks", [])
    first = checks[0] if checks else {}
    print(
        "Gemini live check passed:",
        f"claim_checks={len(checks)}",
        f"first_verdict={first.get('verdict', '')}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
