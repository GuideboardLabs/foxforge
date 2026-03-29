#!/usr/bin/env python3
from __future__ import annotations

import importlib
import json
import os
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

PASS = "PASS"
FAIL = "FAIL"


def _print(status: str, message: str) -> None:
    print(f"[{status}] {message}")


def _ignore_filter(_src: str, names: list[str]) -> set[str]:
    ignored: set[str] = set()
    bulky = {
        ".git",
        ".venv",
        "node_modules",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "Archive",
    }
    for name in names:
        if name in bulky:
            ignored.add(name)
    return ignored


def _ensure_json(path: Path, default) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(json.dumps(default, indent=2), encoding="utf-8")


def _prepare_runtime(repo_root: Path) -> None:
    _ensure_json(repo_root / "Runtime" / "watchtower" / "watches.json", [])
    _ensure_json(repo_root / "Runtime" / "watchtower" / "briefing_state.json", {})
    _ensure_json(repo_root / "Runtime" / "routines" / "routines.json", [])
    _ensure_json(repo_root / "Runtime" / "topics" / "topics.json", [])
    _ensure_json(repo_root / "Runtime" / "project_catalog.json", [])
    _ensure_json(repo_root / "Runtime" / "web" / "settings.json", {"mode": "off", "provider": "auto"})
    (repo_root / "Runtime" / "briefings").mkdir(parents=True, exist_ok=True)
    (repo_root / "Runtime" / "conversations").mkdir(parents=True, exist_ok=True)


class FakeOrchestrator:
    def web_mode_text(self) -> str:
        return "MODE_MARKER"

    def web_provider_text(self) -> str:
        return "PROVIDER_MARKER"

    def set_web_mode(self, mode: str) -> str:
        return f"SET_MODE:{mode}"

    def set_web_provider(self, provider: str) -> str:
        return f"SET_PROVIDER:{provider}"


def main() -> int:
    source_repo = Path(__file__).resolve().parent
    failures = 0

    with tempfile.TemporaryDirectory(prefix="foxforge_smoke_") as tmpdir:
        tmp_repo = Path(tmpdir) / "Foxforge"
        shutil.copytree(source_repo, tmp_repo, ignore=_ignore_filter)
        _prepare_runtime(tmp_repo)

        os.environ.setdefault("FOXFORGE_OWNER_PASSWORD", "smoke-test-password")
        os.environ.setdefault("FOXFORGE_AUTH_ENABLED", "0")

        sys.path.insert(0, str(tmp_repo / "SourceCode"))
        sys.path.insert(0, str(tmp_repo))

        try:
            appmod = importlib.import_module("web_gui.app")
            _print(PASS, "Imported web_gui.app")
        except Exception:
            _print(FAIL, "Importing web_gui.app failed")
            traceback.print_exc()
            return 1

        try:
            wt = getattr(appmod, "_watchtower", None)
            rt = getattr(appmod, "_routine_engine", None)
            wt_alive = bool(wt and getattr(wt, "_thread", None) and wt._thread.is_alive())
            rt_alive = bool(rt and getattr(rt, "_thread", None) and rt._thread.is_alive())
            assert not wt_alive and not rt_alive
            _print(PASS, "Background services are not started at import time")
        except Exception:
            failures += 1
            _print(FAIL, "Background services started too early")
            traceback.print_exc()

        try:
            chat_helpers = importlib.import_module("web_gui.chat_helpers")
            handle_command = chat_helpers.handle_command
            fake = FakeOrchestrator()
            provider_text = handle_command(fake, "/web-provider")
            mode_text = handle_command(fake, "/web-mode")
            assert provider_text == "PROVIDER_MARKER"
            assert mode_text == "MODE_MARKER"
            assert handle_command(fake, "/web-provider auto") == "SET_PROVIDER:auto"
            assert handle_command(fake, "/web-mode ask") == "SET_MODE:ask"
            _print(PASS, "Command routing for /web-provider and /web-mode is correct")
        except Exception:
            failures += 1
            _print(FAIL, "Command routing regression detected")
            traceback.print_exc()

        try:
            app = appmod.create_app()
            _print(PASS, "create_app() returned a Flask app")
        except Exception:
            failures += 1
            _print(FAIL, "create_app() failed")
            traceback.print_exc()
            return 1

        endpoints = [
            "/",
            "/api/health",
            "/api/auth/status",
            "/api/projects",
            "/api/topics",
            "/api/routines",
            "/api/watchtower/watches",
            "/api/panel/status",
            "/api/foraging/state",
        ]

        try:
            with app.test_client() as client:
                for endpoint in endpoints:
                    response = client.get(endpoint)
                    assert response.status_code < 500, f"{endpoint} returned {response.status_code}"
                _print(PASS, "Core routes returned non-5xx responses")

                health_resp = client.get("/api/health")
                health_data = health_resp.get_json()
                assert health_data.get("ok") is True, "/api/health missing ok=True"
                assert "checks" in health_data, "/api/health missing checks payload"
                assert "status" in health_data, "/api/health missing status field"
                _print(PASS, "/api/health returns structured checks payload")
        except Exception:
            failures += 1
            _print(FAIL, "One or more core routes failed")
            traceback.print_exc()

        try:
            wt = appmod._get_watchtower()
            assert wt._thread is not None and wt._thread.is_alive()
            first_wt_ident = wt._thread.ident
            appmod._ensure_background_services_started(app)
            assert wt._thread.ident == first_wt_ident
            _print(PASS, "Background service startup is lazy and idempotent")
        except Exception:
            failures += 1
            _print(FAIL, "Background service startup/idempotency check failed")
            traceback.print_exc()

    if failures:
        _print(FAIL, f"Smoke test completed with {failures} failing check(s)")
        return 1

    _print(PASS, "Smoke test completed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
