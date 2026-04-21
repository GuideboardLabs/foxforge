from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from shared_tools.embedding_memory import _vec_cosine
from shared_tools.ollama_client import OllamaClient

from .turn_replay import replay_turn


@dataclass(slots=True)
class RegressionResult:
    thread_id: str
    passed: bool
    score: float
    expected: str
    actual: str
    reason: str = ""


def _manifest_path(repo_root: Path) -> Path:
    return Path(repo_root) / "Runtime" / "state" / "regression_set.jsonl"


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = str(raw or "").strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        if isinstance(row, dict):
            out.append(row)
    return out


def _fallback_similarity(a: str, b: str) -> float:
    return float(SequenceMatcher(None, str(a or ""), str(b or "")).ratio())


def _embedding_similarity(client: OllamaClient | None, a: str, b: str) -> float:
    if client is None:
        return _fallback_similarity(a, b)
    try:
        va = client.embed("qwen3-embedding:4b", str(a or "")[:3000], timeout=20)
        vb = client.embed("qwen3-embedding:4b", str(b or "")[:3000], timeout=20)
        score = float(_vec_cosine(va, vb))
        if score <= 0.0:
            return _fallback_similarity(a, b)
        return score
    except Exception:
        return _fallback_similarity(a, b)


def run_regression_suite(
    orchestrator: Any,
    *,
    min_similarity: float = 0.90,
) -> dict[str, Any]:
    repo_root = Path(orchestrator.repo_root)
    manifest = _load_manifest(_manifest_path(repo_root))
    if not manifest:
        return {
            "ok": False,
            "error": f"No regression manifest entries found at {_manifest_path(repo_root)}",
            "results": [],
        }

    client: OllamaClient | None
    try:
        client = OllamaClient()
    except Exception:
        client = None

    results: list[RegressionResult] = []
    for row in manifest:
        thread_id = str(row.get("thread_id") or row.get("turn_id") or "").strip()
        expected = str(row.get("composed_answer") or row.get("answer") or row.get("expected") or "").strip()
        if not thread_id or not expected:
            continue
        replay = replay_turn(orchestrator, thread_id=thread_id)
        if not replay.get("ok"):
            results.append(
                RegressionResult(
                    thread_id=thread_id,
                    passed=False,
                    score=0.0,
                    expected=expected,
                    actual="",
                    reason=str(replay.get("error", "replay failed")),
                )
            )
            continue
        state = replay.get("state", {}) if isinstance(replay.get("state", {}), dict) else {}
        actual = str(state.get("composed_answer", "") or state.get("final_reply", "")).strip()
        score = _embedding_similarity(client, expected, actual)
        results.append(
            RegressionResult(
                thread_id=thread_id,
                passed=score >= float(min_similarity),
                score=score,
                expected=expected,
                actual=actual,
                reason="" if score >= float(min_similarity) else "semantic divergence",
            )
        )

    passed = sum(1 for row in results if row.passed)
    failed = len(results) - passed
    return {
        "ok": True,
        "total": len(results),
        "passed": passed,
        "failed": failed,
        "threshold": float(min_similarity),
        "results": [asdict(row) for row in results],
    }
