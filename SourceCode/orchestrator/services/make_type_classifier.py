from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from .make_types import MAKE_TYPES, make_types_hash, normalize_make_type


_TOKEN_RE = re.compile(r"[a-z0-9_]{3,}")
_MODEL_LOCK = Lock()
_MODEL_CACHE: dict[str, dict[str, Any]] = {}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _artifact_dir(repo_root: Path) -> Path:
    path = Path(repo_root) / "Runtime" / "models" / "make_type_setfit"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _artifact_json(repo_root: Path) -> Path:
    return _artifact_dir(repo_root) / "artifact.json"


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(str(text or "").lower())


def _score_keywords(text: str, keyword_model: dict[str, list[str]]) -> tuple[str, float]:
    tokens = _tokenize(text)
    if not tokens or not keyword_model:
        return "", 0.0
    tok_counts = Counter(tokens)
    scores: dict[str, float] = {}
    for label, words in keyword_model.items():
        weight = 0.0
        for word in words:
            if not word:
                continue
            weight += float(tok_counts.get(word, 0))
        if weight > 0.0:
            scores[label] = weight
    if not scores:
        return "", 0.0
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_label, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    confidence = best_score / max(best_score + second_score, 1.0)
    return best_label, max(0.0, min(1.0, confidence))


def _build_keyword_model(rows: list[tuple[str, str]], top_k: int = 18) -> dict[str, list[str]]:
    per_label: dict[str, Counter[str]] = defaultdict(Counter)
    global_counts: Counter[str] = Counter()
    for text, label in rows:
        tokens = _tokenize(text)
        if not tokens:
            continue
        unique = set(tokens)
        global_counts.update(unique)
        per_label[label].update(unique)

    model: dict[str, list[str]] = {}
    for label in MAKE_TYPES:
        counts = per_label.get(label, Counter())
        if not counts:
            model[label] = []
            continue
        scored: list[tuple[str, float]] = []
        for token, c in counts.items():
            # TF-IDF-lite weighting to downrank global/common terms.
            g = max(1.0, float(global_counts.get(token, 1)))
            scored.append((token, float(c) / g))
        scored.sort(key=lambda item: item[1], reverse=True)
        model[label] = [token for token, _ in scored[:top_k]]
    return model


def _load_artifact(repo_root: Path) -> dict[str, Any]:
    key = str(Path(repo_root).resolve())
    with _MODEL_LOCK:
        cached = _MODEL_CACHE.get(key)
        if cached is not None:
            return dict(cached)
        path = _artifact_json(Path(repo_root))
        if not path.exists():
            payload = {}
            _MODEL_CACHE[key] = dict(payload)
            return payload
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        _MODEL_CACHE[key] = dict(payload)
        return payload


def classify(text: str, *, repo_root: Path) -> tuple[str, float]:
    payload = _load_artifact(repo_root)
    keyword_model = payload.get("keyword_model", {}) if isinstance(payload.get("keyword_model", {}), dict) else {}
    label, confidence = _score_keywords(text, {str(k): list(v) for k, v in keyword_model.items() if isinstance(v, list)})
    label = normalize_make_type(label)
    if label:
        return label, confidence
    # Hard fallback: choose a likely default for command-like prompts.
    low = str(text or "").lower()
    if any(tok in low for tok in ("app", "site", "frontend", "landing page")):
        return "web_app", 0.42
    if any(tok in low for tok in ("script", "cli", "python", "bash", "tool")):
        return "tool", 0.42
    if any(tok in low for tok in ("email", "reply to", "subject line")):
        return "email", 0.42
    if any(tok in low for tok in ("post", "twitter", "linkedin", "social")):
        return "social_post", 0.42
    return "", 0.0


def train(
    dataset_rows: list[tuple[str, str]],
    *,
    out_dir: Path,
    backend: str = "keyword",
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[str, str]] = []
    for text, raw_label in dataset_rows:
        label = normalize_make_type(raw_label)
        clean = str(text or "").strip()
        if not label or not clean:
            continue
        rows.append((clean, label))

    keyword_model = _build_keyword_model(rows)

    # Simple macro-F1 estimate using the same keyword classifier on training rows.
    labels = [label for _, label in rows]
    preds: list[str] = []
    for text, _ in rows:
        pred, _conf = _score_keywords(text, keyword_model)
        preds.append(normalize_make_type(pred) or "")
    f1 = _macro_f1(labels, preds, labels=list(MAKE_TYPES))

    artifact = {
        "artifact_version": 1,
        "trained_at": _now_iso(),
        "backend": str(backend or "keyword"),
        "enum_hash": make_types_hash(),
        "samples": len(rows),
        "macro_f1": round(float(f1), 4),
        "keyword_model": keyword_model,
    }
    artifact_path = out_dir / "artifact.json"
    artifact_path.write_text(json.dumps(artifact, indent=2, ensure_ascii=True), encoding="utf-8")
    with _MODEL_LOCK:
        _MODEL_CACHE[str(out_dir.parent.resolve())] = dict(artifact)
        _MODEL_CACHE[str(out_dir.resolve())] = dict(artifact)
    return artifact


def train_from_files(
    *,
    repo_root: Path,
    input_files: list[Path],
) -> dict[str, Any]:
    rows: list[tuple[str, str]] = []
    for path in input_files:
        p = Path(path)
        if not p.exists():
            continue
        for raw in p.read_text(encoding="utf-8").splitlines():
            line = str(raw or "").strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if not isinstance(row, dict):
                continue
            text = str(row.get("text") or row.get("input") or row.get("query") or "").strip()
            label = str(row.get("label") or row.get("suggested_type") or row.get("make_type") or "").strip().lower()
            if text and label:
                rows.append((text, label))
    return train(rows, out_dir=_artifact_dir(repo_root))


def _macro_f1(y_true: list[str], y_pred: list[str], *, labels: list[str]) -> float:
    if not y_true or not y_pred or len(y_true) != len(y_pred):
        return 0.0
    scores: list[float] = []
    for label in labels:
        tp = fp = fn = 0
        for yt, yp in zip(y_true, y_pred):
            if yp == label and yt == label:
                tp += 1
            elif yp == label and yt != label:
                fp += 1
            elif yp != label and yt == label:
                fn += 1
        if tp == 0 and fp == 0 and fn == 0:
            continue
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        if precision + recall <= 0.0:
            scores.append(0.0)
        else:
            scores.append((2.0 * precision * recall) / (precision + recall))
    if not scores:
        return 0.0
    return float(sum(scores) / max(1, len(scores)))


def confidence_band(value: float) -> str:
    score = max(0.0, min(1.0, float(value)))
    if score >= 0.8:
        return "high"
    if score >= 0.6:
        return "medium"
    if score >= 0.4:
        return "low"
    return "very_low"


def entropy_confidence(probabilities: list[float]) -> float:
    probs = [max(0.0, float(p)) for p in probabilities if p is not None]
    total = sum(probs)
    if total <= 0.0:
        return 0.0
    norm = [p / total for p in probs if p > 0.0]
    if not norm:
        return 0.0
    entropy = -sum(p * math.log(p + 1e-12) for p in norm)
    max_entropy = math.log(float(len(norm))) if len(norm) > 1 else 1.0
    if max_entropy <= 0.0:
        return 1.0
    return max(0.0, min(1.0, 1.0 - (entropy / max_entropy)))

