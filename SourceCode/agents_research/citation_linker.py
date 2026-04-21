from __future__ import annotations

import re
import uuid
from typing import Any
from urllib.parse import urlsplit

from shared_tools.embedding_memory import _vec_cosine


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_MARKER_RE = re.compile(r"\[S(\d+)\]", re.IGNORECASE)
_INFERENCE_RE = re.compile(r"\[I\]", re.IGNORECASE)
_TOKEN_RE = re.compile(r"[a-z0-9]{3,}")


def _domain(url: str) -> str:
    try:
        return str(urlsplit(str(url or "")).hostname or "").lower().removeprefix("www.")
    except Exception:
        return ""


def _token_overlap(a: str, b: str) -> float:
    at = set(_TOKEN_RE.findall(str(a or "").lower()))
    bt = set(_TOKEN_RE.findall(str(b or "").lower()))
    if not at or not bt:
        return 0.0
    return len(at & bt) / max(1, len(at))


def _embedding_similarity(client: Any | None, a: str, b: str) -> float:
    if client is None:
        return _token_overlap(a, b)
    try:
        va = client.embed("qwen3-embedding:4b", str(a or "")[:1200], timeout=15)
        vb = client.embed("qwen3-embedding:4b", str(b or "")[:1200], timeout=15)
        score = float(_vec_cosine(va, vb))
        if score > 0.0:
            return score
    except Exception:
        pass
    return _token_overlap(a, b)


def build_retrieved_chunks(findings: list[dict[str, Any]], *, limit: int = 80) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for finding in findings:
        for item in finding.get("source_evidence", []) if isinstance(finding.get("source_evidence", []), list) else []:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url", "")).strip()
            snippet = str(item.get("snippet", "")).strip()
            if not url and not snippet:
                continue
            key = f"{url}|{snippet[:160]}".lower()
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "id": f"c_{uuid.uuid4().hex[:10]}",
                    "url": url,
                    "domain": str(item.get("domain", "")).strip().lower() or _domain(url),
                    "snippet": snippet[:600],
                    "score": float(item.get("score", item.get("source_score", 0.0)) or 0.0),
                }
            )
            if len(rows) >= max(1, int(limit)):
                return rows
    return rows


def link(
    text: str,
    *,
    retrieved_chunks: list[dict[str, Any]],
    threshold: float = 0.45,
    embedding_client: Any | None = None,
) -> dict[str, Any]:
    body = str(text or "").strip()
    if not body:
        return {"text": "", "sentences": [], "retrieved_chunks": list(retrieved_chunks or [])}

    chunks = [dict(row) for row in (retrieved_chunks or []) if isinstance(row, dict)]
    ordered = {idx + 1: row for idx, row in enumerate(chunks)}
    sentence_rows: list[dict[str, Any]] = []

    for raw in _SENTENCE_SPLIT_RE.split(body):
        sentence = str(raw or "").strip()
        if not sentence:
            continue
        marker_ids = [int(m.group(1)) for m in _MARKER_RE.finditer(sentence)]
        is_inference = bool(_INFERENCE_RE.search(sentence))
        clean = _INFERENCE_RE.sub("", _MARKER_RE.sub("", sentence)).strip()
        if not clean:
            continue

        citation_ids: list[str] = []
        # Marker path: [S1], [S2], ...
        for marker in marker_ids:
            row = ordered.get(marker)
            if not row:
                continue
            cid = str(row.get("id", "")).strip()
            if cid and cid not in citation_ids:
                citation_ids.append(cid)

        # Auto-align when markers are absent and this sentence is not explicitly inference.
        if not citation_ids and not is_inference and chunks:
            best_id = ""
            best_score = 0.0
            for row in chunks:
                snippet = str(row.get("snippet", "")).strip()
                if not snippet:
                    continue
                score = _embedding_similarity(embedding_client, clean, snippet)
                if score > best_score:
                    best_score = score
                    best_id = str(row.get("id", "")).strip()
            if best_id and best_score >= float(threshold):
                citation_ids = [best_id]

        sentence_rows.append({"text": clean, "citation_ids": citation_ids})

    # Strip source markers from plain text fallback.
    clean_text = _INFERENCE_RE.sub("", _MARKER_RE.sub("", body)).strip()
    clean_text = re.sub(r"\s{2,}", " ", clean_text)
    return {
        "type": "research_reply",
        "text": clean_text,
        "sentences": sentence_rows,
        "retrieved_chunks": chunks,
    }

