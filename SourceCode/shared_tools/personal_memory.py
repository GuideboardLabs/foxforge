from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from shared_tools.personal_memory_records import PersonalMemoryRecordStore


_PROFILE_DEFAULTS: dict[str, str] = {
    "preferred_name": "",
    "full_name": "",
    "age": "",
    "gender": "",
    "birthday": "",
    "location": "",
    "ancestry": "",
    "health": "",
    "work": "",
    "likes": "",
    "dislikes": "",
    "notes": "",
}
_FAMILY_DEFAULTS: dict[str, str] = {
    "name": "",
    "relationship": "",
    "notes": "",
    "nickname": "",
    "age": "",
    "birthday": "",
    "gender": "",
    "school_or_work": "",
    "likes": "",
    "dislikes": "",
    "important_dates": "",
    "medical_notes": "",
}
_PET_DEFAULTS: dict[str, str] = {
    "name": "",
    "species": "",
    "notes": "",
    "breed": "",
    "sex": "",
    "age": "",
    "birthday": "",
    "weight": "",
    "food": "",
    "medications": "",
    "vet": "",
    "microchip": "",
    "behavior_notes": "",
}
_EMPTY_SCHEMA: dict[str, Any] = {
    "user_profile": {},
    "family_members": [],
    "pets": [],
    "household_notes": "",
}
_TOKEN_RE = re.compile(r"[a-z0-9]{2,}")
_STOP = {
    "the", "and", "for", "with", "from", "that", "this", "have", "your", "about", "they", "them",
    "were", "been", "into", "then", "than", "when", "what", "where", "will", "would", "could",
}
_RELATION_ALIASES = {
    "kid": "child",
    "kids": "child",
    "mom": "mother",
    "dad": "father",
    "nephews": "nephew",
    "nieces": "niece",
    "uncles": "uncle",
    "aunts": "aunt",
}
_PET_SPECIES_CANONICAL = {
    "puppy": "dog",
    "dog": "dog",
    "kitten": "cat",
    "cat": "cat",
    "bird": "bird",
    "rabbit": "rabbit",
    "hamster": "hamster",
    "fish": "fish",
    "snake": "snake",
    "turtle": "turtle",
    "pet": "pet",
}
_FAMILY_QUERY_HINTS = {
    "family", "wife", "husband", "son", "daughter", "child", "kid", "kids", "birthday", "gift",
    "school", "teacher", "parent", "mom", "mother", "dad", "father", "brother", "sister",
    "nephew", "niece", "uncle", "aunt",
}
_PET_QUERY_HINTS = {
    "pet", "pets", "dog", "cat", "puppy", "kitten", "vet", "veterinarian", "food", "walk",
    "groom", "grooming", "neuter", "spay", "animal", "flea", "tick", "medication", "medications",
}
_PROFILE_QUERY_HINTS = {
    "name", "birthday", "location", "near", "local", "travel", "trip", "vacation", "weather",
    "work", "job", "career", "gift", "restaurant", "movie", "music", "food", "age", "ancestry",
    "health",
}
def _atomic_write(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=True), encoding="utf-8")
    tmp.replace(path)


def _copy_schema() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in _EMPTY_SCHEMA.items():
        if isinstance(value, list):
            out[key] = []
        elif isinstance(value, dict):
            out[key] = dict(value)
        else:
            out[key] = value
    return out


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _parse_iso(value: Any) -> datetime | None:
    text = _clean_text(value)
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _normalize_object(raw: Any, defaults: dict[str, str]) -> dict[str, str]:
    row = dict(defaults)
    if not isinstance(raw, dict):
        return row
    legacy_gender = ""
    if "gender" in defaults:
        legacy_gender = _clean_text(raw.get("gender", "")) or _clean_text(raw.get("pronouns", ""))
    for key in defaults:
        if key == "gender":
            row[key] = legacy_gender
            continue
        row[key] = _clean_text(raw.get(key, ""))
    return row


def _normalize_rows(raw: Any, defaults: dict[str, str], required_key: str) -> list[dict[str, str]]:
    if not isinstance(raw, list):
        return []
    out: list[dict[str, str]] = []
    for item in raw:
        row = _normalize_object(item, defaults)
        if not row.get(required_key):
            continue
        out.append(row)
    return out


def _detail_parts(row: dict[str, str], fields: list[tuple[str, str]]) -> list[str]:
    out: list[str] = []
    for key, label in fields:
        value = _clean_text(row.get(key, ""))
        if value:
            out.append(f"{label}: {value}")
    return out


def _name_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", _clean_text(value).lower()).strip()


def _display_name(raw: str) -> str:
    text = " ".join(_clean_text(raw).split())
    if not text:
        return ""
    return " ".join(part[:1].upper() + part[1:] if part else "" for part in text.split(" "))


def _tokens(text: str) -> Counter[str]:
    words = [w for w in _TOKEN_RE.findall(_clean_text(text).lower()) if w not in _STOP]
    return Counter(words)


def _overlap_score(query_terms: set[str], text: str) -> float:
    if not query_terms:
        return 0.0
    row_terms = set(_tokens(text))
    if not row_terms:
        return 0.0
    hits = len(query_terms & row_terms)
    if hits <= 0:
        return 0.0
    return min(0.24, hits * 0.06)


def _set_if_blank(target: dict[str, str], key: str, value: str) -> bool:
    clean = _clean_text(value)
    if not clean:
        return False
    if _clean_text(target.get(key, "")):
        return False
    target[key] = clean
    return True


def _prefer_richer(target: dict[str, str], key: str, value: str) -> bool:
    clean = _clean_text(value)
    if not clean:
        return False
    current = _clean_text(target.get(key, ""))
    if not current or len(clean) > len(current):
        target[key] = clean
        return True
    return False


class PersonalMemory:
    def __init__(self, repo_root: Path) -> None:
        self.path = repo_root / "Runtime" / "memory" / "personal" / "life_context.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._record_store = PersonalMemoryRecordStore(repo_root)

    def load(self) -> dict[str, Any]:
        with self._lock:
            if self.path.exists():
                try:
                    data = json.loads(self.path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    data = {}
            else:
                data = {}
            normalized = self._normalize(data)
            records = self._record_store.ensure_records(normalized)
            rebuilt = self._record_store.snapshot_from_records(records)
            if rebuilt != normalized:
                _atomic_write(self.path, rebuilt)
            return rebuilt

    def save(
        self,
        data: dict[str, Any],
        *,
        sync_source_type: str = "manual",
        sync_source_label: str = "life_admin",
        sync_status: str = "confirmed",
        sync_confidence: float = 1.0,
        authoritative_categories: set[str] | None = None,
    ) -> None:
        with self._lock:
            normalized = self._normalize(data)
            records = self._record_store.sync_from_snapshot(
                normalized,
                source_type=sync_source_type,
                source_label=sync_source_label,
                status=sync_status,
                confidence=sync_confidence,
                replace_scope=str(sync_source_type).strip().lower() == "manual",
                authoritative_categories=authoritative_categories,
            )
            _atomic_write(self.path, self._record_store.snapshot_from_records(records))

    def _normalize(self, data: Any) -> dict[str, Any]:
        payload = data if isinstance(data, dict) else {}
        result = _copy_schema()
        result["user_profile"] = _normalize_object(payload.get("user_profile", {}), _PROFILE_DEFAULTS)
        result["family_members"] = _normalize_rows(payload.get("family_members", []), _FAMILY_DEFAULTS, "name")
        result["pets"] = _normalize_rows(payload.get("pets", []), _PET_DEFAULTS, "name")
        result["household_notes"] = _clean_text(payload.get("household_notes", ""))
        return result

    def list_records(self, include_forgotten: bool = True) -> list[dict[str, Any]]:
        with self._lock:
            return self._record_store.list_records(include_forgotten=include_forgotten)

    def upsert_record(
        self,
        *,
        category: str,
        subject: str,
        field: str,
        value: str,
        status: str = "captured",
        confidence: float = 0.7,
        source_type: str = "chat",
        source_label: str = "chat",
        source_ref: str = "",
        evidence: str = "",
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            record = self._record_store.upsert_record(
                category=category,
                subject=subject,
                field=field,
                value=value,
                status=status,
                confidence=confidence,
                source_type=source_type,
                source_label=source_label,
                source_ref=source_ref,
                evidence=evidence,
                tags=tags,
            )
            _atomic_write(self.path, self._record_store.snapshot_from_records())
            return record

    def confirm_record(self, record_id: str) -> dict[str, Any] | None:
        with self._lock:
            record = self._record_store.update_record_status(record_id, status="confirmed")
            if record is not None:
                _atomic_write(self.path, self._record_store.snapshot_from_records())
            return record

    def pin_record(self, record_id: str) -> dict[str, Any] | None:
        with self._lock:
            record = self._record_store.update_record_status(record_id, status="pinned")
            if record is not None:
                _atomic_write(self.path, self._record_store.snapshot_from_records())
            return record

    def forget_record(self, record_id: str) -> dict[str, Any] | None:
        with self._lock:
            record = self._record_store.update_record_status(record_id, status="forgotten")
            if record is not None:
                _atomic_write(self.path, self._record_store.snapshot_from_records())
            return record

    def mark_used(self, record_id: str) -> dict[str, Any] | None:
        with self._lock:
            record = self._record_store.mark_used(record_id)
            if record is not None:
                _atomic_write(self.path, self._record_store.snapshot_from_records())
            return record

    def get_record(self, record_id: str) -> dict[str, Any] | None:
        with self._lock:
            return self._record_store.get_record(record_id)

    def update_record(
        self,
        record_id: str,
        *,
        value: str | None = None,
        status: str | None = None,
        confidence: float | None = None,
        evidence: str | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, Any] | None:
        with self._lock:
            record = self._record_store.update_record(
                record_id,
                value=value,
                status=status,
                confidence=confidence,
                evidence=evidence,
                tags=tags,
            )
            if record is not None:
                _atomic_write(self.path, self._record_store.snapshot_from_records())
            return record

    def explain_record(self, record_id: str) -> dict[str, Any] | None:
        with self._lock:
            return self._record_store.explain_record(record_id)

    def second_brain_payload(self) -> dict[str, Any]:
        context = self.load()
        records = self.list_records(include_forgotten=True)
        active_records = [row for row in records if _clean_text(row.get("status", "")).lower() != "forgotten"]
        now = datetime.now(timezone.utc)

        def _sort_desc(rows: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
            def _key(row: dict[str, Any]) -> tuple[float, str]:
                dt = _parse_iso(row.get(field, ""))
                return (dt.timestamp() if dt is not None else 0.0, _clean_text(row.get("id", "")))
            return sorted(rows, key=_key, reverse=True)

        def _age_days(row: dict[str, Any]) -> int | None:
            seen = _parse_iso(row.get("updated_at", ""))
            if seen is None:
                return None
            return max(0, int((now - seen).total_seconds() // 86400))

        source_breakdown: dict[str, int] = {}
        status_breakdown: dict[str, int] = {}
        for row in records:
            source = _clean_text(row.get("source_type", "")).lower() or "unknown"
            status = _clean_text(row.get("status", "")).lower() or "captured"
            source_breakdown[source] = int(source_breakdown.get(source, 0)) + 1
            status_breakdown[status] = int(status_breakdown.get(status, 0)) + 1

        pinned = [row for row in active_records if _clean_text(row.get("status", "")).lower() == "pinned"]
        needs_review = [
            row for row in active_records
            if _clean_text(row.get("status", "")).lower() in {"captured", "stale"}
            or float(row.get("confidence", 0.0) or 0.0) < 0.7
        ]
        recently_used = [row for row in active_records if _parse_iso(row.get("last_used_at", "")) is not None]
        fading = []
        for row in active_records:
            status = _clean_text(row.get("status", "")).lower()
            if status == "pinned":
                continue
            days = _age_days(row)
            if status == "stale" or (days is not None and days >= 120):
                fading.append(row)

        overview = {
            "total_records": len(records),
            "active_records": len(active_records),
            "pinned": len(pinned),
            "needs_review": len(needs_review),
            "recently_used": len(recently_used),
            "forgotten": len([row for row in records if _clean_text(row.get("status", "")).lower() == "forgotten"]),
            "family_members": len(context.get("family_members", [])) if isinstance(context.get("family_members", []), list) else 0,
            "pets": len(context.get("pets", [])) if isinstance(context.get("pets", []), list) else 0,
        }

        briefing_lines: list[str] = []
        if overview["pinned"] > 0:
            briefing_lines.append(f"{overview['pinned']} pinned memories are anchoring answers right now.")
        if overview["needs_review"] > 0:
            briefing_lines.append(f"{overview['needs_review']} memory item(s) need review or confirmation.")
        if not briefing_lines:
            briefing_lines.append("Second brain is quiet right now. Add profile details or pinned memories to deepen context.")

        timeline_rows = []
        for row in _sort_desc(active_records, "updated_at")[:12]:
            timeline_rows.append(
                {
                    "id": _clean_text(row.get("id", "")),
                    "subject": _clean_text(row.get("subject", "")) or _clean_text(row.get("category", "")),
                    "field": _clean_text(row.get("field", "")),
                    "value": _clean_text(row.get("value", "")),
                    "status": _clean_text(row.get("status", "")),
                    "source_type": _clean_text(row.get("source_type", "")),
                    "source_label": _clean_text(row.get("source_label", "")),
                    "updated_at": _clean_text(row.get("updated_at", "")),
                    "last_used_at": _clean_text(row.get("last_used_at", "")),
                    "age_days": _age_days(row),
                }
            )

        return {
            "context": context,
            "records": records,
            "overview": overview,
            "status_breakdown": status_breakdown,
            "source_breakdown": source_breakdown,
            "briefing_lines": briefing_lines[:6],
            "pinned_records": _sort_desc(pinned, "updated_at")[:6],
            "needs_review_records": _sort_desc(needs_review, "updated_at")[:6],
            "recent_records": _sort_desc(active_records, "updated_at")[:8],
            "recently_used_records": _sort_desc(recently_used, "last_used_at")[:6],
            "fading_records": _sort_desc(fading, "updated_at")[:6],
            "timeline": timeline_rows,
        }

    def _record_age_penalty(self, updated_at: str) -> float:
        raw = _clean_text(updated_at)
        if not raw:
            return 0.0
        try:
            seen = datetime.fromisoformat(raw)
        except ValueError:
            return 0.0
        if seen.tzinfo is None:
            seen = seen.replace(tzinfo=timezone.utc)
        days = max(0.0, (datetime.now(timezone.utc) - seen).total_seconds() / 86400.0)
        if days >= 365:
            return 0.12
        if days >= 180:
            return 0.08
        if days >= 90:
            return 0.04
        return 0.0

    def _record_line_for_context(self, record: dict[str, Any]) -> str:
        category = _clean_text(record.get("category", "")).lower()
        subject = _clean_text(record.get("subject", ""))
        field = _clean_text(record.get("field", "")).replace("_", " ")
        value = _clean_text(record.get("value", ""))
        if category == "profile":
            return f"You: {field}: {value}"
        if category == "family":
            return f"Family: {subject} | {field}: {value}"
        if category == "pet":
            return f"Pet: {subject} | {field}: {value}"
        if category == "household":
            return f"Household: {value}"
        return f"Memory: {subject or category} | {field}: {value}"

    def _find_row(self, rows: list[dict[str, str]], key: str, value: str) -> dict[str, str] | None:
        target = _name_key(value)
        if not target:
            return None
        for row in rows:
            if _name_key(row.get(key, "")) == target:
                return row
        return None

    def _upsert_family(self, data: dict[str, Any], *, name: str, relationship: str = "", **fields: Any) -> bool:
        clean_name = _display_name(name)
        if not clean_name:
            return False
        rows = data.get("family_members", [])
        if not isinstance(rows, list):
            rows = []
            data["family_members"] = rows
        row = self._find_row(rows, "name", clean_name)
        changed = False
        if row is None:
            row = dict(_FAMILY_DEFAULTS)
            row["name"] = clean_name
            rows.append(row)
            changed = True
        if relationship:
            canonical = _RELATION_ALIASES.get(_clean_text(relationship).lower(), _clean_text(relationship).lower())
            if canonical:
                changed = _prefer_richer(row, "relationship", canonical) or changed
        for key, value in fields.items():
            if key in _FAMILY_DEFAULTS:
                changed = _prefer_richer(row, key, _clean_text(value)) or changed
        return changed

    def _upsert_pet(self, data: dict[str, Any], *, name: str, species: str = "", **fields: Any) -> bool:
        clean_name = _display_name(name)
        if not clean_name:
            return False
        rows = data.get("pets", [])
        if not isinstance(rows, list):
            rows = []
            data["pets"] = rows
        row = self._find_row(rows, "name", clean_name)
        changed = False
        if row is None:
            row = dict(_PET_DEFAULTS)
            row["name"] = clean_name
            rows.append(row)
            changed = True
        if species:
            canonical = _PET_SPECIES_CANONICAL.get(_clean_text(species).lower(), _clean_text(species).lower())
            if canonical:
                changed = _prefer_richer(row, "species", canonical) or changed
        for key, value in fields.items():
            if key in _PET_DEFAULTS:
                changed = _prefer_richer(row, key, _clean_text(value)) or changed
        return changed

    def capture_from_text(self, text: str, source: str = "chat") -> dict[str, Any]:
        body = " ".join(_clean_text(text).split())
        if not body:
            return {"captured": 0, "source": source}

        data = self.load()
        captured = 0
        profile = data.get("user_profile", {})
        if not isinstance(profile, dict):
            profile = dict(_PROFILE_DEFAULTS)
            data["user_profile"] = profile

        for pattern in [
            re.compile(r"\bmy name is\s+([A-Za-z][A-Za-z' -]{1,40})", re.IGNORECASE),
            re.compile(r"\bcall me\s+([A-Za-z][A-Za-z' -]{1,40})", re.IGNORECASE),
        ]:
            match = pattern.search(body)
            if match:
                value = _display_name(match.group(1))
                if " " in value:
                    captured += 1 if _set_if_blank(profile, "full_name", value) else 0
                    first = value.split(" ", 1)[0]
                    captured += 1 if _set_if_blank(profile, "preferred_name", first) else 0
                else:
                    captured += 1 if _set_if_blank(profile, "preferred_name", value) else 0
                break

        match = re.search(r"\bmy gender is\s+([A-Za-z/ -]{2,30})", body, re.IGNORECASE)
        if match and _set_if_blank(profile, "gender", match.group(1)):
            captured += 1
        match = re.search(r"\bi live in\s+([A-Za-z][A-Za-z .,'-]{2,40})", body, re.IGNORECASE)
        if match and _set_if_blank(profile, "location", match.group(1)):
            captured += 1
        match = re.search(r"\bi work (?:at|for|as)\s+([A-Za-z0-9][A-Za-z0-9 .,&'-]{2,50})", body, re.IGNORECASE)
        if match and _set_if_blank(profile, "work", match.group(1)):
            captured += 1
        match = re.search(r"\bmy birthday is\s+([A-Za-z0-9, /-]{3,40})", body, re.IGNORECASE)
        if match and _set_if_blank(profile, "birthday", match.group(1)):
            captured += 1

        family_patterns = [
            re.compile(
                r"\bmy\s+(wife|husband|son|daughter|child|kid|brother|sister|mother|mom|father|dad|partner|girlfriend|boyfriend)\s+([A-Za-z][A-Za-z' -]{1,40})(?:\s+is\s+(\d{1,2})\s*(?:years?|yrs?)\s*old)?",
                re.IGNORECASE,
            ),
            re.compile(
                r"\bmy\s+(nephew|niece|uncle|aunt)\s+([A-Za-z][A-Za-z' -]{1,40})(?:\s+is\s+(\d{1,2})\s*(?:years?|yrs?)\s*old)?",
                re.IGNORECASE,
            ),
            re.compile(
                r"\bour\s+(son|daughter|child|kid|nephew|niece)\s+([A-Za-z][A-Za-z' -]{1,40})(?:\s+is\s+(\d{1,2})\s*(?:years?|yrs?)\s*old)?",
                re.IGNORECASE,
            ),
        ]
        for pattern in family_patterns:
            for match in pattern.finditer(body):
                rel = match.group(1)
                name = match.group(2)
                age = _clean_text(match.group(3))
                if self._upsert_family(data, name=name, relationship=rel, age=age):
                    captured += 1

        pet_pattern = re.compile(
            r"\bmy\s+(dog|puppy|cat|kitten|bird|rabbit|hamster|fish|snake|turtle|pet)\s+([A-Za-z][A-Za-z' -]{1,40})(?:\s+is\s+(\d{1,2})\s*(?:years?|yrs?|months?)\s*old)?",
            re.IGNORECASE,
        )
        for match in pet_pattern.finditer(body):
            species = match.group(1)
            name = match.group(2)
            age = _clean_text(match.group(3))
            if self._upsert_pet(data, name=name, species=species, age=age):
                captured += 1

        family_rows = data.get("family_members", [])
        if isinstance(family_rows, list):
            for row in family_rows:
                name = _clean_text(row.get("name", ""))
                if not name:
                    continue
                match = re.search(
                    rf"\b{re.escape(name)}\s+is\s+(\d{{1,2}})\s*(?:years?|yrs?)\s*old\b",
                    body,
                    re.IGNORECASE,
                )
                if match and _set_if_blank(row, "age", match.group(1)):
                    captured += 1

        pet_rows = data.get("pets", [])
        if isinstance(pet_rows, list):
            for row in pet_rows:
                name = _clean_text(row.get("name", ""))
                if not name:
                    continue
                match = re.search(
                    rf"\b{re.escape(name)}\s+is\s+(\d{{1,2}})\s*(?:years?|yrs?|months?)\s*old\b",
                    body,
                    re.IGNORECASE,
                )
                if match and _set_if_blank(row, "age", match.group(1)):
                    captured += 1

        if captured > 0:
            self.save(
                data,
                sync_source_type="chat",
                sync_source_label=source,
                sync_status="captured",
                sync_confidence=0.74,
            )
        return {"captured": captured, "source": source}

    def relevant_context_for_query(self, query: str, max_chars: int = 1400) -> str:
        query_text = _clean_text(query)
        if not query_text:
            return ""
        records = self.list_records(include_forgotten=False)
        query_terms = set(_tokens(query_text))
        low = query_text.lower()
        candidates: list[tuple[float, str, list[str]]] = []

        for row in records:
            category = _clean_text(row.get("category", "")).lower()
            subject = _clean_text(row.get("subject", ""))
            field = _clean_text(row.get("field", "")).lower()
            value = _clean_text(row.get("value", ""))
            status = _clean_text(row.get("status", "captured")).lower()
            if not category or not field or not value or status == "forgotten":
                continue

            text_blob = " ".join(
                [
                    category,
                    subject,
                    field.replace("_", " "),
                    value,
                    _clean_text(row.get("evidence", "")),
                    " ".join(str(x) for x in row.get("tags", []) if _clean_text(x)),
                ]
            )
            score = _overlap_score(query_terms, text_blob)
            confidence = float(row.get("confidence", 0.7) or 0.7)
            score += max(0.0, min(0.2, (confidence - 0.5) * 0.35))
            score -= self._record_age_penalty(_clean_text(row.get("updated_at", "")))
            if status == "pinned":
                score += 0.14
            elif status == "confirmed":
                score += 0.08
            elif status == "stale":
                score -= 0.08
            elif status == "captured" and confidence < 0.62:
                score -= 0.06

            if category == "profile" and query_terms & _PROFILE_QUERY_HINTS:
                score += 0.1
            if category == "family" and query_terms & _FAMILY_QUERY_HINTS:
                score += 0.08
            if category == "pet" and query_terms & _PET_QUERY_HINTS:
                score += 0.08
            if subject and _name_key(subject) in _name_key(low):
                score += 0.18
            if field and field.replace("_", " ") in low:
                score += 0.08

            threshold = 0.14
            if status == "captured" and confidence < 0.7:
                threshold = 0.2
            if status == "stale":
                threshold = 0.24
            if score < threshold:
                continue
            candidates.append((score, self._record_line_for_context(row), [_clean_text(row.get("id", ""))]))

        if not candidates:
            return ""
        candidates.sort(key=lambda item: item[0], reverse=True)
        lines = [
            "Relevant personal context (use only if it genuinely helps; weave it in naturally and do not mention memory unless asked):"
        ]
        used_ids: list[str] = []
        seen_lines: set[str] = set()
        for _score, line, record_ids in candidates:
            if line in seen_lines:
                continue
            seen_lines.add(line)
            lines.append(f"- {line}")
            used_ids.extend([rid for rid in record_ids if rid])
            if len(lines) >= 5:
                break
        text_out = "\n".join(lines)
        for record_id in used_ids:
            try:
                self.mark_used(record_id)
            except Exception:
                pass
        if len(text_out) <= max_chars:
            return text_out
        return text_out[:max(260, max_chars)].rsplit("\n", 1)[0]

    def format_for_prompt(self) -> str:
        data = self.load()
        sections: list[str] = []

        profile = _normalize_object(data.get("user_profile", {}), _PROFILE_DEFAULTS)
        profile_name = profile.get("preferred_name") or profile.get("full_name")
        if profile_name:
            line = f"- {profile_name}"
            if profile.get("gender"):
                line += f" ({profile['gender']})"
            extras = _detail_parts(
                profile,
                [
                    ("age", "age"),
                    ("birthday", "birthday"),
                    ("location", "location"),
                    ("ancestry", "ancestry"),
                    ("health", "health"),
                    ("work", "work"),
                    ("likes", "likes"),
                    ("dislikes", "dislikes"),
                    ("notes", "notes"),
                ],
            )
            if extras:
                line += " | " + "; ".join(extras)
            sections.append("### You\n" + line)

        family = [m for m in data.get("family_members", []) if isinstance(m, dict) and m.get("name")]
        if family:
            lines = ["### Family"]
            for m in family:
                rel = f" ({m.get('relationship', '')})" if m.get("relationship") else ""
                extras = _detail_parts(
                    m,
                    [
                        ("nickname", "nickname"),
                        ("age", "age"),
                        ("birthday", "birthday"),
                        ("gender", "gender"),
                        ("school_or_work", "school/work"),
                        ("likes", "likes"),
                        ("dislikes", "dislikes"),
                        ("important_dates", "important dates"),
                        ("medical_notes", "medical notes"),
                        ("notes", "notes"),
                    ],
                )
                line = f"- {m['name']}{rel}"
                if extras:
                    line += " | " + "; ".join(extras)
                lines.append(line)
            sections.append("\n".join(lines))

        pets = [p for p in data.get("pets", []) if isinstance(p, dict) and p.get("name")]
        if pets:
            lines = ["### Pets"]
            for p in pets:
                species = f" ({p.get('species', '')})" if p.get("species") else ""
                extras = _detail_parts(
                    p,
                    [
                        ("breed", "breed"),
                        ("sex", "sex"),
                        ("age", "age"),
                        ("birthday", "birthday"),
                        ("weight", "weight"),
                        ("food", "food"),
                        ("medications", "medications"),
                        ("vet", "vet"),
                        ("microchip", "microchip"),
                        ("behavior_notes", "behavior"),
                        ("notes", "notes"),
                    ],
                )
                line = f"- {p['name']}{species}"
                if extras:
                    line += " | " + "; ".join(extras)
                lines.append(line)
            sections.append("\n".join(lines))

        notes = str(data.get("household_notes", "")).strip()
        if notes:
            sections.append(f"### Context Notes\n{notes}")

        return "\n\n".join(sections)
