from __future__ import annotations

import json
import re
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROFILE_DEFAULTS: dict[str, str] = {
    "preferred_name": "",
    "full_name": "",
    "gender": "",
    "birthday": "",
    "location": "",
    "work": "",
    "likes": "",
    "dislikes": "",
    "notes": "",
}
FAMILY_DEFAULTS: dict[str, str] = {
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
PET_DEFAULTS: dict[str, str] = {
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
REMINDER_DEFAULTS: dict[str, str] = {
    "label": "",
    "frequency": "",
    "time": "",
    "notes": "",
    "person": "",
    "start_date": "",
    "end_date": "",
    "location": "",
    "channel": "",
    "priority": "",
}
EMPTY_SNAPSHOT: dict[str, Any] = {
    "user_profile": {},
    "family_members": [],
    "pets": [],
    "recurring_reminders": [],
    "household_notes": "",
}
RECORD_DEFAULTS: dict[str, Any] = {
    "id": "",
    "category": "",
    "subject": "",
    "field": "",
    "value": "",
    "status": "captured",
    "confidence": 0.7,
    "source_type": "",
    "source_label": "",
    "source_ref": "",
    "created_at": "",
    "updated_at": "",
    "last_used_at": "",
    "last_confirmed_at": "",
    "expires_at": "",
    "use_count": 0,
    "evidence": "",
    "tags": [],
}
RECORD_STATUSES = {"captured", "confirmed", "pinned", "stale", "forgotten"}
STATUS_PRIORITY = {
    "forgotten": -1,
    "stale": 0,
    "captured": 1,
    "confirmed": 2,
    "pinned": 3,
}
SOURCE_PRIORITY = {
    "unknown": 0,
    "chat": 1,
    "legacy": 2,
    "manual": 3,
    "import": 3,
}
SYNCABLE_CATEGORIES = {"profile", "family", "pet", "reminder", "household"}


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _name_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", _clean_text(value).lower()).strip()


def _display_name(raw: Any) -> str:
    text = " ".join(_clean_text(raw).split())
    if not text:
        return ""
    return " ".join(part[:1].upper() + part[1:] if part else "" for part in text.split(" "))


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


def _normalize_confidence(value: Any, fallback: float = 0.7) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        out = fallback
    return max(0.0, min(1.0, out))


def _normalize_tags(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in value:
        tag = _clean_text(item).lower()
        if not tag or tag in seen:
            continue
        seen.add(tag)
        out.append(tag[:40])
    return out


def _record_key(category: str, subject: str, field: str) -> str:
    return "|".join([_clean_text(category).lower(), _name_key(subject), _clean_text(field).lower()])


def _record_score(record: dict[str, Any]) -> tuple[int, float, str]:
    status = _clean_text(record.get("status", "captured")).lower()
    return (
        STATUS_PRIORITY.get(status, 0),
        _normalize_confidence(record.get("confidence", 0.7)),
        _clean_text(record.get("updated_at", "")),
    )


def _record_is_active(record: dict[str, Any]) -> bool:
    return _clean_text(record.get("value", "")) != "" and _clean_text(record.get("status", "")).lower() != "forgotten"


def _atomic_write(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=True), encoding="utf-8")
    tmp.replace(path)


class PersonalMemoryRecordStore:
    def __init__(self, repo_root: Path) -> None:
        self.path = repo_root / "Runtime" / "memory" / "personal" / "memory_records.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _normalize_record(self, raw: Any) -> dict[str, Any]:
        row = dict(RECORD_DEFAULTS)
        payload = raw if isinstance(raw, dict) else {}
        now = _now_iso()
        row["id"] = _clean_text(payload.get("id", "")) or uuid.uuid4().hex[:12]
        row["category"] = _clean_text(payload.get("category", "")).lower()
        row["subject"] = _clean_text(payload.get("subject", ""))
        field = _clean_text(payload.get("field", "")).lower()
        if field == "pronouns" and row["category"] in {"profile", "family"}:
            field = "gender"
        row["field"] = field
        row["value"] = _clean_text(payload.get("value", ""))
        status = _clean_text(payload.get("status", "captured")).lower()
        row["status"] = status if status in RECORD_STATUSES else "captured"
        row["confidence"] = _normalize_confidence(payload.get("confidence", 0.7))
        row["source_type"] = _clean_text(payload.get("source_type", "")).lower() or "unknown"
        row["source_label"] = _clean_text(payload.get("source_label", ""))
        row["source_ref"] = _clean_text(payload.get("source_ref", ""))
        row["created_at"] = _clean_text(payload.get("created_at", "")) or now
        row["updated_at"] = _clean_text(payload.get("updated_at", "")) or row["created_at"]
        row["last_used_at"] = _clean_text(payload.get("last_used_at", ""))
        row["last_confirmed_at"] = _clean_text(payload.get("last_confirmed_at", ""))
        row["expires_at"] = _clean_text(payload.get("expires_at", ""))
        try:
            row["use_count"] = max(0, int(payload.get("use_count", 0) or 0))
        except (TypeError, ValueError):
            row["use_count"] = 0
        row["evidence"] = _clean_text(payload.get("evidence", ""))
        row["tags"] = _normalize_tags(payload.get("tags", []))
        return row

    def _normalize_payload(self, payload: Any) -> dict[str, Any]:
        raw_records: list[Any]
        if isinstance(payload, dict):
            raw_records = payload.get("records", [])
        elif isinstance(payload, list):
            raw_records = payload
        else:
            raw_records = []
        records: list[dict[str, Any]] = []
        for item in raw_records:
            row = self._normalize_record(item)
            if not row["category"] or not row["field"]:
                continue
            records.append(row)
        records.sort(key=lambda row: (row["category"], _name_key(row["subject"]), row["field"], row["updated_at"], row["id"]))
        return {"version": 1, "records": records}

    def _load(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            payload = {}
        return self._normalize_payload(payload)["records"]

    def _save(self, records: list[dict[str, Any]]) -> None:
        _atomic_write(self.path, self._normalize_payload({"records": records}))

    def ensure_records(self, snapshot: dict[str, Any]) -> list[dict[str, Any]]:
        records = self._load()
        if records:
            return records
        records = self._records_from_snapshot(
            snapshot,
            source_type="legacy",
            source_label="life_context",
            status="confirmed",
            confidence=0.95,
        )
        self._save(records)
        return records

    def list_records(self, include_forgotten: bool = True) -> list[dict[str, Any]]:
        records = deepcopy(self._load())
        if include_forgotten:
            return records
        return [row for row in records if _clean_text(row.get("status", "")).lower() != "forgotten"]

    def get_record(self, record_id: str) -> dict[str, Any] | None:
        target = _clean_text(record_id)
        if not target:
            return None
        for row in self._load():
            if _clean_text(row.get("id", "")) == target:
                return deepcopy(row)
        return None

    def _find_record_index(self, records: list[dict[str, Any]], *, category: str, subject: str, field: str) -> int:
        key = _record_key(category, subject, field)
        best_idx = -1
        best_score: tuple[int, float, str] | None = None
        for idx, row in enumerate(records):
            if _record_key(row.get("category", ""), row.get("subject", ""), row.get("field", "")) != key:
                continue
            score = _record_score(row)
            if best_idx < 0 or score > best_score:
                best_idx = idx
                best_score = score
        return best_idx

    def _build_record(
        self,
        *,
        category: str,
        subject: str,
        field: str,
        value: str,
        status: str,
        confidence: float,
        source_type: str,
        source_label: str,
        source_ref: str = "",
        evidence: str = "",
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        now = _now_iso()
        return self._normalize_record(
            {
                "id": uuid.uuid4().hex[:12],
                "category": category,
                "subject": subject,
                "field": field,
                "value": value,
                "status": status,
                "confidence": confidence,
                "source_type": source_type,
                "source_label": source_label,
                "source_ref": source_ref,
                "created_at": now,
                "updated_at": now,
                "last_confirmed_at": now if status in {"confirmed", "pinned"} else "",
                "evidence": evidence,
                "tags": tags or [],
            }
        )

    def _records_from_snapshot(
        self,
        snapshot: dict[str, Any],
        *,
        source_type: str,
        source_label: str,
        status: str,
        confidence: float,
    ) -> list[dict[str, Any]]:
        data = self.normalize_snapshot(snapshot)
        records: list[dict[str, Any]] = []
        profile = data.get("user_profile", {})
        if isinstance(profile, dict):
            for field in PROFILE_DEFAULTS:
                value = _clean_text(profile.get(field, ""))
                if not value:
                    continue
                records.append(self._build_record(category="profile", subject="user", field=field, value=value, status=status, confidence=confidence, source_type=source_type, source_label=source_label, evidence=f"{source_label} profile", tags=["profile"]))
        for row in data.get("family_members", []):
            subject = _display_name(row.get("name", ""))
            if not subject:
                continue
            for field in FAMILY_DEFAULTS:
                value = _clean_text(row.get(field, ""))
                if not value:
                    continue
                records.append(self._build_record(category="family", subject=subject, field=field, value=value, status=status, confidence=confidence, source_type=source_type, source_label=source_label, evidence=f"{source_label} family:{subject}", tags=["family"]))
        for row in data.get("pets", []):
            subject = _display_name(row.get("name", ""))
            if not subject:
                continue
            for field in PET_DEFAULTS:
                value = _clean_text(row.get(field, ""))
                if not value:
                    continue
                records.append(self._build_record(category="pet", subject=subject, field=field, value=value, status=status, confidence=confidence, source_type=source_type, source_label=source_label, evidence=f"{source_label} pet:{subject}", tags=["pet"]))
        for row in data.get("recurring_reminders", []):
            subject = _clean_text(row.get("label", ""))
            if not subject:
                continue
            for field in REMINDER_DEFAULTS:
                value = _clean_text(row.get(field, ""))
                if not value:
                    continue
                records.append(self._build_record(category="reminder", subject=subject, field=field, value=value, status=status, confidence=confidence, source_type=source_type, source_label=source_label, evidence=f"{source_label} reminder:{subject}", tags=["reminder"]))
        notes = _clean_text(data.get("household_notes", ""))
        if notes:
            records.append(self._build_record(category="household", subject="household", field="notes", value=notes, status=status, confidence=confidence, source_type=source_type, source_label=source_label, evidence=f"{source_label} household notes", tags=["household"]))
        return records

    def normalize_snapshot(self, data: Any) -> dict[str, Any]:
        payload = data if isinstance(data, dict) else {}
        result = dict(EMPTY_SNAPSHOT)
        result["user_profile"] = _normalize_object(payload.get("user_profile", {}), PROFILE_DEFAULTS)
        result["family_members"] = _normalize_rows(payload.get("family_members", []), FAMILY_DEFAULTS, "name")
        result["pets"] = _normalize_rows(payload.get("pets", []), PET_DEFAULTS, "name")
        result["recurring_reminders"] = _normalize_rows(payload.get("recurring_reminders", []), REMINDER_DEFAULTS, "label")
        result["household_notes"] = _clean_text(payload.get("household_notes", ""))
        return result

    def _summary_for_subject(self, records: list[dict[str, Any]], category: str, subject: str) -> dict[str, str]:
        relevant = [
            row for row in records
            if _record_is_active(row)
            and _clean_text(row.get("category", "")).lower() == category
            and _name_key(row.get("subject", "")) == _name_key(subject)
        ]
        if category == "profile":
            out = dict(PROFILE_DEFAULTS)
            for field in PROFILE_DEFAULTS:
                idx = self._find_record_index(relevant, category=category, subject=subject, field=field)
                if idx >= 0:
                    out[field] = _clean_text(relevant[idx].get("value", ""))
            return out
        if category == "family":
            out = dict(FAMILY_DEFAULTS)
            out["name"] = _display_name(subject)
            for field in FAMILY_DEFAULTS:
                idx = self._find_record_index(relevant, category=category, subject=subject, field=field)
                if idx >= 0:
                    out[field] = _clean_text(relevant[idx].get("value", ""))
            return out
        if category == "pet":
            out = dict(PET_DEFAULTS)
            out["name"] = _display_name(subject)
            for field in PET_DEFAULTS:
                idx = self._find_record_index(relevant, category=category, subject=subject, field=field)
                if idx >= 0:
                    out[field] = _clean_text(relevant[idx].get("value", ""))
            return out
        if category == "reminder":
            out = dict(REMINDER_DEFAULTS)
            out["label"] = _clean_text(subject)
            for field in REMINDER_DEFAULTS:
                idx = self._find_record_index(relevant, category=category, subject=subject, field=field)
                if idx >= 0:
                    out[field] = _clean_text(relevant[idx].get("value", ""))
            return out
        return {}

    def snapshot_from_records(self, records: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        active = [row for row in (records if records is not None else self._load()) if _record_is_active(row)]
        payload = self.normalize_snapshot({})
        payload["user_profile"] = self._summary_for_subject(active, "profile", "user")

        family_subjects = sorted({_display_name(row.get("subject", "")) for row in active if _clean_text(row.get("category", "")).lower() == "family" and _clean_text(row.get("subject", ""))}, key=lambda value: _name_key(value))
        payload["family_members"] = [row for row in (self._summary_for_subject(active, "family", subject) for subject in family_subjects) if row.get("name")]

        pet_subjects = sorted({_display_name(row.get("subject", "")) for row in active if _clean_text(row.get("category", "")).lower() == "pet" and _clean_text(row.get("subject", ""))}, key=lambda value: _name_key(value))
        payload["pets"] = [row for row in (self._summary_for_subject(active, "pet", subject) for subject in pet_subjects) if row.get("name")]

        reminder_subjects = sorted({_clean_text(row.get("subject", "")) for row in active if _clean_text(row.get("category", "")).lower() == "reminder" and _clean_text(row.get("subject", ""))}, key=lambda value: value.lower())
        payload["recurring_reminders"] = [row for row in (self._summary_for_subject(active, "reminder", subject) for subject in reminder_subjects) if row.get("label")]

        idx = self._find_record_index(active, category="household", subject="household", field="notes")
        payload["household_notes"] = _clean_text(active[idx].get("value", "")) if idx >= 0 else ""
        return self.normalize_snapshot(payload)

    def _incoming_should_override(self, current: dict[str, Any], incoming: dict[str, Any]) -> bool:
        current_value = _clean_text(current.get("value", ""))
        incoming_value = _clean_text(incoming.get("value", ""))
        if not current_value:
            return True
        if current_value == incoming_value:
            return (
                STATUS_PRIORITY.get(_clean_text(incoming.get("status", "captured")).lower(), 0),
                SOURCE_PRIORITY.get(_clean_text(incoming.get("source_type", "unknown")).lower(), 0),
                _normalize_confidence(incoming.get("confidence", 0.7)),
            ) > (
                STATUS_PRIORITY.get(_clean_text(current.get("status", "captured")).lower(), 0),
                SOURCE_PRIORITY.get(_clean_text(current.get("source_type", "unknown")).lower(), 0),
                _normalize_confidence(current.get("confidence", 0.7)),
            )
        if _clean_text(incoming.get("source_type", "")).lower() == "manual":
            return True
        if _clean_text(current.get("status", "")).lower() == "forgotten":
            return True
        return (
            SOURCE_PRIORITY.get(_clean_text(incoming.get("source_type", "unknown")).lower(), 0) > SOURCE_PRIORITY.get(_clean_text(current.get("source_type", "unknown")).lower(), 0)
            or STATUS_PRIORITY.get(_clean_text(incoming.get("status", "captured")).lower(), 0) > STATUS_PRIORITY.get(_clean_text(current.get("status", "captured")).lower(), 0)
            or _normalize_confidence(incoming.get("confidence", 0.7)) >= _normalize_confidence(current.get("confidence", 0.7)) + 0.15
        )

    def merge_record(self, records: list[dict[str, Any]], incoming: dict[str, Any]) -> dict[str, Any]:
        record = self._normalize_record(incoming)
        idx = self._find_record_index(records, category=record["category"], subject=record["subject"], field=record["field"])
        if idx < 0:
            records.append(record)
            return record
        current = records[idx]
        merged = dict(current)
        merged["updated_at"] = _now_iso()
        if self._incoming_should_override(current, record):
            merged["value"] = record["value"]
            merged["status"] = record["status"]
            merged["confidence"] = record["confidence"]
            merged["source_type"] = record["source_type"]
            merged["source_label"] = record["source_label"]
            merged["source_ref"] = record["source_ref"]
            merged["evidence"] = record["evidence"]
            merged["tags"] = _normalize_tags(list(current.get("tags", [])) + list(record.get("tags", [])))
        else:
            merged["confidence"] = max(_normalize_confidence(current.get("confidence", 0.7)), _normalize_confidence(record.get("confidence", 0.7)))
            merged["tags"] = _normalize_tags(list(current.get("tags", [])) + list(record.get("tags", [])))
        if merged["status"] in {"confirmed", "pinned"} and not _clean_text(merged.get("last_confirmed_at", "")):
            merged["last_confirmed_at"] = merged["updated_at"]
        records[idx] = self._normalize_record(merged)
        return records[idx]

    def sync_from_snapshot(
        self,
        snapshot: dict[str, Any],
        *,
        source_type: str,
        source_label: str,
        status: str,
        confidence: float,
        replace_scope: bool,
        authoritative_categories: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        records = self.ensure_records(self.normalize_snapshot({}))
        incoming = self._records_from_snapshot(snapshot, source_type=source_type, source_label=source_label, status=status, confidence=confidence)
        seen_keys = {_record_key(row.get("category", ""), row.get("subject", ""), row.get("field", "")) for row in incoming}
        authority = {str(item).strip().lower() for item in (authoritative_categories or set()) if str(item).strip()}
        if replace_scope:
            now = _now_iso()
            for idx, row in enumerate(records):
                if _clean_text(row.get("category", "")).lower() not in SYNCABLE_CATEGORIES:
                    continue
                key = _record_key(row.get("category", ""), row.get("subject", ""), row.get("field", ""))
                if key in seen_keys:
                    continue
                updated = dict(row)
                updated["status"] = "forgotten"
                updated["updated_at"] = now
                records[idx] = self._normalize_record(updated)
        elif authority:
            allowed_subjects: dict[str, set[str]] = {category: set() for category in authority}
            for row in incoming:
                category = _clean_text(row.get("category", "")).lower()
                if category not in authority:
                    continue
                allowed_subjects.setdefault(category, set()).add(_name_key(row.get("subject", "")))
            now = _now_iso()
            for idx, row in enumerate(records):
                category = _clean_text(row.get("category", "")).lower()
                if category not in authority:
                    continue
                subject_key = _name_key(row.get("subject", ""))
                if subject_key in allowed_subjects.get(category, set()):
                    continue
                updated = dict(row)
                updated["status"] = "forgotten"
                updated["updated_at"] = now
                records[idx] = self._normalize_record(updated)
        for row in incoming:
            self.merge_record(records, row)
        self._save(records)
        return records

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
        records = self.ensure_records(self.normalize_snapshot({}))
        stored = self.merge_record(
            records,
            self._build_record(
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
            ),
        )
        self._save(records)
        return deepcopy(stored)

    def update_record_status(self, record_id: str, *, status: str) -> dict[str, Any] | None:
        target = _clean_text(record_id)
        if not target:
            return None
        records = self._load()
        for idx, row in enumerate(records):
            if _clean_text(row.get("id", "")) != target:
                continue
            updated = dict(row)
            updated["status"] = status
            updated["updated_at"] = _now_iso()
            if status in {"confirmed", "pinned"}:
                updated["last_confirmed_at"] = updated["updated_at"]
            records[idx] = self._normalize_record(updated)
            self._save(records)
            return deepcopy(records[idx])
        return None

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
        target = _clean_text(record_id)
        if not target:
            return None
        records = self._load()
        for idx, row in enumerate(records):
            if _clean_text(row.get("id", "")) != target:
                continue
            updated = dict(row)
            if value is not None:
                updated["value"] = _clean_text(value)
            if status is not None:
                clean_status = _clean_text(status).lower()
                if clean_status in RECORD_STATUSES:
                    updated["status"] = clean_status
                    if clean_status in {"confirmed", "pinned"}:
                        updated["last_confirmed_at"] = _now_iso()
            if confidence is not None:
                updated["confidence"] = _normalize_confidence(confidence)
            if evidence is not None:
                updated["evidence"] = _clean_text(evidence)
            if tags is not None:
                updated["tags"] = _normalize_tags(tags)
            updated["updated_at"] = _now_iso()
            records[idx] = self._normalize_record(updated)
            self._save(records)
            return deepcopy(records[idx])
        return None

    def explain_record(self, record_id: str) -> dict[str, Any] | None:
        row = self.get_record(record_id)
        if row is None:
            return None
        return {
            "id": _clean_text(row.get("id", "")),
            "category": _clean_text(row.get("category", "")),
            "subject": _clean_text(row.get("subject", "")),
            "field": _clean_text(row.get("field", "")),
            "value": _clean_text(row.get("value", "")),
            "status": _clean_text(row.get("status", "")),
            "confidence": _normalize_confidence(row.get("confidence", 0.7)),
            "source_type": _clean_text(row.get("source_type", "")),
            "source_label": _clean_text(row.get("source_label", "")),
            "source_ref": _clean_text(row.get("source_ref", "")),
            "updated_at": _clean_text(row.get("updated_at", "")),
            "last_confirmed_at": _clean_text(row.get("last_confirmed_at", "")),
            "last_used_at": _clean_text(row.get("last_used_at", "")),
            "use_count": int(row.get("use_count", 0) or 0),
            "evidence": _clean_text(row.get("evidence", "")),
            "tags": list(row.get("tags", [])) if isinstance(row.get("tags", []), list) else [],
        }

    def mark_used(self, record_id: str) -> dict[str, Any] | None:
        target = _clean_text(record_id)
        if not target:
            return None
        records = self._load()
        for idx, row in enumerate(records):
            if _clean_text(row.get("id", "")) != target:
                continue
            updated = dict(row)
            updated["last_used_at"] = _now_iso()
            try:
                updated["use_count"] = max(0, int(updated.get("use_count", 0) or 0)) + 1
            except (TypeError, ValueError):
                updated["use_count"] = 1
            updated["updated_at"] = updated["last_used_at"]
            records[idx] = self._normalize_record(updated)
            self._save(records)
            return deepcopy(records[idx])
        return None
