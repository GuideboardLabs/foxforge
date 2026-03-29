from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from shared_tools.db import connect, row_to_dict


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _compact(text: str) -> str:
    return " ".join(str(text or "").strip().split())


class ProjectContextMemory:
    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root
        self.lock = Lock()

    def _project_key(self, project: str) -> str:
        return _compact(project).lower() or "general"

    def get_facts(self, project: str) -> dict[str, str]:
        key = self._project_key(project)
        with self.lock, connect(self.repo_root) as conn:
            rows = conn.execute(
                """
                SELECT fact_key, fact_value
                FROM project_facts
                WHERE project_slug = ?
                ORDER BY fact_key ASC;
                """.strip(),
                (key,),
            ).fetchall()
        return {str(row["fact_key"]): str(row["fact_value"]) for row in rows if str(row["fact_value"]).strip()}

    def get_updated_at(self, project: str) -> str:
        key = self._project_key(project)
        with self.lock, connect(self.repo_root) as conn:
            row = conn.execute(
                """
                SELECT MAX(updated_at) AS updated_at
                FROM project_facts
                WHERE project_slug = ?;
                """.strip(),
                (key,),
            ).fetchone()
        if row is None:
            return ""
        return str(row["updated_at"] or "").strip()

    def clear_project(self, project: str) -> bool:
        key = self._project_key(project)
        with self.lock, connect(self.repo_root) as conn:
            cur = conn.execute("DELETE FROM project_facts WHERE project_slug = ?;", (key,))
            conn.commit()
            deleted = cur.rowcount if cur.rowcount != -1 else 0
        return deleted > 0

    def upsert_fact(self, project: str, fact_key: str, fact_value: str, source: str = "derived") -> bool:
        key = self._project_key(project)
        normalized_key = _compact(fact_key)
        normalized_value = _compact(fact_value)
        normalized_source = _compact(source) or "derived"
        if not normalized_key or not normalized_value:
            return False

        now = _now_iso()
        with self.lock, connect(self.repo_root) as conn:
            conn.execute(
                """
                INSERT INTO project_facts(project_slug, fact_key, fact_value, source, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(project_slug, fact_key) DO UPDATE SET
                    fact_value = excluded.fact_value,
                    source = excluded.source,
                    updated_at = excluded.updated_at;
                """.strip(),
                (key, normalized_key, normalized_value, normalized_source, now),
            )
            conn.commit()
        return True

    def _extract_facts(self, text: str) -> dict[str, str]:
        low = text.lower()
        facts: dict[str, str] = {}

        m_weight = re.search(r"\b(\d{1,3}(?:\.\d+)?)\s*(?:lb|lbs|pound|pounds)\b", low)
        if m_weight:
            facts["dog_weight_lbs"] = m_weight.group(1)

        m_age_months = re.search(r"\b(\d{1,2})\s*(?:month|months)\s*old\b", low)
        if m_age_months:
            facts["dog_age_months"] = m_age_months.group(1)

        m_age_years = re.search(r"\b(\d{1,2})\s*(?:year|years)\s*old\b", low)
        if m_age_years:
            facts["dog_age_years"] = m_age_years.group(1)

        if "no known allergies" in low or "no allergies" in low:
            facts["dog_allergies"] = "none known"
        elif "allerg" in low:
            m_allergy = re.search(r"(?:allerg(?:y|ies|ic)?[^.:\n]{0,80})", text, re.IGNORECASE)
            if m_allergy:
                facts["dog_allergies"] = _compact(m_allergy.group(0))

        if "no health issues" in low or "healthy" in low or "no health conditions" in low:
            facts["dog_health_conditions"] = "none reported"
        elif "health condition" in low or "condition" in low:
            m_health = re.search(r"(?:health[^.:\n]{0,100})", text, re.IGNORECASE)
            if m_health:
                facts["dog_health_conditions"] = _compact(m_health.group(0))

        if "active" in low:
            if "very active" in low or "pretty active" in low:
                facts["dog_activity_level"] = "high"
            elif "not active" in low or "low activity" in low:
                facts["dog_activity_level"] = "low"
            else:
                facts["dog_activity_level"] = "moderate"

        m_brand = re.search(
            r"\b(iams|purina|royal canin|blue buffalo|hill'?s|orijen|wellness|pedigree|eukanuba)\b",
            text,
            re.IGNORECASE,
        )
        if m_brand:
            facts["current_food_brand"] = _compact(m_brand.group(1)).upper().replace("'", "")

        m_budget = re.search(r"\b(\d{1,3})\s*-\s*(\d{1,3})\b", low)
        if m_budget:
            facts["budget_range"] = f"{m_budget.group(1)}-{m_budget.group(2)}"
        elif "budget" in low:
            m_budget_line = re.search(r"(?:budget[^.:\n]{0,120})", text, re.IGNORECASE)
            if m_budget_line:
                facts["budget_notes"] = _compact(m_budget_line.group(0))

        m_mix = re.search(r"\b([a-z][a-z \-]{2,80}mix)\b", low)
        if m_mix and any(token in m_mix.group(1) for token in ("terrier", "pit", "russell", "amstaff", "staff")):
            facts["dog_breed_mix"] = _compact(m_mix.group(1))
        else:
            breed_hits = []
            for token in [
                "jack russell",
                "amstaff",
                "pit terrier",
                "staffordshire",
                "terrier",
                "labrador",
                "golden retriever",
                "german shepherd",
                "poodle",
                "beagle",
            ]:
                if token in low:
                    breed_hits.append(token)
            if breed_hits:
                facts["dog_breed_mix"] = ", ".join(dict.fromkeys(breed_hits))

        m_location = re.search(
            r"\b(?:in|located in)\s+([A-Za-z][A-Za-z ]{1,40},?\s*(?:[A-Za-z]{2}|New York|California|Texas|Florida))\b",
            text,
            re.IGNORECASE,
        )
        if m_location:
            facts["location"] = _compact(m_location.group(1))

        return facts

    def ingest_text(self, project: str, text: str, source: str = "derived") -> dict[str, Any]:
        key = self._project_key(project)
        body = _compact(text)
        if not body:
            return {"project": key, "updated": 0, "facts": self.get_facts(key)}
        updates = self._extract_facts(body)
        if not updates:
            return {"project": key, "updated": 0, "facts": self.get_facts(key)}

        changed = 0
        existing = self.get_facts(key)
        for fact_key, fact_value in updates.items():
            normalized_value = _compact(fact_value)
            if not normalized_value:
                continue
            if str(existing.get(fact_key, "")).strip() == normalized_value:
                continue
            if self.upsert_fact(key, fact_key, normalized_value, source=source):
                changed += 1

        return {"project": key, "updated": changed, "facts": self.get_facts(key)}

    def refresh_from_history(self, project: str, history: list[dict[str, str]] | None, reset: bool = True) -> dict[str, Any]:
        key = self._project_key(project)
        rows = history if isinstance(history, list) else []
        if reset:
            self.clear_project(key)
        scanned_user_messages = 0
        updated_fields_total = 0
        for row in rows:
            if not isinstance(row, dict):
                continue
            role = str(row.get("role", "")).strip().lower()
            if role != "user":
                continue
            content = str(row.get("content", "")).strip()
            if not content:
                continue
            scanned_user_messages += 1
            result = self.ingest_text(key, content, source="history_refresh")
            updated_fields_total += int(result.get("updated", 0))
        facts = self.get_facts(key)
        return {
            "project": key,
            "scanned_user_messages": scanned_user_messages,
            "updated_fields": updated_fields_total,
            "facts_count": len(facts),
            "updated_at": self.get_updated_at(key),
            "facts": facts,
        }

    def summary_text(self, project: str, limit_chars: int = 1800) -> str:
        facts = self.get_facts(project)
        if not facts:
            return ""
        labels = {
            "dog_weight_lbs": "Dog weight (lbs)",
            "dog_age_months": "Dog age (months)",
            "dog_age_years": "Dog age (years)",
            "dog_breed_mix": "Breed/mix",
            "dog_activity_level": "Activity level",
            "dog_allergies": "Allergies",
            "dog_health_conditions": "Health conditions",
            "current_food_brand": "Current food brand",
            "budget_range": "Budget range",
            "budget_notes": "Budget notes",
            "location": "Location",
        }
        lines = ["Known project facts from prior user answers (reuse these, do not re-ask unless conflicting):"]
        for fact_key, fact_value in facts.items():
            label = labels.get(fact_key, fact_key.replace("_", " ").title())
            lines.append(f"- {label}: {fact_value}")
        text = "\n".join(lines)
        if len(text) <= limit_chars:
            return text
        return text[: max(300, limit_chars)].rsplit("\n", 1)[0]

    def export_project_rows(self, project: str) -> list[dict[str, Any]]:
        key = self._project_key(project)
        with self.lock, connect(self.repo_root) as conn:
            rows = conn.execute(
                """
                SELECT project_slug, fact_key, fact_value, source, updated_at
                FROM project_facts
                WHERE project_slug = ?
                ORDER BY fact_key ASC;
                """.strip(),
                (key,),
            ).fetchall()
        return [row_to_dict(row) or {} for row in rows]
