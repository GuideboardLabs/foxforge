from __future__ import annotations

import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

_MONTHS = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|sept|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
_DATE_RE = re.compile(rf"\b{_MONTHS}\s+\d{{1,2}},?\s+(?:19|20)\d{{2}}\b", re.IGNORECASE)


def detect_topic_type(query: str, topic_type: str = "general") -> str:
    low = f"{topic_type} {query}".lower()
    if any(token in low for token in ["ufc", "mma", "boxing", "fight", "main card", "weigh-in"]):
        return "combat_sports"
    if any(token in low for token in ["matchup", "odds", "sportsbook", "draftkings", "moneyline", "spread", "kickoff", "tipoff"]):
        return "sports_event"
    if " vs " in low and any(token in low for token in ["nba", "nfl", "mlb", "nhl", "soccer", "football", "boxing", "mma", "ufc"]):
        return "sports_event"
    if any(
        token in low
        for token in [
            "animal care",
            "pet care",
            "veterinary",
            "veterinarian",
            "vet ",
            "dog ",
            "cat ",
            "kitten",
            "puppy",
            "pet food",
            "pet vaccine",
            "flea",
            "tick",
            "neuter",
            "spay",
        ]
    ):
        return "animal_care"
    return str(topic_type or "general")


def classify_fact_volatility(query: str, topic_type: str = "general", text: str = "") -> str:
    low = f"{query} {text} {topic_type}".lower()
    resolved_type = detect_topic_type(query, topic_type)
    if resolved_type == "technical":
        if any(token in low for token in ["cve", "security advisory", "zero-day", "hotfix", "outage"]):
            return "volatile"
        if any(token in low for token in ["release", "changelog", "version", "api", "deprecation", "compatibility"]):
            return "semi_volatile"
    if resolved_type == "finance":
        if any(token in low for token in ["price", "quote", "yield", "cpi", "inflation", "fomc", "fed funds"]):
            return "volatile"
        if any(token in low for token in ["earnings", "guidance", "10-k", "10-q", "sec filing", "analyst"]):
            return "semi_volatile"
    if resolved_type == "current_events":
        if any(token in low for token in ["today", "tonight", "live", "breaking", "developing", "latest"]):
            return "volatile"
        return "semi_volatile"
    if resolved_type == "law":
        if any(token in low for token in ["breaking", "injunction", "emergency order", "today"]):
            return "volatile"
        if any(token in low for token in ["effective date", "ruling", "bill", "statute", "regulation", "deadline"]):
            return "semi_volatile"
    if resolved_type == "education":
        if any(token in low for token in ["admissions deadline", "application deadline", "enrollment window"]):
            return "volatile"
        if any(token in low for token in ["tuition", "curriculum", "accreditation", "requirements"]):
            return "semi_volatile"
    if resolved_type == "travel":
        if any(token in low for token in ["travel advisory", "entry restriction", "flight status", "border closure"]):
            return "volatile"
        if any(token in low for token in ["visa", "entry requirement", "passport", "layover"]):
            return "semi_volatile"
    if resolved_type == "animal_care":
        if any(token in low for token in ["recall", "outbreak", "toxicity", "poison", "emergency", "urgent", "aspca alert"]):
            return "volatile"
        if any(token in low for token in ["vaccine schedule", "dosage", "flea", "tick", "nutrition", "spay", "neuter", "parasite"]):
            return "semi_volatile"
    if resolved_type == "food":
        if any(token in low for token in ["recall", "outbreak", "contamination"]):
            return "volatile"
        if any(token in low for token in ["nutrition", "ingredients", "allergen", "serving size"]):
            return "semi_volatile"
    if resolved_type == "books":
        if any(token in low for token in ["release date", "publication date", "preorder"]):
            return "semi_volatile"
    if resolved_type == "parenting":
        if any(token in low for token in ["product recall", "safety alert", "outbreak"]):
            return "volatile"
        if any(token in low for token in ["age recommendation", "milestone", "dosage", "schedule"]):
            return "semi_volatile"
    if resolved_type == "business":
        if any(token in low for token in ["earnings call", "guidance cut", "ceo resignation", "merger talks"]):
            return "volatile"
        if any(token in low for token in ["quarterly results", "revenue", "margin", "market share", "forecast"]):
            return "semi_volatile"
    if resolved_type == "real_estate":
        if any(token in low for token in ["mortgage rate", "fed decision", "housing shock", "market crash"]):
            return "volatile"
        if any(token in low for token in ["inventory", "median price", "rent", "cap rate", "housing starts"]):
            return "semi_volatile"
    if resolved_type == "gaming":
        if any(token in low for token in ["server outage", "hotfix", "launch delay", "live patch"]):
            return "volatile"
        if any(token in low for token in ["patch notes", "season", "meta", "release window", "dlc"]):
            return "semi_volatile"
    if resolved_type == "automotive":
        if any(token in low for token in ["recall", "safety defect", "stop sale"]):
            return "volatile"
        if any(token in low for token in ["msrp", "range", "mpg", "trim", "model year"]):
            return "semi_volatile"
    if resolved_type == "tv_shows":
        if any(token in low for token in ["cancelled", "renewed", "release moved", "airing tonight"]):
            return "volatile"
        if any(token in low for token in ["episode list", "season release", "showrunner", "cast"]):
            return "semi_volatile"
    if resolved_type == "movies":
        if any(token in low for token in ["opening weekend", "release delay", "festival win"]):
            return "volatile"
        if any(token in low for token in ["box office", "release date", "runtime", "cast", "trailer"]):
            return "semi_volatile"
    if resolved_type == "music":
        if any(token in low for token in ["surprise drop", "tour cancellation", "chart update today"]):
            return "volatile"
        if any(token in low for token in ["album release", "single", "tour dates", "tracklist", "label"]):
            return "semi_volatile"
    if resolved_type == "art":
        if any(token in low for token in ["auction result", "sale price", "exhibition opens today"]):
            return "volatile"
        if any(token in low for token in ["exhibition date", "museum", "curator", "provenance", "artist statement"]):
            return "semi_volatile"
    if any(token in low for token in ["odds", "line", "favorite", "stream", "watch", "time", "today", "tonight", "weigh-in", "live"]):
        return "volatile"
    if any(token in low for token in ["record", "ranking", "injury", "card", "fight date", "broadcast"]):
        return "semi_volatile"
    return "stable"


def freshness_requirement_hours(volatility_class: str) -> int | None:
    if volatility_class == "volatile":
        return 72
    if volatility_class == "semi_volatile":
        return 24 * 180
    return None


def _parse_datetime(value: str) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        dt = datetime.fromisoformat(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        pass
    try:
        dt = parsedate_to_datetime(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _extract_inline_date(text: str) -> datetime | None:
    match = _DATE_RE.search(str(text or ""))
    if not match:
        return None
    chunk = match.group(0)
    for fmt in ["%B %d %Y", "%B %d, %Y", "%b %d %Y", "%b %d, %Y"]:
        try:
            dt = datetime.strptime(chunk.replace("  ", " "), fmt)
            return dt.replace(tzinfo=timezone.utc)
        except Exception:
            continue
    return None


def score_source_freshness(source: dict[str, Any], volatility_class: str, now: datetime | None = None) -> tuple[float, float | None]:
    now = now or datetime.now(timezone.utc)
    dt = (
        _parse_datetime(str(source.get("published_at", "")))
        or _parse_datetime(str(source.get("retrieved_at", "")))
        or _extract_inline_date(f"{source.get('title', '')} {source.get('snippet', '')}")
    )
    if dt is None:
        return 0.45 if volatility_class != "volatile" else 0.2, None
    age_hours = max(0.0, (now - dt).total_seconds() / 3600.0)
    requirement = freshness_requirement_hours(volatility_class)
    if requirement is None:
        if age_hours <= 24 * 365:
            return 0.9, age_hours
        return 0.7, age_hours
    if age_hours <= requirement:
        return 1.0, age_hours
    if age_hours <= requirement * 2:
        return 0.65, age_hours
    if age_hours <= requirement * 8:
        return 0.35, age_hours
    return 0.1, age_hours


def source_type_for_domain(domain: str) -> str:
    host = str(domain or "").lower()
    if any(x in host for x in ["ufcstats", "ufc.com", ".gov", ".edu"]):
        return "official"
    if any(x in host for x in ["draftkings", "fanduel", "betmgm"]):
        return "betting"
    if any(x in host for x in ["wikipedia", "tapology", "sherdog"]):
        return "reference"
    return "news"


def enrich_source_metadata(source: dict[str, Any], query: str, topic_type: str = "general") -> dict[str, Any]:
    payload = dict(source)
    volatility = classify_fact_volatility(query, topic_type, f"{payload.get('title', '')} {payload.get('snippet', '')}")
    freshness_score, age_hours = score_source_freshness(payload, volatility)
    payload["volatility_class"] = volatility
    payload["freshness_score"] = round(float(freshness_score), 3)
    payload["source_age_hours"] = None if age_hours is None else round(float(age_hours), 2)
    payload["source_type"] = source_type_for_domain(str(payload.get("source_domain", "")))
    payload["volatility_fit_score"] = round(float(freshness_score), 3)
    payload["stale_for_query"] = bool(age_hours is not None and freshness_requirement_hours(volatility) is not None and age_hours > float(freshness_requirement_hours(volatility) or 0))
    return payload
