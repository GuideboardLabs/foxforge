from __future__ import annotations

import html
import json
import os
import re
import time
from collections import deque
from html.parser import HTMLParser
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from shared_tools.file_store import ProjectStore
from shared_tools.fact_policy import enrich_source_metadata, detect_topic_type, classify_fact_volatility
from shared_tools.domain_reputation import DomainReputation


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class _PageExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []
        self.links: list[str] = []
        self._in_title = False
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        key = tag.lower()
        if key in {"script", "style", "noscript", "svg", "nav", "header", "footer", "aside", "form", "iframe", "menu"}:
            self._skip_depth += 1
            return
        if key == "title":
            self._in_title = True
            return
        if key != "a":
            return
        for name, value in attrs:
            if name and name.lower() == "href" and value:
                self.links.append(value.strip())
                break

    def handle_endtag(self, tag: str) -> None:
        key = tag.lower()
        if key in {"script", "style", "noscript", "svg", "nav", "header", "footer", "aside", "form", "iframe", "menu"}:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if key == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return
        text = " ".join(str(data or "").split())
        if not text:
            return
        if self._in_title:
            self.title_parts.append(text)
        self.text_parts.append(text)

    def title(self) -> str:
        return " ".join(self.title_parts).strip()

    def snippet(self, max_chars: int = 600) -> str:
        text = " ".join(self.text_parts).strip()
        if len(text) <= max_chars:
            return text
        cut = text[:max_chars].rsplit(" ", 1)[0].strip()
        return (cut or text[:max_chars]).strip() + "..."


_MD_LINK_ONLY_RE = re.compile(r"^\s*(\[([^\]]*)\]\([^)]*\)\s*[|•·\-,]?\s*)+\s*$")
_BOILERPLATE_LOWER = (
    "subscribe to our newsletter",
    "sign up for our newsletter",
    "this site uses cookies",
    "we use cookies",
    "accept all cookies",
    "cookie preferences",
    "manage cookies",
    "skip to main content",
    "skip to content",
    "all rights reserved",
    "share on twitter",
    "share on facebook",
    "share on linkedin",
)


def _clean_crawl4ai_markdown(text: str) -> str:
    """Strip navigation link menus, cookie banners, and share-button boilerplate from Crawl4AI markdown."""
    if not text:
        return text
    lines = text.split("\n")
    cleaned: list[str] = []
    nav_run = 0
    for line in lines:
        stripped = line.strip()
        # Lines composed entirely of markdown links — navigation menus, breadcrumbs
        if stripped and _MD_LINK_ONLY_RE.match(stripped):
            nav_run += 1
            if nav_run <= 1:
                cleaned.append(line)  # keep first link (may be a real article ref)
            continue
        else:
            nav_run = 0
        # Short boilerplate phrases
        low = stripped.lower()
        if len(stripped) < 130 and any(p in low for p in _BOILERPLATE_LOWER):
            continue
        cleaned.append(line)
    # Collapse 3+ blank lines → 2
    result: list[str] = []
    blanks = 0
    for line in cleaned:
        if not line.strip():
            blanks += 1
            if blanks <= 2:
                result.append(line)
        else:
            blanks = 0
            result.append(line)
    return "\n".join(result)


def build_web_progress_payload(result: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize a web-run result into a compact progress payload for the UI."""
    details = result if isinstance(result, dict) else {}
    tier_counts = {}
    scoring = details.get("source_scoring_summary")
    if isinstance(scoring, dict):
        maybe_tiers = scoring.get("tier_counts")
        if isinstance(maybe_tiers, dict):
            tier_counts = maybe_tiers
    raw_sources = details.get("sources") or []
    top_sources: list[dict[str, Any]] = []
    seen_domains: set[str] = set()
    for src in raw_sources:
        if not isinstance(src, dict):
            continue
        url = str(src.get("url") or src.get("source_url") or "").strip()
        domain = str(src.get("source_domain") or src.get("domain") or "").strip().lower()
        if not domain and url:
            try:
                from urllib.parse import urlparse
                domain = urlparse(url).hostname or ""
                domain = domain.removeprefix("www.")
            except Exception:
                pass
        if not domain or domain in seen_domains:
            continue
        seen_domains.add(domain)
        top_sources.append({"domain": domain, "url": url or f"https://{domain}"})
        if len(top_sources) >= 8:
            break
    return {
        "note": "Web stack ready.",
        "mode": str(details.get("mode", "")),
        "source_count": int(details.get("source_count", 0) or 0),
        "seed_count": int(details.get("seed_count", 0) or 0),
        "crawl_pages": int(details.get("crawl_pages", 0) or 0),
        "crawl_gated_links": int(details.get("crawl_gated_links", 0) or 0),
        "query_variants_count": int(details.get("query_variants_count", 0) or 0),
        "conflict_count": int(details.get("conflict_count", 0) or 0),
        "tier1": int(tier_counts.get("tier1", 0) or 0),
        "tier2": int(tier_counts.get("tier2", 0) or 0),
        "tier3": int(tier_counts.get("tier3", 0) or 0),
        "web_sources": top_sources,
    }


class WebResearchEngine:
    VALID_MODES = {"off", "ask", "auto"}
    VALID_PROVIDERS = {"auto", "searxng", "duckduckgo_html", "duckduckgo_api"}
    TRUST_TIER_1 = {
        "reuters.com",
        "apnews.com",
        "bbc.com",
        "nytimes.com",
        "wsj.com",
        "economist.com",
        "ft.com",
        "espn.com",
        "nasa.gov",
        "noaa.gov",
        "cdc.gov",
        "nih.gov",
        "who.int",
        "sec.gov",
        "federalreserve.gov",
        "wikipedia.org",
    }
    TRUST_TIER_2 = {
        "forbes.com",
        "bloomberg.com",
        "cnbc.com",
        "theguardian.com",
        "axios.com",
        "verge.com",
        "techcrunch.com",
        "arstechnica.com",
        "github.com",
        "stackoverflow.com",
        "reddit.com",
        "x.com",
        "twitter.com",
        "medium.com",
        "substack.com",
        "linkedin.com",
        "canva.com",
    }
    PROPAGANDA_TERMS = {
        "shocking", "you won't believe", "exposed", "bombshell", "destroyed",
        "humiliated", "secret agenda", "mainstream media won't", "cover-up",
        "they don't want you to know", "leaked", "breaking truth",
    }
    LOW_SIGNAL_TEXT_TERMS = {
        "unsupported client",
        "unsupported browser",
        "please update your browser",
        "enable javascript",
        "please enable javascript",
        "enable cookies",
        "verify you are human",
        "checking your browser",
        "access denied",
        "request blocked",
        "forbidden",
        "service unavailable",
        "temporarily unavailable",
        "captcha",
        "are you a robot",
        "cloudflare",
        "security check",
        "browser is not supported",
    }
    LOW_SIGNAL_URL_TERMS = {
        "/cdn-cgi/",
        "/captcha",
        "/challenge",
        "/unsupported",
        "/error",
        "/forbidden",
        "/access-denied",
        "/blocked",
        "cf_chl",
        "cf-chl",
    }
    NAVIGATION_NOISE_TERMS = {
        "skip to content",
        "book a demo",
        "privacy policy",
        "terms of service",
        "all rights reserved",
        "cookie preferences",
        "log in",
        "sign up",
        "create account",
    }

    TRUST_TIER_1_SPORTS = {
        "mmafighting.com",
        "bloodyelbow.com",
        "sherdog.com",
        "combatpress.com",
        "tapology.com",
    }

    TRUST_TIER_2_INDIE = {
        "defector.com",
        "propublica.org",
        "theintercept.com",
        "404media.co",
        "therealnews.com",
        "unherd.com",
    }

    # Academic / peer-reviewed — treat as tier1 for factual claims
    TRUST_TIER_1_ACADEMIC = {
        "arxiv.org",
        "pubmed.ncbi.nlm.nih.gov",
        "ncbi.nlm.nih.gov",
        "nature.com",
        "science.org",
        "plos.org",
        "jstor.org",
        "scholar.google.com",
        "semanticscholar.org",
        "biorxiv.org",
        "medrxiv.org",
    }

    # Legal / court records — high-trust primary sources
    TRUST_TIER_1_LEGAL = {
        "law.cornell.edu",
        "oyez.org",
        "scotusblog.com",
        "supremecourt.gov",
        "uscourts.gov",
        "congress.gov",
        "regulations.gov",
    }

    # Mainstream sports (non-MMA) — established beat coverage
    TRUST_TIER_2_MAINSTREAM_SPORTS = {
        "theathletic.com",
        "bleacherreport.com",
        "si.com",
        "basketball-reference.com",
        "baseball-reference.com",
        "pro-football-reference.com",
        "nfl.com",
        "nba.com",
        "mlb.com",
        "nhl.com",
        "skysports.com",
        "goal.com",
    }

    # Prosumer / hobbyist tech — hands-on, independent testing
    TRUST_TIER_2_PROSUMER_TECH = {
        "hackaday.com",
        "tomshardware.com",
        "ifixit.com",
        "rtings.com",
        "notebookcheck.net",
        "makezine.com",
        "instructables.com",
        "thingiverse.com",
        "lttreviews.com",
        "techpowerup.com",
    }

    # Gaming / esports editorial
    TRUST_TIER_2_GAMING = {
        "ign.com",
        "eurogamer.net",
        "pcgamer.com",
        "rockpapershotgun.com",
        "giantbomb.com",
        "gamespot.com",
        "kotaku.com",
        "polygon.com",
        "vg247.com",
    }

    # Film / TV criticism and records
    TRUST_TIER_2_FILM_TV = {
        "imdb.com",
        "rottentomatoes.com",
        "letterboxd.com",
        "criterion.com",
        "rogerebert.com",
        "avclub.com",
    }

    # Music criticism and cataloguing
    TRUST_TIER_2_MUSIC = {
        "pitchfork.com",
        "allmusic.com",
        "discogs.com",
        "rateyourmusic.com",
        "genius.com",
        "stereogum.com",
    }

    # Health / clinical consumer
    TRUST_TIER_2_HEALTH = {
        "mayoclinic.org",
        "clevelandclinic.org",
        "healthline.com",
        "webmd.com",
        "medicalnewstoday.com",
        "nhs.uk",
        "hopkinsmedicine.org",
    }

    # Finance / retail investing
    TRUST_TIER_2_FINANCE = {
        "investopedia.com",
        "morningstar.com",
        "marketwatch.com",
        "seekingalpha.com",
        "fool.com",
        "bankrate.com",
    }
    TRUST_TIER_2_BUSINESS = {
        "hbr.org",
        "mckinsey.com",
        "entrepreneur.com",
        "inc.com",
        "fastcompany.com",
    }
    TRUST_TIER_2_REAL_ESTATE = {
        "zillow.com",
        "redfin.com",
        "realtor.com",
        "apartments.com",
        "co-star.com",
    }
    TRUST_TIER_2_AUTOMOTIVE = {
        "caranddriver.com",
        "motortrend.com",
        "edmunds.com",
        "kbb.com",
        "cars.com",
    }
    TRUST_TIER_2_ART = {
        "artsy.net",
        "moma.org",
        "metmuseum.org",
        "tate.org.uk",
        "smithsonianmag.com",
    }
    TRUST_TIER_2_LEGAL = {
        "justia.com",
        "findlaw.com",
        "canlii.org",
    }
    TRUST_TIER_2_EDUCATION = {
        "coursera.org",
        "edx.org",
        "khanacademy.org",
        "collegeboard.org",
    }
    TRUST_TIER_2_TRAVEL = {
        "tripadvisor.com",
        "lonelyplanet.com",
        "rome2rio.com",
        "seatguru.com",
    }
    TRUST_TIER_2_FOOD = {
        "allrecipes.com",
        "seriouseats.com",
        "nutritionix.com",
        "eatright.org",
    }
    TRUST_TIER_2_BOOKS = {
        "goodreads.com",
        "publishersweekly.com",
        "kirkusreviews.com",
    }
    TRUST_TIER_2_PARENTING = {
        "healthychildren.org",
        "zerotothree.org",
        "parents.com",
    }
    TRUST_TIER_2_ANIMAL_CARE = {
        "avma.org",
        "aaha.org",
        "merckvetmanual.com",
        "vcahospitals.com",
        "aspca.org",
        "akc.org",
        "petmd.com",
    }
    TOPIC_FAMILY_ALIASES = {
        "pet_care": "animal_care",
        "pets": "animal_care",
        "pet_health": "animal_care",
        "veterinary": "animal_care",
        "vet": "animal_care",
    }
    TOPIC_HINTS = {
        "technical": (
            "official documentation",
            "release notes",
            "version compatibility",
            "changelog",
        ),
        "finance": (
            "sec filing",
            "earnings release",
            "guidance update",
            "analyst consensus",
        ),
        "current_events": (
            "official statement",
            "timeline",
            "live updates",
            "breaking news",
        ),
        "law": (
            "statute text",
            "court ruling",
            "effective date",
            "official guidance",
        ),
        "education": (
            "admissions deadline",
            "accreditation status",
            "official program page",
            "curriculum requirements",
        ),
        "travel": (
            "entry requirements",
            "visa rules",
            "travel advisory",
            "official guidance",
        ),
        "animal_care": (
            "veterinary guidance",
            "species specific recommendations",
            "animal welfare guidance",
            "official guidance",
        ),
        "food": (
            "nutrition facts",
            "ingredient list",
            "food safety guidance",
            "official guidance",
        ),
        "books": (
            "publication date",
            "edition details",
            "publisher announcement",
            "author interview",
        ),
        "parenting": (
            "pediatric guideline",
            "age recommendation",
            "development milestone",
            "safety guidance",
        ),
        "business": (
            "quarterly results",
            "management guidance",
            "industry outlook",
            "official filing",
        ),
        "real_estate": (
            "mortgage rates",
            "housing inventory",
            "median home price",
            "official market report",
        ),
        "gaming": (
            "patch notes",
            "release date",
            "developer update",
            "season roadmap",
        ),
        "automotive": (
            "msrp",
            "recall notice",
            "range mpg",
            "official spec sheet",
        ),
        "tv_shows": (
            "season release date",
            "episode schedule",
            "official network announcement",
            "renewed cancelled",
        ),
        "movies": (
            "release date",
            "box office",
            "official trailer",
            "festival premiere",
        ),
        "music": (
            "album release date",
            "tour dates",
            "official announcement",
            "chart update",
        ),
        "art": (
            "exhibition dates",
            "museum announcement",
            "auction result",
            "artist statement",
        ),
    }

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root
        self.store = ProjectStore(repo_root)
        self.root = repo_root / "Runtime" / "web"
        self.pending_path = self.root / "pending_requests.json"
        self.settings_path = self.root / "settings.json"
        self.sources_log_path = self.root / "sources.jsonl"
        self.lock = Lock()
        self._searxng_backoff_until = 0.0
        self._crawl4ai_backoff_until = 0.0
        self._tor_active = False  # Set True during underground run_query execution

        self.root.mkdir(parents=True, exist_ok=True)
        self._domain_rep = DomainReputation(repo_root)
        if not self.pending_path.exists():
            self.pending_path.write_text("[]", encoding="utf-8")
        if not self.settings_path.exists():
            self.settings_path.write_text(
                json.dumps(
                    {
                        "mode": "auto",
                        "provider": "auto",
                        "max_results": 8,
                        "query_expansion_enabled": True,
                        "query_expansion_variants": 4,
                        "source_scoring_enabled": True,
                        "min_quality_sources": 2,
                        "context_min_source_score": 0.52,
                        "conflict_detection_enabled": True,
                        "crawl_relevance_gating_enabled": False,
                        "crawl_relevance_min_score": 0.1,
                        "fact_check_enabled": False,
                        "fact_check_provider": "local",
                        "searxng_base_url": "http://127.0.0.1:8080",
                        "searxng_timeout_sec": 20,
                        "searxng_engines": "",
                        "searxng_categories": "",
                        "searxng_language": "",
                        "crawl_enabled": True,
                        "crawl_depth": 2,
                        "crawl_max_pages": 18,
                        "crawl_links_per_page": 8,
                        "crawl_timeout_sec": 0,
                        "crawl4ai_enabled": True,
                        "crawl4ai_base_url": "http://127.0.0.1:11235",
                        "crawl4ai_timeout_sec": 40,
                        "crawl4ai_retry_attempts": 2,
                        "crawl4ai_css_selector": "article,main,p",
                        "newspaper_enabled": True,
                        "newspaper_language": "",
                        "search_retry_attempts": 3,
                        "crawl_retry_attempts": 3,
                        "crawl_same_domain_only": True,
                        "crawl_text_chars": 800,
                    },
                    indent=2,
                    ensure_ascii=True,
                ),
                encoding="utf-8",
            )
        if not self.sources_log_path.exists():
            self.sources_log_path.write_text("", encoding="utf-8")

    def _load_pending(self) -> list[dict[str, Any]]:
        try:
            data = json.loads(self.pending_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
        return []

    def _save_pending(self, rows: list[dict[str, Any]]) -> None:
        self.pending_path.write_text(json.dumps(rows, indent=2, ensure_ascii=True), encoding="utf-8")

    def _load_settings(self) -> dict[str, Any]:
        try:
            data = json.loads(self.settings_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = {}
        if not isinstance(data, dict):
            data = {}
        mode = str(data.get("mode", "auto")).strip().lower()
        if mode not in self.VALID_MODES:
            mode = "auto"
        data["mode"] = mode
        provider = str(data.get("provider", "auto")).strip().lower() or "auto"
        if provider not in self.VALID_PROVIDERS:
            provider = "auto"
        data["provider"] = provider

        searxng_base_url = str(
            os.getenv("FOXFORGE_SEARXNG_URL", str(data.get("searxng_base_url", "http://127.0.0.1:8080")))
        ).strip()
        data["searxng_base_url"] = searxng_base_url.rstrip("/") or "http://127.0.0.1:8080"

        try:
            searxng_timeout_sec = int(data.get("searxng_timeout_sec", 20))
        except (TypeError, ValueError):
            searxng_timeout_sec = 20
        data["searxng_timeout_sec"] = max(3, min(searxng_timeout_sec, 180))

        data["searxng_engines"] = str(data.get("searxng_engines", "")).strip()
        data["searxng_categories"] = str(data.get("searxng_categories", "")).strip()
        data["searxng_language"] = str(data.get("searxng_language", "")).strip()
        try:
            max_results = int(data.get("max_results", 8))
        except (TypeError, ValueError):
            max_results = 8
        data["max_results"] = max(1, min(max_results, 20))
        data["query_expansion_enabled"] = bool(data.get("query_expansion_enabled", True))
        try:
            query_expansion_variants = int(data.get("query_expansion_variants", 4))
        except (TypeError, ValueError):
            query_expansion_variants = 4
        data["query_expansion_variants"] = max(1, min(query_expansion_variants, 8))
        data["source_scoring_enabled"] = bool(data.get("source_scoring_enabled", True))
        try:
            min_quality_sources = int(data.get("min_quality_sources", 2))
        except (TypeError, ValueError):
            min_quality_sources = 2
        data["min_quality_sources"] = max(1, min(min_quality_sources, 8))
        try:
            context_min_source_score = float(data.get("context_min_source_score", 0.52))
        except (TypeError, ValueError):
            context_min_source_score = 0.52
        data["context_min_source_score"] = max(0.1, min(context_min_source_score, 1.0))
        data["conflict_detection_enabled"] = bool(data.get("conflict_detection_enabled", True))
        data["crawl_relevance_gating_enabled"] = bool(data.get("crawl_relevance_gating_enabled", False))
        try:
            crawl_relevance_min_score = float(data.get("crawl_relevance_min_score", 0.1))
        except (TypeError, ValueError):
            crawl_relevance_min_score = 0.1
        data["crawl_relevance_min_score"] = max(0.0, min(crawl_relevance_min_score, 1.0))
        data["fact_check_enabled"] = bool(data.get("fact_check_enabled", False))
        fact_check_provider = str(data.get("fact_check_provider", "local")).strip().lower() or "local"
        if fact_check_provider not in {"local", "gemini"}:
            fact_check_provider = "local"
        data["fact_check_provider"] = fact_check_provider

        data["crawl_enabled"] = bool(data.get("crawl_enabled", True))

        try:
            crawl_depth = int(data.get("crawl_depth", 2))
        except (TypeError, ValueError):
            crawl_depth = 2
        data["crawl_depth"] = max(0, min(crawl_depth, 4))

        try:
            crawl_max_pages = int(data.get("crawl_max_pages", 18))
        except (TypeError, ValueError):
            crawl_max_pages = 18
        data["crawl_max_pages"] = max(1, min(crawl_max_pages, 80))

        try:
            crawl_links_per_page = int(data.get("crawl_links_per_page", 8))
        except (TypeError, ValueError):
            crawl_links_per_page = 8
        data["crawl_links_per_page"] = max(1, min(crawl_links_per_page, 30))

        try:
            crawl_timeout_sec = int(data.get("crawl_timeout_sec", 0))
        except (TypeError, ValueError):
            crawl_timeout_sec = 0
        data["crawl_timeout_sec"] = max(0, min(crawl_timeout_sec, 180))

        data["crawl4ai_enabled"] = bool(data.get("crawl4ai_enabled", True))
        data["crawl4ai_base_url"] = (
            str(os.getenv("FOXFORGE_CRAWL4AI_URL", str(data.get("crawl4ai_base_url", "http://127.0.0.1:11235"))))
            .strip()
            .rstrip("/")
        )

        try:
            crawl4ai_timeout_sec = int(data.get("crawl4ai_timeout_sec", 40))
        except (TypeError, ValueError):
            crawl4ai_timeout_sec = 40
        data["crawl4ai_timeout_sec"] = max(3, min(crawl4ai_timeout_sec, 300))

        try:
            crawl4ai_retry_attempts = int(data.get("crawl4ai_retry_attempts", 2))
        except (TypeError, ValueError):
            crawl4ai_retry_attempts = 2
        data["crawl4ai_retry_attempts"] = max(1, min(crawl4ai_retry_attempts, 8))
        data["crawl4ai_css_selector"] = str(data.get("crawl4ai_css_selector", "article,main,p")).strip()

        data["newspaper_enabled"] = bool(data.get("newspaper_enabled", True))
        data["newspaper_language"] = str(data.get("newspaper_language", "")).strip()

        try:
            search_retry_attempts = int(data.get("search_retry_attempts", 3))
        except (TypeError, ValueError):
            search_retry_attempts = 3
        data["search_retry_attempts"] = max(1, min(search_retry_attempts, 8))

        try:
            crawl_retry_attempts = int(data.get("crawl_retry_attempts", 3))
        except (TypeError, ValueError):
            crawl_retry_attempts = 3
        data["crawl_retry_attempts"] = max(1, min(crawl_retry_attempts, 8))

        data["crawl_same_domain_only"] = bool(data.get("crawl_same_domain_only", True))

        try:
            crawl_text_chars = int(data.get("crawl_text_chars", 2500))
        except (TypeError, ValueError):
            crawl_text_chars = 2500
        data["crawl_text_chars"] = max(250, min(crawl_text_chars, 6000))

        # TOR proxy settings (disabled by default — enable when TOR daemon is running)
        data["tor_proxy_enabled"] = bool(data.get("tor_proxy_enabled", False))
        data["tor_proxy_url"] = str(data.get("tor_proxy_url", "socks5h://127.0.0.1:9050")).strip()
        try:
            tor_timeout_multiplier = float(data.get("tor_timeout_multiplier", 2.5))
        except (TypeError, ValueError):
            tor_timeout_multiplier = 2.5
        data["tor_timeout_multiplier"] = max(1.0, min(tor_timeout_multiplier, 10.0))

        return data

    def _save_settings(self, settings: dict[str, Any]) -> None:
        self.settings_path.write_text(json.dumps(settings, indent=2, ensure_ascii=True), encoding="utf-8")

    def _urlopen(
        self,
        req: urllib.request.Request,
        timeout: int,
        *,
        use_tor: bool | None = None,
        settings: dict[str, Any] | None = None,
    ) -> Any:
        """Wrapper around urllib.request.urlopen with optional TOR proxy routing.

        When TOR is active (via use_tor=True or self._tor_active) and
        tor_proxy_enabled is set in settings, routes the request through a
        SOCKS5 proxy. The socks5h:// scheme resolves DNS through the proxy to
        prevent leaks — critical for .onion domains.

        Requires PySocks (pip install PySocks) for SOCKS5 support.
        """
        _tor = use_tor if use_tor is not None else self._tor_active
        if _tor:
            _s = settings or self._load_settings()
            if _s.get("tor_proxy_enabled", False):
                proxy_url = str(_s.get("tor_proxy_url", "socks5h://127.0.0.1:9050")).strip()
                multiplier = float(_s.get("tor_timeout_multiplier", 2.5))
                timeout = int(timeout * multiplier)
                proxy_handler = urllib.request.ProxyHandler({
                    "http": proxy_url,
                    "https": proxy_url,
                })
                opener = urllib.request.build_opener(proxy_handler)
                return opener.open(req, timeout=max(timeout, 1))
        if timeout <= 0:
            return urllib.request.urlopen(req)
        return urllib.request.urlopen(req, timeout=timeout)

    def get_mode(self) -> str:
        with self.lock:
            settings = self._load_settings()
            return str(settings.get("mode", "auto"))

    def set_mode(self, mode: str) -> str:
        key = mode.strip().lower()
        if key not in self.VALID_MODES:
            raise ValueError("Invalid web mode. Use: off, ask, auto.")
        with self.lock:
            settings = self._load_settings()
            settings["mode"] = key
            self._save_settings(settings)
        return key

    def mode_text(self) -> str:
        settings = self._load_settings()
        return (
            "Web research mode:\n"
            f"- mode: {settings.get('mode', 'auto')}\n"
            f"- provider: {settings.get('provider', 'auto')}\n"
            f"- max_results: {settings.get('max_results', 8)}\n"
            f"- query_expansion_enabled: {settings.get('query_expansion_enabled', True)}\n"
            f"- query_expansion_variants: {settings.get('query_expansion_variants', 4)}\n"
            f"- source_scoring_enabled: {settings.get('source_scoring_enabled', False)}\n"
            f"- min_quality_sources: {settings.get('min_quality_sources', 2)}\n"
            f"- context_min_source_score: {settings.get('context_min_source_score', 0.52)}\n"
            f"- conflict_detection_enabled: {settings.get('conflict_detection_enabled', False)}\n"
            f"- crawl_relevance_gating_enabled: {settings.get('crawl_relevance_gating_enabled', False)}\n"
            f"- crawl_relevance_min_score: {settings.get('crawl_relevance_min_score', 0.1)}\n"
            f"- fact_check_enabled: {settings.get('fact_check_enabled', False)}\n"
            f"- fact_check_provider: {settings.get('fact_check_provider', 'local')}\n"
            f"- searxng_base_url: {settings.get('searxng_base_url', 'http://127.0.0.1:8080')}\n"
            f"- searxng_timeout_sec: {settings.get('searxng_timeout_sec', 20)}\n"
            f"- searxng_engines: {settings.get('searxng_engines', '') or '(auto)'}\n"
            f"- searxng_categories: {settings.get('searxng_categories', '') or '(auto)'}\n"
            f"- searxng_language: {settings.get('searxng_language', '') or '(auto)'}\n"
            f"- crawl_enabled: {settings.get('crawl_enabled', True)}\n"
            f"- crawl_depth: {settings.get('crawl_depth', 2)}\n"
            f"- crawl_max_pages: {settings.get('crawl_max_pages', 18)}\n"
            f"- crawl_links_per_page: {settings.get('crawl_links_per_page', 8)}\n"
            f"- crawl_timeout_sec: {settings.get('crawl_timeout_sec', 12)}\n"
            f"- crawl4ai_enabled: {settings.get('crawl4ai_enabled', True)}\n"
            f"- crawl4ai_base_url: {settings.get('crawl4ai_base_url', 'http://127.0.0.1:11235')}\n"
            f"- crawl4ai_timeout_sec: {settings.get('crawl4ai_timeout_sec', 40)}\n"
            f"- crawl4ai_retry_attempts: {settings.get('crawl4ai_retry_attempts', 2)}\n"
            f"- crawl4ai_css_selector: {settings.get('crawl4ai_css_selector', '') or '(default)'}\n"
            f"- newspaper_enabled: {settings.get('newspaper_enabled', True)}\n"
            f"- newspaper_language: {settings.get('newspaper_language', '') or '(auto)'}\n"
            f"- search_retry_attempts: {settings.get('search_retry_attempts', 3)}\n"
            f"- crawl_retry_attempts: {settings.get('crawl_retry_attempts', 3)}\n"
            f"- crawl_same_domain_only: {settings.get('crawl_same_domain_only', True)}\n"
            f"- crawl_text_chars: {settings.get('crawl_text_chars', 800)}"
        )

    def get_provider(self) -> str:
        with self.lock:
            settings = self._load_settings()
            return str(settings.get("provider", "auto"))

    def provider_text(self) -> str:
        settings = self._load_settings()
        return (
            "Web research provider:\n"
            f"- provider: {settings.get('provider', 'auto')}\n"
            f"- mode: {settings.get('mode', 'auto')}\n"
            f"- searxng_base_url: {settings.get('searxng_base_url', 'http://127.0.0.1:8080')}"
        )

    def set_provider(self, provider: str) -> str:
        key = provider.strip().lower()
        if key not in self.VALID_PROVIDERS:
            raise ValueError("Invalid web provider. Use: auto, searxng, duckduckgo_html, duckduckgo_api.")
        with self.lock:
            settings = self._load_settings()
            settings["provider"] = key
            self._save_settings(settings)
        return key

    def create_pending(self, *, project: str, lane: str, query: str, reason: str, topic_type: str = "general") -> dict[str, Any]:
        query_text = query.strip()
        if not query_text:
            raise ValueError("Web pending query cannot be empty.")
        normalized_topic_type = str(topic_type or "").strip().lower() or "general"
        row = {
            "id": f"web_{uuid.uuid4().hex[:8]}",
            "type": "web_research",
            "status": "open",
            "project": project.strip() or "general",
            "lane": lane.strip() or "project",
            "topic_type": normalized_topic_type,
            "query": query_text,
            "reason": reason.strip() or "Web freshness/citation check requested.",
            "question": "Allow live web research for this request?",
            "summary": (
                f"Query: {query_text[:220]}"
                + ("" if len(query_text) <= 220 else "...")
                + f" | Reason: {(reason.strip() or 'Web freshness/citation check requested.')[:160]}"
            ),
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
        }
        with self.lock:
            rows = self._load_pending()
            rows.append(row)
            self._save_pending(rows)
        return row

    def list_pending(self, limit: int = 50) -> list[dict[str, Any]]:
        limit = max(1, min(limit, 500))
        with self.lock:
            rows = [x for x in self._load_pending() if str(x.get("status", "")).lower() == "open"]
        rows.sort(key=lambda x: str(x.get("created_at", "")), reverse=True)
        return rows[:limit]

    def get_request(self, request_id: str) -> dict[str, Any] | None:
        key = request_id.strip()
        with self.lock:
            rows = self._load_pending()
            for row in rows:
                if str(row.get("id", "")) == key:
                    return row
        return None

    def ignore(self, request_id: str, reason: str = "") -> dict[str, Any] | None:
        key = request_id.strip()
        with self.lock:
            rows = self._load_pending()
            hit: dict[str, Any] | None = None
            for row in rows:
                if str(row.get("id", "")) != key:
                    continue
                if str(row.get("status", "")).lower() != "open":
                    return None
                row["status"] = "ignored"
                row["ignore_reason"] = reason.strip() or "ignored by user"
                row["updated_at"] = _now_iso()
                row["resolved_at"] = _now_iso()
                hit = row
                break
            if hit is None:
                return None
            self._save_pending(rows)
            return hit

    def mark_routed(self, request_id: str, *, target: str, note: str = "", handoff_id: str = "") -> dict[str, Any] | None:
        key = request_id.strip()
        with self.lock:
            rows = self._load_pending()
            hit: dict[str, Any] | None = None
            for row in rows:
                if str(row.get("id", "")) != key:
                    continue
                if str(row.get("status", "")).lower() != "open":
                    return None
                row["status"] = "routed_external"
                row["routed_target"] = target.strip().lower()
                row["routed_note"] = note.strip()
                row["handoff_id"] = handoff_id.strip()
                row["updated_at"] = _now_iso()
                row["resolved_at"] = _now_iso()
                hit = row
                break
            if hit is None:
                return None
            self._save_pending(rows)
            return hit

    def _strip_tags(self, text: str) -> str:
        return re.sub(r"<[^>]+>", "", text)

    def _unwrap_duckduckgo_url(self, href: str) -> str:
        ref = html.unescape(href.strip())
        if ref.startswith("//"):
            ref = "https:" + ref
        if ref.startswith("/"):
            ref = "https://duckduckgo.com" + ref
        parsed = urllib.parse.urlsplit(ref)
        if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
            params = urllib.parse.parse_qs(parsed.query)
            uddg = params.get("uddg", [])
            if uddg:
                return urllib.parse.unquote(uddg[0])
        return ref

    def _search_duckduckgo_html(self, query: str, max_results: int) -> list[dict[str, str]]:
        encoded = urllib.parse.urlencode({"q": query})
        url = f"https://duckduckgo.com/html/?{encoded}"
        req = urllib.request.Request(
            url=url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Foxforge/1.0",
            },
            method="GET",
        )
        with self._urlopen(req, timeout=25) as resp:
            body = resp.read().decode("utf-8", errors="ignore")

        pattern = re.compile(
            r'<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
            re.IGNORECASE | re.DOTALL,
        )
        out: list[dict[str, str]] = []
        seen: set[str] = set()
        for href, title_html in pattern.findall(body):
            url_value = self._unwrap_duckduckgo_url(href)
            if not url_value.startswith("http://") and not url_value.startswith("https://"):
                continue
            if url_value in seen:
                continue
            seen.add(url_value)
            title = html.unescape(self._strip_tags(title_html)).strip()
            if not title:
                title = url_value
            out.append({"title": title, "url": url_value, "snippet": ""})
            if len(out) >= max_results:
                break
        return out

    def _search_duckduckgo_api(self, query: str, max_results: int) -> list[dict[str, str]]:
        params = urllib.parse.urlencode(
            {
                "q": query,
                "format": "json",
                "no_redirect": "1",
                "no_html": "1",
            }
        )
        url = f"https://api.duckduckgo.com/?{params}"
        req = urllib.request.Request(
            url=url,
            headers={"User-Agent": "Foxforge/1.0"},
            method="GET",
        )
        with self._urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="ignore"))

        out: list[dict[str, str]] = []
        seen: set[str] = set()

        abstract_url = str(data.get("AbstractURL", "")).strip()
        abstract_text = str(data.get("AbstractText", "")).strip()
        heading = str(data.get("Heading", "")).strip() or abstract_url
        if abstract_url and abstract_url not in seen:
            seen.add(abstract_url)
            out.append({"title": heading, "url": abstract_url, "snippet": abstract_text})

        def _walk_topics(items: Any) -> None:
            if not isinstance(items, list):
                return
            for item in items:
                if len(out) >= max_results:
                    return
                if isinstance(item, dict) and "Topics" in item:
                    _walk_topics(item.get("Topics"))
                    continue
                if not isinstance(item, dict):
                    continue
                url_value = str(item.get("FirstURL", "")).strip()
                text = str(item.get("Text", "")).strip()
                if not url_value or url_value in seen:
                    continue
                seen.add(url_value)
                title = text.split(" - ")[0].strip() if text else url_value
                out.append({"title": title, "url": url_value, "snippet": text})

        _walk_topics(data.get("RelatedTopics"))
        return out[:max_results]

    def _search_searxng(self, query: str, max_results: int, settings: dict[str, Any]) -> list[dict[str, str]]:
        base_url = str(settings.get("searxng_base_url", "http://127.0.0.1:8080")).strip().rstrip("/")
        if not base_url:
            return []
        endpoint = f"{base_url}/search"
        timeout_sec = int(settings.get("searxng_timeout_sec", 20))
        params: dict[str, str] = {
            "q": query,
            "format": "json",
        }
        engines = str(settings.get("searxng_engines", "")).strip()
        categories = str(settings.get("searxng_categories", "")).strip()
        language = str(settings.get("searxng_language", "")).strip()
        if engines:
            params["engines"] = engines
        if categories:
            params["categories"] = categories
        if language:
            params["language"] = language
        query_string = urllib.parse.urlencode(params)
        req = urllib.request.Request(
            url=f"{endpoint}?{query_string}",
            headers={"User-Agent": "Foxforge/1.0"},
            method="GET",
        )
        with self._urlopen(req, timeout=timeout_sec) as resp:
            payload = json.loads(resp.read().decode("utf-8", errors="ignore"))
        rows = payload.get("results", [])
        if not isinstance(rows, list):
            return []
        out: list[dict[str, str]] = []
        seen: set[str] = set()
        for row in rows:
            if not isinstance(row, dict):
                continue
            url_value = self._normalize_url(str(row.get("url", "")).strip())
            if not url_value or url_value in seen:
                continue
            seen.add(url_value)
            title = str(row.get("title", "")).strip() or url_value
            snippet = str(row.get("content", "")).strip() or str(row.get("snippet", "")).strip()
            out.append({"title": title, "url": url_value, "snippet": snippet})
            if len(out) >= max_results:
                break
        return out

    def _merge_results(
        self,
        primary: list[dict[str, str]],
        secondary: list[dict[str, str]],
        limit: int,
    ) -> list[dict[str, str]]:
        out: list[dict[str, str]] = []
        seen: set[str] = set()
        for row in list(primary) + list(secondary):
            if len(out) >= limit:
                break
            if not isinstance(row, dict):
                continue
            url_value = self._normalize_url(str(row.get("url", "")).strip())
            if not url_value or url_value in seen:
                continue
            seen.add(url_value)
            payload: dict[str, str] = {
                "title": str(row.get("title", "")).strip() or url_value,
                "url": url_value,
                "snippet": str(row.get("snippet", "")).strip(),
            }
            query_variant = str(row.get("query_variant", "")).strip()
            if query_variant:
                payload["query_variant"] = query_variant
            out.append(payload)
        return out[:limit]

    def _resolve_topic_type(self, query: str, topic_type: str) -> str:
        base = str(topic_type or "").strip().lower() or "general"
        mapped_base = self.TOPIC_FAMILY_ALIASES.get(base, base)
        detected = str(detect_topic_type(query, mapped_base) or mapped_base).strip().lower() or "general"
        if (
            detected in {"combat_sports", "sports_event"}
            and mapped_base not in {"general", "sports", "current_events"}
        ):
            return mapped_base
        return self.TOPIC_FAMILY_ALIASES.get(detected, detected)

    def _expand_queries(self, query: str, settings: dict[str, Any], topic_type: str = "general") -> list[str]:
        base = " ".join(str(query or "").split()).strip()
        if not base:
            return []
        if not bool(settings.get("query_expansion_enabled", True)):
            return [base]

        try:
            max_variants = int(settings.get("query_expansion_variants", 4))
        except (TypeError, ValueError):
            max_variants = 4
        max_variants = max(1, min(max_variants, 8))

        out: list[str] = []
        seen: set[str] = set()

        def _add(value: str) -> None:
            text = " ".join(str(value or "").split()).strip()
            if not text:
                return
            key = text.lower()
            if key in seen:
                return
            seen.add(key)
            out.append(text)

        resolved_topic = self._resolve_topic_type(base, topic_type)
        low = base.lower()
        has_recency_hint = any(
            token in low
            for token in (
                "latest",
                "current",
                "recent",
                "today",
                "this week",
                "this month",
                "what's new",
                "whats new",
                "new in",
                "update",
                "updates",
                "news",
            )
        )
        has_year_hint = bool(re.search(r"\b(19|20)\d{2}\b", base))
        current_year = datetime.now(timezone.utc).year

        _add(base)
        for hint in self.TOPIC_HINTS.get(resolved_topic, ()):
            _add(f"{base} {hint}")
        if not has_recency_hint:
            _add(f"latest {base}")
            _add(f"{base} recent updates")
        if not has_year_hint:
            _add(f"{base} {current_year}")
        _add(f"{base} official announcement")
        _add(f"{base} timeline")
        _add(f"{base} analysis")

        return out[:max_variants]

    def _domain_tier(self, host: str) -> tuple[str, float]:
        value = str(host or "").strip().lower()
        if not value:
            return "tier3", 0.45
        domain = value[4:] if value.startswith("www.") else value
        if self._domain_matches(
            domain,
            (
            self.TRUST_TIER_1
            | self.TRUST_TIER_1_SPORTS
            | self.TRUST_TIER_1_ACADEMIC
            | self.TRUST_TIER_1_LEGAL
            ),
        ):
            return "tier1", 1.0
        if self._domain_matches(
            domain,
            (
            self.TRUST_TIER_2
            | self.TRUST_TIER_2_INDIE
            | self.TRUST_TIER_2_MAINSTREAM_SPORTS
            | self.TRUST_TIER_2_PROSUMER_TECH
            | self.TRUST_TIER_2_GAMING
            | self.TRUST_TIER_2_FILM_TV
            | self.TRUST_TIER_2_MUSIC
            | self.TRUST_TIER_2_HEALTH
            | self.TRUST_TIER_2_FINANCE
            | self.TRUST_TIER_2_BUSINESS
            | self.TRUST_TIER_2_REAL_ESTATE
            | self.TRUST_TIER_2_AUTOMOTIVE
            | self.TRUST_TIER_2_ART
            | self.TRUST_TIER_2_LEGAL
            | self.TRUST_TIER_2_EDUCATION
            | self.TRUST_TIER_2_TRAVEL
            | self.TRUST_TIER_2_ANIMAL_CARE
            | self.TRUST_TIER_2_FOOD
            | self.TRUST_TIER_2_BOOKS
            | self.TRUST_TIER_2_PARENTING
            ),
        ):
            return "tier2", 0.78
        return "tier3", 0.45

    def _domain_matches(self, domain: str, candidates: set[str] | tuple[str, ...]) -> bool:
        root = str(domain or "").strip().lower()
        if not root:
            return False
        for candidate in candidates:
            item = str(candidate or "").strip().lower()
            if not item:
                continue
            if root == item or root.endswith("." + item):
                return True
        return False

    
    def _propaganda_penalty(self, title: str, snippet: str) -> float:
        text = f"{title} {snippet}".lower()
        hits = sum(1 for term in self.PROPAGANDA_TERMS if term in text)
        if hits == 0:
            return 0.0
        return min(0.25, hits * 0.05)

    def _low_signal_penalty(
        self,
        *,
        url: str,
        title: str,
        snippet: str,
        query_terms: set[str],
    ) -> tuple[float, list[str], bool]:
        text = f"{title} {snippet}".lower()
        url_low = str(url or "").lower()
        flags: list[str] = []
        penalty = 0.0

        support_intent_terms = {
            "unsupported",
            "browser",
            "compatibility",
            "javascript",
            "cookies",
            "captcha",
            "blocked",
            "denied",
            "forbidden",
            "access",
            "support",
        }
        support_intent = bool(query_terms & support_intent_terms)

        low_signal_hits = [term for term in self.LOW_SIGNAL_TEXT_TERMS if term in text]
        if low_signal_hits:
            penalty += min(0.52, 0.16 + (0.08 * min(len(low_signal_hits), 4)))
            flags.append("support_or_block_page")

        url_hits = [term for term in self.LOW_SIGNAL_URL_TERMS if term in url_low]
        if url_hits:
            penalty += min(0.22, 0.08 + (0.04 * min(len(url_hits), 3)))
            flags.append("blocked_url_pattern")

        nav_hits = sum(1 for term in self.NAVIGATION_NOISE_TERMS if term in text)
        if nav_hits >= 3:
            penalty += min(0.18, 0.05 + (0.025 * min(nav_hits, 5)))
            flags.append("navigation_heavy")

        if support_intent and ("support_or_block_page" in flags or "blocked_url_pattern" in flags):
            penalty = max(0.0, penalty - 0.24)
            flags.append("support_intent_query")

        quality_blocked = bool(
            ("support_or_block_page" in flags and "support_intent_query" not in flags)
            or ("blocked_url_pattern" in flags and "support_intent_query" not in flags and penalty >= 0.2)
        )
        return round(penalty, 3), flags, quality_blocked

    def _topic_domain_bonus(self, host: str, topic_type: str) -> float:
        topic = str(topic_type or "").strip().lower()
        domain = host[4:] if host.startswith("www.") else host
        if topic == "technical":
            if self._domain_matches(domain, self.TRUST_TIER_1_ACADEMIC) or self._domain_matches(domain, self.TRUST_TIER_2_PROSUMER_TECH):
                return 0.06
            if any(tag in domain for tag in ("docs.", "developer.", "readthedocs", "github.com")):
                return 0.04
        elif topic == "finance":
            if self._domain_matches(domain, self.TRUST_TIER_1) or self._domain_matches(domain, self.TRUST_TIER_2_FINANCE):
                return 0.06
        elif topic == "current_events":
            if self._domain_matches(domain, {"reuters.com", "apnews.com", "bbc.com", "nytimes.com", "wsj.com"}):
                return 0.06
            if self._domain_matches(domain, self.TRUST_TIER_1):
                return 0.04
        elif topic == "law":
            if self._domain_matches(domain, self.TRUST_TIER_1_LEGAL) or self._domain_matches(domain, self.TRUST_TIER_2_LEGAL):
                return 0.06
            if domain.endswith(".gov"):
                return 0.05
        elif topic == "education":
            if self._domain_matches(domain, self.TRUST_TIER_2_EDUCATION) or domain.endswith(".edu"):
                return 0.06
        elif topic == "travel":
            if self._domain_matches(domain, self.TRUST_TIER_2_TRAVEL):
                return 0.06
            if self._domain_matches(domain, {"travel.state.gov", "tsa.gov"}):
                return 0.05
        elif topic == "animal_care":
            if self._domain_matches(domain, self.TRUST_TIER_2_ANIMAL_CARE) or self._domain_matches(domain, self.TRUST_TIER_2_HEALTH):
                return 0.06
        elif topic == "food":
            if self._domain_matches(domain, self.TRUST_TIER_2_FOOD) or self._domain_matches(domain, self.TRUST_TIER_2_HEALTH):
                return 0.05
        elif topic == "books":
            if self._domain_matches(domain, self.TRUST_TIER_2_BOOKS):
                return 0.05
        elif topic == "parenting":
            if self._domain_matches(domain, self.TRUST_TIER_2_PARENTING) or self._domain_matches(domain, self.TRUST_TIER_2_HEALTH):
                return 0.05
        elif topic == "business":
            if self._domain_matches(domain, self.TRUST_TIER_2_BUSINESS) or self._domain_matches(domain, self.TRUST_TIER_2_FINANCE):
                return 0.06
        elif topic == "real_estate":
            if self._domain_matches(domain, self.TRUST_TIER_2_REAL_ESTATE):
                return 0.06
            if self._domain_matches(domain, {"hud.gov", "census.gov"}):
                return 0.05
        elif topic == "gaming":
            if self._domain_matches(domain, self.TRUST_TIER_2_GAMING):
                return 0.06
        elif topic == "automotive":
            if self._domain_matches(domain, self.TRUST_TIER_2_AUTOMOTIVE):
                return 0.06
            if self._domain_matches(domain, {"nhtsa.gov", "fueleconomy.gov"}):
                return 0.05
        elif topic == "tv_shows":
            if self._domain_matches(domain, self.TRUST_TIER_2_FILM_TV):
                return 0.06
        elif topic == "movies":
            if self._domain_matches(domain, self.TRUST_TIER_2_FILM_TV):
                return 0.06
        elif topic == "music":
            if self._domain_matches(domain, self.TRUST_TIER_2_MUSIC):
                return 0.06
        elif topic == "art":
            if self._domain_matches(domain, self.TRUST_TIER_2_ART):
                return 0.06
        return 0.0

    def _topic_signal_bonus(self, title: str, snippet: str, topic_type: str) -> float:
        text = f"{title} {snippet}".lower()
        topic = str(topic_type or "").strip().lower()
        if topic == "technical":
            terms = ("release notes", "changelog", "version", "api", "breaking change", "compatibility")
        elif topic == "finance":
            terms = ("earnings", "guidance", "revenue", "eps", "10-k", "10-q", "sec filing")
        elif topic == "current_events":
            terms = ("breaking", "live", "developing", "statement", "timeline", "update")
        elif topic == "law":
            terms = ("effective date", "statute", "bill", "court", "ruling", "regulation")
        elif topic == "education":
            terms = ("admission", "curriculum", "deadline", "accredited", "tuition", "syllabus")
        elif topic == "travel":
            terms = ("visa", "entry requirement", "travel advisory", "border", "flight", "departure")
        elif topic == "animal_care":
            terms = ("veterinary", "vaccination", "parasite", "pet food", "toxic", "animal welfare")
        elif topic == "food":
            terms = ("nutrition", "ingredient", "allergen", "food safety", "calorie", "recall")
        elif topic == "books":
            terms = ("edition", "isbn", "publisher", "publication date", "hardcover", "paperback")
        elif topic == "parenting":
            terms = ("pediatric", "milestone", "age range", "dosage", "safety", "guideline")
        elif topic == "business":
            terms = ("revenue", "margin", "guidance", "strategy", "market share", "executive")
        elif topic == "real_estate":
            terms = ("median price", "inventory", "mortgage", "cap rate", "vacancy", "housing starts")
        elif topic == "gaming":
            terms = ("patch notes", "season", "meta", "dlc", "release date", "esports")
        elif topic == "automotive":
            terms = ("msrp", "recall", "range", "mpg", "horsepower", "nhtsa")
        elif topic == "tv_shows":
            terms = ("season", "episode", "air date", "renewed", "cancelled", "showrunner")
        elif topic == "movies":
            terms = ("box office", "release date", "cast", "runtime", "director", "trailer")
        elif topic == "music":
            terms = ("album", "single", "tour", "release date", "chart", "label")
        elif topic == "art":
            terms = ("exhibition", "museum", "curator", "artist", "auction", "provenance")
        else:
            return 0.0
        hits = sum(1 for term in terms if term in text)
        if hits <= 0:
            return 0.0
        return min(0.08, hits * 0.02)

    def _score_one_source(self, row: dict[str, Any], query_terms: set[str], query: str = "", topic_type: str = "general") -> dict[str, Any]:
        payload = dict(row)
        url_value = str(payload.get("url", "")).strip()
        host = self._hostname(url_value)
        tier_name, base_score = self._domain_tier(host)
        title = str(payload.get("title", "")).strip()
        snippet = str(payload.get("snippet", "")).strip()
        score = base_score
        payload.setdefault("retrieved_at", _now_iso())

        if url_value.lower().startswith("https://"):
            score += 0.03
        if len(snippet) >= 160:
            score += 0.05
        elif len(snippet) >= 80:
            score += 0.03

        hay = f"{title} {snippet}".lower()
        query_hit_count = 0
        if query_terms:
            query_hit_count = len([term for term in query_terms if term in hay])
            if query_hit_count > 0:
                score += min(0.12, query_hit_count * 0.02)
                if query_hit_count == 1 and len(query_terms) >= 4:
                    score -= 0.03
            else:
                score -= 0.08
        if str(payload.get("query_variant", "")).strip():
            score += 0.02

        score += self._topic_domain_bonus(host, topic_type)
        score += self._topic_signal_bonus(title, snippet, topic_type)
        score -= self._propaganda_penalty(title, snippet)
        low_signal_penalty, quality_flags, quality_blocked = self._low_signal_penalty(
            url=url_value,
            title=title,
            snippet=snippet,
            query_terms=query_terms,
        )
        score -= low_signal_penalty
        payload["source_domain"] = host
        payload["source_tier"] = tier_name
        payload = enrich_source_metadata(payload, query=query, topic_type=topic_type)
        score += float(payload.get("freshness_score", 0.0)) * 0.08
        score += float(payload.get("volatility_fit_score", 0.0)) * 0.05
        if bool(payload.get("stale_for_query", False)):
            score -= 0.08
        score += self._domain_rep.get_adjustment(host)
        score = max(0.05, min(1.0, round(score, 3)))
        payload["source_score"] = score
        payload["query_term_hits"] = int(query_hit_count)
        payload["quality_penalty"] = float(low_signal_penalty)
        payload["quality_flags"] = quality_flags
        payload["quality_blocked"] = bool(quality_blocked)
        return payload

    def _apply_source_scoring(
        self,
        sources: list[dict[str, Any]],
        query: str,
        enabled: bool,
        topic_type: str = "general",
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        resolved_topic = self._resolve_topic_type(query, topic_type)
        if not sources:
            return [], {
                "enabled": bool(enabled),
                "applied": False,
                "strategy": "domain_tier_v1",
                "topic_type": resolved_topic,
                "tier_counts": {"tier1": 0, "tier2": 0, "tier3": 0},
                "top_score": 0.0,
            }

        terms = self._query_terms(query)
        scored = [self._score_one_source(row, terms, query=query, topic_type=resolved_topic) for row in sources]
        if bool(enabled):
            volatility = classify_fact_volatility(query, resolved_topic)
            if volatility == "volatile":
                # For volatile topics (live events, breaking news, prices), freshness
                # must dominate — a fresh tier-2 source beats a stale tier-1 source.
                scored.sort(
                    key=lambda x: (
                        0 if bool(x.get("stale_for_query", False)) else 1,
                        float(x.get("freshness_score", 0.0)),
                        float(x.get("source_score", 0.0)),
                        len(str(x.get("snippet", ""))),
                    ),
                    reverse=True,
                )
            else:
                # Stable and semi-volatile: authority (source_score) primary, freshness tiebreaker.
                scored.sort(
                    key=lambda x: (
                        float(x.get("source_score", 0.0)),
                        float(x.get("freshness_score", 0.0)),
                        -float(x.get("source_age_hours", 0.0) or 0.0),
                        len(str(x.get("snippet", ""))),
                    ),
                    reverse=True,
                )

        tier_counts = {"tier1": 0, "tier2": 0, "tier3": 0}
        for row in scored:
            tier = str(row.get("source_tier", "tier3"))
            if tier not in tier_counts:
                tier = "tier3"
            tier_counts[tier] += 1

        freshness = [float(r.get("freshness_score", 0.0)) for r in scored]
        summary = {
            "enabled": bool(enabled),
            "applied": bool(enabled),
            "strategy": "domain_tier_v2_freshness",
            "topic_type": resolved_topic,
            "tier_counts": tier_counts,
            "top_score": float(scored[0].get("source_score", 0.0)),
            "avg_freshness": round(sum(freshness) / len(freshness), 3) if freshness else 0.0,
            "stale_count": sum(1 for r in scored if bool(r.get("stale_for_query", False))),
        }
        return scored, summary

    def _normalize_money_to_usd(self, raw: str) -> str | None:
        text = str(raw or "").strip().lower()
        if not text:
            return None
        text = text.replace(",", "").replace("$", "").strip()
        factor = 1.0
        if text.endswith("billion"):
            factor = 1_000_000_000.0
            text = text[:-7].strip()
        elif text.endswith("million"):
            factor = 1_000_000.0
            text = text[:-7].strip()
        elif text.endswith("thousand"):
            factor = 1_000.0
            text = text[:-8].strip()
        elif text.endswith("b"):
            factor = 1_000_000_000.0
            text = text[:-1].strip()
        elif text.endswith("m"):
            factor = 1_000_000.0
            text = text[:-1].strip()
        elif text.endswith("k"):
            factor = 1_000.0
            text = text[:-1].strip()
        text = re.sub(r"[^0-9.]+", "", text)
        if not text:
            return None
        try:
            value = float(text) * factor
        except Exception:
            return None
        if value <= 0:
            return None
        return f"${int(round(value)):,}"

    def _normalize_isbn(self, raw: str) -> str | None:
        token = re.sub(r"[^0-9Xx]", "", str(raw or ""))
        if len(token) == 10 and re.fullmatch(r"\d{9}[0-9Xx]", token):
            return token.upper()
        if len(token) == 13 and token.isdigit() and token.startswith(("978", "979")):
            return token
        return None

    def _extract_conflict_values(self, row: dict[str, Any], topic_type: str = "general") -> dict[str, set[str]]:
        title = str(row.get("title", "")).strip()
        snippet = str(row.get("snippet", "")).strip()
        text = f"{title}\n{snippet}"
        low = text.lower()
        topic = str(topic_type or "").strip().lower()

        date_values = {
            v.strip()
            for v in re.findall(
                r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{1,2},?\s+(?:19|20)\d{2}\b",
                low,
                flags=re.IGNORECASE,
            )
        }
        year_values = {
            v.strip()
            for v in re.findall(r"\b(?:19|20)\d{2}\b", low)
        }
        number_values = {
            v.strip()
            for v in re.findall(r"\b\d{1,4}(?:\.\d+)?\b", low)
        }
        number_values = {v for v in number_values if v not in year_values}

        matchup_values: set[str] = set()
        for left, right in re.findall(
            r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+vs\.?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b",
            text,
        ):
            matchup_values.add(f"{left.strip()} vs {right.strip()}".lower())

        isbn_values: set[str] = set()
        msrp_values: set[str] = set()
        range_values: set[str] = set()
        mpg_values: set[str] = set()
        runtime_values: set[str] = set()
        box_office_values: set[str] = set()

        if topic == "books":
            for chunk in re.findall(r"\bISBN(?:-1[03])?:?\s*([0-9Xx\- ]{10,20})\b", text, flags=re.IGNORECASE):
                normalized = self._normalize_isbn(chunk)
                if normalized:
                    isbn_values.add(normalized)
            for chunk in re.findall(r"\b97[89][0-9\- ]{10,20}\b", text):
                normalized = self._normalize_isbn(chunk)
                if normalized:
                    isbn_values.add(normalized)

        if topic == "automotive":
            money_chunks = re.findall(
                r"(?:msrp|starting at|starts at|price(?:d)? at)\s*[:\-]?\s*(\$?\s*\d[\d,]*(?:\.\d+)?(?:\s*[kmb]|(?:\s*(?:million|billion|thousand)))?)",
                low,
                flags=re.IGNORECASE,
            )
            for chunk in money_chunks:
                normalized = self._normalize_money_to_usd(chunk)
                if normalized:
                    msrp_values.add(normalized)
            for miles in re.findall(
                r"\b(?:epa\s+)?(?:range|estimated range|driving range)?\s*(?:of|:|is|up to)?\s*(\d{2,4})\s*(?:mile|miles|mi)\b",
                low,
                flags=re.IGNORECASE,
            ):
                range_values.add(f"{int(miles)} mi")
            for miles in re.findall(r"\b(\d{2,4})\s*-\s*mile\s+range\b", low, flags=re.IGNORECASE):
                range_values.add(f"{int(miles)} mi")
            for city, hwy in re.findall(r"\b(\d{1,3})\s*/\s*(\d{1,3})\s*mpg\b", low, flags=re.IGNORECASE):
                mpg_values.add(f"{int(city)}/{int(hwy)} mpg")
            for mpg in re.findall(r"\b(\d{1,3})\s*mpg\b", low, flags=re.IGNORECASE):
                mpg_values.add(f"{int(mpg)} mpg")

        if topic == "movies":
            for chunk in re.findall(
                r"\b(?:box office|gross|opening weekend|opening)\b[^$]{0,40}(\$?\s*\d[\d,]*(?:\.\d+)?(?:\s*[kmb]|(?:\s*(?:million|billion|thousand)))?)",
                low,
                flags=re.IGNORECASE,
            ):
                normalized = self._normalize_money_to_usd(chunk)
                if normalized:
                    box_office_values.add(normalized)
            for mins in re.findall(
                r"\b(?:runtime|run time|running time)\s*(?:of|:)?\s*(\d{2,3})\s*(?:min|mins|minutes)\b",
                low,
                flags=re.IGNORECASE,
            ):
                runtime_values.add(f"{int(mins)} min")
            if not runtime_values:
                for mins in re.findall(r"\b(\d{2,3})\s*(?:min|mins|minutes)\b", low, flags=re.IGNORECASE):
                    runtime_values.add(f"{int(mins)} min")

        return {
            "date": date_values,
            "year": year_values,
            "number": number_values,
            "matchup": matchup_values,
            "isbn": isbn_values,
            "msrp": msrp_values,
            "range": range_values,
            "mpg": mpg_values,
            "runtime": runtime_values,
            "box_office": box_office_values,
        }

    def _detect_source_conflicts(
        self,
        sources: list[dict[str, Any]],
        query: str,
        enabled: bool,
        topic_type: str = "general",
    ) -> dict[str, Any]:
        resolved_topic = self._resolve_topic_type(query, topic_type)
        summary = {
            "enabled": bool(enabled),
            "applied": bool(enabled),
            "conflict_count": 0,
            "conflicts": [],
            "note": "",
            "topic_type": resolved_topic,
        }
        if not sources or not bool(enabled):
            return summary

        low_query = str(query or "").lower()
        date_intent = any(
            token in low_query
            for token in (
                "when",
                "date",
                "year",
                "timeline",
                "schedule",
                "release date",
                "effective date",
                "deadline",
                "air date",
                "today",
                "tonight",
                "this week",
                "this month",
                "this year",
            )
        )
        numeric_intent = any(
            token in low_query
            for token in (
                "how many",
                "price",
                "cost",
                "score",
                "record",
                "rank",
                "percent",
            )
        )
        matchup_intent = any(token in low_query for token in (" vs ", " versus ", "matchup", "head to head"))

        topic_date_intent_tokens: dict[str, tuple[str, ...]] = {
            "law": ("effective date", "ruling date", "filing date"),
            "education": ("deadline", "enrollment date"),
            "travel": ("departure", "arrival", "travel advisory"),
            "books": ("publication date", "release date"),
            "gaming": ("patch date", "release date", "season start"),
            "tv_shows": ("air date", "episode date", "season release"),
            "movies": ("release date", "premiere date"),
            "music": ("release date", "tour date"),
            "art": ("exhibition date", "auction date"),
        }
        topic_numeric_intent_tokens: dict[str, tuple[str, ...]] = {
            "education": ("tuition", "credit hours", "acceptance rate"),
            "food": ("calories", "grams", "serving size"),
            "books": ("isbn", "page count"),
            "parenting": ("age", "months", "years", "dosage"),
            "business": ("revenue", "margin", "market share", "quarter"),
            "real_estate": ("mortgage rate", "median price", "inventory", "cap rate", "rent"),
            "gaming": ("player count", "meta rank"),
            "automotive": ("msrp", "range", "mpg", "horsepower"),
            "tv_shows": ("runtime", "rating"),
            "movies": ("box office", "runtime", "budget", "rating"),
            "music": ("chart", "streams"),
            "art": ("auction price", "estimate"),
        }
        topic_matchup_tokens: dict[str, tuple[str, ...]] = {
            "combat_sports": ("fight", "bout", "matchup", "main event", "vs"),
            "sports_event": ("game", "matchup", "vs", "head to head"),
        }
        if any(token in low_query for token in topic_date_intent_tokens.get(resolved_topic, ())):
            date_intent = True
        if any(token in low_query for token in topic_numeric_intent_tokens.get(resolved_topic, ())):
            numeric_intent = True
        if any(token in low_query for token in topic_matchup_tokens.get(resolved_topic, ())):
            matchup_intent = True

        if not (date_intent or numeric_intent or matchup_intent):
            return summary

        buckets: dict[str, dict[str, set[int]]] = {
            "date": {},
            "year": {},
            "number": {},
            "matchup": {},
            "isbn": {},
            "msrp": {},
            "range": {},
            "mpg": {},
            "runtime": {},
            "box_office": {},
        }
        for idx, row in enumerate(sources, start=1):
            values = self._extract_conflict_values(row, topic_type=resolved_topic)
            for key, entries in values.items():
                for entry in entries:
                    if not entry:
                        continue
                    buckets[key].setdefault(entry, set()).add(idx)

        conflict_order: list[str] = list(
            {
                "books": ("isbn", "date", "year", "number"),
                "automotive": ("msrp", "range", "mpg", "date", "year", "number"),
                "movies": ("box_office", "runtime", "date", "year", "number"),
            }.get(resolved_topic, ("date", "year", "matchup", "number"))
        )
        if not date_intent:
            conflict_order = [key for key in conflict_order if key not in {"date", "year"}]
        if not matchup_intent:
            conflict_order = [key for key in conflict_order if key != "matchup"]
        if not numeric_intent:
            conflict_order = [key for key in conflict_order if key != "number"]
        if not conflict_order:
            return summary

        conflicts: list[dict[str, Any]] = []
        for key in conflict_order:
            candidates = [
                {"value": value, "sources": sorted(list(indexes))}
                for value, indexes in buckets[key].items()
                if indexes
            ]
            if len(candidates) < 2:
                continue
            candidates.sort(key=lambda x: (len(x["sources"]), x["value"]), reverse=True)
            top_values = candidates[:4]
            source_union = sorted({src for row in top_values for src in row["sources"]})
            if len(source_union) < 2:
                continue
            conflicts.append(
                {
                    "type": key,
                    "values": top_values,
                    "source_coverage": len(source_union),
                }
            )

        summary["conflicts"] = conflicts
        summary["conflict_count"] = len(conflicts)
        summary["topic_type"] = resolved_topic
        if conflicts:
            parts = []
            for row in conflicts[:3]:
                names = [str(x.get("value", "")) for x in row.get("values", [])[:2]]
                key_label = str(row.get("type", "unknown")).replace("_", " ")
                parts.append(f"{key_label}: {' vs '.join(names)}")
            summary["note"] = "Potential source conflicts detected: " + "; ".join(parts)
        return summary

    def search(self, query: str, max_results: int | None = None) -> list[dict[str, str]]:
        settings = self._load_settings()
        k = max_results if max_results is not None else int(settings.get("max_results", 8))
        limit = max(1, min(int(k), 20))
        retry_attempts = max(1, int(settings.get("search_retry_attempts", 3)))
        query_text = query.strip()
        if not query_text:
            return []
        provider = str(settings.get("provider", "auto")).strip().lower()
        searx_rows: list[dict[str, str]] = []
        ddg_html_rows: list[dict[str, str]] = []
        ddg_api_rows: list[dict[str, str]] = []

        if provider in {"auto", "searxng"}:
            now = time.time()
            if now >= self._searxng_backoff_until:
                for _ in range(retry_attempts):
                    try:
                        searx_rows = self._search_searxng(query_text, limit, settings)
                    except Exception:
                        searx_rows = []
                    if searx_rows:
                        break
                if not searx_rows:
                    self._searxng_backoff_until = max(self._searxng_backoff_until, time.time() + 120.0)
            if provider == "searxng":
                return searx_rows[:limit]

        if provider in {"auto", "duckduckgo_html"}:
            for _ in range(retry_attempts):
                try:
                    ddg_html_rows = self._search_duckduckgo_html(query_text, limit)
                except Exception:
                    ddg_html_rows = []
                if ddg_html_rows:
                    break
            if provider == "duckduckgo_html":
                return ddg_html_rows[:limit]

        if provider in {"auto", "duckduckgo_api"}:
            for _ in range(retry_attempts):
                try:
                    ddg_api_rows = self._search_duckduckgo_api(query_text, limit)
                except Exception:
                    ddg_api_rows = []
                if ddg_api_rows:
                    break
            if provider == "duckduckgo_api":
                return ddg_api_rows[:limit]

        merged = self._merge_results(searx_rows, ddg_html_rows, limit)
        merged = self._merge_results(merged, ddg_api_rows, limit)
        return merged[:limit]

    def _hostname(self, url: str) -> str:
        return str(urllib.parse.urlsplit(url).hostname or "").strip().lower()

    def _normalize_url(self, raw_url: str, base_url: str = "") -> str:
        text = str(raw_url or "").strip()
        if not text:
            return ""
        joined = urllib.parse.urljoin(base_url, text) if base_url else text
        parsed = urllib.parse.urlsplit(joined)
        if parsed.scheme not in {"http", "https"}:
            return ""
        if not parsed.netloc:
            return ""
        # Drop fragments, normalize path slashes.
        path = re.sub(r"/{2,}", "/", parsed.path or "/")
        normalized = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc.lower(), path, parsed.query, ""))
        return normalized

    def _can_crawl_url(self, url: str) -> bool:
        low = url.lower()
        blocked_ext = (
            ".pdf",
            ".jpg",
            ".jpeg",
            ".png",
            ".gif",
            ".webp",
            ".svg",
            ".zip",
            ".rar",
            ".7z",
            ".exe",
            ".dmg",
            ".mp3",
            ".mp4",
            ".mov",
            ".avi",
        )
        return not low.endswith(blocked_ext)

    def _extract_urls_from_text(self, text: str, limit: int = 18) -> list[str]:
        rows: list[str] = []
        seen: set[str] = set()
        for match in re.finditer(r"https?://[^\s)\]}>\"']+", str(text or ""), flags=re.IGNORECASE):
            candidate = self._normalize_url(match.group(0))
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            rows.append(candidate)
            if len(rows) >= max(1, limit):
                break
        return rows

    def _fetch_page_basic(self, url: str, timeout_sec: int, text_chars: int, retry_attempts: int) -> dict[str, Any]:
        req = urllib.request.Request(
            url=url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) FoxforgeCrawler/1.0",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            },
            method="GET",
        )
        last_exc = None
        for _ in range(max(1, retry_attempts)):
            try:
                ctx = self._urlopen(req, timeout=int(timeout_sec))
                with ctx as resp:
                    ctype = str(resp.headers.get("Content-Type", "")).lower()
                    if "text/html" not in ctype and "application/xhtml+xml" not in ctype:
                        raise RuntimeError(f"unsupported content type: {ctype or 'unknown'}")
                    raw = resp.read(2_000_000)
                break
            except Exception as exc:
                last_exc = exc
                raw = b""
                continue
        if not raw:
            raise RuntimeError(str(last_exc or "fetch failed"))
        body = raw.decode("utf-8", errors="ignore")
        extractor = _PageExtractor()
        extractor.feed(body)
        title = extractor.title() or url
        snippet = extractor.snippet(max_chars=text_chars)
        return {
            "url": url,
            "title": title,
            "snippet": snippet,
            "links": extractor.links,
        }

    def _fetch_page_crawl4ai(self, url: str, settings: dict[str, Any], text_chars: int) -> dict[str, Any]:
        if not bool(settings.get("crawl4ai_enabled", True)):
            raise RuntimeError("crawl4ai disabled")

        base_url = str(settings.get("crawl4ai_base_url", "http://127.0.0.1:11235")).strip().rstrip("/")
        if not base_url:
            raise RuntimeError("crawl4ai base URL is empty")

        timeout_sec = int(settings.get("crawl4ai_timeout_sec", 40))
        retry_attempts = int(settings.get("crawl4ai_retry_attempts", 2))
        css_selector = str(settings.get("crawl4ai_css_selector", "")).strip()

        payload = {
            "urls": [url],
            "bypass_cache": True,
        }
        if css_selector:
            payload["css_selector"] = css_selector

        req = urllib.request.Request(
            url=f"{base_url}/crawl",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "User-Agent": "Foxforge/1.0",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )

        last_exc: Exception | None = None
        raw_payload: dict[str, Any] | None = None
        for _ in range(max(1, retry_attempts)):
            try:
                with self._urlopen(req, timeout=timeout_sec) as resp:
                    raw_payload = json.loads(resp.read().decode("utf-8", errors="ignore"))
                break
            except Exception as exc:
                last_exc = exc
                raw_payload = None
                continue

        if not isinstance(raw_payload, dict):
            raise RuntimeError(str(last_exc or "crawl4ai response missing"))

        rows: Any = raw_payload.get("results")
        if not isinstance(rows, list):
            rows = raw_payload.get("data")
        if not isinstance(rows, list):
            single = raw_payload.get("result")
            if isinstance(single, dict):
                rows = [single]
        if not isinstance(rows, list) or not rows:
            raise RuntimeError("crawl4ai returned no crawl rows")

        first = rows[0] if isinstance(rows[0], dict) else {}
        title = str(first.get("title", "")).strip()
        markdown = str(first.get("markdown", "")).strip()
        text = str(first.get("text", "")).strip()
        snippet_source = markdown or text
        if not snippet_source:
            # Last fallback for unknown response schemas.
            snippet_source = json.dumps(first, ensure_ascii=True)
        snippet_source = _clean_crawl4ai_markdown(snippet_source)
        snippet = " ".join(snippet_source.split())
        if len(snippet) > text_chars:
            cut = snippet[:text_chars].rsplit(" ", 1)[0].strip()
            snippet = (cut or snippet[:text_chars]).strip() + "..."
        links = self._extract_urls_from_text(markdown or text, limit=28)
        return {
            "url": url,
            "title": title or url,
            "snippet": snippet,
            "links": links,
        }

    def _newspaper_extract(self, url: str, settings: dict[str, Any], text_chars: int) -> dict[str, str] | None:
        if not bool(settings.get("newspaper_enabled", True)):
            return None
        try:
            from newspaper import Article, Config
        except Exception:
            try:
                # Compatibility path for forks/distributions that expose a different top-level module.
                from newspaper4k import Article, Config  # type: ignore
            except Exception:
                return None

        cfg = Config()
        cfg.fetch_images = False
        cfg.memoize_articles = False
        timeout_hint = int(settings.get("crawl_timeout_sec", 20) or 20)
        cfg.request_timeout = max(5, min(timeout_hint if timeout_hint > 0 else 20, 120))
        lang = str(settings.get("newspaper_language", "")).strip()
        if lang:
            cfg.language = lang

        article = Article(url=url, config=cfg)
        try:
            article.download()
            article.parse()
        except Exception:
            return None
        text = " ".join(str(article.text or "").split()).strip()
        if not text:
            return None
        if len(text) > text_chars:
            cut = text[:text_chars].rsplit(" ", 1)[0].strip()
            text = (cut or text[:text_chars]).strip() + "..."
        title = " ".join(str(article.title or "").split()).strip()
        return {"title": title, "snippet": text}

    def _fetch_page(self, url: str, settings: dict[str, Any], text_chars: int) -> dict[str, Any]:
        timeout_sec = int(settings.get("crawl_timeout_sec", 0))
        retry_attempts = int(settings.get("crawl_retry_attempts", 3))
        page: dict[str, Any]
        crawl4ai_ready = bool(settings.get("crawl4ai_enabled", True)) and time.time() >= self._crawl4ai_backoff_until
        if crawl4ai_ready:
            try:
                page = self._fetch_page_crawl4ai(url=url, settings=settings, text_chars=text_chars)
            except Exception:
                self._crawl4ai_backoff_until = max(self._crawl4ai_backoff_until, time.time() + 120.0)
                page = self._fetch_page_basic(
                    url=url,
                    timeout_sec=timeout_sec,
                    text_chars=text_chars,
                    retry_attempts=retry_attempts,
                )
        else:
            page = self._fetch_page_basic(
                url=url,
                timeout_sec=timeout_sec,
                text_chars=text_chars,
                retry_attempts=retry_attempts,
            )

        parsed = self._newspaper_extract(url=url, settings=settings, text_chars=text_chars)
        if isinstance(parsed, dict):
            if parsed.get("title"):
                page["title"] = str(parsed.get("title", "")).strip()
            if parsed.get("snippet"):
                page["snippet"] = str(parsed.get("snippet", "")).strip()
        return page

    def _query_terms(self, query: str) -> set[str]:
        _STOPWORDS = {
            "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "shall",
            "should", "may", "might", "must", "can", "could", "of", "in", "on",
            "at", "to", "for", "with", "by", "from", "up", "about", "into",
            "what", "when", "where", "who", "which", "how", "that", "this",
            "and", "or", "not", "it", "its", "i", "me", "my", "you", "your",
        }
        tokens = set(re.split(r"[^a-z0-9]+", query.lower()))
        tokens.discard("")
        return tokens - _STOPWORDS

    def _link_relevance_score(self, url: str, query_terms: set[str]) -> float:
        """Score a candidate child URL for relevance to query terms (0.0–1.0).
        Returns 0.0 for navigation/structural URLs. Links below
        crawl_relevance_min_score are skipped when gating is enabled."""
        if not url:
            return 0.0
        parsed = urllib.parse.urlsplit(url)
        path = parsed.path.lower()
        query_str = parsed.query.lower()
        path_and_query = path + ("?" + query_str if query_str else "")
        _NAV_PATTERNS = (
            "/login", "/signin", "/signup", "/register", "/logout",
            "/about", "/contact", "/privacy", "/terms", "/cookie",
            "/search", "/tag/", "/tags/", "/category/", "/categories/",
            "/author/", "/feed", "/rss", "/sitemap",
            "?page=", "&page=", "?p=", "&p=",
            "/cdn-cgi/", "/wp-admin", "/wp-login",
        )
        for pat in _NAV_PATTERNS:
            if pat in path_and_query:
                return 0.0
        if not query_terms:
            return 0.5
        path_tokens = set(re.split(r"[^a-z0-9]+", path))
        path_tokens.discard("")
        matches = len(query_terms & path_tokens)
        score = min(1.0, (matches / len(query_terms)) * 0.8)
        if re.search(r"/\d{4}/\d{2}/", path):
            score = min(1.0, score + 0.2)
        elif len(path.split("/")) >= 4 and len(path) > 25:
            score = min(1.0, score + 0.1)
        return round(score, 3)

    def _crawl_sources(self, seeds: list[dict[str, str]], settings: dict[str, Any], query: str = "") -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
        depth_limit = int(settings.get("crawl_depth", 2))
        max_pages = int(settings.get("crawl_max_pages", 18))
        links_per_page = int(settings.get("crawl_links_per_page", 8))
        text_chars = int(settings.get("crawl_text_chars", 800))
        same_domain_only = bool(settings.get("crawl_same_domain_only", True))
        relevance_gating = bool(settings.get("crawl_relevance_gating_enabled", False))
        relevance_min_score = float(settings.get("crawl_relevance_min_score", 0.1))
        query_terms = self._query_terms(query) if relevance_gating else set()

        queue: deque[tuple[str, int, str]] = deque()
        enqueued: set[str] = set()
        visited: set[str] = set()
        pages: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        gated_links: int = 0

        for row in seeds:
            seed_url = self._normalize_url(str(row.get("url", "")).strip())
            if not seed_url:
                continue
            host = self._hostname(seed_url)
            if not host:
                continue
            if seed_url in enqueued:
                continue
            queue.append((seed_url, 0, host))
            enqueued.add(seed_url)

        while queue and len(pages) < max_pages:
            current_url, depth, root_host = queue.popleft()
            if current_url in visited:
                continue
            visited.add(current_url)
            if not self._can_crawl_url(current_url):
                continue
            try:
                page = self._fetch_page(
                    url=current_url,
                    settings=settings,
                    text_chars=text_chars,
                )
                page["depth"] = depth
                page["root_host"] = root_host
                pages.append(page)
            except Exception as exc:
                failures.append({"url": current_url, "depth": depth, "error": str(exc)})
                continue

            if depth >= depth_limit:
                continue

            child_count = 0
            for href in page.get("links", []):
                if child_count >= links_per_page or len(enqueued) >= (max_pages * (links_per_page + 1)):
                    break
                next_url = self._normalize_url(str(href), base_url=current_url)
                if not next_url or next_url in enqueued or next_url in visited:
                    continue
                if not self._can_crawl_url(next_url):
                    continue
                if same_domain_only and self._hostname(next_url) != root_host:
                    continue
                if relevance_gating:
                    rel_score = self._link_relevance_score(next_url, query_terms)
                    if rel_score < relevance_min_score:
                        gated_links += 1
                        continue
                queue.append((next_url, depth + 1, root_host))
                enqueued.add(next_url)
                child_count += 1

        return pages, failures, gated_links

    def _append_source_log(self, payload: dict[str, Any]) -> None:
        line = json.dumps(payload, ensure_ascii=True)
        with self.sources_log_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def run_query(
        self,
        *,
        project: str,
        lane: str,
        query: str,
        reason: str,
        request_id: str = "",
        note: str = "",
        topic_type: str = "general",
    ) -> dict[str, Any]:
        settings = self._load_settings()
        resolved_topic = self._resolve_topic_type(query, topic_type)
        # Activate TOR proxy for underground queries (if enabled in settings)
        self._tor_active = str(resolved_topic).strip().lower() == "underground"
        try:
            return self._run_query_inner(
                project=project, lane=lane, query=query, reason=reason,
                request_id=request_id, note=note, topic_type=topic_type,
                settings=settings, resolved_topic=resolved_topic,
            )
        finally:
            self._tor_active = False

    def _run_query_inner(
        self,
        *,
        project: str,
        lane: str,
        query: str,
        reason: str,
        request_id: str,
        note: str,
        topic_type: str,
        settings: dict[str, Any],
        resolved_topic: str,
    ) -> dict[str, Any]:
        variant_queries = self._expand_queries(query, settings, topic_type=resolved_topic)
        if not variant_queries:
            return {
                "ok": False,
                "project": project,
                "lane": lane,
                "query": query,
                "topic_type": resolved_topic,
                "reason": reason,
                "request_id": request_id,
                "source_count": 0,
                "sources": [],
                "source_path": "",
                "query_expansion_enabled": bool(settings.get("query_expansion_enabled", True)),
                "query_variants_count": 0,
                "query_variants": [],
                "variant_hits": [],
                "source_scoring_enabled": bool(settings.get("source_scoring_enabled", True)),
                "source_scoring_summary": {
                    "enabled": bool(settings.get("source_scoring_enabled", True)),
                    "applied": False,
                    "strategy": "domain_tier_v1",
                    "tier_counts": {"tier1": 0, "tier2": 0, "tier3": 0},
                    "top_score": 0.0,
                },
                "conflict_detection_enabled": bool(settings.get("conflict_detection_enabled", True)),
                "conflict_summary": {"enabled": bool(settings.get("conflict_detection_enabled", True)), "applied": False, "conflict_count": 0, "conflicts": [], "note": ""},
                "crawl_relevance_gating_enabled": bool(settings.get("crawl_relevance_gating_enabled", False)),
                "crawl_gated_links": 0,
                "message": "Query is empty after normalization.",
            }

        max_results = max(1, min(int(settings.get("max_results", 8)), 20))
        seed_limit = max(max_results, min(max_results * max(1, len(variant_queries)), 80))
        seeds: list[dict[str, str]] = []
        variant_hits: list[dict[str, Any]] = []
        for variant in variant_queries:
            rows = self.search(variant, max_results=max_results)
            tagged_rows: list[dict[str, str]] = []
            for row in rows:
                payload = dict(row)
                payload["query_variant"] = variant
                tagged_rows.append(payload)
            seeds = self._merge_results(seeds, tagged_rows, seed_limit)
            variant_hits.append({"query": variant, "seed_hits": len(rows)})

        # --- Quality boost: if T1+T2 seeds are scarce, search wider ---
        # Count T1/T2 seeds using a quick domain-only check (no content scoring yet).
        def _tier12_count(seed_list: list[dict[str, str]]) -> int:
            return sum(
                1 for row in seed_list
                if self._domain_tier(self._hostname(str(row.get("url", ""))))[0] in {"tier1", "tier2"}
            )

        _min_quality = max(1, int(settings.get("min_quality_sources", 2)))
        _boost_max_rounds = 2
        _boost_results = min(20, max_results * 2)
        for _boost_round in range(_boost_max_rounds):
            if _tier12_count(seeds) >= _min_quality:
                break
            _boost_seeds: list[dict[str, str]] = []
            for variant in variant_queries:
                for row in self.search(variant, max_results=_boost_results):
                    payload = dict(row)
                    payload["query_variant"] = variant
                    _boost_seeds.append(payload)
            _before = len(seeds)
            seeds = self._merge_results(seeds, _boost_seeds, seed_limit * 2)
            if len(seeds) == _before:
                break  # nothing new found, no point continuing
            _boost_results = min(20, _boost_results + max_results)  # widen slightly each round

        if not seeds:
            return {
                "ok": False,
                "project": project,
                "lane": lane,
                "query": query,
                "topic_type": resolved_topic,
                "reason": reason,
                "request_id": request_id,
                "source_count": 0,
                "sources": [],
                "source_path": "",
                "query_expansion_enabled": bool(settings.get("query_expansion_enabled", True)),
                "query_variants_count": len(variant_queries),
                "query_variants": variant_queries,
                "variant_hits": variant_hits,
                "source_scoring_enabled": bool(settings.get("source_scoring_enabled", True)),
                "source_scoring_summary": {
                    "enabled": bool(settings.get("source_scoring_enabled", True)),
                    "applied": False,
                    "strategy": "domain_tier_v1",
                    "tier_counts": {"tier1": 0, "tier2": 0, "tier3": 0},
                    "top_score": 0.0,
                },
                "conflict_detection_enabled": bool(settings.get("conflict_detection_enabled", True)),
                "conflict_summary": {"enabled": bool(settings.get("conflict_detection_enabled", True)), "applied": False, "conflict_count": 0, "conflicts": [], "note": ""},
                "crawl_relevance_gating_enabled": bool(settings.get("crawl_relevance_gating_enabled", False)),
                "crawl_gated_links": 0,
                "message": "No web sources found (or network unavailable).",
            }

        provider = str(settings.get("provider", "auto")).strip().lower() or "auto"
        crawl_enabled = bool(settings.get("crawl_enabled", True))
        crawled_pages: list[dict[str, Any]] = []
        crawl_failures: list[dict[str, Any]] = []
        crawl_gated_links: int = 0
        if crawl_enabled:
            crawled_pages, crawl_failures, crawl_gated_links = self._crawl_sources(seeds, settings, query=query)

        if crawled_pages:
            sources_raw: list[dict[str, Any]] = [
                {
                    "title": str(page.get("title", "")).strip(),
                    "url": str(page.get("url", "")).strip(),
                    "snippet": str(page.get("snippet", "")).strip(),
                    "depth": int(page.get("depth", 0)),
                }
                for page in crawled_pages
            ]
        else:
            sources_raw = [dict(row) for row in seeds]

        source_scoring_enabled = bool(settings.get("source_scoring_enabled", True))
        sources, source_scoring_summary = self._apply_source_scoring(
            sources=sources_raw,
            query=query,
            enabled=source_scoring_enabled,
            topic_type=resolved_topic,
        )
        quality_min_score = float(settings.get("context_min_source_score", 0.52))
        quality_min_score = max(0.1, min(quality_min_score, 1.0))
        raw_source_count = len(sources)
        quality_blocked_count = sum(1 for row in sources if bool(row.get("quality_blocked", False)))
        filtered_sources = [
            row
            for row in sources
            if not bool(row.get("quality_blocked", False))
            and float(row.get("source_score", 0.0)) >= quality_min_score
        ]
        if not filtered_sources:
            # If strong domains were captured but scored below threshold, keep a small fallback set.
            filtered_sources = [
                row
                for row in sources
                if not bool(row.get("quality_blocked", False))
                and str(row.get("source_tier", "tier3")) in {"tier1", "tier2"}
            ][:3]
        sources = filtered_sources
        source_scoring_summary["context_min_source_score"] = round(float(quality_min_score), 2)
        source_scoring_summary["quality_blocked_count"] = int(quality_blocked_count)
        source_scoring_summary["quality_filtered_out"] = max(0, raw_source_count - len(sources))
        post_filter_tiers = {"tier1": 0, "tier2": 0, "tier3": 0}
        for row in sources:
            tier = str(row.get("source_tier", "tier3"))
            if tier not in post_filter_tiers:
                tier = "tier3"
            post_filter_tiers[tier] += 1
        source_scoring_summary["post_filter_tier_counts"] = post_filter_tiers

        conflict_detection_enabled = bool(settings.get("conflict_detection_enabled", True))
        if not sources:
            return {
                "ok": False,
                "project": project,
                "lane": lane,
                "query": query,
                "topic_type": resolved_topic,
                "reason": reason,
                "request_id": request_id,
                "source_count": 0,
                "sources": [],
                "source_path": "",
                "provider": provider,
                "seed_count": len(seeds),
                "query_expansion_enabled": bool(settings.get("query_expansion_enabled", True)),
                "query_variants_count": len(variant_queries),
                "query_variants": variant_queries,
                "variant_hits": variant_hits,
                "source_scoring_enabled": source_scoring_enabled,
                "source_scoring_summary": source_scoring_summary,
                "conflict_detection_enabled": conflict_detection_enabled,
                "conflict_summary": {
                    "enabled": conflict_detection_enabled,
                    "applied": False,
                    "conflict_count": 0,
                    "conflicts": [],
                    "note": "",
                    "topic_type": resolved_topic,
                },
                "crawl_relevance_gating_enabled": bool(settings.get("crawl_relevance_gating_enabled", False)),
                "crawl_gated_links": crawl_gated_links,
                "fact_check_enabled": bool(settings.get("fact_check_enabled", False)),
                "fact_check_provider": str(settings.get("fact_check_provider", "local")),
                "crawl_pages": len(crawled_pages),
                "crawl_failures": len(crawl_failures),
                "crawl_enabled": crawl_enabled,
                "message": (
                    "No high-confidence web sources passed relevance and quality filters. "
                    "Try a more specific query or add known trusted domains."
                ),
            }
        conflict_summary = self._detect_source_conflicts(
            sources=sources,
            query=query,
            enabled=conflict_detection_enabled,
            topic_type=resolved_topic,
        )

        # === Real Intelligence Behavior passes ===
        _intel_snippets = [str(s.get("snippet", "")) for s in sources if str(s.get("snippet", "")).strip()]
        _intel_ordered = [(str(s.get("source_domain", "")), str(s.get("snippet", ""))) for s in sources]
        _independence = SourceIndependenceScorer.score(_intel_snippets)
        _mutation = NarrativeMutationTracker.analyze(_intel_ordered)
        _consensus = ConsensusAlarmSystem.evaluate(sources)
        _semantic_contradictions = CrossDomainContradictionDetector.detect(sources)
        intel_summary: dict[str, Any] = {
            "independence": _independence,
            "mutation": _mutation,
            "consensus": _consensus,
            "semantic_contradictions": _semantic_contradictions,
        }

        # Compact alert strings — inserted at the top of the document so they
        # survive context-window trimming and are never "lost in the middle".
        _alert_lines: list[str] = []
        if _independence.get("warning"):
            _alert_lines.append(f"WIRE LAUNDERING SIGNAL: {_independence['warning']}")
        if _mutation.get("mutation_detected"):
            _alert_lines.append(
                f"NARRATIVE MUTATION (confidence={_mutation.get('confidence', 0.0):.2f}): "
                f"{_mutation.get('note', '')}"
            )
        if _consensus.get("alarm"):
            _alert_lines.append(f"CONSENSUS ALARM: {_consensus.get('reason', '')}")
        for _sc in _semantic_contradictions:
            _alert_lines.append(f"CONTRADICTION: {_sc.get('note', '')}")

        lines = [
            "# Web Research Source Cache",
            "",
            f"- request_id: {request_id or 'direct'}",
            f"- project: {project}",
            f"- lane: {lane}",
            f"- query: {query}",
            f"- reason: {reason}",
            f"- note: {note.strip() or 'none'}",
            f"- captured_at: {_now_iso()}",
            f"- seed_count: {len(seeds)}",
            f"- query_expansion_enabled: {bool(settings.get('query_expansion_enabled', True))}",
            f"- query_variants_count: {len(variant_queries)}",
            f"- query_variants: {' | '.join(variant_queries)}",
            f"- provider: {provider}",
            f"- source_scoring_enabled: {source_scoring_enabled}",
            f"- source_scoring_applied: {bool(source_scoring_summary.get('applied', False))}",
            (
                "- source_tier_counts: "
                f"{source_scoring_summary.get('tier_counts', {}).get('tier1', 0)}/"
                f"{source_scoring_summary.get('tier_counts', {}).get('tier2', 0)}/"
                f"{source_scoring_summary.get('tier_counts', {}).get('tier3', 0)}"
            ),
            f"- source_score_top: {float(source_scoring_summary.get('top_score', 0.0)):.2f}",
            f"- context_min_source_score: {float(source_scoring_summary.get('context_min_source_score', settings.get('context_min_source_score', 0.52))):.2f}",
            f"- quality_blocked_count: {int(source_scoring_summary.get('quality_blocked_count', 0))}",
            f"- quality_filtered_out: {int(source_scoring_summary.get('quality_filtered_out', 0))}",
            f"- conflict_detection_enabled: {conflict_detection_enabled}",
            f"- conflict_count: {int(conflict_summary.get('conflict_count', 0))}",
            f"- crawl_relevance_gating_enabled: {bool(settings.get('crawl_relevance_gating_enabled', False))}",
            f"- crawl_relevance_min_score: {float(settings.get('crawl_relevance_min_score', 0.1)):.2f}",
            f"- crawl_gated_links: {crawl_gated_links}",
            f"- fact_check_enabled: {bool(settings.get('fact_check_enabled', False))}",
            f"- fact_check_provider: {settings.get('fact_check_provider', 'local')}",
            f"- crawl_enabled: {crawl_enabled}",
            f"- crawl4ai_enabled: {settings.get('crawl4ai_enabled', True)}",
            f"- newspaper_enabled: {settings.get('newspaper_enabled', True)}",
            f"- crawl_depth: {settings.get('crawl_depth', 2)}",
            f"- crawl_max_pages: {settings.get('crawl_max_pages', 18)}",
            f"- crawl_links_per_page: {settings.get('crawl_links_per_page', 8)}",
            f"- crawl_timeout_sec: {settings.get('crawl_timeout_sec', 12)}",
            f"- crawl_same_domain_only: {settings.get('crawl_same_domain_only', True)}",
            f"- crawl_pages_collected: {len(crawled_pages)}",
            f"- crawl_failures: {len(crawl_failures)}",
        ]
        if _alert_lines:
            lines[2:2] = ["## Active Warnings"] + _alert_lines + [""]
        lines.extend(["", "## Query Variant Hits"])
        for idx, row in enumerate(variant_hits, start=1):
            lines.append(f"{idx}. {row.get('query', '')} | seed_hits={int(row.get('seed_hits', 0))}")

        lines.extend(["", "## Seed Results"])
        for idx, row in enumerate(seeds, start=1):
            title = str(row.get("title", "")).strip() or str(row.get("url", "")).strip()
            url_value = str(row.get("url", "")).strip()
            snippet = str(row.get("snippet", "")).strip()
            source_variant = str(row.get("query_variant", "")).strip()
            if source_variant:
                lines.append(f"{idx}. [{title}]({url_value}) | variant={source_variant}")
            else:
                lines.append(f"{idx}. [{title}]({url_value})")
            if snippet:
                lines.append(f"   - {snippet}")

        lines.extend(
            [
                "",
                "## Traversed Pages" if crawled_pages else "## Traversed Pages (none)",
            ]
        )
        for idx, row in enumerate(crawled_pages, start=1):
            title = str(row.get("title", "")).strip() or str(row.get("url", "")).strip()
            url_value = str(row.get("url", "")).strip()
            snippet = str(row.get("snippet", "")).strip()
            depth = int(row.get("depth", 0))
            lines.append(f"{idx}. d={depth} [{title}]({url_value})")
            if snippet:
                lines.append(f"   - {snippet}")

        if crawl_failures:
            lines.extend(["", "## Crawl Failures (sample)"])
            for idx, row in enumerate(crawl_failures[:20], start=1):
                lines.append(f"{idx}. d={row.get('depth', 0)} {row.get('url', '')}")
                lines.append(f"   - error: {row.get('error', '')}")

        if int(conflict_summary.get("conflict_count", 0)) > 0:
            lines.extend(["", "## Preflight Conflict Flags"])
            for idx, row in enumerate(conflict_summary.get("conflicts", [])[:6], start=1):
                kind = str(row.get("type", "unknown"))
                lines.append(f"{idx}. type={kind} | source_coverage={int(row.get('source_coverage', 0))}")
                values = row.get("values", [])
                if isinstance(values, list):
                    for claim in values[:4]:
                        value = str(claim.get("value", "")).strip()
                        srcs = claim.get("sources", [])
                        lines.append(f"   - {value} | sources={srcs}")

        lines.extend(["", "## Intelligence Analysis"])
        lines.append(f"- source_independence: {float(_independence.get('independence_score', 1.0)):.2f}")
        if _independence.get("warning"):
            lines.append(f"  WARNING: {_independence['warning']}")
        if _mutation.get("mutation_detected"):
            lines.append(
                f"- narrative_mutation: DETECTED (confidence={float(_mutation.get('confidence', 0.0)):.2f})"
            )
            if _mutation.get("note"):
                lines.append(f"  note: {_mutation['note']}")
        else:
            lines.append("- narrative_mutation: none detected")
        if _consensus.get("alarm"):
            lines.append(
                f"- consensus_alarm: TOO-PERFECT"
                f" (uniformity={float(_consensus.get('uniformity_ratio', 0.0)):.2f},"
                f" tier1_anchor={_consensus.get('has_tier1_anchor', False)})"
            )
            lines.append(f"  WARNING: {_consensus.get('reason', '')}")
        else:
            lines.append(
                f"- consensus_alarm: none"
                f" (uniformity={float(_consensus.get('uniformity_ratio', 0.0)):.2f},"
                f" tier1_anchor={_consensus.get('has_tier1_anchor', False)})"
            )
        if _semantic_contradictions:
            lines.append(f"- semantic_contradictions: {len(_semantic_contradictions)}")
            for sc in _semantic_contradictions:
                lines.append(
                    f"  - subject='{sc.get('subject', '')}'"
                    f" | pos={sc.get('positive_sources', [])}"
                    f" vs neg={sc.get('negative_sources', [])}"
                )
        else:
            lines.append("- semantic_contradictions: none detected")

        lines.extend(
            [
                "",
                "## Sources Used By Orchestrator",
            ]
        )
        for idx, row in enumerate(sources, start=1):
            title = str(row.get("title", "")).strip() or str(row.get("url", "")).strip()
            url_value = str(row.get("url", "")).strip()
            snippet = str(row.get("snippet", "")).strip()
            depth = row.get("depth", None)
            score = float(row.get("source_score", 0.0))
            tier = str(row.get("source_tier", "tier3"))
            if isinstance(depth, int):
                lines.append(f"{idx}. d={depth} [{title}]({url_value}) | {tier} score={score:.2f}")
            else:
                lines.append(f"{idx}. [{title}]({url_value}) | {tier} score={score:.2f}")
            if snippet:
                lines.append(f"   - {snippet}")

        filename = self.store.timestamped_name("web_sources")
        source_path = self.store.write_project_file(project, "research_web_sources", filename, "\n".join(lines) + "\n")
        log_payload = {
            "ts": _now_iso(),
            "request_id": request_id,
            "project": project,
            "lane": lane,
            "query": query,
            "topic_type": resolved_topic,
            "reason": reason,
            "note": note.strip(),
            "source_path": str(source_path),
            "provider": provider,
            "seed_count": len(seeds),
            "query_expansion_enabled": bool(settings.get("query_expansion_enabled", True)),
            "query_variants_count": len(variant_queries),
            "query_variants": variant_queries,
            "variant_hits": variant_hits,
            "source_scoring_enabled": source_scoring_enabled,
            "source_scoring_summary": source_scoring_summary,
            "conflict_detection_enabled": conflict_detection_enabled,
            "conflict_summary": conflict_summary,
            "crawl_relevance_gating_enabled": bool(settings.get("crawl_relevance_gating_enabled", False)),
            "crawl_gated_links": crawl_gated_links,
            "fact_check_enabled": bool(settings.get("fact_check_enabled", False)),
            "fact_check_provider": str(settings.get("fact_check_provider", "local")),
            "crawl_enabled": crawl_enabled,
            "crawl_pages": len(crawled_pages),
            "crawl_failures": len(crawl_failures),
            "intel_summary": intel_summary,
            "sources": sources,
        }
        self._append_source_log(log_payload)
        return {
            "ok": True,
            "project": project,
            "lane": lane,
            "query": query,
            "topic_type": resolved_topic,
            "reason": reason,
            "request_id": request_id,
            "source_count": len(sources),
            "provider": provider,
            "seed_count": len(seeds),
            "query_expansion_enabled": bool(settings.get("query_expansion_enabled", True)),
            "query_variants_count": len(variant_queries),
            "query_variants": variant_queries,
            "variant_hits": variant_hits,
            "source_scoring_enabled": source_scoring_enabled,
            "source_scoring_summary": source_scoring_summary,
            "conflict_detection_enabled": conflict_detection_enabled,
            "conflict_summary": conflict_summary,
            "crawl_relevance_gating_enabled": bool(settings.get("crawl_relevance_gating_enabled", False)),
            "crawl_gated_links": crawl_gated_links,
            "fact_check_enabled": bool(settings.get("fact_check_enabled", False)),
            "fact_check_provider": str(settings.get("fact_check_provider", "local")),
            "crawl_pages": len(crawled_pages),
            "crawl_failures": len(crawl_failures),
            "crawl_enabled": crawl_enabled,
            "intel_summary": intel_summary,
            "sources": sources,
            "source_path": str(source_path),
            "message": (
                f"Provider '{provider}' captured {len(seeds)} seed source(s), traversed {len(crawled_pages)} page(s), "
                f"usable source context entries: {len(sources)}."
            ),
        }

    def approve_and_run(self, request_id: str, note: str = "") -> dict[str, Any] | None:
        key = request_id.strip()
        with self.lock:
            rows = self._load_pending()
            target: dict[str, Any] | None = None
            for row in rows:
                if str(row.get("id", "")) != key:
                    continue
                if str(row.get("status", "")).lower() != "open":
                    return None
                target = row
                break
            if target is None:
                return None

        result = self.run_query(
            project=str(target.get("project", "general")),
            lane=str(target.get("lane", "project")),
            query=str(target.get("query", "")),
            reason=str(target.get("reason", "")),
            request_id=key,
            note=note,
            topic_type=str(target.get("topic_type", "general")),
        )

        with self.lock:
            rows = self._load_pending()
            hit: dict[str, Any] | None = None
            for row in rows:
                if str(row.get("id", "")) != key:
                    continue
                if str(row.get("status", "")).lower() != "open":
                    return None
                row["status"] = "resolved"
                row["answer_note"] = note.strip()
                row["updated_at"] = _now_iso()
                row["resolved_at"] = _now_iso()
                row["source_count"] = int(result.get("source_count", 0))
                row["source_path"] = str(result.get("source_path", ""))
                row["run_ok"] = bool(result.get("ok", False))
                hit = row
                break
            if hit is None:
                return None
            self._save_pending(rows)

        result["pending"] = hit
        return result

    def recent_sources_for_project(self, project: str, limit: int = 8) -> list[dict[str, Any]]:
        key = project.strip()
        limit = max(1, min(limit, 100))
        rows: list[dict[str, Any]] = []
        if not self.sources_log_path.exists():
            return rows
        for line in self.sources_log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(data.get("project", "")) != key:
                continue
            if not isinstance(data, dict):
                continue
            rows.append(data)
        rows.sort(key=lambda x: str(x.get("ts", "")), reverse=True)
        return rows[:limit]

    def web_context_for_project(self, project: str, limit: int = 6) -> str:
        logs = self.recent_sources_for_project(project, limit=limit)
        if not logs:
            return ""
        settings = self._load_settings()
        min_score = max(0.1, min(float(settings.get("context_min_source_score", 0.52)), 1.0))
        lines = ["Recent web source cache (use only if relevant):"]
        count = 0
        for log in logs:
            for source in log.get("sources", []) if isinstance(log.get("sources"), list) else []:
                if count >= limit:
                    break
                title = str(source.get("title", "")).strip()
                url_value = str(source.get("url", "")).strip()
                if not url_value:
                    continue
                snippet = str(source.get("snippet", "")).strip()
                _, _, inferred_blocked = self._low_signal_penalty(
                    url=url_value,
                    title=title,
                    snippet=snippet,
                    query_terms=set(),
                )
                quality_blocked = bool(source.get("quality_blocked", False)) or bool(inferred_blocked)
                if quality_blocked:
                    continue
                score = float(source.get("source_score", 0.0))
                if score < min_score:
                    continue
                depth = source.get("depth", None)
                tier = str(source.get("source_tier", "tier3")).strip() or "tier3"
                freshness = float(source.get("freshness_score", 0.0))
                if isinstance(depth, int):
                    lines.append(
                        f"- d={depth} [{tier} {score:.2f} fresh={freshness:.2f}] {title or url_value} | {url_value}"
                    )
                else:
                    lines.append(f"- [{tier} {score:.2f} fresh={freshness:.2f}] {title or url_value} | {url_value}")
                if snippet:
                    lines.append(f"  snippet: {snippet}")
                count += 1
            if count >= limit:
                break
        if count == 0:
            return ""
        return "\n".join(lines)

    def sources_text(self, project: str, limit: int = 10) -> str:
        logs = self.recent_sources_for_project(project, limit=limit)
        if not logs:
            return f"No web source cache yet for project '{project}'."
        lines = [f"Recent web source cache for '{project}' ({len(logs)} runs):"]
        for row in logs:
            ts = str(row.get("ts", ""))
            query = str(row.get("query", ""))
            source_path = str(row.get("source_path", ""))
            sources = row.get("sources", []) if isinstance(row.get("sources"), list) else []
            seed_count = int(row.get("seed_count", 0))
            crawl_pages = int(row.get("crawl_pages", 0))
            crawl_failures = int(row.get("crawl_failures", 0))
            conflict_summary = row.get("conflict_summary", {}) if isinstance(row.get("conflict_summary", {}), dict) else {}
            conflict_count = int(conflict_summary.get("conflict_count", 0))
            lines.append(
                f"- [{ts}] query={query} | used={len(sources)} | seeds={seed_count} | "
                f"crawl_pages={crawl_pages} | crawl_failures={crawl_failures} | "
                f"conflicts={conflict_count} | file={source_path}"
            )
        return "\n".join(lines)




# ============================================================
# REAL INTELLIGENCE BEHAVIORS
# Five active analysis passes run on every query result set.
# ============================================================



class NarrativeMutationTracker:
    """
    Detects how claims mutate as they propagate across sources.

    Early sources tend to hedge ("may", "reportedly", "alleged").
    Later sources that copy-paste or paraphrase often drop hedges and
    inflate certainty ("confirmed", "revealed", "is definitively").

    This is the core fingerprint of citation laundering and PR cascade:
    a claim that starts uncertain and becomes "fact" with no new evidence.
    """

    HEDGE_TERMS = {
        "may", "might", "could", "reportedly", "allegedly", "sources say",
        "claims", "appears to", "seems", "possible", "suggests", "unconfirmed",
        "according to some", "rumored", "speculated", "believed to",
    }
    CERTAINTY_TERMS = {
        "confirmed", "proved", "proven", "revealed", "officially",
        "definitively", "announced", "stated", "declared", "established",
        "verified", "fact", "undeniably",
    }

    @classmethod
    def analyze(cls, ordered_snippets: list[tuple[str, str]]) -> dict[str, Any]:
        """
        ordered_snippets: [(domain, text), ...] in discovery order (earliest first).
        Returns mutation report.
        """
        if len(ordered_snippets) < 2:
            return {"mutation_detected": False, "confidence": 0.0, "note": ""}

        hedge_counts: list[int] = []
        certainty_counts: list[int] = []
        for _domain, text in ordered_snippets:
            tl = text.lower()
            hedge_counts.append(sum(1 for t in cls.HEDGE_TERMS if t in tl))
            certainty_counts.append(sum(1 for t in cls.CERTAINTY_TERMS if t in tl))

        mid = max(1, len(ordered_snippets) // 2)
        early_hedges = sum(hedge_counts[:mid])
        late_hedges = sum(hedge_counts[mid:])
        early_cert = sum(certainty_counts[:mid])
        late_cert = sum(certainty_counts[mid:])

        mutation = (early_hedges > late_hedges) and (late_cert > early_cert)
        if mutation:
            confidence = round(min(1.0, (early_hedges - late_hedges) * 0.15 + (late_cert - early_cert) * 0.1), 3)
            note = (
                f"Hedging language dropped from {early_hedges} to {late_hedges} hits; "
                f"certainty language rose from {early_cert} to {late_cert} hits. "
                "Claim may have been laundered into fact without new primary evidence."
            )
        else:
            confidence = 0.0
            note = ""

        return {
            "mutation_detected": mutation,
            "confidence": confidence,
            "early_hedge_count": early_hedges,
            "late_certainty_count": late_cert,
            "note": note,
        }


class SourceIndependenceScorer:
    """
    Detects wire-service laundering and citation echo chambers.

    When many sources share nearly identical vocabulary, they are almost
    certainly all re-publishing the same wire report or press release.
    High source count with low independence = one claim amplified, not confirmed.
    """

    @staticmethod
    def _token_set(text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]{4,}", text.lower()))

    @classmethod
    def jaccard(cls, a: str, b: str) -> float:
        sa, sb = cls._token_set(a), cls._token_set(b)
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)

    @classmethod
    def score(cls, snippets: list[str]) -> dict[str, Any]:
        """
        Returns independence_score 0.0–1.0 (1.0 = every source is distinct).
        Also flags clone pairs (Jaccard ≥ 0.5).
        """
        if len(snippets) < 2:
            return {"independence_score": 1.0, "clone_pairs": 0, "total_pairs": 0, "warning": ""}

        clone_pairs = 0
        total_pairs = 0
        for i in range(len(snippets)):
            for j in range(i + 1, len(snippets)):
                total_pairs += 1
                if cls.jaccard(snippets[i], snippets[j]) >= 0.5:
                    clone_pairs += 1

        independence = round(1.0 - (clone_pairs / max(1, total_pairs)), 3)
        if independence < 0.5:
            warning = (
                f"Wire laundering likely: {clone_pairs}/{total_pairs} source pairs share >50% vocabulary. "
                "Multiple outlets may be republishing a single press release or wire report."
            )
        elif independence < 0.7:
            warning = (
                f"Echo chamber signal: {clone_pairs}/{total_pairs} source pairs are near-duplicate. "
                "Treat these as one confirmed source, not many."
            )
        else:
            warning = ""

        return {
            "independence_score": independence,
            "clone_pairs": clone_pairs,
            "total_pairs": total_pairs,
            "warning": warning,
        }


class ConsensusAlarmSystem:
    """
    'Too-perfect consensus' detector.

    When all sources agree in near-identical language AND no tier-1 anchor
    is present, the result set is suspicious. Common patterns:
      - Coordinated PR campaigns
      - Astroturf / influencer blast
      - Wire service regurgitation with no original reporting
      - SEO content farms all copying the same source

    A real story backed by real evidence typically produces varied reporting:
    different angles, different wording, some disagreement.
    Uniformity is a red flag, not a quality signal.
    """

    @classmethod
    def evaluate(cls, sources: list[dict[str, Any]]) -> dict[str, Any]:
        if len(sources) < 3:
            return {"alarm": False, "uniformity_ratio": 0.0, "has_tier1_anchor": False, "reason": ""}

        snippets = [str(s.get("snippet", "")) for s in sources if str(s.get("snippet", "")).strip()]
        tiers = [str(s.get("source_tier", "tier3")) for s in sources]

        if len(snippets) < 3:
            return {"alarm": False, "uniformity_ratio": 0.0, "has_tier1_anchor": any(t == "tier1" for t in tiers), "reason": ""}

        has_tier1 = any(t == "tier1" for t in tiers)
        total_pairs = 0
        high_sim_pairs = 0
        for i in range(len(snippets)):
            for j in range(i + 1, len(snippets)):
                total_pairs += 1
                if SourceIndependenceScorer.jaccard(snippets[i], snippets[j]) >= 0.4:
                    high_sim_pairs += 1

        uniformity_ratio = round(high_sim_pairs / max(1, total_pairs), 3)
        alarm = uniformity_ratio >= 0.6 and not has_tier1
        reason = ""
        if alarm:
            reason = (
                f"Too-perfect consensus: {high_sim_pairs}/{total_pairs} source pairs share similar vocabulary, "
                f"no tier-1 anchor present. Possible PR campaign, coordinated messaging, or wire regurgitation. "
                f"Treat with skepticism — look for an original primary source."
            )

        return {
            "alarm": alarm,
            "uniformity_ratio": uniformity_ratio,
            "has_tier1_anchor": has_tier1,
            "reason": reason,
        }


class CrossDomainContradictionDetector:
    """
    Extends numeric conflict detection to directional / semantic contradictions.

    Looks for cases where sources in the same result set make opposing claims
    about the same subject — one says something increases, another says it falls;
    one says X succeeded, another says X failed.

    This catches: stock contradictions, election result disputes, scientific
    finding reversals, sports outcome disagreements, and policy claim fights.
    """

    POSITIVE_SIGNALS = {
        "increases", "rises", "grew", "grows", "improves", "confirms", "proved",
        "succeeds", "wins", "gained", "gains", "advances", "leads", "outperforms",
        "surges", "rallies", "recovers", "beats", "tops",
    }
    NEGATIVE_SIGNALS = {
        "decreases", "falls", "fell", "shrinks", "worsens", "denies", "disproves",
        "fails", "loses", "dropped", "drops", "retreats", "trails", "underperforms",
        "plunges", "collapses", "misses", "loses", "declines",
    }

    @classmethod
    def detect(cls, sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
        subject_signals: dict[str, list[tuple[str, str, str]]] = {}

        for src in sources:
            domain = str(src.get("source_domain", src.get("domain", "")))
            tier = str(src.get("source_tier", "tier3"))
            text = f"{src.get('title', '')} {src.get('snippet', '')}".lower()
            words = text.split()
            for i, word in enumerate(words):
                clean = word.strip(".,;:!?\"'()")
                if clean not in cls.POSITIVE_SIGNALS and clean not in cls.NEGATIVE_SIGNALS:
                    continue
                subject = " ".join(words[max(0, i - 2):i]).strip(".,;:!?\"'()")
                if not subject or len(subject) < 4:
                    continue
                direction = "positive" if clean in cls.POSITIVE_SIGNALS else "negative"
                subject_signals.setdefault(subject, []).append((domain, tier, direction))

        contradictions: list[dict[str, Any]] = []
        for subject, signals in subject_signals.items():
            pos = [s[0] for s in signals if s[2] == "positive"]
            neg = [s[0] for s in signals if s[2] == "negative"]
            if pos and neg and set(pos) != set(neg):
                contradictions.append({
                    "subject": subject,
                    "positive_sources": pos[:3],
                    "negative_sources": neg[:3],
                    "note": (
                        f"Directional contradiction on '{subject}': "
                        f"{len(pos)} source(s) positive, {len(neg)} negative."
                    ),
                })

        return contradictions[:5]
