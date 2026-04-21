"""Recency sensitivity and text signal detection utilities."""

from __future__ import annotations

import re

RECENCY_TERMS: frozenset[str] = frozenset([
    "latest", "recent", "recently", "today", "right now",
    "as of", "this week", "this month", "this year",
    "what's new", "whats new", "new in", "what happened",
    "just released", "just announced", "nowadays",
    "these days", "currently", "real-time", "realtime",
    "trending",
    # sports / event recency signals — compound phrases only to avoid collateral matches
    "next fight", "fight card", "fight night", "this weekend",
    "main event", "next event", "upcoming event", "upcoming fight",
    "ppv", "next ppv", "on the card", "fight week",
    "next match", "upcoming match", "next game", "next bout",
    # Removed: "now" (too common), "live" (too common), "current" (too common),
    #   "update"/"updates" (too ambiguous), "trend" (keep only "trending"),
    #   "breaking" (too broad — use "breaking news" pattern below),
    #   "news" (moved to compound patterns), "card" (too broad — "fight card" kept above)
])

_WEB_OFFER_MARKER_PATTERNS: tuple[str, ...] = (
    # Unambiguous temporal / recency markers
    r"\blatest\b",
    r"\brecent(?:ly)?\b",
    r"\btoday\b",
    r"\bthis week\b",
    r"\bthis month\b",
    r"\bthis year\b",
    r"\bright now\b",
    r"\bas of\b",
    r"\bnowadays\b",
    r"\bthese days\b",
    r"\bcurrently\b",
    r"\bjust released\b",
    r"\bjust announced\b",
    # News — require compound to avoid "that was big news in the 80s"
    r"\bbreaking news\b",
    r"\blatest news\b",
    r"\brecent news\b",
    r"\bin the news\b",
    r"\bnews today\b",
    r"\bwhat'?s new\b",
    r"\bnew in\b",
    r"\bwhat happened\b",
    # Explicit web/search intent — require compound context
    r"\b(?:search|browse|look up|scour) (?:the )?web\b",
    r"\bweb (?:search|result|browser)\b",
    r"\bcrawl\b",
    r"\bscrape\b",
    r"\bfact.?check\b",
    # Live data — require compound (avoid "I live in", "live music")
    r"\blive (?:score|update|stream|result|data|feed|game|event|match|fight)\b",
    r"\bgo live\b",
    # Updates — require qualifier to avoid "update the document"
    r"\b(?:latest|recent|new)\s+update(?:s)?\b",
    # Trending (specific enough as single word; "trend"/"trends" alone are not)
    r"\btrending\b",
    # Social platform references (explicit intent to fetch from platform)
    r"\breddit\b",
    r"\btwitter\b",
    # Citations and explicit sourcing requests — compound to avoid "open source", "source code"
    r"\bcitation(?:s)?\b",
    r"\bcite (?:a |your )?source(?:s)?\b",
    r"\bfind (?:a |the )?source(?:s)?\b",
    r"\bweb source(?:s)?\b",
    r"\bsource(?:s)? for this\b",
    # Raw URL in the message
    r"https?://",
)


_EVOLVING_TOPIC_PATTERNS: tuple[re.Pattern[str], ...] = (
    # "Who is/runs/leads/owns the ..." — positional roles that change
    re.compile(r"\bwho\s+(?:is|are|was|runs?|leads?|owns?|heads?|manages?)\s+(?:the\s+)?", re.I),
    # "Who is the X of Y" — CEO of Tesla, president of France
    re.compile(r"\bwho\s+is\s+(?:the\s+)?\w+\s+of\b", re.I),
    # "What's the [metric]" — unemployment, inflation, price, rate, GDP, population
    re.compile(
        r"\bwhat(?:'s| is| are)\s+(?:the\s+)?(?:current\s+)?"
        r"(?:unemployment|inflation|interest|gdp|population|price|cost|rate|salary|wage|minimum wage"
        r"|gas price|stock price|market cap|net worth|cap rate|mortgage)",
        re.I,
    ),
    # "How much does/is X" — prices, costs, salaries
    re.compile(r"\bhow much\s+(?:does|is|are|do)\b", re.I),
    # "Is X still [state]" / "Does X still [verb]" — checking if something changed
    re.compile(r"\bis\s+\w+\s+still\b", re.I),
    re.compile(r"\bdoes\s+\w+\s+still\b", re.I),
    # Champion / winner / MVP / record holder — titles that rotate
    re.compile(r"\b(?:champion|winner|mvp|record holder|number one|world record|ballon d.or|heisman)\b", re.I),
    # Standings, rankings, leaderboard — inherently time-varying
    re.compile(r"\b(?:standings|rankings?|leaderboard|playoff|seed(?:ing)?|bracket)\b", re.I),
    # Political / leadership office holders
    re.compile(r"\b(?:president|prime minister|chancellor|governor|mayor|ceo|cfo|cto|chairman|secretary of)\b", re.I),
    # "When is the next [event]"
    re.compile(r"\bwhen\s+is\s+(?:the\s+)?next\b", re.I),
    # Software versioning
    re.compile(r"\b(?:latest|newest|current)\s+version\s+of\b", re.I),
    re.compile(r"\bwhat\s+version\s+(?:is|of)\b", re.I),
    # Availability / legality — policies change
    re.compile(r"\bis\s+\w+\s+(?:legal|illegal|banned|available|approved)\s+in\b", re.I),
)


def is_evolving_topic(text: str) -> bool:
    """Return True if the query concerns a topic whose answer changes over time,
    even though the user didn't use explicit recency keywords."""
    low = str(text or "").strip()
    if not low:
        return False
    # Don't double-fire if already caught by recency terms
    if is_recency_sensitive(low):
        return False
    return any(pat.search(low) for pat in _EVOLVING_TOPIC_PATTERNS)


def is_recency_sensitive(text: str) -> bool:
    """Return True if the query explicitly asks for current or live information."""
    low = str(text or "").strip().lower()
    if not low:
        return False
    for term in RECENCY_TERMS:
        pattern = rf"\b{re.escape(term)}\b"
        if re.search(pattern, low, flags=re.IGNORECASE):
            return True
    return False


def is_recency_sensitive_from_history(prior_messages: list, lookback: int = 4) -> bool:
    """Return True if recent assistant messages established a recency/training-data framing."""
    for msg in reversed(prior_messages[-lookback:]):
        if str(msg.get("role", "")).strip().lower() != "assistant":
            continue
        content = str(msg.get("content", "")).strip().lower()
        if any(phrase in content for phrase in [
            "as of ", "last i knew", "my training", "training data",
            "may have changed", "may be outdated", "want me to search",
            "want me to forage", "live search", "live forage",
        ]):
            return True
    return False


def extract_rejected_tool(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    patterns = (
        r"\b(?:don[' ]?t|do not|dont)\s+want\s+(?:to\s+)?(?:use|work with|via)\s+([a-zA-Z0-9._\- ]{2,40})\b",
        r"\b(?:not using|no)\s+([a-zA-Z0-9._\- ]{2,40})\b",
    )
    for pattern in patterns:
        match = re.search(pattern, raw, flags=re.IGNORECASE)
        if not match:
            continue
        candidate = str(match.group(1) or "").strip(" .,!?:;\"'")
        if candidate:
            return candidate[:40]
    return ""


_FACTUAL_LOOKUP_PATTERNS: tuple[str, ...] = (
    # Wh-questions — full and contracted forms
    r"\bwhat(?:'s|'re| is| are| was| were)\s+",
    r"\bwho(?:'s| is| was| were| did| built| created| founded| invented| wrote| made| runs?| owns?| leads?| heads?)\b",
    r"\bwhen(?:'s| did| was| were| does| is)\s+",
    r"\bwhere(?:'s| is| was| did| does)\s+",
    r"\bhow\s+(?:does|did|do|has|have|many|much|often|long|far|old|fast|big|small|tall|serious|dangerous|safe)\s+",
    r"\bwhy\s+(?:does|did|is|was|do|are|were)\s+",
    # Existence and state checks
    r"\bare\s+there\s+",
    r"\bis\s+it\s+true\b",
    r"\bdoes\s+\S",
    r"\bdo\s+(?:you\s+know\s+(?:about|anything|who|what|where|when)|they|people)\b",
    # Explicit lookup and research requests
    r"\btell\s+me\s+about\b",
    r"\bexplain\b",
    r"\bdefine\b",
    r"\blook\s+up\b",
    r"\bfind\s+(?:out|information|info|details?|facts?|anything)\b",
    r"\bresearch\s+\S",
    r"\bany\s+(?:info|information|news|updates?|details?)\s+(?:on|about|regarding)\b",
    r"\bwhat\s+do\s+(?:you\s+know|we\s+know)\s+about\b",
    # Event and cause questions
    r"\bwhat\s+happened\b",
    r"\bwhat\s+(?:caused|led\s+to|came\s+of|became\s+of)\b",
    # "Tell/show/give me" factual requests
    r"\bgive\s+me\s+(?:info|information|details?|facts?|background|context)\b",
    r"\bshow\s+me\s+(?:info|information|details?|facts?|what)\b",
)


def should_offer_web(text: str, lane: str) -> bool:
    """Return True if web research should be offered for this query and lane."""
    lane_key = lane.strip().lower()
    if lane_key == "research":
        return True
    if lane_key != "project":
        return False
    low = str(text or "").strip().lower()
    if not low:
        return False
    if any(re.search(pattern, low, flags=re.IGNORECASE) for pattern in _WEB_OFFER_MARKER_PATTERNS):
        return True
    if is_evolving_topic(text):
        return True
    return any(re.search(p, low, flags=re.IGNORECASE) for p in _FACTUAL_LOOKUP_PATTERNS)
