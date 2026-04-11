"""Delivery target inference — classifies a research/make request into a content type."""

from __future__ import annotations


_TARGET_ALIASES: dict[str, str] = {
    "general": "general",
    "module": "standalone_app",
    "widget": "standalone_app",
    "module_widget": "standalone_app",
    "standalone": "standalone_app",
    "app": "standalone_app",
    "med": "medical",
    "health": "medical",
    "healthcare": "medical",
    "financial": "finance",
    "fin": "finance",
    "sport": "sports",
    "historical": "history",
    "gdd": "game_design_doc",
    "dashboard": "dashboard",
    "blog": "blog", "blog_post": "blog",
    "social_post": "social_post", "social_media": "social_post", "social": "social_post",
    "landing_page": "landing_page", "landing": "landing_page",
    "api": "api", "rest_api": "api", "api_service": "api",
    "screenplay": "screenplay", "play": "screenplay", "teleplay": "screenplay",
}


def infer_delivery_target(text: str, explicit_target: str, mode: str = "research") -> str:
    """Return the canonical delivery target string for a request.

    Args:
        text: The user's request text (used for keyword inference when target is 'auto').
        explicit_target: A target string provided by the caller, or '' / 'auto' to infer.
        mode: Current pipeline mode ('research', 'make', 'build', etc.).
    """
    current_mode = str(mode or "research").strip().lower()
    target = str(explicit_target or "").strip().lower()
    target = _TARGET_ALIASES.get(target, target)

    if target and target != "auto":
        if target == "web_app" and current_mode in {"make", "build"}:
            return "standalone_app"
        return target

    low = text.lower()

    medical_tokens = ("medical", "health", "clinical", "symptom", "diagnosis", "treatment",
                      "medication", "doctor", "hospital", "vet")
    if any(tok in low for tok in medical_tokens):
        return "medical"

    finance_tokens = ("finance", "financial", "stock", "market", "invest", "portfolio",
                      "budget", "economy", "macro", "earnings")
    if any(tok in low for tok in finance_tokens):
        return "finance"

    app_tokens = ("web app", "flask", "vue", "frontend", "backend", "sqlite")
    if any(tok in low for tok in app_tokens):
        if current_mode in {"make", "build"}:
            return "standalone_app"
        return "web_app"

    sports_tokens = ("sports", "nba", "nfl", "mlb", "nhl", "ufc", "mma", "soccer",
                     "football", "baseball", "basketball")
    if any(tok in low for tok in sports_tokens):
        return "sports"

    history_tokens = ("historical", "history", "world war", "ww1", "ww2", "civil war",
                      "timeline", "historiography")
    if any(tok in low for tok in history_tokens):
        return "history"

    general_tokens = ("current events", "pop culture", "culture", "news")
    if any(tok in low for tok in general_tokens):
        return "general"

    if "game design" in low or "gdd" in low:
        return "game_design_doc"
    if "screenplay" in low or "teleplay" in low:
        return "screenplay"
    if "script" in low:
        code_signals = ("python", "bash", "automate", "process", "parse", "convert", "extract", "cron", "cli")
        if any(s in low for s in code_signals):
            return "script"
        return "screenplay"
    if "essay" in low:
        return "essay"
    if "email" in low or "e-mail" in low:
        return "email"
    if "memoir" in low:
        return "memoir"
    if "novel" in low:
        return "novel"
    if "book" in low or "chapter" in low:
        return "book"
    if "dashboard" in low:
        return "dashboard"
    if "blog" in low or "blog post" in low:
        return "blog"
    if "social media" in low or "social post" in low or "tweet" in low or "instagram" in low:
        return "social_post"
    if "landing page" in low:
        return "landing_page"
    if "api" in low and ("rest" in low or "endpoint" in low or "service" in low or "backend" in low):
        return "api"
    return "document"
