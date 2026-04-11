from __future__ import annotations

import json as _json
import re
import threading
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from flask import Blueprint, abort, jsonify, request

from shared_tools.comfyui_client import ComfyUIClient
from shared_tools.content_guardrails import check_content
from shared_tools.image_gen_presets import find_image_gen_preset, resolve_preset_lora_name
from shared_tools.model_routing import lane_model_config
from web_gui.chat_helpers import bg_retitle, bg_summarize, handle_command
from web_gui.utils.file_utils import normalize_project_slug as _normalize_project_slug
from web_gui.utils.history_builders import (
    build_command_history as _build_command_history,
    build_fact_history as _build_fact_history,
    build_talk_history as _build_talk_history,
    extract_talk_text as _extract_talk_text,
)

if TYPE_CHECKING:
    from web_gui.app_context import AppContext

_IMAGE_GEN_DIRECT_RE = re.compile(r"\b(/imagine|text[- ]?to[- ]?image|t2i|recreate)\b", re.IGNORECASE)
_IMAGE_REF_TOKEN_RE = re.compile(r"\{image\s*\d+\}", re.IGNORECASE)
_IMAGE_GEN_RECREATE_RE = re.compile(r"^recreate\b[^a-z]*$", re.IGNORECASE)
_IMAGE_GEN_VERB_RE = re.compile(
    r"\b(draw|paint|generate|create|make|render|illustrate|imagine|design)\b",
    re.IGNORECASE,
)
_IMAGE_GEN_NOUN_RE = re.compile(
    r"\b(image|picture|photo|illustration|art|artwork|portrait|wallpaper)\b",
    re.IGNORECASE,
)
_IMAGE_GEN_OF_RE = re.compile(
    r"\b(?:an?\s+)?(?:image|picture|photo|illustration|portrait|artwork)\s+of\b",
    re.IGNORECASE,
)
_IMAGE_GEN_HELP_RE = re.compile(
    r"\b(?:how\s+to|how\s+do\s+i|how\s+can\s+i|teach\s+me\s+to|guide\s+me\s+to)\b.*\b(?:image|picture|photo|art|illustration)\b",
    re.IGNORECASE,
)
_NIGHT_SCENE_RE = re.compile(r"\b(night|dark)\b", re.IGNORECASE)
_RELATION_SCENE_RE = re.compile(r"\b(at the bottom of|in front of|behind|foreground|background)\b", re.IGNORECASE)
_FIRE_RE = re.compile(r"\b(on fire|burning|ablaze|in flames|flames)\b", re.IGNORECASE)
_SUBJECT_ENTITY_RE = re.compile(
    r"\b("
    r"person|people|human|man|woman|child|hero|warrior|creature|animal|dragon|fox|wolf|cat|dog|bird|"
    r"fortress|castle|citadel|keep|building|house|tower|bridge|ship|boat|vehicle|car|train|robot|monster"
    r")\b",
    re.IGNORECASE,
)
_SETTING_ENTITY_RE = re.compile(
    r"\b("
    r"landscape|mountain|mountains|mountain pass|valley|forest|meadow|field|desert|coast|shore|sea|ocean|"
    r"river|city|town|village|street|sky|horizon|snow|snowy|background|foreground|midground"
    r")\b",
    re.IGNORECASE,
)


def _is_image_gen_request(text: str) -> bool:
    low = str(text or "").strip().lower()
    if not low:
        return False
    if _IMAGE_GEN_HELP_RE.search(low):
        return False
    if _IMAGE_GEN_DIRECT_RE.search(low):
        return True
    if _IMAGE_GEN_OF_RE.search(low):
        return True
    return bool(_IMAGE_GEN_VERB_RE.search(low) and _IMAGE_GEN_NOUN_RE.search(low))


def _normalize_lora_selection(raw: Any) -> list[str]:
    values = raw if isinstance(raw, list) else []
    seen: set[str] = set()
    out: list[str] = []
    for item in values:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text[:220])
        if len(out) >= 32:
            break
    return out


def _parse_selected_loras_value(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return _normalize_lora_selection(raw)
    text = str(raw or "").strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            payload = _json.loads(text)
            if isinstance(payload, list):
                return _normalize_lora_selection(payload)
        except Exception:
            return []
        return []
    return _normalize_lora_selection([part.strip() for part in text.split(",") if part.strip()])


def _to_bool(raw: Any, *, default: bool = False) -> bool:
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    text = str(raw).strip().lower()
    if not text:
        return default
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _to_int(raw: Any, *, default: int | None = None) -> int | None:
    if raw is None:
        return default
    text = str(raw).strip()
    if not text:
        return default
    try:
        return int(text)
    except (TypeError, ValueError):
        return default


def _to_float(raw: Any, *, default: float | None = None) -> float | None:
    if raw is None:
        return default
    text = str(raw).strip()
    if not text:
        return default
    try:
        return float(text)
    except (TypeError, ValueError):
        return default


def _is_simple_image_prompt(prompt: str) -> bool:
    text = " ".join(str(prompt or "").strip().split())
    if not text:
        return False
    words = [w for w in re.split(r"\s+", text) if w]
    if len(words) <= 7:
        return True
    if len(words) <= 12 and "," not in text and "." not in text and ":" not in text:
        return True
    return False


def _has_structured_scene_request(prompt: str) -> bool:
    text = " ".join(str(prompt or "").strip().split())
    if not text:
        return False
    if _RELATION_SCENE_RE.search(text):
        return True
    has_subject = bool(_SUBJECT_ENTITY_RE.search(text))
    has_setting = bool(_SETTING_ENTITY_RE.search(text))
    return has_subject and has_setting


def _canonical_entity_name(raw: str) -> str:
    token = str(raw or "").strip().lower()
    if token in {"castle", "citadel", "keep"}:
        return "fortress"
    if token in {"people", "human", "man", "woman", "child", "hero", "warrior"}:
        return "person"
    if token in {"animal", "dragon", "fox", "wolf", "cat", "dog", "bird", "monster"}:
        return "creature"
    if token in {"boat"}:
        return "ship"
    if token in {"vehicle"}:
        return "car"
    return token


def _extract_required_entities(prompt: str, *, max_items: int = 6) -> list[str]:
    text = " ".join(str(prompt or "").strip().split())
    if not text:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for match in _SUBJECT_ENTITY_RE.finditer(text):
        name = _canonical_entity_name(match.group(0))
        if not name or name in seen:
            continue
        seen.add(name)
        out.append(name)
        if len(out) >= max_items:
            break
    if _FIRE_RE.search(text) and "fire" not in seen and len(out) < max_items:
        out.append("fire")
    return out


def _extract_required_conditions(prompt: str, *, max_items: int = 6) -> list[str]:
    text = " ".join(str(prompt or "").strip().split()).lower()
    if not text:
        return []

    conditions: list[str] = []
    seen: set[str] = set()

    def _push(value: str) -> None:
        key = str(value or "").strip().lower()
        if not key or key in seen:
            return
        seen.add(key)
        conditions.append(value.strip())

    if _FIRE_RE.search(text):
        _push("requested fire is explicit with visible flames and smoke")

    # Generic condition pairing: if a known subject entity appears near fire terms,
    # force that subject's state to be explicit in-frame.
    for match in _SUBJECT_ENTITY_RE.finditer(text):
        entity = _canonical_entity_name(match.group(0))
        if not entity:
            continue
        start, end = match.span()
        near_after = text[end:min(len(text), end + 48)]
        near_before = text[max(0, start - 48):start]
        if _FIRE_RE.search(near_after) or _FIRE_RE.search(near_before):
            _push(f"{entity} is visibly on fire with active flames and smoke")
        if len(conditions) >= max_items:
            break

    return conditions[:max_items]


def _scene_guidance_extras(prompt: str) -> list[str]:
    text = " ".join(str(prompt or "").strip().split())
    has_structured_scene = _has_structured_scene_request(text)
    required_entities = _extract_required_entities(text)
    required_conditions = _extract_required_conditions(text)
    if not has_structured_scene and not required_entities and not required_conditions:
        return []

    extras: list[str] = [
        "single coherent composition where requested elements and environment coexist in one frame",
        "all explicitly requested elements must be clearly visible and recognizable",
        "do not omit, replace, or minimize key requested elements",
    ]
    if required_entities:
        extras.append(f"required visible elements: {', '.join(required_entities)}")
    if required_conditions:
        extras.append(f"required conditions: {'; '.join(required_conditions)}")
        extras.append("required conditions must be literal and clearly visible, not implied")
    if has_structured_scene:
        extras.extend([
            "maintain clear foreground, midground, and background separation",
            "avoid detached split-scene composition",
        ])
    if "fire" in required_entities:
        extras.append("if fire is requested, flames and smoke must be clearly visible on the requested subject")
    return extras


def _merge_negative_prompt_terms(base_negative_prompt: str, extras: list[str]) -> str:
    merged: list[str] = []
    seen: set[str] = set()

    def _append(raw: str) -> None:
        text = str(raw or "").strip()
        if not text:
            return
        key = text.lower()
        if key in seen:
            return
        seen.add(key)
        merged.append(text)

    for chunk in re.split(r"[,\n;]+", str(base_negative_prompt or "")):
        _append(chunk)
    for chunk in extras:
        _append(chunk)
    return ", ".join(merged)


def _refine_negative_prompt(
    negative_prompt: str,
    *,
    prompt: str,
    preset_id: str = "",
) -> str:
    text = " ".join(str(prompt or "").strip().split())
    if not text:
        return str(negative_prompt or "").strip()
    has_structured_scene = _has_structured_scene_request(text)
    required_entities = _extract_required_entities(text)
    required_conditions = _extract_required_conditions(text)
    if not has_structured_scene and not required_entities and not required_conditions:
        return str(negative_prompt or "").strip()

    extras = [
        "split screen",
        "diptych",
        "triptych",
        "collage",
        "comic panel",
        "multi frame",
        "subject omitted",
        "tiny distant subject",
        "out of frame subject",
        "empty landscape",
        "detached foreground/background",
        "substituted subject",
    ]
    extras.extend([f"missing {name}" for name in required_entities if name and name != "fire"])
    if "fire" in required_entities:
        extras.extend([
            "flames not visible",
            "smoke not visible",
            "unlit subject",
            "fire only implied",
        ])
    extras.extend([f"missing condition: {item}" for item in required_conditions if str(item).strip()])

    preset_key = str(preset_id or "").strip().lower()
    _pony_presets = {"and_the_hound", "borderfox", "uwu_figurine", "painterly", "realism", "fixel", "pastels", "unfinished_anime"}
    _strict_animal_presets = {"and_the_hound", "fixel"}
    # Presets that must never produce furry/pony/cartoon source output regardless of subject
    _antifurry_presets = {"pastels", "unfinished_anime"}
    if preset_key in _pony_presets and _pony_is_human_subject(text):
        extras.extend(_PONY_HYBRID_NEGATIVES)
    elif preset_key in _strict_animal_presets and bool(_PONY_ANIMAL_RE.search(text)):
        # Animal subject in a Pony preset — lock out anthro drift from both directions
        extras.extend(_PONY_HYBRID_NEGATIVES)
        extras.extend(["humanoid", "bipedal", "standing upright", "human hands", "human feet", "clothed animal"])
        extras.extend(_PONY_ANIMAL_NSFW_NEGATIVES)
    if preset_key in _antifurry_presets:
        extras.extend(_PONY_SOURCE_ANTIFURRY_NEGATIVES)
    return _merge_negative_prompt_terms(negative_prompt, extras)


_PONY_ANIMAL_RE = re.compile(
    r"\b(dog|cat|fox|wolf|bear|rabbit|bunny|deer|horse|dragon|lion|tiger|bird|owl|raccoon|"
    r"snake|lizard|shark|fish|feline|canine|pony|mare|stallion|beast|creature|monster|"
    r"animal|fur|furry|kemono|anthro)\b",
    re.IGNORECASE,
)
_PONY_HUMAN_RE = re.compile(
    r"\b(human|person|man|woman|girl|boy|lady|gentleman|warrior|wizard|princess|prince|"
    r"knight|mage|elf|dwarf|hero|villain|character|figure|portrait|face)\b",
    re.IGNORECASE,
)
_PONY_HYBRID_NEGATIVES = [
    "tail", "animal tail", "animal ears", "cat ears", "dog ears", "wolf ears", "fox ears",
    "fur", "furry", "kemono", "anthro", "animal features", "hybrid", "beast",
    "snout", "muzzle", "paws", "claws on hands", "animal nose",
]
_PONY_ANIMAL_NSFW_NEGATIVES = [
    "nsfw", "nude", "naked", "explicit", "suggestive", "sexual", "lewd", "adult content",
    "breasts", "large breasts", "huge breasts", "big breasts", "cleavage", "nipples",
    "anthro female", "anthro male", "sexy pose", "pinup",
    "rating:explicit", "rating:questionable",
]
# Pony source-tag negatives — suppress furry/pony/cartoon training data bias entirely
_PONY_SOURCE_ANTIFURRY_NEGATIVES = [
    "source_furry", "source_pony", "source_cartoon",
    "furry", "anthro", "kemono", "animal ears", "animal tail", "fur", "pony",
    "score_4", "score_5",
]


def _pony_is_human_subject(text: str) -> bool:
    """True when the prompt clearly describes a human and mentions no animals."""
    return bool(_PONY_HUMAN_RE.search(text)) and not bool(_PONY_ANIMAL_RE.search(text))


def _refine_image_prompt(
    prompt: str,
    *,
    image_style: str,
    selected_loras: list[str],
    has_references: bool,
    preset_id: str = "",
    refiner_profile: dict[str, Any] | None = None,
    scene_subject: str = "",
) -> str:
    text = " ".join(str(prompt or "").strip().split())
    subject = str(scene_subject or "").strip().lower()
    if not text:
        return text
    is_simple = _is_simple_image_prompt(text)
    scene_extras = _scene_guidance_extras(text)
    if not is_simple and not scene_extras:
        return text

    preset_key = str(preset_id or "").strip().lower()
    if preset_key == "foxo_slyesium":
        extras = [
            "masterpiece",
            "best quality",
            "8k",
            "oil painting",
            "soft lighting",
            "ZaUm",
        ]
        _people_re = re.compile(r"\b(person|people|man|woman|girl|boy|character|figure|face|portrait)\b", re.IGNORECASE)
        is_char = subject == "character" or (not subject and _people_re.search(text))
        extras.append("elysiumChar" if is_char else "elysiumScape")
        extras.extend(scene_extras)
        return f"{text}, {', '.join(extras)}"

    if preset_key == "pixel_forge":
        extras = [
            "pixel",
            "pixel art",
            "pixelated",
            "limited color palette",
            "masterpiece",
            "best quality",
        ]
        if subject == "character":
            extras.extend(["character sprite", "full body", "clean outline"])
        elif subject == "scene":
            extras.extend(["pixel art background", "tileset style", "detailed scenery"])
        else:
            extras.append("retro game style")
        extras.extend(scene_extras)
        return f"{text}, {', '.join(extras)}"

    if preset_key == "fhoxi":
        extras = [
            "chibi",
            "cute",
            "(masterpiece)",
            "(best quality)",
            "(ultra-detailed)",
        ]
        if subject == "scene":
            extras.extend(["chibi scenery", "cute environment", "whimsical background"])
        elif subject == "object":
            extras.extend(["cute item", "chibi style object", "simple background"])
        else:
            extras.extend(["(full body:1.2)", "smile", "(beautiful detailed face)", "(beautiful detailed eyes)"])
        extras.extend(scene_extras)
        return f"{text}, {', '.join(extras)}"

    if preset_key == "faceless_uwu":
        extras = [
            "anime minimalist",
            "flat color",
            "clean lines",
        ]
        if subject == "scene":
            extras.extend(["minimalist landscape", "flat background", "simple scenery"])
        elif subject == "object":
            extras.extend(["simple object", "flat illustration", "white background"])
        else:
            extras.extend(["solo", "simple background", "faceless"])
        extras.extend(scene_extras)
        return f"{text}, {', '.join(extras)}"

    if preset_key == "nutshell":
        extras = [
            "Kurzgesagt style",
            "by Kurzgesagt",
            "vector artwork",
            "2D flat illustration",
            "clean color",
            "clear boundaries",
            "bright colors",
            "high contrast",
            "tidy style",
            "sharp focus",
            "HDR",
            "fine art",
            "masterpiece",
            "best quality",
        ]
        if subject == "character":
            extras.extend(["illustrated character", "expressive pose", "bold silhouette"])
        elif subject == "object":
            extras.extend(["product illustration", "clean white background", "icon style"])
        elif subject == "scene":
            extras.extend(["civilization", "epic landscape", "silhouette"])
        extras.extend(scene_extras)
        return f"{text}, {', '.join(extras)}"

    if preset_key == "foxs_moving_castle":
        extras = [
            "studio ghibli inspired style",
            "rich painterly textures and details",
            "cinematic composition with a clear primary focal subject",
            "single continuous scene in one frame",
            "no split-screen, no diptych, no collage, no multi-panel layout",
        ]
        if subject == "character":
            extras.extend(["character portrait", "expressive face", "soft warm lighting", "detailed clothing"])
        elif subject == "object":
            extras.extend(["detailed object", "whimsical design", "soft focus background"])
        else:
            extras.extend(["whimsical handcrafted architecture", "interior and scenery", "keep buildings and people as dominant frame elements"])
        extras.extend(scene_extras)
        night_terms = [
            str(x).strip().lower()
            for x in ((refiner_profile or {}).get("night_terms", []))
            if str(x).strip()
        ]
        night_pattern = _NIGHT_SCENE_RE if not night_terms else re.compile(
            r"\b(" + "|".join([re.escape(term) for term in night_terms]) + r")\b",
            re.IGNORECASE,
        )
        if night_pattern.search(text):
            extras.extend([
                "nighttime ambience",
                "glowing lamps and warm window light",
                "deep shadows with soft atmospheric haze",
            ])
        if has_references:
            extras.append("preserve key layout cues from reference images")
        return f"{text}, {', '.join(extras)}"

    if preset_key == "painterly":
        extras = [
            "score_9",
            "score_8_up",
            "score_7_up",
            "score_6_up",
            "score_5_up",
            "score_4_up",
            "abstractionism",
            "brush stroke",
            "traditional media",
        ]
        if subject == "scene":
            extras.extend([
                "outdoors",
                "detailed background",
                "painterly landscape",
                "expressive brushwork",
                "rich color palette",
                "atmospheric depth",
            ])
        elif subject == "object":
            extras.extend([
                "still life",
                "painterly composition",
                "expressive texture",
                "rich color",
                "dramatic lighting",
            ])
        else:
            extras.extend([
                "solo",
                "looking at viewer",
                "upper body",
                "expressive brushwork",
                "rich detail",
            ])
            if _pony_is_human_subject(text):
                extras.extend(["human", "fully human"])
        extras.extend(scene_extras)
        return f"{text}, {', '.join(extras)}"

    if preset_key == "borderfox":
        extras = [
            "score_9",
            "score_8_up",
            "score_7_up",
            "zPDXL",
            "Akaburstyle",
        ]
        if subject == "scene":
            extras.extend([
                "Akaburstyle background",
                "detailed environment",
                "painterly scenery",
                "cinematic composition",
                "dramatic lighting",
                "no characters",
            ])
        elif subject == "object":
            extras.extend([
                "Akaburstyle",
                "detailed prop",
                "stylized illustration",
                "clean composition",
                "dramatic lighting",
            ])
        else:
            extras.extend([
                "solo",
                "looking at viewer",
                "close up",
                "dramatic lighting",
                "sharp focus",
            ])
            if _pony_is_human_subject(text):
                extras.extend(["human", "fully human", "detailed face"])
            else:
                extras.extend(["detailed face", "detailed fur"])
        extras.extend(scene_extras)
        return f"{text}, {', '.join(extras)}"

    if preset_key == "uwu_figurine":
        extras = [
            "high resolution",
            "score_9",
            "score_8_up",
            "score_8",
            "figure",
        ]
        if subject == "scene":
            extras.extend(["diorama", "miniature scene", "figurine display", "detailed base", "studio lighting"])
        elif subject == "object":
            extras.extend(["prop figurine", "detailed sculpt", "clean finish", "white background", "studio lighting"])
        else:
            extras.extend(["cute", "solo", "looking at viewer", "smile", "full body", "smooth clean surface", "studio lighting", "white background"])
            if _pony_is_human_subject(text):
                extras.extend(["human", "fully human"])
        extras.extend(scene_extras)
        return f"{text}, {', '.join(extras)}"

    if preset_key == "and_the_hound":
        extras = [
            "score_9",
            "score_8_up",
            "score_7_up",
            "zPDXL",
            "DisneyRenstyle",
        ]
        if subject == "scene":
            # Landscapes/environments: Disney Renaissance painterly backgrounds
            extras.extend([
                "disney renaissance background",
                "lush detailed environment",
                "painterly scenery",
                "rich color palette",
                "cinematic composition",
                "soft atmospheric lighting",
                "no characters",
            ])
        elif subject == "object":
            # Objects: stylized props in Disney aesthetic, avoid the animal-specific anatomy negatives
            extras.extend([
                "disney renaissance style object",
                "stylized prop",
                "vibrant colors",
                "detailed illustration",
                "soft shadows",
                "clean composition",
            ])
        else:
            extras.extend([
                "expressive character",
                "vibrant colors",
                "soft warm lighting",
                "lively pose",
            ])
            if _pony_is_human_subject(text):
                extras.extend(["human", "fully human", "detailed face"])
            else:
                extras.append("detailed fur")
        extras.extend(scene_extras)
        return f"{text}, {', '.join(extras)}"

    if preset_key == "realism":
        extras = [
            "score_9",
            "score_8_up",
            "score_7_up",
            "highly detailed",
            "film grain",
        ]
        if subject == "scene":
            extras.extend(["scenery", "dynamic angle", "atmospheric depth", "natural colors", "environment"])
        elif subject == "object":
            extras.extend(["close-up", "sharp detail", "natural lighting", "film grain"])
        else:
            extras.extend(["dynamic angle", "natural lighting", "detailed skin", "sharp focus"])
            if _pony_is_human_subject(text):
                extras.extend(["human", "fully human"])
        extras.extend(scene_extras)
        return f"{text}, {', '.join(extras)}"

    if preset_key == "fixel":
        extras = [
            "score_9",
            "score_8_up",
            "score_7_up",
            "score_6_up",
            "score_5_up",
        ]
        if subject == "scene":
            extras.extend(["outdoors", "detailed background", "pixel art scenery"])
        elif subject == "object":
            extras.extend(["simple background", "centered", "product style"])
        else:
            is_human = _pony_is_human_subject(text)
            is_animal = bool(_PONY_ANIMAL_RE.search(text))
            if is_human and not is_animal:
                extras.extend(["solo", "looking at viewer", "upper body", "human", "fully human"])
            elif is_animal and not is_human:
                extras.extend([
                    "rating:safe",
                    "full body animal",
                    "quadruped",
                    "no human features",
                    "no clothing",
                    "realistic animal anatomy",
                    "wildlife photography",
                    "non-anthropomorphic",
                ])
            else:
                extras.extend(["solo", "looking at viewer", "upper body"])
        extras.extend(scene_extras)
        return f"{text}, {', '.join(extras)}"

    if preset_key == "sketch_book":
        extras = ["black and white drawing", "on white paper", "pencil sketch", "fine linework", "hand drawn"]
        if subject == "scene":
            extras.extend(["architectural detail", "cross-hatching", "ink wash"])
        elif subject == "character":
            extras.extend(["figure study", "expressive lines"])
        elif subject == "object":
            extras.extend(["still life sketch", "clean outlines"])
        extras.extend(scene_extras)
        return f"{text}, {', '.join(extras)}"

    if preset_key == "shirt_designs":
        extras = ["T shirt design", "TshirtDesignAF", "bold lineart", "fabric texture", "flat design"]
        if subject == "scene":
            extras.extend(["landscape background", "dynamic perspective"])
        elif subject == "character":
            extras.extend(["character illustration", "dynamic pose"])
        elif subject == "object":
            extras.extend(["centered object", "clean composition"])
        extras.extend(scene_extras)
        return f"{text}, {', '.join(extras)}"

    if preset_key == "wallace_vomit":
        extras = ["claymation", "stopmotion", "clay texture", "3d clay render", "soft lighting"]
        if subject == "character":
            extras.extend(["clay figure", "expressive face", "tactile surface"])
        elif subject == "scene":
            extras.extend(["miniature set", "handcrafted environment"])
        extras.extend(scene_extras)
        return f"{text}, {', '.join(extras)}"

    if preset_key == "ms_fainx":
        # Ensure trigger word is present; otherwise pass through untouched
        trigger = "MSPaint Portrait"
        if trigger.lower() not in text.lower():
            return f"{trigger} of {text}"
        return text

    if preset_key == "parchment":
        extras = [
            "on parchment",
            "illustrated",
            "annotated",
            "ink and pigment",
            "aged texture",
            "detailed linework",
            "dramatic composition",
        ]
        if subject == "scene" or not subject:
            extras.extend(["wide establishing view", "atmospheric depth"])
        elif subject == "character":
            extras.extend(["silhouette", "expressive pose", "dramatic light"])
        extras.extend(scene_extras)
        return f"{text}, {', '.join(extras)}"

    if preset_key == "foxjourney":
        extras = [
            "highly detailed",
            "intricate",
            "sharp focus",
            "dynamic lighting",
            "epic composition",
            "vibrant colors",
            "masterpiece",
            "professional digital art",
        ]
        if subject == "character":
            extras.extend(["beautiful", "expressive face", "elegant", "detailed portrait"])
        elif subject == "scene":
            extras.extend(["cinematic", "atmospheric", "rich environment", "ambient light"])
        extras.extend(scene_extras)
        return f"{text}, {', '.join(extras)}"

    if preset_key == "unfinished_anime":
        trigger = "oamhfs"
        quality_tags = "score_9, score_8_up, score_7_up, score_6_up, score_5_up, score_4_up, source_anime, screenshots"
        extras = ["anime style", "hand-drawn linework", "expressive shading", "sketch quality", "cel shaded"]
        if subject == "character":
            extras.extend(["detailed face", "expressive eyes", "dynamic pose", "monochrome"])
        elif subject == "scene":
            extras.extend(["anime background", "detailed environment", "cinematic framing"])
        elif subject == "object":
            extras.extend(["simple background", "centered", "clean lines"])
        extras.extend(scene_extras)
        has_trigger = trigger.lower() in text.lower()
        body = f"{text}, {', '.join(extras)}"
        return f"{quality_tags}, {trigger}, {body}" if not has_trigger else f"{quality_tags}, {body}"

    if preset_key == "pastels":
        trigger_phrase = "ncpy13 style pastels drawing"
        quality_tags = "score_9, score_8_up, score_7_up, score_6_up, score_5_up, score_4_up"
        extras = ["pastel colors", "soft chalk texture", "dark background", "glowing light", "painterly"]
        if subject == "scene":
            extras.extend(["atmospheric depth", "ambient light", "rich environment"])
        elif subject == "character":
            extras.extend(["expressive", "detailed face", "soft shading"])
        elif subject == "object":
            extras.extend(["centered composition", "vivid colors"])
        extras.extend(scene_extras)
        has_trigger = trigger_phrase.lower() in text.lower()
        body = f"{text}, {', '.join(extras)}, {quality_tags}"
        return f"{trigger_phrase}, {body}" if not has_trigger else body

    if preset_key == "illustration":
        # "ch" is the LoRA trigger; inject it if absent
        trigger = "ch"
        has_trigger = bool(re.search(r'\bch\b', text))
        extras = ["flat illustration", "storybook style", "colorful", "clean linework", "simple background", "graphic art"]
        if subject == "scene":
            extras.extend(["scenery", "outdoors", "sky", "cloud", "sun"])
        elif subject == "character":
            extras.extend(["solo", "expressive", "stylized figure"])
        elif subject == "object":
            extras.extend(["centered", "decorative", "white background"])
        extras.extend(scene_extras)
        body = f"{text}, {', '.join(extras)}"
        return f"{trigger}, {body}" if not has_trigger else body

    if preset_key == "foxel":
        extras = ["voxel style", "voxel art", "isometric blocks", "3d pixel art", "cubic geometry", "bright colors", "game asset", "toy-like", "clean render"]
        if subject == "character":
            extras.extend(["action figure", "blocky figure", "centered composition"])
        elif subject == "scene":
            extras.extend(["voxel environment", "isometric view", "miniature world"])
        elif subject == "object":
            extras.extend(["voxel model", "centered", "simple background"])
        extras.extend(scene_extras)
        trigger = "voxel style"
        prefix = trigger if trigger.lower() not in text.lower() else ""
        body = f"{text}, {', '.join(extras)}"
        return f"{prefix}, {body}" if prefix else body

    if preset_key == "storyboard":
        trigger = "storyboard sketch of"
        # Trigger is a prefix phrase — strip any existing variant then prepend cleanly
        stripped = text
        for variant in ("storyboard sketch of ", "storyboard sketch "):
            if stripped.lower().startswith(variant):
                stripped = stripped[len(variant):]
                break
        extras = ["storyboard sketch", "black and white", "rough pencil lines", "dynamic composition", "cinematic framing", "action lines"]
        if subject == "character":
            extras.extend(["dramatic pose", "foreshortening", "motion blur", "dutch angle"])
        elif subject == "scene":
            extras.extend(["establishing shot", "wide angle", "environmental detail"])
        elif subject == "object":
            extras.extend(["centered composition", "bold outlines"])
        extras.extend(scene_extras)
        return f"{trigger} {stripped}, {', '.join(extras)}"

    if preset_key == "fs1":
        trigger = "ps1 style"
        extras = ["game screenshot", "computer generated image", "low poly", "pixelated", "retro 3d", "ps1 graphics", "low resolution render", "n64 style"]
        if subject == "character":
            extras.extend(["blocky character model", "limited texture detail"])
        elif subject == "scene":
            extras.extend(["early 3d environment", "foggy draw distance"])
        elif subject == "object":
            extras.extend(["low poly model", "flat textures"])
        prefix = f"({trigger})" if trigger.lower() not in text.lower() else ""
        body = f"{text}, {', '.join(extras)}"
        return f"{prefix}, {body}" if prefix else body

    if preset_key == "lo_fi":
        trigger = "dreamyvibes artstyle"
        prefix = trigger if trigger.lower() not in text.lower() else ""
        extras = ["dreamy", "soft pastel colors", "atmospheric", "cozy mood", "painterly", "lo-fi aesthetic"]
        if subject == "scene":
            extras.extend(["ambient light", "quiet atmosphere", "depth of field"])
        elif subject == "character":
            extras.extend(["gentle expression", "soft focus", "warm tones"])
        elif subject == "object":
            extras.extend(["still life", "soft shadows", "intimate scale"])
        extras.extend(scene_extras)
        body = f"{text}, {', '.join(extras)}"
        return f"{prefix}, {body}" if prefix else body

    extras: list[str] = []
    if image_style == "realistic":
        extras.extend([
            "photorealistic",
            "detailed skin and textures",
            "natural cinematic lighting",
            "35mm photography look",
            "sharp focus",
        ])
    else:
        extras.extend([
            "highly detailed",
            "clean composition",
            "dynamic lighting",
            "crisp edges and textures",
        ])
    if has_references:
        extras.append("preserve key subjects and composition from reference images")
    if selected_loras:
        extras.append("respect selected LoRA style")
    extras.extend(scene_extras)
    return f"{text}, {', '.join(extras)}"


def register_message_routes(bp: Blueprint, ctx: AppContext) -> None:
    @bp.route("/api/conversations/<conversation_id>/messages", methods=["POST"])
    def add_message(conversation_id: str) -> tuple[dict, int]:
        profile = ctx.require_profile()
        store = ctx.conversation_store_for(profile)
        convo = store.get(conversation_id)
        if convo is None:
            abort(404, description="Conversation not found")

        requested_mode = ""
        raw_content = ""
        request_id = ""
        attachments: list[dict[str, Any]] = []
        upload_errors: list[str] = []
        reply_to_data: dict | None = None
        incoming_image_style: str | None = None
        incoming_selected_loras: list[str] | None = None

        content_type = str(request.content_type or "").strip().lower()
        if content_type.startswith("multipart/form-data"):
            raw_content = str(request.form.get("content", "")).strip()
            requested_mode = str(request.form.get("mode", "")).strip().lower()
            request_id = str(request.form.get("request_id", "")).strip()
            attachments, upload_errors = ctx.save_uploaded_images(profile, conversation_id)
            if "image_style" in request.form:
                incoming_image_style = str(request.form.get("image_style", "")).strip().lower()
            if "selected_loras" in request.form:
                incoming_selected_loras = _parse_selected_loras_value(request.form.get("selected_loras", ""))
        else:
            payload = request.get_json(silent=True) or {}
            raw_content = str(payload.get("content", "")).strip()
            requested_mode = str(payload.get("mode", "")).strip().lower()
            request_id = str(payload.get("request_id", "")).strip()
            if "image_style" in payload:
                incoming_image_style = str(payload.get("image_style", "")).strip().lower()
            if "selected_loras" in payload:
                incoming_selected_loras = _parse_selected_loras_value(payload.get("selected_loras"))
            _rt = payload.get("reply_to")
            if isinstance(_rt, dict) and str(_rt.get("id", "")).strip():
                reply_to_data = {
                    "id": str(_rt.get("id", "")).strip(),
                    "role": str(_rt.get("role", "")).strip(),
                    "excerpt": str(_rt.get("excerpt", ""))[:300].strip(),
                }

        if not raw_content and not attachments:
            return {"error": "Message content or image attachment is required"}, 400

        if incoming_image_style is not None or incoming_selected_loras is not None:
            updated = store.set_image_preferences(
                conversation_id,
                image_style=incoming_image_style,
                selected_loras=incoming_selected_loras,
            )
            if updated is not None:
                convo = updated
        selected_loras: list[str] = _normalize_lora_selection(convo.get("selected_loras", []))
        image_style = str(convo.get("image_style", "realistic")).strip().lower() or "realistic"

        talk_text = _extract_talk_text(raw_content)
        is_forage_request = requested_mode == "forage"
        is_make_request = requested_mode == "make"
        is_make_lane_request = is_make_request and not raw_content.startswith("/")
        is_talk_request = (requested_mode == "talk" or talk_text is not None) and not is_forage_request
        if is_make_lane_request and not raw_content and attachments:
            return {"error": "Describe what to build."}, 400
        normalized_talk = (talk_text if talk_text is not None else raw_content).strip()
        if is_talk_request and not normalized_talk and attachments:
            normalized_talk = "Please analyze the attached file(s)."
        stored_user_content = normalized_talk if is_talk_request else raw_content
        if not stored_user_content and attachments:
            n_docs = sum(1 for a in attachments if str(a.get("type", "")) == "document")
            n_imgs = sum(1 for a in attachments if str(a.get("type", "")) == "image")
            parts = []
            if n_imgs:
                parts.append(f"{n_imgs} image(s)")
            if n_docs:
                parts.append(f"{n_docs} document(s)")
            stored_user_content = f"Uploaded {', '.join(parts)}."
        user_mode = "talk" if is_talk_request else "command"
        request_id = ctx.job_manager.start(
            profile=profile,
            conversation_id=conversation_id,
            request_id=request_id,
            mode=user_mode,
            user_text=stored_user_content,
        )

        convo_project = _normalize_project_slug(convo.get("project"))
        if not str(convo.get("project", "")).strip():
            store.set_project(conversation_id, convo_project)
        project_update = None
        pipeline_store = ctx.pipeline_for(profile)
        project_mode = pipeline_store.get(convo_project)
        request_project_mode = dict(project_mode)
        if is_make_lane_request:
            request_project_mode["mode"] = "make"

        guard = check_content(raw_content)
        if guard.blocked:
            reply_text = guard.reason
            store.add_message(conversation_id, "assistant", reply_text, mode=user_mode, request_id=request_id)
            ctx.job_manager.finish(profile, request_id, reply=reply_text)
            return jsonify({"reply": reply_text, "request_id": request_id}), 200

        orch = ctx.new_orch(profile)
        if orch.project_slug != convo_project:
            orch.set_project(convo_project)

        command_input_base = raw_content if raw_content else "Please analyze the attached image(s)."
        lane_guess = ""
        is_foraging_request = False
        if is_forage_request:
            lane_guess = "research"
            is_foraging_request = True
        elif is_make_lane_request:
            target_value = str(request_project_mode.get("target", "auto")).strip().lower()
            lane_guess = f"build:{target_value or 'auto'}"
            is_foraging_request = True
        elif not is_talk_request and not raw_content.startswith("/"):
            mode_value = str(request_project_mode.get("mode", "discovery")).strip().lower()
            target_value = str(request_project_mode.get("target", "auto")).strip().lower()
            if mode_value == "make":
                lane_guess = f"build:{target_value or 'auto'}"
                is_foraging_request = True
            else:
                try:
                    lane_guess = str(orch.router.route(command_input_base, project_slug=convo_project)).strip().lower()
                except Exception:
                    lane_guess = ""
                is_foraging_request = lane_guess in {"research", "project"}

        is_image_gen_request = _is_image_gen_request(raw_content) and not is_make_lane_request
        has_image_attachments = any(str(a.get("type", "")) == "image" for a in attachments)
        is_image_compose_request = is_image_gen_request and has_image_attachments
        if is_image_gen_request:
            is_foraging_request = False
            is_talk_request = False

        user_msg = store.add_message(
            conversation_id,
            "user",
            stored_user_content,
            mode=user_mode,
            attachments=attachments,
            foraging=is_foraging_request,
            request_id=request_id,
            reply_to=reply_to_data,
        )
        if user_msg is None:
            abort(404, description="Conversation not found")

        def _cancel_requested() -> bool:
            return ctx.job_manager.is_cancel_requested(profile, request_id)

        def _progress(stage: str, detail: str = "", *, summary_path: str = "", raw_path: str = "", web_stack: dict | None = None, agent_event: dict | None = None) -> None:
            ctx.job_manager.update(
                profile,
                request_id,
                stage=stage,
                detail=detail,
                summary_path=summary_path,
                raw_path=raw_path,
                web_stack=web_stack,
                agent_event=agent_event,
            )

        def _cancel_reply() -> str:
            row = ctx.job_manager.get(profile, request_id) or {}
            summary = ctx.job_manager.progress_text(row)
            return (
                "Request cancelled.\n"
                "I stopped this active job at the next safe checkpoint.\n\n"
                "Where I left off:\n"
                f"{summary}"
            )

        _progress("message_received", "Message accepted by API and queued for processing.")
        _progress("orchestrator_ready", f"Active project: {convo_project}")
        if is_foraging_request:
            ctx.foraging_manager.register_job(
                profile=profile,
                conversation_id=conversation_id,
                request_id=request_id,
                project=convo_project,
                lane=lane_guess or "project",
                job_key=ctx.job_manager.key(profile, request_id),
            )
            _progress("foraging_started", f"Foraging task started on lane '{lane_guess or 'project'}'.")
        elif ctx.foraging_manager.active_count() > 0:
            ctx.foraging_manager.request_yield(seconds=150.0)
            _progress("foraging_yield_requested", "Foreground chat/cmd requested temporary Foraging yield.")

        image_context = ""
        doc_context = ""
        image_analysis_failures: list[str] = []
        pipeline_error = ""
        gen_attachments: list[dict] = []
        try:
            image_attachments = [a for a in attachments if str(a.get("type", "")) == "image"]
            doc_attachments = [a for a in attachments if str(a.get("type", "")) == "document"]

            if image_attachments:
                _progress("attachment_analysis", f"Analyzing {len(image_attachments)} image attachment(s).")
                image_context, image_analysis_failures = ctx.describe_image_attachments(
                    profile=profile,
                    conversation_id=conversation_id,
                    orch=orch,
                    attachments=image_attachments,
                    user_text=normalized_talk if is_talk_request else raw_content,
                )
                if image_context.strip():
                    _progress("attachment_analysis_done", "Image context extracted for prompt assembly.")
                elif image_analysis_failures:
                    _progress("attachment_analysis_done", "Image analysis attempted with failures logged.")

            if doc_attachments:
                _progress("attachment_analysis", f"Extracting text from {len(doc_attachments)} document(s).")
                doc_parts: list[str] = []
                for doc_att in doc_attachments:
                    text = str(doc_att.get("extracted_text", "")).strip()
                    name = str(doc_att.get("name", "document"))
                    warning = str(doc_att.get("extraction_warning", "")).strip()
                    if text:
                        doc_parts.append(f"[Document: {name}]\n{text}")
                    elif warning:
                        doc_parts.append(f"[Document: {name} — {warning}]")
                    else:
                        doc_parts.append(f"[Document: {name} — text could not be extracted]")
                doc_context = "\n\n".join(doc_parts)
                if doc_context:
                    _progress("attachment_analysis_done", "Document text extracted for prompt assembly.")

            if _cancel_requested():
                reply_text = _cancel_reply()
                _progress("cancel_acknowledged", "Cancel request accepted before model execution.")
            else:
                reply_text = ""

            if not reply_text and is_image_gen_request:
                _progress("image_gen_queued", "Image generation request detected.")
                attach_dir = ctx.attachment_dir_for(profile, conversation_id)
                attach_dir.mkdir(parents=True, exist_ok=True)
                image_lane = "image_gen_compose" if is_image_compose_request else "image_gen"
                image_positive = _refine_image_prompt(
                    raw_content,
                    image_style=image_style,
                    selected_loras=selected_loras,
                    has_references=is_image_compose_request,
                )
                image_negative = _refine_negative_prompt(
                    "",
                    prompt=raw_content,
                )
                ref_image_paths = [
                    str(ctx.attachment_dir_for(profile, conversation_id) / a["filename"])
                    for a in image_attachments
                    if a.get("filename")
                ] if is_image_compose_request else []
                image_gen_result = orch._run_registered_agent(
                    image_lane,
                    orch._make_agent_task(
                        lane=image_lane,
                        text=image_positive,
                        context={
                            "positive_prompt": image_positive,
                            "negative_prompt": image_negative,
                            "conversation_id": conversation_id,
                            "attach_dir": str(attach_dir),
                            "ref_image_paths": ref_image_paths,
                            "image_style": image_style,
                            "selected_loras": selected_loras,
                        },
                        progress_callback=lambda stage, detail=None: _progress(
                            stage,
                            str(detail.get("note", "") if isinstance(detail, dict) else detail or ""),
                        ),
                    ),
                )
                if image_gen_result.get("ok"):
                    gen_filename = str(image_gen_result.get("filename", ""))
                    gen_url = str(image_gen_result.get("url", ""))
                    gen_seed = int(image_gen_result.get("seed", 0))
                    gen_attachments = [{
                        "id": f"gen_{gen_seed % 100000:05d}",
                        "type": "image",
                        "name": gen_filename,
                        "filename": gen_filename,
                        "mime": "image/png",
                        "size": 0,
                        "url": gen_url,
                    }]
                    reply_text = f"Here is your generated image.\n\n_Prompt: {image_positive}_\n\n_Seed: {gen_seed}_"
                else:
                    reply_text = str(image_gen_result.get("message", "Image generation failed."))
                _progress("image_gen_done", "Image generation completed.")

            elif not reply_text and is_talk_request:
                _progress("talk_mode", "Running conversation-layer reply.")
                talk_input = normalized_talk
                if image_context:
                    talk_input = f"{talk_input}\n\n{image_context}".strip()
                if doc_context:
                    talk_input = f"{talk_input}\n\n{doc_context}".strip()
                if not talk_input:
                    reply_text = "Talk mode message is empty. Send text to continue the conversation."
                else:
                    history = _build_talk_history(convo.get("messages", []), limit_turns=16)
                    capture_history = _build_fact_history(convo.get("messages", []), limit_turns=260)
                    reply_text = orch.conversation_reply(
                        talk_input,
                        history=history,
                        capture_history=capture_history,
                        project=convo_project,
                    )
                _progress("talk_mode_done", "Conversation-layer reply generated.")
            elif not reply_text and raw_content.startswith("/"):
                _progress("command_mode", f"Executing slash command: {raw_content.split(' ', 1)[0]}")
                command_history = _build_command_history(convo.get("messages", []), limit_turns=200)
                fact_history = _build_fact_history(convo.get("messages", []), limit_turns=220)
                history_for_command = fact_history if raw_content.strip().lower() == "/project-facts-refresh" else command_history
                if raw_content.strip().lower() == "/recap":
                    convs = store.list()[:5]
                    lines = ["## Recent Conversations\n"]
                    for row in convs:
                        preview = row.get("summary", "")[:160] or "(no summary yet)"
                        lines.append(f"**{row['title']}** — {row['updated_at'][:10]}\n{preview}\n")
                    reply_text = "\n".join(lines)
                else:
                    reply_text = handle_command(
                        orch,
                        raw_content,
                        command_history=history_for_command,
                        project_mode=project_mode,
                    )
                if raw_content.startswith("/project "):
                    requested = raw_content[len("/project "):].strip()
                    project_update = _normalize_project_slug(requested)
                _progress("command_mode_done", "Slash command execution completed.")
            elif not reply_text:
                _progress("foraging_run", "Running Foraging orchestration.")
                command_input = raw_content if raw_content else "Please analyze the attached file(s)."
                if image_context:
                    command_input = f"{command_input}\n\n{image_context}".strip()
                if doc_context:
                    command_input = f"{command_input}\n\n{doc_context}".strip()
                history = _build_command_history(convo.get("messages", []), limit_turns=18)
                if not orch.project_memory.get_facts(convo_project):
                    orch.refresh_project_facts(history=history, reset=False)
                conversation_summary = store.get_summary(conversation_id) if conversation_id else ""
                reply_text = orch.handle_message(
                    command_input,
                    history=history,
                    project_mode=request_project_mode,
                    cancel_checker=_cancel_requested,
                    pause_checker=ctx.foraging_manager.is_paused,
                    yield_checker=ctx.foraging_manager.should_yield,
                    conversation_summary=conversation_summary,
                    force_research=is_forage_request,
                    force_make=is_make_lane_request,
                    progress_callback=lambda stage, detail=None: _progress(
                        stage,
                        str(detail if not isinstance(detail, dict) else detail.get("note", "") or ""),
                        summary_path=(str(detail.get("summary_path", "")).strip() if isinstance(detail, dict) else ""),
                        raw_path=(str(detail.get("raw_path", "")).strip() if isinstance(detail, dict) else ""),
                        web_stack=(detail if isinstance(detail, dict) and stage == "web_stack_ready" else None),
                        agent_event=(dict(detail, stage=stage) if isinstance(detail, dict) and stage in {"research_pool_started", "research_agent_started", "research_agent_completed"} else None),
                    ),
                )
                _progress("foraging_run_done", "Foraging orchestrator returned final reply.")
        except Exception as exc:
            pipeline_error = str(exc).strip() or "unknown pipeline error"
            _progress("pipeline_error", pipeline_error)
            row = ctx.job_manager.get(profile, request_id) or {}
            progress_summary = ctx.job_manager.progress_text(row)
            if is_foraging_request:
                reply_text = (
                    "Foraging encountered a non-blocking pipeline error after partial progress.\n"
                    "I preserved checkpoints and output paths so you can continue without losing work.\n\n"
                    "Where I left off:\n"
                    f"{progress_summary}\n\n"
                    f"Internal error: {pipeline_error}"
                )
            else:
                reply_text = (
                    "I hit an internal pipeline error while processing this request.\n\n"
                    "Captured progress:\n"
                    f"{progress_summary}\n\n"
                    f"Internal error: {pipeline_error}"
                )
        finally:
            if is_foraging_request:
                ctx.foraging_manager.unregister_job(ctx.job_manager.key(profile, request_id))

        attachment_notes: list[str] = []
        if upload_errors:
            attachment_notes.extend(upload_errors)
        if image_analysis_failures:
            attachment_notes.extend([f"Vision note: {item}" for item in image_analysis_failures[:6]])
        if attachment_notes:
            notes_block = "\n".join([f"- {item}" for item in attachment_notes])
            reply_text = f"{reply_text}\n\nAttachment notes:\n{notes_block}"

        if project_update:
            store.set_project(conversation_id, project=project_update)
        ctx.cache_clear(str(profile.get("id", "")))

        job_row = ctx.job_manager.get(profile, request_id) or {}
        web_stack = job_row.get("web_stack") if isinstance(job_row.get("web_stack"), dict) else {}
        web_sources = [s for s in (web_stack.get("web_sources") or []) if isinstance(s, dict)]
        msg_meta: dict | None = {"web_sources": web_sources} if web_sources else None
        assistant_msg = store.add_message(
            conversation_id,
            "assistant",
            reply_text,
            mode=("talk" if is_talk_request else "command"),
            foraging=is_foraging_request,
            request_id=request_id,
            meta=msg_meta,
            attachments=gen_attachments if gen_attachments else None,
        )
        if assistant_msg is None:
            ctx.job_manager.finish(profile, request_id, status="failed", detail="Failed to persist assistant reply.")
            abort(500, description="Failed to persist assistant reply")

        if is_foraging_request and not pipeline_error:
            try:
                from infra.persistence.repositories import ForageCardRepository as _FCR
                import uuid as _uuid

                card_repo = _FCR(ctx.root)
                job_row = ctx.job_manager.get(profile, request_id) or {}
                summary_path = str(job_row.get("summary_path", "") or "").strip()
                raw_path = str(job_row.get("raw_path", "") or "").strip()
                if summary_path:
                    preview = ""
                    for line in reply_text.strip().splitlines():
                        line = line.strip()
                        if line:
                            preview = line[:300]
                            break
                    card_repo.save_card(
                        {
                            "id": f"fc_{request_id[:12]}_{_uuid.uuid4().hex[:4]}",
                            "title": raw_content[:120] if raw_content else "Forage Research",
                            "project": convo_project or "general",
                            "summary_path": summary_path,
                            "raw_path": raw_path,
                            "query": raw_content[:300] if raw_content else "",
                            "preview": preview,
                            "source_count": 0,
                            "is_pinned": 0,
                            "is_read": 0,
                            "created_at": datetime.now(timezone.utc).isoformat(),
                        }
                    )
            except Exception:
                pass

        updated_early = store.get(conversation_id)
        if updated_early:
            msg_count = len(updated_early.get("messages", []))
            root = ctx.root
            if msg_count >= 4 and msg_count % 4 == 0:
                threading.Thread(target=bg_summarize, args=(conversation_id, store, root), daemon=True).start()
            if msg_count == 4:
                first_user = next((m["content"] for m in updated_early["messages"] if m.get("role") == "user"), "")
                from shared_tools.conversation_store import _clean_title as _ct
                if first_user and updated_early.get("title", "") == _ct(first_user):
                    threading.Thread(target=bg_retitle, args=(conversation_id, store, root), daemon=True).start()

        updated = store.get(conversation_id)
        if updated is None:
            ctx.job_manager.finish(profile, request_id, status="failed", detail="Failed to load updated conversation.")
            abort(500, description="Failed to load updated conversation")

        if bool(updated.get("has_unread", False)):
            push_payload, push_event_key = ctx.conversation_notification_payload(
                profile=profile,
                conversation=updated,
                message=assistant_msg,
            )
            ctx.dispatch_web_push(str(profile.get("id", "")).strip(), push_payload, event_key=push_event_key)

        if _cancel_requested():
            job_status = "canceled"
            job_detail = "Message pipeline cancelled by user."
        elif pipeline_error:
            job_status = "completed_with_warnings"
            job_detail = "Message pipeline completed with non-blocking recovery after internal error."
        else:
            job_status = "completed"
            job_detail = "Message pipeline completed."
        ctx.job_manager.finish(profile, request_id, status=job_status, detail=job_detail)

        return {"conversation": updated, "assistant_message": assistant_msg, "request_id": request_id}, 200

    @bp.route("/api/conversations/<conversation_id>/image-tool/generate", methods=["POST"])
    def image_tool_generate(conversation_id: str) -> tuple[dict, int]:
        profile = ctx.require_profile()
        store = ctx.conversation_store_for(profile)
        convo = store.get(conversation_id)
        if convo is None:
            abort(404, description="Conversation not found")

        content_type = str(request.content_type or "").strip().lower()
        prompt = ""
        negative_prompt = ""
        image_style_in: str | None = None
        selected_loras_in: list[str] | None = None
        refine_prompt = True
        refine_prompt_explicit = False
        style_preset_id = ""
        steps_override: int | None = None
        cfg_override: float | None = None
        width_override: int | None = None
        height_override: int | None = None
        sampler_name_override = ""
        scheduler_override = ""
        lora_strength_model_override: float | None = None
        lora_strength_clip_override: float | None = None
        checkpoint_name_override: str = ""
        workflow_override: str = ""
        vae_name_override: str = ""
        model_family_override: str = ""
        bundled_loras: list[dict] = []
        scene_subject: str = ""
        request_id: str = ""
        attachments: list[dict[str, Any]] = []
        upload_errors: list[str] = []

        if content_type.startswith("multipart/form-data"):
            prompt = str(request.form.get("prompt", "")).strip()
            negative_prompt = str(request.form.get("negative_prompt", "")).strip()
            request_id = str(request.form.get("request_id", "")).strip()
            if "refine_prompt" in request.form:
                refine_prompt = _to_bool(request.form.get("refine_prompt"), default=True)
                refine_prompt_explicit = True
            if "steps" in request.form:
                raw_steps = str(request.form.get("steps", "")).strip()
                if raw_steps:
                    try:
                        steps_override = int(raw_steps)
                    except ValueError:
                        return {"error": "Steps must be an integer."}, 400
            if "cfg" in request.form:
                cfg_override = _to_float(request.form.get("cfg"), default=None)
            if "width" in request.form:
                width_override = _to_int(request.form.get("width"), default=None)
            if "height" in request.form:
                height_override = _to_int(request.form.get("height"), default=None)
            if "sampler_name" in request.form:
                sampler_name_override = str(request.form.get("sampler_name", "")).strip().lower()
            if "scheduler" in request.form:
                scheduler_override = str(request.form.get("scheduler", "")).strip().lower()
            if "lora_strength_model" in request.form:
                lora_strength_model_override = _to_float(request.form.get("lora_strength_model"), default=None)
            if "lora_strength_clip" in request.form:
                lora_strength_clip_override = _to_float(request.form.get("lora_strength_clip"), default=None)
            if "style_preset_id" in request.form:
                style_preset_id = str(request.form.get("style_preset_id", "")).strip().lower()
            if "image_style" in request.form:
                image_style_in = str(request.form.get("image_style", "")).strip().lower()
            if "selected_loras" in request.form:
                selected_loras_in = _parse_selected_loras_value(request.form.get("selected_loras"))
            if "scene_subject" in request.form:
                scene_subject = str(request.form.get("scene_subject", "")).strip().lower()
            if "model_family_override" in request.form:
                model_family_override = str(request.form.get("model_family_override", "")).strip().lower()
            attachments, upload_errors = ctx.save_uploaded_images(profile, conversation_id)
        else:
            payload = request.get_json(silent=True) or {}
            prompt = str(payload.get("prompt", "")).strip()
            negative_prompt = str(payload.get("negative_prompt", "")).strip()
            if "refine_prompt" in payload:
                refine_prompt = _to_bool(payload.get("refine_prompt"), default=True)
                refine_prompt_explicit = True
            if "steps" in payload and str(payload.get("steps", "")).strip():
                try:
                    steps_override = int(payload.get("steps"))
                except (TypeError, ValueError):
                    return {"error": "Steps must be an integer."}, 400
            if "cfg" in payload:
                cfg_override = _to_float(payload.get("cfg"), default=None)
            if "width" in payload:
                width_override = _to_int(payload.get("width"), default=None)
            if "height" in payload:
                height_override = _to_int(payload.get("height"), default=None)
            if "sampler_name" in payload:
                sampler_name_override = str(payload.get("sampler_name", "")).strip().lower()
            if "scheduler" in payload:
                scheduler_override = str(payload.get("scheduler", "")).strip().lower()
            if "lora_strength_model" in payload:
                lora_strength_model_override = _to_float(payload.get("lora_strength_model"), default=None)
            if "lora_strength_clip" in payload:
                lora_strength_clip_override = _to_float(payload.get("lora_strength_clip"), default=None)
            if "style_preset_id" in payload:
                style_preset_id = str(payload.get("style_preset_id", "")).strip().lower()
            if "image_style" in payload:
                image_style_in = str(payload.get("image_style", "")).strip().lower()
            if "selected_loras" in payload:
                selected_loras_in = _parse_selected_loras_value(payload.get("selected_loras"))
            if "scene_subject" in payload:
                scene_subject = str(payload.get("scene_subject", "")).strip().lower()
            if "model_family_override" in payload:
                model_family_override = str(payload.get("model_family_override", "")).strip().lower()
            request_id = str(payload.get("request_id", "")).strip()

        if not prompt:
            return {"error": "Prompt is required"}, 400
        if steps_override is not None and (steps_override < 4 or steps_override > 80):
            return {"error": "Steps must be between 4 and 80."}, 400
        if width_override is not None and (width_override < 256 or width_override > 2048):
            return {"error": "Width must be between 256 and 2048."}, 400
        if height_override is not None and (height_override < 256 or height_override > 2048):
            return {"error": "Height must be between 256 and 2048."}, 400

        request_id = ctx.job_manager.start(
            profile=profile,
            conversation_id=conversation_id,
            request_id=request_id,
            mode="command",
            user_text=prompt,
        )

        if image_style_in is not None or selected_loras_in is not None:
            updated_prefs = store.set_image_preferences(
                conversation_id,
                image_style=image_style_in,
                selected_loras=selected_loras_in,
            )
            if updated_prefs is not None:
                convo = updated_prefs

        image_style = str(convo.get("image_style", "realistic")).strip().lower() or "realistic"
        selected_loras = _normalize_lora_selection(convo.get("selected_loras", []))
        style_preset: dict[str, Any] | None = None
        preset_refiner_profile: dict[str, Any] = {}
        if style_preset_id:
            repo_root = ctx.repo_root_for_profile(profile)
            style_preset = find_image_gen_preset(repo_root, style_preset_id)
            if style_preset is None:
                return {"error": f"Unknown image preset '{style_preset_id}'."}, 400
            preset_defaults = style_preset.get("defaults", {}) if isinstance(style_preset.get("defaults"), dict) else {}
            preset_refiner_profile = style_preset.get("refiner_profile", {}) if isinstance(style_preset.get("refiner_profile"), dict) else {}
            if steps_override is None:
                steps_override = _to_int(preset_defaults.get("steps"), default=steps_override)
            if cfg_override is None:
                cfg_override = _to_float(preset_defaults.get("cfg"), default=cfg_override)
            if width_override is None:
                width_override = _to_int(preset_defaults.get("width"), default=width_override)
            if height_override is None:
                height_override = _to_int(preset_defaults.get("height"), default=height_override)
            if not sampler_name_override:
                sampler_name_override = str(preset_defaults.get("sampler_name", "")).strip().lower()
            if not scheduler_override:
                scheduler_override = str(preset_defaults.get("scheduler", "")).strip().lower()
            if lora_strength_model_override is None:
                lora_strength_model_override = _to_float(
                    preset_defaults.get("lora_strength_model"),
                    default=lora_strength_model_override,
                )
            if lora_strength_clip_override is None:
                lora_strength_clip_override = _to_float(
                    preset_defaults.get("lora_strength_clip"),
                    default=lora_strength_clip_override,
                )
            if not checkpoint_name_override:
                checkpoint_name_override = str(preset_defaults.get("checkpoint_name", "")).strip()
            if not workflow_override:
                workflow_override = str(preset_defaults.get("workflow", "")).strip()
            if not vae_name_override:
                vae_name_override = str(preset_defaults.get("vae_name", "")).strip()
            if not model_family_override:
                model_family_override = (
                    str(style_preset.get("model_family", "")).strip().lower()
                    or str(preset_defaults.get("model_family", "")).strip().lower()
                )
            if model_family_override == "sdxl":
                model_family_override = "xl"
            if not model_family_override:
                base_lane = str(preset_defaults.get("base_lane", "")).strip().lower()
                if base_lane in {"xl", "sdxl", "image_generation_xl"}:
                    model_family_override = "xl"
            if not model_family_override and workflow_override.strip().lower().startswith("sdxl"):
                model_family_override = "xl"
            if not bundled_loras:
                raw_bundled = style_preset.get("bundled_loras", [])
                if isinstance(raw_bundled, list):
                    bundled_loras = [b for b in raw_bundled if isinstance(b, dict) and b.get("name")]
            if not refine_prompt_explicit and "refine_prompt" in preset_defaults:
                refine_prompt = bool(preset_defaults.get("refine_prompt"))
            if not negative_prompt:
                negative_prompt = str(style_preset.get("default_negative_prompt", "")).strip()
            kind = str(style_preset.get("kind", "lora")).strip().lower() or "lora"
            if kind == "lora":
                cfg_sd15 = lane_model_config(repo_root, "image_generation_sd15")
                cfg_xl = lane_model_config(repo_root, "image_generation_xl")
                cfg_compose = lane_model_config(repo_root, "image_generation_compose")
                cfg_realistic = lane_model_config(repo_root, "image_generation")
                base_url = str(
                    cfg_sd15.get("base_url")
                    or cfg_xl.get("base_url")
                    or cfg_compose.get("base_url")
                    or cfg_realistic.get("base_url")
                    or "http://127.0.0.1:8188"
                ).strip()
                client = ComfyUIClient(base_url)
                if not client.is_available():
                    return {
                        "error": f"Preset '{style_preset_id}' requires ComfyUI at {base_url}.",
                    }, 400
                try:
                    available_loras = client.list_loras(timeout=20)
                except Exception as exc:
                    return {"error": f"Could not verify preset LoRA files from ComfyUI: {exc}"}, 400
                resolved_lora = resolve_preset_lora_name(style_preset, available_loras)
                candidates = [
                    str(x).strip()
                    for x in style_preset.get("lora_candidates", [])
                    if str(x).strip()
                ]
                if candidates and not resolved_lora:
                    expected = ", ".join(candidates)
                    return {
                        "error": (
                            f"Preset '{style_preset_id}' is missing LoRA files. "
                            f"Install one of [{expected}] in ComfyUI/models/loras and try again."
                        ),
                    }, 400
                image_style = "lora"
                selected_loras = [resolved_lora] if resolved_lora else []

        image_attachments = [a for a in attachments if str(a.get("type", "")).strip().lower() == "image"]
        ref_image_paths = [
            str(ctx.attachment_dir_for(profile, conversation_id) / str(a.get("filename", "")).strip())
            for a in image_attachments
            if str(a.get("filename", "")).strip()
        ]
        is_compose = bool(ref_image_paths)
        lane = "image_gen_compose" if is_compose else "image_gen"

        # Strip {imageN} reference tokens — UI affordances only, not for the diffusion model
        prompt_for_gen = _IMAGE_REF_TOKEN_RE.sub("", prompt).strip()

        base_negative = str(negative_prompt or "").strip()
        final_negative = (
            _refine_negative_prompt(
                base_negative,
                prompt=prompt_for_gen,
                preset_id=style_preset_id,
            )
            if refine_prompt
            else base_negative
        )
        final_positive = (
            _refine_image_prompt(
                prompt_for_gen,
                image_style=image_style,
                selected_loras=selected_loras,
                has_references=is_compose,
                preset_id=style_preset_id,
                refiner_profile=preset_refiner_profile,
                scene_subject=scene_subject,
            )
            if refine_prompt
            else prompt
        )

        convo_project = _normalize_project_slug(convo.get("project"))
        orch = ctx.new_orch(profile)
        if orch.project_slug != convo_project:
            orch.set_project(convo_project)

        attach_dir = ctx.attachment_dir_for(profile, conversation_id)
        attach_dir.mkdir(parents=True, exist_ok=True)
        ctx.job_manager.update(profile, request_id, stage="image_gen_started")
        image_gen_result = orch._run_registered_agent(
            lane,
            orch._make_agent_task(
                lane=lane,
                text=final_positive,
                context={
                    "positive_prompt": final_positive,
                    "negative_prompt": final_negative,
                    "steps": steps_override,
                    "cfg": cfg_override,
                    "width": width_override,
                    "height": height_override,
                    "sampler_name": sampler_name_override,
                    "scheduler": scheduler_override,
                    "lora_strength_model": lora_strength_model_override,
                    "lora_strength_clip": lora_strength_clip_override,
                    "conversation_id": conversation_id,
                    "attach_dir": str(attach_dir),
                    "ref_image_paths": ref_image_paths,
                    "image_style": image_style,
                    "selected_loras": selected_loras,
                    "style_preset_id": style_preset_id,
                    "checkpoint_name_override": checkpoint_name_override,
                    "workflow_override": workflow_override,
                    "vae_name_override": vae_name_override,
                    "model_family_override": model_family_override,
                    "scene_subject": scene_subject,
                    "bundled_loras": bundled_loras,
                },
            ),
        )
        if not image_gen_result.get("ok"):
            message = str(image_gen_result.get("message", "Image generation failed.")).strip() or "Image generation failed."
            ctx.job_manager.finish(profile, request_id, status="failed", detail=message)
            return {
                "ok": False,
                "error": str(image_gen_result.get("error", "")).strip(),
                "message": message,
                "upload_errors": upload_errors,
            }, 502

        gen_filename = str(image_gen_result.get("filename", "")).strip()
        gen_url = str(image_gen_result.get("url", "")).strip()
        gen_seed = int(image_gen_result.get("seed", 0))
        gen_steps = int(image_gen_result.get("steps", steps_override or 0))
        _gen_model_family = str(model_family_override or "").strip().lower()
        if not _gen_model_family and style_preset_id:
            repo_root_mf = ctx.repo_root_for_profile(profile)
            _preset_mf = find_image_gen_preset(repo_root_mf, style_preset_id)
            if _preset_mf:
                _gen_model_family = str((_preset_mf.get("model_family") or "")).strip().lower()
        gen_attachments = [{
            "id": f"gen_{gen_seed % 100000:05d}",
            "type": "image",
            "name": gen_filename or "generated.png",
            "filename": gen_filename,
            "mime": "image/png",
            "size": 0,
            "url": gen_url,
            "model_family": _gen_model_family,
        }]

        reply_lines = ["Image generated via Image Tool.", f"_Seed: {gen_seed} · Steps: {gen_steps}_"]
        if final_positive != prompt_for_gen:
            reply_lines.append("\nPrompt refined for generation quality.")
        if upload_errors:
            reply_lines.append("\nUpload notes:")
            reply_lines.extend([f"- {item}" for item in upload_errors[:6]])
        reply_text = "\n".join(reply_lines)

        assistant_msg = store.add_message(
            conversation_id,
            "assistant",
            reply_text,
            mode="command",
            foraging=False,
            request_id=request_id,
            attachments=gen_attachments,
            meta={
                "image_tool": True,
                "prompt_original": prompt,
                "prompt_final": final_positive,
                "negative_prompt_base": base_negative,
                "negative_prompt": final_negative,
                "image_style": image_style,
                "selected_loras": selected_loras,
                "compose_mode": is_compose,
                "steps": gen_steps,
                "cfg": cfg_override,
                "width": width_override,
                "height": height_override,
                "sampler_name": sampler_name_override,
                "scheduler": scheduler_override,
                "style_preset_id": style_preset_id,
            },
        )
        if assistant_msg is None:
            ctx.job_manager.finish(profile, request_id, status="failed", detail="Failed to persist assistant reply.")
            abort(500, description="Failed to persist assistant reply")

        ctx.job_manager.finish(profile, request_id, status="completed")
        updated = store.get(conversation_id)
        if updated is None:
            abort(500, description="Failed to load updated conversation")

        if bool(updated.get("has_unread", False)):
            push_payload, push_event_key = ctx.conversation_notification_payload(
                profile=profile,
                conversation=updated,
                message=assistant_msg,
            )
            ctx.dispatch_web_push(str(profile.get("id", "")).strip(), push_payload, event_key=push_event_key)

        return {
            "ok": True,
            "conversation": updated,
            "assistant_message": assistant_msg,
            "request_id": request_id,
            "prompt_original": prompt,
            "prompt_final": final_positive,
            "negative_prompt": final_negative,
            "image_style": image_style,
            "selected_loras": selected_loras,
            "compose_mode": is_compose,
            "steps": gen_steps,
            "style_preset_id": style_preset_id,
            "upload_errors": upload_errors,
        }, 200

    @bp.route("/api/conversations/<conversation_id>/image-tool/video-generate", methods=["POST"])
    def image_tool_video_generate(conversation_id: str) -> tuple[dict, int]:
        profile = ctx.require_profile()
        store = ctx.conversation_store_for(profile)
        convo = store.get(conversation_id)
        if convo is None:
            abort(404, description="Conversation not found")

        body = request.get_json(silent=True) or {}
        prompt = str(body.get("prompt", "")).strip()
        negative_prompt = str(body.get("negative_prompt", "")).strip()
        ref_image_filename = str(body.get("ref_image_filename", "")).strip()
        num_frames = int(body.get("num_frames", 81))
        request_id = str(body.get("request_id", "")).strip()

        if not prompt:
            return {"error": "Prompt is required."}, 400
        if not ref_image_filename:
            return {"error": "ref_image_filename is required."}, 400
        if "/" in ref_image_filename or "\\" in ref_image_filename:
            return {"error": "Invalid ref_image_filename."}, 400
        num_frames = max(17, min(201, num_frames))

        attach_dir = ctx.attachment_dir_for(profile, conversation_id)
        ref_image_path = attach_dir / ref_image_filename
        if not ref_image_path.exists() or not ref_image_path.is_file():
            return {"error": f"Reference image not found: {ref_image_filename}"}, 400

        request_id = ctx.job_manager.start(
            profile=profile,
            conversation_id=conversation_id,
            request_id=request_id,
            mode="command",
            user_text=prompt,
        )

        convo_project = _normalize_project_slug(convo.get("project"))
        orch = ctx.new_orch(profile)
        if orch.project_slug != convo_project:
            orch.set_project(convo_project)

        attach_dir.mkdir(parents=True, exist_ok=True)
        ctx.job_manager.update(profile, request_id, stage="video_gen_started")
        video_gen_result = orch._run_registered_agent(
            "video_gen",
            orch._make_agent_task(
                lane="video_gen",
                text=prompt,
                context={
                    "positive_prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "ref_image_path": str(ref_image_path),
                    "num_frames": num_frames,
                    "conversation_id": conversation_id,
                    "attach_dir": str(attach_dir),
                },
            ),
        )

        if not video_gen_result.get("ok"):
            message = str(video_gen_result.get("message", "Video generation failed.")).strip() or "Video generation failed."
            ctx.job_manager.finish(profile, request_id, status="failed", detail=message)
            return {
                "ok": False,
                "error": str(video_gen_result.get("error", "")).strip(),
                "message": message,
            }, 502

        gen_filename = str(video_gen_result.get("filename", "")).strip()
        gen_url = str(video_gen_result.get("url", "")).strip()
        gen_seed = int(video_gen_result.get("seed", 0))
        gen_frames = int(video_gen_result.get("num_frames", num_frames))
        gen_attachments = [{
            "id": f"vid_{gen_seed % 100000:05d}",
            "type": "video",
            "name": gen_filename or "generated.mp4",
            "filename": gen_filename,
            "mime": "video/mp4",
            "size": 0,
            "url": gen_url,
        }]

        reply_text = f"Video generated via Image Tool.\n_Seed: {gen_seed} · Frames: {gen_frames}_"

        assistant_msg = store.add_message(
            conversation_id,
            "assistant",
            reply_text,
            mode="command",
            foraging=False,
            request_id=request_id,
            attachments=gen_attachments,
            meta={
                "video_tool": True,
                "prompt": prompt,
                "num_frames": gen_frames,
            },
        )
        if assistant_msg is None:
            ctx.job_manager.finish(profile, request_id, status="failed", detail="Failed to persist assistant reply.")
            abort(500, description="Failed to persist assistant reply")

        ctx.job_manager.finish(profile, request_id, status="completed")
        updated = store.get(conversation_id)
        if updated is None:
            abort(500, description="Failed to load updated conversation")

        if bool(updated.get("has_unread", False)):
            push_payload, push_event_key = ctx.conversation_notification_payload(
                profile=profile,
                conversation=updated,
                message=assistant_msg,
            )
            ctx.dispatch_web_push(str(profile.get("id", "")).strip(), push_payload, event_key=push_event_key)

        return {
            "ok": True,
            "conversation": updated,
            "assistant_message": assistant_msg,
            "request_id": request_id,
        }, 200

    @bp.route("/api/conversations/<conversation_id>/image-tool/bg-enhance", methods=["POST"])
    def image_tool_bg_enhance(conversation_id: str) -> tuple[dict, int]:
        profile = ctx.require_profile()
        store = ctx.conversation_store_for(profile)
        convo = store.get(conversation_id)
        if convo is None:
            abort(404, description="Conversation not found")

        body = request.get_json(silent=True) or {}
        source_filename = str(body.get("source_filename", "")).strip()
        request_id = str(body.get("request_id", "")).strip()

        if not source_filename:
            return {"error": "source_filename is required."}, 400
        if "/" in source_filename or "\\" in source_filename:
            return {"error": "Invalid source_filename."}, 400

        attach_dir = ctx.attachment_dir_for(profile, conversation_id)
        source_path = attach_dir / source_filename
        if not source_path.exists() or not source_path.is_file():
            return {"error": f"Source image not found: {source_filename}"}, 400

        request_id = ctx.job_manager.start(
            profile=profile,
            conversation_id=conversation_id,
            request_id=request_id,
            mode="command",
            user_text="BG+ enhance",
        )

        convo_project = _normalize_project_slug(convo.get("project"))
        orch = ctx.new_orch(profile)
        if orch.project_slug != convo_project:
            orch.set_project(convo_project)

        attach_dir.mkdir(parents=True, exist_ok=True)
        ctx.job_manager.update(profile, request_id, stage="enhance_started")
        result = orch._run_registered_agent(
            "image_enhance",
            orch._make_agent_task(
                lane="image_enhance",
                text="BG+ enhance",
                context={
                    "ref_image_path": str(source_path),
                    "conversation_id": conversation_id,
                    "attach_dir": str(attach_dir),
                },
            ),
        )

        if not result.get("ok"):
            message = str(result.get("message", "Enhancement failed.")).strip() or "Enhancement failed."
            ctx.job_manager.finish(profile, request_id, status="failed", detail=message)
            return {"ok": False, "error": str(result.get("error", "")).strip(), "message": message}, 502

        gen_filename = str(result.get("filename", "")).strip()
        gen_url = str(result.get("url", "")).strip()
        gen_seed = int(result.get("seed", 0))
        gen_attachments = [{
            "id": f"gen_{gen_seed % 100000:05d}",
            "type": "image",
            "name": gen_filename or "enhanced.png",
            "filename": gen_filename,
            "mime": "image/png",
            "size": 0,
            "url": gen_url,
            "model_family": "xl",
        }]

        assistant_msg = store.add_message(
            conversation_id,
            "assistant",
            f"BG+ enhancement complete.\n_Seed: {gen_seed}_",
            mode="command",
            foraging=False,
            request_id=request_id,
            attachments=gen_attachments,
            meta={"image_tool": True, "bg_enhance": True},
        )
        if assistant_msg is None:
            ctx.job_manager.finish(profile, request_id, status="failed", detail="Failed to persist assistant reply.")
            abort(500, description="Failed to persist assistant reply")

        ctx.job_manager.finish(profile, request_id, status="completed")
        updated = store.get(conversation_id)
        if updated is None:
            abort(500, description="Failed to load updated conversation")

        if bool(updated.get("has_unread", False)):
            push_payload, push_event_key = ctx.conversation_notification_payload(
                profile=profile,
                conversation=updated,
                message=assistant_msg,
            )
            ctx.dispatch_web_push(str(profile.get("id", "")).strip(), push_payload, event_key=push_event_key)

        return {"ok": True, "conversation": updated, "assistant_message": assistant_msg, "request_id": request_id}, 200
