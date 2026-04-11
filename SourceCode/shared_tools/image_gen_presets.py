from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _catalog_path(repo_root: Path) -> Path:
    return Path(repo_root) / "SourceCode" / "configs" / "image_gen_presets.json"


def _normalize_defaults(raw: Any) -> dict[str, Any]:
    data = raw if isinstance(raw, dict) else {}
    out: dict[str, Any] = {}
    for key in ("checkpoint_name", "workflow", "vae_name", "base_lane", "model_family"):
        text = str(data.get(key, "")).strip()
        if text:
            out[key] = text
    if "steps" in data:
        try:
            out["steps"] = int(data.get("steps"))
        except (TypeError, ValueError):
            pass
    for key in ("width", "height"):
        if key in data:
            try:
                out[key] = int(data.get(key))
            except (TypeError, ValueError):
                pass
    for key in ("cfg", "lora_strength_model", "lora_strength_clip"):
        if key in data:
            try:
                out[key] = float(data.get(key))
            except (TypeError, ValueError):
                pass
    sampler = str(data.get("sampler_name", "")).strip()
    if sampler:
        out["sampler_name"] = sampler
    scheduler = str(data.get("scheduler", "")).strip()
    if scheduler:
        out["scheduler"] = scheduler
    if "refine_prompt" in data:
        out["refine_prompt"] = bool(data.get("refine_prompt"))
    return out


def _normalize_candidates(raw: Any) -> list[str]:
    values = raw if isinstance(raw, list) else []
    out: list[str] = []
    seen: set[str] = set()
    for item in values:
        text = str(item or "").strip()
        if not text:
            continue
        low = text.lower()
        if low in seen:
            continue
        seen.add(low)
        out.append(text)
    return out


def _normalize_refiner_profile(raw: Any) -> dict[str, Any]:
    data = raw if isinstance(raw, dict) else {}
    out: dict[str, Any] = {}
    name = str(data.get("name", "")).strip()
    if name:
        out["name"] = name
    night_terms = _normalize_candidates(data.get("night_terms", []))
    if night_terms:
        out["night_terms"] = night_terms
    return out


def _normalize_bundled_loras(raw: Any) -> list[dict[str, Any]]:
    values = raw if isinstance(raw, list) else []
    out: list[dict[str, Any]] = []
    for item in values:
        row = item if isinstance(item, dict) else {}
        name = str(row.get("name", "")).strip()
        if not name:
            continue
        entry: dict[str, Any] = {"name": name}
        if "strength_model" in row:
            try:
                entry["strength_model"] = float(row.get("strength_model"))
            except (TypeError, ValueError):
                pass
        if "strength_clip" in row:
            try:
                entry["strength_clip"] = float(row.get("strength_clip"))
            except (TypeError, ValueError):
                pass
        out.append(entry)
    return out


def _normalize_preset(raw: Any) -> dict[str, Any] | None:
    row = raw if isinstance(raw, dict) else {}
    preset_id = str(row.get("id", "")).strip().lower()
    label = str(row.get("label", "")).strip()
    kind = str(row.get("kind", "")).strip().lower()
    if not preset_id or not label:
        return None
    if kind not in {"lora", "realistic"}:
        kind = "lora"
    model_family = str(row.get("model_family", "")).strip().lower()
    if model_family in {"sdxl"}:
        model_family = "xl"
    if model_family not in {"", "sd15", "xl", "xl_standard"}:
        model_family = ""
    normalized = {
        "id": preset_id,
        "label": label,
        "kind": kind,
        "model_family": model_family,
        "lora_candidates": _normalize_candidates(row.get("lora_candidates", [])),
        "bundled_loras": _normalize_bundled_loras(row.get("bundled_loras", [])),
        "defaults": _normalize_defaults(row.get("defaults", {})),
        "default_negative_prompt": str(row.get("default_negative_prompt", "")).strip(),
        "refiner_profile": _normalize_refiner_profile(row.get("refiner_profile", {})),
    }
    return normalized


def load_image_gen_presets(repo_root: Path) -> list[dict[str, Any]]:
    path = _catalog_path(repo_root)
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return []
    raw_rows = payload.get("presets", []) if isinstance(payload, dict) else []
    rows = raw_rows if isinstance(raw_rows, list) else []
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in rows:
        normalized = _normalize_preset(item)
        if not normalized:
            continue
        preset_id = str(normalized["id"])
        if preset_id in seen:
            continue
        seen.add(preset_id)
        out.append(normalized)
    return out


def find_image_gen_preset(repo_root: Path, preset_id: str) -> dict[str, Any] | None:
    key = str(preset_id or "").strip().lower()
    if not key:
        return None
    for row in load_image_gen_presets(repo_root):
        if str(row.get("id", "")).strip().lower() == key:
            return row
    return None


def resolve_preset_lora_name(preset: dict[str, Any], available_loras: list[str]) -> str:
    candidates = _normalize_candidates(preset.get("lora_candidates", []))
    if not candidates:
        return ""
    lookup: dict[str, str] = {}
    for item in available_loras or []:
        text = str(item or "").strip()
        if text:
            lookup[text.lower()] = text
    for candidate in candidates:
        hit = lookup.get(candidate.lower())
        if hit:
            return hit
    return ""
