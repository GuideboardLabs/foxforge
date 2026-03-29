from __future__ import annotations

from typing import TYPE_CHECKING, Any

from flask import Blueprint, request

if TYPE_CHECKING:
    from web_gui.app_context import AppContext


def _pm_find_index(rows: Any, key_field: str, raw_value: str) -> int:
    target = str(raw_value or "").strip().lower()
    if not target or not isinstance(rows, list):
        return -1
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        if str(row.get(key_field, "") or "").strip().lower() == target:
            return idx
    return -1


def create_personal_memory_blueprint(ctx: AppContext) -> Blueprint:
    bp = Blueprint('personal_memory_routes', __name__)

    @bp.route('/api/personal-memory', methods=['GET'])
    def personal_memory_get() -> tuple[dict, int]:
        from shared_tools.personal_memory import PersonalMemory
        profile = ctx.require_profile()
        pm = PersonalMemory(ctx.repo_root_for_profile(profile))
        return {"context": pm.load(), "records": pm.list_records()}, 200

    @bp.route('/api/second-brain', methods=['GET'])
    def second_brain_get() -> tuple[dict, int]:
        from shared_tools.personal_memory import PersonalMemory
        profile = ctx.require_profile()
        pm = PersonalMemory(ctx.repo_root_for_profile(profile))
        return pm.second_brain_payload(), 200

    @bp.route('/api/personal-memory/records', methods=['GET'])
    def personal_memory_records_get() -> tuple[dict, int]:
        from shared_tools.personal_memory import PersonalMemory
        profile = ctx.require_profile()
        pm = PersonalMemory(ctx.repo_root_for_profile(profile))
        return {"records": pm.list_records()}, 200

    @bp.route('/api/personal-memory/records/explain', methods=['POST'])
    def personal_memory_record_explain() -> tuple[dict, int]:
        from shared_tools.personal_memory import PersonalMemory
        profile = ctx.require_profile()
        body = request.get_json(silent=True) or {}
        record_id = str(body.get("id", "")).strip()
        if not record_id:
            return {"error": "id is required"}, 400
        pm = PersonalMemory(ctx.repo_root_for_profile(profile))
        explanation = pm.explain_record(record_id)
        if explanation is None:
            return {"error": "record not found"}, 404
        return {"record": explanation}, 200

    @bp.route('/api/personal-memory/records/<record_id>/confirm', methods=['PATCH'])
    def personal_memory_record_confirm(record_id: str) -> tuple[dict, int]:
        from shared_tools.personal_memory import PersonalMemory
        profile = ctx.require_profile()
        pm = PersonalMemory(ctx.repo_root_for_profile(profile))
        record = pm.confirm_record(record_id)
        if record is None:
            return {"error": "record not found"}, 404
        return {"ok": True, "record": record}, 200

    @bp.route('/api/personal-memory/records/<record_id>/pin', methods=['PATCH'])
    def personal_memory_record_pin(record_id: str) -> tuple[dict, int]:
        from shared_tools.personal_memory import PersonalMemory
        profile = ctx.require_profile()
        pm = PersonalMemory(ctx.repo_root_for_profile(profile))
        record = pm.pin_record(record_id)
        if record is None:
            return {"error": "record not found"}, 404
        return {"ok": True, "record": record}, 200

    @bp.route('/api/personal-memory/records/<record_id>/forget', methods=['PATCH'])
    def personal_memory_record_forget(record_id: str) -> tuple[dict, int]:
        from shared_tools.personal_memory import PersonalMemory
        profile = ctx.require_profile()
        pm = PersonalMemory(ctx.repo_root_for_profile(profile))
        record = pm.forget_record(record_id)
        if record is None:
            return {"error": "record not found"}, 404
        return {"ok": True, "record": record}, 200

    @bp.route('/api/personal-memory/records/<record_id>', methods=['PATCH'])
    def personal_memory_record_update(record_id: str) -> tuple[dict, int]:
        from shared_tools.personal_memory import PersonalMemory
        profile = ctx.require_profile()
        body = request.get_json(silent=True) or {}
        pm = PersonalMemory(ctx.repo_root_for_profile(profile))
        record = pm.update_record(
            record_id,
            value=body.get("value"),
            status=body.get("status"),
            confidence=body.get("confidence"),
            evidence=body.get("evidence"),
            tags=body.get("tags"),
        )
        if record is None:
            return {"error": "record not found"}, 404
        return {"ok": True, "record": record}, 200

    @bp.route('/api/personal-memory/profile', methods=['PATCH'])
    def personal_memory_profile() -> tuple[dict, int]:
        from shared_tools.personal_memory import PersonalMemory
        profile = ctx.require_profile()
        body = request.get_json(silent=True) or {}
        pm = PersonalMemory(ctx.repo_root_for_profile(profile))
        data = pm.load()
        data["user_profile"] = {
            "preferred_name": str(body.get("preferred_name", "")).strip(),
            "full_name": str(body.get("full_name", "")).strip(),
            "gender": str(body.get("gender", body.get("pronouns", ""))).strip(),
            "birthday": str(body.get("birthday", "")).strip(),
            "location": str(body.get("location", "")).strip(),
            "work": str(body.get("work", "")).strip(),
            "likes": str(body.get("likes", "")).strip(),
            "dislikes": str(body.get("dislikes", "")).strip(),
            "notes": str(body.get("notes", "")).strip(),
        }
        pm.save(data)
        return {"ok": True}, 200

    @bp.route('/api/personal-memory/notes', methods=['PATCH'])
    def personal_memory_notes() -> tuple[dict, int]:
        from shared_tools.personal_memory import PersonalMemory
        profile = ctx.require_profile()
        body = request.get_json(silent=True) or {}
        pm = PersonalMemory(ctx.repo_root_for_profile(profile))
        data = pm.load()
        data["household_notes"] = str(body.get("notes", "")).strip()
        pm.save(data)
        return {"ok": True}, 200

    @bp.route('/api/personal-memory/family', methods=['POST'])
    def personal_memory_add_family() -> tuple[dict, int]:
        from shared_tools.personal_memory import PersonalMemory
        profile = ctx.require_profile()
        body = request.get_json(silent=True) or {}
        name = str(body.get("name", "")).strip()
        if not name:
            return {"error": "name is required"}, 400
        pm = PersonalMemory(ctx.repo_root_for_profile(profile))
        data = pm.load()
        data["family_members"].append({
            "name": name,
            "relationship": str(body.get("relationship", "")).strip(),
            "notes": str(body.get("notes", "")).strip(),
            "nickname": str(body.get("nickname", "")).strip(),
            "age": str(body.get("age", "")).strip(),
            "birthday": str(body.get("birthday", "")).strip(),
            "gender": str(body.get("gender", body.get("pronouns", ""))).strip(),
            "school_or_work": str(body.get("school_or_work", "")).strip(),
            "likes": str(body.get("likes", "")).strip(),
            "dislikes": str(body.get("dislikes", "")).strip(),
            "important_dates": str(body.get("important_dates", "")).strip(),
            "medical_notes": str(body.get("medical_notes", "")).strip(),
        })
        pm.save(data)
        return {"ok": True}, 201

    @bp.route('/api/personal-memory/family/<name>', methods=['PATCH'])
    def personal_memory_update_family(name: str) -> tuple[dict, int]:
        from shared_tools.personal_memory import PersonalMemory
        profile = ctx.require_profile()
        body = request.get_json(silent=True) or {}
        pm = PersonalMemory(ctx.repo_root_for_profile(profile))
        data = pm.load()
        idx = _pm_find_index(data.get("family_members", []), "name", name)
        if idx < 0:
            return {"error": "family member not found"}, 404
        next_name = str(body.get("name", "")).strip()
        if not next_name:
            return {"error": "name is required"}, 400
        data["family_members"][idx] = {
            "name": next_name,
            "relationship": str(body.get("relationship", "")).strip(),
            "notes": str(body.get("notes", "")).strip(),
            "nickname": str(body.get("nickname", "")).strip(),
            "age": str(body.get("age", "")).strip(),
            "birthday": str(body.get("birthday", "")).strip(),
            "gender": str(body.get("gender", body.get("pronouns", ""))).strip(),
            "school_or_work": str(body.get("school_or_work", "")).strip(),
            "likes": str(body.get("likes", "")).strip(),
            "dislikes": str(body.get("dislikes", "")).strip(),
            "important_dates": str(body.get("important_dates", "")).strip(),
            "medical_notes": str(body.get("medical_notes", "")).strip(),
        }
        pm.save(data)
        return {"ok": True}, 200

    @bp.route('/api/personal-memory/family/<name>', methods=['DELETE'])
    def personal_memory_remove_family(name: str) -> tuple[dict, int]:
        from shared_tools.personal_memory import PersonalMemory
        profile = ctx.require_profile()
        pm = PersonalMemory(ctx.repo_root_for_profile(profile))
        data = pm.load()
        before = len(data["family_members"])
        data["family_members"] = [m for m in data["family_members"] if m.get("name", "").lower() != name.lower()]
        pm.save(data)
        return {"ok": len(data["family_members"]) < before}, 200

    @bp.route('/api/personal-memory/pets', methods=['POST'])
    def personal_memory_add_pet() -> tuple[dict, int]:
        from shared_tools.personal_memory import PersonalMemory
        profile = ctx.require_profile()
        body = request.get_json(silent=True) or {}
        name = str(body.get("name", "")).strip()
        if not name:
            return {"error": "name is required"}, 400
        pm = PersonalMemory(ctx.repo_root_for_profile(profile))
        data = pm.load()
        data["pets"].append({
            "name": name,
            "species": str(body.get("species", "")).strip(),
            "notes": str(body.get("notes", "")).strip(),
            "breed": str(body.get("breed", "")).strip(),
            "sex": str(body.get("sex", "")).strip(),
            "age": str(body.get("age", "")).strip(),
            "birthday": str(body.get("birthday", "")).strip(),
            "weight": str(body.get("weight", "")).strip(),
            "food": str(body.get("food", "")).strip(),
            "medications": str(body.get("medications", "")).strip(),
            "vet": str(body.get("vet", "")).strip(),
            "microchip": str(body.get("microchip", "")).strip(),
            "behavior_notes": str(body.get("behavior_notes", "")).strip(),
        })
        pm.save(data)
        return {"ok": True}, 201

    @bp.route('/api/personal-memory/pets/<name>', methods=['PATCH'])
    def personal_memory_update_pet(name: str) -> tuple[dict, int]:
        from shared_tools.personal_memory import PersonalMemory
        profile = ctx.require_profile()
        body = request.get_json(silent=True) or {}
        pm = PersonalMemory(ctx.repo_root_for_profile(profile))
        data = pm.load()
        idx = _pm_find_index(data.get("pets", []), "name", name)
        if idx < 0:
            return {"error": "pet not found"}, 404
        next_name = str(body.get("name", "")).strip()
        if not next_name:
            return {"error": "name is required"}, 400
        data["pets"][idx] = {
            "name": next_name,
            "species": str(body.get("species", "")).strip(),
            "notes": str(body.get("notes", "")).strip(),
            "breed": str(body.get("breed", "")).strip(),
            "sex": str(body.get("sex", "")).strip(),
            "age": str(body.get("age", "")).strip(),
            "birthday": str(body.get("birthday", "")).strip(),
            "weight": str(body.get("weight", "")).strip(),
            "food": str(body.get("food", "")).strip(),
            "medications": str(body.get("medications", "")).strip(),
            "vet": str(body.get("vet", "")).strip(),
            "microchip": str(body.get("microchip", "")).strip(),
            "behavior_notes": str(body.get("behavior_notes", "")).strip(),
        }
        pm.save(data)
        return {"ok": True}, 200

    @bp.route('/api/personal-memory/pets/<name>', methods=['DELETE'])
    def personal_memory_remove_pet(name: str) -> tuple[dict, int]:
        from shared_tools.personal_memory import PersonalMemory
        profile = ctx.require_profile()
        pm = PersonalMemory(ctx.repo_root_for_profile(profile))
        data = pm.load()
        before = len(data["pets"])
        data["pets"] = [p for p in data["pets"] if p.get("name", "").lower() != name.lower()]
        pm.save(data)
        return {"ok": len(data["pets"]) < before}, 200

    @bp.route('/api/personal-memory/reminders', methods=['POST'])
    def personal_memory_add_reminder() -> tuple[dict, int]:
        from shared_tools.personal_memory import PersonalMemory
        profile = ctx.require_profile()
        body = request.get_json(silent=True) or {}
        label = str(body.get("label", "")).strip()
        if not label:
            return {"error": "label is required"}, 400
        pm = PersonalMemory(ctx.repo_root_for_profile(profile))
        data = pm.load()
        data["recurring_reminders"].append({
            "label": label,
            "frequency": str(body.get("frequency", "")).strip(),
            "time": str(body.get("time", "")).strip(),
            "notes": str(body.get("notes", "")).strip(),
            "person": str(body.get("person", "")).strip(),
            "start_date": str(body.get("start_date", "")).strip(),
            "end_date": str(body.get("end_date", "")).strip(),
            "location": str(body.get("location", "")).strip(),
            "channel": str(body.get("channel", "")).strip(),
            "priority": str(body.get("priority", "")).strip(),
        })
        pm.save(data)
        return {"ok": True}, 201

    @bp.route('/api/personal-memory/reminders/<label>', methods=['PATCH'])
    def personal_memory_update_reminder(label: str) -> tuple[dict, int]:
        from shared_tools.personal_memory import PersonalMemory
        profile = ctx.require_profile()
        body = request.get_json(silent=True) or {}
        pm = PersonalMemory(ctx.repo_root_for_profile(profile))
        data = pm.load()
        idx = _pm_find_index(data.get("recurring_reminders", []), "label", label)
        if idx < 0:
            return {"error": "reminder not found"}, 404
        next_label = str(body.get("label", "")).strip()
        if not next_label:
            return {"error": "label is required"}, 400
        data["recurring_reminders"][idx] = {
            "label": next_label,
            "frequency": str(body.get("frequency", "")).strip(),
            "time": str(body.get("time", "")).strip(),
            "notes": str(body.get("notes", "")).strip(),
            "person": str(body.get("person", "")).strip(),
            "start_date": str(body.get("start_date", "")).strip(),
            "end_date": str(body.get("end_date", "")).strip(),
            "location": str(body.get("location", "")).strip(),
            "channel": str(body.get("channel", "")).strip(),
            "priority": str(body.get("priority", "")).strip(),
        }
        pm.save(data)
        return {"ok": True}, 200

    @bp.route('/api/personal-memory/reminders/<label>', methods=['DELETE'])
    def personal_memory_remove_reminder(label: str) -> tuple[dict, int]:
        from shared_tools.personal_memory import PersonalMemory
        profile = ctx.require_profile()
        pm = PersonalMemory(ctx.repo_root_for_profile(profile))
        data = pm.load()
        before = len(data["recurring_reminders"])
        data["recurring_reminders"] = [
            r for r in data["recurring_reminders"] if r.get("label", "").lower() != label.lower()
        ]
        pm.save(data)
        return {"ok": len(data["recurring_reminders"]) < before}, 200

    return bp
