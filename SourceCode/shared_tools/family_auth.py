from __future__ import annotations

import hashlib
import hmac
import secrets
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from shared_tools.db import connect, row_to_dict, transaction
from shared_tools.migrations import initialize_database


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean_username(value: str) -> str:
    text = "".join(ch for ch in str(value or "").strip().lower() if ch.isalnum() or ch in {"_", "-"})
    return text[:32]


def _clean_role(value: str) -> str:
    role = str(value or "").strip().lower()
    if role in {"child", "kid"}:
        return "child"
    return "adult"


def _clean_color(value: str, default: str = "#4285f4") -> str:
    text = str(value or "").strip()
    if len(text) == 7 and text.startswith("#"):
        hex_part = text[1:]
        if all(ch in "0123456789abcdefABCDEF" for ch in hex_part):
            return text.lower()
    return default


def _clean_pin(value: str) -> str:
    text = str(value or "").strip()
    if len(text) == 4 and text.isdigit():
        return text
    return ""


class FamilyAuthStore:
    DEFAULT_COLORS = [
        "#4285f4",
        "#0f9d58",
        "#db4437",
        "#f4b400",
        "#ab47bc",
        "#5f6368",
        "#00acc1",
    ]

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = Path(repo_root)
        self.lock = Lock()
        initialize_database(self.repo_root)

    @staticmethod
    def _hash_password(password: str, salt_hex: str | None = None) -> dict[str, str]:
        salt = bytes.fromhex(salt_hex) if salt_hex else secrets.token_bytes(16)
        derived = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 210_000)
        return {"salt": salt.hex(), "hash": derived.hex()}

    @staticmethod
    def _verify_password(password: str, password_payload: dict[str, Any]) -> bool:
        salt_hex = str(password_payload.get("salt", "")).strip()
        expected_hex = str(password_payload.get("hash", "")).strip()
        if not salt_hex or not expected_hex:
            return False
        try:
            derived = FamilyAuthStore._hash_password(password, salt_hex=salt_hex)["hash"]
        except ValueError:
            return False
        return hmac.compare_digest(derived, expected_hex)

    @staticmethod
    def _public_profile(row: dict[str, Any] | None) -> dict[str, Any] | None:
        if not row:
            return None
        return {
            "id": str(row.get("id", "")).strip(),
            "username": str(row.get("username", "")).strip(),
            "display_name": str(row.get("display_name", "")).strip() or str(row.get("username", "")).strip(),
            "role": _clean_role(str(row.get("role", "adult"))),
            "color": _clean_color(str(row.get("color", "")), default="#4285f4"),
            "is_owner": bool(row.get("is_owner", False)),
        }

    def _count_users(self) -> int:
        with connect(self.repo_root) as conn:
            row = conn.execute("SELECT COUNT(*) AS count FROM users;").fetchone()
            return int(row["count"]) if row else 0

    def _pick_default_color(self, conn, preferred: str | None = None) -> str:
        if preferred:
            cleaned = _clean_color(preferred, default="")
            if cleaned:
                return cleaned
        rows = conn.execute("SELECT color FROM users WHERE active = 1;").fetchall()
        used_colors = {_clean_color(str(row["color"]), default="") for row in rows}
        for candidate in self.DEFAULT_COLORS:
            if candidate not in used_colors:
                return candidate
        return self.DEFAULT_COLORS[0]

    def ensure_owner(self, owner_password: str, owner_username: str = "owner") -> dict[str, Any]:
        clean_username = _clean_username(owner_username) or "owner"
        password = str(owner_password or "").strip()
        with self.lock, connect(self.repo_root) as conn:
            existing = conn.execute(
                "SELECT * FROM users WHERE is_owner = 1 ORDER BY created_at ASC LIMIT 1;"
            ).fetchone()
            if existing is not None:
                return self._public_profile(row_to_dict(existing)) or {}

            if self._count_users() > 0:
                raise ValueError("Owner account missing but user records already exist.")
            if not password:
                raise ValueError("Owner password required on first boot.")

            password_payload = self._hash_password(password)
            now = _now_iso()
            row = {
                "id": f"u_{secrets.token_hex(4)}",
                "username": clean_username,
                "display_name": "Owner",
                "role": "adult",
                "color": self.DEFAULT_COLORS[0],
                "password_hash": password_payload["hash"],
                "password_salt": password_payload["salt"],
                "created_at": now,
                "updated_at": now,
                "active": 1,
                "is_owner": 1,
            }
            with transaction(conn, immediate=True):
                conn.execute(
                    """
                    INSERT INTO users(
                        id, username, display_name, role, color,
                        password_hash, password_salt,
                        created_at, updated_at, active, is_owner
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """.strip(),
                    (
                        row["id"], row["username"], row["display_name"], row["role"], row["color"],
                        row["password_hash"], row["password_salt"],
                        row["created_at"], row["updated_at"], row["active"], row["is_owner"],
                    ),
                )
            return self._public_profile(row) or {}

    def list_profiles(self) -> list[dict[str, Any]]:
        with self.lock, connect(self.repo_root) as conn:
            rows = conn.execute(
                """
                SELECT id, username, display_name, role, color, is_owner
                FROM users
                WHERE active = 1
                ORDER BY is_owner DESC, LOWER(display_name) ASC, LOWER(username) ASC;
                """.strip()
            ).fetchall()
            return [self._public_profile(row_to_dict(row)) or {} for row in rows]

    def get_profile_by_id(self, user_id: str) -> dict[str, Any] | None:
        uid = str(user_id or "").strip()
        if not uid:
            return None
        with self.lock, connect(self.repo_root) as conn:
            row = conn.execute(
                """
                SELECT id, username, display_name, role, color, is_owner
                FROM users
                WHERE id = ? AND active = 1
                LIMIT 1;
                """.strip(),
                (uid,),
            ).fetchone()
            return self._public_profile(row_to_dict(row)) if row else None

    def verify_login(self, username: str, password: str) -> dict[str, Any] | None:
        clean_username = _clean_username(username)
        if not clean_username:
            return None
        with self.lock, connect(self.repo_root) as conn:
            row = conn.execute(
                """
                SELECT id, username, display_name, role, color, is_owner, password_hash, password_salt
                FROM users
                WHERE username = ? AND active = 1
                LIMIT 1;
                """.strip(),
                (clean_username,),
            ).fetchone()
            target = row_to_dict(row)
            if not target:
                return None
            password_payload = {
                "hash": str(target.get("password_hash", "")),
                "salt": str(target.get("password_salt", "")),
            }
            if not self._verify_password(str(password or ""), password_payload):
                return None
            return self._public_profile(target)

    def create_profile(
        self,
        *,
        username: str,
        pin: str = "",
        password: str = "",
        display_name: str,
        role: str,
        color: str,
    ) -> dict[str, Any]:
        clean_username = _clean_username(username)
        if not clean_username:
            raise ValueError("Username is required.")
        clean_pin = _clean_pin(pin) or _clean_pin(password)
        if not clean_pin:
            raise ValueError("PIN must be exactly 4 digits.")
        clean_role = _clean_role(role)
        clean_display = str(display_name or "").strip() or clean_username

        with self.lock, connect(self.repo_root) as conn:
            existing = conn.execute(
                "SELECT id FROM users WHERE username = ? AND active = 1 LIMIT 1;",
                (clean_username,),
            ).fetchone()
            if existing is not None:
                raise ValueError("Username already exists.")

            clean_color = self._pick_default_color(conn, color if str(color or "").strip() else None)
            password_payload = self._hash_password(clean_pin)
            now = _now_iso()
            row = {
                "id": f"u_{secrets.token_hex(4)}",
                "username": clean_username,
                "display_name": clean_display[:48],
                "role": clean_role,
                "color": clean_color,
                "password_hash": password_payload["hash"],
                "password_salt": password_payload["salt"],
                "created_at": now,
                "updated_at": now,
                "active": 1,
                "is_owner": 0,
            }
            with transaction(conn, immediate=True):
                conn.execute(
                    """
                    INSERT INTO users(
                        id, username, display_name, role, color,
                        password_hash, password_salt,
                        created_at, updated_at, active, is_owner
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """.strip(),
                    (
                        row["id"], row["username"], row["display_name"], row["role"], row["color"],
                        row["password_hash"], row["password_salt"],
                        row["created_at"], row["updated_at"], row["active"], row["is_owner"],
                    ),
                )
            return self._public_profile(row) or {}

    def update_profile(
        self,
        *,
        user_id: str,
        username: str | None = None,
        display_name: str | None = None,
        role: str | None = None,
        color: str | None = None,
        pin: str | None = None,
    ) -> dict[str, Any]:
        uid = str(user_id or "").strip()
        if not uid:
            raise ValueError("Profile id is required.")
        with self.lock, connect(self.repo_root) as conn:
            row = conn.execute("SELECT * FROM users WHERE id = ? AND active = 1 LIMIT 1;", (uid,)).fetchone()
            target = row_to_dict(row)
            if not target:
                raise ValueError("Profile not found.")

            clean_username = target["username"]
            clean_display = str(target.get("display_name", "")).strip() or clean_username
            clean_role = _clean_role(str(target.get("role", "adult")))
            clean_color = _clean_color(str(target.get("color", "")), default="#4285f4")
            password_hash = str(target.get("password_hash", "")).strip()
            password_salt = str(target.get("password_salt", "")).strip()

            if username is not None:
                candidate_username = _clean_username(username)
                if not candidate_username:
                    raise ValueError("Username is required.")
                existing = conn.execute(
                    "SELECT id FROM users WHERE username = ? AND id != ? AND active = 1 LIMIT 1;",
                    (candidate_username, uid),
                ).fetchone()
                if existing is not None:
                    raise ValueError("Username already exists.")
                clean_username = candidate_username

            if display_name is not None:
                clean_display = str(display_name or "").strip() or clean_username
                clean_display = clean_display[:48]

            if role is not None:
                clean_role = _clean_role(role)

            if color is not None:
                clean_color = _clean_color(color, default=clean_color)

            if pin is not None and str(pin).strip() != "":
                clean_pin = _clean_pin(pin)
                if not clean_pin:
                    raise ValueError("PIN must be exactly 4 digits.")
                password_payload = self._hash_password(clean_pin)
                password_hash = password_payload["hash"]
                password_salt = password_payload["salt"]

            now = _now_iso()
            with transaction(conn, immediate=True):
                conn.execute(
                    """
                    UPDATE users
                    SET username = ?,
                        display_name = ?,
                        role = ?,
                        color = ?,
                        password_hash = ?,
                        password_salt = ?,
                        updated_at = ?
                    WHERE id = ?;
                    """.strip(),
                    (clean_username, clean_display, clean_role, clean_color, password_hash, password_salt, now, uid),
                )

            updated = dict(target)
            updated.update(
                {
                    "username": clean_username,
                    "display_name": clean_display,
                    "role": clean_role,
                    "color": clean_color,
                    "password_hash": password_hash,
                    "password_salt": password_salt,
                    "updated_at": now,
                }
            )
            return self._public_profile(updated) or {}
