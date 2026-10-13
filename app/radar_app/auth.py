"""Login berbasis role. Password disimpan sebagai hash PBKDF2-SHA256 (tidak pernah dalam teks asli).

Sumber akun: tabel `app.users` di PostgreSQL (dibuat oleh scripts/bootstrap_app.py). Kalau database tidak tersedia,
dipakai akun demo dari environment variable (DEMO_INVESTIGATOR_PASSWORD, DEMO_MANAGEMENT_PASSWORD).
"""
from __future__ import annotations

import hashlib
import hmac
import os
import secrets

ROLE_LABELS = {"investigator": "Investigator Anti-Fraud", "manajemen": "Manajemen"}
ITERATIONS = 200_000


def hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), ITERATIONS).hex()
    return digest, salt


def demo_users() -> list[dict]:
    users = []
    for username, display_name, role, env_name, default in [
            ("investigator", "Rina Wulandari (Investigator)", "investigator", "DEMO_INVESTIGATOR_PASSWORD", "investigator123"),
            ("manajemen", "Bambang Prasetyo (Manajemen)", "manajemen", "DEMO_MANAGEMENT_PASSWORD", "manajemen123")]:
        digest, salt = hash_password(os.getenv(env_name, default), salt="a1" * 16 if role == "investigator" else "b2" * 16)
        users.append({"username": username, "display_name": display_name, "role": role, "password_hash": digest, "salt": salt})
    return users


def load_users(engine=None) -> dict:
    if engine is not None:
        try:
            import pandas as pd
            frame = pd.read_sql_table("users", engine, schema="app")
            if len(frame):
                return {row["username"]: row.to_dict() for _, row in frame.iterrows()}
        except Exception:
            pass
    return {user["username"]: user for user in demo_users()}


def verify_login(username: str, password: str, engine=None) -> dict | None:
    user = load_users(engine).get((username or "").strip().lower())
    if not user:
        return None
    digest, _ = hash_password(password or "", user["salt"])
    if hmac.compare_digest(digest, user["password_hash"]):
        return {"username": user["username"], "display_name": user["display_name"], "role": user["role"]}
    return None
