"""Siapkan tabel aplikasi (schema app) dan akun login dengan password ter-hash.

Akun dibaca dari environment: DEMO_INVESTIGATOR_PASSWORD, DEMO_MANAGEMENT_PASSWORD (default demo: investigator123 /
manajemen123). Ganti di file .env sebelum dipakai di luar demo.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from radar_app.auth import hash_password  # noqa: E402
from radar_app.data_access import APP_TABLE_DDL  # noqa: E402

USERS = [("investigator", "Rina Wulandari (Investigator)", "investigator", "DEMO_INVESTIGATOR_PASSWORD", "investigator123"),
         ("manajemen", "Bambang Prasetyo (Manajemen)", "manajemen", "DEMO_MANAGEMENT_PASSWORD", "manajemen123")]


def main() -> int:
    url = os.getenv("DATABASE_URL")
    if not url:
        print("DATABASE_URL belum di-set; aplikasi akan memakai akun demo bawaan.")
        return 0
    engine = create_engine(url, pool_pre_ping=True)
    with engine.begin() as connection:
        for statement in [s for s in APP_TABLE_DDL.split(";") if s.strip()]:
            connection.execute(text(statement))
        connection.execute(text("""CREATE TABLE IF NOT EXISTS app.users (username TEXT PRIMARY KEY, display_name TEXT,
                                   role TEXT, password_hash TEXT, salt TEXT)"""))
    rows = []
    for username, display_name, role, env_name, default in USERS:
        digest, salt = hash_password(os.getenv(env_name, default))
        rows.append({"username": username, "display_name": display_name, "role": role, "password_hash": digest, "salt": salt})
    with engine.begin() as connection:
        connection.execute(text("DELETE FROM app.users"))
    pd.DataFrame(rows).to_sql("users", engine, schema="app", if_exists="append", index=False)
    print(f"Schema app siap; {len(rows)} akun dibuat (password tersimpan sebagai hash PBKDF2).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
