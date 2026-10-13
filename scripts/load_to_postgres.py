"""Muat hasil notebook (outputs/app_data/*.csv + *.json) ke PostgreSQL schema `radar`.

Pemakaian:
    python scripts/load_to_postgres.py                 # pakai DATABASE_URL dari environment
    python scripts/load_to_postgres.py --if-empty      # hanya kalau tabel radar.transactions_scored belum ada
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, inspect, text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_DATA_DIR = Path(os.getenv("RADAR_APP_DATA_DIR", PROJECT_ROOT / "outputs" / "app_data"))
DATE_COLUMNS = {"transactions_scored": ["event_time", "request_ts", "approve_ts"], "test_features": ["case_ts"],
                "employees": ["employment_start_date"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--if-empty", action="store_true")
    args = parser.parse_args()
    url = os.getenv("DATABASE_URL")
    if not url:
        print("DATABASE_URL belum di-set.")
        return 1
    engine = create_engine(url, pool_pre_ping=True)
    with engine.begin() as connection:
        connection.execute(text("CREATE SCHEMA IF NOT EXISTS radar"))
    if args.if_empty and inspect(engine).has_table("transactions_scored", schema="radar"):
        print("Tabel radar sudah terisi; lewati pemuatan (gunakan tanpa --if-empty untuk memuat ulang).")
        return 0
    files = sorted(APP_DATA_DIR.glob("*.csv"))
    if not files:
        print(f"Tidak ada CSV di {APP_DATA_DIR}. Jalankan notebook terlebih dahulu.")
        return 1
    for path in files:
        frame = pd.read_csv(path, low_memory=False)
        for column in DATE_COLUMNS.get(path.stem, []):
            if column in frame:
                frame[column] = pd.to_datetime(frame[column], errors="coerce")
        frame.to_sql(path.stem, engine, schema="radar", if_exists="replace", index=False, chunksize=5000, method="multi")
        print(f"radar.{path.stem:<32} {len(frame):>8,} baris")
    documents = [{"name": p.stem, "content": p.read_text(encoding="utf-8")} for p in sorted(APP_DATA_DIR.glob("*.json"))]
    pd.DataFrame(documents).to_sql("documents", engine, schema="radar", if_exists="replace", index=False)
    with engine.begin() as connection:
        for statement in ["CREATE INDEX IF NOT EXISTS ix_tx_request ON radar.transactions_scored (request_id)",
                          "CREATE INDEX IF NOT EXISTS ix_tx_band ON radar.transactions_scored (risk_band)",
                          "CREATE INDEX IF NOT EXISTS ix_tx_time ON radar.transactions_scored (event_time)",
                          "CREATE INDEX IF NOT EXISTS ix_expl_request ON radar.alert_explanations (request_id)",
                          "CREATE INDEX IF NOT EXISTS ix_anomaly_approver ON radar.approver_anomaly (approver_id)"]:
            connection.execute(text(statement))
    print(f"Selesai: {len(files)} tabel + {len(documents)} dokumen JSON dimuat ke schema radar.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
