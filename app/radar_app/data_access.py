"""Akses data untuk dashboard dan agen.

Sumber utama: PostgreSQL (schema `radar` untuk hasil analitik, schema `app` untuk catatan aplikasi).
Cadangan: file CSV hasil notebook di outputs/app_data (aplikasi tetap jalan tanpa Docker/PostgreSQL).
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import SETTINGS, Settings

DATE_COLUMNS = {"transactions_scored": ["event_time", "request_ts", "approve_ts"], "test_features": ["case_ts"],
                "employees": ["employment_start_date"]}
APP_TABLES = {
    "case_reviews": ["reviewed_at", "username", "role", "request_id", "decision", "note", "risk_score", "model_name"],
    "ai_audit_log": ["logged_at", "username", "role", "mode", "question_masked", "guard_flags", "retrieved", "tool_calls",
                     "answer", "latency_ms", "model", "fallback_used", "blocked"],
    "agent_actions": ["created_at", "username", "role", "action_type", "request_id", "content", "status", "approved_by",
                      "approved_at"],
    "judge_results": ["judged_at", "username", "question", "criterion", "score", "rationale", "judge", "answer_mode"],
    "login_log": ["logged_at", "username", "role", "success"],
}
APP_TABLE_DDL = """
CREATE SCHEMA IF NOT EXISTS app;
CREATE TABLE IF NOT EXISTS app.case_reviews (id SERIAL PRIMARY KEY, reviewed_at TIMESTAMPTZ, username TEXT, role TEXT,
    request_id TEXT, decision TEXT, note TEXT, risk_score DOUBLE PRECISION, model_name TEXT);
CREATE TABLE IF NOT EXISTS app.ai_audit_log (id SERIAL PRIMARY KEY, logged_at TIMESTAMPTZ, username TEXT, role TEXT, mode TEXT,
    question_masked TEXT, guard_flags TEXT, retrieved TEXT, tool_calls TEXT, answer TEXT, latency_ms DOUBLE PRECISION,
    model TEXT, fallback_used BOOLEAN, blocked BOOLEAN);
CREATE TABLE IF NOT EXISTS app.agent_actions (id SERIAL PRIMARY KEY, created_at TIMESTAMPTZ, username TEXT, role TEXT,
    action_type TEXT, request_id TEXT, content TEXT, status TEXT, approved_by TEXT, approved_at TIMESTAMPTZ);
CREATE TABLE IF NOT EXISTS app.judge_results (id SERIAL PRIMARY KEY, judged_at TIMESTAMPTZ, username TEXT, question TEXT,
    criterion TEXT, score DOUBLE PRECISION, rationale TEXT, judge TEXT, answer_mode TEXT);
CREATE TABLE IF NOT EXISTS app.login_log (id SERIAL PRIMARY KEY, logged_at TIMESTAMPTZ, username TEXT, role TEXT, success BOOLEAN);
"""


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class DataStore:
    """Satu pintu untuk membaca tabel analitik dan menulis catatan aplikasi."""

    def __init__(self, settings: Settings = SETTINGS):
        self.settings = settings
        self._cache: dict[str, pd.DataFrame] = {}
        self._lock = threading.Lock()
        self.engine = self._connect()

    # ------------------------------------------------------------------ koneksi
    def _connect(self):
        if not self.settings.database_url:
            return None
        try:
            from sqlalchemy import create_engine, text
            engine = create_engine(self.settings.database_url, pool_pre_ping=True)
            with engine.begin() as connection:
                connection.execute(text("SELECT 1"))
                for statement in [s for s in APP_TABLE_DDL.split(";") if s.strip()]:
                    connection.execute(text(statement))
            return engine
        except Exception as error:  # PostgreSQL belum siap / driver tidak ada → mode CSV
            print(f"[DataStore] PostgreSQL tidak tersedia ({error.__class__.__name__}); memakai CSV.")
            return None

    @property
    def backend(self) -> str:
        return "PostgreSQL" if self.engine is not None else "CSV lokal"

    # ------------------------------------------------------------------ tabel analitik
    def table(self, name: str) -> pd.DataFrame:
        """Tabel hasil notebook (jangan diubah in-place; filter menghasilkan frame baru)."""
        with self._lock:
            if name not in self._cache:
                self._cache[name] = self._read_table(name)
            return self._cache[name]

    def _read_table(self, name: str) -> pd.DataFrame:
        frame = None
        if self.engine is not None:
            try:
                frame = pd.read_sql_table(name, self.engine, schema="radar")
            except Exception:
                frame = None
        if frame is None:
            path = self.settings.app_data_dir / f"{name}.csv"
            if not path.exists():
                return pd.DataFrame()
            frame = pd.read_csv(path, low_memory=False)
        for column in DATE_COLUMNS.get(name, []):
            if column in frame:
                frame[column] = pd.to_datetime(frame[column], errors="coerce")
        return frame

    def json_document(self, name: str) -> dict:
        """app_settings.json / model_card.json (di PostgreSQL disimpan di tabel radar.documents)."""
        if self.engine is not None:
            try:
                documents = pd.read_sql_table("documents", self.engine, schema="radar")
                row = documents[documents["name"] == name]
                if len(row):
                    return json.loads(row.iloc[0]["content"])
            except Exception:
                pass
        path = self.settings.app_data_dir / f"{name}.json"
        return json.loads(path.read_text()) if path.exists() else {}

    # ------------------------------------------------------------------ catatan aplikasi
    def _append(self, table: str, record: dict) -> None:
        record = {column: record.get(column) for column in APP_TABLES[table]}
        for key, value in record.items():
            if isinstance(value, (dict, list)):
                record[key] = json.dumps(value, ensure_ascii=False, default=str)
        if self.engine is not None:
            try:
                pd.DataFrame([record]).to_sql(table, self.engine, schema="app", if_exists="append", index=False)
                return
            except Exception as error:
                print(f"[DataStore] gagal menulis ke PostgreSQL ({error}); menulis ke CSV.")
        path = self.settings.state_dir / f"{table}.csv"
        pd.DataFrame([record]).to_csv(path, mode="a", header=not path.exists(), index=False)

    def _read_app_table(self, table: str) -> pd.DataFrame:
        if self.engine is not None:
            try:
                return pd.read_sql_table(table, self.engine, schema="app")
            except Exception:
                pass
        path = self.settings.state_dir / f"{table}.csv"
        return pd.read_csv(path) if path.exists() else pd.DataFrame(columns=APP_TABLES[table])

    def save_review(self, username: str, role: str, request_id: str, decision: str, note: str,
                    risk_score: float, model_name: str) -> None:
        self._append("case_reviews", {"reviewed_at": now_utc(), "username": username, "role": role,
                                      "request_id": request_id, "decision": decision, "note": note,
                                      "risk_score": risk_score, "model_name": model_name})

    def reviews(self) -> pd.DataFrame:
        return self._read_app_table("case_reviews")

    def log_ai_event(self, event: dict) -> None:
        self._append("ai_audit_log", {"logged_at": now_utc(), **event})

    def ai_events(self) -> pd.DataFrame:
        return self._read_app_table("ai_audit_log")

    def save_agent_action(self, username: str, role: str, action_type: str, request_id: str, content: str,
                          status: str = "MENUNGGU_PERSETUJUAN") -> None:
        self._append("agent_actions", {"created_at": now_utc(), "username": username, "role": role,
                                       "action_type": action_type, "request_id": request_id, "content": content,
                                       "status": status})

    def agent_actions(self) -> pd.DataFrame:
        return self._read_app_table("agent_actions")

    def approve_agent_action(self, request_id: str, action_type: str, content: str, approved_by: str) -> None:
        """Persetujuan manusia dicatat sebagai baris baru berstatus DISETUJUI (jejak audit tidak ditimpa)."""
        self._append("agent_actions", {"created_at": now_utc(), "username": approved_by, "role": "investigator",
                                       "action_type": action_type, "request_id": request_id, "content": content,
                                       "status": "DISETUJUI", "approved_by": approved_by, "approved_at": now_utc()})

    def save_judge_result(self, username: str, question: str, criterion: str, score: float, rationale: str,
                          judge: str, answer_mode: str) -> None:
        self._append("judge_results", {"judged_at": now_utc(), "username": username, "question": question,
                                       "criterion": criterion, "score": score, "rationale": rationale,
                                       "judge": judge, "answer_mode": answer_mode})

    def judge_results(self) -> pd.DataFrame:
        return self._read_app_table("judge_results")

    def record_login(self, username: str, role: str, success: bool) -> None:
        self._append("login_log", {"logged_at": now_utc(), "username": username, "role": role, "success": success})

    def logins(self) -> pd.DataFrame:
        return self._read_app_table("login_log")
