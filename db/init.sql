-- Dijalankan otomatis oleh container PostgreSQL saat volume pertama kali dibuat (database: radar).
CREATE DATABASE mlflow;
CREATE SCHEMA IF NOT EXISTS radar;   -- hasil analitik dari notebook (dimuat oleh scripts/load_to_postgres.py)
CREATE SCHEMA IF NOT EXISTS app;     -- catatan aplikasi: akun, keputusan investigator, log AI, aksi agen, penilaian juri

CREATE TABLE IF NOT EXISTS app.users (
    username TEXT PRIMARY KEY, display_name TEXT, role TEXT CHECK (role IN ('investigator', 'manajemen')),
    password_hash TEXT NOT NULL, salt TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS app.case_reviews (
    id SERIAL PRIMARY KEY, reviewed_at TIMESTAMPTZ, username TEXT, role TEXT, request_id TEXT, decision TEXT,
    note TEXT, risk_score DOUBLE PRECISION, model_name TEXT);
CREATE TABLE IF NOT EXISTS app.ai_audit_log (
    id SERIAL PRIMARY KEY, logged_at TIMESTAMPTZ, username TEXT, role TEXT, mode TEXT, question_masked TEXT,
    guard_flags TEXT, retrieved TEXT, tool_calls TEXT, answer TEXT, latency_ms DOUBLE PRECISION, model TEXT,
    fallback_used BOOLEAN, blocked BOOLEAN);
CREATE TABLE IF NOT EXISTS app.agent_actions (
    id SERIAL PRIMARY KEY, created_at TIMESTAMPTZ, username TEXT, role TEXT, action_type TEXT, request_id TEXT,
    content TEXT, status TEXT, approved_by TEXT, approved_at TIMESTAMPTZ);
CREATE TABLE IF NOT EXISTS app.judge_results (
    id SERIAL PRIMARY KEY, judged_at TIMESTAMPTZ, username TEXT, question TEXT, criterion TEXT,
    score DOUBLE PRECISION, rationale TEXT, judge TEXT, answer_mode TEXT);
CREATE TABLE IF NOT EXISTS app.login_log (
    id SERIAL PRIMARY KEY, logged_at TIMESTAMPTZ, username TEXT, role TEXT, success BOOLEAN);
CREATE INDEX IF NOT EXISTS ix_reviews_request ON app.case_reviews (request_id);
CREATE INDEX IF NOT EXISTS ix_ai_log_time ON app.ai_audit_log (logged_at);
