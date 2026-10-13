"""Konfigurasi aplikasi: semua diambil dari environment variable (lihat .env.example), dengan default yang aman."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path


def _project_root() -> Path:
    if os.getenv("RADAR_PROJECT_ROOT"):
        return Path(os.environ["RADAR_PROJECT_ROOT"])
    return Path(__file__).resolve().parents[2]          # .../app/radar_app/config.py → folder proyek


PROJECT_ROOT = _project_root()


def _retrieval_defaults() -> dict:
    """k dense & sparse hasil tuning di notebook (Bagian 8.4); kalau belum ada, pakai default."""
    path = PROJECT_ROOT / "outputs" / "rag_index" / "retrieval_config.json"
    try:
        return json.loads(path.read_text())
    except Exception:
        return {"dense_k": 8, "sparse_k": 8}


@dataclass(frozen=True)
class Settings:
    database_url: str = os.getenv("DATABASE_URL", "")
    app_data_dir: Path = Path(os.getenv("RADAR_APP_DATA_DIR", PROJECT_ROOT / "outputs" / "app_data"))
    model_dir: Path = Path(os.getenv("RADAR_MODEL_DIR", PROJECT_ROOT / "outputs" / "models"))
    state_dir: Path = Path(os.getenv("RADAR_STATE_DIR", PROJECT_ROOT / "outputs" / "app_state"))
    knowledge_base_dir: Path = Path(os.getenv("RADAR_KB_DIR", PROJECT_ROOT / "knowledge_base"))
    rag_index_dir: Path = Path(os.getenv("RADAR_RAG_INDEX_DIR", PROJECT_ROOT / "outputs" / "rag_index"))
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    generator_model: str = os.getenv("GENERATOR_MODEL", "gemma4:e4b")
    guard_model: str = os.getenv("GUARD_MODEL", "llama-guard3:1b")
    judge_model: str = os.getenv("JUDGE_MODEL", "qwen2.5:7b-instruct")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    generator_temperature: float = float(os.getenv("GENERATOR_TEMPERATURE", "0.1"))
    llm_timeout_seconds: int = int(os.getenv("LLM_TIMEOUT_SECONDS", "120"))
    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "")
    mlflow_ui_url: str = os.getenv("MLFLOW_UI_URL", "http://localhost:5000")
    retrieval_top_n: int = 4
    dense_k: int = int(os.getenv("RAG_DENSE_K", _retrieval_defaults().get("dense_k", 8)))
    sparse_k: int = int(os.getenv("RAG_SPARSE_K", _retrieval_defaults().get("sparse_k", 8)))
    chunk_size_tokens: int = 480
    chunk_overlap_tokens: int = 64
    chunk_max_tokens: int = 600
    embedding_window_tokens: int = 110          # MiniLM memotong input di 128 token → jendela ≤ 110 token
    max_agent_steps: int = 4
    max_input_chars: int = 2000
    roles: tuple = field(default=("investigator", "manajemen"))


SETTINGS = Settings()
for folder in (SETTINGS.state_dir, SETTINGS.rag_index_dir):
    try:
        folder.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
