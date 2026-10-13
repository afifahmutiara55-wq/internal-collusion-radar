"""Sumber daya bersama yang di-cache Streamlit: data store, model, indeks RAG, LLM, agen."""
from __future__ import annotations

import joblib
import pandas as pd
import streamlit as st

from ..config import SETTINGS
from ..data_access import DataStore


@st.cache_resource(show_spinner="Memuat data dari database…")
def get_store() -> DataStore:
    return DataStore()


def table(name: str) -> pd.DataFrame:
    return get_store().table(name)


def app_settings() -> dict:
    return get_store().json_document("app_settings")


@st.cache_resource(show_spinner="Memuat model skor risiko…")
def get_risk_model():
    path = SETTINGS.model_dir / "transaction_risk_model.joblib"
    try:
        return joblib.load(path)
    except Exception as error:
        st.session_state["model_load_error"] = f"{error.__class__.__name__}: {error}"
        return None


@st.cache_resource(show_spinner="Memuat knowledge base & indeks RAG…")
def get_retriever():
    from ..genai.retriever import load_or_build_retriever
    return load_or_build_retriever(SETTINGS.rag_index_dir, SETTINGS.knowledge_base_dir, SETTINGS.embedding_model,
                                   {"chunk_size": SETTINGS.chunk_size_tokens, "chunk_overlap": SETTINGS.chunk_overlap_tokens,
                                    "max_tokens": SETTINGS.chunk_max_tokens})


@st.cache_resource
def get_llm():
    from ..genai.llm import OllamaClient
    return OllamaClient(SETTINGS.ollama_base_url, SETTINGS.llm_timeout_seconds)


def known_names_and_units() -> tuple[set, set]:
    employees = table("employees")
    return set(employees["employee_name"]), set(employees["unit_name"])


@st.cache_resource
def get_rag():
    from ..genai.rag import RagAssistant
    names, _ = known_names_and_units()
    return RagAssistant(get_retriever(), get_llm(), get_store(), known_entities=names)


@st.cache_resource
def get_agent():
    from ..genai.agent import InvestigationAgent
    from ..genai.tools import AgentTools
    names, units = known_names_and_units()
    retriever = get_retriever()
    return InvestigationAgent(AgentTools(get_store(), retriever), get_llm(), get_store(), known_names=names,
                              known_units=units, embedder=retriever.embedder)


@st.cache_resource
def get_judge():
    from ..genai.judge import LLMJudge
    return LLMJudge(get_llm(), SETTINGS.judge_model, use_mlflow=False)   # di aplikasi: panggilan Ollama langsung


def employee_label(employee_id) -> str:
    employees = table("employees").set_index("employee_id")
    if employee_id in employees.index:
        return f"{employees.loc[employee_id, 'employee_name']} ({employee_id})"
    return str(employee_id)


def name_map() -> dict:
    return table("employees").set_index("employee_id")["employee_name"].to_dict()


def current_user() -> dict:
    return st.session_state.get("user", {"username": "-", "display_name": "-", "role": "investigator"})


def period_filter(frame: pd.DataFrame, key: str, default_year: int | None = None) -> pd.DataFrame:
    """Filter periode di sidebar (tahun + rentang bulan) untuk tabel transaksi."""
    years = sorted(frame["event_time"].dt.year.dropna().unique().astype(int))
    if not years:
        return frame
    default_year = default_year if default_year in years else years[-1]
    with st.sidebar:
        st.markdown("**Periode**")
        year = st.selectbox("Tahun", years, index=years.index(default_year), key=f"{key}_year")
        months = sorted(frame.loc[frame["event_time"].dt.year == year, "event_time"].dt.month.unique().astype(int))
        month_range = st.select_slider("Bulan", options=months, value=(months[0], months[-1]), key=f"{key}_months") \
            if len(months) > 1 else (months[0], months[0])
    mask = (frame["event_time"].dt.year == year) & frame["event_time"].dt.month.between(*month_range)
    st.session_state[f"{key}_period_label"] = f"{year}, bulan {month_range[0]}–{month_range[1]}"
    return frame[mask]
