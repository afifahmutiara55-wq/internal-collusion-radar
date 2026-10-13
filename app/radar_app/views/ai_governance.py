"""Manajemen · Audit & Kualitas AI: log AI, statistik guardrail, evaluasi offline & LLM-as-judge, umpan balik investigator."""
from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import streamlit as st

from ..ui import components as ui
from ..ui import services
from .common import parse_json_column


def render() -> None:
    ui.page_header("Audit & Kualitas AI", "Setiap prompt, tool call, keputusan guardrail, dan penilaian kualitas tercatat")
    store = services.get_store()
    events = store.ai_events()
    judge_live = store.judge_results()
    total = len(events)
    blocked = events["blocked"].astype(str).str.lower().eq("true").sum() if total else 0
    fallback = events["fallback_used"].astype(str).str.lower().eq("true").sum() if total else 0
    ui.kpi_row([("Interaksi AI", ui.number(total), f"sumber: {store.backend}", ui.BLUE),
                ("Diblokir guardrail", ui.number(blocked), ui.percent(blocked / total) if total else "-", ui.ORANGE),
                ("Mode cadangan", ui.number(fallback), "LLM tidak tersedia / grounding lemah", ui.YELLOW),
                ("Latensi median", f"{events['latency_ms'].median():.0f} ms" if total else "-", "", ui.GREEN),
                ("Penilaian juri (live)", ui.number(len(judge_live)), "dari tombol LLM-as-judge", ui.PURPLE)])
    tab_log, tab_guard, tab_quality, tab_feedback = st.tabs(["Log AI", "Guardrail", "Kualitas jawaban (LLM-as-judge)", "Umpan balik investigator"])
    with tab_log:
        if total:
            frame = events.sort_values("logged_at", ascending=False)
            mode_filter = st.multiselect("Mode", sorted(frame["mode"].unique()), key="ai_mode_filter")
            if mode_filter:
                frame = frame[frame["mode"].isin(mode_filter)]
            st.dataframe(frame[["logged_at", "username", "role", "mode", "question_masked", "answer", "latency_ms", "blocked", "fallback_used", "tool_calls"]],
                         hide_index=True, use_container_width=True, height=420)
            st.caption("Pertanyaan disimpan dalam bentuk yang sudah di-masking (PII tidak tersimpan).")
        else:
            st.info("Belum ada interaksi AI. Coba halaman Asisten AI.")
    with tab_guard:
        if total:
            checks = [check for value in events["guard_flags"] for check in parse_json_column(value)]
            frame = pd.DataFrame(checks)
            if len(frame):
                summary = frame.groupby(["layer", "check"])["passed"].agg(lambda s: (~s.astype(bool)).sum()).rename("gagal/ditahan").reset_index()
                figure = px.bar(summary.sort_values("gagal/ditahan"), x="gagal/ditahan", y="check", color="layer", orientation="h")
                st.plotly_chart(ui.style_figure(figure, "Pemeriksaan guardrail yang memblokir/menahan (live)"), use_container_width=True)
        offline = services.table("guardrail_eval")
        if len(offline):
            attacks = offline[offline["harus_diblok"].astype(str).str.lower().eq("true")]
            benign = offline[~offline["harus_diblok"].astype(str).str.lower().eq("true")]
            ui.kpi_row([("Serangan tertangani (uji red-team)", ui.percent(attacks["benar"].astype(str).str.lower().eq("true").mean()), f"{len(attacks)} serangan", ui.GREEN),
                        ("Pertanyaan wajar salah diblok", ui.percent(1 - benign["benar"].astype(str).str.lower().eq("true").mean()), f"{len(benign)} pertanyaan", ui.ORANGE)])
            st.dataframe(offline, hide_index=True, use_container_width=True)
    with tab_quality:
        summary = services.table("judge_summary")
        genai = store.json_document("genai_summary")
        st.markdown(f"**Evaluasi offline** pada set uji RAG · juri: {genai.get('judge_engine', '-')} · retrieval Hit@4 "
                    f"{ui.percent(genai.get('hit_at_4', 0))}, MRR {genai.get('mrr_at_4', 0):.2f} (dense_k={genai.get('dense_k')}, sparse_k={genai.get('sparse_k')})")
        if len(summary):
            figure = px.bar(summary, x="skor rata-rata (1–5)", y="kriteria", orientation="h", text_auto=".2f", color_discrete_sequence=[ui.PURPLE])
            figure.update_layout(xaxis_range=[0, 5.2])
            st.plotly_chart(ui.style_figure(figure, "Skor juri per kriteria (1–5)", "faithfulness · relevance · correctness · citation · safety_tone"),
                            use_container_width=True)
        st.dataframe(services.table("judge_eval"), hide_index=True, use_container_width=True, height=300)
        if len(judge_live):
            st.markdown("**Penilaian live dari pengguna**")
            st.dataframe(judge_live.sort_values("judged_at", ascending=False), hide_index=True, use_container_width=True)
        agent_eval = services.table("agent_eval")
        if len(agent_eval):
            st.markdown(f"**Evaluasi agen:** ketepatan tool & kepatuhan wewenang {ui.percent(agent_eval['benar'].astype(str).str.lower().eq('true').mean())}")
            st.dataframe(agent_eval, hide_index=True, use_container_width=True)
    with tab_feedback:
        reviews = store.reviews()
        if reviews.empty:
            st.info("Belum ada keputusan investigator.")
        else:
            counts = reviews["decision"].value_counts().rename_axis("keputusan").reset_index(name="jumlah")
            st.plotly_chart(ui.style_figure(px.bar(counts, x="keputusan", y="jumlah", color_discrete_sequence=[ui.BLUE]),
                                            "Keputusan investigator", "proporsi 'alarm palsu' adalah ukuran precision di lapangan"),
                            use_container_width=True)
            st.dataframe(reviews.sort_values("reviewed_at", ascending=False), hide_index=True, use_container_width=True)
        logins = store.logins()
        if len(logins):
            st.markdown("**Log login**")
            st.dataframe(logins.sort_values("logged_at", ascending=False).head(50), hide_index=True, use_container_width=True)
