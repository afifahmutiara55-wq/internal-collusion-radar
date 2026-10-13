"""Asisten AI: RAG (tanya regulasi & panduan) dan Agen (tools sesuai role) dengan guardrail, sumber, dan LLM-as-judge."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from ..config import SETTINGS
from ..genai.tools import ROLE_TOOLS, TOOL_SPECS
from ..ui import components as ui
from ..ui import services
from .common import show_guard_checks

EXAMPLES = {
    "investigator": {"rag": ["Apa saja empat pilar strategi anti fraud menurut POJK 12/2024?",
                             "Kenapa approval malam belum tentu kolusi?", "Berapa target waktu triase alert kategori Tinggi?"],
                     "agent": ["Tampilkan 5 alert dengan skor tertinggi di unit Bekasi",
                               "Siapa saja anggota komunitas risiko dengan skor tertinggi?",
                               "Siapkan draf memo investigasi untuk transaksi {top}"]},
    "manajemen": {"rag": ["Apa kewajiban pelaporan fraud ke OJK?", "Bagaimana rumus manfaat bersih sistem?",
                          "Kapan model harus dilatih ulang?"],
                  "agent": ["Ringkas kondisi risiko kolusi tahun 2026", "Unit mana yang paling berisiko?",
                            "Kalau investigator hanya 4 orang, berapa alert per bulan yang bisa ditangani?"]}}


def render_message(message: dict, index: int) -> None:
    with st.chat_message(message["role"], avatar="🧑‍💼" if message["role"] == "user" else "🛡️"):
        st.markdown(message["content"])
        result = message.get("result")
        if not result:
            return
        mode_label = {"llm": "LLM (Gemma)", "fallback": "cadangan tanpa LLM", "fallback_grounding": "kutipan sumber (grounding)",
                      "abstain": "tidak tahu (konteks lemah)", "blocked": "DIBLOKIR guardrail", "rule_based": "planner aturan (tanpa LLM)"}
        st.caption(f"mode: {mode_label.get(result['mode'], result['mode'])} · {result.get('latency_ms', 0):.0f} ms")
        if result.get("contexts"):
            with st.expander(f"Sumber ({len(result['contexts'])})"):
                for number, context in enumerate(result["contexts"], start=1):
                    st.markdown(f"**[S{number}] {context['title']}** › {context['section']}  \n"
                                f"<span class='small-note'>{context.get('url', '')} · RRF {context['rrf_score']:.3f} · "
                                f"dense #{context.get('dense_rank') or '-'} · BM25 #{context.get('sparse_rank') or '-'}</span>",
                                unsafe_allow_html=True)
                    st.markdown(f"> {context['text'].split(chr(10), 1)[-1][:500]}…")
        if result.get("steps"):
            with st.expander(f"Langkah agen ({len(result['steps'])})"):
                for step in result["steps"]:
                    st.markdown(f"**{step['step']}. `{step['tool']}`** {step['args']} → {step['status']}  \n_{step.get('thought', '')}_")
                    st.json(step["observation"], expanded=False)
        with st.expander("Status guardrail (4 lapis)"):
            show_guard_checks(result.get("checks", []))
        if message.get("judge"):
            scores = message["judge"]
            st.markdown("**Penilaian LLM-as-judge:** " + " · ".join(
                f"{criterion} {value['score']:.1f}" for criterion, value in scores.items() if pd.notna(value["score"])))
        elif result.get("contexts") and result["mode"] not in ("blocked",):
            if st.button("Nilai jawaban ini (LLM-as-judge)", key=f"judge_{index}"):
                judge = services.get_judge()
                with st.spinner(f"Juri: {judge.engine}"):
                    verdict = judge.judge(result["question"], result["answer"], result["contexts"])
                message["judge"] = verdict
                user = services.current_user()
                for criterion, value in verdict.items():
                    services.get_store().save_judge_result(user["username"], result["question"], criterion,
                                                           value["score"], value["rationale"], judge.engine, result["mode"])
                st.rerun()


def render() -> None:
    user = services.current_user()
    role = user["role"]
    ui.page_header("Asisten AI", "Tanya regulasi & panduan (RAG) atau minta agen mengerjakan tugas dengan tools sesuai wewenang Anda")
    llm = services.get_llm()
    retriever = services.get_retriever()
    status = {"Generator": (SETTINGS.generator_model, llm.has_model(SETTINGS.generator_model)),
              "Guardrail model": (SETTINGS.guard_model, llm.has_model(SETTINGS.guard_model)),
              "Juri": (SETTINGS.judge_model, llm.has_model(SETTINGS.judge_model))}
    columns = st.columns(4)
    for column, (label, (model, ok)) in zip(columns, status.items()):
        column.markdown(ui.kpi_card(label, "aktif" if ok else "cadangan", model, ui.GREEN if ok else ui.YELLOW), unsafe_allow_html=True)
    columns[3].markdown(ui.kpi_card("Knowledge base", f"{len(retriever.chunks)} chunk", retriever.embedder.name[:40], ui.BLUE),
                        unsafe_allow_html=True)
    mode = st.radio("Mode", ["Tanya regulasi & panduan (RAG)", "Agen investigasi (tools)"], horizontal=True, key="assistant_mode")
    is_agent = mode.startswith("Agen")
    with st.expander("Tools yang boleh dipakai agen untuk role Anda" if is_agent else "Cara kerja RAG"):
        if is_agent:
            st.dataframe(pd.DataFrame([{"tool": t, "fungsi": TOOL_SPECS[t]["description"]} for t in ROLE_TOOLS[role]]),
                         hide_index=True, use_container_width=True)
            st.caption("Agen hanya membaca data. Draf memo selalu menunggu persetujuan manusia. Maksimal 4 langkah per permintaan.")
        else:
            st.markdown(f"Hybrid retrieval (dense MiniLM + FAISS dan BM25, digabung RRF, top 4; dense_k={SETTINGS.dense_k}, "
                        f"sparse_k={SETTINGS.sparse_k}) → {SETTINGS.generator_model} (temperature {SETTINGS.generator_temperature}) "
                        "menjawab hanya dari konteks dengan sitasi [S#] → guardrail output.")
    history_key = f"chat_{role}_{'agent' if is_agent else 'rag'}"
    st.session_state.setdefault(history_key, [])
    top = services.table("transactions_scored").sort_values("risk_score", ascending=False)["request_id"].iloc[0]
    example_columns = st.columns(3)
    clicked = None
    for column, example in zip(example_columns, EXAMPLES[role]["agent" if is_agent else "rag"]):
        text = example.format(top=top)
        if column.button(text, key=f"ex_{history_key}_{text[:20]}", use_container_width=True):
            clicked = text
    for index, message in enumerate(st.session_state[history_key]):
        render_message(message, index)
    prompt = st.chat_input("Tulis pertanyaan atau tugas…") or clicked
    if prompt:
        st.session_state[history_key].append({"role": "user", "content": prompt})
        with st.spinner("Memproses (guardrail → retrieval/tools → jawaban)…"):
            if is_agent:
                result = services.get_agent().run(prompt, role, user["username"])
                answer = result["final_answer"]
            else:
                result = services.get_rag().answer(prompt, role, user["username"])
                answer = result["answer"]
        st.session_state[history_key].append({"role": "assistant", "content": answer, "result": result})
        if is_agent and result.get("pending_action"):
            st.session_state["pending_memo"] = result["pending_action"]
        st.rerun()
    pending = st.session_state.get("pending_memo")
    if is_agent and pending and role == "investigator":
        st.divider()
        st.markdown(f"#### Draf memo menunggu persetujuan · {pending['request_id']}")
        memo = st.text_area("Tinjau dan sunting bila perlu", pending["memo"], height=320, key="pending_memo_text")
        approve, reject = st.columns(2)
        if approve.button("Setujui & simpan", type="primary"):
            services.get_store().approve_agent_action(pending["request_id"], "memo_investigasi", memo, user["username"])
            st.session_state.pop("pending_memo", None)
            st.success("Memo disetujui dan tersimpan.")
        if reject.button("Tolak draf"):
            st.session_state.pop("pending_memo", None)
            st.rerun()
    if st.session_state[history_key] and st.button("Bersihkan percakapan"):
        st.session_state[history_key] = []
        st.rerun()
