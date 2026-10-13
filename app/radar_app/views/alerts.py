"""Investigator · Monitoring Alert: filter → scorecard → tabel → detail alert → keputusan (human-in-the-loop)."""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from ..ui import components as ui
from ..ui import services
from ..ui.network import LEGEND_HTML, build_graph, plotly_network, pyvis_html
from .common import DECISIONS, RED_FLAG_COLUMNS, flags_text, latest_decisions, show_guard_checks


def filter_alerts(transactions: pd.DataFrame) -> pd.DataFrame:
    frame = services.period_filter(transactions, "alerts", default_year=2026)
    with st.sidebar:
        st.markdown("**Filter alert**")
        levels = st.multiselect("Level risiko", ["Tinggi", "Sedang", "Rendah"], default=["Tinggi", "Sedang"], key="alert_levels")
        flag_labels = st.multiselect("Red flag", list(ui.RED_FLAG_LABELS.values()), key="alert_flags")
        flag_logic = st.radio("Logika red flag", ["minimal satu", "semua terpilih"], horizontal=True, key="alert_flag_logic")
        units = st.multiselect("Unit (requester/approver)", sorted(set(frame["requester_unit"].dropna()) | set(frame["approver_unit"].dropna())),
                               key="alert_units")
        employee_query = st.text_input("Nama / ID pegawai", key="alert_employee", placeholder="mis. Gita atau E00747")
        score_range = st.slider("Rentang skor risiko", 0, 100, (0, 100), key="alert_score")
        status = st.radio("Status keputusan", ["semua", "belum diputus", "sudah diputus"], key="alert_status")
    frame = frame[frame["risk_band"].isin(levels)] if levels else frame
    if flag_labels:
        columns = [c for c, label in ui.RED_FLAG_LABELS.items() if label in flag_labels]
        mask = frame[columns].eq(1).all(axis=1) if flag_logic == "semua terpilih" else frame[columns].eq(1).any(axis=1)
        frame = frame[mask]
    if units:
        frame = frame[frame["requester_unit"].isin(units) | frame["approver_unit"].isin(units)]
    if employee_query:
        employees = services.table("employees")
        matched = employees[employees["employee_name"].str.contains(employee_query, case=False, na=False)
                            | employees["employee_id"].str.upper().eq(employee_query.strip().upper())]["employee_id"]
        frame = frame[frame["requester_id"].isin(matched) | frame["approver_id"].isin(matched)]
    frame = frame[frame["risk_score"].between(*score_range)]
    decisions = latest_decisions()
    if status == "belum diputus":
        frame = frame[~frame["request_id"].isin(decisions.index)]
    elif status == "sudah diputus":
        frame = frame[frame["request_id"].isin(decisions.index)]
    return frame


def score_gauge(score: float, threshold: float) -> go.Figure:
    figure = go.Figure(go.Indicator(
        mode="gauge+number", value=score, number={"suffix": " /100", "font": {"size": 30}},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": ui.INK},
               "steps": [{"range": [0, threshold * 50], "color": "#dbe9fb"},
                         {"range": [threshold * 50, threshold * 100], "color": "#fbe7b5"},
                         {"range": [threshold * 100, 100], "color": "#f8c9b3"}],
               "threshold": {"line": {"color": ui.ORANGE, "width": 3}, "value": threshold * 100}}))
    figure.update_layout(height=210, margin=dict(l=20, r=20, t=20, b=0))
    return figure


def shap_bar(explanations: pd.DataFrame) -> go.Figure:
    frame = explanations.sort_values("rank", ascending=False)
    figure = go.Figure(go.Bar(x=frame["shap_value"], y=frame["reason"], orientation="h",
                              marker_color=[ui.GROUP_COLORS.get(g, ui.BLUE) for g in frame["feature_group"]],
                              text=[f"+{v:.2f}" for v in frame["shap_value"]], textposition="outside",
                              hovertext=frame["feature"] + " · kelompok " + frame["feature_group"]))
    figure.update_layout(xaxis_title="kontribusi SHAP (menaikkan risiko)", yaxis=dict(automargin=True))
    return ui.style_figure(figure, "Tiga alasan utama skor (SHAP)", "warna = kelompok fitur", height=260)


def ego_network(row: pd.Series) -> nx.Graph | None:
    pairs = services.table("pair_network")
    people = {row["requester_id"], row["approver_id"]}
    neighborhood = pairs[pairs["requester_id"].isin(people) | pairs["approver_id"].isin(people)].copy()
    if neighborhood.empty:
        return None
    neighborhood = neighborhood.sort_values(["risky", "n"], ascending=False).head(30)
    edges = pd.DataFrame({"node_a": neighborhood["requester_id"], "node_b": neighborhood["approver_id"],
                          "weight": np.where(neighborhood["risky"].astype(str).str.lower().eq("true"),
                                             neighborhood["z"].clip(lower=1), 0.4),
                          "kind": "RA", "redflag": neighborhood["risky"].astype(str).str.lower().eq("true") & (neighborhood["z"] >= 3),
                          "dependency": neighborhood["dependency"], "n_tx": neighborhood["n"], "z": neighborhood["z"]})
    community_edges = services.table("community_edges")
    shared = community_edges[(community_edges["kind"] == "RR") & (community_edges["node_a"].isin(people) | community_edges["node_b"].isin(people))]
    edges = pd.concat([edges, shared[["node_a", "node_b", "weight", "kind", "shared"]]], ignore_index=True)
    return build_graph(edges, services.table("employees"), highlight=people)


def render_detail(row: pd.Series, settings: dict) -> None:
    names = services.name_map()
    explanations = services.table("alert_explanations")
    reasons = explanations[explanations["request_id"] == row["request_id"]]
    st.markdown(f"### Detail alert {row['request_id']} &nbsp; {ui.band_badge(row['risk_band'])}", unsafe_allow_html=True)
    st.caption(f"approval {row['approve_ts']:%A, %d %b %Y %H:%M} · periode model: {row.get('model_period', '-')}")
    column_a, column_b, column_c = st.columns([1, 1.4, 1.3])
    with column_a:
        st.plotly_chart(score_gauge(float(row["risk_score"]), settings.get("final_threshold", 0.5)), use_container_width=True)
        st.markdown(f"**Requester:** {names.get(row['requester_id'], row['requester_id'])} ({row['requester_id']}) · {row['requester_unit']}  \n"
                    f"**Approver:** {names.get(row['approver_id'], row['approver_id'])} ({row['approver_id']}) · {row['approver_position']}  \n"
                    f"**Debitur:** {row.get('debtor_name_masked', '-')} ({row['debtor_id']}) · {row['debtor_type']} · {row['business_sector']}  \n"
                    f"**Kredit:** {row['loan_product']} · {ui.rupiah(row['amount'], short=False)} · {ui.percent(row['limit_ratio'])} limit approver")
    with column_b:
        if len(reasons):
            st.plotly_chart(shap_bar(reasons), use_container_width=True)
        else:
            st.info("Alasan SHAP disimpan untuk alert Tinggi/Sedang.")
        flags = [ui.RED_FLAG_LABELS[c] for c in RED_FLAG_COLUMNS if row.get(c) == 1]
        st.markdown("**Red flag terpenuhi:** " + (ui.chips(flags) if flags else "tidak ada"), unsafe_allow_html=True)
    with column_c:
        st.markdown("**Bukti dari data**")
        evidence = [e for e in str(row.get("evidence") or "").split(" | ") if e and e != "nan"]
        st.markdown("<ul class='evidence'>" + "".join(f"<li>{e}</li>" for e in evidence) + "</ul>" if evidence else "-",
                    unsafe_allow_html=True)
        anomaly_score = row.get("approver_anomaly_score")
        if pd.notna(anomaly_score):
            st.markdown(f"**Skor anomali approver (IF, model terpisah):** {anomaly_score:.0f}/100")
        st.caption("Skor adalah indikasi berbasis data, bukan bukti. Keputusan ada di tangan investigator.")
    tab_history, tab_network, tab_decision, tab_memo = st.tabs(["Riwayat pasangan", "Jaringan", "Keputusan", "Draf memo (agen AI)"])
    with tab_history:
        transactions = services.table("transactions_scored")
        pair = transactions[(transactions["requester_id"] == row["requester_id"]) & (transactions["approver_id"] == row["approver_id"])]
        figure = px.scatter(pair, x="event_time", y="risk_score", color="risk_band", color_discrete_map=ui.BAND_COLORS,
                            size=pair["amount"] / pair["amount"].max() * 20 + 4, hover_data=["request_id", "amount", "rule_score"],
                            labels={"event_time": "waktu", "risk_score": "skor"})
        st.plotly_chart(ui.style_figure(figure, f"{len(pair)} transaksi pasangan ini (2025–2026)",
                                        "ukuran titik = nominal"), use_container_width=True)
        summary = {"rasio requester → approver": ui.percent(row.get("requester_approver_ratio")),
                   "kali lipat kebiasaan unit (lift)": f"{row.get('pair_dependency_lift', np.nan):.1f}×" if pd.notna(row.get("pair_dependency_lift")) else "-",
                   "% disetujui (1 thn)": ui.percent(row.get("pair_approval_rate")),
                   "debitur sama ≤ 1 thn": ui.number(row.get("debtor_loans_1y"))}
        ui.kpi_row([(k, v, "", ui.PURPLE) for k, v in summary.items()])
    with tab_network:
        graph = ego_network(row)
        if graph is None:
            st.info("Pasangan ini belum punya riwayat di snapshot graph terbaru.")
        else:
            st.markdown(LEGEND_HTML, unsafe_allow_html=True)
            html = pyvis_html(graph, height=480)
            if html:
                components.html(html, height=500)
            else:
                st.plotly_chart(plotly_network(graph, "Jaringan di sekitar pasangan ini"), use_container_width=True)
    with tab_decision:
        user = services.current_user()
        decisions = latest_decisions()
        if row["request_id"] in decisions.index:
            last = decisions.loc[row["request_id"]]
            st.success(f"Keputusan terakhir: **{last['decision']}** oleh {last['username']} ({last['reviewed_at']}). {last['note'] or ''}")
        with st.form(f"decision_{row['request_id']}"):
            decision = st.radio("Keputusan investigator", DECISIONS, horizontal=True)
            note = st.text_area("Catatan (penjelasan sah yang dicek, bukti, rencana klarifikasi)", height=90)
            if st.form_submit_button("Simpan keputusan", type="primary"):
                services.get_store().save_review(user["username"], user["role"], row["request_id"], decision, note,
                                                 float(row["risk_score"]), settings.get("model_name", "-"))
                st.toast("Keputusan tersimpan (jejak audit + umpan balik model).")
                st.rerun()
    with tab_memo:
        memo_key = f"memo_{row['request_id']}"
        if st.button("Minta agen menyusun draf memo investigasi", key=f"btn_{memo_key}"):
            with st.spinner("Agen menjalankan detail_alert → cari_regulasi → draf_memo_investigasi…"):
                st.session_state[memo_key] = services.get_agent().run(
                    f"Siapkan draf memo investigasi untuk transaksi {row['request_id']}", "investigator",
                    services.current_user()["username"])
        result = st.session_state.get(memo_key)
        if result:
            st.caption(f"Mode planner: {result['mode']} · {len(result['steps'])} langkah · {result['latency_ms']:.0f} ms")
            for step in result["steps"]:
                st.markdown(f"- langkah {step['step']}: `{step['tool']}` → {step['status']}")
            if result.get("pending_action"):
                memo = st.text_area("Draf memo (boleh disunting sebelum disetujui)", result["pending_action"]["memo"], height=360,
                                    key=f"text_{memo_key}")
                approve, reject = st.columns(2)
                if approve.button("Setujui & simpan memo", type="primary", key=f"ok_{memo_key}"):
                    services.get_store().approve_agent_action(row["request_id"], "memo_investigasi", memo,
                                                              services.current_user()["username"])
                    st.success("Memo disetujui dan tersimpan (agent_actions, status DISETUJUI).")
                if reject.button("Buang draf", key=f"no_{memo_key}"):
                    st.session_state.pop(memo_key, None)
                    st.rerun()
            with st.expander("Status guardrail"):
                show_guard_checks(result["checks"])


def render() -> None:
    ui.page_header("Monitoring Alert", "Saring alert berdasarkan level risiko, red flag, unit, pegawai, dan skor; buka detail; catat keputusan")
    transactions = services.table("transactions_scored")
    if transactions.empty:
        ui.empty_state("Data belum tersedia.")
        return
    settings = services.app_settings()
    frame = filter_alerts(transactions)
    decisions = latest_decisions()
    ui.kpi_row([("Alert sesuai filter", ui.number(len(frame)), st.session_state.get("alerts_period_label", ""), ui.BLUE),
                ("Tinggi", ui.number((frame["risk_band"] == "Tinggi").sum()), "", ui.ORANGE),
                ("Nominal", ui.rupiah(frame["amount"].sum()), "total kredit pada alert", ui.YELLOW),
                ("Skor rata-rata", ui.number(frame["risk_score"].mean(), 1) if len(frame) else "-", "0–100", ui.PURPLE),
                ("Approver unik", ui.number(frame["approver_id"].nunique()), "", ui.GREEN),
                ("Belum diputus", ui.number((~frame["request_id"].isin(decisions.index)).sum()), "", ui.RED)])
    st.write("")
    names = services.name_map()
    table = frame.sort_values(["risk_score", "event_time"], ascending=False).head(500).copy()
    table["requester"] = table["requester_id"].map(names)
    table["approver"] = table["approver_id"].map(names)
    table["red flag"] = table.apply(flags_text, axis=1)
    table["status"] = np.where(table["request_id"].isin(decisions.index), "diputus", "belum")
    table["nominal"] = table["amount"].map(ui.rupiah)
    table["alasan utama"] = table["top_reasons"].fillna("").str.split(" | ").str[0]
    st.caption(f"Menampilkan {len(table):,} dari {len(frame):,} alert (urut skor tertinggi). Klik satu baris untuk membuka detail.")
    selection = st.dataframe(
        table[["request_id", "event_time", "requester", "approver", "requester_unit", "nominal", "risk_score", "risk_band",
               "red flag", "status", "alasan utama"]],
        hide_index=True, use_container_width=True, height=360, on_select="rerun", selection_mode="single-row", key="alert_table",
        column_config={"risk_score": st.column_config.ProgressColumn("skor", min_value=0, max_value=100, format="%.0f"),
                       "event_time": st.column_config.DatetimeColumn("waktu", format="DD MMM YYYY HH:mm"),
                       "requester_unit": "unit", "risk_band": "level"})
    selected_rows = getattr(getattr(selection, "selection", None), "rows", []) or []
    if len(table) == 0:
        return
    st.download_button("Unduh hasil filter (CSV)", table.drop(columns=["top_reasons"], errors="ignore").to_csv(index=False),
                       "alert_terfilter.csv", "text/csv")
    selected = table.iloc[selected_rows[0]] if selected_rows else None
    if selected is None:
        request_id = st.selectbox("…atau pilih ID transaksi", table["request_id"].tolist(), key="alert_pick")
        selected = table[table["request_id"] == request_id].iloc[0]
    st.divider()
    render_detail(selected, settings)
