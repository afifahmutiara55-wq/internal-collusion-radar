"""Investigator · Profil Pegawai: radar anomali approver (Isolation Forest) + pasangan kerja + alert terkait."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ..ui import components as ui
from ..ui import services

FEATURE_LABELS = {"ex_near_limit": "near-limit vs rekan", "ex_after_hours": "luar jam kerja vs rekan",
                  "ex_fast_approval": "approval kilat vs rekan", "ex_debtor_repeat": "debitur berulang vs rekan",
                  "ex_pair_dependency": "ketergantungan requester vs rekan", "ex_multi_flag": "multi rule vs rekan",
                  "approve_rate": "tingkat approve", "top_requester_share": "konsentrasi 1 requester",
                  "ga_risky_pairs": "[graph] pasangan berisiko", "ga_max_pair_z": "[graph] z pasangan terburuk",
                  "ga_shared_debtor_share": "[graph] % debitur bersama", "ga_comm_score": "[graph] skor komunitas",
                  "ga_comm_size": "[graph] ukuran komunitas", "ga_risk_pagerank": "[graph] sentralitas"}


def employee_table() -> pd.DataFrame:
    employees = services.table("employees").copy()
    anomaly = services.table("approver_anomaly")
    latest_month = anomaly["month"].max()
    latest = anomaly[anomaly["month"] == latest_month].set_index("approver_id")
    employees["skor anomali"] = employees["employee_id"].map(latest["if_score"])
    employees["kategori anomali"] = employees["employee_id"].map(latest["anomaly_band"]).fillna("—")
    transactions = services.table("transactions_scored")
    tx_2026 = transactions[(transactions["event_time"].dt.year == 2026) & (transactions["risk_band"] == "Tinggi")]
    counts = pd.concat([tx_2026["requester_id"], tx_2026["approver_id"]]).value_counts()
    employees["alert Tinggi 2026"] = employees["employee_id"].map(counts).fillna(0).astype(int)
    community = services.table("community_members").drop_duplicates("employee_id").set_index("employee_id")["community_id"]
    employees["komunitas"] = employees["employee_id"].map(community)
    return employees


def render() -> None:
    ui.page_header("Profil Pegawai", "Siapa yang perilakunya menyimpang dari rekan sejabatan, dan dengan siapa ia bekerja")
    employees = employee_table()
    with st.sidebar:
        st.markdown("**Filter pegawai**")
        role = st.radio("Peran", ["Approver", "Requester", "Semua"], key="emp_role")
        units = st.multiselect("Unit", sorted(employees["unit_name"].unique()), key="emp_units")
        positions = st.multiselect("Jabatan", sorted(employees["position"].dropna().unique()), key="emp_positions")
        bands = st.multiselect("Kategori anomali", ["Perlu dilihat", "Pantau", "Normal"], key="emp_bands")
        query = st.text_input("Cari nama / ID", key="emp_query")
        only_alerts = st.checkbox("Hanya yang punya alert Tinggi 2026", key="emp_only_alerts")
    frame = employees if role == "Semua" else employees[employees["role"] == role]
    if units:
        frame = frame[frame["unit_name"].isin(units)]
    if positions:
        frame = frame[frame["position"].isin(positions)]
    if bands:
        frame = frame[frame["kategori anomali"].isin(bands)]
    if query:
        frame = frame[frame["employee_name"].str.contains(query, case=False, na=False) | frame["employee_id"].str.contains(query.upper())]
    if only_alerts:
        frame = frame[frame["alert Tinggi 2026"] > 0]
    frame = frame.sort_values(["skor anomali", "alert Tinggi 2026"], ascending=False, na_position="last")
    ui.kpi_row([("Pegawai sesuai filter", ui.number(len(frame)), "", ui.BLUE),
                ("Approver perlu dilihat", ui.number((frame["kategori anomali"] == "Perlu dilihat").sum()), "skor ≥ 95", ui.ORANGE),
                ("Di komunitas risiko", ui.number(frame["komunitas"].notna().sum()), "", ui.PURPLE),
                ("Punya alert Tinggi 2026", ui.number((frame["alert Tinggi 2026"] > 0).sum()), "", ui.RED)])
    selection = st.dataframe(
        frame[["employee_name", "employee_id", "role", "position", "unit_name", "region", "skor anomali", "kategori anomali",
               "alert Tinggi 2026", "komunitas"]],
        hide_index=True, use_container_width=True, height=320, on_select="rerun", selection_mode="single-row", key="emp_table",
        column_config={"employee_name": "nama", "unit_name": "unit",
                       "skor anomali": st.column_config.ProgressColumn("skor anomali", min_value=0, max_value=100, format="%.0f")})
    if frame.empty:
        return
    rows = getattr(getattr(selection, "selection", None), "rows", []) or []
    employee = frame.iloc[rows[0]] if rows else frame.iloc[0]
    st.divider()
    st.markdown(f"### {employee['employee_name']} ({employee['employee_id']})")
    st.caption(f"{employee['position']} · {employee['unit_name']} ({employee.get('region', '-')}) · bergabung "
               f"{pd.to_datetime(employee['employment_start_date']).strftime('%b %Y') if pd.notna(employee['employment_start_date']) else '-'}")
    anomaly = services.table("approver_anomaly")
    history = anomaly[anomaly["approver_id"] == employee["employee_id"]].sort_values("month")
    transactions = services.table("transactions_scored")
    related = transactions[(transactions["requester_id"] == employee["employee_id"]) | (transactions["approver_id"] == employee["employee_id"])]
    if len(history):
        latest = history.iloc[-1]
        ui.kpi_row([("Skor anomali terakhir", f"{latest['if_score']:.0f}/100", f"{latest['anomaly_band']} · {latest['month']}",
                     ui.ANOMALY_COLORS.get(latest["anomaly_band"], ui.BLUE)),
                    ("Bulan ≥ 95 (12 bln)", ui.number((history.tail(12)["if_score"] >= 95).sum()), "persistensi anomali", ui.ORANGE),
                    ("Keputusan 90 hari", ui.number(latest["n"]), f"tingkat approve {ui.percent(latest['approve_rate'])}", ui.BLUE),
                    ("Alert Tinggi (2025–2026)", ui.number((related["risk_band"] == "Tinggi").sum()), "sebagai requester/approver", ui.RED)])
        st.markdown(f"**Alasan anomali (SHAP Isolation Forest):** {latest['if_reason'] or 'skor di bawah 80, tidak diurai'}")
        left, right = st.columns([1.2, 1])
        figure = px.line(history.tail(24), x="month", y="if_score", markers=True, labels={"month": "bulan", "if_score": "skor anomali"})
        figure.add_hline(y=95, line_dash="dash", line_color=ui.ORANGE, annotation_text="perlu dilihat")
        figure.add_hline(y=80, line_dash="dot", line_color=ui.YELLOW, annotation_text="pantau")
        left.plotly_chart(ui.style_figure(figure, "Tren skor anomali (24 bulan)", "persentil terhadap semua approver-bulan"),
                          use_container_width=True)
        peers = anomaly[(anomaly["month"] == latest["month"]) & (anomaly["position"] == latest["position"])]
        features = [f for f in FEATURE_LABELS if f in history.columns]
        comparison = pd.DataFrame({"fitur": [FEATURE_LABELS[f] for f in features],
                                   "pegawai ini": [latest[f] for f in features],
                                   "median rekan sejabatan": [peers[f].median() for f in features]})
        figure = go.Figure()
        figure.add_bar(y=comparison["fitur"], x=comparison["median rekan sejabatan"], orientation="h", name="median rekan", marker_color=ui.MUTED)
        figure.add_bar(y=comparison["fitur"], x=comparison["pegawai ini"], orientation="h", name="pegawai ini", marker_color=ui.ORANGE)
        figure.update_layout(barmode="group", yaxis=dict(autorange="reversed"))
        right.plotly_chart(ui.style_figure(figure, "Dibanding rekan sejabatan (bulan terakhir)", "ex_* = kelebihan dalam SD", height=460),
                           use_container_width=True)
    else:
        st.info("Pegawai ini bukan approver (atau keputusannya < 10 dalam 90 hari), sehingga tidak dinilai Isolation Forest.")
    pairs = services.table("pair_network")
    column = "approver_id" if employee["role"] == "Requester" else "requester_id"
    own = pairs[pairs["requester_id" if employee["role"] == "Requester" else "approver_id"] == employee["employee_id"]].copy()
    own["pasangan"] = own[column].map(services.name_map()) + " (" + own[column] + ")"
    own = own.sort_values(["risky", "n"], ascending=False).head(15)
    left, right = st.columns(2)
    with left:
        st.markdown("**Pasangan kerja (12 bulan, snapshot graph terbaru)**")
        st.dataframe(own[["pasangan", "n", "k", "z", "risky", "dependency_lift"]], hide_index=True, use_container_width=True,
                     column_config={"n": "transaksi", "k": "ber-red-flag", "z": st.column_config.NumberColumn("z red flag", format="%.1f"),
                                    "risky": "berisiko", "dependency_lift": st.column_config.NumberColumn("lift ketergantungan", format="%.1f")})
    with right:
        st.markdown("**Alert terkait (2025–2026)**")
        st.dataframe(related[related["risk_band"] != "Rendah"].sort_values("risk_score", ascending=False)
                     [["request_id", "event_time", "risk_score", "risk_band", "amount", "top_reasons"]].head(20),
                     hide_index=True, use_container_width=True,
                     column_config={"risk_score": st.column_config.ProgressColumn("skor", min_value=0, max_value=100, format="%.0f"),
                                    "event_time": st.column_config.DatetimeColumn("waktu", format="DD MMM YYYY")})
