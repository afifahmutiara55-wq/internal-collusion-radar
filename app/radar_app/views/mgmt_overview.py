"""Manajemen · Ringkasan Eksekutif: KPI, tren, precision populasi, dampak, dan wilayah berisiko."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ..ui import components as ui
from ..ui import services
from .common import latest_decisions


def render() -> None:
    ui.page_header("Ringkasan Eksekutif", "Seberapa besar risiko kolusi, apakah sistem bekerja, dan berapa nilainya bagi bank")
    settings = services.app_settings()
    transactions = services.table("transactions_scored")
    year = st.sidebar.selectbox("Tahun", sorted(transactions["event_time"].dt.year.unique().astype(int), reverse=True), key="exec_year")
    frame = transactions[transactions["event_time"].dt.year == year]
    months = max(frame["event_time"].dt.to_period("M").nunique(), 1)
    high = frame[frame["risk_band"] == "Tinggi"]
    reviewed = high[high["reviewed"].astype(str).str.lower().eq("true")]
    true_reviewed = int((reviewed["label_audit"] == 1).sum())
    reviewed_precision = true_reviewed / max(len(reviewed), 1)
    lower = true_reviewed / max(len(high), 1)
    upper = (true_reviewed + reviewed_precision * (len(high) - len(reviewed))) / max(len(high), 1)
    impact = services.table("impact")
    model_impact = impact[impact["skenario"].str.startswith("Model")].iloc[0] if len(impact) else None
    decisions = latest_decisions()
    backlog = int((~high["request_id"].isin(decisions.index)).sum())
    ui.kpi_row([
        ("Transaksi dinilai", ui.number(len(frame)), f"{year} · {months} bulan · {ui.rupiah(frame['amount'].sum())}", ui.BLUE),
        ("Alert Tinggi / bulan", ui.number(len(high) / months, 0), f"rules: ±{ui.number((frame['rule_score'] >= settings.get('rules_threshold', 1)).sum() / months)}/bulan", ui.ORANGE),
        ("Nominal berisiko", ui.rupiah(high["amount"].sum()), "total kredit pada alert Tinggi", ui.RED),
        ("Precision populasi", f"{ui.percent(lower)}–{ui.percent(upper)}", "rentang (yang belum direview tidak diketahui)", ui.GREEN),
        ("Manfaat bersih/tahun", ui.rupiah(model_impact["manfaat bersih/tahun"]) if model_impact is not None else "-", "estimasi (lihat Dampak Bisnis)", ui.PURPLE),
        ("Backlog investigasi", ui.number(backlog), "alert Tinggi belum diputus", ui.YELLOW)])
    st.write("")
    with st.container(border=True):
        st.markdown("**Problem statement (dari data):** " + settings.get("problem_statement", "-"))
    kpis = services.table("monthly_kpis")
    left, right = st.columns([1.5, 1])
    figure = go.Figure()
    figure.add_bar(x=kpis["month"], y=kpis["rule_flagged"], name="ditandai rules", marker_color="#d8d7d2")
    figure.add_bar(x=kpis["month"], y=kpis["alert_tinggi"], name="alert Tinggi (model)", marker_color=ui.ORANGE)
    figure.add_scatter(x=kpis["month"], y=kpis["kolusi_terkonfirmasi"], name="kolusi terkonfirmasi audit", mode="lines",
                       line=dict(color=ui.INK, width=2))
    figure.add_vrect(x0="2025-01", x1="2025-12", fillcolor=ui.YELLOW, opacity=.08, line_width=0, annotation_text="validasi")
    figure.add_vrect(x0="2026-01", x1=kpis["month"].max(), fillcolor=ui.GREEN, opacity=.08, line_width=0, annotation_text="test (out-of-sample)")
    figure.update_layout(barmode="overlay")
    left.plotly_chart(ui.style_figure(figure, "Model jauh lebih selektif daripada rules", "per bulan 2021–2026; 2021–2024 = periode latih"),
                      use_container_width=True)
    unit_risk = services.table("unit_risk")
    region = unit_risk[unit_risk["year"] == year].groupby("region").agg(transaksi=("transaksi", "sum"),
                                                                       alert=("alert_tinggi", "sum")).reset_index()
    region["alert_rate"] = region["alert"] / region["transaksi"]
    figure = px.bar(region.sort_values("alert_rate"), x="alert_rate", y="region", orientation="h", color="alert_rate",
                    color_continuous_scale=["#dbe9fb", ui.ORANGE], labels={"alert_rate": "% transaksi alert Tinggi", "region": ""})
    figure.update_layout(coloraxis_showscale=False, xaxis_tickformat=".1%")
    right.plotly_chart(ui.style_figure(figure, "Wilayah dengan tingkat alert tertinggi", f"{year}"), use_container_width=True)
    left, middle, right = st.columns(3)
    communities = services.table("communities").head(5)
    with left:
        st.markdown("**5 komunitas risiko terkuat**")
        st.dataframe(communities[["community_id", "skor_komunitas", "anggota", "unit", "aktif_bulan"]], hide_index=True,
                     use_container_width=True, column_config={"skor_komunitas": st.column_config.ProgressColumn("skor", min_value=0, max_value=100, format="%.0f")})
    with middle:
        metrics = services.table("model_metrics")
        test = metrics[metrics["periode"] == "test 2026"].set_index("model")
        st.markdown("**Kinerja di data uji 2026**")
        st.dataframe(test[["precision", "recall", "PR-AUC", "FP"]].style.format({"precision": "{:.0%}", "recall": "{:.0%}", "PR-AUC": "{:.3f}"}),
                     use_container_width=True)
    with right:
        st.markdown("**Dampak (estimasi per tahun)**")
        if len(impact):
            display = impact.set_index("skenario")[["kerugian dicegah/tahun", "biaya pemeriksaan/tahun", "manfaat bersih/tahun"]].map(ui.rupiah)
            st.dataframe(display, use_container_width=True)
    st.caption("Angka 2021–2024 adalah periode latih (in-sample); evaluasi yang jujur memakai 2026 (test). "
               "Semua skor adalah indikasi, keputusan tetap oleh investigator.")
