"""Investigator · Ringkasan: kondisi antrean kerja dan sinyal risiko periode terpilih."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ..ui import components as ui
from ..ui import services
from .common import latest_decisions, red_flag_summary


def render() -> None:
    ui.page_header("Ringkasan Investigasi", "Apa yang perlu ditangani hari ini, dan dari mana risikonya datang")
    transactions = services.table("transactions_scored")
    if transactions.empty:
        ui.empty_state("Data belum tersedia. Jalankan notebook lalu muat ke PostgreSQL (lihat README).")
        return
    period = services.period_filter(transactions, "inv_overview", default_year=2026)
    decisions = latest_decisions()
    high = period[period["risk_band"] == "Tinggi"]
    open_high = high[~high["request_id"].isin(decisions.index)]
    anomaly = services.table("approver_anomaly")
    latest_month = anomaly["month"].max() if len(anomaly) else None
    flagged_approvers = anomaly[(anomaly["month"] == latest_month) & (anomaly["if_score"] >= 95)] if latest_month else anomaly.head(0)
    ui.kpi_row([
        ("Transaksi dinilai", ui.number(len(period)), st.session_state.get("inv_overview_period_label", ""), ui.BLUE),
        ("Alert Tinggi", ui.number(len(high)), f"nominal {ui.rupiah(high['amount'].sum())}", ui.ORANGE),
        ("Belum diputus", ui.number(len(open_high)), "alert Tinggi tanpa keputusan investigator", ui.RED),
        ("Alert Sedang", ui.number((period["risk_band"] == "Sedang").sum()), "pantau / sampling", ui.YELLOW),
        ("Approver perlu dilihat", ui.number(len(flagged_approvers)), f"skor anomali ≥ 95 · {latest_month}", ui.PURPLE),
        ("Komunitas risiko", ui.number(len(services.table("communities"))), "snapshot graph terbaru", ui.GREEN)])
    st.write("")
    left, right = st.columns([1.35, 1])
    monthly = (period.groupby([period["event_time"].dt.to_period("M").astype(str).rename("bulan"), "risk_band"])
               .size().rename("jumlah").reset_index())
    monthly = monthly[monthly["risk_band"] != "Rendah"]
    figure = px.bar(monthly, x="bulan", y="jumlah", color="risk_band", color_discrete_map=ui.BAND_COLORS,
                    category_orders={"risk_band": ["Tinggi", "Sedang"]}, labels={"risk_band": "kategori"})
    left.plotly_chart(ui.style_figure(figure, "Alert per bulan", "jumlah alert Tinggi & Sedang"), use_container_width=True)
    flags = red_flag_summary(period)
    figure = go.Figure()
    figure.add_bar(y=flags["rule"], x=flags["% semua transaksi"] * 100, orientation="h", name="semua transaksi", marker_color=ui.MUTED)
    figure.add_bar(y=flags["rule"], x=flags["% alert Tinggi"] * 100, orientation="h", name="alert Tinggi", marker_color=ui.ORANGE)
    figure.update_layout(barmode="group", xaxis_title="% transaksi yang memenuhi rule")
    right.plotly_chart(ui.style_figure(figure, "Red flag pada alert Tinggi", "dibanding seluruh transaksi"), use_container_width=True)
    left, right = st.columns([1, 1.35])
    by_unit = (high.groupby("requester_unit").agg(alert=("request_id", "size"), nominal=("amount", "sum"))
               .sort_values("alert", ascending=True).tail(10).reset_index())
    figure = px.bar(by_unit, x="alert", y="requester_unit", orientation="h", color_discrete_sequence=[ui.ORANGE],
                    hover_data={"nominal": ":,.0f"}, labels={"requester_unit": "", "alert": "alert Tinggi"})
    left.plotly_chart(ui.style_figure(figure, "10 unit dengan alert Tinggi terbanyak", "unit requester"), use_container_width=True)
    with right:
        st.markdown("**Antrean prioritas: alert Tinggi terbaru yang belum diputus**")
        names = services.name_map()
        queue = open_high.sort_values(["risk_score", "event_time"], ascending=False).head(12).assign(
            requester=lambda df: df["requester_id"].map(names), approver=lambda df: df["approver_id"].map(names),
            nominal=lambda df: df["amount"].map(ui.rupiah))
        st.dataframe(queue[["request_id", "event_time", "requester", "approver", "requester_unit", "nominal", "risk_score"]],
                     hide_index=True, use_container_width=True, height=380,
                     column_config={"risk_score": st.column_config.ProgressColumn("skor", min_value=0, max_value=100, format="%.0f"),
                                    "event_time": st.column_config.DatetimeColumn("waktu", format="DD MMM YYYY HH:mm"),
                                    "requester_unit": "unit"})
        st.caption("Buka halaman **Monitoring Alert** untuk filter lengkap, detail, dan form keputusan.")
    if len(decisions):
        st.markdown("**Keputusan investigator pada periode ini**")
        decided = decisions[decisions.index.isin(period["request_id"])]["decision"].value_counts().rename_axis("keputusan").reset_index(name="jumlah")
        st.plotly_chart(ui.style_figure(px.pie(decided, names="keputusan", values="jumlah", hole=.55,
                                               color_discrete_sequence=[ui.ORANGE, ui.GREEN, ui.YELLOW]), "Komposisi keputusan", height=300),
                        use_container_width=True)
