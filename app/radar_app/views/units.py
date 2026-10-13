"""Manajemen · Risiko Unit & Wilayah: treemap wilayah → unit, tren, dan komposisi red flag (agregat)."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from ..ui import components as ui
from ..ui import services
from .common import red_flag_summary


def render() -> None:
    ui.page_header("Risiko Unit & Wilayah", "Di mana risiko terkonsentrasi, untuk alokasi pengawasan dan kapasitas investigasi")
    transactions = services.table("transactions_scored")
    unit_risk = services.table("unit_risk")
    year = st.sidebar.selectbox("Tahun", sorted(unit_risk["year"].unique().astype(int), reverse=True), key="unit_year")
    frame = unit_risk[unit_risk["year"] == year].copy()
    frame["region"] = frame["region"].fillna("Lainnya")
    left, right = st.columns([1.3, 1])
    figure = px.treemap(frame, path=["region", "requester_unit"], values="transaksi", color="alert_rate",
                        color_continuous_scale=["#eef4fc", "#f3c35a", ui.ORANGE], hover_data={"alert_tinggi": True, "skor_rata": ":.1f"})
    figure.update_layout(coloraxis_colorbar=dict(title="% alert", tickformat=".0%"))
    left.plotly_chart(ui.style_figure(figure, "Peta risiko: wilayah → unit", "ukuran = volume transaksi · warna = % alert Tinggi", height=520),
                      use_container_width=True)
    top = frame.sort_values("alert_rate", ascending=False).head(15)
    figure = px.bar(top.iloc[::-1], x="alert_rate", y="requester_unit", orientation="h", color_discrete_sequence=[ui.ORANGE],
                    hover_data=["transaksi", "alert_tinggi", "region"], labels={"alert_rate": "% alert Tinggi", "requester_unit": ""})
    figure.update_layout(xaxis_tickformat=".1%")
    right.plotly_chart(ui.style_figure(figure, "15 unit dengan % alert tertinggi", f"{year}", height=520), use_container_width=True)
    unit = st.selectbox("Lihat unit", top["requester_unit"].tolist() + sorted(set(frame["requester_unit"]) - set(top["requester_unit"])),
                        key="unit_pick")
    unit_tx = transactions[(transactions["requester_unit"] == unit) & (transactions["event_time"].dt.year == year)]
    ui.kpi_row([("Transaksi", ui.number(len(unit_tx)), unit, ui.BLUE),
                ("Alert Tinggi", ui.number((unit_tx["risk_band"] == "Tinggi").sum()), ui.percent((unit_tx["risk_band"] == "Tinggi").mean(), 1), ui.ORANGE),
                ("Nominal alert Tinggi", ui.rupiah(unit_tx.loc[unit_tx["risk_band"] == "Tinggi", "amount"].sum()), "", ui.RED),
                ("Approver terlibat", ui.number(unit_tx.loc[unit_tx["risk_band"] == "Tinggi", "approver_id"].nunique()), "", ui.PURPLE)])
    left, right = st.columns(2)
    monthly = unit_tx.groupby(unit_tx["event_time"].dt.to_period("M").astype(str))["risk_band"].apply(lambda s: (s == "Tinggi").mean()).reset_index()
    figure = px.line(monthly, x="event_time", y="risk_band", markers=True, labels={"event_time": "bulan", "risk_band": "% alert Tinggi"})
    figure.update_layout(yaxis_tickformat=".0%")
    left.plotly_chart(ui.style_figure(figure, f"Tren % alert Tinggi · {unit}"), use_container_width=True)
    flags = red_flag_summary(unit_tx)
    figure = px.bar(flags, x="% alert Tinggi", y="rule", orientation="h", color_discrete_sequence=[ui.ORANGE])
    figure.update_layout(xaxis_tickformat=".0%")
    right.plotly_chart(ui.style_figure(figure, "Red flag pada alert Tinggi unit ini", "membantu menentukan fokus pengawasan"),
                       use_container_width=True)
    st.caption("Halaman ini agregat; detail transaksi dan nama pegawai hanya tersedia untuk role investigator.")
