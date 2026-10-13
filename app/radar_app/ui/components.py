"""Komponen tampilan bersama: warna, kartu KPI, badge, format angka, template grafik Plotly (light mode)."""
from __future__ import annotations

import html

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

BLUE, ORANGE, GREEN, YELLOW, PURPLE, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#7b5cd6", "#c2362b"
INK, INK_SECONDARY, MUTED, GRID, SURFACE = "#1f2328", "#52514e", "#8a8984", "#e6e5e1", "#f6f7f9"
BAND_COLORS = {"Tinggi": ORANGE, "Sedang": YELLOW, "Rendah": "#9ec5f0"}
ANOMALY_COLORS = {"Perlu dilihat": ORANGE, "Pantau": YELLOW, "Normal": "#9ec5f0"}
GROUP_COLORS = {"Transaksi": BLUE, "Riwayat": PURPLE, "Rules": YELLOW, "Graph": ORANGE}
RED_FLAG_LABELS = {"rf_near_limit": "Near limit 95–99%", "rf_after_hours": "Luar jam kerja (19–07/akhir pekan)",
                   "rf_fast_approval": "Approval kilat < 20 menit", "rf_debtor_repeat": "Debitur sama ≥ 2×/tahun",
                   "rf_pair_dependency": "Ketergantungan approver"}

pio.templates["radar"] = go.layout.Template(layout=go.Layout(
    font=dict(family="Inter, Segoe UI, sans-serif", size=12, color=INK), paper_bgcolor="white", plot_bgcolor="white",
    colorway=[BLUE, ORANGE, GREEN, YELLOW, PURPLE, MUTED], margin=dict(l=10, r=10, t=48, b=10),
    xaxis=dict(gridcolor=GRID, zerolinecolor=GRID), yaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0), hoverlabel=dict(bgcolor="white")))
pio.templates.default = "radar"

CSS = f"""
<style>
.block-container {{padding-top: 1.6rem; padding-bottom: 2rem;}}
.kpi-card {{background: white; border: 1px solid {GRID}; border-radius: 12px; padding: 14px 16px; height: 100%;
           box-shadow: 0 1px 2px rgba(0,0,0,.04);}}
.kpi-label {{font-size: .78rem; color: {INK_SECONDARY}; text-transform: uppercase; letter-spacing: .03em;}}
.kpi-value {{font-size: 1.55rem; font-weight: 700; color: {INK}; line-height: 1.25; margin-top: 2px;}}
.kpi-sub {{font-size: .78rem; color: {MUTED}; margin-top: 2px;}}
.kpi-accent {{border-left: 4px solid var(--accent);}}
.badge {{display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: .78rem; font-weight: 600;
        color: white; margin-right: 4px;}}
.chip {{display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: .78rem; background: {SURFACE};
       border: 1px solid {GRID}; color: {INK}; margin: 2px 4px 2px 0;}}
.page-subtitle {{color: {INK_SECONDARY}; margin-top: -10px; margin-bottom: 14px;}}
.evidence li {{margin-bottom: 4px;}}
.guard-ok {{color: {GREEN}; font-weight: 600;}} .guard-fail {{color: {RED}; font-weight: 600;}}
.small-note {{font-size: .8rem; color: {MUTED};}}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def page_header(title: str, subtitle: str = "") -> None:
    st.title(title)
    if subtitle:
        st.markdown(f"<div class='page-subtitle'>{html.escape(subtitle)}</div>", unsafe_allow_html=True)


def kpi_card(label: str, value: str, sub: str = "", accent: str = BLUE) -> str:
    return (f"<div class='kpi-card kpi-accent' style='--accent:{accent}'><div class='kpi-label'>{html.escape(label)}</div>"
            f"<div class='kpi-value'>{html.escape(str(value))}</div><div class='kpi-sub'>{html.escape(sub)}</div></div>")


def kpi_row(cards: list[tuple]) -> None:
    columns = st.columns(len(cards))
    for column, card in zip(columns, cards):
        column.markdown(kpi_card(*card), unsafe_allow_html=True)


def badge(text: str, color: str) -> str:
    return f"<span class='badge' style='background:{color}'>{html.escape(str(text))}</span>"


def band_badge(band: str) -> str:
    return badge(band, BAND_COLORS.get(band, MUTED))


def chips(items: list[str]) -> str:
    return "".join(f"<span class='chip'>{html.escape(str(i))}</span>" for i in items if i)


def rupiah(value, short: bool = True) -> str:
    if value is None or pd.isna(value):
        return "-"
    if short:
        for size, suffix in [(1e12, "T"), (1e9, "M"), (1e6, "jt")]:
            if abs(value) >= size:
                return f"Rp{value / size:,.1f} {suffix}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"Rp{value:,.0f}".replace(",", ".")


def number(value, decimals: int = 0) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{value:,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def percent(value, decimals: int = 0) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{value * 100:.{decimals}f}%".replace(".", ",")


def style_figure(fig, title: str = "", subtitle: str = "", height: int = 360):
    text = f"<b>{html.escape(title)}</b>" + (f"<br><span style='font-size:11px;color:{INK_SECONDARY}'>{html.escape(subtitle)}</span>" if subtitle else "")
    fig.update_layout(title=dict(text=text, x=0, xanchor="left", font=dict(size=14)), height=height)
    return fig


def empty_state(message: str) -> None:
    st.info(message)
