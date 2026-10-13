"""Fungsi bantu yang dipakai beberapa halaman."""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from ..ui import components as ui
from ..ui import services

RED_FLAG_COLUMNS = list(ui.RED_FLAG_LABELS)
DECISIONS = ["Tindak lanjut / eskalasi", "Bukan kolusi (alarm palsu)", "Perlu data tambahan"]


def latest_decisions() -> pd.DataFrame:
    """Keputusan investigator terakhir per request_id (index = request_id)."""
    reviews = services.get_store().reviews()
    if reviews.empty:
        return pd.DataFrame(columns=["decision", "username", "reviewed_at", "note"]).rename_axis("request_id")
    reviews = reviews.sort_values("reviewed_at")
    return reviews.groupby("request_id").last()[["decision", "username", "reviewed_at", "note"]]


def red_flag_summary(frame: pd.DataFrame) -> pd.DataFrame:
    high = frame[frame["risk_band"] == "Tinggi"]
    return pd.DataFrame({"rule": [ui.RED_FLAG_LABELS[c] for c in RED_FLAG_COLUMNS],
                         "% semua transaksi": [frame[c].mean() if len(frame) else 0 for c in RED_FLAG_COLUMNS],
                         "% alert Tinggi": [high[c].mean() if len(high) else 0 for c in RED_FLAG_COLUMNS]})


def flags_text(row: pd.Series) -> str:
    return ", ".join(ui.RED_FLAG_LABELS[c].split(" (")[0] for c in RED_FLAG_COLUMNS if row.get(c) == 1)


def show_guard_checks(checks: list) -> None:
    """Tabel status guardrail (lapis, pemeriksaan, hasil, detail)."""
    if not checks:
        return
    frame = pd.DataFrame(checks)
    frame["hasil"] = frame["passed"].map({True: "✅ lolos", False: "⛔ gagal/ditahan"})
    st.dataframe(frame[["layer", "check", "hasil", "detail"]].rename(columns={"layer": "lapis", "check": "pemeriksaan"}),
                 hide_index=True, use_container_width=True)


def parse_json_column(value) -> list:
    try:
        return json.loads(value) if isinstance(value, str) else []
    except json.JSONDecodeError:
        return []
