"""Investigator · Riwayat Keputusan: jejak audit keputusan dan memo yang disetujui."""
from __future__ import annotations

import plotly.express as px
import streamlit as st

from ..ui import components as ui
from ..ui import services


def render() -> None:
    ui.page_header("Riwayat Keputusan", "Jejak audit keputusan investigator dan memo hasil agen yang telah disetujui")
    store = services.get_store()
    reviews, actions = store.reviews(), store.agent_actions()
    user = services.current_user()
    only_mine = st.toggle("Hanya keputusan saya", value=False)
    if only_mine and len(reviews):
        reviews = reviews[reviews["username"] == user["username"]]
    ui.kpi_row([("Keputusan tercatat", ui.number(len(reviews)), f"sumber: {store.backend}", ui.BLUE),
                ("Tindak lanjut", ui.number((reviews["decision"].str.startswith("Tindak")).sum()) if len(reviews) else "0", "", ui.ORANGE),
                ("Alarm palsu", ui.number((reviews["decision"].str.startswith("Bukan")).sum()) if len(reviews) else "0",
                 "umpan balik untuk pelatihan ulang", ui.GREEN),
                ("Memo disetujui", ui.number((actions["status"] == "DISETUJUI").sum()) if len(actions) else "0", "", ui.PURPLE)])
    if reviews.empty:
        st.info("Belum ada keputusan. Catat keputusan dari halaman Monitoring Alert.")
    else:
        left, right = st.columns([1, 1.4])
        counts = reviews["decision"].value_counts().rename_axis("keputusan").reset_index(name="jumlah")
        left.plotly_chart(ui.style_figure(px.pie(counts, names="keputusan", values="jumlah", hole=.55), "Komposisi keputusan", height=300),
                          use_container_width=True)
        right.dataframe(reviews.sort_values("reviewed_at", ascending=False), hide_index=True, use_container_width=True, height=300)
    if len(actions):
        st.markdown("**Memo & aksi agen**")
        st.dataframe(actions.sort_values("created_at", ascending=False), hide_index=True, use_container_width=True)
