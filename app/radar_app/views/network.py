"""Jaringan Risiko: komunitas dari graph analytics, anggota (nama, jabatan, unit), dan graph interaktif."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

from ..ui import components as ui
from ..ui import services
from ..ui.network import LEGEND_HTML, build_graph, plotly_network, pyvis_html


def render() -> None:
    user = services.current_user()
    ui.page_header("Jaringan Risiko", "Lingkaran pegawai yang hubungannya menyimpang (graph analytics, tanpa label, 12 bulan terakhir)")
    communities, members = services.table("communities"), services.table("community_members")
    edges, history = services.table("community_edges"), services.table("community_history")
    anomaly = services.table("approver_anomaly")
    if communities.empty:
        ui.empty_state("Belum ada komunitas risiko.")
        return
    with st.sidebar:
        st.markdown("**Filter komunitas**")
        priorities = st.multiselect("Prioritas", ["Tinggi", "Sedang", "Rendah"], default=["Tinggi", "Sedang", "Rendah"], key="net_prio")
        unit_query = st.text_input("Unit mengandung", key="net_unit")
        min_members = st.slider("Anggota minimal", 3, int(max(communities["anggota"].max(), 3)), 3, key="net_min")
    frame = communities[communities["prioritas"].isin(priorities) & (communities["anggota"] >= min_members)]
    if unit_query:
        frame = frame[frame["unit"].str.contains(unit_query, case=False, na=False)]
    ui.kpi_row([("Komunitas risiko", ui.number(len(frame)), "≥ 3 orang, snapshot terbaru", ui.ORANGE),
                ("Pegawai terlibat", ui.number(frame["anggota"].sum()), "", ui.BLUE),
                ("Prioritas Tinggi", ui.number((frame["prioritas"] == "Tinggi").sum()), "kuartil skor teratas", ui.RED),
                ("Rata-rata aktif", f"{frame['aktif_bulan'].mean():.1f} bln" if len(frame) else "-", "dari 12 snapshot terakhir", ui.GREEN)])
    st.write("")
    left, right = st.columns([1.1, 1])
    with left:
        st.markdown("**Daftar komunitas** (klik satu baris)")
        selection = st.dataframe(
            frame[["community_id", "skor_komunitas", "prioritas", "anggota", "approver_utama", "unit", "edge_red_flag",
                   "edge_ketergantungan", "edge_debitur_bersama", "aktif_bulan"]],
            hide_index=True, use_container_width=True, height=380, on_select="rerun", selection_mode="single-row", key="net_table",
            column_config={"skor_komunitas": st.column_config.ProgressColumn("skor", min_value=0, max_value=100, format="%.0f"),
                           "community_id": "ID", "edge_red_flag": "red flag", "edge_ketergantungan": "ketergantungan",
                           "edge_debitur_bersama": "debitur bersama", "aktif_bulan": "aktif (bln)"})
    with right:
        figure = px.scatter(frame, x="aktif_bulan", y="skor_komunitas", size="anggota", color="prioritas",
                            color_discrete_map={"Tinggi": ui.ORANGE, "Sedang": ui.YELLOW, "Rendah": "#9ec5f0"},
                            hover_data=["community_id", "unit", "approver_utama"],
                            labels={"aktif_bulan": "persistensi (bulan aktif dari 12)", "skor_komunitas": "skor komunitas"})
        st.plotly_chart(ui.style_figure(figure, "Kuat & persisten = paling patut diperiksa", "ukuran = jumlah anggota"),
                        use_container_width=True)
    rows = getattr(getattr(selection, "selection", None), "rows", []) or []
    community_id = int(frame.iloc[rows[0]]["community_id"]) if rows else st.selectbox(
        "…atau pilih ID komunitas", frame["community_id"].tolist(), key="net_pick")
    info = communities[communities["community_id"] == community_id].iloc[0]
    group = members[members["community_id"] == community_id].copy()
    latest_month = anomaly["month"].max() if len(anomaly) else None
    latest_scores = anomaly[anomaly["month"] == latest_month].set_index("approver_id")["if_score"] if latest_month else pd.Series(dtype=float)
    group["skor anomali (IF)"] = group["employee_id"].map(latest_scores)
    st.divider()
    st.markdown(f"### Komunitas {community_id} &nbsp; {ui.badge('prioritas ' + info['prioritas'], ui.ORANGE if info['prioritas'] == 'Tinggi' else ui.YELLOW)}",
                unsafe_allow_html=True)
    ui.kpi_row([("Skor komunitas", f"{info['skor_komunitas']:.0f}/100", "relatif thd komunitas terkuat 12 bln", ui.ORANGE),
                ("Anggota", ui.number(info["anggota"]), f"{info['n_approver']} approver · {info['n_requester']} requester", ui.BLUE),
                ("Hubungan berisiko", ui.number(info["edge_red_flag"] + info["edge_ketergantungan"]),
                 f"{info['edge_red_flag']} red flag · {info['edge_ketergantungan']} ketergantungan", ui.PURPLE),
                ("Debitur bersama", ui.number(info["edge_debitur_bersama"]), "pasangan requester berbagi debitur", ui.GREEN),
                ("Persistensi", f"{info['aktif_bulan']:.0f}/12 bln", "anggota muncul di komunitas risiko", ui.YELLOW)])
    graph_column, members_column = st.columns([1.5, 1])
    with graph_column:
        include_context = st.toggle("Tampilkan juga hubungan normal anggota (12 bulan)", value=False, key="net_context")
        community_edges = edges[(edges["community_a"] == community_id) & (edges["community_b"] == community_id)]
        frame_edges = community_edges[["node_a", "node_b", "weight", "kind", "redflag", "dependency", "n_tx", "z", "shared"]]
        if include_context:
            pairs = services.table("pair_network")
            people = set(group["employee_id"])
            normal = pairs[(pairs["requester_id"].isin(people) | pairs["approver_id"].isin(people))
                           & ~pairs["risky"].astype(str).str.lower().eq("true")].nlargest(25, "n")
            frame_edges = pd.concat([frame_edges, pd.DataFrame({"node_a": normal["requester_id"], "node_b": normal["approver_id"],
                                                                 "weight": 0.5, "kind": "RA", "redflag": False, "dependency": False,
                                                                 "n_tx": normal["n"], "z": normal["z"]})], ignore_index=True)
        graph = build_graph(frame_edges, services.table("employees"), highlight=set(group["employee_id"]),
                            node_scores=latest_scores.to_dict())
        st.markdown(LEGEND_HTML, unsafe_allow_html=True)
        html = pyvis_html(graph, height=540)
        if html:
            components.html(html, height=560)
        else:
            st.plotly_chart(plotly_network(graph, f"Komunitas {community_id}"), use_container_width=True)
    with members_column:
        st.markdown("**Anggota komunitas**")
        st.dataframe(group[["employee_name", "employee_id", "role", "position", "unit", "pagerank", "skor anomali (IF)", "aktif_bulan"]],
                     hide_index=True, use_container_width=True, height=300,
                     column_config={"employee_name": "nama", "pagerank": st.column_config.NumberColumn("sentralitas", format="%.2f"),
                                    "skor anomali (IF)": st.column_config.NumberColumn(format="%.0f"), "aktif_bulan": "aktif (bln)"})
        presence = history[history["employee_id"].isin(group["employee_id"])].groupby("month")["employee_id"].nunique().reset_index()
        figure = px.line(presence, x="month", y="employee_id", markers=True, labels={"employee_id": "anggota terdeteksi", "month": "bulan"})
        st.plotly_chart(ui.style_figure(figure, "Berapa anggota muncul tiap bulan", "12 snapshot terakhir", height=250),
                        use_container_width=True)
    if user["role"] == "investigator" and st.button("Minta agen merangkum komunitas ini", key=f"net_agent_{community_id}"):
        with st.spinner("Agen membaca komunitas…"):
            result = services.get_agent().run(f"Ringkas komunitas {community_id} beserta anggotanya", "investigator", user["username"])
        st.markdown(result["final_answer"])
    st.caption("Komunitas adalah pola statistik dari hubungan transaksi, bukan bukti kolusi. Gunakan sebagai prioritas pemeriksaan.")
