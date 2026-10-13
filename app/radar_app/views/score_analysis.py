"""Analisis Skor: distribusi, hubungan dengan rules, kalibrasi, dan SHAP global/dependence."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ..ui import components as ui
from ..ui import services


def render() -> None:
    ui.page_header("Analisis Skor", "Bagaimana skor risiko terbentuk, seberapa bisa dipercaya, dan fitur apa yang menentukannya")
    transactions = services.table("transactions_scored")
    settings = services.app_settings()
    period = services.period_filter(transactions, "score", default_year=2026)
    threshold = settings.get("final_threshold", 0.5) * 100
    ui.kpi_row([("Threshold Tinggi", f"≥ {threshold:.0f}", "dipilih di validasi 2025 (F1 terbaik)", ui.ORANGE),
                ("Threshold Sedang", f"≥ {threshold / 2:.0f}", "separuh threshold Tinggi", ui.YELLOW),
                ("Model", settings.get("model_name", "-"), "kalibrasi isotonic", ui.BLUE),
                ("Metode XAI", settings.get("shap_method", "SHAP").split(" (")[0], "alasan per alert", ui.PURPLE)])
    tab_distribution, tab_calibration, tab_shap, tab_dependence = st.tabs(
        ["Distribusi skor", "Kalibrasi (bisa dipercaya?)", "SHAP global", "Pengaruh fitur (dependence)"])
    with tab_distribution:
        left, right = st.columns(2)
        figure = px.histogram(period, x="risk_score", color="risk_band", nbins=50, color_discrete_map=ui.BAND_COLORS,
                              log_y=True, labels={"risk_score": "skor risiko", "risk_band": "kategori"})
        figure.add_vline(x=threshold, line_dash="dash", line_color=ui.ORANGE, annotation_text="threshold Tinggi")
        left.plotly_chart(ui.style_figure(figure, "Sebagian besar transaksi berskor rendah", "jumlah transaksi per skor (skala log)"),
                          use_container_width=True)
        figure = px.box(period.assign(rule=period["rule_score"].clip(upper=3).map({0: "0", 1: "1", 2: "2", 3: "≥3"})),
                        x="rule", y="risk_score", color_discrete_sequence=[ui.BLUE], points=False,
                        category_orders={"rule": ["0", "1", "2", "≥3"]}, labels={"rule": "jumlah rule terpenuhi", "risk_score": "skor"})
        right.plotly_chart(ui.style_figure(figure, "Skor naik seiring jumlah rule, tapi tidak identik",
                                           "model juga menimbang graph & riwayat; 0 rule pun bisa tinggi bila jaringannya berisiko"),
                           use_container_width=True)
    with tab_calibration:
        curves = services.table("model_curves")
        calibration = curves[curves["curve"] == "CALIBRATION"]
        figure = go.Figure()
        figure.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dot", color=ui.MUTED), name="sempurna"))
        for name, color in [("mentah", ui.BLUE), ("terkalibrasi", ui.ORANGE)]:
            subset = calibration[calibration["model"] == name]
            figure.add_trace(go.Scatter(x=subset["x"], y=subset["y"], mode="lines+markers", name=name, line=dict(color=color, width=3)))
        figure.update_layout(xaxis_title="skor prediksi", yaxis_title="proporsi kolusi aktual (test 2026)")
        st.plotly_chart(ui.style_figure(figure, "Reliability diagram: skor ≈ peluang", "titik dekat diagonal = skor bisa dibaca sebagai peluang"),
                        use_container_width=True)
        st.markdown("**Cara membaca:** skor 80 berarti dari kasus serupa yang direview audit, ±80% terbukti kolusi. Pada seluruh "
                    "transaksi (bukan hanya yang direview) proporsi kolusinya jauh lebih kecil, sehingga skor dibaca sebagai "
                    "**prioritas relatif**; precision populasi ditampilkan sebagai rentang di Ringkasan Eksekutif.")
    shap_global = services.table("shap_global")
    beeswarm = services.table("shap_beeswarm")
    with tab_shap:
        left, right = st.columns([0.9, 1.2])
        top = shap_global.head(15).iloc[::-1]
        figure = go.Figure(go.Bar(x=top["mean |SHAP|"], y=top["feature"], orientation="h",
                                  marker_color=[ui.GROUP_COLORS.get(g, ui.BLUE) for g in top["kelompok"]],
                                  hovertext=top["arti"]))
        left.plotly_chart(ui.style_figure(figure, "Fitur paling berpengaruh", "rata-rata |SHAP|; warna = kelompok fitur", height=480),
                          use_container_width=True)
        share = shap_global.groupby("kelompok")["mean |SHAP|"].sum()
        left.plotly_chart(ui.style_figure(px.pie(values=share.values, names=share.index, hole=.55, color=share.index,
                                                 color_discrete_map=ui.GROUP_COLORS), "Porsi pengaruh per kelompok", height=280),
                          use_container_width=True)
        features = top["feature"].tolist()[::-1]
        swarm = beeswarm[beeswarm["feature"].isin(features)].copy()
        swarm["persentil nilai"] = swarm.groupby("feature")["feature_value"].rank(pct=True)
        figure = px.scatter(swarm, x="shap_value", y="feature", color="persentil nilai", category_orders={"feature": features},
                            color_continuous_scale="RdBu_r", hover_data=["feature_value", "fraud_label"], opacity=.6)
        figure.add_vline(x=0, line_color=ui.MUTED)
        right.plotly_chart(ui.style_figure(figure, "Beeswarm: arah pengaruh", "merah = nilai fitur tinggi; kanan = menaikkan risiko",
                                           height=780), use_container_width=True)
    with tab_dependence:
        feature = st.selectbox("Pilih fitur", shap_global["feature"].head(15).tolist(), key="dependence_feature")
        subset = beeswarm[beeswarm["feature"] == feature]
        figure = px.scatter(subset, x="feature_value", y="shap_value", color=subset["fraud_label"].map({0: "normal", 1: "kolusi"}),
                            color_discrete_map={"normal": ui.BLUE, "kolusi": ui.ORANGE}, opacity=.6,
                            labels={"feature_value": feature, "shap_value": "kontribusi SHAP", "color": "label audit"})
        figure.add_hline(y=0, line_color=ui.MUTED)
        description = shap_global.set_index("feature").loc[feature, "arti"] if feature in set(shap_global["feature"]) else ""
        st.plotly_chart(ui.style_figure(figure, f"Pengaruh '{feature}' terhadap skor", description), use_container_width=True)
        st.caption("Pola yang masuk akal secara bisnis (mis. risiko melonjak di 95–99% limit atau jam malam) memperkuat kepercayaan pada "
                   "model. Pola yang janggal adalah tanda bias yang harus diperiksa.")
