"""Performa Model: metrik yang cocok untuk tiap model (classifier, Isolation Forest, graph, rules) + fairness + seleksi."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import confusion_matrix

from ..ui import components as ui
from ..ui import services


def classifier_tab(settings: dict) -> None:
    metrics, test = services.table("model_metrics"), services.table("test_features")
    st.markdown("**Metrik utama: PR-AUC** (paling jujur untuk data imbalanced). Accuracy dilaporkan, tetapi tidak dipakai memilih model.")
    view = metrics.set_index(["periode", "model"])[["accuracy", "precision", "recall", "F1", "F2", "MCC", "ROC-AUC", "PR-AUC", "TP", "FP", "FN", "TN"]]
    st.dataframe(view.style.format({c: "{:.3f}" for c in ["accuracy", "precision", "recall", "F1", "F2", "MCC", "ROC-AUC", "PR-AUC"]}),
                 use_container_width=True)
    left, right = st.columns([1, 1.2])
    with left:
        threshold = st.slider("Threshold (skor minimum) untuk confusion matrix", 5, 95, int(round(settings.get("final_threshold", 0.5) * 100))) / 100
        labels, scores = test["fraud_label"].to_numpy(), test["risk_probability"].to_numpy()
        matrix = confusion_matrix(labels, (scores >= threshold).astype(int), labels=[0, 1])
        text = [[f"TN<br>{matrix[0, 0]:,}", f"FP (alarm palsu)<br>{matrix[0, 1]:,}"], [f"FN (lolos)<br>{matrix[1, 0]:,}", f"TP (tertangkap)<br>{matrix[1, 1]:,}"]]
        figure = go.Figure(go.Heatmap(z=matrix, x=["prediksi normal", "prediksi kolusi"], y=["aktual normal", "aktual kolusi"],
                                      text=text, texttemplate="%{text}", colorscale="Blues", showscale=False))
        figure.update_layout(yaxis=dict(autorange="reversed"))
        precision = matrix[1, 1] / max(matrix[:, 1].sum(), 1)
        recall = matrix[1, 1] / max(matrix[1].sum(), 1)
        st.plotly_chart(ui.style_figure(figure, "Confusion matrix · test 2026", f"precision {ui.percent(precision)} · recall {ui.percent(recall)}",
                                        height=330), use_container_width=True)
    with right:
        curves = services.table("model_curves")
        curve_type = st.radio("Kurva", ["PR", "ROC", "CALIBRATION"], horizontal=True,
                              format_func={"PR": "Precision–Recall", "ROC": "ROC", "CALIBRATION": "Kalibrasi"}.get)
        subset = curves[curves["curve"] == curve_type]
        figure = px.line(subset, x="x", y="y", color="model", labels={"x": {"PR": "recall", "ROC": "false positive rate", "CALIBRATION": "skor prediksi"}[curve_type],
                                                                      "y": {"PR": "precision", "ROC": "true positive rate", "CALIBRATION": "proporsi aktual"}[curve_type]})
        if curve_type != "PR":
            figure.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dot", color=ui.MUTED), name="acak/sempurna")
        st.plotly_chart(ui.style_figure(figure, "Kurva evaluasi · test 2026", height=380), use_container_width=True)
    left, middle, right = st.columns(3)
    with left:
        st.markdown("**Rentang ketidakpastian (bootstrap 95%)**")
        st.dataframe(services.table("bootstrap_ci").set_index("metrik").style.format("{:.3f}"), use_container_width=True)
    with middle:
        monthly = services.table("monthly_performance")
        figure = px.line(monthly.melt(id_vars="month", value_vars=["precision", "recall", "PR-AUC"]), x="month", y="value", color="variable",
                         markers=True, labels={"value": "", "month": "bulan"})
        st.plotly_chart(ui.style_figure(figure, "Stabil per bulan (2026)", height=300), use_container_width=True)
    with right:
        ablation = services.table("ablation")
        figure = px.bar(ablation, x="fitur", y="PR-AUC test", text_auto=".3f", color_discrete_sequence=[ui.BLUE])
        figure.update_layout(yaxis_range=[0.5, 1])
        st.plotly_chart(ui.style_figure(figure, "Ablation: nilai tiap kelompok fitur", "PR-AUC test", height=300), use_container_width=True)
    st.markdown("**Analisis error per pola (test 2026)** — pola kolusi idealnya tertangkap; hard negative (`Normal - …`) idealnya tidak ditandai.")
    st.dataframe(services.table("pattern_analysis").style.format({"label_kolusi": "{:.0%}", "ditandai_model": "{:.0%}", "ditandai_rules": "{:.0%}"}),
                 use_container_width=True, hide_index=True)


def render() -> None:
    ui.page_header("Performa Model", "Setiap model dinilai dengan metrik yang cocok dengan jenis dan kegunaannya")
    settings = services.app_settings()
    tabs = st.tabs(["Classifier (skor transaksi)", "Isolation Forest (radar approver)", "Graph (komunitas)", "Rules (baseline)",
                    "Fairness", "Seleksi & tuning"])
    with tabs[0]:
        classifier_tab(settings)
    with tabs[1]:
        radar = services.table("radar_evaluation")
        st.markdown("**Tanpa label:** kriteria Excess-Mass & stabilitas antar-seed (dipakai untuk tuning). **Dengan label (hanya mengukur):** "
                    "apakah approver yang terbukti menyetujui kasus kolusi berada di puncak radar?")
        st.dataframe(radar.set_index(["periode", "radar"]).style.format("{:.3f}"), use_container_width=True)
        left, right = st.columns(2)
        recent = radar[radar["periode"] == "2025–2026"]
        figure = px.bar(recent, x="radar", y="precision@5%", text_auto=".0%", color_discrete_sequence=[ui.PURPLE])
        figure.add_hline(y=recent["acak (prevalensi)"].iloc[0], line_dash="dot", annotation_text="acak")
        left.plotly_chart(ui.style_figure(figure, "Precision di 5% teratas (2025–2026)", height=340), use_container_width=True)
        stability = services.table("iforest_stability")
        right.markdown("**Stabilitas & Excess-Mass**")
        right.dataframe(stability, hide_index=True, use_container_width=True)
        complementarity = services.table("if_classifier_complementarity")
        right.markdown(f"**Saling melengkapi dengan classifier** (Spearman {settings.get('spearman_if_classifier', 0):.2f})")
        right.dataframe(complementarity.style.format({"terbukti_menyetujui_kolusi": "{:.0%}"}), hide_index=True, use_container_width=True)
    with tabs[2]:
        graph = services.table("graph_evaluation")
        figure = go.Figure()
        for column, color in [("modularity", ui.BLUE), ("persistensi", ui.GREEN), ("precision", ui.ORANGE), ("recall", ui.PURPLE)]:
            figure.add_scatter(x=graph["bulan"], y=graph[column], name=column, line=dict(color=color, width=2))
        st.plotly_chart(ui.style_figure(figure, "Kualitas struktur & daya peringatan dini komunitas",
                                        "modularity & persistensi tanpa label; precision/recall = anggota yang terlibat kasus terbukti ≤ 3 bulan"),
                        use_container_width=True)
        recent = graph[graph["bulan"].astype(str) >= "2025-01"]
        ui.kpi_row([("Modularity", f"{recent['modularity'].mean():.2f}", "rata-rata 2025–2026", ui.BLUE),
                    ("Persistensi", f"{recent['persistensi'].mean():.2f}", "Jaccard bulan ke bulan", ui.GREEN),
                    ("Precision peringatan dini", ui.percent(recent["precision"].mean()), "", ui.ORANGE),
                    ("Lift", f"{recent['lift'].mean():.1f}×", "dibanding pegawai acak", ui.PURPLE)])
    with tabs[3]:
        st.dataframe(services.table("rule_definitions").style.format({"coverage": "{:.1%}", "precision": "{:.0%}", "recall": "{:.0%}"}),
                     hide_index=True, use_container_width=True)
        st.caption("Rules dipakai sebagai baseline dan alasan. Setiap rule punya kembaran sah, sehingga rules sendirian berisik.")
    with tabs[4]:
        fairness = services.table("fairness")
        dimension = st.selectbox("Dimensi", fairness["dimensi"].unique())
        subset = fairness[fairness["dimensi"] == dimension]
        figure = go.Figure()
        figure.add_bar(x=subset["segmen"], y=subset["FPR rules"], name="FPR rules", marker_color=ui.MUTED)
        figure.add_bar(x=subset["segmen"], y=subset["FPR model"], name="FPR model", marker_color=ui.ORANGE)
        figure.update_layout(barmode="group", yaxis_tickformat=".0%")
        st.plotly_chart(ui.style_figure(figure, f"False positive rate per {dimension}", "rendah & merata = adil bagi pegawai/debitur yang bekerja sah"),
                        use_container_width=True)
        st.dataframe(subset, hide_index=True, use_container_width=True)
    with tabs[5]:
        st.markdown(f"**Seleksi algoritma:** {settings.get('screening_engine', '-')} · **Tuning:** {settings.get('tuning_engine', '-')}")
        leaderboard = services.table("screening_leaderboard")
        st.dataframe(leaderboard, hide_index=True, use_container_width=True)
        trials = services.table("tuning_trials")
        if len(trials) and "value" in trials:
            trials = trials.assign(percobaan=trials.groupby("model").cumcount())
            figure = px.scatter(trials, x="percobaan", y="value", color="model", labels={"value": "PR-AUC CV deret waktu"})
            st.plotly_chart(ui.style_figure(figure, "Riwayat percobaan tuning"), use_container_width=True)
