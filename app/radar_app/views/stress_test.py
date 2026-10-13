"""Manajemen · Stress Testing: kapasitas & threshold, pelaku beradaptasi, kegagalan graph, prevalensi, drift."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ..ui import components as ui
from ..ui import services

RED_FLAGS = list(ui.RED_FLAG_LABELS)
GRAPH_FEATURES = ["g_pair_weight", "g_appr_wdeg", "g_debtor_n_requesters", "g_req_shared_debtors", "g_risk_pair_z",
                  "g_risk_pair_weight", "g_risk_comm_score", "g_risk_comm_size", "g_same_comm", "g_appr_risk_pagerank"]


def apply_evasion(frame: pd.DataFrame, hide_near_limit: bool, hide_night: bool, hide_fast: bool, share: float, seed: int = 42):
    """Sebagian pelaku (share) menghindari rule yang diketahui; pola hubungan (graph) tetap."""
    rng = np.random.default_rng(seed)
    adapted = frame.copy()
    positives = adapted.index[(adapted["fraud_label"] == 1)]
    chosen = rng.choice(positives, size=int(len(positives) * share), replace=False) if len(positives) else []
    for column in ["limit_ratio", "amount", "approve_hour", "approval_minutes"] + RED_FLAGS + ["rule_score"]:
        adapted[column] = adapted[column].astype(float)
    if hide_near_limit:
        ratio = rng.uniform(0.80, 0.93, len(chosen))
        adapted.loc[chosen, "limit_ratio"] = ratio
        adapted.loc[chosen, "amount"] = ratio * adapted.loc[chosen, "approval_limit"].to_numpy()
        adapted.loc[chosen, "rf_near_limit"] = 0
    if hide_night:
        adapted.loc[chosen, "approve_hour"] = rng.integers(9, 16, len(chosen))
        adapted.loc[chosen, "rf_after_hours"] = 0
    if hide_fast:
        adapted.loc[chosen, "approval_minutes"] = adapted.loc[chosen, "approval_minutes"].clip(lower=0) + rng.uniform(30, 180, len(chosen))
        adapted.loc[chosen, "rf_fast_approval"] = 0
    adapted["rule_score"] = adapted[RED_FLAGS].sum(axis=1)
    return adapted


def recall_precision(labels, scores, threshold) -> tuple[float, float]:
    predicted = scores >= threshold
    tp = (predicted & (labels == 1)).sum()
    return tp / max((labels == 1).sum(), 1), tp / max(predicted.sum(), 1)


def render() -> None:
    ui.page_header("Stress Testing", "Simulasi: bagaimana kalau kapasitas, perilaku pelaku, atau kondisi data berubah?")
    settings = services.app_settings()
    test = services.table("test_features")
    transactions = services.table("transactions_scored")
    population_2026 = transactions[transactions["event_time"].dt.year == 2026]
    months = max(population_2026["event_time"].dt.to_period("M").nunique(), 1)
    model = services.get_risk_model()
    tab_capacity, tab_evasion, tab_outage, tab_prevalence, tab_drift = st.tabs(
        ["A. Kapasitas & threshold", "B. Pelaku beradaptasi", "C. Pipeline graph gagal", "D. Prevalensi berubah", "E. Drift data"])
    with tab_capacity:
        c1, c2, c3, c4 = st.columns(4)
        investigators = c1.slider("Jumlah investigator", 1, 30, int(settings.get("investigators_available", 6)))
        hours = c2.slider("Jam investigasi/orang/bulan", 40, 160, int(settings.get("hours_per_investigator_month", 120)), step=10)
        per_alert = c3.slider("Jam per alert", 0.5, 8.0, float(settings.get("audit_hours_per_alert", 2.0)), step=0.5)
        threshold = c4.slider("Skor minimum alert", 5, 95, int(round(settings.get("final_threshold", 0.5) * 100)), step=1) / 100
        capacity = investigators * hours / per_alert
        alerts_per_month = (population_2026["risk_probability"] >= threshold).sum() / months
        recall, precision = recall_precision(test["fraud_label"].to_numpy(), test["risk_probability"].to_numpy(), threshold)
        missed = int(((test["risk_probability"] < threshold) & (test["fraud_label"] == 1)).sum())
        ui.kpi_row([("Kapasitas", f"{ui.number(capacity)} alert/bln", f"{investigators} orang × {hours} jam ÷ {per_alert} jam", ui.BLUE),
                    ("Beban alert", f"{ui.number(alerts_per_month)} /bln", "populasi 2026", ui.ORANGE if alerts_per_month > capacity else ui.GREEN),
                    ("Utilisasi", ui.percent(alerts_per_month / capacity), "beban ÷ kapasitas", ui.RED if alerts_per_month > capacity else ui.GREEN),
                    ("Recall (test)", ui.percent(recall), f"{missed} kasus kolusi lolos", ui.PURPLE),
                    ("Precision (test)", ui.percent(precision), "di kasus yang direview", ui.YELLOW)])
        if alerts_per_month > capacity:
            st.warning("Beban melebihi kapasitas: naikkan skor minimum, tambah investigator, atau gunakan antrean berprioritas (skor tertinggi dulu).")
        grid = np.arange(0.05, 0.96, 0.01)
        sweep = pd.DataFrame({"skor minimum": grid * 100,
                              "alert/bulan": [(population_2026["risk_probability"] >= t).sum() / months for t in grid],
                              "recall": [recall_precision(test["fraud_label"].to_numpy(), test["risk_probability"].to_numpy(), t)[0] for t in grid]})
        figure = go.Figure()
        figure.add_scatter(x=sweep["skor minimum"], y=sweep["alert/bulan"], name="alert/bulan", line=dict(color=ui.BLUE, width=3))
        figure.add_scatter(x=sweep["skor minimum"], y=sweep["recall"] * 100, name="recall (%)", yaxis="y2", line=dict(color=ui.GREEN, width=3))
        figure.add_hline(y=capacity, line_dash="dash", line_color=ui.ORANGE, annotation_text="kapasitas")
        figure.add_vline(x=threshold * 100, line_dash="dot", line_color=ui.INK)
        figure.update_layout(yaxis=dict(title="alert per bulan"), yaxis2=dict(title="recall (%)", overlaying="y", side="right", range=[0, 105]),
                             xaxis_title="skor minimum")
        st.plotly_chart(ui.style_figure(figure, "Trade-off beban kerja vs kolusi yang tertangkap"), use_container_width=True)
    with tab_evasion:
        st.markdown("Pelaku yang tahu rules bisa menghindarinya. Pilih perilaku yang dihindari dan berapa persen pelaku yang beradaptasi; "
                    "model menilai ulang kasus test 2026 secara langsung.")
        c1, c2, c3, c4 = st.columns(4)
        hide_near = c1.checkbox("Hindari near-limit (nominal 80–93%)", value=True)
        hide_night = c2.checkbox("Hindari jam malam (approve 09–16)", value=True)
        hide_fast = c3.checkbox("Hindari approval kilat (+30–180 mnt)", value=True)
        share = c4.slider("% pelaku beradaptasi", 0, 100, 100, step=10) / 100
        threshold = settings.get("final_threshold", 0.5)
        rules_threshold = settings.get("rules_threshold", 1)
        if model is None:
            st.info("Model tidak bisa dimuat di lingkungan ini (versi pustaka berbeda); menampilkan hasil yang dihitung di notebook.")
            st.dataframe(services.table("stress_evasion"), hide_index=True, use_container_width=True)
        else:
            adapted = apply_evasion(test, hide_near, hide_night, hide_fast, share)
            new_probability = model.predict_proba(adapted)
            positives = adapted["fraud_label"] == 1
            result = pd.DataFrame({
                "pendekatan": ["Rules", "Model (rules + graph + riwayat)"],
                "recall sebelum": [(test.loc[positives, "rule_score"] >= rules_threshold).mean(), (test.loc[positives, "risk_probability"] >= threshold).mean()],
                "recall sesudah": [(adapted.loc[positives, "rule_score"] >= rules_threshold).mean(), (new_probability[positives.to_numpy()] >= threshold).mean()]})
            figure = go.Figure()
            figure.add_bar(x=result["pendekatan"], y=result["recall sebelum"] * 100, name="sebelum", marker_color=ui.MUTED)
            figure.add_bar(x=result["pendekatan"], y=result["recall sesudah"] * 100, name="sesudah pelaku beradaptasi", marker_color=ui.ORANGE)
            figure.update_layout(barmode="group", yaxis_title="recall (%)", yaxis_range=[0, 105])
            st.plotly_chart(ui.style_figure(figure, "Rules runtuh, model bertahan karena pola hubungan sulit disembunyikan"),
                            use_container_width=True)
            st.caption("Dalam jangka panjang fitur graph juga ikut melemah karena dibangun dari red flag; karena itu model dilatih ulang berkala.")
    with tab_outage:
        st.dataframe(services.table("stress_graph_outage"), hide_index=True, use_container_width=True)
        if model is not None and st.button("Hitung ulang langsung: kosongkan semua fitur graph"):
            outage = test.copy()
            outage[GRAPH_FEATURES] = np.nan
            probability = model.predict_proba(outage)
            recall, precision = recall_precision(test["fraud_label"].to_numpy(), probability, settings.get("final_threshold", 0.5))
            st.metric("Recall tanpa fitur graph", ui.percent(recall), f"{(recall - settings.get('test_recall', recall)) * 100:+.0f} poin")
        st.caption("Sistem tetap berjalan (imputasi), tetapi kinerja turun: kegagalan pipeline graph harus memicu alarm monitoring.")
    with tab_prevalence:
        multiplier = st.slider("Proporsi kolusi dibanding data uji (×)", 0.1, 3.0, 1.0, step=0.1)
        threshold = settings.get("final_threshold", 0.5)
        labels, scores = test["fraud_label"].to_numpy(), test["risk_probability"].to_numpy()
        predicted = scores >= threshold
        tp, fp = (predicted & (labels == 1)).sum(), (predicted & (labels == 0)).sum() / multiplier
        rules_predicted = test["rule_score"].to_numpy() >= settings.get("rules_threshold", 1)
        rules_tp, rules_fp = (rules_predicted & (labels == 1)).sum(), (rules_predicted & (labels == 0)).sum() / multiplier
        ui.kpi_row([("Precision model", ui.percent(tp / max(tp + fp, 1e-9)), "recall tidak berubah", ui.ORANGE),
                    ("Precision rules", ui.percent(rules_tp / max(rules_tp + rules_fp, 1e-9)), "", ui.MUTED)])
        st.dataframe(services.table("stress_prevalence"), hide_index=True, use_container_width=True)
        st.caption("Di seluruh transaksi proporsi kolusi jauh lebih kecil daripada di kasus yang direview, sehingga precision turun — "
                   "alasan precision populasi disajikan sebagai rentang.")
    with tab_drift:
        drift = services.table("drift_psi")
        figure = px.bar(drift.sort_values("PSI 2026 vs train"), x="PSI 2026 vs train", y="fitur", orientation="h", color="status",
                        color_discrete_map={"stabil": ui.GREEN, "pantau": ui.YELLOW, "bergeser": ui.ORANGE})
        figure.add_vline(x=0.1, line_dash="dot", line_color=ui.YELLOW)
        figure.add_vline(x=0.25, line_dash="dash", line_color=ui.ORANGE)
        st.plotly_chart(ui.style_figure(figure, "Population Stability Index fitur utama (2026 vs data latih)",
                                        "< 0,1 stabil · 0,1–0,25 pantau · > 0,25 bergeser → evaluasi & latih ulang"), use_container_width=True)
