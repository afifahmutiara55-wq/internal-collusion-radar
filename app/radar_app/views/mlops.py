"""Manajemen · Model & MLflow: registry, run eksperimen, model card, alur MLOps."""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from ..config import SETTINGS
from ..ui import components as ui
from ..ui import services


def mlflow_tables() -> tuple[pd.DataFrame | None, pd.DataFrame | None, str]:
    if not SETTINGS.mlflow_tracking_uri:
        return None, None, "MLFLOW_TRACKING_URI belum di-set"
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
        mlflow.set_tracking_uri(SETTINGS.mlflow_tracking_uri)
        client = MlflowClient()
        models = []
        for model in client.search_registered_models():
            for version in model.latest_versions or []:
                models.append({"model": model.name, "versi": version.version, "stage/alias": version.current_stage,
                               "run_id": version.run_id, "dibuat": pd.to_datetime(version.creation_timestamp, unit="ms")})
        runs = mlflow.search_runs(experiment_names=["kur-collusion-radar"], max_results=100)
        return pd.DataFrame(models), runs, "terhubung ke " + SETTINGS.mlflow_tracking_uri
    except Exception as error:
        return None, None, f"gagal terhubung ({error.__class__.__name__})"


def render() -> None:
    ui.page_header("Model & MLflow", "Model mana yang dipakai, dari eksperimen apa, dan bagaimana tata kelolanya")
    card = services.get_store().json_document("model_card")
    settings = services.app_settings()
    ui.kpi_row([("Model transaksi", card.get("algoritma", "-"), card.get("kalibrasi", ""), ui.BLUE),
                ("Threshold Tinggi", f"{card.get('threshold_tinggi', 0) * 100:.0f}", f"Sedang ≥ {card.get('threshold_sedang', 0) * 100:.0f}", ui.ORANGE),
                ("PR-AUC test", f"{card.get('metrik_test', {}).get('PR-AUC', 0):.3f}", "data 2026 (dibuka sekali)", ui.GREEN),
                ("Pakai output IF?", "Tidak" if not card.get("memakai_output_isolation_forest") else "Ya", "aturan independensi", ui.PURPLE)])
    st.link_button("Buka MLflow UI", SETTINGS.mlflow_ui_url)
    models, runs, status = mlflow_tables()
    st.caption(f"MLflow: {status}")
    left, right = st.columns([1, 1.2])
    with left:
        st.markdown("**Model registry**")
        if models is not None and len(models):
            st.dataframe(models, hide_index=True, use_container_width=True)
        else:
            st.dataframe(pd.DataFrame([{"model": "kur_collusion_classifier", "keterangan": "classifier skor transaksi"},
                                       {"model": "kur_approver_iforest", "keterangan": "Isolation Forest radar approver"}]),
                         hide_index=True, use_container_width=True)
        st.markdown("**Model card**")
        st.json(card, expanded=False)
    with right:
        st.markdown("**Run eksperimen**")
        if runs is not None and len(runs):
            columns = [c for c in runs.columns if c in ("tags.mlflow.runName", "start_time", "status") or c.startswith("metrics.")]
            st.dataframe(runs[columns], hide_index=True, use_container_width=True, height=420)
        else:
            exported = services.table("mlflow_runs")
            if len(exported):
                exported = exported.assign(metrik=exported["metrics"].map(lambda m: ", ".join(f"{k}={v:.3f}" for k, v in json.loads(m).items())[:160]))
                st.dataframe(exported[["run_name", "registered_model", "metrik"]], hide_index=True, use_container_width=True, height=420)
    st.markdown("**Alur MLOps**")
    st.graphviz_chart("""digraph { rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#eef4fc", color="#2a78d6", fontname="Helvetica"];
        data [label="7 tabel\\n+ data contract"]; feat [label="fitur as-of\\nrules + graph"]; screen [label="PyCaret\\nscreening"];
        tune [label="Optuna\\nCV deret waktu"]; val [label="validasi 2025\\nkalibrasi + threshold"]; test [label="test 2026\\n(sekali)"];
        reg [label="MLflow\\nregistry", fillcolor="#e7f6ef", color="#1baf7a"]; db [label="PostgreSQL\\nskor + alasan"]; app [label="Streamlit\\n2 role"];
        mon [label="monitoring\\nPSI, keputusan", fillcolor="#fdf0e9", color="#eb6834"];
        data -> feat -> screen -> tune -> val -> test -> reg -> db -> app -> mon; mon -> feat [style=dashed, label="latih ulang"]; }""")
    st.markdown(f"Kebijakan: pelatihan ulang kuartalan atau bila PSI > 0,25 pada fitur utama; setiap versi baru melewati validasi yang sama "
                f"dan dicatat di MLflow. Seleksi: {settings.get('screening_engine', '-')}; tuning: {settings.get('tuning_engine', '-')}.")
