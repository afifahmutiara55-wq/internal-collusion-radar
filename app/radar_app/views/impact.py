"""Manajemen · Dampak Bisnis: asumsi bisa diubah (slider), manfaat bersih, sensitivitas, nilai strategis."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ..ui import components as ui
from ..ui import services


def render() -> None:
    ui.page_header("Dampak Bisnis", "Estimasi transparan: ubah asumsi, angka dihitung ulang langsung")
    settings = services.app_settings()
    with st.sidebar:
        st.markdown("**Asumsi**")
        lgd = st.slider("LGD (kerugian saat macet)", 0.2, 1.0, float(settings.get("lgd", 0.6)), 0.05)
        prevention = st.slider("% alert benar yang berhasil dicegah", 0.2, 1.0, float(settings.get("prevention_rate", 0.75)), 0.05)
        hours = st.slider("Jam investigasi per alert", 0.5, 8.0, float(settings.get("audit_hours_per_alert", 2.0)), 0.5)
        cost = st.number_input("Biaya per jam investigator (Rp)", 50_000, 1_000_000, int(settings.get("investigator_cost_per_hour", 150_000)), 10_000)
        system_cost = st.number_input("Biaya sistem per tahun (Rp, opsional)", 0, 20_000_000_000, 750_000_000, 50_000_000)
    base_loss = settings.get("loss_per_fraud", 0) / max(settings.get("lgd", 0.6), 1e-9)
    fraud_per_year = settings.get("fraud_cases_per_year", 0)
    scenarios = {"Rules saja": (settings.get("rules_test_recall", 0), settings.get("rules_alerts_per_month", 0)),
                 "Model final": (settings.get("test_recall", 0), settings.get("model_alerts_per_month", 0))}
    rows = []
    for name, (recall, alerts) in scenarios.items():
        prevented = fraud_per_year * recall * prevention * base_loss * lgd
        review = alerts * 12 * hours * cost
        extra = system_cost if name == "Model final" else 0
        rows.append({"skenario": name, "recall": recall, "alert/bulan": alerts, "kerugian dicegah": prevented,
                     "biaya pemeriksaan": review, "biaya sistem": extra, "manfaat bersih": prevented - review - extra,
                     "FTE investigator": alerts * hours / 120})
    table = pd.DataFrame(rows).set_index("skenario")
    model, rules = table.loc["Model final"], table.loc["Rules saja"]
    roi = model["manfaat bersih"] / max(model["biaya pemeriksaan"] + model["biaya sistem"], 1)
    ui.kpi_row([("Kerugian dicegah/tahun", ui.rupiah(model["kerugian dicegah"]), f"{fraud_per_year:,.0f} kasus/thn × recall {ui.percent(model['recall'])}", ui.GREEN),
                ("Biaya pemeriksaan/tahun", ui.rupiah(model["biaya pemeriksaan"]), f"{model['alert/bulan']:.0f} alert/bln × {hours} jam", ui.YELLOW),
                ("Manfaat bersih/tahun", ui.rupiah(model["manfaat bersih"]), "setelah biaya sistem", ui.BLUE),
                ("ROI", f"{roi:.1f}×", "manfaat bersih ÷ biaya", ui.PURPLE),
                ("Hemat jam investigasi", f"{ui.number((rules['alert/bulan'] - model['alert/bulan']) * hours * 12)} jam/thn", "dibanding rules", ui.ORANGE)])
    left, right = st.columns([1.2, 1])
    figure = go.Figure()
    for column, color in [("kerugian dicegah", ui.GREEN), ("biaya pemeriksaan", ui.YELLOW), ("biaya sistem", ui.MUTED), ("manfaat bersih", ui.BLUE)]:
        figure.add_bar(x=table.index, y=table[column] / 1e9, name=column, marker_color=color)
    figure.update_layout(barmode="group", yaxis_title="Rp miliar / tahun")
    left.plotly_chart(ui.style_figure(figure, "Model vs rules", "asumsi di sidebar"), use_container_width=True)
    grid = pd.DataFrame([[fraud_per_year * model["recall"] * p * base_loss * l - model["biaya pemeriksaan"] - model["biaya sistem"]
                          for l in (0.4, 0.6, 0.8)] for p in (0.5, 0.75, 1.0)],
                        index=["50% dicegah", "75% dicegah", "100% dicegah"], columns=["LGD 40%", "LGD 60%", "LGD 80%"])
    figure = px.imshow(grid / 1e9, text_auto=".1f", color_continuous_scale=["#fbe7b5", ui.GREEN], aspect="auto",
                       labels=dict(color="Rp M"))
    right.plotly_chart(ui.style_figure(figure, "Sensitivitas manfaat bersih (Rp miliar)", "tetap positif di skenario pesimistis?"),
                       use_container_width=True)
    st.dataframe(table.style.format({"recall": "{:.0%}", "alert/bulan": "{:,.0f}", "kerugian dicegah": ui.rupiah, "biaya pemeriksaan": ui.rupiah,
                                     "biaya sistem": ui.rupiah, "manfaat bersih": ui.rupiah, "FTE investigator": "{:.1f}"}), use_container_width=True)
    st.markdown("#### Dampak non-finansial & nilai strategis")
    c1, c2 = st.columns(2)
    c1.markdown("""
- **Deteksi lebih cepat**: dari ±8 bulan setelah cair (saat NPL) menjadi sesaat setelah approval.
- **Efisiensi investigator**: alarm palsu jauh lebih sedikit dibanding rules; antrean berprioritas.
- **Kualitas bukti**: alasan SHAP + red flag + peta jaringan + jejak login per alert.
- **Efek jera**: pola kolusi dipantau sistematis; prinsip empat mata diperkuat.""")
    c2.markdown("""
| Pilar POJK 12/2024 | Peran sistem |
|---|---|
| Pencegahan | efek jera, profil approver berisiko |
| Deteksi | rules + graph + IF + classifier |
| Investigasi & pelaporan | antrean, bukti, draf memo (disetujui manusia) |
| Pemantauan & evaluasi | dashboard, stress test, drift, LLM-as-judge |""")
    st.caption("Estimasi memakai data sintetis dan asumsi yang dinyatakan terbuka; validasi dengan data internal sebelum dipakai untuk keputusan anggaran.")
