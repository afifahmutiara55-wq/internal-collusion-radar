"""Tools agen: fungsi HANYA-BACA di atas hasil model (PostgreSQL/CSV) + knowledge base.

Setiap tool punya: deskripsi (untuk LLM planner), skema argumen (divalidasi sebelum dijalankan), dan daftar role yang
boleh memakainya. Tool yang menghasilkan aksi (draf memo) tidak menyimpan apa pun: hasilnya berstatus
MENUNGGU_PERSETUJUAN dan baru tersimpan setelah investigator menekan tombol persetujuan di dashboard (human-in-the-loop).
"""
from __future__ import annotations

import difflib
import re

import numpy as np
import pandas as pd

ROLE_TOOLS = {
    "investigator": ["cari_alert", "detail_alert", "profil_pegawai", "komunitas_risiko", "cari_regulasi",
                     "draf_memo_investigasi", "performa_model"],
    "manajemen": ["ringkasan_kpi", "risiko_unit", "komunitas_risiko", "simulasi_kapasitas", "performa_model", "cari_regulasi"],
}
RED_FLAG_NAMES = {"rf_near_limit": "near limit 95–99%", "rf_after_hours": "luar jam kerja", "rf_fast_approval": "approval kilat",
                  "rf_debtor_repeat": "debitur berulang", "rf_pair_dependency": "ketergantungan approver"}
TOOL_SPECS = {
    "cari_alert": {"description": "Cari alert transaksi. Argumen opsional: level ('Tinggi'/'Sedang'), unit (nama unit), "
                                  "red_flag (salah satu: near_limit, after_hours, fast_approval, debtor_repeat, pair_dependency), "
                                  "nama_pegawai (nama/ID requester atau approver), bulan ('YYYY-MM'), limit (1–20).",
                   "args": {"level": str, "unit": str, "red_flag": str, "nama_pegawai": str, "bulan": str, "limit": int}},
    "detail_alert": {"description": "Detail satu transaksi: skor, kategori, alasan SHAP, red flag, bukti graph, skor anomali "
                                    "approver, status review. Argumen wajib: request_id (format R0000000).",
                     "args": {"request_id": str}},
    "profil_pegawai": {"description": "Profil pegawai (requester/approver): jabatan, unit, skor anomali terakhir & alasannya, "
                                      "jumlah alert 2026, komunitas risiko, pasangan kerja teratas. Argumen: nama_atau_id.",
                       "args": {"nama_atau_id": str}},
    "komunitas_risiko": {"description": "Komunitas risiko dari graph. Argumen opsional: community_id (angka) untuk detail "
                                        "anggota, atau nama_atau_id untuk mencari komunitas seorang pegawai, atau top (1–10).",
                         "args": {"community_id": int, "nama_atau_id": str, "top": int}},
    "cari_regulasi": {"description": "Cari potongan knowledge base (regulasi KUR, POJK, UU PDP, SOP, panduan skor) yang relevan. "
                                     "Argumen: pertanyaan.", "args": {"pertanyaan": str}},
    "draf_memo_investigasi": {"description": "Susun DRAF memo investigasi untuk satu transaksi (menunggu persetujuan "
                                             "investigator). Argumen: request_id.", "args": {"request_id": str}},
    "ringkasan_kpi": {"description": "Ringkasan KPI risiko kolusi per tahun: transaksi, alert, nominal berisiko, precision "
                                     "populasi, dampak. Argumen opsional: tahun (2025/2026).", "args": {"tahun": int}},
    "risiko_unit": {"description": "Peringkat unit berdasarkan tingkat alert Tinggi (agregat, tanpa nama pegawai). "
                                   "Argumen opsional: tahun, top (1–20).", "args": {"tahun": int, "top": int}},
    "simulasi_kapasitas": {"description": "Simulasi kapasitas investigasi: alert/bulan, recall, precision untuk jumlah "
                                          "investigator atau skor minimum tertentu. Argumen opsional: investigator (1–50), "
                                          "skor_minimum (10–95).", "args": {"investigator": int, "skor_minimum": int}},
    "performa_model": {"description": "Kinerja model di data uji 2026: classifier vs baseline, radar Isolation Forest, "
                                      "graph, fairness. Tanpa argumen.", "args": {}},
}
ARG_PATTERNS = {"request_id": r"^R\d{7}$", "bulan": r"^20\d\d-\d\d$"}


class ToolError(ValueError):
    pass


def records(frame: pd.DataFrame, columns: list, limit: int = 10) -> list[dict]:
    subset = frame[[c for c in columns if c in frame.columns]].head(limit).copy()
    for column in subset.columns:
        if pd.api.types.is_datetime64_any_dtype(subset[column]):
            subset[column] = subset[column].dt.strftime("%Y-%m-%d %H:%M")
        elif pd.api.types.is_float_dtype(subset[column]):
            subset[column] = subset[column].round(3)
    return subset.replace({np.nan: None}).to_dict("records")


class AgentTools:
    def __init__(self, store, retriever=None):
        self.store, self.retriever = store, retriever

    # ------------------------------------------------------------------ helper data
    @property
    def employees(self) -> pd.DataFrame:
        return self.store.table("employees")

    def name_of(self, employee_id) -> str:
        match = self.employees.loc[self.employees["employee_id"] == employee_id, "employee_name"]
        return match.iloc[0] if len(match) else str(employee_id)

    def resolve_employee(self, query: str) -> pd.Series:
        query = (query or "").strip()
        employees = self.employees
        if re.fullmatch(r"E\d{5}", query.upper()):
            match = employees[employees["employee_id"] == query.upper()]
        else:
            lowered = employees["employee_name"].str.lower()
            match = employees[lowered == query.lower()]
            if match.empty:
                match = employees[lowered.str.contains(re.escape(query.lower()), na=False)]
            if match.empty:
                close = difflib.get_close_matches(query.lower(), lowered.tolist(), n=1, cutoff=0.75)
                match = employees[lowered.isin(close)]
        if match.empty:
            raise ToolError(f"pegawai '{query}' tidak ditemukan")
        return match.iloc[0]

    # ------------------------------------------------------------------ eksekusi aman
    def call(self, tool: str, args: dict, role: str) -> dict:
        if tool not in TOOL_SPECS:
            raise ToolError(f"tool '{tool}' tidak dikenal")
        if tool not in ROLE_TOOLS.get(role, []):
            raise ToolError(f"tool '{tool}' tidak diizinkan untuk role {role}")
        clean = {}
        for name, value in (args or {}).items():
            expected = TOOL_SPECS[tool]["args"].get(name)
            if expected is None or value in (None, ""):
                continue
            try:
                value = expected(value)
            except (TypeError, ValueError):
                raise ToolError(f"argumen {name} harus bertipe {expected.__name__}")
            if isinstance(value, str):
                value = value.strip()[:120]
                if name in ARG_PATTERNS and not re.fullmatch(ARG_PATTERNS[name], value.upper() if name == "request_id" else value):
                    raise ToolError(f"format {name} tidak valid: {value}")
                if name == "request_id":
                    value = value.upper()
            clean[name] = value
        return getattr(self, tool)(**clean)

    # ------------------------------------------------------------------ tools investigator
    def cari_alert(self, level: str = None, unit: str = None, red_flag: str = None, nama_pegawai: str = None,
                   bulan: str = None, limit: int = 10) -> dict:
        frame = self.store.table("transactions_scored")
        frame = frame[frame["risk_band"].isin([level.capitalize()] if level else ["Tinggi", "Sedang"])]
        if unit:
            frame = frame[frame["requester_unit"].str.contains(unit, case=False, na=False)
                          | frame["approver_unit"].str.contains(unit, case=False, na=False)]
        if red_flag:
            column = "rf_" + red_flag.replace("rf_", "")
            if column not in frame:
                raise ToolError(f"red flag '{red_flag}' tidak dikenal")
            frame = frame[frame[column] == 1]
        if nama_pegawai:
            employee = self.resolve_employee(nama_pegawai)
            frame = frame[(frame["requester_id"] == employee["employee_id"]) | (frame["approver_id"] == employee["employee_id"])]
        if bulan:
            frame = frame[frame["month"].astype(str) == bulan]
        frame = frame.sort_values(["risk_score", "event_time"], ascending=[False, False])
        top = frame.head(max(1, min(int(limit or 10), 20)))
        top = top.assign(requester=top["requester_id"].map(self.name_of), approver=top["approver_id"].map(self.name_of))
        rows = records(top, ["request_id", "event_time", "requester", "approver", "requester_unit", "amount", "risk_score",
                             "risk_band", "top_reasons"], len(top))
        return {"jumlah_ditemukan": int(len(frame)), "ditampilkan": len(rows), "alert": rows}

    def detail_alert(self, request_id: str) -> dict:
        frame = self.store.table("transactions_scored")
        match = frame[frame["request_id"] == request_id]
        if match.empty:
            raise ToolError(f"transaksi {request_id} tidak ada di data 2025–2026")
        row = match.iloc[0]
        explanations = self.store.table("alert_explanations")
        reasons = explanations[explanations["request_id"] == request_id].sort_values("rank")
        flags = [name for column, name in RED_FLAG_NAMES.items() if row.get(column) == 1]
        community = None
        members = self.store.table("community_members")
        member_row = members[members["employee_id"] == row["requester_id"]]
        if len(member_row):
            community = int(member_row.iloc[0]["community_id"])
        return {"request_id": request_id, "tanggal_approval": str(row["approve_ts"])[:16],
                "requester": f"{self.name_of(row['requester_id'])} ({row['requester_id']}, {row['requester_unit']})",
                "approver": f"{self.name_of(row['approver_id'])} ({row['approver_id']}, {row['approver_position']})",
                "debitur": f"{row.get('debtor_name_masked', '-')} ({row['debtor_id']}, {row['debtor_type']}, {row['business_sector']})",
                "produk": row["loan_product"], "nominal": float(row["amount"]), "limit_ratio": round(float(row["limit_ratio"]), 3),
                "skor_risiko": float(row["risk_score"]), "kategori": row["risk_band"],
                "alasan_shap": [{"alasan": r["reason"], "kontribusi_shap": round(float(r["shap_value"]), 3)} for _, r in reasons.iterrows()],
                "red_flag": flags, "bukti": str(row.get("evidence") or "").split(" | "),
                "skor_anomali_approver": None if pd.isna(row.get("approver_anomaly_score")) else round(float(row["approver_anomaly_score"]), 1),
                "alasan_anomali_approver": row.get("approver_anomaly_reason") or "",
                "komunitas_requester": community, "sudah_direview_audit": bool(row.get("reviewed")),
                "catatan": "Skor adalah indikasi berbasis data, bukan bukti. Keputusan oleh investigator."}

    def profil_pegawai(self, nama_atau_id: str) -> dict:
        employee = self.resolve_employee(nama_atau_id)
        employee_id = employee["employee_id"]
        transactions = self.store.table("transactions_scored")
        tx_2026 = transactions[transactions["event_time"].dt.year == 2026]
        as_requester = tx_2026[tx_2026["requester_id"] == employee_id]
        as_approver = tx_2026[tx_2026["approver_id"] == employee_id]
        profile = {"employee_id": employee_id, "nama": employee["employee_name"], "jabatan": employee["position"],
                   "unit": employee["unit_name"], "wilayah": employee.get("region"), "peran": employee["role"],
                   "transaksi_2026_sebagai_requester": int(len(as_requester)),
                   "transaksi_2026_sebagai_approver": int(len(as_approver)),
                   "alert_tinggi_2026": int(((as_requester["risk_band"] == "Tinggi").sum()) + (as_approver["risk_band"] == "Tinggi").sum())}
        anomaly = self.store.table("approver_anomaly")
        history = anomaly[anomaly["approver_id"] == employee_id].sort_values("month")
        if len(history):
            latest = history.iloc[-1]
            profile.update(skor_anomali_terakhir=round(float(latest["if_score"]), 1), bulan_anomali=str(latest["month"]),
                           kategori_anomali=latest["anomaly_band"], alasan_anomali=latest.get("if_reason") or "",
                           bulan_skor_anomali_95_ke_atas_12bln=int((history.tail(12)["if_score"] >= 95).sum()))
        members = self.store.table("community_members")
        membership = members[members["employee_id"] == employee_id]
        profile["komunitas_risiko"] = int(membership.iloc[0]["community_id"]) if len(membership) else None
        partner_column, partner_frame = (("requester_id", as_approver) if employee["role"] == "Approver" else ("approver_id", as_requester))
        partners = partner_frame.groupby(partner_column).agg(transaksi=("request_id", "size"),
                                                             alert_tinggi=("risk_band", lambda s: int((s == "Tinggi").sum())))
        partners = partners.sort_values(["alert_tinggi", "transaksi"], ascending=False).head(5).reset_index()
        profile["pasangan_teratas_2026"] = [{"pegawai": f"{self.name_of(r[partner_column])} ({r[partner_column]})",
                                             "transaksi": int(r["transaksi"]), "alert_tinggi": int(r["alert_tinggi"])}
                                            for _, r in partners.iterrows()]
        return profile

    def komunitas_risiko(self, community_id: int = None, nama_atau_id: str = None, top: int = 5) -> dict:
        communities, members = self.store.table("communities"), self.store.table("community_members")
        if nama_atau_id and community_id is None:
            employee = self.resolve_employee(nama_atau_id)
            membership = members[members["employee_id"] == employee["employee_id"]]
            if membership.empty:
                return {"pesan": f"{employee['employee_name']} tidak berada di komunitas risiko pada snapshot terbaru."}
            community_id = int(membership.iloc[0]["community_id"])
        if community_id is not None:
            info = communities[communities["community_id"] == community_id]
            if info.empty:
                raise ToolError(f"komunitas {community_id} tidak ditemukan")
            group = members[members["community_id"] == community_id]
            return {"komunitas": records(info, ["community_id", "anggota", "skor_komunitas", "prioritas", "unit",
                                                "edge_red_flag", "edge_ketergantungan", "edge_debitur_bersama", "aktif_bulan"], 1)[0],
                    "anggota": records(group, ["employee_id", "employee_name", "role", "position", "unit", "pagerank",
                                               "aktif_bulan"], 30)}
        top_communities = communities.sort_values("skor_komunitas", ascending=False).head(max(1, min(int(top or 5), 10)))
        return {"jumlah_komunitas": int(len(communities)),
                "teratas": records(top_communities, ["community_id", "anggota", "skor_komunitas", "prioritas",
                                                     "approver_utama", "unit", "aktif_bulan"], 10)}

    def cari_regulasi(self, pertanyaan: str) -> dict:
        if self.retriever is None:
            raise ToolError("knowledge base belum dimuat")
        hits = self.retriever.hybrid_search(pertanyaan, top_n=4)
        return {"sumber": [{"no": f"S{i}", "dokumen": h["title"], "bagian": h["section"], "chunk_id": h["chunk_id"],
                            "kutipan": h["text"].split("\n", 1)[-1][:700]} for i, h in enumerate(hits, start=1)]}

    def draf_memo_investigasi(self, request_id: str) -> dict:
        detail = self.detail_alert(request_id)
        sop = self.cari_regulasi("struktur memo investigasi langkah klarifikasi SOP alert") if self.retriever else {"sumber": []}
        reasons = "\n".join(f"   - {r['alasan']} (kontribusi SHAP {r['kontribusi_shap']:+.2f})" for r in detail["alasan_shap"]) or "   - (tidak ada)"
        evidence = "\n".join(f"   - {e}" for e in detail["bukti"] if e) or "   - (tidak ada)"
        memo = f"""DRAF MEMO INVESTIGASI — MENUNGGU PERSETUJUAN INVESTIGATOR
1. Identitas alert: {detail['request_id']} · approval {detail['tanggal_approval']} · skor {detail['skor_risiko']:.0f}/100 · kategori {detail['kategori']}
   Requester: {detail['requester']}
   Approver : {detail['approver']}
   Debitur  : {detail['debitur']} · {detail['produk']} · nominal Rp{detail['nominal']:,.0f} ({detail['limit_ratio']:.0%} limit approver)
2. Ringkasan indikasi (alasan model):
{reasons}
   Red flag terpenuhi: {', '.join(detail['red_flag']) or 'tidak ada'}
3. Bukti pendukung dari data:
{evidence}
   Skor anomali approver: {detail['skor_anomali_approver'] if detail['skor_anomali_approver'] is not None else '-'} {('— ' + detail['alasan_anomali_approver']) if detail['alasan_anomali_approver'] else ''}
4. Penjelasan sah yang perlu dicek: lembur akhir bulan terdokumentasi, surat delegasi cuti/SK Plt, memo pengalihan portofolio, top-up debitur lama.
5. Rencana pengumpulan bukti: dokumen kredit lengkap, log login & aksi sistem, riwayat pasangan requester–approver, verifikasi keberadaan usaha debitur.
6. Rencana klarifikasi: klarifikasi tertulis/wawancara terpisah dengan requester dan approver.
7. Rekomendasi sementara: {'usulkan penahanan pencairan kepada pejabat berwenang dan triase ≤ 1×24 jam' if detail['kategori'] == 'Tinggi' else 'pemantauan dan triase ≤ 3 hari kerja'}.
Catatan: memo disusun dari indikasi berbasis data dan tidak memuat kesimpulan bersalah. Rujukan prosedur: {', '.join(sorted({s['dokumen'] for s in sop['sumber']})) or 'SOP investigasi'}."""
        return {"status": "MENUNGGU_PERSETUJUAN", "request_id": request_id, "memo": memo,
                "catatan": "Agen tidak menyimpan atau mengirim memo. Investigator harus meninjau dan menekan 'Setujui & simpan'."}

    # ------------------------------------------------------------------ tools manajemen
    def ringkasan_kpi(self, tahun: int = 2026) -> dict:
        transactions = self.store.table("transactions_scored")
        frame = transactions[transactions["event_time"].dt.year == int(tahun)]
        if frame.empty:
            raise ToolError(f"tidak ada data tahun {tahun}")
        months = frame["month"].nunique()
        high = frame[frame["risk_band"] == "Tinggi"]
        reviewed = high[high["reviewed"] == True]  # noqa: E712
        true_reviewed = int((reviewed["label_audit"] == 1).sum())
        precision_reviewed = true_reviewed / max(len(reviewed), 1)
        settings = self.store.json_document("app_settings")
        impact = self.store.table("impact")
        model_row = impact[impact["skenario"].str.startswith("Model")]
        return {"tahun": int(tahun), "bulan_data": int(months), "transaksi_dinilai": int(len(frame)),
                "nominal_total": float(frame["amount"].sum()), "alert_tinggi": int(len(high)),
                "alert_tinggi_per_bulan": round(len(high) / max(months, 1), 1), "alert_sedang": int((frame["risk_band"] == "Sedang").sum()),
                "nominal_alert_tinggi": float(high["amount"].sum()),
                "precision_populasi_batas_bawah": round(true_reviewed / max(len(high), 1), 3),
                "precision_populasi_batas_atas": round((true_reviewed + precision_reviewed * (len(high) - len(reviewed))) / max(len(high), 1), 3),
                "transaksi_rules_menandai": int((frame["rule_score"] >= settings.get("rules_threshold", 1)).sum()),
                "manfaat_bersih_per_tahun_estimasi": float(model_row["manfaat bersih/tahun"].iloc[0]) if len(model_row) else None,
                "komunitas_risiko_aktif": int(len(self.store.table("communities")))}

    def risiko_unit(self, tahun: int = 2026, top: int = 10) -> dict:
        unit_risk = self.store.table("unit_risk")
        frame = unit_risk[unit_risk["year"] == int(tahun)].sort_values("alert_rate", ascending=False)
        return {"tahun": int(tahun), "unit_teratas": records(frame, ["requester_unit", "region", "transaksi", "alert_tinggi",
                                                                     "alert_rate", "skor_rata"], max(1, min(int(top or 10), 20)))}

    def simulasi_kapasitas(self, investigator: int = None, skor_minimum: int = None) -> dict:
        sweep = self.store.table("stress_threshold")
        settings = self.store.json_document("app_settings")
        hours = settings.get("hours_per_investigator_month", 120)
        per_alert = settings.get("audit_hours_per_alert", 2)
        if skor_minimum is not None:
            row = sweep.iloc[(sweep["skor_minimum"] - float(skor_minimum)).abs().argsort().iloc[0]]
            capacity_needed = row["alert/bulan (populasi 2026)"] * per_alert / hours
            return {"skor_minimum": float(row["skor_minimum"]), "alert_per_bulan": round(float(row["alert/bulan (populasi 2026)"]), 1),
                    "recall_test": round(float(row["recall (kasus direview)"]), 3),
                    "precision_test": round(float(row["precision (kasus direview)"]), 3),
                    "investigator_dibutuhkan": round(float(capacity_needed), 1)}
        investigator = int(investigator or settings.get("investigators_available", 6))
        capacity = investigator * hours / per_alert
        feasible = sweep[sweep["alert/bulan (populasi 2026)"] <= capacity].sort_values("skor_minimum")
        row = feasible.iloc[0] if len(feasible) else sweep.sort_values("skor_minimum").iloc[-1]
        return {"investigator": investigator, "kapasitas_alert_per_bulan": round(capacity, 1),
                "skor_minimum_disarankan": float(row["skor_minimum"]),
                "alert_per_bulan": round(float(row["alert/bulan (populasi 2026)"]), 1),
                "recall_test": round(float(row["recall (kasus direview)"]), 3),
                "precision_test": round(float(row["precision (kasus direview)"]), 3),
                "catatan": "skor minimum terendah yang beban alertnya masih muat kapasitas"}

    def performa_model(self) -> dict:
        metrics = self.store.table("model_metrics")
        test = metrics[metrics["periode"] == "test 2026"]
        radar = self.store.table("radar_evaluation")
        radar_recent = radar[radar["periode"] == "2025–2026"]
        graph = self.store.table("graph_evaluation")
        fairness = self.store.table("fairness")
        return {"classifier_test_2026": records(test, ["model", "precision", "recall", "F1", "ROC-AUC", "PR-AUC", "FP", "FN"], 5),
                "radar_isolation_forest_2025_2026": records(radar_recent, ["radar", "PR-AUC", "precision@5%", "lift@5%"], 5),
                "graph_peringatan_dini": {"precision_rata": round(float(graph["precision"].tail(21).mean()), 3),
                                          "lift_rata": round(float(graph["lift"].tail(21).mean()), 2)},
                "fairness_FPR_model_rentang": [round(float(fairness["FPR model"].min()), 3), round(float(fairness["FPR model"].max()), 3)]}
