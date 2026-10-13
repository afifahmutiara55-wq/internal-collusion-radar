#!/usr/bin/env python3
"""Validasi dataset v5: membuktikan cacat yang ditemukan di v4 tidak muncul lagi.

Pemakaian:
    python validate_dataset.py --clean out_clean --raw data
(--clean = output generator dengan --no-noise; --raw = output dengan noise)
"""
import argparse
import re
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)
LINES = []


def out(s=""):
    print(s)
    LINES.append(str(s))


def check(name, ok, detail=""):
    out(f"[{'LULUS' if ok else 'GAGAL'}] {name}" + (f" — {detail}" if detail else ""))
    return ok


def load(d):
    T = {t: pd.read_csv(f"{d}/{t}.csv") for t in ["employees", "debtors", "requests", "approvals", "access_logs", "fraud_cases"]}
    for t, c in [("requests", "timestamp"), ("approvals", "timestamp"), ("access_logs", "timestamp")]:
        T[t][c] = pd.to_datetime(T[t][c])
    T["fraud_cases"]["case_date"] = pd.to_datetime(T["fraud_cases"]["case_date"])
    T["employees"]["employment_start_date"] = pd.to_datetime(T["employees"]["employment_start_date"])
    T["debtors"]["customer_since"] = pd.to_datetime(T["debtors"]["customer_since"])
    return T


def build_features(T):
    """Fitur as-of per request (hanya memakai kejadian SEBELUM request tsb)."""
    r, a, e, d = T["requests"], T["approvals"], T["employees"], T["debtors"]
    E = e.set_index("employee_id")
    D = d.set_index("debtor_id")
    tx = r.merge(a[["request_id", "approver_id", "timestamp"]].rename(columns={"timestamp": "approve_ts"}), on="request_id")
    tx = tx.rename(columns={"timestamp": "request_ts"}).sort_values("request_ts").reset_index(drop=True)
    tx["limit"] = tx.approver_id.map(E.approval_limit)
    tx["limit_ratio"] = tx.amount / tx["limit"]
    tx["near_limit"] = tx.limit_ratio.between(0.95, 0.9999).astype(int)    # strictly below limit
    h = tx.approve_ts.dt.hour
    tx["night_approval"] = (((h >= 19) | (h < 7)) | (tx.approve_ts.dt.dayofweek >= 5)).astype(int)
    tx["lag_min"] = (tx.approve_ts - tx.request_ts).dt.total_seconds() / 60
    tx["fast_approval"] = (tx.lag_min < 20).astype(int)
    tx["debtor_tenure_days"] = (tx.request_ts - tx.debtor_id.map(D.customer_since)).dt.days
    tx["req_unit"] = tx.requester_id.map(E.unit_name)
    tx["appr_unit"] = tx.approver_id.map(E.unit_name)
    tx["same_unit"] = (tx.req_unit == tx.appr_unit).astype(int)
    parents = {}
    for u in e.unit_name.unique():
        parents[u] = u
    import importlib.util, pathlib
    spec = importlib.util.spec_from_file_location("g", pathlib.Path(__file__).with_name("generate_kur_dataset.py"))
    g = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(g)
    parent = {k: v[1] for k, v in g.KCP.items()}
    region = {**g.KC, **{k: v[0] for k, v in g.KCP.items()}}
    tx["outside_routing"] = [int(au not in (ru, parent.get(ru, ru))) for ru, au in zip(tx.req_unit, tx.appr_unit)]
    tx["debtor_region_mismatch"] = (tx.debtor_id.map(D.region) != tx.req_unit.map(region)).astype(int)
    # riwayat as-of
    tx["pair_prior_n"] = tx.groupby(["requester_id", "approver_id"]).cumcount()
    tx["req_prior_n"] = tx.groupby("requester_id").cumcount()
    tx["appr_prior_n"] = tx.groupby("approver_id").cumcount()
    tx["triad_prior_n"] = tx.groupby(["requester_id", "approver_id", "debtor_id"]).cumcount()
    tx["pair_share_of_requester"] = np.where(tx.req_prior_n > 0, tx.pair_prior_n / tx.req_prior_n.clip(lower=1), 0)
    tx["pair_share_of_approver"] = np.where(tx.appr_prior_n > 0, tx.pair_prior_n / tx.appr_prior_n.clip(lower=1), 0)
    # pinjaman ke debitur yang sama dalam 30 hari sebelumnya
    s = tx.sort_values(["debtor_id", "request_ts"])
    code = s.debtor_id.astype("category").cat.codes.to_numpy().astype(np.int64)
    sec = (s.request_ts.astype("int64") // 10**9).to_numpy()
    key = code * 10**10 + sec
    cnt = np.arange(len(s)) - np.searchsorted(key, key - 30 * 86400, side="left")
    tx.loc[s.index, "debtor_loans_30d"] = cnt
    return tx


FEATS = ["limit_ratio", "near_limit", "night_approval", "lag_min", "fast_approval", "debtor_tenure_days",
         "same_unit", "outside_routing", "debtor_region_mismatch", "pair_prior_n", "req_prior_n",
         "triad_prior_n", "pair_share_of_requester", "pair_share_of_approver", "debtor_loans_30d", "amount"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", required=True)
    ap.add_argument("--raw", default=None)
    ap.add_argument("--report", default=None)
    args = ap.parse_args()
    T = load(args.clean)
    r, a, e, d, l, f = (T[k] for k in ["requests", "approvals", "employees", "debtors", "access_logs", "fraud_cases"])
    E = e.set_index("employee_id")
    ok_all = True

    out("=" * 90)
    out("1. KONSISTENSI TABEL LABEL vs LOG  (cacat v4: approver label ≠ approver log)")
    out("=" * 90)
    j = f.merge(a, on="request_id", suffixes=("", "_log")).merge(r[["request_id", "timestamp"]], on="request_id")
    for c in ["requester_id", "approver_id", "debtor_id"]:
        ok_all &= check(f"fraud_cases.{c} == log", (j[c] == j[f"{c}_log"]).all(), f"{(j[c] == j[c + '_log']).mean():.2%} cocok")
    ok_all &= check("case_date == tanggal request", (j.case_date == j.timestamp_y.dt.normalize()).all()
                    if "timestamp_y" in j else (j.case_date == j.timestamp.dt.normalize()).all())
    out(pd.crosstab(f.pattern, f.fraud_label).to_string())
    out(f"Positive rate: {f.fraud_label.mean():.2%}  (rasio 1 : {(1 - f.fraud_label.mean()) / f.fraud_label.mean():.1f})")

    out("\n" + "=" * 90)
    out("2. INTEGRITAS WAKTU & REFERENSI")
    out("=" * 90)
    tx = r.merge(a[["request_id", "approver_id", "timestamp"]], on="request_id", suffixes=("", "_a"))
    ok_all &= check("approval tidak pernah sebelum request", (tx.timestamp_a.dropna() >= tx.loc[tx.timestamp_a.notna(), "timestamp"]).all())
    ok_all &= check("request tidak sebelum pegawai bergabung",
                    (tx.timestamp >= tx.requester_id.map(E.employment_start_date)).all())
    ok_all &= check("request tidak sebelum debitur jadi nasabah",
                    (tx.timestamp >= tx.debtor_id.map(d.set_index("debtor_id").customer_since)).all())
    lim = tx.approver_id.map(E.approval_limit)
    ok_all &= check("tidak ada amount > limit approver", (tx.amount <= lim).all())
    ok_all &= check("tidak ada self-approval", (tx.requester_id != tx.approver_id).all())
    ok_all &= check("approver selalu jabatan ber-limit", (lim > 0).all())
    dec = a[a.timestamp.notna()]
    logs_appr = l[l.action == "Approve"]
    m = dec.merge(logs_appr, left_on=["approver_id", "timestamp"], right_on=["employee_id", "timestamp"], how="left")
    ok_all &= check("setiap keputusan approval punya jejak di access_logs", m.log_id.notna().all(),
                    f"{m.log_id.notna().mean():.2%}")
    pend = a[a.decision == "Pending"]
    ok_all &= check("Pending hanya di 20 hari terakhir & tanpa timestamp",
                    pend.timestamp.isna().all() and
                    (r.set_index("request_id").loc[pend.request_id, "timestamp"] > pd.Timestamp("2026-09-10")).all(),
                    f"{len(pend)} baris")
    desc_amt = r.request_description.str.extract(r"Rp([\d,]+)")[0].str.replace(",", "").astype(float)
    ok_all &= check("nominal di deskripsi == amount", (desc_amt == r.amount).all())
    pref = r.request_description.str.split(":").str[0]
    ok_all &= check("prefix deskripsi == request_purpose (cacat v4: 12%)", (pref == r.request_purpose).all())

    out("\n" + "=" * 90)
    out("3. REALISME POLA WAKTU  (cacat v4: approval merata 24 jam)")
    out("=" * 90)
    F = build_features(T)
    normal = F[~F.request_id.isin(f.loc[f.fraud_label == 1, "request_id"])]
    h = normal.approve_ts.dt.hour
    share_work = ((h >= 8) & (h < 18)).mean()
    ok_all &= check("approval normal di jam kerja 08–18 ≥ 85%", share_work >= 0.85, f"{share_work:.1%}")
    out(f"   approval normal pukul 19:00–06:59 atau akhir pekan: {normal.night_approval.mean():.2%} (lembur closing yang sah)")
    hq = (F.request_ts.dt.hour.value_counts(normalize=True).sort_index() * 100).round(1)
    out("   distribusi jam request (%): " + ", ".join(f"{k:02d}:{v}" for k, v in hq.items()))

    out("\n" + "=" * 90)
    out("4. PASANGAN NORMAL JUGA BERULANG  (cacat v4: normal median 1×, kolusi ±120×)")
    out("=" * 90)
    pair_total = F.groupby(["requester_id", "approver_id"]).size()
    lab = F.merge(f[["request_id", "fraud_label", "pattern"]], on="request_id")
    lab["pair_total"] = [pair_total[(q, p)] for q, p in zip(lab.requester_id, lab.approver_id)]
    desc = lab.groupby("fraud_label").pair_total.describe(percentiles=[.1, .5, .9])[["10%", "50%", "90%", "max"]]
    out(desc.to_string())
    overlap = (lab.loc[lab.fraud_label == 0, "pair_total"] >= lab.loc[lab.fraud_label == 1, "pair_total"].median()).mean()
    ok_all &= check("pasangan normal sering berulang (median ≥ 10×)", desc.loc[0, "50%"] >= 10, f"median {desc.loc[0, '50%']:.0f}×")
    ok_all &= check("rentang pasangan normal & kolusi tumpang tindih", overlap >= 0.2,
                    f"{overlap:.0%} kasus normal punya pasangan ≥ median kasus kolusi")

    out("\n" + "=" * 90)
    out("5. TIDAK ADA FITUR TUNGGAL YANG MEMISAHKAN SEMPURNA  (cacat v4: lag>4j = 98% fraud, pair_n PR-AUC 0,96)")
    out("=" * 90)
    rows = []
    for c in FEATS:
        x = lab[c].fillna(lab[c].median())
        auc = roc_auc_score(lab.fraud_label, x)
        auc = max(auc, 1 - auc)
        pr = max(average_precision_score(lab.fraud_label, x), average_precision_score(lab.fraud_label, -x))
        rows.append((c, auc, pr))
    sf = pd.DataFrame(rows, columns=["fitur", "ROC-AUC", "PR-AUC"]).sort_values("PR-AUC", ascending=False)
    out(sf.round(3).to_string(index=False))
    out(f"   (baseline PR-AUC acak = positive rate = {lab.fraud_label.mean():.3f})")
    ok_all &= check("PR-AUC fitur tunggal terbaik < 0,60", sf["PR-AUC"].max() < 0.60, f"maks {sf['PR-AUC'].max():.3f}")
    lagb = pd.cut(lab.lag_min / 60, [-0.01, 1, 4, 8, 24, 1e4], labels=["0–1j", "1–4j", "4–8j", "8–24j", ">24j"])
    rate = lab.groupby(lagb, observed=True).fraud_label.agg(["mean", "size"])
    out("   fraud rate per lag approval:\n" + rate.round(3).to_string())
    ok_all &= check("tidak ada kelompok lag dengan fraud rate > 60%", rate["mean"].max() < 0.6, f"maks {rate['mean'].max():.0%}")

    out("\n" + "=" * 90)
    out("6. BASELINE MODEL (Random Forest, temporal split)  — target: bagus tapi tidak sempurna")
    out("=" * 90)
    lab["year"] = lab.request_ts.dt.year
    tr, va, te = lab[lab.year <= 2024], lab[lab.year == 2025], lab[lab.year == 2026]
    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced", n_jobs=-1, random_state=0)
    X = lambda df: df[FEATS].fillna(-1)
    rf.fit(X(tr), tr.fraud_label)
    res = []
    for nm, df in [("train 2021–24", tr), ("val 2025", va), ("test 2026", te)]:
        p = rf.predict_proba(X(df))[:, 1]
        res.append((nm, len(df), df.fraud_label.mean(), roc_auc_score(df.fraud_label, p), average_precision_score(df.fraud_label, p)))
    res = pd.DataFrame(res, columns=["split", "n", "pos_rate", "ROC-AUC", "PR-AUC"])
    out(res.round(3).to_string(index=False))
    te_pr = res.loc[2, "PR-AUC"]
    ok_all &= check("PR-AUC test di rentang realistis 0,55–0,90", 0.55 <= te_pr <= 0.90, f"{te_pr:.3f}")
    imp = pd.Series(rf.feature_importances_, FEATS).sort_values(ascending=False)
    out("   feature importance:\n" + imp.round(3).to_string())
    ok_all &= check("tidak ada satu fitur yang mendominasi (importance < 0,35)", imp.max() < 0.35, f"maks {imp.idxmax()} = {imp.max():.2f}")
    te = te.assign(p=rf.predict_proba(X(te))[:, 1])
    thr = np.quantile(te.p, 1 - te.fraud_label.mean())
    out("   recall per pola di test (threshold = top-k sebesar positive rate):")
    out(te[te.fraud_label == 1].groupby("pattern").apply(lambda g: (g.p >= thr).mean()).round(2).to_string())
    out("   false positive rate per skenario normal di test:")
    out(te[te.fraud_label == 0].groupby("pattern").apply(lambda g: (g.p >= thr).mean()).round(3).to_string())

    if args.raw:
        out("\n" + "=" * 90)
        out("7. DATA MENTAH: GANGGUAN YANG DISUNTIKKAN BISA DIPULIHKAN")
        out("=" * 90)
        R = {t: pd.read_csv(f"{args.raw}/{t}.csv", dtype=str, keep_default_na=False) for t in T}
        rr = R["requests"].drop_duplicates()
        miss = rr.amount == ""
        amt = rr.loc[miss, "request_description"].str.extract(r"Rp([\d,]+)")[0]
        ok_all &= check("amount kosong bisa diambil dari deskripsi", amt.notna().all(), f"{amt.notna().sum()}/{miss.sum()}")
        aa = R["approvals"].drop_duplicates()
        miss = aa.approver_id == ""
        ll = R["access_logs"]
        ap_logs = ll[(ll.action == "Approve")]
        mm = aa[miss].merge(ap_logs, on="timestamp", how="left")
        rec = mm.groupby("approval_id").employee_id.nunique()
        ok_all &= check("approver_id kosong bisa dipulihkan dari access_logs", (rec >= 1).mean() > 0.99,
                        f"{(rec == 1).mean():.1%} unik, {(rec >= 1).mean():.1%} ditemukan")
        rq = rr.set_index("request_id")
        for col_r, col_a in [("status", "decision"), ("debtor_id", "debtor_id")]:
            mr = rr[col_r] == ""
            src = aa.drop_duplicates("request_id").set_index("request_id")[col_a]
            ok_all &= check(f"requests.{col_r} kosong bisa diisi dari approvals.{col_a}",
                            (src.reindex(rr.loc[mr, "request_id"]) != "").all(), f"{mr.sum()} baris")
        inj = pd.read_csv(f"{args.raw}/data_quality_injected.csv")
        out(inj.to_string(index=False))

    out("\n" + ("SEMUA CEK LULUS" if ok_all else "ADA CEK YANG GAGAL"))
    if args.report:
        open(args.report, "w", encoding="utf-8").write("\n".join(LINES))
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
