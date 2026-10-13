#!/usr/bin/env python3
"""
KUR Internal Collusion Radar — Synthetic Dataset Generator v6.1
===============================================================

Membangkitkan 7 tabel (employees, debtors, requests, approvals, access_logs, fraud_cases,
loan_performance)
dari SATU simulasi yang konsisten. Semua tabel diturunkan dari kejadian yang sama, sehingga:

  * fraud_cases selalu menunjuk ke request/approval yang benar-benar ada di log
    (requester, approver, debitur, dan tanggal identik dengan log);
  * approval dirouting sesuai matriks kewenangan KUR (limit jabatan + eskalasi ke KC induk),
    sehingga pasangan requester–approver NORMAL juga berulang (RM → manager di unitnya);
  * aktivitas normal terkonsentrasi di jam kerja, dengan lembur akhir bulan yang sah;
  * setiap approval meninggalkan jejak di access_logs (Login + Approve) pada jam yang sama;
  * pola kolusi tidak "rapi": tiap transaksi kolusi hanya membawa SEBAGIAN sinyal,
    dan ada hard negatives (kasus normal yang mirip kolusi).

Lapisan "data kotor" (missing, duplikat, variasi kapitalisasi) ditambahkan SETELAH simulasi,
dengan aturan bahwa setiap nilai yang dihapus masih bisa dipulihkan dari tabel lain
(kecuali yang memang didokumentasikan tidak bisa). Matikan dengan --no-noise.

Pemakaian:
    python generate_kur_dataset.py --out data            # data mentah (dengan noise)
    python generate_kur_dataset.py --out data_clean --no-noise

v6.1 (untuk notebook v9): + kolom `employee_name` (nama fiktif) di employees, + tabel `loan_performance`
(kolektibilitas per 30 Sep 2026, right-censored), default 90 ring.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------
# 0. Parameter global
# --------------------------------------------------------------------------------------
SEED = 2026
START = pd.Timestamp("2021-01-01")
END = pd.Timestamp("2026-09-30")              # data berhenti sebelum "hari ini" (Okt 2026)

LIMIT = {
    "Sub Branch Manager (AMGR)": 150_000_000,
    "Sub Branch Manager (AVP/MGR)": 250_000_000,
    "Business Branch Manager": 400_000_000,
    "Branch Manager": 1_000_000_000,
}
TIERS = list(LIMIT)                             # urut dari limit terkecil

PLAFON_MAX = 500_000_000                        # plafon KUR Kecil
PRODUCTS = [("KUR Super Mikro", 10_000_000), ("KUR Mikro", 100_000_000), ("KUR Kecil", 500_000_000)]

# --- cerita masalah bisnis (diatur di sini) -------------------------------------------
HOTSPOT_UNITS = ["Bekasi", "Jababeka", "Cikarang Central City", "Tambun"]   # klaster koridor Bekasi
P_HOTSPOT = 0.30                    # porsi ring yang lahir di klaster hotspot
RING_START_YEAR_W = {2021: 0.05, 2022: 0.15, 2023: 0.24, 2024: 0.28, 2025: 0.21, 2026: 0.07}
RING_MONTH_END_BOOST = 2.4          # ring lebih aktif di 5 hari kerja terakhir (tekanan target)
P_DEFAULT_NORMAL = 0.035            # peluang kredit normal akhirnya macet
P_DEFAULT_RING = 0.40               # peluang kredit kolusi akhirnya macet
P_PLT = 0.015                       # approval oleh Plt dari unit lain se-region (sah)
P_HANDOVER = 0.03                   # debitur pindahan dari RM lain di unit yang sama (sah)

SECTORS = {"Perdagangan": 0.45, "Pertanian": 0.23, "Industri": 0.11, "Jasa": 0.10,
           "Akomodasi & Makan Minum": 0.07, "Perikanan": 0.04}
SECTOR_REGION_TILT = {   # pengali bobot sektor per region (komposisi ekonomi lokal)
    "DKI Jakarta": {"Perdagangan": 1.25, "Jasa": 1.8, "Akomodasi & Makan Minum": 1.6, "Pertanian": 0.15, "Perikanan": 0.3},
    "Banten": {"Industri": 1.6, "Perdagangan": 1.1, "Pertanian": 0.6},
    "Jawa Barat": {"Industri": 1.5, "Pertanian": 0.9},
    "Jawa Timur": {"Pertanian": 1.3, "Industri": 1.2},
    "Jawa Tengah & DIY": {"Pertanian": 1.4, "Akomodasi & Makan Minum": 1.3},
    "Sulawesi": {"Pertanian": 1.6, "Perikanan": 2.6, "Industri": 0.5},
    "Sumatera": {"Pertanian": 1.8, "Perikanan": 1.4, "Industri": 0.6},
    "Nusa Tenggara": {"Pertanian": 1.5, "Perikanan": 2.2, "Akomodasi & Makan Minum": 1.8, "Industri": 0.4},
}
PURPOSE_BY_SECTOR = {  # Modal Kerja, Stok, Bahan Baku, Peralatan, Kendaraan, Pengembangan, Renovasi, Tempat
    "Perdagangan":             [0.50, 0.28, 0.02, 0.04, 0.06, 0.04, 0.03, 0.03],
    "Pertanian":               [0.55, 0.02, 0.20, 0.12, 0.04, 0.04, 0.01, 0.02],
    "Industri":                [0.45, 0.03, 0.22, 0.20, 0.04, 0.03, 0.01, 0.02],
    "Jasa":                    [0.50, 0.02, 0.02, 0.22, 0.10, 0.05, 0.05, 0.04],
    "Akomodasi & Makan Minum": [0.48, 0.05, 0.14, 0.15, 0.02, 0.05, 0.07, 0.04],
    "Perikanan":               [0.55, 0.02, 0.12, 0.18, 0.05, 0.05, 0.01, 0.02],
}
PURPOSE_ORDER = ["Modal Kerja", "Pembelian Stok", "Pembelian Bahan Baku", "Peralatan Usaha", "Kendaraan Usaha",
                 "Pengembangan Usaha", "Renovasi Usaha", "Tempat Usaha"]
WORKING_CAPITAL = {"Modal Kerja", "Pembelian Stok", "Pembelian Bahan Baku"}

# nama outlet: nama KC/KCP BNI dari informasi publik; penempatan pegawai sepenuhnya sintetis
KC = {
    "Jakarta Kota": "DKI Jakarta", "Jakarta Pluit": "DKI Jakarta", "Jakarta Timur": "DKI Jakarta",
    "Fatmawati": "DKI Jakarta", "BSD": "Banten", "Tangerang": "Banten", "Serang": "Banten",
    "Bogor": "Jawa Barat", "Cibinong": "Jawa Barat", "Bekasi": "Jawa Barat", "Karawang": "Jawa Barat",
    "Bandung": "Jawa Barat", "Tasikmalaya": "Jawa Barat", "Surabaya": "Jawa Timur", "Gresik": "Jawa Timur",
    "Malang": "Jawa Timur", "Jombang": "Jawa Timur", "Makassar": "Sulawesi", "Padang": "Sumatera",
    "Pekanbaru": "Sumatera", "Batam": "Sumatera",
}
KCP = {  # nama: (region, KC induk)
    "Gajah Mada": ("DKI Jakarta", "Jakarta Kota"), "JIExpo Kemayoran": ("DKI Jakarta", "Jakarta Kota"),
    "Modernland": ("Banten", "Tangerang"), "Kutabumi": ("Banten", "Tangerang"),
    "Jatiuwung": ("Banten", "Tangerang"), "Cimone": ("Banten", "Tangerang"), "Pamulang": ("Banten", "BSD"),
    "Palima": ("Banten", "Serang"), "Margonda": ("Jawa Barat", "Cibinong"),
    "Jababeka": ("Jawa Barat", "Bekasi"), "Cikarang Central City": ("Jawa Barat", "Bekasi"),
    "Tambun": ("Jawa Barat", "Bekasi"), "Cikampek": ("Jawa Barat", "Karawang"),
    "Rengasdengklok": ("Jawa Barat", "Karawang"), "Juanda": ("Jawa Timur", "Surabaya"),
    "Sutomo": ("Jawa Timur", "Surabaya"), "Dieng": ("Jawa Timur", "Malang"),
    "Kanaka": ("Sulawesi", "Makassar"), "Arief Rate": ("Sulawesi", "Makassar"),
    "Sandubaya": ("Nusa Tenggara", "Surabaya"), "Wonosari": ("Jawa Tengah & DIY", "Malang"),
    "Sidokarto": ("Jawa Tengah & DIY", "Malang"),
}
REGIONS = sorted(set(KC.values()) | {r for r, _ in KCP.values()})

FIRST = ["Andi", "Budi", "Citra", "Dewi", "Eko", "Fajar", "Gita", "Hadi", "Indra", "Joko", "Kartika",
         "Lina", "Maya", "Nadia", "Oscar", "Putri", "Rina", "Sari", "Taufik", "Wahyu", "Agus", "Bayu",
         "Dimas", "Endang", "Fitri", "Hendra", "Intan", "Lukman", "Mega", "Nur", "Rizky", "Siti", "Teguh",
         "Yanti", "Yusuf", "Ayu", "Dian", "Galih", "Ratna", "Slamet"]
LAST = ["Santoso", "Rahmawati", "Wijaya", "Wibowo", "Saputra", "Kurniawan", "Siregar", "Maulana", "Permata",
        "Nugroho", "Pratama", "Lestari", "Firmansyah", "Hidayat", "Setiawan", "Hakim", "Halim", "Gunawan",
        "Susanto", "Purnomo", "Nasution", "Simanjuntak", "Daulay", "Ramadhan", "Utami", "Anggraini",
        "Syahputra", "Harahap", "Tanjung", "Kusuma"]

PURPOSE = {
    "Modal Kerja": ("penambahan modal kerja untuk operasional usaha",),
    "Pembelian Bahan Baku": ("pembelian bahan baku produksi",),
    "Pembelian Stok": ("penambahan stok barang dagangan",),
    "Peralatan Usaha": ("pembelian peralatan untuk menunjang kegiatan usaha",),
    "Kendaraan Usaha": ("pembelian kendaraan untuk mendukung kegiatan usaha",),
    "Renovasi Usaha": ("renovasi tempat usaha",),
    "Tempat Usaha": ("sewa atau pembelian tempat usaha",),
    "Pengembangan Usaha": ("pengembangan kapasitas dan perluasan kegiatan usaha",),
}
GOALS = ["untuk menjaga kelancaran operasional usaha", "untuk mendukung peningkatan volume penjualan",
         "untuk memenuhi kebutuhan usaha menjelang periode permintaan tinggi",
         "untuk memperluas kapasitas layanan kepada pelanggan", "untuk meningkatkan efisiensi produksi"]

# pola kolusi (nama dipakai di fraud_cases.pattern)
RING_PATTERNS = {               # bobot kemunculan
    "Near Limit": 0.28,
    "Split Loan": 0.14,
    "After Hours": 0.16,
    "Routing Bypass": 0.14,
    "Nominee Debtor": 0.16,
    "Multiple Red Flags": 0.12,
}

# probabilitas sinyal per transaksi kolusi (sengaja < 1 supaya tidak "rapi")
SIGNAL_P = {
    #                     near_limit after_hours fast_approval nominee
    "Near Limit":         (0.70,      0.10,       0.35,         0.10),
    "Split Loan":         (0.00,      0.10,       0.35,         0.45),
    "After Hours":        (0.10,      0.65,       0.40,         0.10),
    "Routing Bypass":     (0.10,      0.15,       0.40,         0.15),
    "Nominee Debtor":     (0.10,      0.10,       0.35,         0.85),
    "Multiple Red Flags": (0.50,      0.45,       0.55,         0.50),
}


# --------------------------------------------------------------------------------------
# helper waktu
# --------------------------------------------------------------------------------------
def business_calendar():
    days = pd.date_range(START, END, freq="D")
    cal = pd.DataFrame({"date": days})
    cal["dow"] = cal.date.dt.dayofweek
    # 3 hari kerja terakhir tiap bulan = periode closing (volume & lembur naik)
    wk = cal[cal.dow < 5].copy()
    wk["ym"] = wk.date.dt.to_period("M")
    wk["rank_from_end"] = wk.groupby("ym").date.rank(ascending=False)
    cal["month_end"] = cal.date.isin(wk.loc[wk.rank_from_end <= 3, "date"])
    cal["month_end5"] = cal.date.isin(wk.loc[wk.rank_from_end <= 5, "date"])
    years = (cal.date - START).dt.days / 365.25
    w = np.where(cal.dow < 5, 1.0, np.where(cal.dow == 5, 0.04, 0.0))
    w = w * np.where(cal.month_end, 1.6, 1.0) * (1.06 ** years)      # tren tumbuh ~6%/tahun
    cal["w"] = w
    return cal.reset_index(drop=True)


def at(day, hour_float):
    return day + pd.to_timedelta(hour_float, unit="h")


def next_business_morning(ts, rng):
    d = ts.normalize() + pd.Timedelta(days=1)
    while d.dayofweek >= 5:
        d += pd.Timedelta(days=1)
    return at(d, 8 + rng.uniform(0, 2.5))


def adjust_to_working_hours(ts, month_end_days, rng, overtime_ok):
    """Approver normal hanya memutus di jam kerja (08:00–17:30). Saat closing akhir bulan
    sebagian approver lembur sampai 21:00 (aktivitas malam yang SAH = hard negative)."""
    h = ts.hour + ts.minute / 60
    end_h = 21.0 if (overtime_ok and ts.normalize() in month_end_days) else 17.5
    if ts.dayofweek >= 5 or h >= end_h:
        return next_business_morning(ts, rng)
    if h < 8:
        return at(ts.normalize(), 8 + rng.uniform(0, 1.5))
    return ts


def round_amount(x, rng):
    r = rng.random()
    if x < 10_000_000:
        q = 500_000 if r < 0.7 else 100_000
    elif r < 0.55:
        q = 5_000_000
    elif r < 0.85:
        q = 1_000_000
    else:
        q = 1_000
    return float(max(q, np.round(x / q) * q))


# --------------------------------------------------------------------------------------
# 1. Organisasi & pegawai
# --------------------------------------------------------------------------------------
def build_employees(rng):
    rows = []
    rand_date = lambda a, b: pd.Timestamp(a) + pd.Timedelta(days=int(rng.integers(0, (pd.Timestamp(b) - pd.Timestamp(a)).days)))
    for unit in list(KC) + list(KCP):
        utype = "KC" if unit in KC else "KCP"
        if utype == "KC":
            mgrs = ["Branch Manager"] + ["Business Branch Manager"] * int(rng.integers(2, 4))
            n_rm, n_ca = int(rng.integers(24, 44)), int(rng.integers(9, 17))
        else:
            # sebagian KCP tidak punya AMGR -> satu approver menangani semua: konsentrasi tinggi yang WAJAR
            mgrs = ["Sub Branch Manager (AVP/MGR)"] + ["Sub Branch Manager (AMGR)"] * int(rng.choice([0, 1, 1, 2]))
            n_rm, n_ca = int(rng.integers(10, 21)), int(rng.integers(4, 10))
        for p in mgrs:
            rows.append(dict(unit_name=unit, unit_type=utype, position=p, approval_limit=float(LIMIT[p]),
                             employment_start_date=rand_date("2006-01-01", "2020-06-30")))
        for p, n in [("Relationship Manager", n_rm), ("Credit Analyst", n_ca)]:
            for _ in range(n):
                late = rng.random() < 0.18          # pegawai yang baru bergabung selama periode data
                d = rand_date("2021-01-01", "2026-03-31") if late else rand_date("2012-01-01", "2020-12-31")
                rows.append(dict(unit_name=unit, unit_type=utype, position=p, approval_limit=0.0,
                                 employment_start_date=d))
    emp = pd.DataFrame(rows).sample(frac=1, random_state=SEED).reset_index(drop=True)
    emp.insert(0, "employee_id", [f"E{i:05d}" for i in range(1, len(emp) + 1)])
    # nama pegawai fiktif (kombinasi unik) -> dibutuhkan dashboard investigator
    combos = [f"{a} {b}" for a in FIRST for b in LAST] + [f"{a} {m} {b}" for a in FIRST for b in LAST
                                                         for m in ["Adi", "Dwi", "Nur", "Sri", "Tri", "Putra"]]
    name_rng = np.random.default_rng(SEED + 7)          # RNG terpisah: simulasi lain tidak berubah
    picked = name_rng.choice(len(combos), size=len(emp), replace=False)
    emp.insert(1, "employee_name", [combos[i] for i in picked])
    return emp


def build_routing(emp, rng):
    """Untuk tiap requester: daftar tier yang bisa ia tuju (unit sendiri, lalu KC induk)
    beserta approver 'utama' per tier (tim tetap)."""
    mgr = emp[emp.approval_limit > 0]
    by_unit_tier = defaultdict(list)
    for r in mgr.itertuples():
        by_unit_tier[(r.unit_name, r.position)].append(r.employee_id)

    def tiers_for(unit):
        if unit in KC:
            return [(t, by_unit_tier[(unit, t)]) for t in ["Business Branch Manager", "Branch Manager"]]
        parent = KCP[unit][1]
        return ([(t, by_unit_tier[(unit, t)]) for t in ["Sub Branch Manager (AMGR)", "Sub Branch Manager (AVP/MGR)"]]
                + [(t, by_unit_tier[(parent, t)]) for t in ["Business Branch Manager", "Branch Manager"]])

    unit_tiers = {u: tiers_for(u) for u in list(KC) + list(KCP)}
    req = emp[emp.approval_limit == 0]
    primary = {}
    for r in req.itertuples():
        primary[r.employee_id] = {t: (rng.choice(lst) if lst else None) for t, lst in unit_tiers[r.unit_name]}

    # cuti approver: 2 blok/tahun, 5–10 hari kerja -> memicu delegasi (routing menyimpang yang SAH)
    leave = defaultdict(list)
    for m in mgr.employee_id:
        for y in range(START.year, END.year + 1):
            for _ in range(2):
                s = pd.Timestamp(f"{y}-01-01") + pd.Timedelta(days=int(rng.integers(0, 350)))
                leave[m].append((s, s + pd.Timedelta(days=int(rng.integers(7, 15)))))
    return unit_tiers, primary, leave


def on_leave(m, day, leave):
    return any(s <= day <= e for s, e in leave[m])


def required_tier(unit, amount, unit_tiers):
    return next(t for t, lst in unit_tiers[unit] if LIMIT[t] >= amount and lst)


def route(req_id, unit, amount, day, unit_tiers, primary, leave, rng):
    """Matriks kewenangan: tier terendah yang limitnya >= amount. Kalau approver tier itu cuti,
    didelegasikan ke rekan setier, lalu ke tier di atasnya."""
    tiers = unit_tiers[unit]
    idx = next(i for i, (t, lst) in enumerate(tiers) if LIMIT[t] >= amount and lst)
    for j in range(idx, len(tiers)):
        t, lst = tiers[j]
        avail = [m for m in lst if not on_leave(m, day, leave)]
        if not avail:
            continue
        p = primary[req_id].get(t)
        delegated = j > idx or (p is not None and p not in avail)
        if p in avail and rng.random() < 0.8:
            return p, delegated
        return rng.choice(avail), delegated
    t, lst = tiers[idx]
    return rng.choice(lst), True


def acting_approver(unit, amount, emp_by_region_tier, region_of_unit, rng):
    """Plt (pelaksana tugas) dari unit LAIN se-region dengan limit cukup -> routing menyimpang yang SAH."""
    region = region_of_unit[unit]
    pool = [(m, u) for t in TIERS if LIMIT[t] >= amount and t != "Branch Manager"
            for m, u in emp_by_region_tier.get((region, t), []) if u != unit]
    if not pool:
        return None
    return pool[int(rng.integers(len(pool)))][0]


# --------------------------------------------------------------------------------------
# 2. Simulasi transaksi normal
# --------------------------------------------------------------------------------------
class DebtorBook:
    def __init__(self, rng):
        self.rng = rng
        self.rows = []
        self.last_loan = {}

    def new(self, region, before, kind=None, since_days=(30, 2900)):
        rng = self.rng
        did = f"D{len(self.rows) + 1:06d}"
        dtype = kind or rng.choice(["Individual", "SME"], p=[0.68, 0.32])
        since = before - pd.Timedelta(days=int(rng.integers(*since_days)))
        since = max(since, pd.Timestamp("2010-01-01"))
        tilt = SECTOR_REGION_TILT.get(region, {})
        w = np.array([p * tilt.get(s, 1.0) for s, p in SECTORS.items()])
        sector = rng.choice(list(SECTORS), p=w / w.sum())
        self.rows.append(dict(debtor_id=did, debtor_name=f"{rng.choice(FIRST)} {rng.choice(LAST)}",
                              debtor_type=dtype, business_sector=sector, region=region,
                              customer_since=since.normalize()))
        return did

    def dtype(self, did):
        return self.rows[int(did[1:]) - 1]["debtor_type"]

    def sector(self, did):
        return self.rows[int(did[1:]) - 1]["business_sector"]


def draw_amount(dtype, rng):
    med, sig = {"Individual": (25e6, 0.80), "SME": (120e6, 0.65)}[dtype]
    x = float(np.clip(rng.lognormal(np.log(med), sig), 2e6, PLAFON_MAX))
    return min(round_amount(x, rng), float(PLAFON_MAX))


def request_hour(day, month_end_days, rng):
    if day.dayofweek == 5:
        return rng.uniform(8, 12.5), False
    if day in month_end_days and rng.random() < 0.25:
        return rng.uniform(17.0, 20.5), True                           # lembur closing
    return float(np.clip(rng.normal(11.5, 2.3), 7.6, 17.4)), False


def normal_approval_ts(req_ts, month_end_days, rng):
    if rng.random() < 0.07:                                            # approval cepat yang wajar
        ts = req_ts + pd.Timedelta(minutes=float(rng.uniform(3, 20)))
    else:
        ts = req_ts + pd.Timedelta(hours=float(rng.lognormal(np.log(3.0), 0.85)))
    return adjust_to_working_hours(ts, month_end_days, rng, overtime_ok=rng.random() < 0.65)


def simulate(rng):
    emp = build_employees(rng)
    unit_tiers, primary, leave = build_routing(emp, rng)
    cal = business_calendar()
    month_end_days = set(cal.loc[cal.month_end, "date"])
    E = emp.set_index("employee_id")
    region_of_unit = {**KC, **{k: v[0] for k, v in KCP.items()}}
    book = DebtorBook(rng)
    portfolio = defaultdict(list)
    recs = []
    emp_by_region_tier = defaultdict(list)
    for r in emp[emp.approval_limit > 0].itertuples():
        emp_by_region_tier[(region_of_unit[r.unit_name], r.position)].append((r.employee_id, r.unit_name))
    unit_staff = emp[emp.approval_limit == 0].groupby("unit_name").employee_id.apply(list).to_dict()

    requesters = emp[emp.approval_limit == 0]
    rate = rng.gamma(4.0, 0.55, size=len(requesters))                   # request / bulan / requester
    for (rid, row), lam in zip(requesters.set_index("employee_id").iterrows(), rate):
        start = max(START, row.employment_start_date + pd.Timedelta(days=45))
        mask = cal.date >= start
        if mask.sum() == 0:
            continue
        months = mask.sum() / 30.44
        n = rng.poisson(lam * months * (0.6 if row.position == "Credit Analyst" else 1.0))
        if n == 0:
            continue
        w = cal.w[mask].to_numpy()
        days = np.sort(rng.choice(cal.date[mask].to_numpy(), size=n, p=w / w.sum()))
        region = region_of_unit[row.unit_name]
        for d in pd.to_datetime(days):
            scen = "Normal"
            # debitur: ulangan (>= 180 hari sejak pinjaman terakhir) atau baru
            elig = [x for x in portfolio[rid] if (d - book.last_loan[x]).days >= 180]
            handover = None
            if rng.random() < P_HANDOVER:
                mates = [m for m in unit_staff.get(row.unit_name, []) if m != rid and portfolio[m]]
                if mates:
                    mate = mates[int(rng.integers(len(mates)))]
                    cands = [x for x in portfolio[mate] if (d - book.last_loan[x]).days >= 180]
                    if cands:
                        handover = cands[int(rng.integers(len(cands)))]
            if handover is not None:                                   # pengalihan portofolio yang sah
                did = handover
                portfolio[mate].remove(did)
                portfolio[rid].append(did)
                scen = "Normal - Portfolio Handover"
            elif portfolio[rid] and rng.random() < 0.02:
                did = rng.choice(portfolio[rid])                       # top-up cepat yang sah
                scen = "Normal - Top-up"
            elif elig and rng.random() < 0.7:
                did = rng.choice(elig)
            else:
                did = book.new(region if rng.random() < 0.93 else rng.choice(REGIONS), d)
                portfolio[rid].append(did)
            book.last_loan[did] = d
            amount = draw_amount(book.dtype(did), rng)
            hr, overtime = request_hour(d, month_end_days, rng)
            req_ts = at(d, hr).floor("min")
            appr, delegated = route(rid, row.unit_name, amount, d, unit_tiers, primary, leave, rng)
            plt_appr = acting_approver(row.unit_name, amount, emp_by_region_tier, region_of_unit, rng) \
                if rng.random() < P_PLT else None
            if plt_appr is not None:
                appr = plt_appr
            appr_ts = normal_approval_ts(req_ts, month_end_days, rng).floor("min")
            if scen == "Normal - Portfolio Handover":
                pass
            elif plt_appr is not None:
                scen = "Normal - Acting Approver (Plt)"
            elif overtime or (appr_ts.hour >= 18):
                scen = "Normal - Month-end Overtime"
            elif delegated:
                scen = "Normal - Leave Delegation"
            elif amount == LIMIT[E.loc[appr, "position"]]:
                scen = "Normal - Round Amount at Limit"
            recs.append(dict(requester_id=rid, debtor_id=did, amount=amount, request_ts=req_ts,
                             approver_id=appr, approve_ts=appr_ts, ring_id=None, scenario=scen,
                             dtype=book.dtype(did)))
    return emp, unit_tiers, primary, leave, cal, month_end_days, book, portfolio, recs


# --------------------------------------------------------------------------------------
# 3. Ring kolusi
# --------------------------------------------------------------------------------------
def build_rings(emp, unit_tiers, primary, n_rings, rng):
    E = emp.set_index("employee_id")
    staff = emp[emp.approval_limit == 0]
    mgr = emp[emp.approval_limit > 0]
    rings, used_req = [], set()
    names, w = list(RING_PATTERNS), np.array(list(RING_PATTERNS.values()))
    for k in range(n_rings):
        pattern = rng.choice(names, p=w / w.sum())
        years, yw = list(RING_START_YEAR_W), np.array(list(RING_START_YEAR_W.values()))
        year = int(rng.choice(years, p=yw / yw.sum()))
        y0 = max(pd.Timestamp(f"{year}-01-01"), pd.Timestamp("2021-02-01"))
        y1 = min(pd.Timestamp(f"{year}-12-31"), END - pd.Timedelta(days=60))
        start = y0 + pd.Timedelta(days=int(rng.integers(0, max(1, (y1 - y0).days))))
        dur = int(rng.integers(5, 22))
        end = min(start + pd.DateOffset(months=dur), END)
        # requester pertama: RM yang sudah aktif saat ring mulai (sebagian besar di klaster hotspot)
        in_hotspot = rng.random() < P_HOTSPOT
        cand = staff[(staff.position == "Relationship Manager")
                     & (staff.employment_start_date < start - pd.Timedelta(days=120))
                     & (~staff.employee_id.isin(used_req))
                     & (staff.unit_name.isin(HOTSPOT_UNITS) == in_hotspot)]
        r0 = cand.sample(1, random_state=int(rng.integers(1e9))).iloc[0]
        if pattern == "Routing Bypass":
            # approver di LUAR jalur routing requester (bukan unitnya, bukan KC induknya)
            own = {m for _, lst in unit_tiers[r0.unit_name] for m in lst}
            pool = mgr[~mgr.employee_id.isin(own) & (mgr.position != "Branch Manager")]
            appr = pool.sample(1, random_state=int(rng.integers(1e9))).iloc[0].employee_id
        else:
            # tier ditentukan dari nominal "khas" requester -> colluder = approver yang memang biasa ia temui
            t = required_tier(r0.unit_name, draw_amount(rng.choice(["Individual", "SME"], p=[0.68, 0.32]), rng), unit_tiers)
            if t == "Branch Manager":                                   # nominal ≤ plafon: BM jarang jadi colluder
                t = "Business Branch Manager"
            lst = dict(unit_tiers[r0.unit_name])[t]
            appr = primary[r0.employee_id][t] if rng.random() < 0.5 else rng.choice(lst)
        members = [r0.employee_id]
        # sebagian ring beranggotakan 2–3 requester dari unit yang sama (lingkaran kolusi)
        extra = int(rng.choice([0, 0, 1, 2]))
        mates = staff[(staff.unit_name == r0.unit_name) & (staff.employee_id != r0.employee_id)
                      & (staff.employment_start_date < start) & (~staff.employee_id.isin(used_req))]
        if extra and len(mates):
            members += list(mates.sample(min(extra, len(mates)), random_state=int(rng.integers(1e9))).employee_id)
        used_req.update(members)
        rings.append(dict(ring_id=f"RING{k + 1:03d}", pattern=pattern, approver_id=appr, members=members,
                          start=start, end=end, unit=r0.unit_name))
    return rings


def ring_amount(ring, unit, lim, dtype, unit_tiers, appr_tier, rng):
    """Nominal kolusi NON-near-limit diambil dari distribusi yang SAMA dengan transaksi normal,
    dibatasi pada rentang yang secara alami dirouting ke approver colluder."""
    for _ in range(60):
        x = draw_amount(dtype, rng)
        if x > lim:
            continue
        if ring["pattern"] == "Routing Bypass" or required_tier(unit, x, unit_tiers) == appr_tier:
            return x
    return round_amount(rng.uniform(0.3, 0.9) * lim, rng)


def simulate_rings(rings, emp, book, portfolio, unit_tiers, cal, month_end_days, rng):
    E = emp.set_index("employee_id")
    region_of_unit = {**KC, **{k: v[0] for k, v in KCP.items()}}
    lower = {t: (LIMIT[TIERS[i - 1]] if i else 0) for i, t in enumerate(TIERS)}
    recs = []
    for ring in rings:
        appr_tier = E.loc[ring["approver_id"], "position"]
        lim = LIMIT[appr_tier]
        p_near, p_night, p_fast, p_nom = SIGNAL_P[ring["pattern"]]
        region = region_of_unit[ring["unit"]]
        nominees = [book.new(region if rng.random() < 0.5 else rng.choice(REGIONS),
                             ring["start"], kind=rng.choice(["Individual", "SME"]), since_days=(10, 90))
                    for _ in range(int(rng.integers(3, 9)))]
        days_mask = (cal.date >= ring["start"]) & (cal.date <= ring["end"]) & (cal.dow < 5)
        ring_days = cal.date[days_mask].to_numpy()
        if len(ring_days) == 0:
            continue
        day_w = np.where(cal.month_end5[days_mask].to_numpy(), RING_MONTH_END_BOOST, 1.0)
        day_w = day_w / day_w.sum()
        for rid in ring["members"]:
            months = len(ring_days) / 21.7
            n = rng.poisson(rng.uniform(1.2, 3.4) * months)
            if ring["pattern"] == "Split Loan":
                n = max(1, n // 3)                                       # n = jumlah "burst"
            for d in pd.to_datetime(np.sort(rng.choice(ring_days, size=n, p=day_w))):
                if ring["pattern"] == "Split Loan":
                    did = rng.choice(nominees) if rng.random() < p_nom else book.new(region, d)
                    k = int(rng.integers(2, 5))
                    parts = [(d + pd.Timedelta(days=int(x))) for x in np.sort(rng.integers(0, 10, size=k))]
                    amts = [min(round(lim * rng.uniform(0.55, 0.92) / 1e5) * 1e5, PLAFON_MAX) for _ in range(k)]
                    items = list(zip(parts, amts, [did] * k))
                else:
                    if rng.random() < p_nom:
                        did = rng.choice(nominees)
                    else:                                                # debitur biasa, sama seperti alur normal
                        elig = [x for x in portfolio[rid] if (d - book.last_loan.get(x, d)).days >= 180]
                        if elig and rng.random() < 0.7:
                            did = rng.choice(elig)
                        else:
                            did = book.new(region if rng.random() < 0.93 else rng.choice(REGIONS), d)
                            portfolio[rid].append(did)
                        book.last_loan[did] = d
                    if rng.random() < p_near:
                        amt = round(lim * rng.uniform(0.95, 0.996) / 1e5) * 1e5   # sedikit di bawah limit
                    else:
                        amt = ring_amount(ring, E.loc[rid, "unit_name"], lim, book.dtype(did), unit_tiers, appr_tier, rng)
                    items = [(d, min(amt, lim, PLAFON_MAX), did)]
                for dd, amt, did_ in items:
                    if dd > END:
                        continue
                    if dd.dayofweek >= 5:
                        dd = dd - pd.Timedelta(days=dd.dayofweek - 4)
                    night = rng.random() < p_night
                    if night:
                        req_ts = at(dd, rng.uniform(16.5, 21.0) if rng.random() < 0.6
                                    else float(np.clip(rng.normal(11.5, 2.3), 7.6, 17.4)))
                        base = max(req_ts, at(dd, 19.25))
                        appr_ts = base + pd.Timedelta(minutes=float(rng.uniform(5, 200)))
                        if appr_ts.hour < 19 and appr_ts.normalize() == dd:
                            appr_ts = at(dd, rng.uniform(19.5, 23.5))
                    else:
                        hr, _ = request_hour(dd, month_end_days, rng)
                        req_ts = at(dd, hr)
                        if rng.random() < p_fast:
                            appr_ts = req_ts + pd.Timedelta(minutes=float(rng.uniform(1, 15)))
                        else:
                            appr_ts = normal_approval_ts(req_ts, month_end_days, rng)
                    recs.append(dict(requester_id=rid, debtor_id=did_, amount=float(amt), request_ts=req_ts.floor("min"),
                                     approver_id=ring["approver_id"], approve_ts=appr_ts.floor("min"),
                                     ring_id=ring["ring_id"], scenario=ring["pattern"],
                                     dtype=book.dtype(did_)))
    return recs


# --------------------------------------------------------------------------------------
# 4. Merakit tabel
# --------------------------------------------------------------------------------------
def describe(purpose, product, amount, rng):
    return (f"{purpose}: pengajuan pembiayaan {product} sebesar Rp{amount:,.0f} untuk {PURPOSE[purpose][0]}, "
            f"{rng.choice(GOALS)}.")


def product_of(amount):
    return next(name for name, cap in PRODUCTS if amount <= cap)


def draw_tenor(purpose, product, rng):
    if purpose in WORKING_CAPITAL:
        t = rng.choice([12, 18, 24, 36], p=[0.35, 0.20, 0.30, 0.15])
    else:
        t = rng.choice([24, 36, 48, 60], p=[0.20, 0.35, 0.25, 0.20])
    return int(min(t, 36) if product == "KUR Super Mikro" else t)


def assemble(emp, book, recs, rng):
    tx = pd.DataFrame(recs)
    # log sistem mencatat sampai detik
    tx["request_ts"] = tx.request_ts + pd.to_timedelta(rng.integers(0, 60, len(tx)), unit="s")
    tx["approve_ts"] = tx.approve_ts + pd.to_timedelta(rng.integers(0, 60, len(tx)), unit="s")
    bad = tx.approve_ts <= tx.request_ts
    tx.loc[bad, "approve_ts"] = tx.loc[bad, "request_ts"] + pd.to_timedelta(rng.integers(30, 120, bad.sum()), unit="s")
    tx = tx.sort_values("request_ts").reset_index(drop=True)
    tx["request_id"] = [f"R{i:07d}" for i in range(1, len(tx) + 1)]
    tx["approval_id"] = "A" + tx["request_id"].str[1:]
    sector_of = {r["debtor_id"]: r["business_sector"] for r in book.rows}
    tx["sector"] = tx.debtor_id.map(sector_of)
    tx["request_purpose"] = [PURPOSE_ORDER[rng.choice(8, p=PURPOSE_BY_SECTOR[s])] for s in tx.sector]
    tx["loan_product"] = [product_of(a) for a in tx.amount]
    tx["tenor_months"] = [draw_tenor(p, pr, rng) for p, pr in zip(tx.request_purpose, tx.loan_product)]
    tx["request_description"] = [describe(p, pr, a, rng) for p, pr, a in
                                 zip(tx.request_purpose, tx.loan_product, tx.amount)]

    # keputusan: ring hampir selalu disetujui; normal ditolak berdasar risiko sederhana
    p_rej = 0.05 + 0.05 * (tx.amount > 300e6) + 0.04 * (tx.dtype == "Individual")
    p_rej = p_rej + 0.03 * (tx.loan_product == "KUR Super Mikro")
    p_rej = np.where(tx.ring_id.notna(), 0.01, p_rej)
    decision = np.where(rng.random(len(tx)) < p_rej, "Rejected", "Approved").astype(object)
    recent = tx.request_ts > END - pd.Timedelta(days=20)
    pending = recent & (rng.random(len(tx)) < 0.55) & tx.ring_id.isna()
    decision[pending.to_numpy()] = "Pending"
    tx["decision"] = decision
    # Pending = belum diputus -> tidak ada timestamp keputusan (struktural, bukan missing)
    tx.loc[tx.decision == "Pending", "approve_ts"] = pd.NaT
    # keputusan tidak mungkin melewati akhir periode data
    late = tx.approve_ts > END + pd.Timedelta(hours=23, minutes=59)
    tx.loc[late, "decision"] = "Pending"
    tx.loc[late, "approve_ts"] = pd.NaT

    requests = pd.DataFrame({
        "request_id": tx.request_id, "loan_product": tx.loan_product, "requester_id": tx.requester_id,
        "debtor_id": tx.debtor_id, "amount": tx.amount, "tenor_months": tx.tenor_months, "timestamp": tx.request_ts,
        "status": tx.decision, "request_purpose": tx.request_purpose,
        "request_description": tx.request_description})
    approvals = pd.DataFrame({
        "approval_id": tx.approval_id, "request_id": tx.request_id, "requester_id": tx.requester_id,
        "approver_id": tx.approver_id, "debtor_id": tx.debtor_id, "decision": tx.decision,
        "timestamp": tx.approve_ts, "request_purpose": tx.request_purpose,
        "request_description": tx.request_description})
    debtors = pd.DataFrame(book.rows)
    employees = emp.copy()
    return tx, employees, debtors, requests, approvals


def build_access_logs(tx, emp, rng):
    is_mgr = set(emp.loc[emp.approval_limit > 0, "employee_id"])
    ev = []
    dec = tx[tx.approve_ts.notna()]
    ev.append(pd.DataFrame({"employee_id": dec.approver_id, "timestamp": dec.approve_ts,
                            "system": "Approval System", "action": "Approve"}))
    v = dec.sample(frac=0.15, random_state=SEED)
    vts = v.approve_ts - pd.to_timedelta(rng.uniform(1, 25, len(v)), unit="min")
    ok = vts > v.request_ts
    ev.append(pd.DataFrame({"employee_id": v.approver_id[ok], "timestamp": vts[ok].dt.floor("min"),
                            "system": "Credit System", "action": "View"}))
    ev.append(pd.DataFrame({"employee_id": tx.requester_id, "timestamp": tx.request_ts,
                            "system": "Credit System", "action": "Edit"}))
    core = pd.concat(ev, ignore_index=True)

    # aktivitas latar (CRM / Core Banking) di jam kerja, sedikit di sore/malam (sah)
    n_bg = 40_000
    emps = rng.choice(emp.employee_id.to_numpy(), size=n_bg)
    days = pd.to_datetime(rng.choice(pd.bdate_range(START, END).to_numpy(), size=n_bg))
    hrs = np.where(rng.random(n_bg) < 0.05, rng.uniform(17.5, 21.5, n_bg), np.clip(rng.normal(12, 2.4, n_bg), 7.8, 17.3))
    bg = pd.DataFrame({"employee_id": emps, "timestamp": (days + pd.to_timedelta(hrs, unit="h")).floor("min"),
                       "system": rng.choice(["CRM", "Core Banking"], size=n_bg, p=[0.45, 0.55]),
                       "action": rng.choice(["View", "Export"], size=n_bg, p=[0.8, 0.2])})
    start = emp.set_index("employee_id").employment_start_date
    bg = bg[bg.timestamp >= bg.employee_id.map(start)]
    core = pd.concat([core, bg], ignore_index=True)

    # Login: satu per (pegawai, hari, sesi siang/malam), sebelum aktivitas pertama sesi tsb
    core["session"] = np.where((core.timestamp.dt.hour >= 18) | (core.timestamp.dt.hour < 7), "night", "day")
    core["day"] = core.timestamp.dt.normalize()
    first = core.groupby(["employee_id", "day", "session"]).timestamp.min().reset_index()
    first["timestamp"] = (first.timestamp - pd.to_timedelta(rng.uniform(1, 30, len(first)), unit="min")).dt.floor("min")
    first["system"] = np.where(first.employee_id.isin(is_mgr), "Approval System", "Credit System")
    first["action"] = "Login"
    logs = pd.concat([core[["employee_id", "timestamp", "system", "action"]],
                      first[["employee_id", "timestamp", "system", "action"]]], ignore_index=True)
    logs = logs.sort_values(["timestamp", "employee_id"]).reset_index(drop=True)
    logs.insert(0, "log_id", [f"L{i:07d}" for i in range(1, len(logs) + 1)])
    return logs


def build_loan_performance(tx, rng):
    """Kinerja kredit per akhir data (30 Sep 2026). OUTCOME yang terjadi SETELAH approval.

    - Kredit cair 1-5 hari kerja setelah approval.
    - Peluang akhirnya macet: kolusi P_DEFAULT_RING, normal P_DEFAULT_NORMAL.
    - Waktu sampai NPL: kolusi lebih cepat (median ±8,5 bln, early default) vs normal (median ±18 bln).
    - Right-censoring: kalau waktu NPL > umur kredit saat ini, kredit masih terlihat lancar.
    """
    ok = tx[(tx.decision == "Approved") & tx.approve_ts.notna()].copy()
    is_ring = ok.ring_id.notna().to_numpy()
    disb = (ok.approve_ts.dt.normalize() + pd.to_timedelta(rng.integers(1, 6, len(ok)), unit="D"))
    disb = disb.where(disb <= END, END.normalize())
    mob = ((END - disb).dt.days / 30.44).to_numpy()                       # months on book
    tenor = ok.tenor_months.to_numpy().astype(float)
    p_def = np.where(is_ring, P_DEFAULT_RING, P_DEFAULT_NORMAL)
    will_default = rng.random(len(ok)) < p_def
    ttd = np.where(is_ring, rng.lognormal(np.log(8.5), 0.45, len(ok)), rng.lognormal(np.log(18), 0.55, len(ok)))
    ttd = np.clip(ttd, 2.5, np.maximum(tenor - 1, 3))
    defaulted = will_default & (ttd <= mob)
    amount = ok.amount.to_numpy()
    paid_frac = np.clip(mob / tenor, 0, 1)
    outstanding = np.where(defaulted, amount * np.clip(1 - ttd / tenor, 0.05, 1) * rng.uniform(0.9, 1.0, len(ok)),
                           amount * (1 - paid_frac))
    months_since_npl = np.where(defaulted, mob - ttd, np.nan)
    coll = np.where(defaulted, np.select([months_since_npl < 3, months_since_npl < 6], [3, 4], 5), 1)
    dpk = (~defaulted) & (paid_frac < 1) & (rng.random(len(ok)) < np.where(is_ring, 0.10, 0.035))
    coll = np.where(dpk, 2, coll)
    dpd = np.select([coll == 2, coll == 3, coll == 4, coll == 5],
                    [rng.integers(1, 90, len(ok)), rng.integers(91, 121, len(ok)), rng.integers(121, 181, len(ok)),
                     np.clip(np.nan_to_num(months_since_npl) * 30.44 + 90, 181, 2000).astype(int)], 0)
    outstanding = np.where((coll == 1) & (paid_frac >= 1), 0.0, np.round(outstanding, -3))
    first_npl = pd.Series(pd.NaT, index=ok.index, dtype="datetime64[ns]")
    first_npl[defaulted] = (disb[defaulted] + pd.to_timedelta(ttd[defaulted] * 30.44, unit="D")).dt.normalize()
    status = np.where((outstanding == 0) & (coll == 1), "Lunas",
                      pd.Series(coll).map({1: "Lancar", 2: "Dalam Perhatian Khusus", 3: "Kurang Lancar",
                                           4: "Diragukan", 5: "Macet"}).to_numpy())
    return pd.DataFrame({"request_id": ok.request_id.to_numpy(), "disbursement_date": disb.to_numpy(),
                         "tenor_months": tenor.astype(int), "outstanding_amount": outstanding,
                         "days_past_due": dpd.astype(int), "collectibility": coll.astype(int),
                         "collectibility_status": status, "first_npl_date": first_npl.to_numpy()})


def build_labels(tx, rng, n_cases=30_000):
    """Kasus berlabel = sampel request yang sudah ditinjau auditor.
    Semua entitas diambil LANGSUNG dari log (tidak ada ketidakcocokan)."""
    ring = tx[tx.ring_id.notna()]
    detected = ring.sample(frac=0.90, random_state=SEED)               # 10% kolusi tidak terdeteksi
    normal = tx[tx.ring_id.isna()]
    w = np.where(normal.scenario == "Normal", 1.0, 3.0)                 # auditor lebih sering meninjau yg mirip mencurigakan
    take = n_cases - len(detected)
    sampled = normal.sample(n=take, weights=w, random_state=SEED)
    lab = pd.concat([detected.assign(fraud_label=1), sampled.assign(fraud_label=0)])
    lab["pattern"] = lab.scenario
    # label noise yang terdokumentasi: hasil review auditor yang ambigu
    flip1 = lab.index[(lab.fraud_label == 1)].to_series().sample(frac=0.04, random_state=SEED + 1)
    flip0 = lab.index[(lab.fraud_label == 0)].to_series().sample(frac=0.004, random_state=SEED + 2)
    lab.loc[flip1, "fraud_label"] = 0
    lab.loc[flip0, "fraud_label"] = 1
    lab.loc[flip1.index.union(flip0.index), "pattern"] = "Ambiguous / Noisy Label"
    lab = lab.sort_values("request_ts").reset_index(drop=True)
    return pd.DataFrame({
        "case_id": [f"C{i:06d}" for i in range(1, len(lab) + 1)], "request_id": lab.request_id,
        "requester_id": lab.requester_id, "approver_id": lab.approver_id, "debtor_id": lab.debtor_id,
        "pattern": lab.pattern, "fraud_label": lab.fraud_label.astype(int),
        "case_date": lab.request_ts.dt.normalize()})


# --------------------------------------------------------------------------------------
# 5. Lapisan data kotor (setiap gangguan bisa dipulihkan dari tabel lain)
# --------------------------------------------------------------------------------------
def inject_noise(T, rng):
    rep = []
    T = {k: v.copy() for k, v in T.items()}
    r, a, f, e, d, l = (T[k] for k in ["requests", "approvals", "fraud_cases", "employees", "debtors", "access_logs"])

    # pilih request yang BERBEDA untuk tiap jenis gangguan -> setiap request maksimal 1 gangguan
    perm = rng.permutation(len(r))
    cursor = [0]

    def take(frac):
        n = int(len(r) * frac)
        idx = perm[cursor[0]:cursor[0] + n]
        cursor[0] += n
        return idx

    def blank(df, idx, col, why):
        df[col] = df[col].astype(object)
        df.loc[df.index[idx], col] = np.nan
        rep.append(dict(table=name_of(df), column=col, rows=len(idx), type="missing", recoverable_from=why))

    names = {id(r): "requests", id(a): "approvals", id(f): "fraud_cases", id(e): "employees",
             id(d): "debtors", id(l): "access_logs"}
    name_of = lambda df: names[id(df)]

    blank(r, take(0.005), "amount", "angka di request_description (100% konsisten)")
    blank(r, take(0.005), "debtor_id", "approvals.debtor_id")
    blank(r, take(0.003), "timestamp", "tidak dipulihkan; pakai approvals.timestamp / access_logs Edit")
    blank(r, take(0.003), "status", "approvals.decision")
    blank(r, take(0.004), "request_description", "approvals.request_description")
    blank(a, take(0.004), "approver_id", "access_logs (action=Approve, timestamp sama)")
    blank(a, take(0.003), "decision", "requests.status")
    blank(a, take(0.003), "timestamp", "access_logs (action=Approve oleh approver yang sama)")
    blank(a, take(0.004), "request_description", "requests.request_description")
    # variasi penulisan
    for df, col, frac, fn in [(r, "status", 0.01, str.lower), (a, "decision", 0.008, str.upper),
                              (r, "request_purpose", 0.008, str.upper), (a, "request_purpose", 0.008, str.upper)]:
        idx = take(frac)
        df.loc[df.index[idx], col] = df.loc[df.index[idx], col].map(fn)
        rep.append(dict(table=name_of(df), column=col, rows=len(idx), type="inconsistent_case", recoverable_from="standarisasi"))

    # tabel label: kolom kosong bisa diisi dari log lewat request_id
    fperm = rng.permutation(len(f))
    for col, n, why in [("approver_id", 0.004, "approvals via request_id"), ("debtor_id", 0.005, "requests via request_id"),
                        ("case_date", 0.003, "requests.timestamp via request_id")]:
        idx, fperm = fperm[:int(len(f) * n)], fperm[int(len(f) * n):]
        blank(f, idx, col, why)

    # employees: jabatan <-> limit saling melengkapi
    eperm = rng.permutation(len(e))
    blank(e, eperm[:8], "position", "approval_limit (limit>0 -> jabatan unik; limit=0 -> staf)")
    blank(e, eperm[8:16], "approval_limit", "position (1 jabatan = 1 limit)")

    # debtors
    dperm = rng.permutation(len(d))
    n1 = int(len(d) * 0.01)
    blank(d, dperm[:n1], "debtor_name", "tidak dipulihkan (PII, bukan fitur)")
    blank(d, dperm[n1:2 * n1], "region", "region unit requester yang melayani debitur")
    i3 = dperm[2 * n1:2 * n1 + int(len(d) * 0.008)]
    d.loc[d.index[i3], "debtor_type"] = "unknown"
    rep.append(dict(table="debtors", column="debtor_type", rows=len(i3), type="placeholder", recoverable_from="tidak; jadikan kategori Unknown"))
    i4 = dperm[2 * n1 + len(i3):2 * n1 + len(i3) + int(len(d) * 0.008)]
    d.loc[d.index[i4], "debtor_type"] = d.loc[d.index[i4], "debtor_type"] + " "
    rep.append(dict(table="debtors", column="debtor_type", rows=len(i4), type="whitespace", recoverable_from="strip"))

    # access logs: JANGAN ganggu log Approve (sumber pemulihan approvals)
    non_appr = np.flatnonzero(l.action.to_numpy() != "Approve")
    lp = rng.permutation(non_appr)
    nl = int(len(l) * 0.003)
    blank(l, lp[:nl], "employee_id", "tidak; baris dibuang saat cleaning")
    blank(l, lp[nl:2 * nl], "system", "Login approver=Approval System, Login staf=Credit System; lainnya Unknown")
    blank(l, lp[2 * nl:3 * nl], "timestamp", "tidak; baris dibuang saat cleaning")

    # duplikat persis (re-ingest), disisipkan di posisi acak
    for k, frac in [("employees", 0.003), ("debtors", 0.01), ("requests", 0.006), ("approvals", 0.007), ("access_logs", 0.006)]:
        df = T[k] if k not in ("requests", "approvals", "employees", "debtors", "access_logs") else \
            {"requests": r, "approvals": a, "employees": e, "debtors": d, "access_logs": l}[k]
        dup = df.sample(frac=frac, random_state=SEED + len(k))
        out = pd.concat([df, dup]).sample(frac=1, random_state=SEED).sort_index(kind="stable")
        T[k] = out.reset_index(drop=True)
        rep.append(dict(table=k, column="(baris)", rows=len(dup), type="exact_duplicate", recoverable_from="drop_duplicates"))
    T["fraud_cases"] = f
    # loan_performance: status kolektibilitas kosong -> pulihkan dari collectibility (1-5) + outstanding
    lpf = T["loan_performance"]
    lperm = rng.permutation(len(lpf))[:int(len(lpf) * 0.01)]
    lpf["collectibility_status"] = lpf["collectibility_status"].astype(object)
    lpf.loc[lpf.index[lperm], "collectibility_status"] = np.nan
    rep.append(dict(table="loan_performance", column="collectibility_status", rows=len(lperm), type="missing",
                    recoverable_from="collectibility (1-5) + outstanding_amount (0 = Lunas)"))
    return T, pd.DataFrame(rep)


# --------------------------------------------------------------------------------------
# 6. main
# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--no-noise", action="store_true")
    ap.add_argument("--rings", type=int, default=90)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    emp, unit_tiers, primary, leave, cal, month_end_days, book, portfolio, recs = simulate(rng)
    rings = build_rings(emp, unit_tiers, primary, args.rings, rng)
    recs += simulate_rings(rings, emp, book, portfolio, unit_tiers, cal, month_end_days, rng)
    tx, employees, debtors, requests, approvals = assemble(emp, book, recs, rng)
    logs = build_access_logs(tx, emp, rng)
    labels = build_labels(tx, rng)
    performance = build_loan_performance(tx, rng)

    T = {"employees": employees, "debtors": debtors, "requests": requests, "approvals": approvals,
         "access_logs": logs, "fraud_cases": labels, "loan_performance": performance}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    noise_rep = None
    if not args.no_noise:
        T, noise_rep = inject_noise(T, np.random.default_rng(args.seed + 99))
        noise_rep.to_csv(out / "data_quality_injected.csv", index=False)

    fmt = {"employees": {"employment_start_date": "%Y-%m-%d"}, "debtors": {"customer_since": "%Y-%m-%d"},
           "requests": {"timestamp": "%Y-%m-%d %H:%M:%S"}, "approvals": {"timestamp": "%Y-%m-%d %H:%M:%S"},
           "access_logs": {"timestamp": "%Y-%m-%d %H:%M:%S"}, "fraud_cases": {"case_date": "%Y-%m-%d"},
           "loan_performance": {"disbursement_date": "%Y-%m-%d", "first_npl_date": "%Y-%m-%d"}}
    for name, df in T.items():
        df = df.copy()
        for col, f_ in fmt[name].items():
            s = pd.to_datetime(df[col], errors="coerce")
            df[col] = s.dt.strftime(f_).where(s.notna(), "")
        df.to_csv(out / f"{name}.csv", index=False)

    # simpan kebenaran internal (untuk validasi), TIDAK untuk modeling
    if args.no_noise:
        tx.to_csv(out / "_truth_transactions.csv", index=False)
        pd.DataFrame([{**r, "members": ";".join(r["members"])} for r in rings]).to_csv(out / "_truth_rings.csv", index=False)
    summary = {name: len(df) for name, df in T.items()}
    summary["rings"] = len(rings)
    summary["ring_transactions"] = int(tx.ring_id.notna().sum())
    summary["label_positive_rate"] = float(labels.fraud_label.mean())
    perf = performance.merge(tx[["request_id", "ring_id"]], on="request_id")
    summary["npl_rate_ring"] = float((perf[perf.ring_id.notna()].collectibility >= 3).mean())
    summary["npl_rate_normal"] = float((perf[perf.ring_id.isna()].collectibility >= 3).mean())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
