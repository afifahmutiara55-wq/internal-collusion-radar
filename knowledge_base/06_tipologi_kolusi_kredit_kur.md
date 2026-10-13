---
doc_id: 06_tipologi_kolusi_kredit_kur
title: Tipologi kolusi requester–approver pada kredit KUR dan kembaran sahnya (hard negative)
category: internal
source: Analisis tim Collusion Radar atas data sintetis v6.1 dan praktik umum audit kredit
url: internal://collusion-radar/tipologi
last_verified: 2026-10-11
disclaimer: Tipologi disusun untuk prototype berbasis data sintetis. Pola nyata di bank dapat berbeda dan harus divalidasi unit anti-fraud.
---

# Tipologi kolusi kredit KUR

## Siapa yang terlibat

- **Requester**: Relationship Manager (RM) atau Credit Analyst yang menyiapkan dan mengajukan kredit.
- **Approver**: pejabat ber-limit kewenangan (AMGR, AVP/MGR, Business Branch Manager, Branch Manager) yang memutus.
- **Debitur**: bisa debitur sungguhan, debitur titipan (nominee), atau debitur fiktif.

Kolusi terjadi bila requester dan approver bekerja sama sehingga keputusan kredit tidak lagi independen. Satu "lingkaran"
kolusi biasanya terdiri dari satu approver dan beberapa requester yang berulang kali bekerja sama.

## Enam pola kolusi

| Pola | Ciri di data | Sinyal utama |
|---|---|---|
| **Near Limit** | nominal diatur sedikit di bawah limit approver kenalan (95–99%) supaya tidak perlu eskalasi | rule 1, `limit_ratio` |
| **Split Loan** | satu kebutuhan dipecah menjadi beberapa pengajuan ke debitur yang sama dalam waktu berdekatan | rule 4, `debtor_loans_1y` |
| **After Hours** | approval malam hari atau akhir pekan saat pengawasan longgar | rule 2, jam approval, log login malam |
| **Routing Bypass** | requester rutin memakai approver di luar jalur routing unitnya | rule 5, `pair_dependency_lift`, beda unit |
| **Nominee Debtor** | debitur titipan/fiktif, sering baru menjadi nasabah dan dipakai beberapa RM | debitur baru, debitur bersama di graph |
| **Multiple Red Flags** | kombinasi beberapa sinyal sekaligus | `rule_score` ≥ 2 |

Pelaku tidak selalu konsisten: satu transaksi kolusi sering hanya membawa sebagian sinyalnya. Itulah sebabnya sistem
memakai banyak sinyal sekaligus dan pola hubungan antarorang.

## Kembaran sah (hard negative) — jangan langsung dicurigai

| Situasi sah | Mirip dengan | Pembeda |
|---|---|---|
| Lembur closing akhir bulan | After Hours | umumnya selesai ±21:00, banyak pegawai lain juga aktif, tidak terkonsentrasi pada satu pasangan |
| Delegasi saat approver cuti | Routing Bypass | sesekali, ada catatan cuti, rasio pasangan kecil |
| Plt (pelaksana tugas) dari unit lain | Routing Bypass | periode terbatas dan resmi |
| Pengalihan portofolio antar-RM | debitur bersama | ada memo pengalihan, terjadi sekali |
| Top-up debitur lama | approval cepat, debitur berulang | debitur lama dengan riwayat lancar, sekali setahun |
| Nominal bulat tepat di limit | Near Limit | tepat 100% (angka bulat), bukan 95–99% |

## Tanda perilaku pegawai (untuk konteks investigasi, bukan fitur model)

Literatur fraud menyebut tanda seperti gaya hidup melebihi penghasilan, kesulitan keuangan, menolak cuti atau rotasi,
dan kedekatan yang tidak wajar dengan nasabah tertentu. Tanda ini **tidak** dipakai model, tetapi dapat menjadi bahan
klarifikasi oleh investigator.

## Masalah bisnis yang terlihat di data prototype

1. Proporsi kasus kolusi pada kasus yang direview meningkat tajam sejak 2021.
2. Kasus terkonsentrasi di klaster unit tertentu (koridor Bekasi) — ciri jaringan.
3. Kasus menumpuk di lima hari kerja terakhir setiap bulan (tekanan target).
4. Kredit kolusi jauh lebih sering macet dibanding kredit normal.
5. Tanpa sistem, kolusi baru terlihat sekitar 8 bulan setelah dana cair, yaitu saat kredit menjadi NPL.

Angka terkini untuk setiap temuan tersedia di dashboard dan notebook (Bagian 1, EDA Q3–Q4).
