---
doc_id: 07_kamus_rules_dan_fitur
title: Kamus 5 rules, fitur graph, dan fitur Isolation Forest pada Collusion Radar
category: internal
source: Notebook Internal Collusion Radar v9 (Bagian 1.8, 2.3, 2.4)
url: internal://collusion-radar/kamus-fitur
last_verified: 2026-10-11
disclaimer: Definisi mengikuti konfigurasi notebook v9. Bila ambang diubah, dokumen ini harus diperbarui.
---

# Kamus rules dan fitur

## Lima rules dasar

Rules dihitung dari log transaksi tanpa memakai label. Ambangnya dikalibrasi sekali di data latih 2021–2024.

| Rule | Definisi | Alasan ambang | Kembaran sah |
|---|---|---|---|
| **R1 `rf_near_limit`** | nominal ≥ 95% dan < 100% limit kewenangan approver | 90% menambah alarm palsu; 100% (angka bulat) lazim | nominal bulat tepat di limit |
| **R2 `rf_after_hours`** | approval pukul 19:00–06:59 atau Sabtu/Minggu | jam 17–19 masih ramai lembur closing yang sah | lembur akhir bulan |
| **R3 `rf_fast_approval`** | approval kurang dari 20 menit setelah pengajuan | kompromi antara precision (10 menit) dan recall (60 menit) | top-up debitur lama |
| **R4 `rf_debtor_repeat`** | ≥ 2 pengajuan lain ke debitur yang sama dalam 1 tahun terakhir | 1×/tahun (perpanjangan, top-up) itu normal | perpanjangan tahunan |
| **R5 `rf_pair_dependency`** | `requester_approver_ratio` ≥ 10%, minimal 5 pengajuan pasangan setahun, dan ≥ 5× porsi approver itu di unit requester | rasio mentah tidak membedakan karena RM memang bergantung pada manager unitnya | Plt/delegasi cuti (sesekali) |

`rule_score` = jumlah rule yang terpenuhi (0–5). Rules dipakai sebagai **baseline** dan **alasan**, bukan skor akhir.

### Rumus indikator hubungan

- `requester_approver_ratio` = jumlah pengajuan pasangan (requester → approver) ÷ total pengajuan requester, dalam 1 tahun
  sebelum transaksi.
- `unit_approver_share` = porsi approver tersebut di seluruh pengajuan unit requester (1 tahun).
- `pair_dependency_lift` = `requester_approver_ratio` ÷ `unit_approver_share`. Lift 10× berarti requester memakai approver
  itu 10 kali lebih sering daripada kebiasaan rekan satu unitnya.
- `pair_approval_rate` = pengajuan pasangan yang disetujui ÷ seluruh keputusan pasangan (fitur model, bukan rule).

## Fitur graph per transaksi (input classifier)

| Fitur | Arti |
|---|---|
| `g_pair_weight` | frekuensi hubungan requester–approver dalam 12 bulan |
| `g_appr_wdeg` | total transaksi approver dalam 12 bulan |
| `g_debtor_n_requesters` | jumlah requester berbeda yang melayani debitur ini |
| `g_req_shared_debtors` | jumlah debitur requester yang juga dilayani RM lain |
| `g_risk_pair_z` | z-score red flag pasangan (0 bila wajar) |
| `g_risk_pair_weight` | bobot hubungan di graph risiko (red flag berlebih + ketergantungan) |
| `g_risk_comm_score`, `g_risk_comm_size` | skor dan ukuran komunitas risiko tempat requester berada |
| `g_same_comm` | requester dan approver berada di komunitas risiko yang sama |
| `g_appr_risk_pagerank` | sentralitas approver di jaringan berisiko (PageRank) |

Graph dibangun per bulan dari approval 12 bulan sebelumnya, tanpa label. Hubungan disimpan di graph risiko bila red flag
transaksinya jauh di atas wajar (z ≥ 3 dan minimal 3 transaksi ber-red-flag) atau memenuhi rule ketergantungan.
Requester yang berbagi debitur juga dihubungkan. Komunitas dicari dengan algoritma Louvain.

## Fitur Isolation Forest per approver-bulan

| Fitur | Arti |
|---|---|
| `ex_near_limit`, `ex_after_hours`, `ex_fast_approval`, `ex_debtor_repeat`, `ex_pair_dependency`, `ex_multi_flag` | kelebihan jumlah transaksi ber-rule dibanding rekan sejabatan (excess z-score, 90 hari) |
| `approve_rate` | persentase pengajuan yang disetujui |
| `top_requester_share` | porsi keputusan dari satu requester teratas |
| `ga_risky_pairs`, `ga_max_pair_z`, `ga_shared_debtor_share`, `ga_comm_score`, `ga_comm_size`, `ga_risk_pagerank` | posisi approver di jaringan berisiko |

Excess z-score = (jumlah teramati − jumlah wajar) ÷ simpangan baku binomial, dengan "wajar" dihitung dari rekan
sejabatan. Contoh: rekan sejabatan rata-rata 4% near-limit; approver dengan 40 keputusan wajarnya punya 1,6 transaksi
near-limit; bila ia punya 9, excess-nya sekitar +6 SD.

## Yang sengaja tidak dijadikan fitur

ID pegawai/debitur (model akan menghafal orang), nama unit dan wilayah (risiko bias), kolom `pattern` (kunci jawaban),
data kinerja kredit (terjadi setelah approval), dan skor Isolation Forest (dijaga terpisah dari classifier).
