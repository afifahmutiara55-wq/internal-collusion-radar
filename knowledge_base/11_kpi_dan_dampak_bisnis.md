---
doc_id: 11_kpi_dan_dampak_bisnis
title: Definisi KPI dashboard manajemen dan cara menghitung dampak bisnis
category: internal
source: Notebook Internal Collusion Radar v9 (Bagian 6–7) dan halaman Dampak Bisnis dashboard
url: internal://collusion-radar/kpi-dampak
last_verified: 2026-10-11
disclaimer: Angka dampak adalah estimasi dengan asumsi yang dinyatakan terbuka; angka terkini tersedia di dashboard.
---

# KPI dan dampak bisnis

## Definisi KPI

| KPI | Definisi |
|---|---|
| Transaksi dinilai | jumlah approval yang diberi skor risiko pada periode terpilih |
| Alert Tinggi / Sedang | jumlah transaksi dengan kategori skor Tinggi / Sedang |
| Nominal berisiko | total nominal kredit pada alert Tinggi |
| Precision populasi (rentang) | batas bawah: alert yang belum direview dianggap alarm palsu; batas atas: precision sama dengan alert yang sudah direview |
| Recall (test) | persentase kasus kolusi terkonfirmasi di data uji yang tertangkap sebagai alert Tinggi |
| Alarm palsu per bulan | alert Tinggi yang ternyata normal, per bulan |
| Backlog investigasi | alert Tinggi yang belum diputuskan investigator |
| Approver perlu dilihat | approver dengan skor anomali ≥ 95 pada bulan terakhir |
| Komunitas risiko aktif | komunitas beranggota ≥ 3 orang pada snapshot graph terbaru |

## Rumus dampak finansial

- **Kolusi per tahun** = kasus kolusi terkonfirmasi di data uji, disetahunkan.
- **Kerugian per kasus** = rata-rata outstanding NPL kredit kolusi yang sudah ≥ 18 bulan cair × LGD.
- **Kerugian dicegah per tahun** = kolusi per tahun × recall × tingkat pencegahan × kerugian per kasus.
- **Biaya pemeriksaan per tahun** = alert per bulan (populasi) × 12 × jam per alert × biaya per jam investigator.
- **Manfaat bersih** = kerugian dicegah − biaya pemeriksaan.

Asumsi dasar: LGD 60%, 75% alert yang benar berhasil dicegah sebelum dana cair, 2 jam per alert, Rp150.000 per jam.
Analisis sensitivitas menggeser LGD (40–80%) dan tingkat pencegahan (50–100%). Halaman Dampak Bisnis menyediakan slider
untuk mengubah semua asumsi.

## Dampak non-finansial

- Deteksi sesaat setelah approval, bukan saat kredit menjadi NPL.
- Beban investigasi lebih kecil karena alarm palsu jauh lebih sedikit dibanding rules.
- Bukti terstruktur per alert dan jejak audit keputusan.
- Jaringan kolusi terlihat sebagai komunitas, bukan kasus terpisah.
- Efek jera dan penguatan prinsip empat mata.

## Kapasitas investigasi

Kapasitas per bulan = jumlah investigator × jam kerja investigasi per bulan ÷ jam per alert. Contoh: 6 investigator × 120
jam ÷ 2 jam = 360 alert per bulan. Manajemen dapat menurunkan threshold bila kapasitas masih tersisa (recall naik) atau
menaikkannya bila antrean menumpuk (precision naik, recall turun). Halaman Stress Testing menampilkan trade-off ini.
