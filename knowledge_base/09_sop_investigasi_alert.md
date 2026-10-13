---
doc_id: 09_sop_investigasi_alert
title: Contoh SOP penanganan alert indikasi kolusi (prototype)
category: internal
source: Disusun tim Collusion Radar sebagai contoh, mengacu pada empat pilar strategi anti fraud POJK 12/2024
url: internal://collusion-radar/sop-investigasi
last_verified: 2026-10-11
disclaimer: SOP contoh untuk prototype dan demo. Bank wajib memakai SOP resmi unit anti-fraud/SKAI yang berlaku.
---

# Contoh SOP penanganan alert indikasi kolusi

## Prinsip

1. **Praduga tak bersalah**: alert adalah indikasi, bukan tuduhan.
2. **Kerahasiaan**: identitas pihak yang diperiksa hanya diketahui investigator yang ditugaskan.
3. **Berbasis bukti**: setiap kesimpulan harus didukung dokumen dan data yang bisa diverifikasi.
4. **Proporsional**: tindakan sesuai tingkat risiko dan kekuatan bukti.
5. **Tercatat**: setiap keputusan dicatat di sistem sebagai jejak audit.

## Target waktu (SLA) per kategori

| Kategori alert | Target triase | Target keputusan awal | Tindakan sementara |
|---|---|---|---|
| Tinggi | 1 × 24 jam | 3 hari kerja | usulkan penahanan pencairan kepada pejabat berwenang bila dana belum cair |
| Sedang | 3 hari kerja | 10 hari kerja | pemantauan, tanpa penahanan |
| Approver "Perlu dilihat" | 5 hari kerja | 20 hari kerja | reviu portofolio approver secara sampling |

## Langkah 1 — Triase

- Buka detail alert: skor, kategori, tiga alasan SHAP, rules yang terpenuhi, dan bukti graph.
- Periksa apakah ada **penjelasan sah** yang jelas: lembur akhir bulan yang terdokumentasi, surat delegasi cuti, SK Plt,
  memo pengalihan portofolio, atau top-up debitur lama.
- Bila penjelasan sah terverifikasi → tandai **bukan kolusi (alarm palsu)** dengan catatan alasannya.

## Langkah 2 — Pengumpulan bukti

- Dokumen kredit: permohonan, analisis kredit, hasil kunjungan, agunan, dan memo persetujuan.
- Jejak sistem: log login dan aksi (misalnya login malam sebelum approval malam), waktu pengajuan dan approval.
- Riwayat pasangan requester–approver: frekuensi, rasio terhadap rekan unit, debitur yang dipakai bersama.
- Peta komunitas risiko dan skor anomali approver sebagai konteks.
- Status kredit terkini bila sudah cair (kolektibilitas).

## Langkah 3 — Klarifikasi

- Klarifikasi tertulis atau wawancara dengan requester dan approver secara terpisah.
- Verifikasi keberadaan dan usaha debitur bila ada indikasi debitur titipan/fiktif.
- Dokumentasikan semua jawaban dan bukti pendukung.

## Langkah 4 — Keputusan dan eskalasi

| Keputusan | Kriteria | Tindak lanjut |
|---|---|---|
| Bukan kolusi | penjelasan sah terverifikasi | tutup alert; catat alasan (umpan balik model) |
| Perlu data tambahan | bukti belum cukup | jadwalkan ulang, tetapkan data yang dibutuhkan |
| Tindak lanjut / eskalasi | indikasi kuat | laporkan ke unit anti-fraud/SKAI sesuai kewenangan; proses investigasi formal |

Fraud yang berdampak signifikan dilaporkan sesuai ketentuan pelaporan strategi anti fraud. Sanksi mengikuti peraturan
kepegawaian dan tata kelola bank, bukan keputusan sistem.

## Langkah 5 — Dokumentasi dan pembelajaran

- Simpan memo investigasi final (draf dari agen AI wajib ditinjau dan disetujui investigator).
- Keputusan investigator menjadi **label baru** untuk pelatihan ulang model.
- Pola baru yang ditemukan diusulkan sebagai rule baru atau fitur baru melalui tata kelola model.

## Struktur memo investigasi

1. Identitas alert (ID transaksi, tanggal, skor, kategori).
2. Ringkasan indikasi (alasan SHAP, rules, bukti graph, skor anomali approver).
3. Penjelasan sah yang sudah dicek.
4. Bukti yang dikumpulkan dan sumbernya.
5. Hasil klarifikasi.
6. Kesimpulan sementara dan rekomendasi tindak lanjut.
7. Catatan: memo disusun berdasarkan indikasi; tidak memuat kesimpulan bersalah.
