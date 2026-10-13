---
doc_id: 08_panduan_membaca_skor
title: Panduan membaca tiga skor Collusion Radar dan batasannya
category: internal
source: Notebook Internal Collusion Radar v9 (Bagian 3.4, 5, 6.1) dan model card
url: internal://collusion-radar/panduan-skor
last_verified: 2026-10-11
disclaimer: Nilai threshold terkini tercantum di model card dan halaman Performa Model; dokumen ini menjelaskan cara membacanya.
---

# Panduan membaca skor

Collusion Radar menghasilkan **tiga skor yang terpisah**. Skor-skor ini **tidak digabung** (tidak di-ensemble), karena
masing-masing menjawab pertanyaan berbeda dan dihasilkan model yang berbeda.

## 1. Skor Risiko Transaksi (0–100)

- **Model**: classifier supervised (algoritma terpilih tercantum di model card), dilatih dari kasus yang sudah direview
  audit 2021–2024, dipilih dan dikalibrasi di data 2025, diuji di data 2026.
- **Cara membaca**: skor adalah probabilitas setelah **kalibrasi isotonic** dikali 100. Skor 80 berarti: di antara kasus
  dengan skor serupa pada populasi kasus yang direview, sekitar 80% terbukti kolusi.
- **Kategori**: **Tinggi** bila skor ≥ threshold kerja; **Sedang** bila ≥ separuh threshold; selain itu **Rendah**.
  Threshold dipilih di data validasi (F1 terbaik) dan dapat diubah manajemen sesuai kapasitas tim (lihat halaman Stress
  Testing).
- **Alasan**: tiga alasan teratas berasal dari nilai **SHAP** — kontribusi setiap fitur terhadap skor. Setiap alasan
  menunjuk data yang bisa diverifikasi (jam approval, rasio pasangan, anggota komunitas, dan sebagainya).

## 2. Skor Anomali Approver (persentil 0–100)

- **Model**: Isolation Forest **tanpa label**, unit approver × bulan, memakai perilaku 90 hari terakhir dan posisi di
  jaringan berisiko.
- **Cara membaca**: skor 97 berarti perilaku approver bulan itu lebih menyimpang daripada 97% approver-bulan lain.
- **Kategori**: **Perlu dilihat** ≥ 95; **Pantau** 80–95; selebihnya Normal.
- **Kegunaan**: menemukan approver yang perilakunya menyimpang walaupun belum ada transaksi berskor tinggi
  (pemeriksaan proaktif). Skor ini **bukan** input skor transaksi.

## 3. Skor Komunitas (0–100)

- **Model**: graph analytics tanpa label (Louvain pada graph hubungan berisiko 12 bulan terakhir).
- **Cara membaca**: total bobot hubungan berisiko di dalam komunitas, dibandingkan komunitas terbesar dalam 12 bulan
  terakhir. Semakin tinggi, semakin padat dan kuat lingkarannya.
- **Konteks penting**: kolom "aktif (bulan)" menunjukkan berapa bulan anggota terus muncul; komunitas yang persisten lebih
  patut diperiksa.

## Apa yang BOLEH dan TIDAK BOLEH disimpulkan

| Boleh | Tidak boleh |
|---|---|
| "Transaksi ini **berindikasi** kolusi dengan skor 85 karena ..." | "Pegawai X **terbukti** melakukan kolusi" |
| memprioritaskan pemeriksaan | memberi sanksi hanya berdasarkan skor |
| meminta klarifikasi dan dokumen pendukung | mengumumkan nama pihak yang sedang diperiksa |
| memakai skor approver untuk pemeriksaan proaktif | menganggap skor approver tinggi sebagai bukti |

## Batasan yang harus diketahui pengguna

1. **Kalibrasi relatif terhadap populasi kasus yang direview.** Proporsi kolusi di seluruh transaksi jauh lebih kecil,
   sehingga pada populasi skor lebih tepat dibaca sebagai **prioritas relatif**. Precision populasi disajikan sebagai
   rentang di dashboard.
2. **Data sintetis.** Prototype memakai data simulasi; validasi ulang dengan data internal wajib sebelum produksi.
3. **Pelaku beradaptasi.** Kalau pola penghindaran berubah, kinerja model bisa turun; pemantauan drift dan pelatihan ulang
   berkala diperlukan.
4. **Ketergantungan pada pipeline graph.** Bila fitur graph tidak tersedia, kinerja turun; kegagalan pipeline harus
   memicu peringatan.
5. **Label tidak sempurna.** Sebagian hasil review auditor ambigu; skor tidak lebih benar dari label yang dipelajarinya.

## Alur keputusan yang disarankan

Skor Tinggi → investigator membuka detail alert → memverifikasi alasan dan bukti → memeriksa komunitas dan profil approver
→ memutuskan: **tindak lanjut / eskalasi**, **bukan kolusi (alarm palsu)**, atau **perlu data tambahan** → keputusan
tercatat sebagai jejak audit dan umpan balik pelatihan ulang model.
