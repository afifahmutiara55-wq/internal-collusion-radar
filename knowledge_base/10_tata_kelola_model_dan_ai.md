---
doc_id: 10_tata_kelola_model_dan_ai
title: Tata kelola model ML dan penggunaan GenAI pada Collusion Radar
category: internal
source: Praktik manajemen risiko model, prinsip AI guardrail capstone ODP, dan arsitektur prototype v9
url: internal://collusion-radar/tata-kelola-model
last_verified: 2026-10-11
disclaimer: Kebijakan contoh untuk prototype; bank harus menyesuaikan dengan kebijakan manajemen risiko model yang berlaku.
---

# Tata kelola model dan AI

## Siklus hidup model

1. **Pengembangan**: data dibersihkan dengan data contract; fitur dihitung as-of (tanpa melihat masa depan); split waktu
   train 2021–2024, validation 2025, test 2026.
2. **Seleksi dan tuning**: PyCaret untuk screening algoritma, Optuna untuk tuning dengan validasi silang deret waktu.
   Isolation Forest dituning tanpa label (kriteria Excess-Mass).
3. **Validasi**: evaluasi di data test yang dibuka sekali, interval kepercayaan bootstrap, uji fairness antarsegmen,
   analisis error per pola, uji stres.
4. **Registrasi**: model dicatat di **MLflow Model Registry** beserta parameter, metrik, dan model card.
5. **Penerapan**: dashboard membaca skor dari PostgreSQL; model yang dipakai selalu versi terdaftar.
6. **Pemantauan**: Population Stability Index (PSI) bulanan untuk fitur utama, kinerja per bulan, volume alert, dan rasio
   keputusan "alarm palsu" dari investigator.
7. **Pelatihan ulang**: kuartalan, atau lebih cepat bila PSI > 0,25 pada fitur utama, precision populasi turun, atau
   muncul pola baru. Versi baru wajib melewati validasi yang sama sebelum menggantikan versi lama.

## Peran dan tanggung jawab (contoh)

| Peran | Tanggung jawab |
|---|---|
| Pemilik model (unit anti-fraud) | menyetujui penggunaan, threshold, dan perubahan model |
| Pengembang model (data scientist) | membangun, mendokumentasikan, mengusulkan perubahan |
| Validator independen (manajemen risiko) | menguji ulang sebelum produksi dan secara berkala |
| Pengguna (investigator, manajemen) | memakai skor sesuai panduan; mencatat keputusan |
| Audit intern | mengaudit kepatuhan proses dan jejak keputusan |

## Kebijakan penggunaan GenAI

- **Model lokal**: LLM (Gemma 4 E4B) dijalankan lokal melalui Ollama; tidak ada data yang dikirim ke layanan eksternal.
- **Grounding**: asisten RAG hanya menjawab dari knowledge base dan wajib menyebut sumber. Bila konteks tidak memadai,
  asisten menjawab tidak tahu.
- **Bukan pengambil keputusan**: agen AI hanya membaca data, merangkum, dan menyusun draf. Aksi penting (menyimpan memo,
  eskalasi) memerlukan persetujuan manusia.
- **Hak akses mengikuti role**: investigator dapat membuka detail alert dan profil pegawai; manajemen mendapat data agregat,
  kinerja model, dan simulasi.
- **Bahasa netral**: keluaran AI tidak boleh menuduh; istilah yang dipakai adalah "indikasi" dan "perlu klarifikasi".

## Empat lapis AI guardrail

| Lapis | Kontrol |
|---|---|
| Input | deteksi prompt injection (pola + kemiripan semantik), pemeriksaan keamanan dengan Llama Guard 3, masking PII, pembatasan topik |
| Proses / agen | whitelist tool per role, validasi argumen, batas langkah, tool hanya-baca, persetujuan manusia untuk aksi |
| Output | cek sitasi dan grounding, filter bahasa menuduh, masking PII, pemeriksaan keamanan output |
| Monitoring & audit | log prompt, tool call, keluaran, keputusan guardrail, latensi; evaluasi kualitas dengan LLM-as-a-judge; fallback bila LLM gagal |

## Evaluasi kualitas jawaban AI (LLM-as-a-judge)

Jawaban asisten dievaluasi secara berkala pada set pertanyaan uji dengan kriteria: **kesetiaan pada konteks
(faithfulness)**, **relevansi**, **kebenaran terhadap jawaban acuan**, **ketepatan sitasi**, dan **kepatuhan nada/keamanan**.
Juri memakai model dari keluarga berbeda dengan generator untuk mengurangi bias memihak diri sendiri, temperatur 0, rubrik
berskala 1–5 dengan deskripsi tiap skor, dan diminta menuliskan alasan sebelum skor. Kesesuaian juri dengan penilaian
manusia diuji pada sampel berlabel (persentase kesepakatan dan Cohen's kappa).
