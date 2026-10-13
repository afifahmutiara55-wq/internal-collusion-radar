# Internal Collusion Radar v9 — Deteksi Indikasi Kolusi pada Persetujuan KUR

Capstone ODP Data Analytics Batch 2. Sistem end-to-end untuk mendeteksi indikasi kolusi **requester ↔ approver**
pada proses persetujuan Kredit Usaha Rakyat: data → EDA → 5 rules → graph analytics → Isolation Forest (individu) &
classifier (transaksi) → SHAP → dashboard dua role → asisten AI (RAG + agen) dengan guardrail dan LLM-as-judge.

> **Data sintetis.** Semua pegawai, debitur, dan kasus fiktif (generator v6.1, `data_generator/`). Angka di bawah
> berasal dari data sintetis dan **bukan** estimasi kondisi bank mana pun.

---

## 1. Isi paket

| Folder / file | Isi |
|---|---|
| `notebooks/Internal_Collusion_Radar_EndToEnd_v9.ipynb` | Notebook utama (Bagian 0–12), sudah dieksekusi beserta output |
| `data/raw/` | 7 tabel sintetis: employees, debtors, requests, approvals, access_logs, fraud_cases, loan_performance |
| `data_generator/` | Generator dataset v6.1 + validator |
| `outputs/app_data/` | Hasil notebook untuk dashboard (CSV + JSON) |
| `outputs/models/` | `transaction_risk_model.joblib` (classifier terkalibrasi), `approver_anomaly_model.joblib` (Isolation Forest), `model_card.json` |
| `outputs/rag_index/` | Indeks RAG (chunk, vektor jendela kalimat, konfigurasi k hasil tuning) |
| `knowledge_base/` | 11 dokumen acuan RAG (KUR, POJK, UU PDP, ACFE, tipologi, SOP, tata kelola model) + set evaluasi (`eval/`) |
| `app/` | Aplikasi Streamlit (`streamlit_app.py`, paket `radar_app/`) |
| `scripts/` | Muat ke PostgreSQL, bootstrap akun, bangun indeks RAG, evaluasi GenAI offline |
| `docker-compose.yml`, `docker/`, `db/` | Infrastruktur: PostgreSQL, MLflow, Ollama, app, JupyterLab (opsional), query DBeaver |

## 2. Arsitektur

```
        data/raw (7 tabel)                           knowledge_base/*.md
              │                                              │ chunking ≤600 token
   ┌──────────▼───────────┐                       ┌──────────▼──────────┐
   │ Notebook (Bagian 1–7)│ ──── MLflow ────────▶ │ MiniLM-L12 (384d)   │
   │ clean → fitur as-of  │  (params, metrik,     │ FAISS + BM25 → RRF  │
   │ 5 rules → GRAPH ─┬──▶│   model registry)     │ top 4               │
   │                  │   │                       └──────────┬──────────┘
   │   g_* (transaksi)│ ga_* (approver)                      │
   │        ▼         ▼   │                     Gemma 4 E4B (Ollama, t=0,1)
   │  Classifier   Isolation Forest               + guardrail 4 lapis
   │  (transaksi)  (individu/bulan)               + agen ber-tools (role)
   └────┬──────────────┬──┘                       + LLM-as-judge (Qwen2.5)
        │ CSV / joblib │                                     │
        ▼              ▼                                     ▼
   PostgreSQL (schema radar = hasil analitik, app = keputusan, log audit AI) ◀── DBeaver
        │
        ▼
   Streamlit (login investigator / manajemen, light mode)
```

**Prinsip independensi (tidak ada ensemble).** Graph analytics dihitung lebih dulu dari keputusan *masa lalu*
(snapshot bulanan, jendela 12 bulan sebelum bulan berjalan). Outputnya dipakai sebagai fitur oleh dua model yang
**tidak saling memakai output**:

| | Isolation Forest | Classifier terkalibrasi |
|---|---|---|
| Pertanyaan | "Apakah **approver ini** berperilaku tidak wajar bulan ini?" | "Seberapa mungkin **transaksi ini** kolusi?" |
| Unit | approver × bulan (jendela 90 hari) | satu persetujuan kredit |
| Label | tidak dipakai saat melatih (unsupervised) | label kasus terkonfirmasi, split waktu |
| Skor | persentil 0–100 (peringkat relatif) | probabilitas terkalibrasi (isotonic) → 0–100 |
| Dipakai untuk | prioritas audit perilaku pegawai | antrean alert transaksi |

Alasan tidak digabung: unit analisis dan arti skornya berbeda (persentil vs probabilitas). Rata-rata keduanya tidak
punya arti probabilistik, merusak kalibrasi, dan membuat keputusan sulit dipertanggungjawabkan. Notebook menguji
independensi dengan `assert` (tidak ada kolom IF di fitur classifier dan sebaliknya; label tidak dipakai IF).

## 3. Hasil utama (data sintetis, test = 2026)

| Komponen | Metrik yang dipakai & hasil |
|---|---|
| Rules (5 aturan, ≥1 aktif) | precision 0,53 · recall 0,85 · PR-AUC 0,60 · 474 alarm palsu |
| Baseline Logistic Regression | precision 0,77 · recall 0,92 · PR-AUC 0,90 · 168 alarm palsu |
| Classifier final (Random Forest terkalibrasi, dipilih di validasi 2025) | precision 0,86 · recall 0,91 · F1 0,89 · MCC 0,87 · PR-AUC 0,90 (CI bootstrap 95% 0,88–0,93) · 90 alarm palsu |
| Isolation Forest (approver-bulan, 2025–2026) | PR-AUC 0,68 · precision@5% 0,87 (acak 0,18) · stabilitas top-5% antar seed (Jaccard) 0,82 → 0,87 setelah tuning EM tanpa label |
| Graph (snapshot bulanan) | 32 komunitas risiko · modularity 0,95–0,97 · persistensi komunitas 0,65 · peringatan dini precision ±33% (lift ±6,5×) |
| RAG (33 pertanyaan) | Hit@4 1,00 · MRR 0,96 · 38 chunk (174–470 token, median 264) · k optimal dense 3, BM25 5 |
| Guardrail (34 kasus red-team) | 22/22 serangan tertangani · 0/12 pertanyaan wajar terblokir |
| Agen (14 skenario) | 14/14 benar (tools tepat, penolakan sesuai wewenang role) |

Membaca dengan jujur: RF dan LR **setara** dalam kemampuan mengurutkan (CI PR-AUC bertumpang tindih); keunggulan RF
ada pada threshold kerja (alarm palsu 168 → 90). Lompatan terbesar datang dari **Rules → ML** (alarm palsu −81% dengan
recall setara) dan dari **penambahan fitur graph** (ablasi PR-AUC test 0,87 → 0,91). IF dan classifier saling
melengkapi: 34 approver-bulan hanya ditandai IF (71% terbukti menyetujui kasus kolusi), korelasi Spearman 0,48.
Stress test: bila pelaku menghindari semua rule transaksi, recall rules turun ke 0,25 sedangkan model tetap 0,85.
Dampak (asumsi dapat diubah di dashboard): manfaat bersih ±Rp12,1 M/tahun (rentang sensitivitas Rp5,1–21,8 M)
dengan beban investigasi −62% dibanding rules.

## 4. Menjalankan dengan Docker (disarankan)

Prasyarat: Docker Desktop / Docker Engine + Compose v2, RAM ≥ 16 GB dan ruang disk ±20 GB untuk tiga model Ollama (GPU opsional).

```bash
cp .env.example .env            # ubah password sebelum dipakai di luar demo
docker compose up -d --build    # postgres, mlflow, ollama (+ unduh 3 model), app
docker compose logs -f ollama-pull   # tunggu sampai ketiga model selesai diunduh
```

| Layanan | Alamat | Keterangan |
|---|---|---|
| Dashboard | http://localhost:8501 | login `investigator` / `investigator123` atau `manajemen` / `manajemen123` |
| MLflow | http://localhost:5000 | experiment `kur-collusion-radar`, model `kur_collusion_classifier`, `kur_approver_iforest` |
| PostgreSQL | localhost:5432, db `radar` | buka di **DBeaver**: New Connection → PostgreSQL → user/password dari `.env`. Query siap pakai: `db/dbeaver_queries.sql` |
| Ollama | http://localhost:11434 | `gemma4:e4b` (generator), `llama-guard3:1b` (guardrail), `qwen2.5:7b-instruct` (juri) |

Saat container app pertama kali jalan: CSV hasil notebook dimuat ke schema `radar`, akun dibuat (password di-hash
PBKDF2), lalu indeks RAG disiapkan. Tanpa Ollama, asisten tetap jalan dalam **mode cadangan** (jawaban ekstraktif dari
sumber, planner agen berbasis aturan, juri heuristik), dan setiap jawaban menyebut mode yang dipakai.

### Melatih ulang di lingkungan yang sama (penting)

```bash
docker compose --profile lab up -d lab     # JupyterLab: http://localhost:8888 (token: radar)
# buka notebooks/Internal_Collusion_Radar_EndToEnd_v9.ipynb → Run All (±30–60 menit dengan PyCaret + Optuna)
docker compose exec app python scripts/load_to_postgres.py      # muat ulang hasil ke PostgreSQL
docker compose exec app python scripts/build_rag_index.py --force
docker compose restart app
```

File model `.joblib` harus dibuat dengan versi scikit-learn/pandas yang **sama** dengan aplikasi
(`app/requirements.txt`). Notebook terlampir dieksekusi di lingkungan pengembangan yang berbeda (lihat §7), jadi
jalankan ulang notebook di container `lab` sebelum demo agar model bisa dimuat aplikasi. Bila model tidak bisa dimuat,
aplikasi tetap menampilkan skor yang sudah dihitung notebook dan hanya menonaktifkan rescoring langsung (stress test).

## 5. Menjalankan tanpa Docker

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r notebooks/requirements-notebook.txt     # notebook (PyCaret butuh Python 3.11)
cd notebooks && jupyter lab                            # Run All
cd .. && pip install -r app/requirements.txt
streamlit run app/streamlit_app.py                     # tanpa DATABASE_URL → membaca CSV di outputs/app_data
```

Opsional: `ollama pull gemma4:e4b llama-guard3:1b qwen2.5:7b-instruct` dan `mlflow server --port 5000`
(set `MLFLOW_TRACKING_URI=http://localhost:5000` sebelum menjalankan notebook).

## 6. Peta halaman dashboard

| Role | Halaman | Tugas utama |
|---|---|---|
| Investigator | Ringkasan | scorecard alert, tren bulanan, red flag teratas, antrean kerja |
| | Monitoring Alert | filter level risiko / red flag / unit / nama pegawai → scorecard, tabel, detail (gauge, alasan SHAP, bukti, riwayat pasangan, jaringan, form keputusan, draf memo agen) |
| | Jaringan Risiko | komunitas risiko (skor, anggota bernama, PageRank), graf interaktif dengan garis tebal = hubungan berisiko |
| | Profil Pegawai | skor anomali IF per bulan, alasan, transaksi terkait, filter tabel |
| | Riwayat Keputusan | keputusan yang sudah dibuat (umpan balik untuk precision lapangan) |
| | Analisis Skor | distribusi skor, kalibrasi, kontribusi grup fitur, perbandingan skor IF vs classifier |
| | Performa Model | metrik per model (rules, graph, IF, classifier), confusion matrix interaktif |
| | Asisten AI / Knowledge Base | RAG + agen; telusuri dokumen, uji pencarian dense vs BM25 vs hybrid |
| Manajemen | Ringkasan Eksekutif | KPI, eksposur, tren, unit & komunitas teratas |
| | Risiko Unit & Wilayah | treemap dan tabel tingkat alert per unit |
| | Jaringan Risiko | sama seperti investigator |
| | Stress Testing | kapasitas investigator, evasion (pelaku menghindari rules), graph mati, prevalensi, drift |
| | Dampak Bisnis | simulasi manfaat bersih dengan slider asumsi |
| | Performa Model / Analisis Skor / Model & MLflow | metrik, run MLflow, model registry, lineage |
| | Audit & Kualitas AI | log audit AI, hasil guardrail, skor LLM-as-judge, kappa vs label manusia |
| | Knowledge Base | sama + **tambah dokumen** (validasi PII, bangun ulang indeks, tercatat di log audit) |

## 7. Catatan jujur tentang lingkungan pengembangan

- Notebook terlampir dieksekusi di sandbox tanpa akses PyPI/HuggingFace. Karena itu beberapa langkah memakai
  **pengganti yang setara secara metodologi** dan notebook menuliskannya di tabel status pustaka: screening model manual
  dengan CV deret waktu yang sama (pengganti PyCaret), random search (pengganti Optuna TPE), sampling Shapley
  (pengganti SHAP TreeExplainer), embedding LSA (pengganti MiniLM), pencarian numpy (pengganti FAISS), dan mode
  cadangan GenAI (tanpa Ollama). Di container `lab` semua pustaka asli terpasang dan notebook otomatis memakainya.
- Angka GenAI (juri, red-team) pada notebook terlampir berasal dari mode cadangan. Jalankan
  `docker compose exec app python scripts/evaluate_genai.py` setelah model Ollama terunduh untuk skor dengan juri LLM.
- Versi dependensi di `requirements*.txt` dan Dockerfile disusun agar saling kompatibel, tetapi belum dapat dibangun
  di sandbox ini. Bila pip melaporkan konflik (biasanya karena PyCaret), ikuti catatan di `requirements-notebook.txt`.
- Nilai k optimal (dense 3, BM25 5) di `retrieval_config.json` dipilih saat embedding pengganti (LSA) aktif. Container
  app membangun ulang indeks dengan MiniLM, tetapi k baru akan dituning ulang hanya bila notebook dijalankan lagi
  (Bagian 8) di container `lab`.
- Set red-team (34 kasus) dan set evaluasi RAG (33 pertanyaan) masih kecil; perlakukan hasilnya sebagai uji fungsi,
  bukan jaminan keamanan.
- Ringkasan regulasi di `knowledge_base/` berasal dari sumber resmi yang dicek 11 Oktober 2026 dan diberi disclaimer;
  bukan teks lengkap peraturan.

## 8. Informasi tambahan yang akan memperkuat sistem

SOP investigasi internal dan matriks kewenangan memutus kredit (limit per jabatan), definisi jam kerja dan pola shift
resmi, daftar kasus fraud historis yang sudah diputus (untuk kalibrasi ulang), kebijakan retensi data dan klasifikasi
informasi, serta teks lengkap kebijakan anti-fraud bank. Dokumen tersebut bisa ditambahkan lewat halaman
Knowledge Base (role manajemen) tanpa mengubah kode.
