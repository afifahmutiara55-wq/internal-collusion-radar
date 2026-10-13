#!/bin/sh
set -e
cd /workspace
echo "[1/3] Memuat hasil notebook ke PostgreSQL (hanya jika belum ada)…"
python scripts/load_to_postgres.py --if-empty || echo "  peringatan: pemuatan gagal, aplikasi memakai CSV"
echo "[2/3] Menyiapkan akun & tabel aplikasi…"
python scripts/bootstrap_app.py || echo "  peringatan: bootstrap gagal, memakai akun demo"
echo "[3/3] Menyiapkan indeks RAG…"
python scripts/build_rag_index.py || echo "  peringatan: indeks dibangun saat aplikasi pertama dibuka"
exec streamlit run app/streamlit_app.py --server.address=0.0.0.0 --server.port=8501
