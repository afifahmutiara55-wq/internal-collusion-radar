"""Knowledge Base RAG: lihat dokumen & chunk, uji pencarian (dense vs BM25 vs hybrid), dan tambah dokumen baru.

Tata kelola: semua role boleh MELIHAT dan MENGUJI pencarian; hanya role manajemen yang boleh MENAMBAH dokumen,
karena isi knowledge base menentukan jawaban asisten AI (perubahan dicatat di log audit AI).
"""
from __future__ import annotations

import re
from datetime import date

import pandas as pd
import plotly.express as px
import streamlit as st

from ..config import SETTINGS
from ..genai.guardrails import mask_pii
from ..genai.knowledge_base import chunk_documents, load_documents, save_chunks
from ..ui import components as ui
from ..ui import services

CATEGORIES = ["regulasi", "kebijakan_internal", "sop", "analisis_fraud", "panduan_model", "lainnya"]


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:50] or "dokumen"


def rebuild_index() -> dict:
    """Chunk ulang semua dokumen lalu bangun indeks dense + BM25; kosongkan cache supaya asisten memakai indeks baru."""
    from ..genai.retriever import HybridRetriever
    documents = load_documents(SETTINGS.knowledge_base_dir)
    chunks, _ = chunk_documents(documents, SETTINGS.chunk_size_tokens, SETTINGS.chunk_overlap_tokens, SETTINGS.chunk_max_tokens)
    save_chunks(chunks, SETTINGS.knowledge_base_dir / "chunks.jsonl")
    retriever = HybridRetriever.build(chunks, SETTINGS.embedding_model, SETTINGS.embedding_window_tokens)
    retriever.save(SETTINGS.rag_index_dir)
    for cached in (services.get_retriever, services.get_rag, services.get_agent):
        cached.clear()
    return {"dokumen": len(documents), "chunk": len(chunks)}


def render_documents(chunks: pd.DataFrame) -> None:
    documents = load_documents(SETTINGS.knowledge_base_dir)
    rows = []
    for document in documents:
        own = chunks[chunks["doc_id"] == document["doc_id"]]
        rows.append({"doc_id": document["doc_id"], "judul": document["title"], "kategori": document.get("category", ""),
                     "sumber": document.get("source", ""), "url": document.get("url", ""),
                     "terakhir dicek": document.get("last_verified", ""), "ditambahkan oleh": document.get("added_by", "tim proyek"),
                     "chunk": len(own), "token median": int(own["n_tokens"].median()) if len(own) else 0})
    table = pd.DataFrame(rows)
    st.dataframe(table, hide_index=True, use_container_width=True,
                 column_config={"url": st.column_config.LinkColumn("url", display_text="buka sumber")})
    choice = st.selectbox("Baca dokumen", [d["doc_id"] for d in documents],
                          format_func=lambda doc_id: next(d["title"] for d in documents if d["doc_id"] == doc_id))
    document = next(d for d in documents if d["doc_id"] == choice)
    if document.get("disclaimer"):
        st.info(f"Catatan sumber: {document['disclaimer']}")
    with st.expander("Isi dokumen", expanded=False):
        st.markdown(document["text"])
    own = chunks[chunks["doc_id"] == choice]
    with st.expander(f"Chunk dokumen ini ({len(own)})"):
        for _, chunk in own.iterrows():
            st.markdown(f"**{chunk['chunk_id']}** · {chunk['section']} · {chunk['n_tokens']} token")
            st.code(chunk["text"][:1200], language=None)


def render_search_test(retriever) -> None:
    st.markdown("Bandingkan hasil **dense** (makna, MiniLM + FAISS), **BM25** (kata persis), dan **hybrid** (RRF, top 4) "
                "untuk satu pertanyaan. Berguna untuk mengecek apakah dokumen baru sudah bisa ditemukan.")
    query = st.text_input("Pertanyaan uji", "Apa kewajiban pelaporan fraud menurut POJK 12/2024?", key="kb_query")
    if not query.strip():
        return
    dense = retriever.dense_search(query, SETTINGS.dense_k)
    sparse = retriever.sparse_search(query, SETTINGS.sparse_k)
    hybrid = retriever.hybrid_search(query, SETTINGS.dense_k, SETTINGS.sparse_k, 4)
    columns = st.columns(3)
    for column, title, results in [(columns[0], f"Dense (k={SETTINGS.dense_k})", dense),
                                   (columns[1], f"BM25 (k={SETTINGS.sparse_k})", sparse)]:
        column.markdown(f"**{title}**")
        column.dataframe(pd.DataFrame([{"#": rank, "chunk": retriever.chunks[i]["chunk_id"], "skor": round(score, 3)}
                                       for rank, (i, score) in enumerate(results, start=1)]),
                         hide_index=True, use_container_width=True, height=300)
    columns[2].markdown("**Hybrid RRF → top 4 (dipakai LLM)**")
    columns[2].dataframe(pd.DataFrame([{"#": rank, "chunk": r["chunk_id"], "RRF": round(r["rrf_score"], 4),
                                        "dense #": r["dense_rank"], "BM25 #": r["sparse_rank"]}
                                       for rank, r in enumerate(hybrid, start=1)]),
                         hide_index=True, use_container_width=True, height=300)
    for rank, result in enumerate(hybrid, start=1):
        with st.expander(f"[S{rank}] {result['title']} › {result['section']}"):
            st.write(result["text"].split("\n", 1)[-1][:1500])


def render_add_document(user: dict) -> None:
    if user["role"] != "manajemen":
        st.info("Penambahan dokumen hanya untuk role manajemen (perubahan knowledge base mengubah jawaban asisten AI). "
                "Usulkan dokumen baru ke manajemen/kepatuhan.")
        return
    st.markdown("Unggah file **.md / .txt** atau tempel teks. Gunakan judul bagian `## ...` supaya chunking per bagian rapi. "
                "Dokumen berisi data pribadi (NIK, nomor rekening, telepon, email, alamat) **ditolak** sesuai UU PDP.")
    with st.form("kb_add_form", clear_on_submit=False):
        title = st.text_input("Judul dokumen*")
        left, right = st.columns(2)
        category = left.selectbox("Kategori", CATEGORIES)
        source = right.text_input("Sumber (instansi / nomor dokumen)*")
        url = st.text_input("URL sumber (opsional)")
        uploaded = st.file_uploader("File", type=["md", "txt"])
        pasted = st.text_area("…atau tempel isi dokumen", height=180)
        confirmed = st.checkbox("Saya sudah memeriksa isi dokumen benar dan boleh dipakai sebagai acuan asisten AI")
        submitted = st.form_submit_button("Tambahkan & bangun ulang indeks", type="primary")
    if not submitted:
        return
    body = uploaded.read().decode("utf-8", errors="ignore") if uploaded is not None else pasted
    body = re.sub(r"^---.*?---\s*", "", body, flags=re.S).strip()   # buang front matter lama bila ada
    problems = []
    if not title.strip() or not source.strip():
        problems.append("Judul dan sumber wajib diisi.")
    if len(body.split()) < 40:
        problems.append("Isi dokumen terlalu pendek (minimal ±40 kata).")
    _, pii_found = mask_pii(body)
    if pii_found:
        problems.append(f"Dokumen mengandung pola data pribadi: {', '.join(sorted(set(pii_found)))}. Hapus dulu sebelum diunggah.")
    if not confirmed:
        problems.append("Centang konfirmasi pemeriksaan isi.")
    if problems:
        for problem in problems:
            st.error(problem)
        return
    doc_id = f"90_{slugify(title)}"
    path = SETTINGS.knowledge_base_dir / f"{doc_id}.md"
    if not body.lstrip().startswith("#"):
        body = f"# {title}\n\n## Isi\n\n{body}"
    front_matter = (f"---\ndoc_id: {doc_id}\ntitle: {title.strip()}\ncategory: {category}\nsource: {source.strip()}\n"
                    f"url: {url.strip()}\nlast_verified: {date.today().isoformat()}\nadded_by: {user['username']}\n---\n\n")
    path.write_text(front_matter + body + "\n", encoding="utf-8")
    with st.spinner("Chunking + embedding + membangun indeks FAISS & BM25…"):
        summary = rebuild_index()
    services.get_store().log_ai_event({"username": user["username"], "role": user["role"], "mode": "kb_update",
                                        "question_masked": f"tambah dokumen {doc_id}", "guard_flags": "", "retrieved": "",
                                        "tool_calls": "", "answer": f"indeks: {summary}", "latency_ms": 0.0,
                                        "model": SETTINGS.embedding_model, "fallback_used": False, "blocked": False})
    st.success(f"Dokumen '{title}' ditambahkan. Indeks sekarang {summary['dokumen']} dokumen / {summary['chunk']} chunk. "
               "Jalankan ulang evaluasi retrieval (notebook Bagian 8 atau scripts/evaluate_genai.py) agar mutu tetap terpantau.")


def render() -> None:
    user = services.current_user()
    ui.page_header("Knowledge Base RAG", "Dokumen acuan asisten AI: regulasi KUR & OJK, analisis fraud, SOP, dan panduan model")
    retriever = services.get_retriever()
    chunks = pd.DataFrame(retriever.chunks)
    info = retriever.describe()
    ui.kpi_row([("Dokumen", f"{chunks['doc_id'].nunique()}", "file markdown", ui.BLUE),
                ("Chunk", f"{len(chunks)}", f"token {chunks['n_tokens'].min()}–{chunks['n_tokens'].max()} · median "
                                             f"{int(chunks['n_tokens'].median())}", ui.PURPLE),
                ("Embedding", f"{info['dimension']} dimensi", info["embedder"][:38], ui.GREEN),
                ("Retrieval", f"top 4", f"dense k={SETTINGS.dense_k} · BM25 k={SETTINGS.sparse_k} · RRF", ui.ORANGE)])
    tab_docs, tab_search, tab_add = st.tabs(["Dokumen", "Uji pencarian", "Tambah dokumen"])
    with tab_docs:
        left, right = st.columns([2, 1])
        with right:
            figure = px.histogram(chunks, x="n_tokens", nbins=15, color_discrete_sequence=[ui.BLUE])
            figure.add_vline(x=SETTINGS.chunk_max_tokens, line_dash="dash", line_color=ui.RED, annotation_text="batas 600")
            st.plotly_chart(ui.style_figure(figure, "Panjang chunk (token)", "semua di bawah batas", 280),
                            use_container_width=True)
            per_category = chunks.groupby("category")["doc_id"].nunique().reset_index(name="dokumen")
            st.plotly_chart(ui.style_figure(px.bar(per_category, x="dokumen", y="category", orientation="h"),
                                            "Dokumen per kategori", "", 240), use_container_width=True)
        with left:
            render_documents(chunks)
    with tab_search:
        render_search_test(retriever)
    with tab_add:
        render_add_document(user)
