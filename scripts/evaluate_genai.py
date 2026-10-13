"""Evaluasi offline asisten RAG dengan LLM-as-a-judge dan catat ke MLflow (bisa dijadwalkan, mis. mingguan).

Pemakaian (di container app atau lab):
    python scripts/evaluate_genai.py --judge-model qwen2.5:7b-instruct
Hasil: outputs/app_data/judge_eval.csv & judge_summary.csv diperbarui; run baru di experiment MLflow 'kur-collusion-radar'.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from radar_app.config import SETTINGS  # noqa: E402
from radar_app.data_access import DataStore  # noqa: E402
from radar_app.genai.judge import CRITERIA, LLMJudge  # noqa: E402
from radar_app.genai.llm import OllamaClient  # noqa: E402
from radar_app.genai.rag import RagAssistant  # noqa: E402
from radar_app.genai.retriever import load_or_build_retriever  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge-model", default=SETTINGS.judge_model)
    args = parser.parse_args()
    eval_set = [json.loads(line) for line in (SETTINGS.knowledge_base_dir / "eval" / "rag_eval_set.jsonl").read_text().splitlines() if line.strip()]
    client = OllamaClient(SETTINGS.ollama_base_url, SETTINGS.llm_timeout_seconds)
    store = DataStore()
    retriever = load_or_build_retriever(SETTINGS.rag_index_dir, SETTINGS.knowledge_base_dir, SETTINGS.embedding_model)
    assistant = RagAssistant(retriever, client, store, known_entities=set(store.table("employees").get("employee_name", [])))
    judge = LLMJudge(client, args.judge_model, use_mlflow=bool(SETTINGS.mlflow_tracking_uri))
    print("Juri:", judge.engine)
    rows = []
    for item in eval_set:
        response = assistant.answer(item["question"], "investigator", "evaluator")
        verdict = judge.judge(item["question"], response["answer"], response["contexts"], item["reference_answer"])
        rows.append({"id": item["id"], "pertanyaan": item["question"], "mode_jawaban": response["mode"],
                     **{c: verdict[c]["score"] for c in CRITERIA}, "jawaban": response["answer"][:400]})
        print(item["id"], response["mode"], {c: verdict[c]["score"] for c in CRITERIA})
    results = pd.DataFrame(rows)
    summary = results[CRITERIA].mean().rename("skor rata-rata (1–5)").to_frame()
    summary["% skor ≥ 4"] = [(results[c] >= 4).mean() for c in CRITERIA]
    results.to_csv(SETTINGS.app_data_dir / "judge_eval.csv", index=False)
    summary.reset_index().rename(columns={"index": "kriteria"}).to_csv(SETTINGS.app_data_dir / "judge_summary.csv", index=False)
    if SETTINGS.mlflow_tracking_uri:
        try:
            import mlflow
            mlflow.set_tracking_uri(SETTINGS.mlflow_tracking_uri)
            mlflow.set_experiment("kur-collusion-radar")
            with mlflow.start_run(run_name="genai_eval_llm_judge"):
                mlflow.log_params({"judge": judge.engine, "generator": SETTINGS.generator_model, "dense_k": SETTINGS.dense_k,
                                   "sparse_k": SETTINGS.sparse_k})
                mlflow.log_metrics({f"judge_{c}": float(results[c].mean()) for c in CRITERIA if results[c].notna().any()})
                mlflow.log_text(results.to_csv(index=False), "judge_eval.csv")
            print("Tercatat di MLflow.")
        except Exception as error:
            print("MLflow tidak tersedia:", error)
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
