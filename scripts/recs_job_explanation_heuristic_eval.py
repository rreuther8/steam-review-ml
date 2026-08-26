"""Stage 4 explanation heuristic eval: reference-free groundedness/relevance proxies.

Diagnostic-only (no promotion bar, no regression baseline) -- catches obviously broken
or hallucinating explanations cheaply, before spending real LLM-judge API calls. See
``docs/plans/rag_extension_plan.md`` (Stage 5, Track B) and
``notebooks/ranking/recs_030_stage4_explanation_heuristic_eval.ipynb`` for the
exploratory version this mirrors.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from steam_review_ml.evaluation.explanation_eval_pipeline import (
    generate_or_load_explanations,
    score_explanations,
    summarize_explanation_scores,
)
from steam_review_ml.recommender.rag_recommender import RAGRecommender
from steam_review_ml.utils import load_config


def main() -> None:
    t_start = time.perf_counter()
    parser = argparse.ArgumentParser(description="Run Stage 4 explanation heuristic eval.")
    parser.add_argument("config", type=str, help="Path to JSON config.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    repo_root = Path(__file__).resolve().parents[1]

    def _resolve(rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else repo_root / p

    cohort_parquet = _resolve(str(cfg["cohort_parquet"]))
    explanations_cache = _resolve(str(cfg["explanations_cache"]))
    gguf_path = _resolve(str(cfg["gguf_path"]))
    igdb_enriched_path = str(_resolve(str(cfg["igdb_enriched_path"])))
    output_dir = _resolve(str(cfg.get("output_dir", "artifacts/recs/explanation_eval/runs/latest")))
    verbose = bool(cfg.get("verbose", True))
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Running Stage 4 explanation heuristic eval job")
    print(f"cohort_parquet={cohort_parquet}")
    print(f"explanations_cache={explanations_cache}")
    print(f"output_dir={output_dir}")

    rec = RAGRecommender.from_serve_config(repo_root=repo_root)
    app_name_by_id = dict(zip(rec.retriever.index_frame["app_id"], rec.retriever.index_frame["app_name"]))

    cohort_df = pd.read_parquet(cohort_parquet)[["query_app_id", "query_text"]]
    results_df = generate_or_load_explanations(
        rec,
        cohort_df,
        app_name_by_id,
        cache_path=explanations_cache,
        gguf_path=gguf_path,
        verbose=verbose,
    )

    from sentence_transformers import SentenceTransformer

    embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    scored_df = score_explanations(results_df, embed_model, enriched_path=igdb_enriched_path)

    scores_path = output_dir / "explanation_heuristic_scores.parquet"
    scored_df.to_parquet(scores_path, index=False)
    print(f"Wrote {scores_path}")

    summary = summarize_explanation_scores(scored_df)
    summary_path = output_dir / "explanation_heuristic_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {summary_path}")
    print(json.dumps(summary, indent=2))

    print(f"Total script runtime: {time.perf_counter() - t_start:.2f}s")


if __name__ == "__main__":
    main()
