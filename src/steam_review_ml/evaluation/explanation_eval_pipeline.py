"""Orchestration for Stage 4 explanation generation + heuristic scoring.

Shared by the exploratory notebook (``recs_030``) and the reproducible pipeline job
(``scripts/recs_job_explanation_heuristic_eval.py``) so the two don't drift apart.
Scoring primitives themselves live in ``explanation_heuristics.py``; this module owns
the I/O and the (slow) generation loop.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd

from steam_review_ml.evaluation.candidate_text import build_candidate_text_lookup
from steam_review_ml.evaluation.explanation_heuristics import (
    content_word_overlap_ratio,
    cosine_similarity,
    find_ungrounded_tags,
    is_degenerate_output,
    load_catalog_tag_vocabulary,
    load_igdb_tags,
)


def generate_explanations_for_cohort(
    rec: Any,
    backend: Any,
    cohort_df: pd.DataFrame,
    app_name_by_id: dict[int, str],
    *,
    verbose: bool = True,
) -> pd.DataFrame:
    """Top-1 ``rec.recommend()`` + ``backend.generate_explanation()`` for each cohort row.

    ``cohort_df`` needs ``query_app_id``/``query_text`` columns. Generation dominates
    runtime (~2s/example vs ~0.1-0.2s retrieve, measured on ``val_llm_mini_v1`` -- see
    ``recs_030``), so progress/timing is printed every 20 examples when ``verbose``.
    """
    rows: list[dict[str, Any]] = []
    retrieve_times: list[float] = []
    generate_times: list[float] = []
    n = len(cohort_df)
    run_start = time.perf_counter()

    for i, ex in enumerate(cohort_df.to_dict("records")):
        query_app_id = int(ex["query_app_id"])

        t0 = time.perf_counter()
        hits = rec.recommend(ex["query_text"], query_app_id=query_app_id)
        retrieve_times.append(time.perf_counter() - t0)
        top = hits.iloc[0]
        rec_app_id = int(top["app_id"])

        game_texts = build_candidate_text_lookup([query_app_id, rec_app_id])

        t0 = time.perf_counter()
        explanation = backend.generate_explanation(game_texts[query_app_id], game_texts[rec_app_id])
        generate_times.append(time.perf_counter() - t0)

        rows.append(
            {
                "query_app_id": query_app_id,
                "query_app_name": app_name_by_id.get(query_app_id, ""),
                "query_text": ex["query_text"],
                "rec_app_id": rec_app_id,
                "rec_app_name": top.get("app_name", ""),
                "rec_score": float(top["score"]),
                "candidate_text": game_texts[rec_app_id],
                "explanation": explanation,
            }
        )

        if verbose and ((i + 1) % 20 == 0 or (i + 1) == n):
            elapsed = time.perf_counter() - run_start
            print(
                f"[{i + 1}/{n}] elapsed={elapsed:.0f}s "
                f"retrieve_avg={sum(retrieve_times) / len(retrieve_times):.2f}s "
                f"generate_avg={sum(generate_times) / len(generate_times):.2f}s"
            )

    if verbose and retrieve_times:
        print(
            f"\nTotal: {time.perf_counter() - run_start:.0f}s over {n} examples\n"
            f"  retrieve: total={sum(retrieve_times):.0f}s mean={sum(retrieve_times) / n:.2f}s "
            f"max={max(retrieve_times):.2f}s\n"
            f"  generate: total={sum(generate_times):.0f}s mean={sum(generate_times) / n:.2f}s "
            f"max={max(generate_times):.2f}s"
        )

    return pd.DataFrame(rows)


def generate_or_load_explanations(
    rec: Any,
    cohort_df: pd.DataFrame,
    app_name_by_id: dict[int, str],
    *,
    cache_path: Path,
    gguf_path: Path,
    verbose: bool = True,
) -> pd.DataFrame:
    """Load ``cache_path`` if present, else generate (constructing the backend lazily) and cache it."""
    if cache_path.is_file():
        results_df = pd.read_parquet(cache_path)
        if verbose:
            print(f"Loaded {len(results_df)} cached explanations from {cache_path}")
        return results_df

    from steam_review_ml.recommender.llm_backends import LlamaCppBackend

    load_start = time.perf_counter()
    backend = LlamaCppBackend(str(gguf_path), n_gpu_layers=-1)
    if verbose:
        print(f"Backend load: {time.perf_counter() - load_start:.1f}s")

    results_df = generate_explanations_for_cohort(rec, backend, cohort_df, app_name_by_id, verbose=verbose)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_parquet(cache_path, index=False)
    if verbose:
        print(f"Generated and cached {len(results_df)} explanations to {cache_path}")
    return results_df


def score_explanations(
    results_df: pd.DataFrame,
    embed_model: Any,
    *,
    enriched_path: str | None = None,
) -> pd.DataFrame:
    """Groundedness proxy (content overlap + tag leakage) and two relevance proxies (embedding cosine).

    ``relevance_cosine`` (vs. ``query_text``, the user's raw review) is confounded by how
    descriptive the review happened to be -- a terse review ("Just get it.") scores low
    regardless of explanation quality, since there's little for the embedding to match
    against. ``relevance_cosine_query_game`` (vs. the query game's own IGDB text -- what
    ``generate_explanation`` was actually grounded in, alongside ``candidate_text``) isn't
    subject to that confound and is the more apples-to-apples relevance signal. Keep both:
    a gap between them is itself informative (it's exactly the review-vs-metadata mismatch
    calibration surfaced).
    """
    rec_app_ids = results_df["rec_app_id"].unique().tolist()
    tags_by_app = load_igdb_tags(rec_app_ids, enriched_path=enriched_path)
    tag_vocabulary = load_catalog_tag_vocabulary(enriched_path=enriched_path)

    query_app_ids = results_df["query_app_id"].unique().tolist()
    query_igdb_text_by_app = build_candidate_text_lookup(query_app_ids, enriched_path=enriched_path)

    scored_rows = []
    for row in results_df.to_dict("records"):
        explanation = row["explanation"]
        exp_vec = embed_model.encode([explanation])[0]
        query_vec = embed_model.encode([row["query_text"]])[0]
        query_game_vec = embed_model.encode([query_igdb_text_by_app[row["query_app_id"]]])[0]

        scored_rows.append(
            {
                **row,
                "is_degenerate": is_degenerate_output(explanation),
                "content_overlap_ratio": content_word_overlap_ratio(explanation, row["candidate_text"]),
                "ungrounded_tags": find_ungrounded_tags(
                    explanation, tags_by_app[row["rec_app_id"]], tag_vocabulary
                ),
                "relevance_cosine": cosine_similarity(exp_vec, query_vec),
                "relevance_cosine_query_game": cosine_similarity(exp_vec, query_game_vec),
            }
        )
    return pd.DataFrame(scored_rows)


def summarize_explanation_scores(scored_df: pd.DataFrame) -> dict[str, Any]:
    return {
        "n_examples": int(len(scored_df)),
        "degenerate_rate": float(scored_df["is_degenerate"].mean()),
        "any_ungrounded_tag_rate": float((scored_df["ungrounded_tags"].str.len() > 0).mean()),
        "content_overlap_ratio_mean": float(scored_df["content_overlap_ratio"].mean()),
        "content_overlap_ratio_median": float(scored_df["content_overlap_ratio"].median()),
        "relevance_cosine_mean": float(scored_df["relevance_cosine"].mean()),
        "relevance_cosine_median": float(scored_df["relevance_cosine"].median()),
        "relevance_cosine_query_game_mean": float(scored_df["relevance_cosine_query_game"].mean()),
        "relevance_cosine_query_game_median": float(scored_df["relevance_cosine_query_game"].median()),
    }
