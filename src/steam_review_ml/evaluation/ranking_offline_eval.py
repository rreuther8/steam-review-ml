"""Offline ranking eval on frozen retrieval pools (no retriever re-score)."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from steam_review_ml.evaluation.example_cohort import load_retrieval_pools_jsonl
from steam_review_ml.evaluation.heuristic_ranker import PoolRerankSpec, pool_rerank_registry, rerank_scores_on_pool
from steam_review_ml.evaluation.retrieval_offline_eval import (
    RANKING_REPORT_METRIC_COLS,
    _append_personalization_metrics,
    _attach_retrieval_method,
    _oracle_ranked_indices_from_retrieved,
    _personalization_by_group,
    _rank_rows,
    _slice_and_bucket_for_example,
    _table_by_slice_for_metrics,
    _table_by_support_for_metrics,
    _table_overall_ranking,
    _table_personalization,
    _table_popularity,
    average_precision_at_k,
    hit_rate_at_k,
    load_eval_examples_from_parquet,
    load_ranking_catalog_context,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from steam_review_ml.recommender.retrieve import ContentRetriever


@dataclass(frozen=True)
class RankingEvalTables:
    ranking_overall: pd.DataFrame
    ranking_by_slice: pd.DataFrame
    ranking_by_support_bucket: pd.DataFrame
    ranking_by_pop_decile: pd.DataFrame
    ranking_pop_delta_vs_popularity: pd.DataFrame
    personalization: pd.DataFrame
    run_meta: dict[str, Any]


def _parse_pool_row(row: dict[str, Any]) -> tuple[list[int], list[float], int]:
    pool_apps = [int(x) for x in json.loads(row["retrieved_app_ids_json"])]
    retr_scores = [float(x) for x in json.loads(row["retrieved_scores_json"])]
    k_retrieval = int(row.get("retrieval_k", len(pool_apps)))
    return pool_apps, retr_scores, k_retrieval


def _catalog_indices_for_pool(
    pool_app_ids: list[int], *, app_ids: np.ndarray, app_to_row: dict[int, int]
) -> np.ndarray:
    return np.asarray([app_to_row[int(a)] for a in pool_app_ids], dtype=np.int64)


def _ranking_metrics_from_pool(
    row: dict[str, Any],
    *,
    method_name: str,
    spec: PoolRerankSpec | None,
    app_ids: np.ndarray,
    app_to_row: dict[int, int],
    pop_row: np.ndarray,
    k_final: int,
) -> dict[str, Any] | None:
    positives = set(int(x) for x in json.loads(row["validation_positive_app_ids_json"]))
    if not positives:
        return None

    pool_apps, retr_scores, _k_ret = _parse_pool_row(row)
    retrieved_indices = _catalog_indices_for_pool(pool_apps, app_ids=app_ids, app_to_row=app_to_row)

    if spec is None:
        order = np.argsort(-np.asarray(retr_scores, dtype=np.float64))
        ranked_indices = retrieved_indices[order[:k_final]]
    else:
        blend = rerank_scores_on_pool(
            pool_apps,
            retr_scores,
            spec,
            pop_row=pop_row,
            app_to_row=app_to_row,
            query_app_id=int(row["query_app_id"]),
        )
        order = np.argsort(-np.asarray(blend, dtype=np.float64))
        ranked_indices = retrieved_indices[order[:k_final]]

    oracle_indices = _oracle_ranked_indices_from_retrieved(retrieved_indices, positives, app_ids)

    ex_stub = {
        "n_eval_targets": int(row["n_eval_targets"]),
        "train_review_rows": [{}] * int(row.get("n_support_train", 0)),
    }
    slice_name, _bucket = _slice_and_bucket_for_example(ex_stub)
    if row.get("slice_name"):
        slice_name = str(row["slice_name"])

    return {
        "method": method_name,
        "ex_idx": int(row["ex_idx"]),
        "slice_name": slice_name,
        "user_id": row["user_id"],
        "query_app_id": int(row["query_app_id"]),
        "n_eval_targets": int(row["n_eval_targets"]),
        "n_support_train": int(row.get("n_support_train", 0)),
        "n_unique_train_apps": int(row.get("n_unique_train_apps", 0)),
        "Hit@K": hit_rate_at_k(ranked_indices, positives, k_final, app_ids),
        "Precision@K": precision_at_k(ranked_indices, positives, k_final, app_ids),
        "Recall@K": recall_at_k(ranked_indices, positives, k_final, app_ids),
        "MAP@K": average_precision_at_k(ranked_indices, positives, k_final, app_ids),
        "NDCG@K": ndcg_at_k(ranked_indices, positives, k_final, app_ids),
        "MRR": mrr(ranked_indices, positives, app_ids),
        "OracleHit@K": hit_rate_at_k(oracle_indices, positives, k_final, app_ids),
        "OracleNDCG@K": ndcg_at_k(oracle_indices, positives, k_final, app_ids),
    }


def _make_frozen_pool_score_fn(
    *,
    pools_by_ex: dict[int, dict[str, Any]],
    spec: PoolRerankSpec,
    app_ids: np.ndarray,
    app_to_row: dict[int, int],
    pop_row: np.ndarray,
) -> Callable[[dict], np.ndarray]:
    def score(ex: dict) -> np.ndarray:
        ex_idx = int(ex["ex_idx"])
        row = pools_by_ex[ex_idx]
        pool_apps, retr_scores, _ = _parse_pool_row(row)
        blend = rerank_scores_on_pool(
            pool_apps,
            retr_scores,
            spec,
            pop_row=pop_row,
            app_to_row=app_to_row,
            query_app_id=int(ex["query_app_id"]),
        )
        retrieved_indices = _catalog_indices_for_pool(pool_apps, app_ids=app_ids, app_to_row=app_to_row)
        full = np.full(len(app_ids), -np.inf, dtype=np.float64)
        for idx, score_val in zip(retrieved_indices, blend):
            full[int(idx)] = float(score_val)
        return full.astype(np.float32)

    return score


def _make_frozen_pool_retrieval_score_fn(
    *,
    pools_by_ex: dict[int, dict[str, Any]],
    app_ids: np.ndarray,
    app_to_row: dict[int, int],
) -> Callable[[dict], np.ndarray]:
    """Top-k personalization scores = frozen pool retrieval order (no rerank)."""

    def score(ex: dict) -> np.ndarray:
        ex_idx = int(ex["ex_idx"])
        row = pools_by_ex[ex_idx]
        pool_apps, retr_scores, _ = _parse_pool_row(row)
        retrieved_indices = _catalog_indices_for_pool(pool_apps, app_ids=app_ids, app_to_row=app_to_row)
        full = np.full(len(app_ids), -np.inf, dtype=np.float64)
        for idx, score_val in zip(retrieved_indices, retr_scores):
            full[int(idx)] = float(score_val)
        return full.astype(np.float32)

    return score


def _load_pools_by_ex(jsonl_path: Path, pool_method: str) -> dict[int, dict[str, Any]]:
    rows = load_retrieval_pools_jsonl(jsonl_path, method=pool_method)
    return {int(r["ex_idx"]): r for r in rows}


def _popularity_train_catalog_scores(
    ex: dict, *, pop_row: np.ndarray, app_to_row: dict[int, int]
) -> np.ndarray:
    """Same scorer as retrieval eval: global train popularity, query app masked."""
    s = np.asarray(pop_row, dtype=np.float64).copy()
    row = app_to_row.get(int(ex["query_app_id"]))
    if row is not None:
        s[row] = -np.inf
    return s


def _ranking_metrics_from_catalog_scores(
    ex: dict,
    ex_idx: int,
    *,
    method_name: str,
    scores: np.ndarray,
    app_ids: np.ndarray,
    k_retrieval: int,
    k_final: int,
) -> dict[str, Any] | None:
    positives = set(int(x) for x in ex["validation_positive_app_ids"])
    if not positives:
        return None

    full_order = _rank_rows(scores)
    retrieved_indices = np.asarray(full_order[:k_retrieval], dtype=np.int64)
    ranked_indices = np.asarray(full_order[:k_final], dtype=np.int64)
    oracle_indices = _oracle_ranked_indices_from_retrieved(retrieved_indices, positives, app_ids)
    slice_name, _bucket = _slice_and_bucket_for_example(ex)
    n_support = len(ex.get("train_review_rows", []))

    return {
        "method": method_name,
        "ex_idx": int(ex_idx),
        "slice_name": slice_name,
        "user_id": ex["user_id"],
        "query_app_id": int(ex["query_app_id"]),
        "n_eval_targets": int(ex["n_eval_targets"]),
        "n_support_train": int(n_support),
        "n_unique_train_apps": int(len({int(r["app_id"]) for r in ex.get("train_review_rows", [])})),
        "Hit@K": hit_rate_at_k(ranked_indices, positives, k_final, app_ids),
        "Precision@K": precision_at_k(ranked_indices, positives, k_final, app_ids),
        "Recall@K": recall_at_k(ranked_indices, positives, k_final, app_ids),
        "MAP@K": average_precision_at_k(ranked_indices, positives, k_final, app_ids),
        "NDCG@K": ndcg_at_k(ranked_indices, positives, k_final, app_ids),
        "MRR": mrr(ranked_indices, positives, app_ids),
        "OracleHit@K": hit_rate_at_k(oracle_indices, positives, k_final, app_ids),
        "OracleNDCG@K": ndcg_at_k(oracle_indices, positives, k_final, app_ids),
    }


def run_ranking_eval(
    *,
    repo_root: Path,
    pools_jsonl: Path,
    pool_methods: list[str],
    ranker_methods: list[str],
    examples_parquet: Path,
    catalog_methods: list[str] | None = None,
    k_final: int = 10,
    k_retrieval: int = 100,
    k_personalization: int = 10,
    min_review_chars: int = 30,
    enable_popularity_decile_diagnostics: bool = True,
    include_personalization: bool = True,
    artifact_dir: Path | None = None,
    verbose: bool = False,
) -> RankingEvalTables:
    if not pools_jsonl.is_file():
        raise FileNotFoundError(f"pools_jsonl not found: {pools_jsonl}")
    if not examples_parquet.is_file():
        raise FileNotFoundError(f"examples_parquet not found: {examples_parquet}")

    rerank_specs = pool_rerank_registry()
    unknown = sorted(set(ranker_methods).difference(rerank_specs.keys()))
    if unknown:
        raise ValueError(f"Unknown ranker_methods: {unknown}. Available={sorted(rerank_specs.keys())}")

    retriever = ContentRetriever(artifact_dir=artifact_dir, repo_root=repo_root)
    catalog = load_ranking_catalog_context(
        repo_root=repo_root,
        min_review_chars=min_review_chars,
        artifact_dir=artifact_dir,
        retriever=retriever,
    )
    app_ids = catalog.app_ids
    app_to_row = catalog.app_to_row
    pop_row = catalog.pop_row

    X = retriever.embedding_matrix

    if catalog_methods is None:
        catalog_methods = ["popularity_train"]

    examples = load_eval_examples_from_parquet(examples_parquet)
    for ex_idx, ex in enumerate(examples):
        ex["ex_idx"] = ex_idx

    pools_by_method: dict[str, dict[int, dict[str, Any]]] = {}
    for pm in pool_methods:
        pools_by_method[pm] = _load_pools_by_ex(pools_jsonl, pm)

    ranking_frames: list[pd.DataFrame] = []
    t0 = time.perf_counter()

    for catalog_method in catalog_methods:
        if catalog_method != "popularity_train":
            raise ValueError(
                f"Unsupported catalog_methods entry: {catalog_method!r}; only 'popularity_train' is wired."
            )
        rows_out: list[dict[str, Any]] = []
        ex_iter = enumerate(examples)
        if verbose:
            ex_iter = tqdm(ex_iter, total=len(examples), desc=f"rank {catalog_method}", unit="ex")
        for ex_idx, ex in ex_iter:
            scores = _popularity_train_catalog_scores(ex, pop_row=pop_row, app_to_row=app_to_row)
            m = _ranking_metrics_from_catalog_scores(
                ex,
                ex_idx,
                method_name=catalog_method,
                scores=scores,
                app_ids=app_ids,
                k_retrieval=int(k_retrieval),
                k_final=k_final,
            )
            if m is not None:
                rows_out.append(m)
        df_catalog = pd.DataFrame(rows_out)
        if not df_catalog.empty:
            df_catalog["retrieval_method"] = catalog_method
        ranking_frames.append(df_catalog)

    methods_to_score: list[tuple[str, str | None, PoolRerankSpec | None]] = []
    for rm in ranker_methods:
        spec = rerank_specs[rm]
        if spec.base_method not in pools_by_method:
            raise ValueError(
                f"ranker {rm!r} needs pool_method {spec.base_method!r} in pool_methods; got {pool_methods}"
            )
        methods_to_score.append((rm, spec.base_method, spec))

    for pm in pool_methods:
        if pm not in [m[0] for m in methods_to_score]:
            methods_to_score.append((pm, pm, None))

    for method_name, pool_method, spec in methods_to_score:
        pools = pools_by_method[pool_method]
        rows_out: list[dict[str, Any]] = []
        pool_iter = pools.values()
        if verbose:
            pool_iter = tqdm(pool_iter, total=len(pools), desc=f"rank {method_name}", unit="ex")
        for row in pool_iter:
            m = _ranking_metrics_from_pool(
                row,
                method_name=method_name,
                spec=spec,
                app_ids=app_ids,
                app_to_row=app_to_row,
                pop_row=pop_row,
                k_final=k_final,
            )
            if m is not None:
                rows_out.append(m)
        df_pool = pd.DataFrame(rows_out)
        if not df_pool.empty:
            df_pool["retrieval_method"] = pool_method
        ranking_frames.append(df_pool)

    df_ex_ranking = pd.concat(ranking_frames, ignore_index=True)
    if df_ex_ranking.empty:
        raise RuntimeError("No ranking metrics produced; check pools_jsonl and positives.")

    overall_ranking = _table_overall_ranking(df_ex_ranking)
    by_slice_ranking = _table_by_slice_for_metrics(
        df_ex_ranking, metric_cols=RANKING_REPORT_METRIC_COLS, ranking=True
    )
    by_support_ranking = _table_by_support_for_metrics(
        df_ex_ranking, metric_cols=RANKING_REPORT_METRIC_COLS, ranking=True
    )
    pop_rank, pop_delta_rank, ex_pop_map = _table_popularity(
        df_ex_metrics=df_ex_ranking,
        examples=examples,
        app_ids=app_ids,
        pop_row=pop_row,
        enable_popularity_decile_diagnostics=enable_popularity_decile_diagnostics,
        metric_cols=RANKING_REPORT_METRIC_COLS,
    )

    personalization = pd.DataFrame()
    if include_personalization:

        def _popularity_train_score(ex: dict) -> np.ndarray:
            s = pop_row.copy()
            row = app_to_row.get(int(ex["query_app_id"]))
            if row is not None:
                s[row] = -np.inf
            return s

        pers_methods: dict[str, Callable[[dict], np.ndarray]] = {
            "popularity_train": _popularity_train_score,
        }
        for pm in pool_methods:
            pers_methods[pm] = _make_frozen_pool_retrieval_score_fn(
                pools_by_ex=pools_by_method[pm],
                app_ids=app_ids,
                app_to_row=app_to_row,
            )
        for rm in ranker_methods:
            spec = rerank_specs[rm]
            pers_methods[rm] = _make_frozen_pool_score_fn(
                pools_by_ex=pools_by_method[spec.base_method],
                spec=spec,
                app_ids=app_ids,
                app_to_row=app_to_row,
                pop_row=pop_row,
            )

        personalization = _table_personalization(
            methods=pers_methods,
            examples=examples,
            X=X,
            app_ids=app_ids,
            pop_row=pop_row,
            k_personalization=k_personalization,
            verbose=verbose,
        )

        ex_meta = df_ex_ranking[["ex_idx", "n_eval_targets", "n_support_train"]].drop_duplicates("ex_idx").copy()
        ex_meta["slice_name"] = np.select(
            [ex_meta["n_eval_targets"] >= 2, ex_meta["n_eval_targets"] == 1, ex_meta["n_eval_targets"] == 0],
            ["slice_a_multi_target", "slice_b_single_target", "slice_c_zero_target"],
            default="slice_other",
        )
        from steam_review_ml.evaluation.retrieval_offline_eval import _support_bucket

        ex_meta["train_support_bucket"] = ex_meta["n_support_train"].fillna(0).astype(int).map(_support_bucket)

        slice_personalization = _personalization_by_group(
            methods=pers_methods,
            examples=examples,
            X=X,
            app_ids=app_ids,
            pop_row=pop_row,
            k_personalization=k_personalization,
            group_map=ex_meta[["ex_idx", "slice_name"]],
            group_col="slice_name",
            verbose=verbose,
        )
        support_personalization = _personalization_by_group(
            methods=pers_methods,
            examples=examples,
            X=X,
            app_ids=app_ids,
            pop_row=pop_row,
            k_personalization=k_personalization,
            group_map=ex_meta[["ex_idx", "train_support_bucket"]],
            group_col="train_support_bucket",
            verbose=verbose,
        )
        pop_personalization = _personalization_by_group(
            methods=pers_methods,
            examples=examples,
            X=X,
            app_ids=app_ids,
            pop_row=pop_row,
            k_personalization=k_personalization,
            group_map=ex_pop_map[["ex_idx", "pos_pop_decile"]] if not ex_pop_map.empty else pd.DataFrame(),
            group_col="pos_pop_decile",
            verbose=verbose,
        )

        overall_ranking = _append_personalization_metrics(overall_ranking, personalization, on_keys=["method"])
        by_slice_ranking = _append_personalization_metrics(
            by_slice_ranking, slice_personalization, on_keys=["method", "slice_name"]
        )
        by_support_ranking = _append_personalization_metrics(
            by_support_ranking, support_personalization, on_keys=["method", "train_support_bucket"]
        )
        pop_rank = _append_personalization_metrics(
            pop_rank, pop_personalization, on_keys=["method", "pos_pop_decile"]
        )
        pop_delta_rank = _append_personalization_metrics(
            pop_delta_rank, pop_personalization, on_keys=["method", "pos_pop_decile"]
        )

    overall_ranking = _attach_retrieval_method(overall_ranking, df_ex_ranking)
    by_slice_ranking = _attach_retrieval_method(by_slice_ranking, df_ex_ranking)
    by_support_ranking = _attach_retrieval_method(by_support_ranking, df_ex_ranking)
    pop_rank = _attach_retrieval_method(pop_rank, df_ex_ranking)
    pop_delta_rank = _attach_retrieval_method(pop_delta_rank, df_ex_ranking)

    t1 = time.perf_counter()
    run_meta = {
        "job": "ranking_eval",
        "pools_jsonl": str(pools_jsonl.resolve()),
        "examples_parquet": str(examples_parquet.resolve()),
        "pool_methods": pool_methods,
        "ranker_methods": ranker_methods,
        "catalog_methods": list(catalog_methods),
        "k_final": int(k_final),
        "k_retrieval": int(k_retrieval),
        "k_personalization": int(k_personalization),
        "include_personalization": bool(include_personalization),
        "n_examples_ranked": int(df_ex_ranking["ex_idx"].nunique()),
        "timing_seconds": {"total": round(t1 - t0, 3)},
    }
    return RankingEvalTables(
        ranking_overall=overall_ranking,
        ranking_by_slice=by_slice_ranking,
        ranking_by_support_bucket=by_support_ranking,
        ranking_by_pop_decile=pop_rank,
        ranking_pop_delta_vs_popularity=pop_delta_rank,
        personalization=personalization,
        run_meta=run_meta,
    )
