"""Offline retrieval + ranking evaluation (contract tables, baselines, fusion methods)."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from steam_review_ml.constants import PROJECT_RANDOM_SEED
from steam_review_ml.data.loaders import (
    NORMALIZED_SPLIT_FILENAMES,
    load_normalized_split_df,
    resolve_normalized_split_parquet,
)
from steam_review_ml.recommender.math_utils import l2_normalize
from steam_review_ml.recommender.retrieve import (
    ContentRetriever,
    METHOD_FUSION_C_RAW_PLUS_BEHAVIOR,
    fusion_c_raw_plus_behavior_query_vector,
)

USER_COL = "author.steamid"
TIME_COL = "timestamp_created"
METRIC_COLS = ["Hit@K", "Precision@K", "Recall@K", "MAP@K", "NDCG@K", "MRR"]
RETRIEVAL_METRIC_COLS = ["Hit@K", "Precision@K", "Recall@K"]
MASKING_POLICY_VERSION = "mask_query_app_v1"
SUPPORT_BUCKET_ORDER = ["0", "1", "2-3", "4-7", "8+"]
REQUIRED_PHASE1_METHODS = frozenset({"raw", "popularity_train", "multi_mean_train"})
PERSONALIZATION_METRIC_PREFIXES = (
    "ILD@",
    "CatalogCoverage@",
    "Novelty@",
    "PersonalizationGapVsPopularity@",
)
TRAIN_ROW_OPTIONAL_FIELDS = (
    "_norm_author__playtime_at_review",
    "_norm_author__playtime_last_two_weeks",
    "_norm_author__num_games_owned",
    "_norm_author__num_reviews",
    "_norm_review_word_count",
)

@dataclass(frozen=True)
class EvalInputs:
    retriever: ContentRetriever
    examples: list[dict]
    embedding_matrix: np.ndarray
    app_ids: np.ndarray
    app_to_row: dict[int, int]
    pop_row: np.ndarray
    eval_split_name: str
    prep_diagnostics: dict


@dataclass(frozen=True)
class EvalTables:
    """Ranking tables mirror legacy full-metric aggregates; retrieval tables subset metrics."""

    retrieval_overall: pd.DataFrame
    retrieval_by_slice: pd.DataFrame
    retrieval_by_support_bucket: pd.DataFrame
    retrieval_by_pop_decile: pd.DataFrame
    retrieval_pop_delta_vs_popularity: pd.DataFrame
    ranking_overall: pd.DataFrame
    ranking_by_slice: pd.DataFrame
    ranking_by_support_bucket: pd.DataFrame
    ranking_by_pop_decile: pd.DataFrame
    ranking_pop_delta_vs_popularity: pd.DataFrame
    personalization: pd.DataFrame
    artifact_rows: list[dict[str, Any]]
    run_meta: dict


def _support_bucket(n: int) -> str:
    n = int(n)
    if n <= 0:
        return "0"
    if n == 1:
        return "1"
    if n <= 3:
        return "2-3"
    if n <= 7:
        return "4-7"
    return "8+"


def _rank_rows(scores: np.ndarray) -> np.ndarray:
    return np.argsort(-scores)


def recall_at_k(ranked_rows: np.ndarray, positives: set[int], k: int, app_ids: np.ndarray) -> float:
    if not positives:
        return float("nan")
    top = set(int(app_ids[i]) for i in ranked_rows[:k])
    return len(top & positives) / len(positives)


def hit_rate_at_k(ranked_rows: np.ndarray, positives: set[int], k: int, app_ids: np.ndarray) -> float:
    top = set(int(app_ids[i]) for i in ranked_rows[:k])
    return 1.0 if (top & positives) else 0.0


def precision_at_k(ranked_rows: np.ndarray, positives: set[int], k: int, app_ids: np.ndarray) -> float:
    if k <= 0:
        return float("nan")
    top = set(int(app_ids[i]) for i in ranked_rows[:k])
    return len(top & positives) / float(k)


def mrr(ranked_rows: np.ndarray, positives: set[int], app_ids: np.ndarray) -> float:
    for rank, i in enumerate(ranked_rows.tolist(), start=1):
        if int(app_ids[i]) in positives:
            return 1.0 / rank
    return 0.0


def average_precision_at_k(ranked_rows: np.ndarray, positives: set[int], k: int, app_ids: np.ndarray) -> float:
    if not positives:
        return float("nan")
    hits = 0
    prec_sum = 0.0
    for rank, i in enumerate(ranked_rows[:k].tolist(), start=1):
        if int(app_ids[i]) in positives:
            hits += 1
            prec_sum += hits / rank
    return prec_sum / len(positives)


def ndcg_at_k(ranked_rows: np.ndarray, positives: set[int], k: int, app_ids: np.ndarray) -> float:
    if not positives:
        return float("nan")
    gains = [1.0 if int(app_ids[i]) in positives else 0.0 for i in ranked_rows[:k]]

    def dcg(g: list[float]) -> float:
        return sum(rel / math.log2(idx + 2) for idx, rel in enumerate(g))

    ideal_len = min(len(positives), k)
    ideal_gains = [1.0] * ideal_len + [0.0] * max(0, k - ideal_len)
    idcg = dcg(ideal_gains)
    if idcg <= 1e-12:
        return 0.0
    return dcg(gains) / idcg


def _build_eval_records(df_val_all: pd.DataFrame, df_train_all: pd.DataFrame) -> pd.DataFrame:
    val_pos_user_stats = (
        df_val_all.loc[df_val_all["recommended"] == 1]
        .groupby(USER_COL)
        .agg(n_val_pos_rows=("app_id", "size"), n_val_pos_apps=("app_id", "nunique"))
    )
    train_any = df_train_all.groupby(USER_COL).size().rename("n_train_support")
    train_pos = (
        df_train_all.loc[df_train_all["recommended"] == 1]
        .groupby(USER_COL)
        .size()
        .rename("n_train_pos_support")
    )
    cohort_user = (
        pd.DataFrame(index=pd.Index(df_val_all[USER_COL].unique(), name=USER_COL))
        .join(train_any, how="left")
        .join(train_pos, how="left")
        .join(val_pos_user_stats, how="left")
        .fillna(0)
        .astype(
            {
                "n_train_support": int,
                "n_train_pos_support": int,
                "n_val_pos_rows": int,
                "n_val_pos_apps": int,
            }
        )
        .reset_index()
    )
    cohort_user["eval_pos_cohort"] = np.select(
        [
            cohort_user["n_val_pos_apps"] >= 2,
            cohort_user["n_val_pos_apps"] == 1,
            cohort_user["n_val_pos_apps"] == 0,
        ],
        ["val_multi_pos_eval", "val_single_pos_eval", "val_no_pos_eval"],
        default="val_no_pos_eval",
    )
    cohort_user["cohort"] = np.select(
        [
            cohort_user["n_train_pos_support"] >= 2,
            cohort_user["n_train_pos_support"] == 1,
            (cohort_user["n_train_support"] >= 1) & (cohort_user["n_train_pos_support"] == 0),
            cohort_user["n_train_support"] == 0,
        ],
        ["val_multi_pos_train", "val_pos_train", "val_train", "val_no_train"],
        default="val_no_train",
    )
    df_val_pos = df_val_all.loc[df_val_all["recommended"] == 1].copy()
    df_eval_records = df_val_pos.merge(
        cohort_user[
            [
                USER_COL,
                "cohort",
                "eval_pos_cohort",
                "n_train_support",
                "n_train_pos_support",
                "n_val_pos_rows",
                "n_val_pos_apps",
            ]
        ],
        on=USER_COL,
        how="left",
        validate="many_to_one",
    )
    return df_eval_records[
        [
            USER_COL,
            "review_id",
            "app_id",
            "review",
            "ts",
            "cohort",
            "eval_pos_cohort",
            "n_train_support",
            "n_train_pos_support",
            "n_val_pos_rows",
            "n_val_pos_apps",
        ]
    ].reset_index(drop=True)


def _sample_eval_base(
    df_eval_records: pd.DataFrame,
    *,
    active_cohort: str,
    max_examples: int,
    cohort_sizing: dict[tuple[str, str], float],
    rng: np.random.Generator,
) -> pd.DataFrame:
    eval_base = (
        df_eval_records.copy()
        if active_cohort == "all"
        else df_eval_records[df_eval_records["cohort"] == active_cohort].copy()
    )
    if len(eval_base) == 0:
        raise RuntimeError(f"No eval records available for ACTIVE_COHORT={active_cohort!r}.")
    if max_examples <= 0 or len(eval_base) <= max_examples:
        return eval_base.reset_index(drop=True)
    if not cohort_sizing:
        take = rng.choice(eval_base.index.to_numpy(), size=max_examples, replace=False)
        return eval_base.loc[take].reset_index(drop=True)

    chosen_idx: list[int] = []
    for (eval_pos_cohort, cohort), pct in cohort_sizing.items():
        n_take = int(round(max_examples * float(pct)))
        if n_take <= 0:
            continue
        mask = (eval_base["eval_pos_cohort"] == eval_pos_cohort) & (eval_base["cohort"] == cohort)
        eligible = eval_base.loc[mask]
        if len(eligible) == 0:
            continue
        n_take = min(n_take, len(eligible))
        take_idx = rng.choice(eligible.index.to_numpy(), size=n_take, replace=False)
        chosen_idx.extend(take_idx.tolist())
    chosen_idx = list(dict.fromkeys(chosen_idx))
    if len(chosen_idx) < max_examples:
        remaining = eval_base.loc[~eval_base.index.isin(chosen_idx)]
        fill_take = min(max_examples - len(chosen_idx), len(remaining))
        if fill_take > 0:
            fill_idx = rng.choice(remaining.index.to_numpy(), size=fill_take, replace=False)
            chosen_idx.extend(fill_idx.tolist())
    elif len(chosen_idx) > max_examples:
        chosen_idx = rng.choice(np.asarray(chosen_idx, dtype=int), size=max_examples, replace=False).tolist()
    return eval_base.loc[chosen_idx].reset_index(drop=True)


def _build_examples(
    eval_base: pd.DataFrame,
    user_to_eval_apps: dict[str, list[int]],
    *,
    df_train_all: pd.DataFrame,
    max_train_rows_per_user: int,
    support_app_filter_mode: str,
    rng: np.random.Generator,
    verbose: bool = False,
) -> tuple[list[dict], dict]:
    train_pos = df_train_all[df_train_all["recommended"] == 1].copy()
    optional_fields = [c for c in TRAIN_ROW_OPTIONAL_FIELDS if c in train_pos.columns]
    train_rows_by_user: dict[str, list[dict]] = {}
    for uid, g in train_pos.groupby(USER_COL):
        rows_out: list[dict] = []
        for rec in g.to_dict(orient="records"):
            row = {
                "app_id": int(rec["app_id"]),
                "text": str(rec["review"]),
                "ts": float(rec["ts"]),
            }
            for f in optional_fields:
                v = rec.get(f)
                row[f] = None if pd.isna(v) else v
            rows_out.append(row)
        train_rows_by_user[str(uid)] = rows_out
    examples: list[dict] = []
    drop_reasons: dict[str, int] = {
        "no_other_positive_app": 0,
    }
    row_iter = eval_base.iterrows()
    if verbose:
        row_iter = tqdm(
            row_iter,
            total=len(eval_base),
            desc="build eval examples",
            unit="row",
        )
    for _, r in row_iter:
        uid = str(r[USER_COL])
        q_app = int(r["app_id"])
        apps = user_to_eval_apps.get(uid, [])
        validation_positive_app_ids = {int(a) for a in apps if int(a) != q_app}
        if not validation_positive_app_ids:
            drop_reasons["no_other_positive_app"] += 1
            continue
        rows = train_rows_by_user.get(uid, [])
        if support_app_filter_mode == "strict":
            exclude_apps = set(validation_positive_app_ids) | {q_app}
            rows = [x for x in rows if int(x["app_id"]) not in exclude_apps]
        elif support_app_filter_mode == "query_only":
            rows = [x for x in rows if int(x["app_id"]) != q_app]
        else:
            raise ValueError(f"support_app_filter_mode must be query_only|strict, got {support_app_filter_mode!r}")
        if max_train_rows_per_user > 0 and len(rows) > max_train_rows_per_user:
            chosen = rng.choice(np.arange(len(rows)), size=max_train_rows_per_user, replace=False)
            rows = [rows[i] for i in sorted(chosen.tolist())]
        examples.append(
            {
                "user_id": uid,
                "query_app_id": q_app,
                "query_text": str(r["review"]),
                "query_ts": float(r["ts"]),
                "validation_positive_app_ids": validation_positive_app_ids,
                "n_eval_targets": len(validation_positive_app_ids),
                "train_review_rows": rows,
                "cohort": str(r["cohort"]),
                "eval_pos_cohort": str(r["eval_pos_cohort"]),
            }
        )
    diagnostics = {
        "sampled_rows": int(len(eval_base)),
        "evaluable_examples": int(len(examples)),
        "dropped_rows": int(len(eval_base) - len(examples)),
        "drop_reasons": drop_reasons,
    }
    return examples, diagnostics


def _query_vector_raw(retriever: ContentRetriever, ex: dict) -> np.ndarray:
    return retriever.embed_text(ex["query_text"])


def _query_vector_multi_mean_train(retriever: ContentRetriever, ex: dict, *, multi_max_reviews: int) -> np.ndarray:
    texts = [str(ex["query_text"]).strip()]
    for row in ex.get("train_review_rows", [])[: max(0, int(multi_max_reviews) - 1)]:
        t = row.get("text")
        if t and str(t).strip():
            texts.append(str(t).strip())
    if len(texts) == 1:
        return retriever.embed_text(texts[0])
    vecs = np.stack([retriever.embed_text(t) for t in texts], axis=0).astype(np.float32)
    return l2_normalize(vecs.mean(axis=0))


def _scores_from_query(
    q: np.ndarray,
    *,
    X: np.ndarray,
    app_to_row: dict[int, int],
    query_app_id: int,
    mask_query_app: bool,
) -> np.ndarray:
    s = (X @ q).astype(np.float32)
    if mask_query_app:
        row = app_to_row.get(int(query_app_id))
        if row is not None:
            s[row] = -np.inf
    return s


def _build_method_registry(
    *,
    retriever: ContentRetriever,
    X: np.ndarray,
    pop_row: np.ndarray,
    app_to_row: dict[int, int],
    multi_max_reviews: int,
    rng: np.random.Generator,
    mask_query_app: bool,
) -> dict[str, Callable[[dict], np.ndarray]]:
    def score_raw(ex: dict) -> np.ndarray:
        q = _query_vector_raw(retriever, ex)
        return _scores_from_query(q, X=X, app_to_row=app_to_row, query_app_id=int(ex["query_app_id"]), mask_query_app=mask_query_app)

    def score_popularity_train(ex: dict) -> np.ndarray:
        s = pop_row.copy()
        if mask_query_app:
            row = app_to_row.get(int(ex["query_app_id"]))
            if row is not None:
                s[row] = -np.inf
        return s

    def score_multi_mean_train(ex: dict) -> np.ndarray:
        q = _query_vector_multi_mean_train(retriever, ex, multi_max_reviews=multi_max_reviews)
        return _scores_from_query(q, X=X, app_to_row=app_to_row, query_app_id=int(ex["query_app_id"]), mask_query_app=mask_query_app)

    def score_random(ex: dict) -> np.ndarray:
        s = rng.random(X.shape[0]).astype(np.float32)
        if mask_query_app:
            row = app_to_row.get(int(ex["query_app_id"]))
            if row is not None:
                s[row] = -np.inf
        return s

    def score_fusion_c_raw_plus_behavior(ex: dict) -> np.ndarray:
        q = fusion_c_raw_plus_behavior_query_vector(
            retriever,
            ex,
            embedding_matrix=X,
            app_to_row=app_to_row,
        )
        return _scores_from_query(
            q,
            X=X,
            app_to_row=app_to_row,
            query_app_id=int(ex["query_app_id"]),
            mask_query_app=mask_query_app,
        )

    return {
        "raw": score_raw,
        "popularity_train": score_popularity_train,
        "multi_mean_train": score_multi_mean_train,
        METHOD_FUSION_C_RAW_PLUS_BEHAVIOR: score_fusion_c_raw_plus_behavior,
        "random": score_random,
    }


def prepare_eval_inputs(
    *,
    repo_root: Path,
    split: str,
    active_cohort: str,
    max_examples: int,
    support_app_filter_mode: str,
    cohort_sizing: dict[tuple[str, str], float],
    min_review_chars: int,
    max_train_rows_per_user: int,
    random_seed: int = PROJECT_RANDOM_SEED,
    artifact_dir: Path | None = None,
    verbose: bool = False,
) -> EvalInputs:
    retriever = ContentRetriever(artifact_dir=artifact_dir, repo_root=repo_root)
    X = retriever.embedding_matrix
    app_ids = retriever.app_ids
    app_to_row = {int(a): i for i, a in enumerate(app_ids)}
    indexed_apps = set(int(a) for a in app_ids.tolist())

    processed = repo_root / "data" / "processed"
    eval_parquet, eval_split_name = resolve_normalized_split_parquet(processed, split)
    train_parquet = processed / NORMALIZED_SPLIT_FILENAMES["train"]
    if not eval_parquet.is_file():
        raise FileNotFoundError(f"Missing eval parquet: {eval_parquet}")
    if not train_parquet.is_file():
        raise FileNotFoundError(f"Missing train parquet: {train_parquet}")

    df_val_all = load_normalized_split_df(
        eval_parquet,
        indexed_apps=indexed_apps,
        min_review_chars=min_review_chars,
        user_col=USER_COL,
        time_col=TIME_COL,
    )
    df_train_all = load_normalized_split_df(
        train_parquet,
        indexed_apps=indexed_apps,
        min_review_chars=min_review_chars,
        user_col=USER_COL,
        time_col=TIME_COL,
        extra_columns=TRAIN_ROW_OPTIONAL_FIELDS,
    )
    if verbose:
        print(
            "Loaded split rows: "
            f"eval={len(df_val_all):,} train={len(df_train_all):,} "
            f"split_used={eval_split_name}"
        )
    df_eval_records = _build_eval_records(df_val_all, df_train_all)
    user_to_eval_apps = (
        df_eval_records.groupby(USER_COL)["app_id"]
        .apply(lambda s: sorted({int(x) for x in s}))
        .to_dict()
    )

    rng = np.random.default_rng(int(random_seed))
    eval_base = _sample_eval_base(
        df_eval_records,
        active_cohort=active_cohort,
        max_examples=int(max_examples),
        cohort_sizing=cohort_sizing,
        rng=rng,
    )
    examples, build_diagnostics = _build_examples(
        eval_base,
        user_to_eval_apps={str(k): v for k, v in user_to_eval_apps.items()},
        df_train_all=df_train_all,
        max_train_rows_per_user=max_train_rows_per_user,
        support_app_filter_mode=support_app_filter_mode,
        rng=rng,
        verbose=verbose,
    )
    if len(examples) == 0:
        raise RuntimeError("No evaluable examples created from selected cohort/split settings.")
    if verbose:
        print(
            "Prepared evaluation inputs: "
            f"records={len(df_eval_records):,} sampled={len(eval_base):,} "
            f"evaluable_examples={len(examples):,}"
        )
        print("Drop reasons:", build_diagnostics.get("drop_reasons", {}))

    train_pos = df_train_all[df_train_all["recommended"] == 1]
    vc = train_pos.groupby("app_id").size()
    pop_row = np.asarray([float(vc.get(int(a), 0.0)) for a in app_ids], dtype=np.float32)
    pop_row = np.maximum(pop_row, 1e-6).astype(np.float32)

    return EvalInputs(
        retriever=retriever,
        examples=examples,
        embedding_matrix=X,
        app_ids=app_ids,
        app_to_row=app_to_row,
        pop_row=pop_row,
        eval_split_name=eval_split_name,
        prep_diagnostics={
            "eval_records_count": int(len(df_eval_records)),
            "sampled_rows_count": int(len(eval_base)),
            "full_eval_user_count": int(df_eval_records[USER_COL].nunique()),
            "full_eval_multi_pos_user_count": int(
                sum(1 for apps in user_to_eval_apps.values() if len(apps) >= 2)
            ),
            **build_diagnostics,
        },
    )


def load_eval_examples_from_parquet(parquet_path: Path) -> list[dict]:
    """Load eval examples produced by ``recs_job_build_eval_examples.py`` (stable parquet schema).

    Returns the same ``list[dict]`` shape ``prepare_eval_inputs`` uses so ``run_retrieval_eval`` stays unchanged.
    """
    df = pd.read_parquet(parquet_path)
    required_cols = {
        "user_id",
        "query_app_id",
        "query_text",
        "query_ts",
        "n_eval_targets",
        "cohort",
        "eval_pos_cohort",
        "validation_positive_app_ids_json",
        "train_review_rows_json",
    }
    missing = sorted(c for c in required_cols if c not in df.columns)
    if missing:
        raise ValueError(f"Cached eval examples parquet missing columns: {missing}")

    examples: list[dict] = []
    for rec in df.to_dict(orient="records"):
        examples.append(
            {
                "user_id": str(rec["user_id"]),
                "query_app_id": int(rec["query_app_id"]),
                "query_text": str(rec["query_text"]),
                "query_ts": float(rec["query_ts"]),
                "validation_positive_app_ids": set(
                    int(a) for a in json.loads(rec["validation_positive_app_ids_json"])
                ),
                "n_eval_targets": int(rec["n_eval_targets"]),
                "train_review_rows": json.loads(rec["train_review_rows_json"]),
                "cohort": str(rec["cohort"]),
                "eval_pos_cohort": str(rec["eval_pos_cohort"]),
            }
        )
    return examples


def prepare_eval_inputs_from_cache(
    *,
    repo_root: Path,
    split: str,
    min_review_chars: int,
    examples_parquet: Path,
    artifact_dir: Path | None = None,
    verbose: bool = False,
) -> EvalInputs:
    """Retriever + popularity row from artifacts; examples from parquet (no cohort resampling).

    Caller is responsible for using a parquet that matches offline eval semantics (typically built via
    ``recs_job_build_eval_examples.py`` with the intended split/cohort/seed frozen in its meta).
    ``split`` is still resolved for ``eval_split_name`` parity with scripted runs on the same config.
    """
    retriever = ContentRetriever(artifact_dir=artifact_dir, repo_root=repo_root)
    X = retriever.embedding_matrix
    app_ids = retriever.app_ids
    app_to_row = {int(a): i for i, a in enumerate(app_ids)}
    indexed_apps = set(int(a) for a in app_ids.tolist())

    processed = repo_root / "data" / "processed"
    eval_parquet, eval_split_name = resolve_normalized_split_parquet(processed, split)
    train_parquet = processed / NORMALIZED_SPLIT_FILENAMES["train"]
    if not train_parquet.is_file():
        raise FileNotFoundError(f"Missing train parquet: {train_parquet}")

    df_train_all = load_normalized_split_df(
        train_parquet,
        indexed_apps=indexed_apps,
        min_review_chars=min_review_chars,
        user_col=USER_COL,
        time_col=TIME_COL,
        extra_columns=TRAIN_ROW_OPTIONAL_FIELDS,
    )
    train_pos = df_train_all[df_train_all["recommended"] == 1]
    vc = train_pos.groupby("app_id").size()
    pop_row = np.asarray([float(vc.get(int(a), 0.0)) for a in app_ids], dtype=np.float32)
    pop_row = np.maximum(pop_row, 1e-6).astype(np.float32)

    examples = load_eval_examples_from_parquet(examples_parquet)
    if not examples:
        raise RuntimeError(f"No examples loaded from {examples_parquet}")
    if verbose:
        print(f"Loaded {len(examples):,} cached eval examples from {examples_parquet}")

    return EvalInputs(
        retriever=retriever,
        examples=examples,
        embedding_matrix=X,
        app_ids=app_ids,
        app_to_row=app_to_row,
        pop_row=pop_row,
        eval_split_name=eval_split_name,
        prep_diagnostics={
            "examples_source": "parquet_cache",
            "examples_parquet": str(examples_parquet.resolve()),
            "n_examples_loaded": int(len(examples)),
            "eval_split_parquet": str(eval_parquet),
        },
    )


def _embedding_model_snapshot(retriever: ContentRetriever | None) -> str:
    if retriever is None:
        return "unknown"
    url = getattr(retriever, "_tfhub_url", None)
    return str(url) if url else "unknown"


def _slice_and_bucket_for_example(ex: dict) -> tuple[str, str]:
    n_eval = int(ex.get("n_eval_targets", 0))
    if n_eval >= 2:
        slice_name = "slice_a_multi_target"
    elif n_eval == 1:
        slice_name = "slice_b_single_target"
    elif n_eval == 0:
        slice_name = "slice_c_zero_target"
    else:
        slice_name = "slice_other"
    n_sup = len(ex.get("train_review_rows", []))
    return slice_name, _support_bucket(int(n_sup))


def _per_example_retrieval_ranking(
    *,
    method_name: str,
    score_fn: Callable[[dict], np.ndarray],
    examples: list[dict],
    app_ids: np.ndarray,
    k_retrieval: int,
    k_final: int,
    masking_policy_version: str,
    model_version: str,
    verbose: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    """Score each example once; derive retrieval vs ranking metric rows and JSONL-ready artifact dicts."""
    retrieval_rows: list[dict[str, Any]] = []
    ranking_rows: list[dict[str, Any]] = []
    artifact_rows: list[dict[str, Any]] = []
    ex_iter = enumerate(examples)
    if verbose:
        ex_iter = tqdm(
            ex_iter,
            total=len(examples),
            desc=f"score {method_name}",
            unit="example",
        )
    for ex_idx, ex in ex_iter:
        validation_positive_app_ids_set = set(int(x) for x in ex["validation_positive_app_ids"])
        positives = validation_positive_app_ids_set
        if not positives:
            continue
        scores = score_fn(ex)
        full_order = _rank_rows(scores)
        retrieved_indices = np.asarray(full_order[:k_retrieval], dtype=np.int64)
        ranked_indices = np.asarray(full_order[:k_final], dtype=np.int64)

        retr_ids_list = [int(app_ids[int(i)]) for i in retrieved_indices]
        retr_scores_list = [float(scores[int(i)]) for i in retrieved_indices]
        ranked_ids_list = [int(app_ids[int(i)]) for i in ranked_indices]
        ranked_scores_list = [float(scores[int(i)]) for i in ranked_indices]

        slice_name, bucket = _slice_and_bucket_for_example(ex)
        retrieved_app_ids_set = set(retr_ids_list)
        n_positive_in_retrieved = int(len(positives & retrieved_app_ids_set))

        retrieval_rows.append(
            {
                "method": method_name,
                "ex_idx": ex_idx,
                "slice_name": slice_name,
                "user_id": ex["user_id"],
                "query_app_id": int(ex["query_app_id"]),
                "n_eval_targets": int(ex["n_eval_targets"]),
                "n_support_train": int(len(ex.get("train_review_rows", []))),
                "n_unique_train_apps": int(
                    len({int(r["app_id"]) for r in ex.get("train_review_rows", [])})
                ),
                "n_positive_in_retrieved": n_positive_in_retrieved,
                "Hit@K": hit_rate_at_k(retrieved_indices, positives, k_retrieval, app_ids),
                "Precision@K": precision_at_k(retrieved_indices, positives, k_retrieval, app_ids),
                "Recall@K": recall_at_k(retrieved_indices, positives, k_retrieval, app_ids),
            }
        )
        ranking_rows.append(
            {
                "method": method_name,
                "ex_idx": ex_idx,
                "slice_name": slice_name,
                "user_id": ex["user_id"],
                "query_app_id": int(ex["query_app_id"]),
                "n_eval_targets": int(ex["n_eval_targets"]),
                "n_support_train": int(len(ex.get("train_review_rows", []))),
                "n_unique_train_apps": int(
                    len({int(r["app_id"]) for r in ex.get("train_review_rows", [])})
                ),
                "Hit@K": hit_rate_at_k(ranked_indices, positives, k_final, app_ids),
                "Precision@K": precision_at_k(ranked_indices, positives, k_final, app_ids),
                "Recall@K": recall_at_k(ranked_indices, positives, k_final, app_ids),
                "MAP@K": average_precision_at_k(ranked_indices, positives, k_final, app_ids),
                "NDCG@K": ndcg_at_k(ranked_indices, positives, k_final, app_ids),
                "MRR": mrr(ranked_indices, positives, app_ids),
            }
        )

        ranked_set = set(ranked_ids_list)
        retr_set = set(retr_ids_list)
        assert ranked_set <= retr_set, "ranked ids must stay within retrieved candidates"

        artifact_rows.append(
            {
                "ex_idx": int(ex_idx),
                "method": method_name,
                "query_app_id": int(ex["query_app_id"]),
                "user_id": ex["user_id"],
                "validation_positive_app_ids_json": json.dumps(sorted(positives)),
                "retrieved_app_ids_json": json.dumps(retr_ids_list),
                "retrieved_scores_json": json.dumps(retr_scores_list),
                "ranked_app_ids_json": json.dumps(ranked_ids_list),
                "ranked_scores_json": json.dumps(ranked_scores_list),
                "n_eval_targets": int(ex["n_eval_targets"]),
                "slice_name": slice_name,
                "n_support_train": int(len(ex.get("train_review_rows", []))),
                "train_support_bucket": bucket,
                "retrieval_k": int(k_retrieval),
                "final_k": int(k_final),
                "masking_policy_version": masking_policy_version,
                "model_version": model_version,
            }
        )

    df_retrieval = pd.DataFrame(retrieval_rows)
    df_ranking = pd.DataFrame(ranking_rows)
    return df_retrieval, df_ranking, artifact_rows


def _coverage_from_examples(examples: list[dict]) -> pd.DataFrame:
    ex_npos = pd.DataFrame(
        {
            "ex_idx": np.arange(len(examples), dtype=int),
            "n_eval_targets": [int(ex.get("n_eval_targets", 0)) for ex in examples],
        }
    )
    n_total = int(len(ex_npos))
    n_multi_pos = int((ex_npos["n_eval_targets"] >= 2).sum())
    n_single_pos = int((ex_npos["n_eval_targets"] == 1).sum())
    n_zero_pos = int((ex_npos["n_eval_targets"] == 0).sum())
    coverage_multi_pos = float(n_multi_pos / n_total) if n_total else np.nan
    return pd.DataFrame(
        [
            {
                "n_total": n_total,
                "n_multi_pos": n_multi_pos,
                "n_single_pos": n_single_pos,
                "n_zero_pos": n_zero_pos,
                "coverage_multi_pos": coverage_multi_pos,
            }
        ]
    )


def _table_overall_generic(
    df_ex_metrics: pd.DataFrame,
    *,
    metric_cols: list[str],
    sort_cols: list[str],
    ascending: list[bool],
) -> pd.DataFrame:
    return (
        df_ex_metrics.groupby("method", observed=True)[metric_cols]
        .mean()
        .reset_index()
        .sort_values(sort_cols, ascending=ascending)
        .reset_index(drop=True)
    )


def _table_by_slice_for_metrics(df_ex_metrics: pd.DataFrame, *, metric_cols: list[str], ranking: bool) -> pd.DataFrame:
    df = df_ex_metrics.copy()
    df["slice_name"] = np.select(
        [df["n_eval_targets"] >= 2, df["n_eval_targets"] == 1, df["n_eval_targets"] == 0],
        ["slice_a_multi_target", "slice_b_single_target", "slice_c_zero_target"],
        default="slice_other",
    )
    if ranking:
        sort_spec = ["slice_name", "NDCG@K", "Hit@K"]
        ascend = [True, False, False]
    else:
        sort_spec = ["slice_name", "Recall@K", "Hit@K"]
        ascend = [True, False, False]
    return (
        df.groupby(["slice_name", "method"], observed=True)[metric_cols]
        .mean()
        .reset_index()
        .sort_values(sort_spec, ascending=ascend)
        .reset_index(drop=True)
    )


def _table_by_support_for_metrics(df_ex_metrics: pd.DataFrame, *, metric_cols: list[str], ranking: bool) -> pd.DataFrame:
    df = df_ex_metrics.copy()
    df["train_support_bucket"] = df["n_support_train"].fillna(0).astype(int).map(_support_bucket)
    df["train_support_bucket"] = pd.Categorical(
        df["train_support_bucket"], categories=SUPPORT_BUCKET_ORDER, ordered=True
    )
    if ranking:
        sort_spec = ["train_support_bucket", "NDCG@K", "MAP@K", "MRR"]
        ascend = [True, False, False, False]
    else:
        sort_spec = ["train_support_bucket", "Recall@K", "Hit@K", "Precision@K"]
        ascend = [True, False, False, False]
    return (
        df.groupby(["train_support_bucket", "method"], observed=True)[metric_cols]
        .mean()
        .reset_index()
        .sort_values(sort_spec, ascending=ascend)
        .reset_index(drop=True)
    )


def _table_overall_ranking(df_ex_metrics: pd.DataFrame) -> pd.DataFrame:
    return _table_overall_generic(
        df_ex_metrics,
        metric_cols=METRIC_COLS,
        sort_cols=["NDCG@K", "MAP@K", "MRR"],
        ascending=[False, False, False],
    )


def _table_overall_retrieval(df_ex_metrics: pd.DataFrame) -> pd.DataFrame:
    return _table_overall_generic(
        df_ex_metrics,
        metric_cols=RETRIEVAL_METRIC_COLS,
        sort_cols=["Recall@K", "Hit@K", "Precision@K"],
        ascending=[False, False, False],
    )


def _example_popularity_segments(
    *,
    examples: list[dict],
    app_ids: np.ndarray,
    pop_row: np.ndarray,
) -> pd.DataFrame:
    app_pop = {int(a): float(c) for a, c in zip(app_ids, pop_row)}
    pos_pop_rows = []
    for ex_idx, ex in enumerate(examples):
        vals = [app_pop.get(int(a), 0.0) for a in ex["validation_positive_app_ids"]]
        pos_pop_rows.append(
            {
                "ex_idx": ex_idx,
                "pos_pop_mean": float(np.mean(vals)) if vals else np.nan,
                "pos_pop_median": float(np.median(vals)) if vals else np.nan,
            }
        )
    df_ex_pop = pd.DataFrame(pos_pop_rows)
    valid = df_ex_pop["pos_pop_mean"].notna()
    if valid.sum() == 0:
        df_ex_pop["pos_pop_decile"] = pd.Series(dtype="string")
        return df_ex_pop
    df_ex_pop.loc[valid, "pos_pop_decile"] = pd.qcut(
        df_ex_pop.loc[valid, "pos_pop_mean"],
        q=10,
        labels=[f"D{i}" for i in range(1, 11)],
        duplicates="drop",
    )
    df_ex_pop["pos_pop_decile"] = df_ex_pop["pos_pop_decile"].astype("string")
    return df_ex_pop


def _table_popularity(
    *,
    df_ex_metrics: pd.DataFrame,
    examples: list[dict],
    app_ids: np.ndarray,
    pop_row: np.ndarray,
    enable_popularity_decile_diagnostics: bool,
    metric_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not enable_popularity_decile_diagnostics:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    df_ex_pop = _example_popularity_segments(examples=examples, app_ids=app_ids, pop_row=pop_row)
    if df_ex_pop["pos_pop_decile"].dropna().empty:
        return pd.DataFrame(), pd.DataFrame(), df_ex_pop
    df_seg_pop = df_ex_metrics.merge(df_ex_pop, on="ex_idx", how="left")
    pop_table = (
        df_seg_pop.dropna(subset=["pos_pop_decile"])
        .groupby(["pos_pop_decile", "method"], observed=True)[metric_cols]
        .mean()
        .reset_index()
        .sort_values(["pos_pop_decile", "method"])
        .reset_index(drop=True)
    )
    if pop_table.empty:
        return pop_table, pd.DataFrame(), df_ex_pop
    pop_ref = pop_table[pop_table["method"] == "popularity_train"][["pos_pop_decile"] + metric_cols].rename(
        columns={m: f"{m}_pop_ref" for m in metric_cols}
    )
    pop_delta = pop_table.merge(pop_ref, on="pos_pop_decile", how="left")
    for m in metric_cols:
        pop_delta[f"{m}_delta_vs_pop"] = pop_delta[m] - pop_delta[f"{m}_pop_ref"]
    pop_delta = pop_delta.sort_values(["pos_pop_decile", "method"]).reset_index(drop=True)
    return pop_table, pop_delta, df_ex_pop


def _table_personalization(
    *,
    methods: dict[str, Callable[[dict], np.ndarray]],
    examples: list[dict],
    X: np.ndarray,
    app_ids: np.ndarray,
    pop_row: np.ndarray,
    k_personalization: int,
    example_indices: set[int] | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    pop_counts = np.asarray(pop_row, dtype=np.float64)
    pop_share = (pop_counts + 1.0) / float(pop_counts.sum() + len(pop_counts))
    item_novelty = -np.log2(pop_share)

    def topk_rows(scores: np.ndarray, k: int) -> np.ndarray:
        return _rank_rows(scores)[:k]

    def ild_from_rows(rows: np.ndarray) -> float:
        if len(rows) <= 1:
            return 0.0
        emb = X[rows]
        sim = emb @ emb.T
        tri = np.triu_indices(len(rows), k=1)
        if len(tri[0]) == 0:
            return 0.0
        return float(np.mean(1.0 - sim[tri]))

    def jaccard(a: set[int], b: set[int]) -> float:
        union = a | b
        if not union:
            return 1.0
        return float(len(a & b) / len(union))

    if "popularity_train" not in methods:
        raise RuntimeError("popularity_train method is required for personalization gap metric")

    pop_topk_sets: dict[int, set[int]] = {}
    pop_ex_iter = enumerate(examples)
    if verbose:
        pop_ex_iter = tqdm(
            pop_ex_iter,
            total=len(examples),
            desc="personalization baseline popularity",
            unit="example",
        )
    for ex_idx, ex in pop_ex_iter:
        if example_indices is not None and ex_idx not in example_indices:
            continue
        rows_pop = topk_rows(methods["popularity_train"](ex), k_personalization)
        pop_topk_sets[ex_idx] = set(int(app_ids[i]) for i in rows_pop)

    out_rows: list[dict] = []
    method_iter = methods.items()
    if verbose:
        method_iter = tqdm(method_iter, total=len(methods), desc="personalization methods", unit="method")
    for method_name, score_fn in method_iter:
        ild_vals: list[float] = []
        novelty_vals: list[float] = []
        gap_vals: list[float] = []
        seen_items: set[int] = set()
        for ex_idx, ex in enumerate(examples):
            if example_indices is not None and ex_idx not in example_indices:
                continue
            top_rows = topk_rows(score_fn(ex), k_personalization)
            top_set = set(int(app_ids[i]) for i in top_rows)
            seen_items.update(top_set)
            ild_vals.append(ild_from_rows(top_rows))
            novelty_vals.append(float(np.mean(item_novelty[top_rows])))
            gap_vals.append(1.0 - jaccard(top_set, pop_topk_sets[ex_idx]))
        out_rows.append(
            {
                "method": method_name,
                f"ILD@{k_personalization}": float(np.mean(ild_vals)) if ild_vals else np.nan,
                f"CatalogCoverage@{k_personalization}": float(len(seen_items) / len(app_ids)) if len(app_ids) else np.nan,
                f"Novelty@{k_personalization}": float(np.mean(novelty_vals)) if novelty_vals else np.nan,
                f"PersonalizationGapVsPopularity@{k_personalization}": float(np.mean(gap_vals)) if gap_vals else np.nan,
            }
        )
    return pd.DataFrame(out_rows).sort_values("method").reset_index(drop=True)


def _append_personalization_metrics(
    table: pd.DataFrame,
    personalization: pd.DataFrame,
    *,
    on_keys: list[str],
) -> pd.DataFrame:
    """Attach method-level personalization metrics to method-indexed tables."""
    if table.empty or personalization.empty or "method" not in table.columns:
        return table
    metric_cols = [
        c
        for c in personalization.columns
        if any(c.startswith(prefix) for prefix in PERSONALIZATION_METRIC_PREFIXES)
    ]
    if not metric_cols:
        return table
    return table.merge(
        personalization[on_keys + metric_cols],
        on=on_keys,
        how="left",
        validate="many_to_one" if on_keys == ["method"] else "many_to_many",
    )


def _personalization_by_group(
    *,
    methods: dict[str, Callable[[dict], np.ndarray]],
    examples: list[dict],
    X: np.ndarray,
    app_ids: np.ndarray,
    pop_row: np.ndarray,
    k_personalization: int,
    group_map: pd.DataFrame,
    group_col: str,
    verbose: bool = False,
) -> pd.DataFrame:
    if group_map.empty or group_col not in group_map.columns:
        return pd.DataFrame()
    out: list[pd.DataFrame] = []
    for group_value, g in group_map.dropna(subset=[group_col]).groupby(group_col, observed=True):
        ex_idx_set = set(g["ex_idx"].astype(int).tolist())
        if not ex_idx_set:
            continue
        t = _table_personalization(
            methods=methods,
            examples=examples,
            X=X,
            app_ids=app_ids,
            pop_row=pop_row,
            k_personalization=k_personalization,
            example_indices=ex_idx_set,
            verbose=verbose,
        )
        if t.empty:
            continue
        t[group_col] = group_value
        out.append(t)
    if not out:
        return pd.DataFrame()
    return pd.concat(out, ignore_index=True)


def _retrieval_bottleneck_summary(
    df_ex: pd.DataFrame,
    *,
    k_retrieval: int,
    candidate_pool_size: int,
) -> dict[str, Any]:
    """Phase-3 bottleneck stats: positives in retrieved top-``k_retrieval`` (not ranking top-``k_final``)."""
    need = {"method", "slice_name", "n_positive_in_retrieved"}
    if df_ex.empty or not need.issubset(df_ex.columns):
        return {
            "candidate_pool_size": int(candidate_pool_size),
            "k_retrieval": int(k_retrieval),
            "by_method": {},
            "by_method_slice": [],
            "note": "no per-example bottleneck columns",
        }
    by_method: dict[str, Any] = {}
    for method, g in df_ex.groupby("method", observed=True):
        n = int(len(g))
        z = float((g["n_positive_in_retrieved"].astype(int) == 0).mean()) if n else float("nan")
        avg_p = float(g["n_positive_in_retrieved"].astype(float).mean()) if n else float("nan")
        by_method[str(method)] = {
            "n_examples": n,
            "frac_queries_zero_positive_in_topk_retrieval": z,
            "avg_positive_items_in_topk_retrieval": avg_p,
        }
    by_method_slice: list[dict[str, Any]] = []
    for (method, slice_name), g in df_ex.groupby(["method", "slice_name"], observed=True):
        n = int(len(g))
        z = float((g["n_positive_in_retrieved"].astype(int) == 0).mean()) if n else float("nan")
        avg_p = float(g["n_positive_in_retrieved"].astype(float).mean()) if n else float("nan")
        by_method_slice.append(
            {
                "method": str(method),
                "slice_name": str(slice_name),
                "n_examples": n,
                "frac_queries_zero_positive_in_topk_retrieval": z,
                "avg_positive_items_in_topk_retrieval": avg_p,
            }
        )
    return {
        "candidate_pool_size": int(candidate_pool_size),
        "k_retrieval": int(k_retrieval),
        "definition": "positive_items = |validation_positive ∩ retrieved_app_ids[0:k_retrieval]|",
        "by_method": by_method,
        "by_method_slice": by_method_slice,
    }


def _slice_b_empirical_std(
    df_ex: pd.DataFrame,
    *,
    metric_cols: tuple[str, ...],
) -> dict[str, dict[str, float]]:
    """Per-method std-dev of per-example metrics in slice_b (spread across users, not a CI)."""
    if df_ex.empty or "slice_name" not in df_ex.columns:
        return {}
    sb = df_ex[df_ex["slice_name"] == "slice_b_single_target"]
    out: dict[str, dict[str, float]] = {}
    for method, g in sb.groupby("method", observed=True):
        method = str(method)
        row: dict[str, float] = {}
        for col in metric_cols:
            if col in g.columns and len(g):
                row[f"{col}_std_across_examples"] = float(pd.to_numeric(g[col], errors="coerce").std(ddof=0))
            else:
                row[f"{col}_std_across_examples"] = float("nan")
        out[method] = row
    return out


def run_retrieval_eval(
    *,
    repo_root: Path,
    split: str,
    methods: list[str],
    active_cohort: str,
    max_examples: int,
    support_app_filter_mode: str,
    cohort_sizing: dict[tuple[str, str], float],
    min_review_chars: int,
    max_train_rows_per_user: int,
    multi_max_reviews: int,
    k_final: int,
    k_retrieval: int | None = None,
    k_personalization: int,
    enable_popularity_decile_diagnostics: bool,
    include_random_sanity: bool,
    random_seed: int = PROJECT_RANDOM_SEED,
    artifact_dir: Path | None = None,
    verbose: bool = False,
    examples_parquet: Path | None = None,
) -> EvalTables:
    if not REQUIRED_PHASE1_METHODS.issubset(set(methods)):
        raise ValueError(
            "methods must include required baselines "
            f"{sorted(REQUIRED_PHASE1_METHODS)}; got {sorted(set(methods))}"
        )

    k_r = int(k_final) if k_retrieval is None else int(k_retrieval)
    if k_r < int(k_final):
        raise ValueError(f"k_retrieval ({k_r}) must be >= k_final ({k_final})")

    t0 = time.perf_counter()
    if examples_parquet is not None:
        p = examples_parquet
        if not p.is_file():
            raise FileNotFoundError(f"examples_parquet not found: {p}")
        inputs = prepare_eval_inputs_from_cache(
            repo_root=repo_root,
            split=split,
            min_review_chars=min_review_chars,
            examples_parquet=p,
            artifact_dir=artifact_dir,
            verbose=verbose,
        )
    else:
        inputs = prepare_eval_inputs(
            repo_root=repo_root,
            split=split,
            active_cohort=active_cohort,
            max_examples=max_examples,
            support_app_filter_mode=support_app_filter_mode,
            cohort_sizing=cohort_sizing,
            min_review_chars=min_review_chars,
            max_train_rows_per_user=max_train_rows_per_user,
            random_seed=random_seed,
            artifact_dir=artifact_dir,
            verbose=verbose,
        )
    t_inputs = time.perf_counter()
    rng = np.random.default_rng(int(random_seed))
    registry = _build_method_registry(
        retriever=inputs.retriever,
        X=inputs.embedding_matrix,
        pop_row=inputs.pop_row,
        app_to_row=inputs.app_to_row,
        multi_max_reviews=multi_max_reviews,
        rng=rng,
        mask_query_app=True,
    )
    unknown = sorted(set(methods).difference(registry.keys()))
    if unknown:
        raise ValueError(f"Unknown methods requested: {unknown}. Available={sorted(registry.keys())}")

    run_methods = methods.copy()
    if include_random_sanity and "random" not in run_methods:
        run_methods.append("random")
    selected_registry = {m: registry[m] for m in run_methods}

    method_iter = selected_registry.items()
    if verbose:
        method_iter = tqdm(method_iter, total=len(selected_registry), desc="methods", unit="method")
    retrieval_frames: list[pd.DataFrame] = []
    ranking_frames: list[pd.DataFrame] = []
    artifact_rows_acc: list[dict[str, Any]] = []
    model_snap = _embedding_model_snapshot(inputs.retriever)
    for name, fn in method_iter:
        df_r, df_k, arts = _per_example_retrieval_ranking(
            method_name=name,
            score_fn=fn,
            examples=inputs.examples,
            app_ids=inputs.app_ids,
            k_retrieval=k_r,
            k_final=k_final,
            masking_policy_version=MASKING_POLICY_VERSION,
            model_version=model_snap,
            verbose=verbose,
        )
        retrieval_frames.append(df_r)
        ranking_frames.append(df_k)
        artifact_rows_acc.extend(arts)

    df_ex_retrieval = pd.concat(retrieval_frames, ignore_index=True)
    df_ex_ranking = pd.concat(ranking_frames, ignore_index=True)
    if len(df_ex_ranking) == 0:
        raise RuntimeError("No per-example metrics produced; check split/cohort/method settings.")
    t_metrics = time.perf_counter()

    overall_retrieval = _table_overall_retrieval(df_ex_retrieval)
    overall_ranking = _table_overall_ranking(df_ex_ranking)
    by_slice_retrieval = _table_by_slice_for_metrics(
        df_ex_retrieval, metric_cols=RETRIEVAL_METRIC_COLS, ranking=False
    )
    by_slice_ranking = _table_by_slice_for_metrics(df_ex_ranking, metric_cols=METRIC_COLS, ranking=True)
    by_support_retrieval = _table_by_support_for_metrics(
        df_ex_retrieval, metric_cols=RETRIEVAL_METRIC_COLS, ranking=False
    )
    by_support_ranking = _table_by_support_for_metrics(df_ex_ranking, metric_cols=METRIC_COLS, ranking=True)
    pop_ret, pop_delta_ret, ex_pop_map = _table_popularity(
        df_ex_metrics=df_ex_retrieval,
        examples=inputs.examples,
        app_ids=inputs.app_ids,
        pop_row=inputs.pop_row,
        enable_popularity_decile_diagnostics=enable_popularity_decile_diagnostics,
        metric_cols=RETRIEVAL_METRIC_COLS,
    )
    pop_rank, pop_delta_rank, _ = _table_popularity(
        df_ex_metrics=df_ex_ranking,
        examples=inputs.examples,
        app_ids=inputs.app_ids,
        pop_row=inputs.pop_row,
        enable_popularity_decile_diagnostics=enable_popularity_decile_diagnostics,
        metric_cols=METRIC_COLS,
    )
    personalization = _table_personalization(
        methods=selected_registry,
        examples=inputs.examples,
        X=inputs.embedding_matrix,
        app_ids=inputs.app_ids,
        pop_row=inputs.pop_row,
        k_personalization=k_personalization,
        verbose=verbose,
    )
    ex_meta = df_ex_ranking[["ex_idx", "n_eval_targets", "n_support_train"]].drop_duplicates("ex_idx").copy()
    ex_meta["slice_name"] = np.select(
        [ex_meta["n_eval_targets"] >= 2, ex_meta["n_eval_targets"] == 1, ex_meta["n_eval_targets"] == 0],
        ["slice_a_multi_target", "slice_b_single_target", "slice_c_zero_target"],
        default="slice_other",
    )
    ex_meta["train_support_bucket"] = ex_meta["n_support_train"].fillna(0).astype(int).map(_support_bucket)
    slice_personalization = _personalization_by_group(
        methods=selected_registry,
        examples=inputs.examples,
        X=inputs.embedding_matrix,
        app_ids=inputs.app_ids,
        pop_row=inputs.pop_row,
        k_personalization=k_personalization,
        group_map=ex_meta[["ex_idx", "slice_name"]],
        group_col="slice_name",
        verbose=verbose,
    )
    support_personalization = _personalization_by_group(
        methods=selected_registry,
        examples=inputs.examples,
        X=inputs.embedding_matrix,
        app_ids=inputs.app_ids,
        pop_row=inputs.pop_row,
        k_personalization=k_personalization,
        group_map=ex_meta[["ex_idx", "train_support_bucket"]],
        group_col="train_support_bucket",
        verbose=verbose,
    )
    pop_personalization = _personalization_by_group(
        methods=selected_registry,
        examples=inputs.examples,
        X=inputs.embedding_matrix,
        app_ids=inputs.app_ids,
        pop_row=inputs.pop_row,
        k_personalization=k_personalization,
        group_map=ex_pop_map[["ex_idx", "pos_pop_decile"]] if not ex_pop_map.empty else pd.DataFrame(),
        group_col="pos_pop_decile",
        verbose=verbose,
    )
    # Guardrails (ILD, catalog coverage, novelty, personalization gap) align with method top-k@k_personalization;
    # attach to both ranking and retrieval summaries for comparable reporting.
    overall_retrieval = _append_personalization_metrics(overall_retrieval, personalization, on_keys=["method"])
    by_slice_retrieval = _append_personalization_metrics(
        by_slice_retrieval, slice_personalization, on_keys=["method", "slice_name"]
    )
    by_support_retrieval = _append_personalization_metrics(
        by_support_retrieval, support_personalization, on_keys=["method", "train_support_bucket"]
    )
    pop_ret = _append_personalization_metrics(
        pop_ret, pop_personalization, on_keys=["method", "pos_pop_decile"]
    )
    pop_delta_ret = _append_personalization_metrics(
        pop_delta_ret, pop_personalization, on_keys=["method", "pos_pop_decile"]
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
    t_tables = time.perf_counter()
    coverage = _coverage_from_examples(inputs.examples).iloc[0].to_dict()
    ex_diag = pd.DataFrame(
        {
            "n_eval_targets": [int(ex.get("n_eval_targets", 0)) for ex in inputs.examples],
            "n_support_train": [int(len(ex.get("train_review_rows", []))) for ex in inputs.examples],
        }
    )
    ex_diag["slice_name"] = np.select(
        [ex_diag["n_eval_targets"] >= 2, ex_diag["n_eval_targets"] == 1, ex_diag["n_eval_targets"] == 0],
        ["slice_a_multi_target", "slice_b_single_target", "slice_c_zero_target"],
        default="slice_other",
    )
    ex_diag["train_support_bucket"] = ex_diag["n_support_train"].map(_support_bucket)
    slice_counts = ex_diag["slice_name"].value_counts().sort_index().to_dict()
    support_counts = (
        ex_diag["train_support_bucket"]
        .value_counts()
        .reindex(SUPPORT_BUCKET_ORDER, fill_value=0)
        .to_dict()
    )
    pool_n = int(len(inputs.app_ids))
    retr_bottleneck = _retrieval_bottleneck_summary(
        df_ex_retrieval, k_retrieval=k_r, candidate_pool_size=pool_n
    )
    slice_b_std_retrieval = _slice_b_empirical_std(
        df_ex_retrieval, metric_cols=tuple(RETRIEVAL_METRIC_COLS)
    )
    slice_b_std_ranking = _slice_b_empirical_std(df_ex_ranking, metric_cols=tuple(METRIC_COLS))

    run_meta = {
        "split_requested": split,
        "split_used": inputs.eval_split_name,
        "active_cohort": active_cohort,
        "max_examples": int(max_examples),
        "n_examples_evaluable": int(len(inputs.examples)),
        "methods_requested": methods,
        "methods_run": run_methods,
        "k_final": int(k_final),
        "k_retrieval": int(k_r),
        "k_personalization": int(k_personalization),
        "random_seed": int(random_seed),
        "masking_policy_version": MASKING_POLICY_VERSION,
        "model_version": model_snap,
        "coverage": coverage,
        "prep_diagnostics": inputs.prep_diagnostics,
        "counts_by_slice": {k: int(v) for k, v in slice_counts.items()},
        "counts_by_support_bucket": {k: int(v) for k, v in support_counts.items()},
        "retrieval_bottleneck": retr_bottleneck,
        "slice_b_empirical_std": {
            "retrieval": slice_b_std_retrieval,
            "ranking": slice_b_std_ranking,
            "note": "std across eval examples in slice_b_single_target (not bootstrap CI)",
        },
        "timing_seconds": {
            "prepare_inputs": round(t_inputs - t0, 3),
            "score_methods": round(t_metrics - t_inputs, 3),
            "build_tables": round(t_tables - t_metrics, 3),
            "total": round(t_tables - t0, 3),
        },
    }
    return EvalTables(
        retrieval_overall=overall_retrieval,
        retrieval_by_slice=by_slice_retrieval,
        retrieval_by_support_bucket=by_support_retrieval,
        retrieval_by_pop_decile=pop_ret,
        retrieval_pop_delta_vs_popularity=pop_delta_ret,
        ranking_overall=overall_ranking,
        ranking_by_slice=by_slice_ranking,
        ranking_by_support_bucket=by_support_ranking,
        ranking_by_pop_decile=pop_rank,
        ranking_pop_delta_vs_popularity=pop_delta_rank,
        personalization=personalization,
        artifact_rows=artifact_rows_acc,
        run_meta=run_meta,
    )
