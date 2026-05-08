"""Centralized offline evaluation pipeline for query-embedding retrieval (retrieval phase)."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

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
from steam_review_ml.recommender.retrieve import ContentRetriever

USER_COL = "author.steamid"
TIME_COL = "timestamp_created"
METRIC_COLS = ["Hit@K", "Recall@K", "MAP@K", "NDCG@K", "MRR"]
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
    overall: pd.DataFrame
    by_slice: pd.DataFrame
    by_support_bucket: pd.DataFrame
    by_pop_decile: pd.DataFrame
    pop_delta_vs_popularity: pd.DataFrame
    personalization: pd.DataFrame
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
        positives = {int(a) for a in apps if int(a) != q_app}
        if not positives:
            drop_reasons["no_other_positive_app"] += 1
            continue
        rows = train_rows_by_user.get(uid, [])
        if support_app_filter_mode == "strict":
            exclude_apps = set(positives) | {q_app}
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
                "positives": positives,
                "n_eval_targets": len(positives),
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

    return {
        "raw": score_raw,
        "popularity_train": score_popularity_train,
        "multi_mean_train": score_multi_mean_train,
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


def _per_example_metrics(
    *,
    method_name: str,
    score_fn: Callable[[dict], np.ndarray],
    examples: list[dict],
    app_ids: np.ndarray,
    k_final: int,
    verbose: bool = False,
) -> pd.DataFrame:
    rows: list[dict] = []
    ex_iter = enumerate(examples)
    if verbose:
        ex_iter = tqdm(
            ex_iter,
            total=len(examples),
            desc=f"score {method_name}",
            unit="example",
        )
    for ex_idx, ex in ex_iter:
        positives = ex["positives"]
        if not positives:
            continue
        ranked_rows = _rank_rows(score_fn(ex))
        rows.append(
            {
                "method": method_name,
                "ex_idx": ex_idx,
                "user_id": ex["user_id"],
                "query_app_id": int(ex["query_app_id"]),
                "n_eval_targets": int(ex["n_eval_targets"]),
                "n_support_train": int(len(ex.get("train_review_rows", []))),
                "n_unique_train_apps": int(len({int(r["app_id"]) for r in ex.get("train_review_rows", [])})),
                "Hit@K": hit_rate_at_k(ranked_rows, positives, k_final, app_ids),
                "Recall@K": recall_at_k(ranked_rows, positives, k_final, app_ids),
                "MAP@K": average_precision_at_k(ranked_rows, positives, k_final, app_ids),
                "NDCG@K": ndcg_at_k(ranked_rows, positives, k_final, app_ids),
                "MRR": mrr(ranked_rows, positives, app_ids),
            }
        )
    return pd.DataFrame(rows)


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


def _table_overall(df_ex_metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        df_ex_metrics.groupby("method", observed=True)[METRIC_COLS]
        .mean()
        .reset_index()
        .sort_values(["NDCG@K", "MAP@K", "MRR"], ascending=False)
        .reset_index(drop=True)
    )


def _table_by_slice(df_ex_metrics: pd.DataFrame) -> pd.DataFrame:
    df = df_ex_metrics.copy()
    df["slice_name"] = np.select(
        [df["n_eval_targets"] >= 2, df["n_eval_targets"] == 1, df["n_eval_targets"] == 0],
        ["slice_a_multi_target", "slice_b_single_target", "slice_c_zero_target"],
        default="slice_other",
    )
    return (
        df.groupby(["slice_name", "method"], observed=True)[METRIC_COLS]
        .mean()
        .reset_index()
        .sort_values(["slice_name", "NDCG@K", "Hit@K"], ascending=[True, False, False])
        .reset_index(drop=True)
    )


def _table_by_support_bucket(df_ex_metrics: pd.DataFrame) -> pd.DataFrame:
    df = df_ex_metrics.copy()
    df["train_support_bucket"] = df["n_support_train"].fillna(0).astype(int).map(_support_bucket)
    df["train_support_bucket"] = pd.Categorical(
        df["train_support_bucket"], categories=SUPPORT_BUCKET_ORDER, ordered=True
    )
    return (
        df.groupby(["train_support_bucket", "method"], observed=True)[METRIC_COLS]
        .mean()
        .reset_index()
        .sort_values(
            ["train_support_bucket", "NDCG@K", "MAP@K", "MRR"],
            ascending=[True, False, False, False],
        )
        .reset_index(drop=True)
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
        vals = [app_pop.get(int(a), 0.0) for a in ex["positives"]]
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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not enable_popularity_decile_diagnostics:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    df_ex_pop = _example_popularity_segments(examples=examples, app_ids=app_ids, pop_row=pop_row)
    if df_ex_pop["pos_pop_decile"].dropna().empty:
        return pd.DataFrame(), pd.DataFrame(), df_ex_pop
    df_seg_pop = df_ex_metrics.merge(df_ex_pop, on="ex_idx", how="left")
    pop_table = (
        df_seg_pop.dropna(subset=["pos_pop_decile"])
        .groupby(["pos_pop_decile", "method"], observed=True)[METRIC_COLS]
        .mean()
        .reset_index()
        .sort_values(["pos_pop_decile", "method"])
        .reset_index(drop=True)
    )
    if pop_table.empty:
        return pop_table, pd.DataFrame(), df_ex_pop
    pop_ref = pop_table[pop_table["method"] == "popularity_train"][["pos_pop_decile"] + METRIC_COLS].rename(
        columns={m: f"{m}_pop_ref" for m in METRIC_COLS}
    )
    pop_delta = pop_table.merge(pop_ref, on="pos_pop_decile", how="left")
    for m in METRIC_COLS:
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
    k_personalization: int,
    enable_popularity_decile_diagnostics: bool,
    include_random_sanity: bool,
    random_seed: int = PROJECT_RANDOM_SEED,
    artifact_dir: Path | None = None,
    verbose: bool = False,
) -> EvalTables:
    if not REQUIRED_PHASE1_METHODS.issubset(set(methods)):
        raise ValueError(
            "methods must include required baselines "
            f"{sorted(REQUIRED_PHASE1_METHODS)}; got {sorted(set(methods))}"
        )

    t0 = time.perf_counter()
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
    metric_frames: list[pd.DataFrame] = []
    for name, fn in method_iter:
        metric_frames.append(
            _per_example_metrics(
                method_name=name,
                score_fn=fn,
                examples=inputs.examples,
                app_ids=inputs.app_ids,
                k_final=k_final,
                verbose=verbose,
            )
        )
    df_ex_metrics = pd.concat(metric_frames, ignore_index=True)
    if len(df_ex_metrics) == 0:
        raise RuntimeError("No per-example metrics produced; check split/cohort/method settings.")
    t_metrics = time.perf_counter()

    overall = _table_overall(df_ex_metrics)
    by_slice = _table_by_slice(df_ex_metrics)
    by_support = _table_by_support_bucket(df_ex_metrics)
    pop_table, pop_delta, ex_pop_map = _table_popularity(
        df_ex_metrics=df_ex_metrics,
        examples=inputs.examples,
        app_ids=inputs.app_ids,
        pop_row=inputs.pop_row,
        enable_popularity_decile_diagnostics=enable_popularity_decile_diagnostics,
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
    ex_meta = df_ex_metrics[["ex_idx", "n_eval_targets", "n_support_train"]].drop_duplicates("ex_idx").copy()
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
    overall = _append_personalization_metrics(overall, personalization, on_keys=["method"])
    by_slice = _append_personalization_metrics(by_slice, slice_personalization, on_keys=["method", "slice_name"])
    by_support = _append_personalization_metrics(
        by_support, support_personalization, on_keys=["method", "train_support_bucket"]
    )
    pop_table = _append_personalization_metrics(pop_table, pop_personalization, on_keys=["method", "pos_pop_decile"])
    pop_delta = _append_personalization_metrics(pop_delta, pop_personalization, on_keys=["method", "pos_pop_decile"])
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
    run_meta = {
        "split_requested": split,
        "split_used": inputs.eval_split_name,
        "active_cohort": active_cohort,
        "max_examples": int(max_examples),
        "n_examples_evaluable": int(len(inputs.examples)),
        "methods_requested": methods,
        "methods_run": run_methods,
        "k_final": int(k_final),
        "k_personalization": int(k_personalization),
        "random_seed": int(random_seed),
        "coverage": coverage,
        "prep_diagnostics": inputs.prep_diagnostics,
        "counts_by_slice": {k: int(v) for k, v in slice_counts.items()},
        "counts_by_support_bucket": {k: int(v) for k, v in support_counts.items()},
        "timing_seconds": {
            "prepare_inputs": round(t_inputs - t0, 3),
            "score_methods": round(t_metrics - t_inputs, 3),
            "build_tables": round(t_tables - t_metrics, 3),
            "total": round(t_tables - t0, 3),
        },
    }
    return EvalTables(
        overall=overall,
        by_slice=by_slice,
        by_support_bucket=by_support,
        by_pop_decile=pop_table,
        pop_delta_vs_popularity=pop_delta,
        personalization=personalization,
        run_meta=run_meta,
    )
