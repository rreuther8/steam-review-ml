"""Same-split contrastive training/eval example builders (query review → other liked apps in split)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from steam_review_ml.constants import PROJECT_RANDOM_SEED
from steam_review_ml.data.loaders import (
    NORMALIZED_SPLIT_FILENAMES,
    load_normalized_split_df,
    resolve_normalized_split_parquet,
)
from steam_review_ml.recommender.retrieve import ContentRetriever

USER_COL = "author.steamid"
TIME_COL = "timestamp_created"
TRAIN_ROW_OPTIONAL_FIELDS = (
    "_norm_author__playtime_at_review",
    "_norm_author__playtime_last_two_weeks",
    "_norm_author__num_games_owned",
    "_norm_author__num_reviews",
    "_norm_review_word_count",
)


def positive_app_ids_from_example(ex: dict) -> set[int]:
    """Train-facing and eval-facing positive app id keys."""
    raw = ex.get("positive_app_ids")
    if raw is None:
        raw = ex["validation_positive_app_ids"]
    if isinstance(raw, set):
        return {int(app_id) for app_id in raw}
    return {int(app_id) for app_id in list(raw)}


def build_eval_records(df_query_split: pd.DataFrame, df_train_all: pd.DataFrame) -> pd.DataFrame:
    """Attach cohort metadata for query-split positive reviews (eval contract)."""
    val_pos_user_stats = (
        df_query_split.loc[df_query_split["recommended"] == 1]
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
        pd.DataFrame(index=pd.Index(df_query_split[USER_COL].unique(), name=USER_COL))
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
    df_query_pos = df_query_split.loc[df_query_split["recommended"] == 1].copy()
    df_eval_records = df_query_pos.merge(
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


def sample_query_rows_random(
    df_eval_records: pd.DataFrame,
    *,
    max_examples: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if max_examples <= 0 or len(df_eval_records) <= max_examples:
        return df_eval_records.reset_index(drop=True)
    take = rng.choice(df_eval_records.index.to_numpy(), size=max_examples, replace=False)
    return df_eval_records.loc[take].reset_index(drop=True)


def sample_query_rows_by_cohort(
    df_eval_records: pd.DataFrame,
    *,
    active_cohort: str,
    max_examples: int,
    cohort_sizing: dict[tuple[str, str], float],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Eval-job stratified sampling (not used for tower training)."""
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
        return sample_query_rows_random(eval_base, max_examples=max_examples, rng=rng)

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


def build_example_dicts(
    eval_base: pd.DataFrame,
    user_to_query_split_apps: dict[str, list[int]],
    *,
    df_train_all: pd.DataFrame,
    max_train_rows_per_user: int,
    support_app_filter_mode: str,
    rng: np.random.Generator,
    verbose: bool = False,
) -> tuple[list[dict], dict[str, Any]]:
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
    drop_reasons: dict[str, int] = {"no_other_positive_app": 0}
    row_iter = eval_base.iterrows()
    if verbose:
        row_iter = tqdm(row_iter, total=len(eval_base), desc="build contrastive examples", unit="row")
    for _, r in row_iter:
        uid = str(r[USER_COL])
        q_app = int(r["app_id"])
        apps = user_to_query_split_apps.get(uid, [])
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
        ex = {
            "user_id": uid,
            "query_app_id": q_app,
            "query_text": str(r["review"]),
            "query_ts": float(r["ts"]),
            "validation_positive_app_ids": positives,
            "positive_app_ids": positives,
            "n_eval_targets": len(positives),
            "train_review_rows": rows,
            "cohort": str(r["cohort"]),
            "eval_pos_cohort": str(r["eval_pos_cohort"]),
        }
        examples.append(ex)
    diagnostics = {
        "sampled_rows": int(len(eval_base)),
        "evaluable_examples": int(len(examples)),
        "dropped_rows": int(len(eval_base) - len(examples)),
        "drop_reasons": drop_reasons,
    }
    return examples, diagnostics


def build_contrastive_examples(
    *,
    repo_root: Path,
    split: str,
    max_examples: int,
    min_review_chars: int = 30,
    max_train_rows_per_user: int = 5,
    support_app_filter_mode: str = "strict",
    random_seed: int = PROJECT_RANDOM_SEED,
    artifact_dir: Path | None = None,
    verbose: bool = False,
) -> tuple[list[dict], dict[str, Any]]:
    """Build same-split contrastive examples for tower training (random cap, no eval cohort sizing)."""
    retriever = ContentRetriever(artifact_dir=artifact_dir, repo_root=repo_root)
    indexed_apps = set(int(a) for a in retriever.app_ids.tolist())

    processed = repo_root / "data" / "processed"
    query_parquet, split_name = resolve_normalized_split_parquet(processed, split)
    train_parquet = processed / NORMALIZED_SPLIT_FILENAMES["train"]
    if not query_parquet.is_file():
        raise FileNotFoundError(f"Missing split parquet: {query_parquet}")
    if not train_parquet.is_file():
        raise FileNotFoundError(f"Missing train parquet: {train_parquet}")

    df_query = load_normalized_split_df(
        query_parquet,
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
            f"Loaded split rows: query={len(df_query):,} train_support={len(df_train_all):,} "
            f"split_used={split_name}"
        )

    df_records = build_eval_records(df_query, df_train_all)
    user_to_apps = (
        df_records.groupby(USER_COL)["app_id"]
        .apply(lambda s: sorted({int(x) for x in s}))
        .to_dict()
    )
    rng = np.random.default_rng(int(random_seed))
    sampled = sample_query_rows_random(df_records, max_examples=int(max_examples), rng=rng)
    examples, diagnostics = build_example_dicts(
        sampled,
        user_to_query_split_apps={str(k): v for k, v in user_to_apps.items()},
        df_train_all=df_train_all,
        max_train_rows_per_user=max_train_rows_per_user,
        support_app_filter_mode=support_app_filter_mode,
        rng=rng,
        verbose=verbose,
    )
    if len(examples) == 0:
        raise RuntimeError(f"No contrastive examples for split={split!r}.")
    if verbose:
        print(
            f"Built contrastive examples: split={split_name} records={len(df_records):,} "
            f"sampled={len(sampled):,} examples={len(examples):,}"
        )
        print("Drop reasons:", diagnostics.get("drop_reasons", {}))
    diagnostics["split_name"] = split_name
    diagnostics["query_split"] = split
    return examples, diagnostics
