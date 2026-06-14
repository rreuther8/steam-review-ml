"""Build, validate, and load frozen example cohorts for eval or ranker training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

COHORT_PARQUET_NAME = "example_cohort.parquet"
LEGACY_COHORT_PARQUET_NAME = "eval_examples.parquet"
COHORT_IDENTITY_COLS = ("user_id", "query_app_id", "query_ts")


def cohort_parquet_path(cache_dir: Path) -> Path:
    """Prefer ``example_cohort.parquet``; fall back to legacy ``eval_examples.parquet``."""
    cache_dir = Path(cache_dir)
    primary = cache_dir / COHORT_PARQUET_NAME
    if primary.is_file():
        return primary
    legacy = cache_dir / LEGACY_COHORT_PARQUET_NAME
    if legacy.is_file():
        return legacy
    raise FileNotFoundError(
        f"No cohort parquet in {cache_dir} (expected {COHORT_PARQUET_NAME} or {LEGACY_COHORT_PARQUET_NAME})"
    )


def resolve_reference_cohort_parquet(*, cache_root: Path, cache_name: str) -> Path:
    return cohort_parquet_path(cache_root / cache_name)


def assert_cohort_disjoint(
    new_df: pd.DataFrame,
    reference_parquet: Path,
    *,
    context: str = "",
) -> None:
    """Fail if any (user_id, query_app_id, query_ts) appears in both cohorts."""
    ref = pd.read_parquet(reference_parquet)
    for col in COHORT_IDENTITY_COLS:
        if col not in new_df.columns or col not in ref.columns:
            raise ValueError(f"Missing identity column {col!r} for disjoint check")
    left = new_df[list(COHORT_IDENTITY_COLS)].drop_duplicates()
    right = ref[list(COHORT_IDENTITY_COLS)].drop_duplicates()
    overlap = left.merge(right, on=list(COHORT_IDENTITY_COLS), how="inner")
    if not overlap.empty:
        prefix = f"{context}: " if context else ""
        sample = overlap.head(5).to_dict(orient="records")
        raise ValueError(
            f"{prefix}found {len(overlap)} overlapping examples vs {reference_parquet}; "
            f"sample={sample}"
        )


def slice_name_from_n_targets(n_eval_targets: int) -> str:
    if n_eval_targets >= 2:
        return "slice_a_multi_target"
    if n_eval_targets == 1:
        return "slice_b_single_target"
    if n_eval_targets == 0:
        return "slice_c_zero_target"
    return "slice_other"


def support_bucket(n: int) -> str:
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


def examples_to_frame(examples: list[dict]) -> pd.DataFrame:
    rows: list[dict] = []
    for ex_idx, ex in enumerate(examples):
        train_rows = ex.get("train_review_rows", [])
        n_support_train = int(len(train_rows))
        n_unique_train_apps = int(len({int(r["app_id"]) for r in train_rows if "app_id" in r}))
        n_eval_targets = int(ex.get("n_eval_targets", 0))
        rows.append(
            {
                "ex_idx": ex_idx,
                "user_id": str(ex.get("user_id")),
                "query_app_id": int(ex.get("query_app_id")),
                "query_text": str(ex.get("query_text", "")),
                "query_ts": float(ex.get("query_ts", 0.0)),
                "n_eval_targets": n_eval_targets,
                "slice_name": slice_name_from_n_targets(n_eval_targets),
                "cohort": str(ex.get("cohort", "")),
                "eval_pos_cohort": str(ex.get("eval_pos_cohort", "")),
                "n_support_train": n_support_train,
                "n_unique_train_apps": n_unique_train_apps,
                "train_support_bucket": support_bucket(n_support_train),
                "validation_positive_app_ids_json": json.dumps(
                    sorted(int(a) for a in ex.get("validation_positive_app_ids", []))
                ),
                "train_review_rows_json": json.dumps(train_rows),
            }
        )
    return pd.DataFrame(rows)


def load_retrieval_pool_rows(parquet_path: Path) -> list[dict[str, Any]]:
    """Load ranker **train** pools from export parquet (recs_014 ``train_pools``).

    One dict per example from ``recs_job_export_retrieval_pools`` (e.g.
    ``artifacts/recs/ranker_pools/train_ranker_v1/two_tower_v1.parquet``).
    Disjoint train cohort — not the val eval jsonl.

    Required columns include ``retrieved_*_json``, ``validation_positive_app_ids_json``,
    ``slice_name``, ``pool_method``. JSON fields are strings (same as jsonl rows).
    """
    df = pd.read_parquet(parquet_path)
    required = {
        "ex_idx",
        "pool_method",
        "retrieved_app_ids_json",
        "retrieved_scores_json",
        "validation_positive_app_ids_json",
        "n_eval_targets",
        "slice_name",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Pool parquet missing columns: {sorted(missing)}")
    return df.to_dict(orient="records")


def load_retrieval_pools_jsonl(jsonl_path: Path, *, method: str) -> list[dict[str, Any]]:
    """Load ranker **val** pools from offline eval jsonl (recs_014 ``val_pools``).

    Reads ``eval_offline_examples.jsonl`` (multi-method audit file) and keeps
    only rows where ``method`` matches (typically ``two_tower_v1``). Use for
    report-only head-to-head — do not tune hyperparameters on these pools.

    Extra val-only fields (e.g. ``ranked_app_ids_json``) are ignored by pool rerankers.
    """
    pools: list[dict[str, Any]] = []
    with Path(jsonl_path).open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("method") != method:
                continue
            pools.append(row)
    if not pools:
        raise RuntimeError(f"No rows for method={method!r} in {jsonl_path}")
    return pools
