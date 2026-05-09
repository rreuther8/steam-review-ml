"""Build and cache eval examples for retrieval experiments.

Creates a static examples artifact from `prepare_eval_inputs(...)` so repeated
notebook/script runs do not rebuild the cohort each time.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from steam_review_ml.constants import PROJECT_RANDOM_SEED
from steam_review_ml.recommender.evaluation import prepare_eval_inputs
from steam_review_ml.utils import load_config


def _parse_cohort_sizing(raw: dict[str, float] | None) -> dict[tuple[str, str], float]:
    if not raw:
        return {}
    out: dict[tuple[str, str], float] = {}
    for key, pct in raw.items():
        if "|" not in key:
            raise ValueError(
                "cohort_sizing keys must be 'eval_pos_cohort|cohort' (e.g. "
                "'val_multi_pos_eval|val_multi_pos_train')."
            )
        left, right = key.split("|", 1)
        out[(left.strip(), right.strip())] = float(pct)
    return out


def _slice_name_from_n_targets(n_eval_targets: int) -> str:
    if n_eval_targets >= 2:
        return "slice_a_multi_target"
    if n_eval_targets == 1:
        return "slice_b_single_target"
    if n_eval_targets == 0:
        return "slice_c_zero_target"
    return "slice_other"


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


def _examples_to_frame(examples: list[dict]) -> pd.DataFrame:
    rows: list[dict] = []
    for ex_idx, ex in enumerate(examples):
        train_rows = ex.get("train_review_rows", [])
        n_support_train = int(len(train_rows))
        n_unique_train_apps = int(len({int(r["app_id"]) for r in train_rows if "app_id" in r}))
        n_eval_targets = int(ex.get("n_eval_targets", 0))
        row = {
            "ex_idx": ex_idx,
            "user_id": str(ex.get("user_id")),
            "query_app_id": int(ex.get("query_app_id")),
            "query_text": str(ex.get("query_text", "")),
            "query_ts": float(ex.get("query_ts", 0.0)),
            "n_eval_targets": n_eval_targets,
            "slice_name": _slice_name_from_n_targets(n_eval_targets),
            "cohort": str(ex.get("cohort", "")),
            "eval_pos_cohort": str(ex.get("eval_pos_cohort", "")),
            "n_support_train": n_support_train,
            "n_unique_train_apps": n_unique_train_apps,
            "train_support_bucket": _support_bucket(n_support_train),
            "validation_positive_app_ids_json": json.dumps(
                sorted(int(a) for a in ex.get("validation_positive_app_ids", []))
            ),
            "train_review_rows_json": json.dumps(train_rows),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    t_start = time.perf_counter()
    parser = argparse.ArgumentParser(description="Build cached eval examples artifact from config.")
    parser.add_argument("config", type=str, help="Path to JSON config.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    repo_root = Path(__file__).resolve().parents[1]

    split = str(cfg.get("split", "val"))
    active_cohort = str(cfg.get("active_cohort", "all"))
    max_examples = int(cfg.get("max_examples", 12_500))
    support_app_filter_mode = str(cfg.get("support_app_filter_mode", "strict"))
    min_review_chars = int(cfg.get("min_review_chars", 30))
    max_train_rows_per_user = int(cfg.get("max_train_rows_per_user", 5))
    random_seed = int(cfg.get("random_seed", PROJECT_RANDOM_SEED))
    artifact_dir = repo_root / str(cfg.get("artifact_dir", "artifacts/recs"))
    output_dir = repo_root / str(cfg.get("output_dir", "artifacts/recs/eval_cache"))
    cache_name = str(cfg.get("cache_name", "val_dev_cache"))
    output_dir = output_dir / cache_name
    output_dir.mkdir(parents=True, exist_ok=True)

    cohort_sizing = _parse_cohort_sizing(cfg.get("cohort_sizing"))

    print("Building cached eval examples")
    print(
        f"split={split} active_cohort={active_cohort} max_examples={max_examples} "
        f"support_mode={support_app_filter_mode}"
    )
    print(f"artifact_dir={artifact_dir} output_dir={output_dir}")

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
        verbose=bool(cfg.get("verbose", True)),
    )

    examples_df = _examples_to_frame(inputs.examples)
    summary = (
        examples_df.groupby(["slice_name", "train_support_bucket"], observed=True)
        .size()
        .rename("n_examples")
        .reset_index()
        .sort_values(["slice_name", "train_support_bucket"])
        .reset_index(drop=True)
    )

    examples_path = output_dir / "eval_examples.parquet"
    summary_path = output_dir / "eval_examples_summary.csv"
    meta_path = output_dir / "eval_examples_meta.json"

    examples_df.to_parquet(examples_path, index=False)
    summary.to_csv(summary_path, index=False)

    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(Path(args.config)),
        "cache_name": cache_name,
        "split_requested": split,
        "split_used": inputs.eval_split_name,
        "active_cohort": active_cohort,
        "max_examples": max_examples,
        "support_app_filter_mode": support_app_filter_mode,
        "min_review_chars": min_review_chars,
        "max_train_rows_per_user": max_train_rows_per_user,
        "random_seed": random_seed,
        "artifact_dir": str(artifact_dir),
        "output_dir": str(output_dir),
        "n_examples": int(len(examples_df)),
        "n_users": int(examples_df["user_id"].nunique()),
        "counts_by_slice": {
            k: int(v) for k, v in examples_df["slice_name"].value_counts().sort_index().to_dict().items()
        },
        "prep_diagnostics": dict(inputs.prep_diagnostics),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Wrote {examples_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {meta_path}")
    print(f"Total script runtime: {time.perf_counter() - t_start:.2f}s")


if __name__ == "__main__":
    main()

