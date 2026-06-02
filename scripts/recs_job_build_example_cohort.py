"""Build and cache a frozen example cohort for offline eval or ranker training.

Writes ``example_cohort.parquet`` plus summary/meta under ``artifacts/recs/eval_cache/<cache_name>/``.
Use ``split: train`` + ``purpose: ranker_train`` for ranker fit/tune cohorts (must not overlap val eval cache).
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from steam_review_ml.constants import PROJECT_RANDOM_SEED
from steam_review_ml.evaluation.example_cohort import (
    COHORT_PARQUET_NAME,
    assert_cohort_disjoint,
    examples_to_frame,
    resolve_reference_cohort_parquet,
)
from steam_review_ml.evaluation.retrieval_offline_eval import prepare_eval_inputs
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


def main() -> None:
    t_start = time.perf_counter()
    parser = argparse.ArgumentParser(description="Build cached example cohort from JSON config.")
    parser.add_argument("config", type=str, help="Path to JSON config.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    repo_root = Path(__file__).resolve().parents[1]

    split = str(cfg.get("split", "val"))
    purpose = str(cfg.get("purpose", "offline_eval"))
    active_cohort = str(cfg.get("active_cohort", "all"))
    max_examples = int(cfg.get("max_examples", 12_500))
    support_app_filter_mode = str(cfg.get("support_app_filter_mode", "strict"))
    min_review_chars = int(cfg.get("min_review_chars", 30))
    max_train_rows_per_user = int(cfg.get("max_train_rows_per_user", 5))
    random_seed = int(cfg.get("random_seed", PROJECT_RANDOM_SEED))
    artifact_dir = repo_root / str(cfg.get("artifact_dir", "artifacts/recs"))
    cache_root = repo_root / str(cfg.get("output_dir", "artifacts/recs/eval_cache"))
    cache_name = str(cfg.get("cache_name", "example_cohort_v1"))
    output_dir = cache_root / cache_name
    output_dir.mkdir(parents=True, exist_ok=True)

    cohort_sizing = _parse_cohort_sizing(cfg.get("cohort_sizing"))
    disjoint_from_cache = cfg.get("disjoint_from_cache")

    print("Building cached example cohort")
    print(
        f"purpose={purpose} split={split} active_cohort={active_cohort} "
        f"max_examples={max_examples} support_mode={support_app_filter_mode}"
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

    examples_df = examples_to_frame(inputs.examples)
    disjoint_check: dict[str, object] = {"required": bool(disjoint_from_cache), "passed": True}
    if disjoint_from_cache:
        ref_path = resolve_reference_cohort_parquet(cache_root=cache_root, cache_name=str(disjoint_from_cache))
        assert_cohort_disjoint(
            examples_df,
            ref_path,
            context=f"cache_name={cache_name}",
        )
        disjoint_check["reference_cache"] = str(disjoint_from_cache)
        disjoint_check["reference_parquet"] = str(ref_path)
        print(f"Disjoint check passed vs {ref_path}")

    summary = (
        examples_df.groupby(["slice_name", "train_support_bucket"], observed=True)
        .size()
        .rename("n_examples")
        .reset_index()
        .sort_values(["slice_name", "train_support_bucket"])
        .reset_index(drop=True)
    )

    examples_path = output_dir / COHORT_PARQUET_NAME
    summary_path = output_dir / "example_cohort_summary.csv"
    meta_path = output_dir / "example_cohort_meta.json"

    examples_df.to_parquet(examples_path, index=False)
    summary.to_csv(summary_path, index=False)

    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(Path(args.config)),
        "cache_name": cache_name,
        "purpose": purpose,
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
        "disjoint_check": disjoint_check,
        "prep_diagnostics": dict(inputs.prep_diagnostics),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Wrote {examples_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {meta_path}")
    print(f"Total script runtime: {time.perf_counter() - t_start:.2f}s")


if __name__ == "__main__":
    main()
