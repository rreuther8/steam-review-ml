"""Subsample an existing frozen eval cohort down to a smaller frozen cohort.

Used for Stage 4 (LLM reranker spike): running an LLM over the full ~12k-example
`val_dev_12k_v1` cohort is too slow/expensive to iterate on, so this draws a
deterministic, seeded subset of it (same examples, not a fresh independent sample)
and freezes that subset as its own named cohort under artifacts/recs/eval_cache/.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from steam_review_ml.evaluation.example_cohort import (
    SubsampleEvalCohortConfig,
    cohort_parquet_path,
)
from steam_review_ml.utils import load_config


def subsample_eval_cohort(config: SubsampleEvalCohortConfig) -> dict[str, Any]:
    """Draw a deterministic subset of an existing frozen cohort and freeze it under a new cache_name.

    Same examples as the source cohort (not a fresh independent sample) -- keeps the
    small cohort a true subset of whatever population other rankers were measured against.
    Returns a metadata dict (also written alongside the subset parquet) with paths and counts.
    """
    source_path = cohort_parquet_path(config.cache_root / config.source_cache_name)
    source_df = pd.read_parquet(source_path)
    if config.n > len(source_df):
        raise ValueError(f"Requested n={config.n} exceeds source cohort size {len(source_df)}")

    subset_df = (
        source_df.sample(n=config.n, random_state=config.random_seed).sort_index().reset_index(drop=True)
    )
    subset_df["ex_idx"] = subset_df.index

    config.output_dir.mkdir(parents=True, exist_ok=True)
    subset_df.to_parquet(config.output_path, index=False)

    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cache_name": config.cache_name,
        "source_cache_name": config.source_cache_name,
        "source_parquet": str(source_path),
        "n": config.n,
        "random_seed": config.random_seed,
        "n_examples": int(len(subset_df)),
        "n_users": int(subset_df["user_id"].nunique()),
        "counts_by_slice": {
            k: int(v) for k, v in subset_df["slice_name"].value_counts().sort_index().to_dict().items()
        },
        "output_path": str(config.output_path),
    }
    config.meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    meta["meta_path"] = str(config.meta_path)

    return meta


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Subsample a frozen eval cohort parquet into a smaller frozen cohort."
    )
    parser.add_argument("config", type=str, help="Path to JSON config.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    repo_root = Path(__file__).resolve().parents[1]
    config = SubsampleEvalCohortConfig.from_job_config(cfg, repo_root=repo_root)

    print(f"Subsampling {config.source_cache_name} -> {config.cache_name} (n={config.n}, seed={config.random_seed})")
    meta = subsample_eval_cohort(config)

    print(f"Wrote {meta['output_path']} ({meta['n_examples']} examples)")
    print(f"Wrote {meta['meta_path']}")


if __name__ == "__main__":
    main()
