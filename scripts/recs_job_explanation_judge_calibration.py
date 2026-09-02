"""Stage 5 Track B calibration: sample a small cohort for hand-labeling, then compare
those hand labels against the LLM judge and the cheap heuristic proxies.

Two-step, human-in-the-loop process -- there is no single "run" of this job:

    python scripts/recs_job_explanation_judge_calibration.py configs/recs_job_explanation_judge_calibration.json --sample
    # -> fill in human_faithfulness/human_relevance (1-5 ints) in the written CSV by hand
    python scripts/recs_job_explanation_judge_eval.py configs/recs_job_explanation_judge_eval.json  # if not already run
    python scripts/recs_job_explanation_judge_calibration.py configs/recs_job_explanation_judge_calibration.json --compare

``--sample`` bounds its pool to whatever the judge has already scored (if
``judge_scores_path`` exists), so the sample is always comparable. If the judge
hasn't run yet, it bounds the pool to ``n_sample`` itself and prints the
``--limit`` the judge job needs to cover it -- run the sample step first (free),
then size the paid judge run to match.

If ``sample_path`` already exists, ``--sample`` tops it up rather than overwriting it:
existing rows (and any hand-written labels) are left untouched, and only enough new rows
to reach ``n_sample`` total are appended. To grow a sample, raise ``n_sample`` in the
config, make sure the judge has scored at least that many, and rerun ``--sample``.
Pass ``--force`` to overwrite the existing file with a fresh sample instead.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from steam_review_ml.constants import PROJECT_RANDOM_SEED
from steam_review_ml.evaluation.judge_calibration import (
    build_calibration_comparison,
    load_hand_labels,
    sample_for_hand_labeling,
    summarize_calibration,
)
from steam_review_ml.utils import load_config


def _run_sample(cfg: dict, *, resolve, force: bool) -> None:
    explanations_cache = resolve(str(cfg["explanations_cache"]))
    judge_scores_path = resolve(str(cfg["judge_scores_path"]))
    sample_path = resolve(str(cfg["sample_path"]))
    n_sample = int(cfg.get("n_sample", 25))
    seed = int(cfg.get("seed") or PROJECT_RANDOM_SEED)
    igdb_enriched_path = cfg.get("igdb_enriched_path")
    igdb_enriched_path = str(resolve(str(igdb_enriched_path))) if igdb_enriched_path else None

    if not explanations_cache.is_file():
        raise FileNotFoundError(
            f"{explanations_cache} not found -- run recs_job_explanation_heuristic_eval.py first."
        )

    if judge_scores_path.is_file():
        max_example_id = len(pd.read_parquet(judge_scores_path))
        print(f"Judge scores already cover the first {max_example_id} cached explanations; sampling only from those.")
    else:
        max_example_id = n_sample
        print(
            f"No judge scores yet at {judge_scores_path}. Sampling from the first {max_example_id} "
            f"cached explanations -- run recs_job_explanation_judge_eval.py with --limit >= {max_example_id} "
            "before --compare."
        )

    explanations_df = pd.read_parquet(explanations_cache).head(max_example_id)

    if sample_path.is_file() and not force:
        existing_df = pd.read_csv(sample_path)
        n_new = n_sample - len(existing_df)
        if n_new <= 0:
            print(
                f"{sample_path} already has {len(existing_df)} row(s) (target n_sample={n_sample}) -- "
                "nothing to add. Raise n_sample in the config to grow it, or pass --force to overwrite."
            )
            return
        new_rows_df = sample_for_hand_labeling(
            explanations_df,
            n=n_new,
            seed=seed,
            enriched_path=igdb_enriched_path,
            exclude_example_ids=existing_df["example_id"].tolist(),
        )
        if new_rows_df.empty:
            print(f"No unsampled rows left in the current pool (bounded to {max_example_id}) to add.")
            return
        sample_df = pd.concat([existing_df, new_rows_df], ignore_index=True).sort_values("example_id")
        print(
            f"Appending {len(new_rows_df)} new row(s) to {sample_path} "
            f"(existing {len(existing_df)} row(s), and any labels on them, left untouched)."
        )
    else:
        if sample_path.is_file() and force:
            print(f"{sample_path} exists -- --force set, overwriting.")
        sample_df = sample_for_hand_labeling(explanations_df, n=n_sample, seed=seed, enriched_path=igdb_enriched_path)

    sample_path.parent.mkdir(parents=True, exist_ok=True)
    sample_df.to_csv(sample_path, index=False)
    print(f"Wrote {len(sample_df)}-row calibration sample to {sample_path}")
    print("Fill in human_faithfulness/human_relevance (1-5 ints) for every row, then rerun with --compare.")


def _run_compare(cfg: dict, *, resolve) -> None:
    sample_path = resolve(str(cfg["sample_path"]))
    heuristic_scores_path = resolve(str(cfg["heuristic_scores_path"]))
    judge_scores_path = resolve(str(cfg["judge_scores_path"]))
    summary_path = resolve(str(cfg["summary_path"]))

    for path, hint in (
        (sample_path, "run this script with --sample first"),
        (heuristic_scores_path, "run recs_job_explanation_heuristic_eval.py first"),
        (judge_scores_path, "run recs_job_explanation_judge_eval.py first"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found -- {hint}")

    hand_labels_df = load_hand_labels(sample_path)
    heuristic_scores_df = pd.read_parquet(heuristic_scores_path)
    judge_scores_df = pd.read_parquet(judge_scores_path)

    comparison_df = build_calibration_comparison(hand_labels_df, heuristic_scores_df, judge_scores_df)
    summary = summarize_calibration(comparison_df)

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {summary_path}")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample a cohort for hand-labeling, or compare filled-in labels against judge/heuristic scores."
    )
    parser.add_argument("config", type=str, help="Path to JSON config.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--sample", action="store_true", help="Write a CSV sample for hand-labeling.")
    mode.add_argument(
        "--compare", action="store_true", help="Compare a filled-in CSV against judge/heuristic scores."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="With --sample: overwrite an existing sample CSV instead of appending to it "
        "(destroys existing hand-written labels). Ignored with --compare.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    repo_root = Path(__file__).resolve().parents[1]

    def _resolve(rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else repo_root / p

    if args.sample:
        _run_sample(cfg, resolve=_resolve, force=args.force)
    else:
        _run_compare(cfg, resolve=_resolve)


if __name__ == "__main__":
    main()
