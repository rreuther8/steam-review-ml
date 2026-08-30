"""Stage 5 (Track B): LLM-as-judge scoring of cached Stage 4 explanations.

Scores the already-generated, already-cached explanations from
``recs_job_explanation_heuristic_eval.py`` (same input parquet) using a different,
stronger model (Claude via the Anthropic API) than the local Llama-3.1-8B that
generated them -- avoids a model grading its own family's output.

This is a paid, network-calling job: cost is proportional to ``limit`` (or the
full cached cohort if ``limit`` is unset/null). Use ``--dry-run`` to preview the
first judge prompt for free, and a small ``limit`` for a pilot before scaling up
-- the config ships with ``limit: 10`` for exactly that reason. Judge scores are
cached (``judge_scores_cache``) so a rerun never re-pays the API.

See ``docs/plans/rag_extension_plan.md`` (Stage 5, Track B).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from steam_review_ml.evaluation.llm_judge import (
    DEFAULT_JUDGE_MODEL,
    build_judge_prompt,
    load_anthropic_client,
    score_or_load_judge_scores,
    summarize_judge_scores,
)
from steam_review_ml.utils import load_config


def main() -> None:
    t_start = time.perf_counter()
    parser = argparse.ArgumentParser(description="Run Stage 5 LLM-as-judge eval on cached Stage 4 explanations.")
    parser.add_argument("config", type=str, help="Path to JSON config.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the first example's judge prompt and exit -- no API calls, no cost.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Judge only the first N cached explanations. Overrides config key 'limit'.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    repo_root = Path(__file__).resolve().parents[1]

    def _resolve(rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else repo_root / p

    explanations_cache = _resolve(str(cfg["explanations_cache"]))
    judge_scores_cache = _resolve(str(cfg["judge_scores_cache"]))
    output_dir = _resolve(str(cfg.get("output_dir", "artifacts/recs/explanation_eval/runs/latest")))
    model = str(cfg.get("model", DEFAULT_JUDGE_MODEL))
    limit = args.limit if args.limit is not None else cfg.get("limit")
    verbose = bool(cfg.get("verbose", True))

    if not explanations_cache.is_file():
        raise FileNotFoundError(
            f"{explanations_cache} not found -- run recs_job_explanation_heuristic_eval.py first "
            "(it generates and caches the explanations this job scores)."
        )
    results_df = pd.read_parquet(explanations_cache)

    print("Running Stage 5 LLM-as-judge eval job")
    print(f"explanations_cache={explanations_cache} (n={len(results_df)})")
    print(f"judge_scores_cache={judge_scores_cache}")
    print(f"model={model} limit={limit}")

    if args.dry_run:
        row = results_df.iloc[0]
        prompt = build_judge_prompt(row["query_text"], row["candidate_text"], row["explanation"])
        print("\n--- dry-run: first judge prompt (no API call, no cost) ---\n")
        print(prompt)
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    client = load_anthropic_client(repo_root=repo_root)
    scored_df = score_or_load_judge_scores(
        results_df,
        client,
        cache_path=judge_scores_cache,
        model=model,
        limit=limit,
        verbose=verbose,
    )

    scores_path = output_dir / "explanation_judge_scores.parquet"
    scored_df.to_parquet(scores_path, index=False)
    print(f"Wrote {scores_path}")

    summary = summarize_judge_scores(scored_df)
    summary_path = output_dir / "explanation_judge_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {summary_path}")
    print(json.dumps(summary, indent=2))

    print(f"Total script runtime: {time.perf_counter() - t_start:.2f}s")


if __name__ == "__main__":
    main()
