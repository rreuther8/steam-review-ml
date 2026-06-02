"""Offline ranking eval on frozen retrieval pools (read-only jsonl; no retriever re-score).

Writes ``eval_ranking_*`` under ``runs/latest_ranking`` by default.
``--write-baseline`` merges ranking snapshots into the dual baseline JSON
(shared with ``recs_job_eval_retrieval.py``).
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from steam_review_ml.evaluation.eval_baseline import merge_ranking_into_baseline
from steam_review_ml.evaluation.ranking_offline_eval import run_ranking_eval
from steam_review_ml.utils import load_config


def main() -> None:
    t_start = time.perf_counter()
    parser = argparse.ArgumentParser(description="Run offline ranking eval on frozen retrieval pools.")
    parser.add_argument("config", type=str, help="Path to JSON config.")
    parser.add_argument(
        "--pools-jsonl",
        type=str,
        default=None,
        metavar="PATH",
        help="Frozen pools jsonl (overrides config pools_jsonl). Read-only.",
    )
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="Merge ranking overall into dual baseline JSON (retrieval section unchanged).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    repo_root = Path(__file__).resolve().parents[1]

    pools_jsonl = Path(args.pools_jsonl) if args.pools_jsonl else Path(str(cfg["pools_jsonl"]))
    if not pools_jsonl.is_absolute():
        pools_jsonl = repo_root / pools_jsonl

    examples_parquet = Path(str(cfg["examples_parquet"]))
    if not examples_parquet.is_absolute():
        examples_parquet = repo_root / examples_parquet

    pool_methods = [str(m) for m in cfg.get("pool_methods", [])]
    ranker_methods = [str(m) for m in cfg.get("ranker_methods", [])]
    catalog_methods = [str(m) for m in cfg.get("catalog_methods", ["popularity_train"])]
    if not pool_methods:
        raise ValueError("Config must include non-empty 'pool_methods'.")
    if not ranker_methods:
        raise ValueError("Config must include non-empty 'ranker_methods'.")

    k_final = int(cfg.get("k_final", 10))
    k_retrieval = int(cfg.get("k_retrieval", 100))
    k_personalization = int(cfg.get("k_personalization", 10))
    min_review_chars = int(cfg.get("min_review_chars", 30))
    enable_pop = bool(cfg.get("enable_popularity_decile_diagnostics", True))
    include_personalization = bool(cfg.get("include_personalization", True))
    verbose = bool(cfg.get("verbose", True))

    artifact_dir = repo_root / str(cfg.get("artifact_dir", "artifacts/recs"))
    output_dir = repo_root / str(cfg.get("output_dir", "artifacts/recs/offline_eval/runs/latest_ranking"))
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_rel = cfg.get(
        "baseline_path", "artifacts/recs/offline_eval/runs/latest/eval_retrieval_baseline_overall.json"
    )
    baseline_path = Path(str(baseline_rel))
    if not baseline_path.is_absolute():
        baseline_path = repo_root / baseline_path

    print("Running offline ranking eval job")
    print(f"pools_jsonl={pools_jsonl}")
    print(f"examples_parquet={examples_parquet}")
    print(
        f"pool_methods={pool_methods} ranker_methods={ranker_methods} "
        f"catalog_methods={catalog_methods} k_final={k_final} k_retrieval={k_retrieval}"
    )
    print(f"output_dir={output_dir}")

    tables = run_ranking_eval(
        repo_root=repo_root,
        pools_jsonl=pools_jsonl,
        pool_methods=pool_methods,
        ranker_methods=ranker_methods,
        catalog_methods=catalog_methods,
        examples_parquet=examples_parquet,
        k_final=k_final,
        k_retrieval=k_retrieval,
        k_personalization=k_personalization,
        min_review_chars=min_review_chars,
        enable_popularity_decile_diagnostics=enable_pop,
        include_personalization=include_personalization,
        artifact_dir=artifact_dir,
        verbose=verbose,
    )

    paths = {
        "eval_ranking_overall.csv": tables.ranking_overall,
        "eval_ranking_by_slice.csv": tables.ranking_by_slice,
        "eval_ranking_by_support_bucket.csv": tables.ranking_by_support_bucket,
        "eval_ranking_by_pop_decile.csv": tables.ranking_by_pop_decile,
        "eval_ranking_pop_delta_vs_popularity.csv": tables.ranking_pop_delta_vs_popularity,
        "eval_ranking_personalization.csv": tables.personalization,
    }
    for name, df in paths.items():
        p = output_dir / name
        df.to_csv(p, index=False)
        print(f"Wrote {p}")

    meta_path = output_dir / "eval_ranking_run_meta.json"
    run_meta = dict(tables.run_meta)
    run_meta.update(
        {
            "run_utc": datetime.now(timezone.utc).isoformat(),
            "config_path": str(Path(args.config)),
            "output_dir": str(output_dir),
            "baseline_path": str(baseline_path),
        }
    )
    meta_path.write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
    print(f"Wrote {meta_path}")

    if args.write_baseline:
        merge_ranking_into_baseline(ranking_overall=tables.ranking_overall, baseline_path=baseline_path)
        print(f"Merged ranking baseline into {baseline_path}")

    print(f"Total script runtime: {time.perf_counter() - t_start:.2f}s")


if __name__ == "__main__":
    main()
