"""Run full offline evaluation from JSON config (retrieval + ranking contracts).

Entry point: ``recs_job_eval_offline`` (formerly ``recs_job_eval_retrieval``).
Re-scores every method in config, then writes paired retrieval- and ranking-contract
tables, per-example ``eval_offline_examples.jsonl`` (frozen pools), and run metadata
under the configured directory (default: ``artifacts/recs/offline_eval/runs/latest``).

For rank-only eval on frozen pools, use ``recs_job_eval_ranking.py`` instead.

The config ``methods`` list must include **three** required baselines (contract + regression):

- ``raw``
- ``popularity_train``
- ``multi_mean_train``

You may append additional registered scorer names (wired in ``steam_review_ml.evaluation.retrieval_offline_eval``).
The default JSON config also runs **Candidate C** as ``fusion_c_raw_plus_behavior``: a
hand-fused query vector (session USE embed + playtime-weighted train catalog blend) implemented
in ``steam_review_ml.recommender.retrieve.fusion_c_raw_plus_behavior_query_vector`` — same game
matrix as ``raw``, not a separately trained two-tower model.

Optional non-gating sanity baseline:

- ``random``
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

from steam_review_ml.constants import PROJECT_RANDOM_SEED
from steam_review_ml.evaluation.eval_baseline import write_offline_baseline_dual
from steam_review_ml.evaluation.retrieval_offline_eval import REQUIRED_PHASE1_METHODS, run_retrieval_eval
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


def _archive_run(
    *,
    latest_dir: Path,
    run_utc: str,
    run_tag: str,
    output_root: Path,
) -> Path:
    ts = run_utc.replace("-", "").replace(":", "").replace("T", "_").split(".", 1)[0].replace("+0000", "Z")
    safe_tag = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in run_tag.strip()) or "manual"
    archive_dir = output_root / f"{ts}__{safe_tag}"
    if archive_dir.exists():
        raise FileExistsError(f"Archive directory already exists: {archive_dir}")
    shutil.copytree(latest_dir, archive_dir)
    return archive_dir


def main() -> None:
    t_start = time.perf_counter()
    parser = argparse.ArgumentParser(
        description="Run centralized offline retrieval+ranking eval and write artifact tables."
    )
    parser.add_argument("config", type=str, help="Path to JSON config.")
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="Freeze baseline JSON (ranking + retrieval overall snapshots) alongside summary CSVs.",
    )
    parser.add_argument(
        "--examples-parquet",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "If set, load example cohort from this parquet (from recs_job_build_example_cohort) "
            "instead of resampling prepare_eval_inputs. Overrides config key examples_parquet."
        ),
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    repo_root = Path(__file__).resolve().parents[1]

    split = str(cfg.get("split", "val"))
    methods = [str(m) for m in cfg.get("methods", [])]
    if not methods:
        raise ValueError("Config must include non-empty 'methods' list.")

    if not REQUIRED_PHASE1_METHODS.issubset(set(methods)):
        raise ValueError(
            f"'methods' must include required baselines {sorted(REQUIRED_PHASE1_METHODS)}; "
            f"got {sorted(set(methods))}"
        )

    active_cohort = str(cfg.get("active_cohort", "all"))
    max_examples = int(cfg.get("max_examples", 12_500))
    support_app_filter_mode = str(cfg.get("support_app_filter_mode", "strict"))
    min_review_chars = int(cfg.get("min_review_chars", 30))
    max_train_rows_per_user = int(cfg.get("max_train_rows_per_user", 5))
    multi_max_reviews = int(cfg.get("multi_max_reviews", 5))
    k_final = int(cfg.get("k_final", 10))
    k_retrieval = cfg.get("k_retrieval")
    k_retrieval_arg = None if k_retrieval is None else int(k_retrieval)
    k_personalization = int(cfg.get("k_personalization", 10))
    include_random_sanity = bool(cfg.get("include_random_sanity", False))
    verbose = bool(cfg.get("verbose", True))
    enable_popularity_decile_diagnostics = bool(
        cfg.get("enable_popularity_decile_diagnostics", True)
    )
    random_seed = int(cfg.get("random_seed", PROJECT_RANDOM_SEED))

    artifact_dir_rel = cfg.get("artifact_dir", "artifacts/recs")
    artifact_dir = repo_root / str(artifact_dir_rel)
    output_dir_rel = cfg.get("output_dir", "artifacts/recs/offline_eval/runs/latest")
    output_dir = repo_root / str(output_dir_rel)
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_run = bool(cfg.get("archive_run", False))
    run_tag = str(cfg.get("run_tag", "eval"))

    cohort_sizing = _parse_cohort_sizing(cfg.get("cohort_sizing"))

    examples_parquet: Path | None = None
    if args.examples_parquet:
        p = Path(args.examples_parquet)
        examples_parquet = p if p.is_absolute() else repo_root / p
    elif cfg.get("examples_parquet"):
        p = Path(str(cfg["examples_parquet"]).strip())
        examples_parquet = p if p.is_absolute() else repo_root / p

    print("Running offline eval job")
    kr_display = k_retrieval_arg if k_retrieval_arg is not None else k_final
    print(
        f"split={split} methods={methods} max_examples={max_examples} "
        f"k_retrieval={kr_display} k_final={k_final} k_personalization={k_personalization}"
    )
    print(
        f"active_cohort={active_cohort} support_mode={support_app_filter_mode} "
        f"artifact_dir={artifact_dir} output_dir={output_dir}"
    )
    if examples_parquet is not None:
        print(f"examples_parquet={examples_parquet} (cached cohort; max_examples/cohort_sizing unused for sampling)")

    two_tower_model_path: Path | None = None
    if cfg.get("two_tower_model_path"):
        p = Path(str(cfg["two_tower_model_path"]).strip())
        two_tower_model_path = p if p.is_absolute() else repo_root / p
    two_tower_catalog_item_batch = int(cfg.get("two_tower_catalog_item_batch", 256))
    if two_tower_model_path is not None:
        print(f"two_tower_model_path={two_tower_model_path}")

    rag_chroma_persist_dir: Path | None = None
    if cfg.get("rag_chroma_persist_dir"):
        p = Path(str(cfg["rag_chroma_persist_dir"]).strip())
        rag_chroma_persist_dir = p if p.is_absolute() else repo_root / p
    rag_variant = str(cfg.get("rag_variant", "any_polarity__flat"))
    rag_query_blend_weight = float(cfg.get("rag_query_blend_weight", 0.5))
    if rag_chroma_persist_dir is not None:
        print(
            f"rag_chroma_persist_dir={rag_chroma_persist_dir} rag_variant={rag_variant} "
            f"rag_query_blend_weight={rag_query_blend_weight}"
        )

    tables = run_retrieval_eval(
        repo_root=repo_root,
        split=split,
        methods=methods,
        active_cohort=active_cohort,
        max_examples=max_examples,
        support_app_filter_mode=support_app_filter_mode,
        cohort_sizing=cohort_sizing,
        min_review_chars=min_review_chars,
        max_train_rows_per_user=max_train_rows_per_user,
        multi_max_reviews=multi_max_reviews,
        k_final=k_final,
        k_retrieval=k_retrieval_arg,
        k_personalization=k_personalization,
        enable_popularity_decile_diagnostics=enable_popularity_decile_diagnostics,
        include_random_sanity=include_random_sanity,
        random_seed=random_seed,
        artifact_dir=artifact_dir,
        verbose=verbose,
        examples_parquet=examples_parquet,
        two_tower_model_path=two_tower_model_path,
        two_tower_catalog_item_batch=two_tower_catalog_item_batch,
        rag_chroma_persist_dir=rag_chroma_persist_dir,
        rag_variant=rag_variant,
        rag_query_blend_weight=rag_query_blend_weight,
    )

    retr_overall_path = output_dir / "eval_retrieval_overall.csv"
    retr_slice_path = output_dir / "eval_retrieval_by_slice.csv"
    retr_support_path = output_dir / "eval_retrieval_by_support_bucket.csv"
    retr_pop_path = output_dir / "eval_retrieval_by_pop_decile.csv"
    retr_pop_delta_path = output_dir / "eval_retrieval_pop_delta_vs_popularity.csv"

    rank_overall_path = output_dir / "eval_ranking_overall.csv"
    rank_slice_path = output_dir / "eval_ranking_by_slice.csv"
    rank_support_path = output_dir / "eval_ranking_by_support_bucket.csv"
    rank_pop_path = output_dir / "eval_ranking_by_pop_decile.csv"
    rank_pop_delta_path = output_dir / "eval_ranking_pop_delta_vs_popularity.csv"
    rank_person_path = output_dir / "eval_ranking_personalization.csv"

    examples_path = output_dir / "eval_offline_examples.jsonl"
    meta_path = output_dir / "eval_offline_run_meta.json"
    baseline_path = output_dir / "eval_retrieval_baseline_overall.json"

    tables.retrieval_overall.to_csv(retr_overall_path, index=False)
    tables.retrieval_by_slice.to_csv(retr_slice_path, index=False)
    tables.retrieval_by_support_bucket.to_csv(retr_support_path, index=False)
    tables.retrieval_by_pop_decile.to_csv(retr_pop_path, index=False)
    tables.retrieval_pop_delta_vs_popularity.to_csv(retr_pop_delta_path, index=False)

    tables.ranking_overall.to_csv(rank_overall_path, index=False)
    tables.ranking_by_slice.to_csv(rank_slice_path, index=False)
    tables.ranking_by_support_bucket.to_csv(rank_support_path, index=False)
    tables.ranking_by_pop_decile.to_csv(rank_pop_path, index=False)
    tables.ranking_pop_delta_vs_popularity.to_csv(rank_pop_delta_path, index=False)
    tables.personalization.to_csv(rank_person_path, index=False)

    with examples_path.open("w", encoding="utf-8") as f_out:
        for row in tables.artifact_rows:
            f_out.write(json.dumps(row, ensure_ascii=False) + "\n")

    run_utc = datetime.now(timezone.utc).isoformat()
    run_meta = dict(tables.run_meta)
    run_meta.update(
        {
            "run_utc": run_utc,
            "config_path": str(Path(args.config)),
            "output_dir": str(output_dir),
            "archive_run": archive_run,
            "run_tag": run_tag,
        }
    )
    meta_path.write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    print(f"Wrote {examples_path}")
    print(f"Wrote {meta_path}")

    paths_msg = (
        retr_overall_path,
        retr_slice_path,
        retr_support_path,
        retr_pop_path,
        retr_pop_delta_path,
        rank_overall_path,
        rank_slice_path,
        rank_support_path,
        rank_pop_path,
        rank_pop_delta_path,
        rank_person_path,
    )
    for p in paths_msg:
        print(f"Wrote {p}")

    if args.write_baseline:
        write_offline_baseline_dual(
            ranking_overall=tables.ranking_overall,
            retrieval_overall=tables.retrieval_overall,
            baseline_path=baseline_path,
        )
        print(f"Wrote {baseline_path}")
    if archive_run:
        output_root = output_dir.parent
        archive_dir = _archive_run(
            latest_dir=output_dir,
            run_utc=run_utc,
            run_tag=run_tag,
            output_root=output_root,
        )
        print(f"Archived run snapshot: {archive_dir}")
    elapsed = time.perf_counter() - t_start
    print(f"Total script runtime: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
