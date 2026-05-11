"""Migrate existing artifacts/recs files into a clearer layout.

Usage:
  python scripts/recs_migrate_artifacts_layout.py --dry-run
  python scripts/recs_migrate_artifacts_layout.py --apply
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _plan_moves(root: Path) -> list[tuple[Path, Path]]:
    moves: list[tuple[Path, Path]] = []

    # Evaluation summaries and experiment outputs
    experiments = {
        "eval_two_stage_summary.csv": "experiments/two_stage/eval_two_stage_summary.csv",
        "eval_history_blend_gridsearch_at5.csv": "experiments/history_blend/eval_history_blend_gridsearch_at5.csv",
        "eval_review_style_4way_proxy_baseline_raw_raw.json": "experiments/review_style/4way_proxy/eval_review_style_4way_proxy_baseline_raw_raw.json",
        "eval_review_style_4way_proxy_metrics.csv": "experiments/review_style/4way_proxy/eval_review_style_4way_proxy_metrics.csv",
        "eval_review_style_4way_rows.jsonl": "experiments/review_style/4way_proxy/eval_review_style_4way_rows.jsonl",
        "eval_review_style_4way_summary.csv": "experiments/review_style/4way_proxy/eval_review_style_4way_summary.csv",
        "eval_review_style_ab_rows.jsonl": "experiments/review_style/ab/eval_review_style_ab_rows.jsonl",
        "eval_review_style_ab_summary.csv": "experiments/review_style/ab/eval_review_style_ab_summary.csv",
    }
    for name, rel_dst in experiments.items():
        src = root / name
        if src.exists():
            moves.append((src, root / rel_dst))

    # Query datasets
    query_sets = {
        "eval_queries.jsonl": "datasets/eval_queries/default/eval_queries.jsonl",
        "eval_queries_review_style.jsonl": "datasets/eval_queries/review_style/eval_queries_review_style.jsonl",
    }
    for name, rel_dst in query_sets.items():
        src = root / name
        if src.exists():
            moves.append((src, root / rel_dst))

    # Embedding artifacts
    embeddings_default = {
        "game_profile_embedding_meta.json": "embeddings/game_profile/default/game_profile_embedding_meta.json",
        "game_profile_embedding_index.parquet": "embeddings/game_profile/default/game_profile_embedding_index.parquet",
        "game_profile_embeddings.npz": "embeddings/game_profile/default/game_profile_embeddings.npz",
    }
    for name, rel_dst in embeddings_default.items():
        src = root / name
        if src.exists():
            moves.append((src, root / rel_dst))

    embeddings_structured = {
        "game_profile_embedding_meta_structured_eval.json": "embeddings/game_profile/structured_eval/game_profile_embedding_meta_structured_eval.json",
        "game_profile_embedding_index_structured_eval.parquet": "embeddings/game_profile/structured_eval/game_profile_embedding_index_structured_eval.parquet",
        "game_profile_embeddings_structured_eval.npz": "embeddings/game_profile/structured_eval/game_profile_embeddings_structured_eval.npz",
    }
    for name, rel_dst in embeddings_structured.items():
        src = root / name
        if src.exists():
            moves.append((src, root / rel_dst))

    passthrough = {
        "game_profile_reviews.parquet": "embeddings/game_profile/default/game_profile_reviews.parquet",
        "user_facing_qual_eval_sheet.csv": "qualitative/user_facing/user_facing_qual_eval_sheet.csv",
        "user_facing_qual_eval_summary.csv": "qualitative/user_facing/user_facing_qual_eval_summary.csv",
        "active_retrieval_config.json": "retrieval/configs/active_retrieval_config.json",
    }
    for name, rel_dst in passthrough.items():
        src = root / name
        if src.exists():
            moves.append((src, root / rel_dst))

    # Legacy eval directory -> offline_eval/runs/legacy_snapshot
    legacy_eval_dir = root / "eval"
    if legacy_eval_dir.exists() and legacy_eval_dir.is_dir():
        moves.append((legacy_eval_dir, root / "offline_eval" / "runs" / "legacy_snapshot"))

    # retrieval/runs -> offline_eval/runs (rename umbrella for offline retrieval+ranking eval)
    old_runs_root = root / "retrieval" / "runs"
    new_runs_root = root / "offline_eval" / "runs"
    if old_runs_root.exists() and old_runs_root.is_dir() and not new_runs_root.exists():
        moves.append((old_runs_root, new_runs_root))

    return moves


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate artifacts/recs to a cleaner layout.")
    parser.add_argument("--apply", action="store_true", help="Apply move operations.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned operations only.")
    args = parser.parse_args()

    if args.apply and args.dry_run:
        raise ValueError("Use either --apply or --dry-run, not both.")

    repo_root = Path(__file__).resolve().parents[1]
    root = repo_root / "artifacts" / "recs"
    moves = _plan_moves(root)

    if not moves:
        print("No matching files found to migrate.")
        return

    print(f"Planned operations: {len(moves)}")
    for src, dst in moves:
        print(f"- {src} -> {dst}")

    if not args.apply:
        print("\nDry-run mode. Re-run with --apply to execute.")
        return

    for src, dst in moves:
        dst.parent.mkdir(parents=True, exist_ok=True)
        src.rename(dst)
    print("\nMigration complete.")


if __name__ == "__main__":
    main()
