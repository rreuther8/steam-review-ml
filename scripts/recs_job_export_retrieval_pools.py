"""Export frozen retrieval pools for a cached example cohort (ranker train/tune).

Scores one retrieval method (default ``two_tower_v1``) on every row in
``example_cohort.parquet`` and writes a slim pool parquet reusable by multiple rankers.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from steam_review_ml.evaluation.example_cohort import cohort_parquet_path, slice_name_from_n_targets
from steam_review_ml.evaluation.retrieval_offline_eval import (
    METHOD_TWO_TOWER_V1,
    _rank_rows,
    load_eval_examples_from_parquet,
    prepare_eval_inputs_from_cache,
)
from steam_review_ml.recommender.two_tower_score import load_two_tower_model, make_two_tower_score_fn
from steam_review_ml.recommender.two_tower_train import load_hub_settings
from steam_review_ml.utils import load_config


def _examples_parquet_from_cfg(cfg: dict, repo_root: Path) -> Path:
    if cfg.get("examples_parquet"):
        p = Path(str(cfg["examples_parquet"]))
        return p if p.is_absolute() else repo_root / p
    cache_root = repo_root / str(cfg.get("cohort_cache_root", "artifacts/recs/eval_cache"))
    cache_name = str(cfg["cohort_cache_name"])
    return cohort_parquet_path(cache_root / cache_name)


def main() -> None:
    t_start = time.perf_counter()
    parser = argparse.ArgumentParser(description="Export retrieval pools for a cached example cohort.")
    parser.add_argument("config", type=str, help="Path to JSON config.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    repo_root = Path(__file__).resolve().parents[1]

    pool_method = str(cfg.get("pool_method", METHOD_TWO_TOWER_V1))
    k_retrieval = int(cfg.get("k_retrieval", 100))
    min_review_chars = int(cfg.get("min_review_chars", 30))
    artifact_dir = repo_root / str(cfg.get("artifact_dir", "artifacts/recs"))
    catalog_item_batch = int(cfg.get("catalog_item_batch", 256))
    verbose = bool(cfg.get("verbose", True))

    examples_parquet = _examples_parquet_from_cfg(cfg, repo_root)
    if not examples_parquet.is_file():
        raise FileNotFoundError(f"Example cohort not found: {examples_parquet}")

    output_path = Path(str(cfg["output_path"]))
    if not output_path.is_absolute():
        output_path = repo_root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    two_tower_model_path: Path | None = None
    if pool_method == METHOD_TWO_TOWER_V1:
        if not cfg.get("two_tower_model_path"):
            raise ValueError("two_tower_model_path is required when pool_method is two_tower_v1")
        p = Path(str(cfg["two_tower_model_path"]))
        two_tower_model_path = p if p.is_absolute() else repo_root / p

    print(f"examples_parquet={examples_parquet}")
    print(f"pool_method={pool_method} k_retrieval={k_retrieval}")
    print(f"output_path={output_path}")

    inputs = prepare_eval_inputs_from_cache(
        repo_root=repo_root,
        split=str(cfg.get("split", "val")),
        min_review_chars=min_review_chars,
        examples_parquet=examples_parquet,
        artifact_dir=artifact_dir,
        verbose=verbose,
    )
    examples = load_eval_examples_from_parquet(examples_parquet)
    if len(examples) != len(inputs.examples):
        raise RuntimeError(
            f"Example count mismatch: parquet={len(examples)} prepare={len(inputs.examples)}"
        )

    if pool_method == METHOD_TWO_TOWER_V1:
        assert two_tower_model_path is not None
        hub_url, hub_max_chars = load_hub_settings(inputs.retriever)
        tower_model = load_two_tower_model(
            two_tower_model_path,
            hub_url=hub_url,
            n_items=len(inputs.retriever.app_ids),
            embed_dim=int(inputs.retriever.embedding_matrix.shape[1]),
        )
        score_fn = make_two_tower_score_fn(
            tower_model,
            inputs.retriever,
            max_chars=hub_max_chars,
            catalog_item_batch=catalog_item_batch,
            mask_query_app=True,
        )
    else:
        raise ValueError(f"Unsupported pool_method for export: {pool_method!r}")

    app_ids = inputs.app_ids
    rows: list[dict] = []
    ex_iter = enumerate(examples)
    if verbose:
        ex_iter = tqdm(ex_iter, total=len(examples), desc=f"pool {pool_method}", unit="example")

    for ex_idx, ex in ex_iter:
        scores = score_fn(ex)
        order = _rank_rows(scores)
        retrieved = order[:k_retrieval]
        retr_ids = [int(app_ids[int(i)]) for i in retrieved]
        retr_scores = [float(scores[int(i)]) for i in retrieved]
        positives = sorted(int(a) for a in ex["validation_positive_app_ids"])
        rows.append(
            {
                "ex_idx": int(ex_idx),
                "pool_method": pool_method,
                "user_id": str(ex["user_id"]),
                "query_app_id": int(ex["query_app_id"]),
                "query_ts": float(ex["query_ts"]),
                "slice_name": slice_name_from_n_targets(int(ex["n_eval_targets"])),
                "n_eval_targets": int(ex["n_eval_targets"]),
                "validation_positive_app_ids_json": json.dumps(positives),
                "retrieved_app_ids_json": json.dumps(retr_ids),
                "retrieved_scores_json": json.dumps(retr_scores),
                "retrieval_k": int(k_retrieval),
            }
        )

    out_df = pd.DataFrame(rows)
    out_df.to_parquet(output_path, index=False)

    meta_path = output_path.with_suffix(".meta.json")
    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(Path(args.config).resolve()),
        "examples_parquet": str(examples_parquet.resolve()),
        "pool_method": pool_method,
        "k_retrieval": k_retrieval,
        "n_pools": int(len(out_df)),
        "output_path": str(output_path.resolve()),
        "two_tower_model_path": str(two_tower_model_path) if two_tower_model_path else None,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Wrote {output_path} ({len(out_df):,} pools)")
    print(f"Wrote {meta_path}")
    print(f"Total script runtime: {time.perf_counter() - t_start:.2f}s")


if __name__ == "__main__":
    main()
