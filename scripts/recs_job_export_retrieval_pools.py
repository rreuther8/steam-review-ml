"""Export frozen retrieval pools for a cached example cohort (ranker train/tune).

Scores exactly one retrieval pipeline on every row in ``example_cohort.parquet`` and writes a
slim pool parquet reusable by multiple rankers. ``pool_method`` in the config selects which
pipeline dataclass gets built below -- a discriminated union, not a flat dict of optional keys,
so a config can't accidentally supply fields for both pipelines and have the job run both.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from steam_review_ml.evaluation.example_cohort import cohort_parquet_path, slice_name_from_n_targets
from steam_review_ml.evaluation.retrieval_offline_eval import (
    EvalInputs,
    _rank_rows,
    load_eval_examples_from_parquet,
    prepare_eval_inputs_from_cache,
)
from steam_review_ml.recommender.chroma_retrieve import ChromaGameProfileRetriever
from steam_review_ml.recommender.job_config import (
    PoolExportConfig,
    RagPoolExportConfig,
    TwoTowerPoolExportConfig,
)
from steam_review_ml.recommender.math_utils import l2_normalize
from steam_review_ml.recommender.two_tower_score import load_two_tower_model, make_two_tower_score_fn
from steam_review_ml.recommender.two_tower_train import load_hub_settings
from steam_review_ml.utils import load_config

ScoreFn = Callable[[int, dict], np.ndarray]


def _examples_parquet_from_cfg(cfg: dict, repo_root: Path) -> Path:
    if cfg.get("examples_parquet"):
        p = Path(str(cfg["examples_parquet"]))
        return p if p.is_absolute() else repo_root / p
    cache_root = repo_root / str(cfg.get("cohort_cache_root", "artifacts/recs/eval_cache"))
    cache_name = str(cfg["cohort_cache_name"])
    return cohort_parquet_path(cache_root / cache_name)


def _build_two_tower_score_fn(inputs: EvalInputs, pipeline: TwoTowerPoolExportConfig) -> ScoreFn:
    hub_url, hub_max_chars = load_hub_settings(inputs.retriever)
    tower_model = load_two_tower_model(
        pipeline.two_tower_model_path,
        hub_url=hub_url,
        n_items=len(inputs.retriever.app_ids),
        embed_dim=int(inputs.retriever.embedding_matrix.shape[1]),
    )
    scorer = make_two_tower_score_fn(
        tower_model,
        inputs.retriever,
        max_chars=hub_max_chars,
        catalog_item_batch=pipeline.catalog_item_batch,
        mask_query_app=True,
    )
    return lambda ex_idx, ex: scorer(ex)


def _build_rag_score_fn(
    inputs: EvalInputs,
    pipeline: RagPoolExportConfig,
    examples: list[dict],
    *,
    repo_root: Path,
    verbose: bool,
) -> ScoreFn:
    """Batched: one encode call for all reviews, one for unique descriptions, one (chunked)
    Chroma call for all queries -- see ``chroma_retrieve.embed_texts_batch`` /
    ``score_batch_against_catalog`` for why this replaces the per-example approach without
    changing what's computed (same Chroma distances, not a local reimplementation).
    """
    rag_retriever = ChromaGameProfileRetriever(
        variant=pipeline.rag_variant,
        chroma_persist_dir=pipeline.rag_chroma_persist_dir,
        repo_root=repo_root,
    )
    description_by_app_id = rag_retriever.load_all_description_texts()

    query_app_ids = np.array([int(ex["query_app_id"]) for ex in examples])
    review_texts = [ex["query_text"] for ex in examples]

    if verbose:
        print(f"Encoding {len(review_texts):,} review texts...")
    review_vecs = rag_retriever.embed_texts_batch(review_texts)

    unique_app_ids = sorted({int(a) for a in query_app_ids})
    desc_texts_by_app = {
        a: description_by_app_id[a] for a in unique_app_ids if description_by_app_id.get(a, "").strip()
    }
    if verbose:
        print(f"Encoding {len(desc_texts_by_app):,} unique description texts...")
    desc_vec_by_app: dict[int, np.ndarray] = {}
    if desc_texts_by_app:
        desc_app_ids = list(desc_texts_by_app.keys())
        desc_vecs = rag_retriever.embed_texts_batch([desc_texts_by_app[a] for a in desc_app_ids])
        desc_vec_by_app = dict(zip(desc_app_ids, desc_vecs))

    w = pipeline.rag_query_blend_weight
    blended = np.empty_like(review_vecs)
    for i, qid in enumerate(query_app_ids):
        desc_vec = desc_vec_by_app.get(int(qid))
        blended[i] = (
            review_vecs[i] if desc_vec is None else l2_normalize((1.0 - w) * review_vecs[i] + w * desc_vec)
        )

    if verbose:
        print(f"Scoring {len(blended):,} queries against the catalog...")
    scores_matrix = rag_retriever.score_batch_against_catalog(
        blended, query_app_ids=query_app_ids, app_ids=inputs.app_ids
    )
    return lambda ex_idx, ex: scores_matrix[ex_idx]


def main() -> None:
    t_start = time.perf_counter()
    parser = argparse.ArgumentParser(description="Export retrieval pools for a cached example cohort.")
    parser.add_argument("config", type=str, help="Path to JSON config.")
    args = parser.parse_args()

    raw_cfg = load_config(args.config)
    repo_root = Path(__file__).resolve().parents[1]
    examples_parquet = _examples_parquet_from_cfg(raw_cfg, repo_root)
    cfg = PoolExportConfig.from_json(repo_root, raw_cfg, examples_parquet=examples_parquet)

    if not cfg.examples_parquet.is_file():
        raise FileNotFoundError(f"Example cohort not found: {cfg.examples_parquet}")
    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"examples_parquet={cfg.examples_parquet}")
    print(f"pool_method={cfg.pool_method} k_retrieval={cfg.k_retrieval}")
    print(f"output_path={cfg.output_path}")

    inputs = prepare_eval_inputs_from_cache(
        repo_root=repo_root,
        split=cfg.split,
        min_review_chars=cfg.min_review_chars,
        examples_parquet=cfg.examples_parquet,
        artifact_dir=cfg.artifact_dir,
        verbose=cfg.verbose,
    )
    examples = load_eval_examples_from_parquet(cfg.examples_parquet)
    if len(examples) != len(inputs.examples):
        raise RuntimeError(f"Example count mismatch: parquet={len(examples)} prepare={len(inputs.examples)}")

    if isinstance(cfg.pipeline, TwoTowerPoolExportConfig):
        score_fn = _build_two_tower_score_fn(inputs, cfg.pipeline)
    else:
        score_fn = _build_rag_score_fn(inputs, cfg.pipeline, examples, repo_root=repo_root, verbose=cfg.verbose)

    app_ids = inputs.app_ids
    rows: list[dict] = []
    ex_iter = enumerate(examples)
    if cfg.verbose:
        ex_iter = tqdm(ex_iter, total=len(examples), desc=f"pool {cfg.pool_method}", unit="example")

    for ex_idx, ex in ex_iter:
        scores = score_fn(ex_idx, ex)
        order = _rank_rows(scores)
        retrieved = order[: cfg.k_retrieval]
        retr_ids = [int(app_ids[int(i)]) for i in retrieved]
        retr_scores = [float(scores[int(i)]) for i in retrieved]
        positives = sorted(int(a) for a in ex["validation_positive_app_ids"])
        rows.append(
            {
                "ex_idx": int(ex_idx),
                "pool_method": cfg.pool_method,
                "user_id": str(ex["user_id"]),
                "query_app_id": int(ex["query_app_id"]),
                "query_ts": float(ex["query_ts"]),
                "slice_name": slice_name_from_n_targets(int(ex["n_eval_targets"])),
                "n_eval_targets": int(ex["n_eval_targets"]),
                "validation_positive_app_ids_json": json.dumps(positives),
                "retrieved_app_ids_json": json.dumps(retr_ids),
                "retrieved_scores_json": json.dumps(retr_scores),
                "retrieval_k": int(cfg.k_retrieval),
            }
        )

    out_df = pd.DataFrame(rows)
    out_df.to_parquet(cfg.output_path, index=False)

    meta_path = cfg.output_path.with_suffix(".meta.json")
    meta: dict = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(Path(args.config).resolve()),
        "examples_parquet": str(cfg.examples_parquet.resolve()),
        "pool_method": cfg.pool_method,
        "k_retrieval": cfg.k_retrieval,
        "n_pools": int(len(out_df)),
        "output_path": str(cfg.output_path.resolve()),
    }
    if isinstance(cfg.pipeline, TwoTowerPoolExportConfig):
        meta["two_tower_model_path"] = str(cfg.pipeline.two_tower_model_path)
    else:
        meta["rag_variant"] = cfg.pipeline.rag_variant
        meta["rag_query_blend_weight"] = cfg.pipeline.rag_query_blend_weight
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Wrote {cfg.output_path} ({len(out_df):,} pools)")
    print(f"Wrote {meta_path}")
    print(f"Total script runtime: {time.perf_counter() - t_start:.2f}s")


if __name__ == "__main__":
    main()
