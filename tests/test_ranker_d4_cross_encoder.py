from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from steam_review_ml.recommender.ranker_d4_cross_encoder import (
    DEFAULT_CE_BLEND_W_GRID,
    blend_ce_retr,
    blend_ce_retr_logpop,
    build_app_candidate_texts,
    query_text_map_from_examples,
    tune_ce_retr_blend,
)


def test_blend_ce_retr_endpoints() -> None:
    ce = np.asarray([0.0, 1.0, 0.5])
    retr = np.asarray([1.0, 0.0, 0.5])
    all_ce = blend_ce_retr(ce, retr, w=1.0)
    all_retr = blend_ce_retr(ce, retr, w=0.0)
    assert float(all_ce[1]) > float(all_ce[0])
    assert float(all_retr[0]) > float(all_retr[1])


def test_blend_ce_retr_logpop_uses_pop() -> None:
    ce = np.asarray([0.2, 0.8, 0.5])
    retr = np.asarray([0.2, 0.8, 0.5])
    pool_apps = [1, 2, 3]
    pop_row = np.asarray([1.0, 100.0, 10.0], dtype=np.float64)
    app_to_row = {1: 0, 2: 1, 3: 2}
    s = blend_ce_retr_logpop(ce, retr, pool_apps, pop_row=pop_row, app_to_row=app_to_row, w=0.0, alpha=0.0)
    assert float(s[1]) > float(s[0])


def test_build_app_candidate_texts_uses_longest_review(tmp_path: Path) -> None:
    path = tmp_path / "profiles.parquet"
    pd.DataFrame(
        {
            "app_id": [1, 1, 2],
            "app_name": ["A", "A", "B"],
            "review": ["short", "much longer candidate review text", "b"],
            "review_len": [5, 33, 1],
        }
    ).to_parquet(path)
    texts = build_app_candidate_texts(path, max_chars_per_app=200)
    assert "much longer candidate review text" in texts[1]


def test_default_ce_blend_w_grid_excludes_zero() -> None:
    assert 0.0 not in DEFAULT_CE_BLEND_W_GRID
    assert DEFAULT_CE_BLEND_W_GRID[0] == 0.1
    assert DEFAULT_CE_BLEND_W_GRID[-1] == 1.0


def test_tune_ce_retr_blend_picks_best_w() -> None:
    pools = [
        {
            "ex_idx": 0,
            "validation_positive_app_ids_json": json.dumps([2]),
            "retrieved_app_ids_json": json.dumps([1, 2, 3]),
            "retrieved_scores_json": json.dumps([0.9, 0.1, 0.5]),
        }
    ]
    ce_by_ex = {0: np.asarray([0.1, 0.95, 0.2])}
    app_ids = np.asarray([1, 2, 3])
    app_to_row = {1: 0, 2: 1, 3: 2}
    w, ndcg = tune_ce_retr_blend(pools, ce_by_ex, app_ids=app_ids, app_to_row=app_to_row, w_grid=[0.0, 1.0])
    assert w == 1.0
    assert ndcg > 0


def test_query_text_map_from_examples() -> None:
    examples = [{"query_text": "hello"}, {"query_text": "world"}]
    m = query_text_map_from_examples(examples)
    assert m[0] == "hello"
