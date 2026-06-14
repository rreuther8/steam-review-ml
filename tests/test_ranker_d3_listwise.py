from __future__ import annotations

import json

import numpy as np

from steam_review_ml.recommender.ranker_d3_listwise import (
    listnet_target_probs,
    listwise_rows_from_pools,
)


def test_listnet_target_uniform_over_positives() -> None:
    target = listnet_target_probs(np.asarray([0.0, 1.0, 1.0]))
    assert target.tolist() == [0.0, 0.5, 0.5]


def test_listnet_target_zero_when_no_positive() -> None:
    target = listnet_target_probs(np.asarray([0.0, 0.0]))
    assert float(target.sum()) == 0.0


def test_listwise_rows_one_per_pool() -> None:
    pop_row = np.asarray([1.0, 10.0, 100.0], dtype=np.float32)
    app_to_row = {1: 0, 2: 1, 3: 2}
    pools = [
        {
            "ex_idx": 0,
            "validation_positive_app_ids_json": json.dumps([2]),
            "retrieved_app_ids_json": json.dumps([1, 2, 3]),
            "retrieved_scores_json": json.dumps([0.1, 0.9, 0.5]),
        }
    ]
    df = listwise_rows_from_pools(pools, pop_row=pop_row, app_to_row=app_to_row)
    assert len(df) == 1
    labels = json.loads(df.iloc[0]["labels_json"])
    assert labels == [0.0, 1.0, 0.0]
