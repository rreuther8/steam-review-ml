from __future__ import annotations

import json

import numpy as np
import pandas as pd

from steam_review_ml.recommender.ranker_d2_pointwise import (
    FEATURE_COLS,
    _class_weights,
    _sample_weights,
    pointwise_rows_from_pools,
)


def test_pointwise_rows_from_pools_labels() -> None:
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
    df = pointwise_rows_from_pools(pools, pop_row=pop_row, app_to_row=app_to_row)
    assert len(df) == 3
    assert set(FEATURE_COLS).issubset(df.columns)
    assert float(df.loc[df["retr_score"] == 0.9, "label"].iloc[0]) == 1.0
    assert float(df["label"].sum()) == 1.0


def test_class_weights_balances_positives() -> None:
    labels = np.asarray([0, 0, 0, 1], dtype=np.float32)
    w = _class_weights(labels)
    assert w[1] > w[0]


def test_sample_weights_match_class_weight_dict() -> None:
    labels = np.asarray([0, 1, 0], dtype=np.float32)
    cw = _class_weights(labels)
    sw = _sample_weights(labels, cw)
    assert sw.tolist() == [cw[0], cw[1], cw[0]]
