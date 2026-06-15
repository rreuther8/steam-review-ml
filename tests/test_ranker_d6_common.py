from __future__ import annotations

import json

import numpy as np

from steam_review_ml.recommender.ranker_d6_common import attach_query_text, padded_pool_batch
from steam_review_ml.recommender.ranker_d6_rank_head import rank_head_feature_dim


def test_rank_head_feature_dim() -> None:
    assert rank_head_feature_dim(64) == 193


def test_attach_query_text() -> None:
    pools = [{"ex_idx": 1, "retrieved_app_ids_json": "[]"}]
    out = attach_query_text(pools, {1: "hello"})
    assert out[0]["query_text"] == "hello"


def test_padded_pool_batch_shapes() -> None:
    app_to_row = {10: 0, 20: 1}
    pools = [
        {
            "ex_idx": 0,
            "query_text": "q",
            "validation_positive_app_ids_json": json.dumps([20]),
            "retrieved_app_ids_json": json.dumps([10, 20]),
            "retrieved_scores_json": json.dumps([0.2, 0.8]),
        }
    ]
    q, ids, retr, labels = padded_pool_batch(pools, app_to_row=app_to_row, max_list_len=100)
    assert q == ["q"]
    assert ids[0] == [0, 1]
    assert retr[0] == [0.2, 0.8]
    assert labels[0] == [0.0, 1.0]
