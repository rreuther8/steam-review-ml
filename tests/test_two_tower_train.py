"""Tests for two-tower training helpers (no GPU / full data required)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from steam_review_ml.recommender.contrastive_examples import (
    build_eval_records,
    build_example_dicts,
    positive_app_ids_from_example,
    sample_query_rows_random,
)
from steam_review_ml.recommender.two_tower_train import (
    build_tower_contrastive_rows,
    history_to_dataframe,
    item_init_matrix_from_catalog,
)


class _FakeRetriever:
    app_ids = np.array([10, 20, 30], dtype=np.int64)
    embedding_matrix = np.eye(3, dtype=np.float32)


def test_positive_app_ids_from_example_both_keys():
    ex = {"positive_app_ids": {20}, "validation_positive_app_ids": {30}}
    assert positive_app_ids_from_example(ex) == {20}


def test_build_eval_records_minimal():
    df_query = pd.DataFrame(
        {
            "author.steamid": ["u1", "u1"],
            "app_id": [1, 2],
            "review_id": [1, 2],
            "review": ["a", "b"],
            "ts": [1.0, 2.0],
            "recommended": [1, 1],
        }
    )
    df_train = pd.DataFrame(
        {
            "author.steamid": ["u1"],
            "app_id": [3],
            "review_id": [3],
            "review": ["c"],
            "ts": [3.0],
            "recommended": [1],
        }
    )
    records = build_eval_records(df_query, df_train)
    assert len(records) == 2
    assert "cohort" in records.columns


def test_build_example_dicts_positives():
    df_base = pd.DataFrame(
        {
            "author.steamid": ["u1"],
            "app_id": [1],
            "review_id": [1],
            "review": ["query"],
            "ts": [1.0],
            "cohort": ["val_multi_pos_train"],
            "eval_pos_cohort": ["val_multi_pos_eval"],
        }
    )
    user_to_apps = {"u1": [1, 2]}
    df_train = pd.DataFrame(
        {
            "author.steamid": ["u1", "u1"],
            "app_id": [1, 2],
            "review": ["q", "other"],
            "ts": [1.0, 2.0],
            "recommended": [1, 1],
        }
    )
    rng = np.random.default_rng(0)
    examples, diag = build_example_dicts(
        df_base,
        user_to_apps,
        df_train_all=df_train,
        max_train_rows_per_user=5,
        support_app_filter_mode="strict",
        rng=rng,
    )
    assert len(examples) == 1
    assert positive_app_ids_from_example(examples[0]) == {2}
    assert diag["evaluable_examples"] == 1


def test_build_tower_contrastive_rows_grouping():
    examples = [
        {
            "query_text": "hello",
            "positive_app_ids": {20, 30},
        }
    ]
    texts, item_rows, groups = build_tower_contrastive_rows(
        examples,
        _FakeRetriever(),
        max_examples=10,
        max_chars=None,
    )
    assert len(texts) == 2
    assert list(item_rows) == [1, 2]
    assert list(groups) == [0, 0]


def test_item_init_matrix_from_catalog():
    mat = item_init_matrix_from_catalog(_FakeRetriever())
    assert mat.shape == (3, 3)


def test_history_to_dataframe():
    class _Hist:
        history = {
            "loss": [1.0, 0.9],
            "val_loss": [1.1, 0.85],
            "user_to_item_loss": [0.5, 0.4],
        }

    df = history_to_dataframe(_Hist())
    assert list(df.columns) == ["epoch", "loss", "val_loss", "user_to_item_loss"]
    assert len(df) == 2


def test_sample_query_rows_random_caps():
    df = pd.DataFrame({"author.steamid": [f"u{i}" for i in range(10)], "app_id": range(10)})
    rng = np.random.default_rng(0)
    out = sample_query_rows_random(df, max_examples=3, rng=rng)
    assert len(out) == 3
