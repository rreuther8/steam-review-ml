"""Tests for example cohort builder helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from steam_review_ml.evaluation.example_cohort import (
    assert_cohort_disjoint,
    cohort_parquet_path,
    examples_to_frame,
)


def test_assert_cohort_disjoint_passes_when_no_overlap(tmp_path: Path) -> None:
    left = pd.DataFrame(
        {
            "user_id": ["u1", "u2"],
            "query_app_id": [1, 2],
            "query_ts": [1.0, 2.0],
        }
    )
    right_path = tmp_path / "ref.parquet"
    pd.DataFrame(
        {
            "user_id": ["u3"],
            "query_app_id": [3],
            "query_ts": [3.0],
        }
    ).to_parquet(right_path, index=False)
    assert_cohort_disjoint(left, right_path)


def test_assert_cohort_disjoint_fails_on_overlap(tmp_path: Path) -> None:
    left = pd.DataFrame(
        {
            "user_id": ["u1"],
            "query_app_id": [10],
            "query_ts": [100.0],
        }
    )
    right_path = tmp_path / "ref.parquet"
    pd.DataFrame(
        {
            "user_id": ["u1"],
            "query_app_id": [10],
            "query_ts": [100.0],
        }
    ).to_parquet(right_path, index=False)
    with pytest.raises(ValueError, match="overlapping"):
        assert_cohort_disjoint(left, right_path)


def test_cohort_parquet_path_prefers_new_name(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "example_cohort.parquet").write_bytes(b"")
    assert cohort_parquet_path(cache).name == "example_cohort.parquet"


def test_cohort_parquet_path_falls_back_to_legacy(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "eval_examples.parquet").write_bytes(b"")
    assert cohort_parquet_path(cache).name == "eval_examples.parquet"


def test_examples_to_frame_roundtrip_positives() -> None:
    ex = {
        "user_id": "u1",
        "query_app_id": 5,
        "query_text": "hello",
        "query_ts": 1.0,
        "n_eval_targets": 2,
        "validation_positive_app_ids": {7, 8},
        "train_review_rows": [],
        "cohort": "c",
        "eval_pos_cohort": "e",
    }
    df = examples_to_frame([ex])
    loaded = json.loads(df.iloc[0]["validation_positive_app_ids_json"])
    assert loaded == [7, 8]
    assert df.iloc[0]["slice_name"] == "slice_a_multi_target"
