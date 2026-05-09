"""Tests for centralized recommender evaluation contract."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import pytest

from steam_review_ml.recommender import evaluation


def _fake_eval_inputs(fake_retriever: object) -> evaluation.EvalInputs:
    app_ids = np.array([1, 2, 3], dtype=np.int64)
    return evaluation.EvalInputs(
        retriever=fake_retriever,  # type: ignore[arg-type]
        examples=[
            {
                "user_id": "u1",
                "query_app_id": 1,
                "query_text": "a",
                "query_ts": 1.0,
                "validation_positive_app_ids": {2},
                "n_eval_targets": 1,
                "train_review_rows": [],
                "cohort": "val_no_train",
                "eval_pos_cohort": "val_single_pos_eval",
            },
            {
                "user_id": "u2",
                "query_app_id": 2,
                "query_text": "b",
                "query_ts": 2.0,
                "validation_positive_app_ids": {1, 3},
                "n_eval_targets": 2,
                "train_review_rows": [{"app_id": 1, "text": "x", "ts": 1.5}],
                "cohort": "val_pos_train",
                "eval_pos_cohort": "val_multi_pos_eval",
            },
            {
                "user_id": "u3",
                "query_app_id": 3,
                "query_text": "c",
                "query_ts": 3.0,
                "validation_positive_app_ids": {1},
                "n_eval_targets": 1,
                "train_review_rows": [
                    {"app_id": 1, "text": "y", "ts": 1.1},
                    {"app_id": 2, "text": "z", "ts": 1.2},
                ],
                "cohort": "val_multi_pos_train",
                "eval_pos_cohort": "val_single_pos_eval",
            },
        ],
        embedding_matrix=np.eye(3, dtype=np.float32),
        app_ids=app_ids,
        app_to_row={1: 0, 2: 1, 3: 2},
        pop_row=np.array([10.0, 5.0, 1.0], dtype=np.float32),
        eval_split_name="val",
        prep_diagnostics={
            "eval_records_count": 3,
            "sampled_rows_count": 3,
            "full_eval_user_count": 3,
            "full_eval_multi_pos_user_count": 1,
            "sampled_rows": 3,
            "evaluable_examples": 3,
            "dropped_rows": 0,
            "drop_reasons": {"no_other_positive_app": 0},
        },
    )


def _fake_registry(_: object) -> dict[str, Callable[[dict], np.ndarray]]:
    def score_raw(ex: dict) -> np.ndarray:
        return {
            1: np.array([0.1, 0.9, 0.8], dtype=np.float32),
            2: np.array([0.8, 0.1, 0.9], dtype=np.float32),
            3: np.array([0.9, 0.8, 0.1], dtype=np.float32),
        }[int(ex["query_app_id"])]

    def score_pop(ex: dict) -> np.ndarray:
        _ = ex
        return np.array([0.9, 0.8, 0.7], dtype=np.float32)

    def score_multi(ex: dict) -> np.ndarray:
        return {
            1: np.array([0.2, 0.7, 0.9], dtype=np.float32),
            2: np.array([0.7, 0.2, 0.9], dtype=np.float32),
            3: np.array([0.9, 0.7, 0.2], dtype=np.float32),
        }[int(ex["query_app_id"])]

    def score_random(ex: dict) -> np.ndarray:
        _ = ex
        return np.array([0.4, 0.3, 0.2], dtype=np.float32)

    return {
        "raw": score_raw,
        "popularity_train": score_pop,
        "multi_mean_train": score_multi,
        "random": score_random,
    }


def test_run_retrieval_eval_reuses_prepared_retriever(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_retriever = object()
    fake_inputs = _fake_eval_inputs(fake_retriever)

    def fake_prepare_eval_inputs(**kwargs):  # noqa: ANN003
        _ = kwargs
        return fake_inputs

    def fake_content_retriever(*args, **kwargs):  # noqa: ANN002, ANN003
        _ = (args, kwargs)
        raise AssertionError("run_retrieval_eval should not create a second ContentRetriever")

    def fake_build_method_registry(**kwargs):  # noqa: ANN003
        assert kwargs["retriever"] is fake_retriever
        return _fake_registry(fake_retriever)

    monkeypatch.setattr(evaluation, "prepare_eval_inputs", fake_prepare_eval_inputs)
    monkeypatch.setattr(evaluation, "ContentRetriever", fake_content_retriever)
    monkeypatch.setattr(evaluation, "_build_method_registry", fake_build_method_registry)

    tables = evaluation.run_retrieval_eval(
        repo_root=Path("."),
        split="val",
        methods=["raw", "popularity_train", "multi_mean_train"],
        active_cohort="all",
        max_examples=100,
        support_app_filter_mode="strict",
        cohort_sizing={},
        min_review_chars=1,
        max_train_rows_per_user=5,
        multi_max_reviews=5,
        k_final=2,
        k_personalization=2,
        enable_popularity_decile_diagnostics=True,
        include_random_sanity=False,
        random_seed=1,
        artifact_dir=None,
        verbose=False,
    )
    assert not tables.overall.empty


def test_eval_outputs_include_personalization_columns(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_retriever = object()
    fake_inputs = _fake_eval_inputs(fake_retriever)

    monkeypatch.setattr(evaluation, "prepare_eval_inputs", lambda **kwargs: fake_inputs)
    monkeypatch.setattr(evaluation, "_build_method_registry", lambda **kwargs: _fake_registry(fake_retriever))

    tables = evaluation.run_retrieval_eval(
        repo_root=Path("."),
        split="val",
        methods=["raw", "popularity_train", "multi_mean_train"],
        active_cohort="all",
        max_examples=100,
        support_app_filter_mode="strict",
        cohort_sizing={},
        min_review_chars=1,
        max_train_rows_per_user=5,
        multi_max_reviews=5,
        k_final=2,
        k_personalization=2,
        enable_popularity_decile_diagnostics=True,
        include_random_sanity=False,
        random_seed=1,
        artifact_dir=None,
        verbose=False,
    )

    expected_prefixes = (
        "ILD@",
        "CatalogCoverage@",
        "Novelty@",
        "PersonalizationGapVsPopularity@",
    )

    def has_all_prefixes(df: pd.DataFrame) -> bool:
        return all(any(c.startswith(p) for c in df.columns) for p in expected_prefixes)

    assert has_all_prefixes(tables.overall)
    assert has_all_prefixes(tables.by_slice)
    assert has_all_prefixes(tables.by_support_bucket)
    assert has_all_prefixes(tables.by_pop_decile)
    assert has_all_prefixes(tables.pop_delta_vs_popularity)
