"""Tests for centralized recommender evaluation contract."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import pytest

import steam_review_ml.evaluation.retrieval_offline_eval as evaluation
from steam_review_ml.recommender.math_utils import l2_normalize
from steam_review_ml.recommender.retrieve import fusion_c_raw_plus_behavior_query_vector


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
        evaluation.RetrievalEvalConfig(
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
    )
    assert not tables.retrieval_overall.empty
    assert not tables.ranking_overall.empty


def test_include_query_text_flag_adds_query_text_to_artifact_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_retriever = object()
    fake_inputs = _fake_eval_inputs(fake_retriever)

    monkeypatch.setattr(evaluation, "prepare_eval_inputs", lambda **kwargs: fake_inputs)
    monkeypatch.setattr(evaluation, "_build_method_registry", lambda **kwargs: _fake_registry(fake_retriever))

    tables = evaluation.run_retrieval_eval(
        evaluation.RetrievalEvalConfig(
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
            include_query_text=True,
        )
    )
    expected_query_text = {ex["query_app_id"]: ex["query_text"] for ex in fake_inputs.examples}
    assert tables.artifact_rows
    for row in tables.artifact_rows:
        assert row["query_text"] == expected_query_text[row["query_app_id"]]


def test_fusion_c_query_vector_fuses_session_and_weighted_behavior() -> None:
    class FakeRetriever:
        def embed_text(self, text: str) -> np.ndarray:  # noqa: ANN001
            t = str(text)
            if t == "query":
                return np.array([1.0, 0.0, 0.0], dtype=np.float32)
            if t == "x":
                return np.array([0.25, 0.0, 0.0], dtype=np.float32)
            raise AssertionError(f"unexpected embed text: {t!r}")

    ex = {
        "query_text": "query",
        "train_review_rows": [
            {"app_id": 2, "_norm_author__playtime_at_review": 1.0, "text": "x"},
        ],
    }
    X = np.eye(3, dtype=np.float32)
    app_to_row = {2: 1}
    out = fusion_c_raw_plus_behavior_query_vector(
        FakeRetriever(),  # type: ignore[arg-type]
        ex,
        embedding_matrix=X,
        app_to_row=app_to_row,
    )
    expected = l2_normalize(np.array([1.0, 1.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-6)


def test_oracle_ranking_metrics_ceiling_within_retrieved_pool() -> None:
    app_ids = np.array([10, 20, 30], dtype=np.int64)
    retrieved = np.array([0, 1, 2], dtype=np.int64)
    positives = {20}
    ranked_bad = np.array([0, 1], dtype=np.int64)
    oracle = evaluation._oracle_ranked_indices_from_retrieved(retrieved, positives, app_ids)

    assert list(oracle) == [1, 0, 2]
    assert evaluation.hit_rate_at_k(ranked_bad, positives, 1, app_ids) == 0.0
    assert evaluation.hit_rate_at_k(oracle, positives, 1, app_ids) == 1.0
    assert evaluation.ndcg_at_k(oracle, positives, 1, app_ids) == 1.0


def test_ranking_overall_includes_oracle_columns(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_retriever = object()
    fake_inputs = _fake_eval_inputs(fake_retriever)

    monkeypatch.setattr(evaluation, "prepare_eval_inputs", lambda **kwargs: fake_inputs)
    monkeypatch.setattr(evaluation, "_build_method_registry", lambda **kwargs: _fake_registry(fake_retriever))

    tables = evaluation.run_retrieval_eval(
        evaluation.RetrievalEvalConfig(
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
            k_retrieval=3,
            k_personalization=2,
            enable_popularity_decile_diagnostics=True,
            include_random_sanity=False,
            random_seed=1,
            artifact_dir=None,
            verbose=False,
        )
    )

    for col in evaluation.ORACLE_RANKING_METRIC_COLS:
        assert col in tables.ranking_overall.columns
    assert (tables.ranking_overall["OracleHit@K"] >= tables.ranking_overall["Hit@K"]).all()
    assert (tables.ranking_overall["OracleNDCG@K"] >= tables.ranking_overall["NDCG@K"]).all()


def test_eval_outputs_include_personalization_columns(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_retriever = object()
    fake_inputs = _fake_eval_inputs(fake_retriever)

    monkeypatch.setattr(evaluation, "prepare_eval_inputs", lambda **kwargs: fake_inputs)
    monkeypatch.setattr(evaluation, "_build_method_registry", lambda **kwargs: _fake_registry(fake_retriever))

    tables = evaluation.run_retrieval_eval(
        evaluation.RetrievalEvalConfig(
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
    )

    expected_prefixes = (
        "ILD@",
        "CatalogCoverage@",
        "Novelty@",
        "PersonalizationGapVsPopularity@",
    )

    def has_all_prefixes(df: pd.DataFrame) -> bool:
        return all(any(c.startswith(p) for c in df.columns) for p in expected_prefixes)

    assert has_all_prefixes(tables.retrieval_overall)
    assert has_all_prefixes(tables.retrieval_by_slice)
    assert has_all_prefixes(tables.retrieval_by_support_bucket)
    assert has_all_prefixes(tables.retrieval_by_pop_decile)
    assert has_all_prefixes(tables.retrieval_pop_delta_vs_popularity)

    assert has_all_prefixes(tables.ranking_overall)
    assert has_all_prefixes(tables.ranking_by_slice)
    assert has_all_prefixes(tables.ranking_by_support_bucket)
    assert has_all_prefixes(tables.ranking_by_pop_decile)
    assert has_all_prefixes(tables.ranking_pop_delta_vs_popularity)


def test_build_method_registry_omits_rag_chunk_methods_when_persist_dir_unset() -> None:
    registry = evaluation._build_method_registry(
        retriever=object(),  # type: ignore[arg-type]
        X=np.eye(3, dtype=np.float32),
        pop_row=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        app_to_row={1: 0, 2: 1, 3: 2},
        multi_max_reviews=5,
        rng=np.random.default_rng(0),
        mask_query_app=True,
    )
    assert evaluation.METHOD_RAG_CHUNK_RAW_QUERY not in registry
    assert evaluation.METHOD_RAG_CHUNK_QUERY_PLUS_DESC not in registry
