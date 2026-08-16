"""Tests for the pure pooling/blend math in scripts/recs_job_game_chunk_embeddings.py."""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.recs_job_game_chunk_embeddings import _blend_with_description, _pool_review_variant


def _review_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"app_id": 1, "review_id": 10, "votes_helpful": 9, "recommended": 1},
            {"app_id": 1, "review_id": 11, "votes_helpful": 1, "recommended": 0},
            {"app_id": 2, "review_id": 20, "votes_helpful": 0, "recommended": 1},
        ]
    )


def _basis_vectors() -> np.ndarray:
    # Orthonormal basis so weighted-average direction is easy to check by inspection.
    return np.array(
        [
            [1.0, 0.0, 0.0],  # app 1, review 10 (recommended=1, votes_helpful=9)
            [0.0, 1.0, 0.0],  # app 1, review 11 (recommended=0, votes_helpful=1)
            [0.0, 0.0, 1.0],  # app 2, review 20 (recommended=1, votes_helpful=0)
        ],
        dtype=np.float32,
    )


def test_pool_flat_any_polarity_is_uniform_average() -> None:
    out = _pool_review_variant(_review_df(), _basis_vectors(), polarity="any_polarity", weighting="flat")

    app1 = out[1]
    assert app1["n_reviews_pooled"] == 2
    assert app1["recommended_rate"] == 0.5
    expected = np.array([1.0, 1.0, 0.0]) / np.linalg.norm([1.0, 1.0, 0.0])
    assert np.allclose(app1["vector"], expected, atol=1e-5)


def test_pool_log_weighted_favors_higher_votes_helpful() -> None:
    out = _pool_review_variant(
        _review_df(), _basis_vectors(), polarity="any_polarity", weighting="log_weighted"
    )
    app1 = out[1]
    # review 10 has votes_helpful=9 > review 11's 1, so its axis should dominate.
    assert app1["vector"][0] > app1["vector"][1] > 0


def test_pool_recommended_only_filters_before_pooling() -> None:
    out = _pool_review_variant(
        _review_df(), _basis_vectors(), polarity="recommended_only", weighting="flat"
    )
    assert 1 in out
    assert out[1]["n_reviews_pooled"] == 1
    assert out[1]["recommended_rate"] == 1.0
    assert np.allclose(out[1]["vector"], [1.0, 0.0, 0.0], atol=1e-5)


def test_pool_log_weighted_falls_back_to_uniform_when_all_weights_zero() -> None:
    # app 2's only review has votes_helpful=0 -> log1p weight is 0 -> must not divide by zero.
    out = _pool_review_variant(
        _review_df(), _basis_vectors(), polarity="any_polarity", weighting="log_weighted"
    )
    assert np.allclose(out[2]["vector"], [0.0, 0.0, 1.0], atol=1e-5)


def test_blend_with_description_weights_and_normalizes() -> None:
    pooled = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    description = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    blended_zero = _blend_with_description(pooled, description, 0.0)
    assert np.allclose(blended_zero, pooled, atol=1e-5)

    blended_one = _blend_with_description(pooled, description, 1.0)
    assert np.allclose(blended_one, description, atol=1e-5)

    blended_half = _blend_with_description(pooled, description, 0.5)
    assert np.isclose(np.linalg.norm(blended_half), 1.0, atol=1e-5)
    assert blended_half[0] > 0 and blended_half[1] > 0
