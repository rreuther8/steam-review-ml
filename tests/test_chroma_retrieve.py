"""Tests for steam_review_ml.recommender.chroma_retrieve (pure helpers only; no real Chroma DB)."""

from __future__ import annotations

import numpy as np

from steam_review_ml.recommender.chroma_retrieve import (
    UNSCORED_SENTINEL,
    _align_batch_query_results,
)


def test_align_batch_query_results_maps_scores_to_app_id_columns() -> None:
    metadatas = [[{"app_id": 20}, {"app_id": 10}]]
    distances = [[0.2, 0.5]]
    app_ids = np.array([10, 20, 30])

    scores = _align_batch_query_results(
        metadatas, distances, query_app_ids=np.array([30]), app_ids=app_ids
    )

    assert scores.shape == (1, 3)
    assert scores[0, 0] == -0.5  # app_id 10
    assert scores[0, 1] == -0.2  # app_id 20


def test_align_batch_query_results_masks_query_app_id() -> None:
    metadatas = [[{"app_id": 10}, {"app_id": 30}]]
    distances = [[0.1, 0.1]]
    app_ids = np.array([10, 20, 30])

    scores = _align_batch_query_results(
        metadatas, distances, query_app_ids=np.array([10]), app_ids=app_ids
    )

    assert scores[0, 0] == UNSCORED_SENTINEL  # masked: this row's own query game
    assert scores[0, 2] == -0.1  # app_id 30 unaffected


def test_align_batch_query_results_unscored_for_missing_catalog_row() -> None:
    """A game in `app_ids` that Chroma never returned for this variant stays unscored."""
    metadatas = [[{"app_id": 10}]]
    distances = [[0.3]]
    app_ids = np.array([10, 20])

    scores = _align_batch_query_results(
        metadatas, distances, query_app_ids=np.array([999]), app_ids=app_ids
    )

    assert scores[0, 0] == -0.3
    assert scores[0, 1] == UNSCORED_SENTINEL


def test_align_batch_query_results_handles_multiple_queries() -> None:
    metadatas = [[{"app_id": 10}], [{"app_id": 20}]]
    distances = [[0.4], [0.1]]
    app_ids = np.array([10, 20])

    scores = _align_batch_query_results(
        metadatas, distances, query_app_ids=np.array([20, 10]), app_ids=app_ids
    )

    assert scores.shape == (2, 2)
    assert scores[0, 0] == -0.4
    assert scores[0, 1] == UNSCORED_SENTINEL  # row 0's own query game (20) masked
    assert scores[1, 0] == UNSCORED_SENTINEL  # row 1's own query game (10) masked
    assert scores[1, 1] == -0.1
