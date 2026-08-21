"""Tests for the RAG chunk-building logic in scripts/recs_job_game_chunks.py."""

from __future__ import annotations

import pandas as pd

from scripts.recs_job_game_chunks import _build_description_chunks, _build_review_chunks


def _reviews_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            # app 1: 3 eligible reviews, top-2 cap by votes_helpful, tie broken by review_id.
            {"app_id": 1, "app_name": "A", "review_id": 10, "review": "x" * 40, "votes_helpful": 5, "recommended": 1},
            {"app_id": 1, "app_name": "A", "review_id": 11, "review": "y" * 40, "votes_helpful": 5, "recommended": 0},
            {"app_id": 1, "app_name": "A", "review_id": 12, "review": "z" * 40, "votes_helpful": 1, "recommended": 1},
            # app 1: below min_review_chars, dropped.
            {"app_id": 1, "app_name": "A", "review_id": 13, "review": "short", "votes_helpful": 9, "recommended": 1},
            # app 2: single review, any polarity kept (recommended==0).
            {"app_id": 2, "app_name": "B", "review_id": 20, "review": "w" * 40, "votes_helpful": 0, "recommended": 0},
        ]
    )


def test_build_review_chunks_caps_by_votes_helpful_with_review_id_tiebreak() -> None:
    out = _build_review_chunks(_reviews_df(), max_reviews_per_game=2, min_review_chars=30)

    app1 = out[out["app_id"] == 1].sort_values("review_id")
    assert list(app1["review_id"]) == [10, 11]  # tie on votes_helpful=5, review_id asc wins
    assert list(app1["chunk_type"]) == ["review", "review"]
    assert list(app1["text_len"]) == [40, 40]

    app2 = out[out["app_id"] == 2]
    assert list(app2["review_id"]) == [20]
    assert list(app2["recommended"]) == [0]  # any-polarity, not just recommended==1


def test_build_review_chunks_drops_short_reviews() -> None:
    out = _build_review_chunks(_reviews_df(), max_reviews_per_game=50, min_review_chars=30)
    assert 13 not in set(out["review_id"])


def test_build_description_chunks_concatenates_summary_and_storyline() -> None:
    igdb = pd.DataFrame(
        [
            {"app_id": 1, "summary": "Summary text.", "storyline": "Storyline text."},
            {"app_id": 2, "summary": "Summary only.", "storyline": None},
            {"app_id": 3, "summary": None, "storyline": None},  # dropped: no text
        ]
    )
    out = _build_description_chunks(igdb)

    assert set(out["app_id"]) == {1, 2}
    assert (out["chunk_type"] == "description").all()
    assert out["review_id"].isna().all()

    row1 = out.loc[out["app_id"] == 1].iloc[0]
    assert row1["text"] == "Summary text.\n\nStoryline text."

    row2 = out.loc[out["app_id"] == 2].iloc[0]
    assert row2["text"] == "Summary only."
