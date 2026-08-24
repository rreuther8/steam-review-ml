"""Tests for the Stage 4 candidate-text lookup (see steam_review_ml.evaluation.candidate_text)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from steam_review_ml.evaluation.candidate_text import build_candidate_text_lookup


def _write_fixture_parquet(path: Path) -> None:
    pd.DataFrame(
        {
            "app_id": [1, 2, 3],
            "app_name": ["Game One", "Game Two", "Game Three"],
            "summary": ["A summary of game one.", "A summary of game two.", "A summary of game three."],
            # app_id 2 has no storyline, matching real IGDB rows where ~half are null.
            "storyline": ["Story of game one.", None, "Story of game three."],
        }
    ).to_parquet(path, index=False)


def test_lookup_joins_name_summary_and_storyline(tmp_path: Path) -> None:
    fixture_path = tmp_path / "igdb_games__enriched.parquet"
    _write_fixture_parquet(fixture_path)

    result = build_candidate_text_lookup([1], enriched_path=str(fixture_path))

    text = result[1]
    assert "Game One" in text
    assert "A summary of game one." in text
    assert "Story of game one." in text


def test_lookup_omits_storyline_when_missing(tmp_path: Path) -> None:
    fixture_path = tmp_path / "igdb_games__enriched.parquet"
    _write_fixture_parquet(fixture_path)

    result = build_candidate_text_lookup([2], enriched_path=str(fixture_path))

    text = result[2]
    assert "Game Two" in text
    assert "A summary of game two." in text
    # Regression guard: a NaN storyline must not leak in as the literal string "nan".
    assert "nan" not in text.lower()


def test_lookup_subsets_to_requested_app_ids(tmp_path: Path) -> None:
    fixture_path = tmp_path / "igdb_games__enriched.parquet"
    _write_fixture_parquet(fixture_path)

    result = build_candidate_text_lookup([1, 3], enriched_path=str(fixture_path))

    assert set(result.keys()) == {1, 3}


def test_lookup_raises_on_missing_app_id(tmp_path: Path) -> None:
    fixture_path = tmp_path / "igdb_games__enriched.parquet"
    _write_fixture_parquet(fixture_path)

    with pytest.raises(ValueError, match=r"\[999\]"):
        build_candidate_text_lookup([1, 999], enriched_path=str(fixture_path))
