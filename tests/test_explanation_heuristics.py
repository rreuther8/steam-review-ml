"""Tests for the Stage 4 explanation heuristics (see steam_review_ml.evaluation.explanation_heuristics)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from steam_review_ml.evaluation.explanation_heuristics import (
    content_word_overlap_ratio,
    cosine_similarity,
    find_ungrounded_tags,
    is_degenerate_output,
    load_catalog_tag_vocabulary,
    load_igdb_tags,
)


def test_is_degenerate_output_flags_empty_text() -> None:
    assert is_degenerate_output("") is True
    assert is_degenerate_output("   ") is True


def test_is_degenerate_output_flags_refusal() -> None:
    assert is_degenerate_output("I cannot provide an explanation for this.") is True


def test_is_degenerate_output_flags_repetition_loop() -> None:
    assert is_degenerate_output("great great great great great great great") is True


def test_is_degenerate_output_false_for_normal_text() -> None:
    text = "This game shares the same fast-paced platforming and roguelite structure."
    assert is_degenerate_output(text) is False


def test_content_word_overlap_ratio_full_match() -> None:
    explanation = "This game features fast platforming."
    source = "A fast platforming game that features tight controls."
    assert content_word_overlap_ratio(explanation, source) == 1.0


def test_content_word_overlap_ratio_no_match() -> None:
    explanation = "This describes underwater exploration and submarines."
    source = "A racing vehicle with fast cars."
    ratio = content_word_overlap_ratio(explanation, source)
    assert ratio == 0.0


def test_content_word_overlap_ratio_no_content_words_returns_one() -> None:
    assert content_word_overlap_ratio("it is on at", "anything") == 1.0


def test_find_ungrounded_tags_flags_tag_not_in_candidate() -> None:
    flagged = find_ungrounded_tags(
        "This open-world RPG has deep strategy elements.",
        candidate_tags=["Platform", "Indie"],
        tag_vocabulary=["RPG", "Strategy", "Platform", "Indie"],
    )
    assert "RPG" in flagged
    assert "Strategy" in flagged
    assert "Platform" not in flagged


def test_find_ungrounded_tags_empty_when_grounded() -> None:
    flagged = find_ungrounded_tags(
        "This platform game is a great indie pick.",
        candidate_tags=["Platform", "Indie"],
        tag_vocabulary=["RPG", "Platform", "Indie"],
    )
    assert flagged == []


def test_cosine_similarity_identical_vectors() -> None:
    v = np.array([1.0, 2.0, 3.0])
    assert cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors() -> None:
    assert cosine_similarity(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(0.0)


def _write_fixture_parquet(path: Path) -> None:
    pd.DataFrame(
        {
            "app_id": [1, 2],
            "genres_names": [np.array(["Platform", "Indie"]), np.array(["Racing"])],
            "themes_names": [np.array(["Action"]), np.array(["Open world"])],
        }
    ).to_parquet(path, index=False)


def test_load_igdb_tags_joins_genres_and_themes(tmp_path: Path) -> None:
    fixture_path = tmp_path / "igdb_games__enriched.parquet"
    _write_fixture_parquet(fixture_path)

    result = load_igdb_tags([1], enriched_path=str(fixture_path))

    assert result[1] == {"Platform", "Indie", "Action"}


def test_load_igdb_tags_raises_on_missing_app_id(tmp_path: Path) -> None:
    fixture_path = tmp_path / "igdb_games__enriched.parquet"
    _write_fixture_parquet(fixture_path)

    with pytest.raises(ValueError, match="99"):
        load_igdb_tags([99], enriched_path=str(fixture_path))


def test_load_catalog_tag_vocabulary_unions_all_games(tmp_path: Path) -> None:
    fixture_path = tmp_path / "igdb_games__enriched.parquet"
    _write_fixture_parquet(fixture_path)

    vocabulary = load_catalog_tag_vocabulary(enriched_path=str(fixture_path))

    assert vocabulary == {"Platform", "Indie", "Action", "Racing", "Open world"}
