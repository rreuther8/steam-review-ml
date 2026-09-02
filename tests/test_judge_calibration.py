"""Tests for steam_review_ml.evaluation.judge_calibration (no API/model deps)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from steam_review_ml.evaluation.judge_calibration import (
    build_calibration_comparison,
    load_hand_labels,
    sample_for_hand_labeling,
    summarize_calibration,
)


def _explanations_df(n: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "query_app_id": 100 + i,
                "query_app_name": f"Query {i}",
                "rec_app_id": 200 + i,
                "rec_app_name": f"Rec {i}",
                "query_text": f"query text {i}",
                "candidate_text": f"candidate text {i}",
                "explanation": f"explanation {i}",
            }
            for i in range(n)
        ]
    )


def _write_igdb_fixture(path: Path, app_ids: list[int]) -> None:
    pd.DataFrame(
        {
            "app_id": app_ids,
            "app_name": [f"Query {i}" for i in range(len(app_ids))],
            "summary": [f"IGDB summary for query app {a}." for a in app_ids],
            "storyline": [None] * len(app_ids),
        }
    ).to_parquet(path, index=False)


@pytest.fixture(autouse=True)
def _clear_candidate_text_cache():
    from steam_review_ml.evaluation import candidate_text

    candidate_text._load_igdb_text_by_app.cache_clear()
    yield
    candidate_text._load_igdb_text_by_app.cache_clear()


def test_sample_for_hand_labeling_has_example_id_and_blank_label_columns(tmp_path: Path) -> None:
    igdb_path = tmp_path / "igdb.parquet"
    _write_igdb_fixture(igdb_path, list(range(100, 120)))

    sample = sample_for_hand_labeling(_explanations_df(20), n=5, seed=2026, enriched_path=str(igdb_path))

    assert len(sample) == 5
    assert list(sample["example_id"]) == sorted(sample["example_id"])
    assert sample["example_id"].max() < 20
    assert (sample["human_faithfulness"] == "").all()
    assert (sample["human_relevance"] == "").all()
    assert "query_text" in sample.columns
    assert "explanation" in sample.columns
    assert "query_igdb_text" in sample.columns
    for _, row in sample.iterrows():
        assert f"IGDB summary for query app {row['query_app_id']}." in row["query_igdb_text"]
    # query_igdb_text (grounding text) must not be confused with query_text (the raw review)
    assert list(sample.columns).index("query_igdb_text") < list(sample.columns).index("candidate_text")


def test_sample_for_hand_labeling_is_deterministic_given_seed(tmp_path: Path) -> None:
    igdb_path = tmp_path / "igdb.parquet"
    _write_igdb_fixture(igdb_path, list(range(100, 120)))
    df = _explanations_df(20)
    a = sample_for_hand_labeling(df, n=5, seed=2026, enriched_path=str(igdb_path))
    b = sample_for_hand_labeling(df, n=5, seed=2026, enriched_path=str(igdb_path))
    assert a["example_id"].tolist() == b["example_id"].tolist()


def test_sample_for_hand_labeling_caps_at_pool_size(tmp_path: Path) -> None:
    igdb_path = tmp_path / "igdb.parquet"
    _write_igdb_fixture(igdb_path, list(range(100, 103)))
    sample = sample_for_hand_labeling(_explanations_df(3), n=10, seed=2026, enriched_path=str(igdb_path))
    assert len(sample) == 3


def test_sample_for_hand_labeling_excludes_given_ids(tmp_path: Path) -> None:
    igdb_path = tmp_path / "igdb.parquet"
    _write_igdb_fixture(igdb_path, list(range(100, 120)))
    df = _explanations_df(20)

    first = sample_for_hand_labeling(df, n=5, seed=2026, enriched_path=str(igdb_path))
    topped_up = sample_for_hand_labeling(
        df, n=5, seed=2026, enriched_path=str(igdb_path), exclude_example_ids=first["example_id"].tolist()
    )

    assert set(topped_up["example_id"]).isdisjoint(set(first["example_id"]))
    assert len(topped_up) == 5


def test_sample_for_hand_labeling_exclude_shrinks_pool_below_n(tmp_path: Path) -> None:
    igdb_path = tmp_path / "igdb.parquet"
    _write_igdb_fixture(igdb_path, list(range(100, 103)))
    df = _explanations_df(3)

    sample = sample_for_hand_labeling(df, n=5, seed=2026, enriched_path=str(igdb_path), exclude_example_ids=[0, 1])

    assert sample["example_id"].tolist() == [2]


def test_load_hand_labels_raises_on_unfilled_rows(tmp_path: Path) -> None:
    path = tmp_path / "sample.csv"
    pd.DataFrame(
        [
            {"example_id": 0, "human_faithfulness": 4, "human_relevance": ""},
            {"example_id": 1, "human_faithfulness": 3, "human_relevance": 3},
        ]
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="still unlabeled"):
        load_hand_labels(path)


def test_load_hand_labels_raises_on_out_of_range_score(tmp_path: Path) -> None:
    path = tmp_path / "sample.csv"
    pd.DataFrame([{"example_id": 0, "human_faithfulness": 9, "human_relevance": 3}]).to_csv(path, index=False)

    with pytest.raises(ValueError, match="out of range"):
        load_hand_labels(path)


def test_load_hand_labels_returns_int_columns(tmp_path: Path) -> None:
    path = tmp_path / "sample.csv"
    pd.DataFrame([{"example_id": 0, "human_faithfulness": 4, "human_relevance": 5}]).to_csv(path, index=False)

    df = load_hand_labels(path)
    assert df["human_faithfulness"].tolist() == [4]
    assert df["human_relevance"].tolist() == [5]


def test_build_calibration_comparison_joins_on_example_id() -> None:
    hand_labels_df = pd.DataFrame(
        [
            {"example_id": 0, "human_faithfulness": 5, "human_relevance": 4},
            {"example_id": 2, "human_faithfulness": 1, "human_relevance": 2},
        ]
    )
    heuristic_scores_df = pd.DataFrame(
        [
            {
                "content_overlap_ratio": 0.5,
                "relevance_cosine": 0.6,
                "relevance_cosine_query_game": 0.55,
                "is_degenerate": False,
            },
            {
                "content_overlap_ratio": 0.1,
                "relevance_cosine": 0.2,
                "relevance_cosine_query_game": 0.25,
                "is_degenerate": False,
            },
            {
                "content_overlap_ratio": 0.9,
                "relevance_cosine": 0.8,
                "relevance_cosine_query_game": 0.85,
                "is_degenerate": False,
            },
        ]
    )
    judge_scores_df = pd.DataFrame(
        [
            {"judge_faithfulness": 5, "judge_relevance": 4, "judge_rationale": "a"},
            {"judge_faithfulness": 4, "judge_relevance": 4, "judge_rationale": "b"},
            {"judge_faithfulness": 2, "judge_relevance": 1, "judge_rationale": "c"},
        ]
    )

    comparison = build_calibration_comparison(hand_labels_df, heuristic_scores_df, judge_scores_df)

    assert len(comparison) == 2
    row0 = comparison[comparison["example_id"] == 0].iloc[0]
    assert row0["judge_faithfulness"] == 5
    assert row0["content_overlap_ratio"] == 0.5
    row2 = comparison[comparison["example_id"] == 2].iloc[0]
    assert row2["judge_faithfulness"] == 2
    assert row2["content_overlap_ratio"] == 0.9


def test_build_calibration_comparison_raises_when_judge_missing_example() -> None:
    hand_labels_df = pd.DataFrame([{"example_id": 5, "human_faithfulness": 5, "human_relevance": 4}])
    heuristic_scores_df = pd.DataFrame(
        [
            {
                "content_overlap_ratio": 0.5,
                "relevance_cosine": 0.6,
                "relevance_cosine_query_game": 0.5,
                "is_degenerate": False,
            }
        ]
        * 6
    )
    judge_scores_df = pd.DataFrame(
        [{"judge_faithfulness": 5, "judge_relevance": 4, "judge_rationale": "a"}] * 2
    )

    with pytest.raises(ValueError, match="not present in judge scores"):
        build_calibration_comparison(hand_labels_df, heuristic_scores_df, judge_scores_df)


def test_summarize_calibration_perfect_agreement() -> None:
    comparison_df = pd.DataFrame(
        [
            {
                "human_faithfulness": 5,
                "human_relevance": 4,
                "judge_faithfulness": 5,
                "judge_relevance": 4,
                "content_overlap_ratio": 0.9,
                "relevance_cosine": 0.8,
                "relevance_cosine_query_game": 0.7,
            },
            {
                "human_faithfulness": 1,
                "human_relevance": 2,
                "judge_faithfulness": 1,
                "judge_relevance": 2,
                "content_overlap_ratio": 0.1,
                "relevance_cosine": 0.2,
                "relevance_cosine_query_game": 0.3,
            },
        ]
    )

    summary = summarize_calibration(comparison_df)

    assert summary["n_examples"] == 2
    assert summary["judge_faithfulness_exact_agreement_rate"] == pytest.approx(1.0)
    assert summary["judge_relevance_exact_agreement_rate"] == pytest.approx(1.0)
    assert summary["judge_faithfulness_within_1_rate"] == pytest.approx(1.0)
    assert summary["judge_faithfulness_spearman"] == pytest.approx(1.0)
    assert summary["heuristic_groundedness_spearman"] == pytest.approx(1.0)
    assert summary["heuristic_relevance_vs_review_spearman"] == pytest.approx(1.0)
    assert summary["heuristic_relevance_vs_query_game_spearman"] == pytest.approx(1.0)


def test_summarize_calibration_disagreement() -> None:
    comparison_df = pd.DataFrame(
        [
            {
                "human_faithfulness": 5,
                "human_relevance": 5,
                "judge_faithfulness": 1,
                "judge_relevance": 1,
                "content_overlap_ratio": 0.5,
                "relevance_cosine": 0.5,
                "relevance_cosine_query_game": 0.5,
            },
            {
                "human_faithfulness": 1,
                "human_relevance": 1,
                "judge_faithfulness": 5,
                "judge_relevance": 5,
                "content_overlap_ratio": 0.5,
                "relevance_cosine": 0.5,
                "relevance_cosine_query_game": 0.5,
            },
        ]
    )

    summary = summarize_calibration(comparison_df)

    assert summary["judge_faithfulness_exact_agreement_rate"] == pytest.approx(0.0)
    assert summary["judge_faithfulness_within_1_rate"] == pytest.approx(0.0)
    assert summary["judge_faithfulness_spearman"] == pytest.approx(-1.0)
    # constant heuristic column -> undefined correlation, not a crash
    assert summary["heuristic_groundedness_spearman"] is None
    assert summary["heuristic_relevance_vs_query_game_spearman"] is None
