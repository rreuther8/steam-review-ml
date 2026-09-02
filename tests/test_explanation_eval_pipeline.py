"""Tests for steam_review_ml.evaluation.explanation_eval_pipeline (fake rec/backend, no local LLM)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from steam_review_ml.evaluation.explanation_eval_pipeline import (
    generate_explanations_for_cohort,
    generate_or_load_explanations,
    score_explanations,
    summarize_explanation_scores,
)


class _FakeRec:
    """Always recommends app_id=2, mirroring the shape ``Recommender.recommend`` returns."""

    def recommend(self, query_text: str, *, query_app_id: int) -> pd.DataFrame:
        return pd.DataFrame([{"app_id": 2, "app_name": "Game Two", "score": 0.9}])


class _FakeBackend:
    def generate_explanation(self, query_text: str, recommended_text: str) -> str:
        return "A short grounded explanation."


class _FakeEmbedModel:
    def encode(self, texts: list[str]) -> np.ndarray:
        return np.array([[float(len(t)), 1.0] for t in texts])


def _write_igdb_fixture(path: Path) -> None:
    pd.DataFrame(
        {
            "app_id": [1, 2],
            "app_name": ["Game One", "Game Two"],
            "summary": ["Summary one.", "Summary two."],
            "storyline": [None, None],
            "genres_names": [np.array(["Indie"]), np.array(["Platform"])],
            "themes_names": [np.array(["Action"]), np.array(["Action"])],
        }
    ).to_parquet(path, index=False)


@pytest.fixture(autouse=True)
def _clear_candidate_text_cache():
    from steam_review_ml.evaluation import candidate_text, explanation_heuristics

    candidate_text._load_igdb_text_by_app.cache_clear()
    explanation_heuristics._load_igdb_tags_by_app.cache_clear()
    yield
    candidate_text._load_igdb_text_by_app.cache_clear()
    explanation_heuristics._load_igdb_tags_by_app.cache_clear()


def test_generate_explanations_for_cohort_uses_top1_pick(tmp_path: Path, monkeypatch) -> None:
    igdb_path = tmp_path / "igdb_games__enriched.parquet"
    _write_igdb_fixture(igdb_path)

    import steam_review_ml.evaluation.explanation_eval_pipeline as pipeline_module

    original_lookup = pipeline_module.build_candidate_text_lookup
    monkeypatch.setattr(
        pipeline_module,
        "build_candidate_text_lookup",
        lambda app_ids, **kwargs: original_lookup(app_ids, enriched_path=str(igdb_path)),
    )

    cohort_df = pd.DataFrame([{"query_app_id": 1, "query_text": "I like indie platformers."}])
    result = generate_explanations_for_cohort(
        _FakeRec(), _FakeBackend(), cohort_df, {1: "Game One", 2: "Game Two"}, verbose=False
    )

    assert len(result) == 1
    row = result.iloc[0]
    assert row["query_app_id"] == 1
    assert row["query_app_name"] == "Game One"
    assert row["rec_app_id"] == 2
    assert row["rec_app_name"] == "Game Two"
    assert row["explanation"] == "A short grounded explanation."


def test_generate_or_load_explanations_uses_cache(tmp_path: Path) -> None:
    cache_path = tmp_path / "explanations.parquet"
    cached = pd.DataFrame([{"query_app_id": 1, "rec_app_id": 2, "explanation": "cached"}])
    cached.to_parquet(cache_path, index=False)

    result = generate_or_load_explanations(
        rec=None,
        cohort_df=pd.DataFrame(),
        app_name_by_id={},
        cache_path=cache_path,
        gguf_path=Path("unused.gguf"),
        verbose=False,
    )

    assert result["explanation"].tolist() == ["cached"]


def test_score_explanations_flags_ungrounded_tags(tmp_path: Path) -> None:
    igdb_path = tmp_path / "igdb_games__enriched.parquet"
    _write_igdb_fixture(igdb_path)

    results_df = pd.DataFrame(
        [
            {
                "query_app_id": 1,
                "query_text": "I like action games.",
                "rec_app_id": 2,
                "candidate_text": "Game Two\n\nSummary two.",
                "explanation": "This role-playing game (RPG) has deep systems.",
            }
        ]
    )

    scored = score_explanations(results_df, _FakeEmbedModel(), enriched_path=str(igdb_path))

    assert scored.iloc[0]["is_degenerate"] == False  # noqa: E712 -- numpy bool, not python bool
    assert "Role-playing (RPG)" not in scored.iloc[0]["ungrounded_tags"]  # not in fixture vocab at all
    assert scored.iloc[0]["content_overlap_ratio"] < 1.0
    assert "relevance_cosine_query_game" in scored.columns


def test_summarize_explanation_scores() -> None:
    scored_df = pd.DataFrame(
        [
            {
                "is_degenerate": False,
                "ungrounded_tags": [],
                "content_overlap_ratio": 0.8,
                "relevance_cosine": 0.5,
                "relevance_cosine_query_game": 0.6,
            },
            {
                "is_degenerate": True,
                "ungrounded_tags": ["RPG"],
                "content_overlap_ratio": 0.2,
                "relevance_cosine": 0.1,
                "relevance_cosine_query_game": 0.2,
            },
        ]
    )

    summary = summarize_explanation_scores(scored_df)

    assert summary["n_examples"] == 2
    assert summary["degenerate_rate"] == pytest.approx(0.5)
    assert summary["any_ungrounded_tag_rate"] == pytest.approx(0.5)
    assert summary["content_overlap_ratio_mean"] == pytest.approx(0.5)
    assert summary["relevance_cosine_query_game_mean"] == pytest.approx(0.4)
