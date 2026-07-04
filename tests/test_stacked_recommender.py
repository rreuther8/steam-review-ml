"""Tests for stacked two-tower + v2a serve path."""

from __future__ import annotations

from pathlib import Path

import pytest

from steam_review_ml.recommender.retrieve import default_repo_root
from steam_review_ml.recommender.serve_config import load_serve_config
from steam_review_ml.recommender.stacked_recommender import StackedRecommender


def _has_serve_artifacts() -> bool:
    root = default_repo_root()
    cfg = load_serve_config()
    tower = Path(cfg["two_tower_model_path"])
    igdb = Path(cfg["igdb_enriched_path"])
    embeddings = (
        root
        / "artifacts"
        / "recs"
        / "embeddings"
        / "game_profile"
        / "default"
        / "game_profile_embeddings.npz"
    )
    legacy = root / "artifacts" / "recs" / "game_profile_embeddings.npz"
    return tower.is_file() and igdb.is_file() and (embeddings.is_file() or legacy.is_file())


def test_load_serve_config_resolves_paths() -> None:
    cfg = load_serve_config()
    assert cfg["default_method"] == "two_tower_v1_v2a_embed_query_logpop_blend"
    assert Path(cfg["two_tower_model_path"]).is_file()
    assert Path(cfg["igdb_enriched_path"]).is_file()
    assert cfg["k_retrieval"] == 100
    assert cfg["k_final"] == 10


@pytest.mark.skipif(not _has_serve_artifacts(), reason="serve stack artifacts not present")
def test_stacked_recommender_recommend_smoke() -> None:
    pytest.importorskip("tensorflow")
    pytest.importorskip("tensorflow_hub")
    rec = StackedRecommender.from_serve_config()
    # Civilization V app_id from catalog smoke tests
    df = rec.recommend(
        "Great strategy game with deep mechanics.",
        query_app_id=8930,
        k=5,
    )
    assert len(df) == 5
    assert "score" in df.columns
    assert "app_id" in df.columns
    assert 8930 not in set(df["app_id"].astype(int).tolist())


@pytest.mark.skipif(not _has_serve_artifacts(), reason="serve stack artifacts not present")
def test_stacked_recommender_from_config_matches_method() -> None:
    pytest.importorskip("tensorflow")
    rec = StackedRecommender.from_serve_config()
    assert rec.method_id == "two_tower_v1_v2a_embed_query_logpop_blend"
