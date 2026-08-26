"""Tests for the RAG chunk-retrieval + v2a serve path (``RAGRecommender``, shipped default)."""

from __future__ import annotations

from pathlib import Path

import pytest

from steam_review_ml.recommender.rag_recommender import RAGRecommender
from steam_review_ml.recommender.serve_config import load_serve_config


def _has_serve_artifacts() -> bool:
    cfg = load_serve_config()
    chroma_dir = Path(cfg["rag_chroma_persist_dir"])
    igdb = Path(cfg["igdb_enriched_path"])
    embeddings = chroma_dir.parents[1] / "game_profile" / "default" / "game_profile_embeddings.npz"
    legacy = chroma_dir.parents[3] / "recs" / "game_profile_embeddings.npz"
    return chroma_dir.is_dir() and igdb.is_file() and (embeddings.is_file() or legacy.is_file())


def test_load_serve_config_resolves_paths() -> None:
    cfg = load_serve_config()
    assert cfg["default_method"] == "two_tower_v1_v2a_embed_query_logpop_blend"
    assert Path(cfg["rag_chroma_persist_dir"]).is_dir()
    assert Path(cfg["igdb_enriched_path"]).is_file()
    assert cfg["k_retrieval"] == 100
    assert cfg["k_final"] == 10


@pytest.mark.skipif(not _has_serve_artifacts(), reason="serve stack artifacts not present")
def test_rag_recommender_recommend_smoke() -> None:
    pytest.importorskip("chromadb")
    pytest.importorskip("sentence_transformers")
    rec = RAGRecommender.from_serve_config()
    # Civilization V app_id from catalog smoke tests
    df = rec.recommend(
        "Great strategy game with deep mechanics.",
        query_app_id=8930,
    )
    assert len(df) == rec.k_final
    assert "score" in df.columns
    assert "app_id" in df.columns
    assert 8930 not in set(df["app_id"].astype(int).tolist())
    assert rec.method_id == "two_tower_v1_v2a_embed_query_logpop_blend"


@pytest.mark.skipif(not _has_serve_artifacts(), reason="serve stack artifacts not present")
def test_rag_recommender_from_config_matches_method() -> None:
    pytest.importorskip("chromadb")
    pytest.importorskip("sentence_transformers")
    rec = RAGRecommender.from_serve_config()
    assert rec.method_id == "two_tower_v1_v2a_embed_query_logpop_blend"
    assert rec.rag_variant == "any_polarity__log_weighted"
