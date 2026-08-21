"""Tests for the RAG extension job configs (GameChunksJobConfig, GameChunkEmbeddingsJobConfig)."""

from __future__ import annotations

from pathlib import Path

import pytest

from steam_review_ml.recommender.job_config import (
    GameChunkEmbeddingsJobConfig,
    GameChunksJobConfig,
)


def test_game_chunks_from_json_uses_defaults_and_resolves_paths(tmp_path: Path) -> None:
    cfg = GameChunksJobConfig.from_json(
        tmp_path,
        {
            "train_input_path": "data/processed/train.parquet",
            "igdb_input_path": "artifacts/igdb/igdb_games__enriched.parquet",
            "output_path": "artifacts/recs/embeddings/game_chunks/default/game_review_chunks.parquet",
        },
    )

    assert cfg.train_input_path == tmp_path / "data/processed/train.parquet"
    assert cfg.igdb_input_path == tmp_path / "artifacts/igdb/igdb_games__enriched.parquet"
    assert cfg.max_reviews_per_game == 50
    assert cfg.min_review_chars == 30


def test_game_chunks_from_json_requires_paths(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="train_input_path"):
        GameChunksJobConfig.from_json(tmp_path, {})


def test_game_chunks_from_json_overrides_params(tmp_path: Path) -> None:
    cfg = GameChunksJobConfig.from_json(
        tmp_path,
        {
            "train_input_path": "a.parquet",
            "igdb_input_path": "b.parquet",
            "output_path": "c.parquet",
            "max_reviews_per_game": 20,
            "min_review_chars": 10,
        },
    )
    assert cfg.max_reviews_per_game == 20
    assert cfg.min_review_chars == 10


def test_game_chunk_embeddings_from_json_uses_defaults_and_resolves_paths(tmp_path: Path) -> None:
    cfg = GameChunkEmbeddingsJobConfig.from_json(
        tmp_path,
        {
            "input_path": "artifacts/recs/embeddings/game_chunks/default/game_review_chunks.parquet",
            "chroma_persist_dir": "artifacts/recs/embeddings/game_chunks/chroma",
        },
    )

    assert cfg.input_path == tmp_path / "artifacts/recs/embeddings/game_chunks/default/game_review_chunks.parquet"
    assert cfg.chroma_persist_dir == tmp_path / "artifacts/recs/embeddings/game_chunks/chroma"
    assert cfg.review_chunks_collection == "game_review_chunks"
    assert cfg.game_profiles_collection == "game_profiles"
    assert cfg.batch_size == 64
    assert cfg.max_chars_per_chunk == 8000
    assert cfg.description_blend_weight == 0.1


def test_game_chunk_embeddings_from_json_requires_paths(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="input_path"):
        GameChunkEmbeddingsJobConfig.from_json(tmp_path, {})


def test_game_chunk_embeddings_from_json_overrides_params(tmp_path: Path) -> None:
    cfg = GameChunkEmbeddingsJobConfig.from_json(
        tmp_path,
        {
            "input_path": "a.parquet",
            "chroma_persist_dir": "chroma",
            "description_blend_weight": 0.25,
            "batch_size": 32,
        },
    )
    assert cfg.description_blend_weight == 0.25
    assert cfg.batch_size == 32
