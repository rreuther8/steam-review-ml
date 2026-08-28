"""Tests for the recommender job configs (RAG extension jobs + pool-export config)."""

from __future__ import annotations

from pathlib import Path

import pytest

from steam_review_ml.recommender.job_config import (
    GameChunkEmbeddingsJobConfig,
    GameChunksJobConfig,
    PoolExportConfig,
    RagPoolExportConfig,
    TwoTowerPoolExportConfig,
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


def test_two_tower_pool_export_from_json_uses_defaults_and_resolves_path(tmp_path: Path) -> None:
    cfg = TwoTowerPoolExportConfig.from_json(tmp_path, {"two_tower_model_path": "model.keras"})

    assert cfg.two_tower_model_path == tmp_path / "model.keras"
    assert cfg.catalog_item_batch == 256


def test_two_tower_pool_export_from_json_requires_model_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="two_tower_model_path"):
        TwoTowerPoolExportConfig.from_json(tmp_path, {})


def test_rag_pool_export_from_json_uses_defaults(tmp_path: Path) -> None:
    cfg = RagPoolExportConfig.from_json(tmp_path, {})

    assert cfg.rag_chroma_persist_dir is None
    assert cfg.rag_variant == "any_polarity__log_weighted"
    assert cfg.rag_query_blend_weight == 0.5


def test_rag_pool_export_from_json_resolves_chroma_dir_when_given(tmp_path: Path) -> None:
    cfg = RagPoolExportConfig.from_json(tmp_path, {"rag_chroma_persist_dir": "chroma"})

    assert cfg.rag_chroma_persist_dir == tmp_path / "chroma"


def test_pool_export_config_from_json_dispatches_to_two_tower(tmp_path: Path) -> None:
    cfg = PoolExportConfig.from_json(
        tmp_path,
        {"output_path": "out.parquet", "two_tower_model_path": "model.keras"},
        examples_parquet=tmp_path / "examples.parquet",
    )

    assert isinstance(cfg.pipeline, TwoTowerPoolExportConfig)
    assert cfg.pool_method == "two_tower_v1"
    assert cfg.output_path == tmp_path / "out.parquet"


def test_pool_export_config_from_json_dispatches_to_rag(tmp_path: Path) -> None:
    cfg = PoolExportConfig.from_json(
        tmp_path,
        {"output_path": "out.parquet", "pool_method": "rag_chunk_v1_vector_blend_query"},
        examples_parquet=tmp_path / "examples.parquet",
    )

    assert isinstance(cfg.pipeline, RagPoolExportConfig)
    assert cfg.pool_method == "rag_chunk_v1_vector_blend_query"


def test_pool_export_config_from_json_rejects_unknown_pool_method(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="bogus"):
        PoolExportConfig.from_json(
            tmp_path,
            {"output_path": "out.parquet", "pool_method": "bogus"},
            examples_parquet=tmp_path / "examples.parquet",
        )


def test_pool_export_config_from_json_uses_defaults(tmp_path: Path) -> None:
    cfg = PoolExportConfig.from_json(
        tmp_path,
        {"output_path": "out.parquet", "two_tower_model_path": "model.keras"},
        examples_parquet=tmp_path / "examples.parquet",
    )

    assert cfg.split == "val"
    assert cfg.k_retrieval == 100
    assert cfg.min_review_chars == 30
    assert cfg.verbose is True
