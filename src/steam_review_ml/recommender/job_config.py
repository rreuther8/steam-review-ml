"""Job configs for the RAG extension chunking + embedding jobs (Stages 1-2).

Mirrors the frozen-dataclass + ``from_json(repo_root, cfg)`` shape already used by
``steam_review_ml.igdb.job_config`` (see ``docs/todo_job_config_boilerplate_refactor.md``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

from steam_review_ml.utils import require_str


@dataclass(frozen=True)
class GameChunksJobConfig:
    repo_root: Path
    train_input_path: Path
    igdb_input_path: Path
    output_path: Path
    max_reviews_per_game: int = 50
    min_review_chars: int = 30

    @classmethod
    def from_json(cls, repo_root: Path, cfg: dict[str, Any]) -> GameChunksJobConfig:
        return cls(
            repo_root=repo_root,
            train_input_path=repo_root / require_str(cfg, "train_input_path"),
            igdb_input_path=repo_root / require_str(cfg, "igdb_input_path"),
            output_path=repo_root / require_str(cfg, "output_path"),
            max_reviews_per_game=int(cfg.get("max_reviews_per_game", 50)),
            min_review_chars=int(cfg.get("min_review_chars", 30)),
        )


@dataclass(frozen=True)
class GameChunkEmbeddingsJobConfig:
    repo_root: Path
    input_path: Path
    chroma_persist_dir: Path
    review_chunks_collection: str = "game_review_chunks"
    game_profiles_collection: str = "game_profiles"
    embedding_model_name: str = "BAAI/bge-small-en-v1.5"
    batch_size: int = 64
    max_chars_per_chunk: int = 8000
    description_blend_weight: float = 0.1

    @classmethod
    def from_json(cls, repo_root: Path, cfg: dict[str, Any]) -> GameChunkEmbeddingsJobConfig:
        return cls(
            repo_root=repo_root,
            input_path=repo_root / require_str(cfg, "input_path"),
            chroma_persist_dir=repo_root / require_str(cfg, "chroma_persist_dir"),
            review_chunks_collection=str(
                cfg.get("review_chunks_collection", "game_review_chunks")
            ),
            game_profiles_collection=str(cfg.get("game_profiles_collection", "game_profiles")),
            embedding_model_name=str(
                cfg.get("embedding_model_name", "BAAI/bge-small-en-v1.5")
            ),
            batch_size=int(cfg.get("batch_size", 64)),
            max_chars_per_chunk=int(cfg.get("max_chars_per_chunk", 8000)),
            description_blend_weight=float(cfg.get("description_blend_weight", 0.1)),
        )


@dataclass(frozen=True)
class TwoTowerPoolExportConfig:
    """``pool_method == "two_tower_v1"``: score every example with the trained two-tower model."""

    two_tower_model_path: Path
    catalog_item_batch: int = 256

    @classmethod
    def from_json(cls, repo_root: Path, cfg: dict[str, Any]) -> TwoTowerPoolExportConfig:
        return cls(
            two_tower_model_path=repo_root / require_str(cfg, "two_tower_model_path"),
            catalog_item_batch=int(cfg.get("catalog_item_batch", 256)),
        )


@dataclass(frozen=True)
class RagPoolExportConfig:
    """``pool_method == "rag_chunk_v1_vector_blend_query"``: score with Stage 3's RAG retriever."""

    rag_chroma_persist_dir: Path | None
    rag_variant: str = "any_polarity__log_weighted"
    rag_query_blend_weight: float = 0.5

    @classmethod
    def from_json(cls, repo_root: Path, cfg: dict[str, Any]) -> RagPoolExportConfig:
        chroma_dir = cfg.get("rag_chroma_persist_dir")
        return cls(
            rag_chroma_persist_dir=(repo_root / str(chroma_dir)) if chroma_dir else None,
            rag_variant=str(cfg.get("rag_variant", "any_polarity__log_weighted")),
            rag_query_blend_weight=float(cfg.get("rag_query_blend_weight", 0.5)),
        )


RetrievalPipelineConfig = Union[TwoTowerPoolExportConfig, RagPoolExportConfig]


@dataclass(frozen=True)
class PoolExportConfig:
    """Fully parsed config for ``recs_job_export_retrieval_pools.py`` -- exactly one retrieval
    pipeline, chosen by ``pool_method``. A discriminated union (``pipeline``'s concrete type),
    not a flat dict of optional keys, so a config can't supply fields for both pipelines and
    have the job run both.
    """

    examples_parquet: Path
    output_path: Path
    split: str
    k_retrieval: int
    min_review_chars: int
    artifact_dir: Path
    verbose: bool
    pipeline: RetrievalPipelineConfig

    @property
    def pool_method(self) -> str:
        from steam_review_ml.evaluation.retrieval_offline_eval import (
            METHOD_RAG_CHUNK_VECTOR_BLEND,
            METHOD_TWO_TOWER_V1,
        )

        if isinstance(self.pipeline, TwoTowerPoolExportConfig):
            return METHOD_TWO_TOWER_V1
        return METHOD_RAG_CHUNK_VECTOR_BLEND

    @classmethod
    def from_json(cls, repo_root: Path, cfg: dict[str, Any], *, examples_parquet: Path) -> PoolExportConfig:
        from steam_review_ml.evaluation.retrieval_offline_eval import (
            METHOD_RAG_CHUNK_VECTOR_BLEND,
            METHOD_TWO_TOWER_V1,
        )

        pool_method = str(cfg.get("pool_method", METHOD_TWO_TOWER_V1))
        pipeline: RetrievalPipelineConfig
        if pool_method == METHOD_TWO_TOWER_V1:
            pipeline = TwoTowerPoolExportConfig.from_json(repo_root, cfg)
        elif pool_method == METHOD_RAG_CHUNK_VECTOR_BLEND:
            pipeline = RagPoolExportConfig.from_json(repo_root, cfg)
        else:
            raise ValueError(
                f"Unsupported pool_method for export: {pool_method!r}. "
                f"Supported: {METHOD_TWO_TOWER_V1!r}, {METHOD_RAG_CHUNK_VECTOR_BLEND!r}"
            )

        return cls(
            examples_parquet=examples_parquet,
            output_path=repo_root / require_str(cfg, "output_path"),
            split=str(cfg.get("split", "val")),
            k_retrieval=int(cfg.get("k_retrieval", 100)),
            min_review_chars=int(cfg.get("min_review_chars", 30)),
            artifact_dir=repo_root / str(cfg.get("artifact_dir", "artifacts/recs")),
            verbose=bool(cfg.get("verbose", True)),
            pipeline=pipeline,
        )
