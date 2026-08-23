"""Job configs for the RAG extension chunking + embedding jobs (Stages 1-2).

Mirrors the frozen-dataclass + ``from_json(repo_root, cfg)`` shape already used by
``steam_review_ml.igdb.job_config`` (see ``docs/todo_job_config_boilerplate_refactor.md``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
