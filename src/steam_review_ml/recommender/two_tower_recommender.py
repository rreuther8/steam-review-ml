"""Retrieve with two_tower_v1, then rerank with a registered pool reranker (shipped v2a stack)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from steam_review_ml.recommender.recommender_base import METHOD_V2A, Recommender
from steam_review_ml.recommender.retrieve import default_repo_root
from steam_review_ml.recommender.serve_config import load_serve_config


class TwoTowerRecommender(Recommender):
    """Two-stage recommend: ``two_tower_v1`` retrieve @k_retrieval → pool rerank @k_final."""

    def __init__(
        self,
        *,
        method_id: str = METHOD_V2A,
        two_tower_model_path: Path | str,
        repo_root: Path | None = None,
        artifact_dir: Path | None = None,
        igdb_enriched_path: Path | str | None = None,
        k_retrieval: int = 100,
        k_final: int = 10,
        min_review_chars: int = 30,
        catalog_item_batch: int = 256,
    ) -> None:
        super().__init__(
            method_id=method_id,
            repo_root=repo_root,
            artifact_dir=artifact_dir,
            igdb_enriched_path=igdb_enriched_path,
            k_retrieval=k_retrieval,
            k_final=k_final,
            min_review_chars=min_review_chars,
        )
        from steam_review_ml.recommender.two_tower_score import (
            encode_query_vector,
            load_two_tower_model,
            precompute_catalog_item_vectors,
            score_catalog,
        )
        from steam_review_ml.recommender.two_tower_train import load_hub_settings

        self._two_tower_model_path = Path(two_tower_model_path)

        hub_url, hub_max_chars = load_hub_settings(self._retriever)
        self._hub_max_chars = hub_max_chars
        embed_dim = int(self._retriever.embedding_matrix.shape[1])
        self._model = load_two_tower_model(
            self._two_tower_model_path,
            hub_url=hub_url,
            n_items=len(self._retriever.app_ids),
            embed_dim=embed_dim,
        )
        self._item_vectors = precompute_catalog_item_vectors(
            self._model,
            len(self._retriever.app_ids),
            batch_size=int(catalog_item_batch),
        )
        self._encode_query_vector = encode_query_vector
        self._score_catalog_fn = score_catalog

    @classmethod
    def from_serve_config(
        cls,
        config_path: Path | str | None = None,
        *,
        repo_root: Path | None = None,
        artifact_dir: Path | None = None,
    ) -> TwoTowerRecommender:
        cfg = load_serve_config(config_path, repo_root=repo_root)
        root = repo_root or default_repo_root()
        return cls(
            method_id=str(cfg.get("default_method", METHOD_V2A)),
            two_tower_model_path=cfg["two_tower_model_path"],
            repo_root=root,
            artifact_dir=artifact_dir,
            igdb_enriched_path=cfg.get("igdb_enriched_path"),
            k_retrieval=int(cfg.get("k_retrieval", 100)),
            k_final=int(cfg.get("k_final", 10)),
        )

    @property
    def two_tower_model_path(self) -> Path:
        return self._two_tower_model_path

    def _score_catalog(self, query_text: str, *, query_app_id: int) -> np.ndarray:
        user_vector = self._encode_query_vector(
            self._model,
            query_text,
            max_chars=self._hub_max_chars,
        )
        mask_row = self._app_to_row.get(int(query_app_id))
        return self._score_catalog_fn(user_vector, self._item_vectors, mask_row=mask_row)
