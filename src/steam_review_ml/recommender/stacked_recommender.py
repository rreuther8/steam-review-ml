"""Retrieve with two_tower_v1, then rerank with a registered pool reranker (shipped v2a stack)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from steam_review_ml.evaluation.heuristic_ranker import (
    PoolRerankSpec,
    pool_rerank_registry,
    rerank_scores_on_pool,
)
from steam_review_ml.evaluation.v2a_metadata_ranker import (
    METHOD_TWO_TOWER_V1_V2A_EMBED_QUERY_LOGPOP_BLEND,
)
from steam_review_ml.recommender.retrieve import ContentRetriever, default_repo_root
from steam_review_ml.recommender.serve_config import load_serve_config

METHOD_V2A = METHOD_TWO_TOWER_V1_V2A_EMBED_QUERY_LOGPOP_BLEND


def _resolve_rerank_spec(
    method_id: str,
    *,
    igdb_enriched_path: str | None,
) -> PoolRerankSpec:
    specs = pool_rerank_registry()
    if method_id not in specs:
        raise ValueError(
            f"Unknown stacked method {method_id!r}; available pool rerankers: {sorted(specs)}"
        )
    spec = specs[method_id]
    if igdb_enriched_path:
        params = {**spec.params, "enriched_path": igdb_enriched_path}
        return replace(spec, params=params)
    return spec


class StackedRecommender:
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
        from steam_review_ml.evaluation.retrieval_offline_eval import load_ranking_catalog_context
        from steam_review_ml.recommender.two_tower_score import (
            encode_query_vector,
            load_two_tower_model,
            precompute_catalog_item_vectors,
            score_catalog,
        )
        from steam_review_ml.recommender.two_tower_train import load_hub_settings

        self._repo_root = repo_root or default_repo_root()
        self._method_id = str(method_id)
        self._k_retrieval = int(k_retrieval)
        self._k_final = int(k_final)
        self._validate_k_bounds()
        self._two_tower_model_path = Path(two_tower_model_path)
        self._igdb_enriched_path = str(igdb_enriched_path) if igdb_enriched_path else None

        self._retriever = ContentRetriever(artifact_dir=artifact_dir, repo_root=self._repo_root)
        catalog = load_ranking_catalog_context(
            repo_root=self._repo_root,
            min_review_chars=min_review_chars,
            artifact_dir=artifact_dir,
            retriever=self._retriever,
        )
        self._app_ids = catalog.app_ids
        self._app_to_row = catalog.app_to_row
        self._pop_row = catalog.pop_row
        self._rerank_spec = _resolve_rerank_spec(
            self._method_id,
            igdb_enriched_path=self._igdb_enriched_path,
        )

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
        self._score_catalog = score_catalog

    def _validate_k_bounds(self) -> None:
        if self._k_final > self._k_retrieval:
            raise ValueError(
                f"k_final ({self._k_final}) cannot exceed k_retrieval ({self._k_retrieval}) — "
                "the rerank stage can only select from the retrieved pool"
            )

    @classmethod
    def from_serve_config(
        cls,
        config_path: Path | str | None = None,
        *,
        repo_root: Path | None = None,
        artifact_dir: Path | None = None,
    ) -> StackedRecommender:
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
    def method_id(self) -> str:
        return self._method_id

    @property
    def retriever(self) -> ContentRetriever:
        return self._retriever

    @property
    def two_tower_model_path(self) -> Path:
        return self._two_tower_model_path

    @property
    def igdb_enriched_path(self) -> str | None:
        return self._igdb_enriched_path

    @property
    def k_retrieval(self) -> int:
        return self._k_retrieval

    @property
    def k_final(self) -> int:
        return self._k_final

    def recommend(
        self,
        query_text: str,
        *,
        query_app_id: int,
    ) -> pd.DataFrame:
        """Return top-``k_final`` catalog rows with rerank scores (query game masked at retrieve)."""
        k_out = self._k_final
        k_pool = self._k_retrieval

        user_vector = self._encode_query_vector(
            self._model,
            str(query_text),
            max_chars=self._hub_max_chars,
        )
        mask_row = self._app_to_row.get(int(query_app_id))
        base_scores = self._score_catalog(
            user_vector,
            self._item_vectors,
            mask_row=mask_row,
        )

        retrieved_indices = np.argsort(-base_scores)[:k_pool]
        pool_apps = [int(self._app_ids[int(i)]) for i in retrieved_indices]
        pool_retr_scores = [float(base_scores[int(i)]) for i in retrieved_indices]
        rerank_scores = rerank_scores_on_pool(
            pool_apps,
            pool_retr_scores,
            self._rerank_spec,
            pop_row=self._pop_row,
            app_to_row=self._app_to_row,
            query_app_id=int(query_app_id),
        )

        top_pool_order = np.argsort(-np.asarray(rerank_scores, dtype=np.float64))[:k_out]
        selected_apps = [pool_apps[int(i)] for i in top_pool_order]
        selected_scores = [float(rerank_scores[int(i)]) for i in top_pool_order]

        idx_df = self._retriever.index_frame
        row_indices = [self._app_to_row[int(a)] for a in selected_apps]
        out = idx_df.iloc[row_indices].copy()
        out["score"] = selected_scores
        return out.reset_index(drop=True)
