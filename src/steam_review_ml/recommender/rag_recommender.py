"""Retrieve with the Stage 3 RAG chunk pipeline (Chroma), then rerank with a registered pool reranker."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from steam_review_ml.evaluation.v2a_metadata_ranker import (
    METHOD_RAG_CHUNK_V1_VECTOR_BLEND_QUERY_V2A_EMBED_QUERY_LOGPOP_BLEND,
)
from steam_review_ml.recommender.recommender_base import Recommender
from steam_review_ml.recommender.retrieve import default_repo_root
from steam_review_ml.recommender.serve_config import load_serve_config

DEFAULT_RAG_VARIANT = "any_polarity__log_weighted"
DEFAULT_RAG_QUERY_BLEND_WEIGHT = 0.5
# RAGRecommender's own default -- not the shared `default_method` config key, which now
# names the API's two_tower_v1 default (see docs/retrieval_decision_log.md 2026-08-28).
DEFAULT_RAG_METHOD = METHOD_RAG_CHUNK_V1_VECTOR_BLEND_QUERY_V2A_EMBED_QUERY_LOGPOP_BLEND


class RAGRecommender(Recommender):
    """Two-stage recommend: ``rag_chunk_v1_vector_blend_query`` retrieve @k_retrieval → pool rerank @k_final."""

    def __init__(
        self,
        *,
        method_id: str = DEFAULT_RAG_METHOD,
        rag_chroma_persist_dir: Path | str | None = None,
        rag_variant: str = DEFAULT_RAG_VARIANT,
        rag_query_blend_weight: float = DEFAULT_RAG_QUERY_BLEND_WEIGHT,
        repo_root: Path | None = None,
        artifact_dir: Path | None = None,
        igdb_enriched_path: Path | str | None = None,
        k_retrieval: int = 100,
        k_final: int = 10,
        min_review_chars: int = 30,
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
        from steam_review_ml.recommender.chroma_retrieve import ChromaGameProfileRetriever

        persist_dir = Path(rag_chroma_persist_dir) if rag_chroma_persist_dir else None
        self._rag_variant = str(rag_variant)
        self._rag_query_blend_weight = float(rag_query_blend_weight)
        self._rag_retriever = ChromaGameProfileRetriever(
            variant=self._rag_variant,
            chroma_persist_dir=persist_dir,
            repo_root=self._repo_root,
        )
        self._description_by_app_id = self._rag_retriever.load_all_description_texts()

    @classmethod
    def from_serve_config(
        cls,
        config_path: Path | str | None = None,
        *,
        repo_root: Path | None = None,
        artifact_dir: Path | None = None,
    ) -> RAGRecommender:
        cfg = load_serve_config(config_path, repo_root=repo_root)
        root = repo_root or default_repo_root()
        return cls(
            method_id=DEFAULT_RAG_METHOD,
            rag_chroma_persist_dir=cfg.get("rag_chroma_persist_dir"),
            rag_variant=str(cfg.get("rag_variant", DEFAULT_RAG_VARIANT)),
            rag_query_blend_weight=float(
                cfg.get("rag_query_blend_weight", DEFAULT_RAG_QUERY_BLEND_WEIGHT)
            ),
            repo_root=root,
            artifact_dir=artifact_dir,
            igdb_enriched_path=cfg.get("igdb_enriched_path"),
            k_retrieval=int(cfg.get("k_retrieval", 100)),
            k_final=int(cfg.get("k_final", 10)),
        )

    @property
    def rag_variant(self) -> str:
        return self._rag_variant

    def _score_catalog(self, query_text: str, *, query_app_id: int) -> np.ndarray:
        description = self._description_by_app_id.get(int(query_app_id), "")
        query_vector = self._rag_retriever.embed_query_vector_blend(
            query_text, description, blend_weight=self._rag_query_blend_weight
        )
        return self._rag_retriever.score_against_catalog(
            query_vector, query_app_id=int(query_app_id), app_ids=self._app_ids
        )
