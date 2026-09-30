"""Shared retrieve-then-rerank contract for production recommenders.

Every recommender here is the same two-stage shape: a backend-specific retrieval
score over the full catalog, then a shared pool-rerank stage keyed by ``app_id``
(popularity + IGDB taxonomy blend). Subclasses only differ in how they produce
that first full-catalog score vector.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
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

METHOD_V2A = METHOD_TWO_TOWER_V1_V2A_EMBED_QUERY_LOGPOP_BLEND


def _resolve_rerank_spec(method_id: str, *, igdb_enriched_path: str | None) -> PoolRerankSpec:
    specs = pool_rerank_registry()
    if method_id not in specs:
        raise ValueError(
            f"Unknown rerank method {method_id!r}; available pool rerankers: {sorted(specs)}"
        )
    spec = specs[method_id]
    if igdb_enriched_path:
        params = {**spec.params, "enriched_path": igdb_enriched_path}
        return replace(spec, params=params)
    return spec


class Recommender(ABC):
    """Two-stage recommend: backend-specific retrieval @k_retrieval → pool rerank @k_final."""

    def __init__(
        self,
        *,
        method_id: str = METHOD_V2A,
        repo_root: Path | None = None,
        artifact_dir: Path | None = None,
        igdb_enriched_path: Path | str | None = None,
        k_retrieval: int = 100,
        k_final: int = 10,
        min_review_chars: int = 30,
    ) -> None:
        from steam_review_ml.evaluation.retrieval_offline_eval import load_ranking_catalog_context

        self._repo_root = repo_root or default_repo_root()
        self._method_id = str(method_id)
        self._k_retrieval = int(k_retrieval)
        self._k_final = int(k_final)
        self._validate_k_bounds()
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
            self._method_id, igdb_enriched_path=self._igdb_enriched_path
        )

    def _validate_k_bounds(self) -> None:
        if self._k_final > self._k_retrieval:
            raise ValueError(
                f"k_final ({self._k_final}) cannot exceed k_retrieval ({self._k_retrieval}) — "
                "the rerank stage can only select from the retrieved pool"
            )

    @classmethod
    @abstractmethod
    def from_serve_config(
        cls,
        config_path: Path | str | None = None,
        *,
        repo_root: Path | None = None,
        artifact_dir: Path | None = None,
    ) -> Recommender:
        """Build from ``configs/recs_serve.json`` (or an explicit path); backend-specific keys."""

    @property
    def method_id(self) -> str:
        return self._method_id

    @property
    def retriever(self) -> ContentRetriever:
        return self._retriever

    @property
    def igdb_enriched_path(self) -> str | None:
        return self._igdb_enriched_path

    @property
    def k_retrieval(self) -> int:
        return self._k_retrieval

    @property
    def k_final(self) -> int:
        return self._k_final

    @abstractmethod
    def _score_catalog(self, query_text: str, *, query_app_id: int) -> np.ndarray:
        """Full-catalog score vector aligned to ``self._app_ids``; query app excluded/masked."""

    def recommend(self, query_text: str, *, query_app_id: int) -> pd.DataFrame:
        """Return top-``k_final`` catalog rows with rerank scores (query game excluded at retrieve).

        Per-stage timing (retrieve vs. rerank) is attached to the result via ``DataFrame.attrs``
        (``retrieve_ms``, ``rerank_ms``) rather than changing the return type -- this method's
        DataFrame contract is depended on by eval jobs, notebooks, and tests; ``.attrs`` is
        pandas's sanctioned side channel for exactly this kind of per-call metadata, and each
        call produces a fresh DataFrame/dict so there's no shared mutable state across requests.
        """
        k_out = self._k_final
        k_pool = self._k_retrieval

        t0 = time.perf_counter()
        base_scores = self._score_catalog(str(query_text), query_app_id=int(query_app_id))
        retrieve_ms = (time.perf_counter() - t0) * 1000

        retrieved_indices = np.argsort(-base_scores)[:k_pool]
        pool_apps = [int(self._app_ids[int(i)]) for i in retrieved_indices]
        pool_retr_scores = [float(base_scores[int(i)]) for i in retrieved_indices]

        t0 = time.perf_counter()
        rerank_scores = rerank_scores_on_pool(
            pool_apps,
            pool_retr_scores,
            self._rerank_spec,
            pop_row=self._pop_row,
            app_to_row=self._app_to_row,
            query_app_id=int(query_app_id),
        )
        rerank_ms = (time.perf_counter() - t0) * 1000

        top_pool_order = np.argsort(-np.asarray(rerank_scores, dtype=np.float64))[:k_out]
        selected_apps = [pool_apps[int(i)] for i in top_pool_order]
        selected_scores = [float(rerank_scores[int(i)]) for i in top_pool_order]

        idx_df = self._retriever.index_frame
        row_indices = [self._app_to_row[int(a)] for a in selected_apps]
        out = idx_df.iloc[row_indices].copy()
        out["score"] = selected_scores
        out = out.reset_index(drop=True)
        out.attrs["retrieve_ms"] = retrieve_ms
        out.attrs["rerank_ms"] = rerank_ms
        return out
