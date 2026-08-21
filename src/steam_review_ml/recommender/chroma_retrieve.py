"""Chroma-backed game profile retriever (Stage 3 of the RAG extension plan).

Analogous to ``ContentRetriever`` (retrieve.py), but queries the ``game_profiles``
Chroma collection built by Stage 2 instead of a static ``.npz`` cosine-sim matrix.
Chroma stores the vectors; this class still owns query-text embedding (same TF Hub
USE model), same as ``ContentRetriever`` owns it for the static-matrix path.
"""

from __future__ import annotations

from pathlib import Path

import chromadb
import numpy as np
import pandas as pd
from chromadb.api import ClientAPI

from steam_review_ml.recommender.math_utils import l2_normalize
from steam_review_ml.recommender.retrieve import default_repo_root

DEFAULT_CHROMA_PERSIST_DIR = "artifacts/recs/embeddings/game_chunks/chroma"
DEFAULT_GAME_PROFILES_COLLECTION = "game_profiles"
DEFAULT_REVIEW_CHUNKS_COLLECTION = "game_review_chunks"
DEFAULT_TFHUB_URL = "https://tfhub.dev/google/universal-sentence-encoder/4"

# Sentinel for a catalog row Chroma didn't return for this variant (the query's own
# app, or a game skipped from this variant during Stage 2 pooling) -- never ranked.
UNSCORED_SENTINEL = float("-inf")


class ChromaGameProfileRetriever:
    """Query one fixed pooling ``variant`` of the Stage 2 ``game_profiles`` collection."""

    def __init__(
        self,
        *,
        variant: str,
        chroma_persist_dir: Path | None = None,
        collection_name: str = DEFAULT_GAME_PROFILES_COLLECTION,
        review_chunks_collection_name: str = DEFAULT_REVIEW_CHUNKS_COLLECTION,
        tfhub_url: str = DEFAULT_TFHUB_URL,
        repo_root: Path | None = None,
    ) -> None:
        root = repo_root or default_repo_root()
        self._persist_dir = chroma_persist_dir or (root / DEFAULT_CHROMA_PERSIST_DIR)
        if not self._persist_dir.is_dir():
            raise FileNotFoundError(
                f"Missing Chroma persist dir: {self._persist_dir}. "
                "Run scripts/recs_job_game_chunk_embeddings.py first."
            )
        self._variant = variant
        self._tfhub_url = tfhub_url
        self._embed_fn = None

        client: ClientAPI = chromadb.PersistentClient(path=str(self._persist_dir))
        self._collection = client.get_collection(collection_name)
        if self._collection.count() == 0:
            raise RuntimeError(f"Chroma collection '{collection_name}' is empty.")
        self._review_chunks_collection = client.get_collection(review_chunks_collection_name)

        sample = self._collection.get(where={"variant": variant}, limit=1)
        if not sample["ids"]:
            raise ValueError(f"No rows found for variant={variant!r} in '{collection_name}'.")

    # --- embedding (mirrors ContentRetriever's lazy TF Hub load) ---

    def _ensure_embedder(self) -> None:
        if self._embed_fn is not None:
            return
        try:
            import os

            os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
            import tensorflow as tf
            import tensorflow_hub as hub
        except ImportError as e:
            raise ImportError(
                "TensorFlow Hub is required for retrieval. Install TF + Hub (conda-forge recommended), "
                "or pip install -e '.[recs-pip]' in a pip-only env."
            ) from e
        for gpu in tf.config.list_physical_devices("GPU"):
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                pass
        self._embed_fn = hub.load(self._tfhub_url)

    def embed_text(self, text: str) -> np.ndarray:
        """Run the same Hub model as the rest of the pipeline; return an L2-normalized vector."""
        self._ensure_embedder()
        cleaned = (text or "").strip()
        raw_embedding = self._embed_fn([cleaned])
        return l2_normalize(raw_embedding)

    # --- retrieval ---

    def score_against_catalog(
        self,
        query_vector: np.ndarray,
        *,
        query_app_id: int,
        app_ids: np.ndarray,
    ) -> np.ndarray:
        """Full-length score vector aligned to ``app_ids`` (eval-contract shape).

        Self-exclusion happens via the Chroma ``where`` filter, not a post-hoc mask.
        Any ``app_id`` in the catalog that this variant has no row for (self, or a
        game skipped in Stage 2 pooling for this variant) gets ``UNSCORED_SENTINEL``.
        """
        result = self._collection.query(
            query_embeddings=[np.asarray(query_vector, dtype=np.float32).tolist()],
            n_results=len(app_ids),
            where={
                "$and": [
                    {"variant": self._variant},
                    {"app_id": {"$ne": int(query_app_id)}},
                ]
            },
        )
        score_by_app_id = {
            int(meta["app_id"]): -float(dist)
            for meta, dist in zip(result["metadatas"][0], result["distances"][0])
        }

        app_id_to_row = {int(a): i for i, a in enumerate(app_ids)}
        scores = np.full(len(app_ids), UNSCORED_SENTINEL, dtype=np.float32)
        for app_id, score in score_by_app_id.items():
            row = app_id_to_row.get(app_id)
            if row is not None:
                scores[row] = score
        return scores

    def top_k(self, query_text: str, k: int = 10, *, query_app_id: int | None = None) -> pd.DataFrame:
        """Standalone/demo usage: rank this variant by query text (no eval-contract catalog needed)."""
        query_vector = self.embed_text(query_text)
        where: dict = {"variant": self._variant}
        if query_app_id is not None:
            where = {"$and": [{"variant": self._variant}, {"app_id": {"$ne": int(query_app_id)}}]}
        result = self._collection.query(
            query_embeddings=[query_vector.tolist()], n_results=k, where=where
        )
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]
        return pd.DataFrame(
            {
                "app_id": [int(m["app_id"]) for m in metadatas],
                "score": [-float(d) for d in distances],
                "recommended_rate": [m.get("recommended_rate") for m in metadatas],
                "n_reviews_pooled": [m.get("n_reviews_pooled") for m in metadatas],
            }
        )

    def load_all_description_texts(self) -> dict[int, str]:
        """One bulk fetch of every game's description chunk text, keyed by ``app_id``.

        Backed by ``game_review_chunks`` (Stage 1/2 already stored this text there) --
        avoids re-reading the IGDB parquet or doing one Chroma call per eval example.
        """
        result = self._review_chunks_collection.get(
            where={"chunk_type": "description"}, include=["metadatas", "documents"]
        )
        return {
            int(meta["app_id"]): doc
            for meta, doc in zip(result["metadatas"], result["documents"])
        }

    @property
    def variant(self) -> str:
        return self._variant
