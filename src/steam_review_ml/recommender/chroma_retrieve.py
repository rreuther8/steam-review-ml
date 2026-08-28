"""Chroma-backed game profile retriever (Stage 3 of the RAG extension plan).

Analogous to ``ContentRetriever`` (retrieve.py), but queries the ``game_profiles``
Chroma collection built by Stage 2 instead of a static ``.npz`` cosine-sim matrix.
Chroma stores the vectors; this class still owns query-text embedding (same
sentence-transformers model used to build the index), same as ``ContentRetriever``
owns it for the static-matrix path.
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
DEFAULT_EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# BGE's own convention: prepend this instruction to the query side only (not
# passages/chunks) -- see docs/plans/rag_extension_plan.md decision #4.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Sentinel for a catalog row Chroma didn't return for this variant (the query's own
# app, or a game skipped from this variant during Stage 2 pooling) -- never ranked.
UNSCORED_SENTINEL = float("-inf")


def _align_batch_query_results(
    metadatas: list[list[dict]],
    distances: list[list[float]],
    *,
    query_app_ids: np.ndarray,
    app_ids: np.ndarray,
) -> np.ndarray:
    """Reshape Chroma's own per-query result lists into one aligned score matrix.

    Pure function -- takes Chroma's already-computed distances as input and only restructures
    them; it does not recompute or approximate similarity itself. Masks each row's own
    ``query_app_id`` after the fact, since the batched call below drops the per-example
    ``$ne`` exclusion filter (one ``where`` clause now applies to the whole batch).
    """
    app_id_to_col = {int(a): i for i, a in enumerate(app_ids)}
    scores = np.full((len(metadatas), len(app_ids)), UNSCORED_SENTINEL, dtype=np.float32)
    for row, (metas, dists) in enumerate(zip(metadatas, distances)):
        for meta, dist in zip(metas, dists):
            col = app_id_to_col.get(int(meta["app_id"]))
            if col is not None:
                scores[row, col] = -float(dist)
    for row, qid in enumerate(query_app_ids):
        col = app_id_to_col.get(int(qid))
        if col is not None:
            scores[row, col] = UNSCORED_SENTINEL
    return scores


class ChromaGameProfileRetriever:
    """Query one fixed pooling ``variant`` of the Stage 2 ``game_profiles`` collection."""

    def __init__(
        self,
        *,
        variant: str,
        chroma_persist_dir: Path | None = None,
        collection_name: str = DEFAULT_GAME_PROFILES_COLLECTION,
        review_chunks_collection_name: str = DEFAULT_REVIEW_CHUNKS_COLLECTION,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL_NAME,
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
        self._embedding_model_name = embedding_model_name
        self._embed_fn = None

        client: ClientAPI = chromadb.PersistentClient(path=str(self._persist_dir))
        self._collection = client.get_collection(collection_name)
        if self._collection.count() == 0:
            raise RuntimeError(f"Chroma collection '{collection_name}' is empty.")
        self._review_chunks_collection = client.get_collection(review_chunks_collection_name)

        sample = self._collection.get(where={"variant": variant}, limit=1)
        if not sample["ids"]:
            raise ValueError(f"No rows found for variant={variant!r} in '{collection_name}'.")

    # --- embedding (mirrors ContentRetriever's lazy model load) ---

    def _ensure_embedder(self) -> None:
        if self._embed_fn is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required for retrieval. "
                "Install it via the 'rag' extra: pip install -e '.[rag]'"
            ) from e
        model = SentenceTransformer(self._embedding_model_name)
        self._embed_fn = model.encode

    def embed_text(self, text: str) -> np.ndarray:
        """Run the same embedding model as the rest of the pipeline; return an L2-normalized vector."""
        self._ensure_embedder()
        cleaned = (text or "").strip()
        raw_embedding = self._embed_fn([BGE_QUERY_PREFIX + cleaned])
        return l2_normalize(raw_embedding)

    def embed_query_vector_blend(
        self, review_text: str, description_text: str, *, blend_weight: float = 0.5
    ) -> np.ndarray:
        """Vector-blend alternative to text-concatenation (Ablation B): embed review and
        description separately, blend the two vectors -- see docs/plans/rag_extension_plan.md
        Stage 3, recs_026. Falls back to the review vector alone if there's no description.
        """
        review_vec = self.embed_text(review_text)
        cleaned_desc = (description_text or "").strip()
        if not cleaned_desc:
            return review_vec
        desc_vec = self.embed_text(description_text)
        return l2_normalize((1.0 - blend_weight) * review_vec + blend_weight * desc_vec)

    def embed_texts_batch(self, texts: list[str]) -> np.ndarray:
        """Batched sibling of ``embed_text`` -- one model call for many texts, not one per text.

        For bulk offline scoring (e.g. exporting retrieval pools for a large cohort), where
        calling ``embed_text`` per example would mean thousands of separate small model calls.
        """
        self._ensure_embedder()
        cleaned = [BGE_QUERY_PREFIX + (t or "").strip() for t in texts]
        raw = self._embed_fn(cleaned)
        return np.vstack([l2_normalize(row) for row in raw])

    def score_batch_against_catalog(
        self,
        query_vectors: np.ndarray,
        *,
        query_app_ids: np.ndarray,
        app_ids: np.ndarray,
        chroma_batch_size: int = 100,
    ) -> np.ndarray:
        """Batched sibling of ``score_against_catalog``: one Chroma call for many queries.

        Still goes through Chroma's own distance computation -- this does not reimplement
        similarity search locally. It only drops the per-example ``app_id != query_app_id``
        filter (masking that afterward instead, in ``_align_batch_query_results``) so many
        queries can share one ``where={"variant": ...}`` call, chunked by
        ``chroma_batch_size``. Chroma's SQLite backend hits a "too many SQL variables" error
        well before any request-size limit you'd expect -- empirically ~100-150 query vectors
        per call at this collection's embedding dim (384) and ``n_results`` (~315); 100 is a
        confirmed-safe default, not an arbitrary round number.
        """
        query_vectors = np.asarray(query_vectors, dtype=np.float32)
        query_app_ids = np.asarray(query_app_ids)
        n = query_vectors.shape[0]
        chunks: list[np.ndarray] = []
        for start in range(0, n, chroma_batch_size):
            end = min(start + chroma_batch_size, n)
            result = self._collection.query(
                query_embeddings=query_vectors[start:end].tolist(),
                n_results=len(app_ids),
                where={"variant": self._variant},
            )
            chunks.append(
                _align_batch_query_results(
                    result["metadatas"],
                    result["distances"],
                    query_app_ids=query_app_ids[start:end],
                    app_ids=app_ids,
                )
            )
        return np.vstack(chunks)

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
