"""Content-based retrieval: frozen USE query vector vs precomputed game profile matrix."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .math_utils import L2_EPS, l2_normalize


def default_repo_root() -> Path:
    """Return the directory containing ``pyproject.toml``, walking up from cwd."""
    here = Path.cwd().resolve()
    for directory in (here, *here.parents):
        if (directory / "pyproject.toml").is_file():
            return directory
    raise RuntimeError("Could not find repo root (no pyproject.toml in parents).")


class ContentRetriever:
    """Load ``recs_002`` artifacts and run top-K cosine retrieval (same logic as ``recs_003``).

    TensorFlow Hub is loaded lazily on first embed. Install TensorFlow + Hub (e.g. conda-forge), then
    ``pip install -e .``; pip-only envs may use ``pip install -e '.[recs-pip]'``.
    """

    def __init__(self, artifact_dir: Path | None = None, *, repo_root: Path | None = None) -> None:
        root = repo_root or default_repo_root()
        if artifact_dir is not None:
            self._artifact_dir = Path(artifact_dir)
        else:
            self._artifact_dir = root / "artifacts" / "recs"

        self._embed_fn = None
        self._load_artifacts()

    # --- artifact I/O ---

    def _artifact_paths(self) -> tuple[Path, Path, Path]:
        """
        Resolve artifact paths across supported layouts.

        Supported:
        - Legacy: artifacts/recs/game_profile_*
        - Current: artifacts/recs/embeddings/game_profile/default/game_profile_*
        """
        base = self._artifact_dir
        candidate_bases = [
            base,
            base / "embeddings" / "game_profile" / "default",
        ]
        for candidate in candidate_bases:
            npz = candidate / "game_profile_embeddings.npz"
            idx = candidate / "game_profile_embedding_index.parquet"
            meta = candidate / "game_profile_embedding_meta.json"
            if npz.is_file() and idx.is_file() and meta.is_file():
                return npz, idx, meta

        # Fall back to legacy path tuple so error message stays explicit.
        return (
            base / "game_profile_embeddings.npz",
            base / "game_profile_embedding_index.parquet",
            base / "game_profile_embedding_meta.json",
        )

    def _load_artifacts(self) -> None:
        npz_path, idx_path, meta_path = self._artifact_paths()
        for path in (npz_path, idx_path, meta_path):
            if not path.is_file():
                raise FileNotFoundError(f"Missing recommender artifact: {path}")

        self._X, self._app_ids = self._read_embedding_npz(npz_path)
        self._idx_df = self._read_index_aligned_with_matrix(idx_path, self._X, self._app_ids)
        self._tfhub_url, self._max_chars = self._read_hub_meta(meta_path)

    @staticmethod
    def _read_embedding_npz(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
        with np.load(npz_path) as z:
            matrix = np.asarray(z["embeddings"], dtype=np.float32)
            app_ids = np.asarray(z["app_id"], dtype=np.int64)
        return matrix, app_ids

    @staticmethod
    def _read_index_aligned_with_matrix(
        idx_path: Path,
        embedding_matrix: np.ndarray,
        npz_app_ids: np.ndarray,
    ) -> pd.DataFrame:
        index_df = pd.read_parquet(idx_path)
        n_rows = len(index_df)
        n_embed = embedding_matrix.shape[0]
        if n_rows != n_embed:
            raise ValueError(
                f"Index row count ({n_rows}) != embedding row count ({n_embed})"
            )
        index_app_ids = index_df["app_id"].to_numpy()
        if not np.array_equal(index_app_ids, npz_app_ids):
            raise ValueError("game_profile_embedding_index.parquet app_id order does not match npz")
        return index_df

    @staticmethod
    def _read_hub_meta(meta_path: Path) -> tuple[str, int | None]:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        model_url = meta["model_name"]
        max_chars = meta.get("max_chars_per_review")
        return model_url, max_chars

    # --- embedding ---

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
        """Run the same Hub model as ``recs_002`` / ``recs_003``; return L2-normalized query vector."""
        self._ensure_embedder()
        cleaned = (text or "").strip()
        if self._max_chars is not None:
            cleaned = cleaned[: int(self._max_chars)]
        raw_embedding = self._embed_fn([cleaned])
        return l2_normalize(raw_embedding)

    # --- retrieval ---

    def _text_to_embed(
        self,
        query_text: str,
        *,
        structured: bool,
        context: str | None,
    ) -> str:
        """Return the string that should be passed to the Hub model."""
        from .preferences import build_embedding_input, extract_preferences

        raw = (query_text or "").strip()
        if not structured:
            return raw
        context_text = raw if context is None else context
        prefs = extract_preferences(raw)
        return build_embedding_input(prefs, context_text)

    @staticmethod
    def _row_indices_top_k(
        similarities: np.ndarray,
        k: int,
        app_ids_per_row: np.ndarray,
        exclude_app_ids: set[int] | None,
    ) -> np.ndarray:
        """Row indices into ``embedding_matrix`` / ``index_frame``, best-first, length ``k`` (or fewer)."""
        n = len(similarities)
        k = min(int(k), n)

        if not exclude_app_ids:
            # argpartition is O(n) vs full sort O(n log n); we only need the top k.
            candidate_rows = np.argpartition(-similarities, k - 1)[:k]
            order_within = np.argsort(-similarities[candidate_rows])
            return candidate_rows[order_within].astype(np.int64)

        # Exclusions break argpartition (we may need more than k candidates), so scan sorted order.
        sorted_rows = np.argsort(-similarities)
        chosen: list[int] = []
        for row in sorted_rows:
            app_id = int(app_ids_per_row[int(row)])
            if app_id in exclude_app_ids:
                continue
            chosen.append(int(row))
            if len(chosen) >= k:
                break
        return np.asarray(chosen, dtype=np.int64)

    def _relevant_history_vectors(
        self,
        query_vector: np.ndarray,
        history_texts: list[str] | None,
        *,
        min_similarity: float,
    ) -> list[tuple[float, np.ndarray]]:
        if not history_texts:
            return []
        cleaned_history = [str(text).strip() for text in history_texts if str(text).strip()]
        if not cleaned_history:
            return []

        vectors_with_sim: list[tuple[float, np.ndarray]] = []
        for text in cleaned_history:
            history_vector = self.embed_text(text)
            similarity = float(np.dot(query_vector, history_vector))
            if similarity >= float(min_similarity):
                vectors_with_sim.append((similarity, history_vector))
        return vectors_with_sim

    def _blend_with_history(
        self,
        query_vector: np.ndarray,
        vectors_with_sim: list[tuple[float, np.ndarray]],
        *,
        alpha: float,
        history_top_k: int,
    ) -> np.ndarray:
        if not vectors_with_sim:
            return query_vector

        vectors_with_sim.sort(key=lambda item: item[0], reverse=True)
        selected = vectors_with_sim[: max(1, int(history_top_k))]
        sims = np.asarray([max(0.0, sim) for sim, _ in selected], dtype=np.float32)
        vecs = np.stack([vec for _, vec in selected], axis=0).astype(np.float32)
        if float(sims.sum()) > L2_EPS:
            history_vector = l2_normalize((vecs * sims[:, None]).sum(axis=0))
        else:
            history_vector = l2_normalize(vecs.mean(axis=0))
        return l2_normalize((1.0 - alpha) * query_vector + alpha * history_vector)

    def top_k(
        self,
        query_text: str,
        k: int = 10,
        *,
        structured: bool = False,
        context: str | None = None,
        exclude_app_ids: set[int] | None = None,
        history_texts: list[str] | None = None,
        history_blend_alpha: float = 0.0,
        history_top_k: int = 3,
        history_min_similarity: float = 0.2,
    ) -> pd.DataFrame:
        """Rank indexed games by dot product (cosine, since vectors are L2-normalized).

        Parameters
        ----------
        query_text
            Raw user text. If ``structured`` is True, this is passed through
            ``extract_preferences`` / ``build_embedding_input``.
        k
            Number of rows to return (after exclusions).
        structured
            If True, embed the rules-based structured string instead of raw ``query_text``.
        context
            Optional second argument to ``build_embedding_input`` (defaults to ``query_text``).
        exclude_app_ids
            e.g. mask the query game so it never appears in results.
        history_texts
            Optional prior user review texts to blend with the query embedding.
        history_blend_alpha
            Blend weight in [0, 1] for history contribution. Default 0 keeps behavior unchanged.
        history_top_k
            Maximum number of prior reviews to blend after relevance filtering.
        history_min_similarity
            Keep prior reviews with cosine(query, review) >= this threshold to avoid unrelated history.
        """
        to_embed = self._text_to_embed(query_text, structured=structured, context=context)
        query_vector = self.embed_text(to_embed)
        alpha = float(np.clip(history_blend_alpha, 0.0, 1.0))
        if alpha > 0.0:
            relevant_vectors = self._relevant_history_vectors(
                query_vector,
                history_texts,
                min_similarity=history_min_similarity,
            )
            query_vector = self._blend_with_history(
                query_vector,
                relevant_vectors,
                alpha=alpha,
                history_top_k=history_top_k,
            )

        similarities = (self._X @ query_vector).astype(np.float32)

        row_indices = self._row_indices_top_k(
            similarities,
            k,
            self._app_ids,
            exclude_app_ids,
        )
        out = self._idx_df.iloc[row_indices].copy()
        out["score"] = similarities[row_indices]
        return out.reset_index(drop=True)

    @property
    def embedding_matrix(self) -> np.ndarray:
        """Shape ``(n_games, dim)`` — same order as ``index_frame`` / ``app_ids``."""
        return self._X

    @property
    def app_ids(self) -> np.ndarray:
        return self._app_ids

    @property
    def index_frame(self) -> pd.DataFrame:
        return self._idx_df


# --- Offline-eval query fusion (fixed USE + catalog vectors; not trained two-tower models) ---

# Train-row keys for behavior pooling weights (aligned with recs_011 Candidate C).
BEHAVIOR_WEIGHT_PLAYTIME_KEYS: tuple[str, ...] = ("_norm_author__playtime_at_review",)

# Registry id for Candidate C; must match ``configs/recs_job_eval_retrieval.json`` methods entry.
METHOD_FUSION_C_RAW_PLUS_BEHAVIOR = "fusion_c_raw_plus_behavior"


def _extract_behavior_playtime_weight(row: dict[str, Any]) -> float:
    """Prefer pre-normalized playtime-at-review; zero if missing."""
    vals: list[float] = []
    for key in BEHAVIOR_WEIGHT_PLAYTIME_KEYS:
        v = row.get(key)
        if v is None:
            continue
        vals.append(max(0.0, float(v)))
    return float(max(vals)) if vals else 0.0


def _mean_train_review_text_embedding(retriever: ContentRetriever, ex: dict[str, Any]) -> np.ndarray:
    """Mean embedding of train review texts; mirrors recs_011 when no texts fall back to query."""
    texts = [
        str(r.get("text", "")).strip()
        for r in ex.get("train_review_rows", [])
        if str(r.get("text", "")).strip()
    ]
    if texts:
        vecs = np.stack([retriever.embed_text(t) for t in texts], axis=0).astype(np.float32)
        return l2_normalize(vecs.mean(axis=0)).astype(np.float32, copy=False)
    return np.asarray(retriever.embed_text(ex["query_text"]), dtype=np.float32)


def _weighted_mean_behavior_embedding(
    ex: dict[str, Any],
    *,
    embedding_matrix: np.ndarray,
    app_to_row: dict[int, int],
    fallback: np.ndarray,
) -> tuple[np.ndarray, bool]:
    """Weighted mean catalog vectors for train apps using playtime weights; parity with recs_011."""
    rows = ex.get("train_review_rows", [])
    weighted_vecs: list[np.ndarray] = []
    weights: list[float] = []
    for r in rows:
        aid = int(r.get("app_id", -1))
        row_idx = app_to_row.get(aid)
        if row_idx is None:
            continue
        w = _extract_behavior_playtime_weight(r)
        if w <= 0.0:
            continue
        weighted_vecs.append(embedding_matrix[row_idx].astype(np.float32, copy=False))
        weights.append(w)
    if weighted_vecs:
        mat = np.stack(weighted_vecs, axis=0)
        w_arr = np.asarray(weights, dtype=np.float32)
        vec = (mat * w_arr[:, None]).sum(axis=0) / float(np.maximum(w_arr.sum(), 1e-12))
        return l2_normalize(vec).astype(np.float32, copy=False), True
    support_ids = sorted(
        {int(r.get("app_id")) for r in rows if int(r.get("app_id", -1)) in app_to_row}
    )
    if support_ids:
        mat = np.stack([embedding_matrix[app_to_row[a]] for a in support_ids], axis=0).astype(np.float32)
        return l2_normalize(mat.mean(axis=0)).astype(np.float32, copy=False), False
    return np.asarray(fallback, dtype=np.float32), False


def fusion_c_raw_plus_behavior_query_vector(
    retriever: ContentRetriever,
    ex: dict[str, Any],
    *,
    embedding_matrix: np.ndarray,
    app_to_row: dict[int, int],
) -> np.ndarray:
    """Candidate C: ``l2_normalize(embed(query_text) + weighted_mean_behavior_embed)``.

    Same recipe as ``notebooks/retrieval_ranking/recs_011_eval_retrieval_two_tower_comparison.ipynb``.
    Uses ``BEHAVIOR_WEIGHT_PLAYTIME_KEYS`` on train-history rows for weights, then fuses with the
    session vector before a final ``l2_normalize``.
    """
    q_session = np.asarray(retriever.embed_text(ex["query_text"]), dtype=np.float32)
    u_rev = _mean_train_review_text_embedding(retriever, ex)
    u_behavior, _used_w = _weighted_mean_behavior_embedding(
        ex,
        embedding_matrix=embedding_matrix,
        app_to_row=app_to_row,
        fallback=u_rev,
    )
    return l2_normalize(q_session + u_behavior).astype(np.float32, copy=False)
