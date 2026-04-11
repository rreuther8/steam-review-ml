"""Content-based retrieval: frozen USE query vector vs precomputed game profile matrix."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# Vectors this small are treated as zero to avoid divide-by-zero in L2 normalize.
_L2_EPS = 1e-12


def default_repo_root() -> Path:
    """Return the directory containing ``pyproject.toml``, walking up from cwd."""
    here = Path.cwd().resolve()
    for directory in (here, *here.parents):
        if (directory / "pyproject.toml").is_file():
            return directory
    raise RuntimeError("Could not find repo root (no pyproject.toml in parents).")


class ContentRetriever:
    """Load ``recs_002`` artifacts and run top‑K cosine retrieval (same logic as ``recs_003``).

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
        base = self._artifact_dir
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

    @staticmethod
    def _l2_normalize(vector: np.ndarray) -> np.ndarray:
        v = np.asarray(vector, dtype=np.float32).ravel()
        norm = np.linalg.norm(v)
        if norm <= _L2_EPS:
            return v
        return (v / norm).astype(np.float32)

    def embed_text(self, text: str) -> np.ndarray:
        """Run the same Hub model as ``recs_002`` / ``recs_003``; return L2-normalized query vector."""
        self._ensure_embedder()
        cleaned = (text or "").strip()
        if self._max_chars is not None:
            cleaned = cleaned[: int(self._max_chars)]
        raw_embedding = self._embed_fn([cleaned])
        return self._l2_normalize(raw_embedding)

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

    def top_k(
        self,
        query_text: str,
        k: int = 10,
        *,
        structured: bool = False,
        context: str | None = None,
        exclude_app_ids: set[int] | None = None,
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
        """
        to_embed = self._text_to_embed(query_text, structured=structured, context=context)
        query_vector = self.embed_text(to_embed)
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
