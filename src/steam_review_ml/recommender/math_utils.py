"""Shared math helpers for recommender modules."""

from __future__ import annotations

import numpy as np

L2_EPS = 1e-12


def l2_normalize(vector: np.ndarray, *, eps: float = L2_EPS) -> np.ndarray:
    """Return a flattened float32 L2-normalized vector."""
    v = np.asarray(vector, dtype=np.float32).ravel()
    norm = float(np.linalg.norm(v))
    if norm <= eps:
        return v
    return (v / norm).astype(np.float32)
