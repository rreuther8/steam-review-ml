"""Tests for IGDB USE text embedding helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from steam_review_ml.igdb.constants import USE_EMBEDDING_FIELD_SUFFIX
from steam_review_ml.igdb.text_embeddings import embed_text_columns, use_embedding_column


def test_use_embedding_column_suffix() -> None:
    assert use_embedding_column("summary") == f"summary{USE_EMBEDDING_FIELD_SUFFIX}"


def test_embed_text_columns_adds_l2_normalized_vectors() -> None:
    df = pd.DataFrame(
        [
            {"app_id": 1, "summary": "A tactical shooter."},
            {"app_id": 2, "summary": "A cozy farming game."},
        ]
    )

    def fake_embed(texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), 4), dtype=np.float32)
        for i, text in enumerate(texts):
            out[i, 0] = float(len(text))
            out[i, 1] = float(sum(ord(c) for c in text[:3]))
        return out

    out = embed_text_columns(df, ["summary"], embed_fn=fake_embed, batch_size=8)

    assert "summary" in out.columns
    assert use_embedding_column("summary") in out.columns
    vec = np.asarray(out.loc[0, use_embedding_column("summary")], dtype=np.float32)
    assert vec.shape == (4,)
    assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-5)


def test_embed_text_columns_requires_field() -> None:
    df = pd.DataFrame([{"app_id": 1, "summary": "hello"}])
    with pytest.raises(KeyError, match="storyline"):
        embed_text_columns(df, ["storyline"], embed_fn=lambda texts: np.zeros((len(texts), 2)))
