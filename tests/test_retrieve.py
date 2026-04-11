"""Tests for ``steam_review_ml.recommender.retrieve``."""

from __future__ import annotations

from pathlib import Path

import pytest

from steam_review_ml.recommender.retrieve import ContentRetriever, default_repo_root


def test_default_repo_root_points_at_pyproject() -> None:
    root = default_repo_root()
    assert (root / "pyproject.toml").is_file()


def test_content_retriever_raises_on_missing_artifacts(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Missing recommender artifact"):
        ContentRetriever(artifact_dir=tmp_path)


@pytest.mark.skipif(
    not (default_repo_root() / "artifacts" / "recs" / "game_profile_embeddings.npz").is_file(),
    reason="recs_002 artifacts not present",
)
def test_content_retriever_loads_matrix_without_embedding() -> None:
    r = ContentRetriever()
    assert r.embedding_matrix.ndim == 2
    assert len(r.app_ids) == r.embedding_matrix.shape[0]
    assert len(r.index_frame) == r.embedding_matrix.shape[0]


@pytest.mark.skipif(
    not (default_repo_root() / "artifacts" / "recs" / "game_profile_embeddings.npz").is_file(),
    reason="recs_002 artifacts not present",
)
def test_top_k_raw_smoke() -> None:
    pytest.importorskip("tensorflow")
    pytest.importorskip("tensorflow_hub")
    r = ContentRetriever()
    df = r.top_k("I enjoy tactical RPGs with good story.", k=3)
    assert len(df) == 3
    assert "score" in df.columns
    assert "app_id" in df.columns
