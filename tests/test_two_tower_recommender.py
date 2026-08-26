"""Tests for the two-tower + v2a serve path (``TwoTowerRecommender``).

No longer wired to ``configs/recs_serve.json`` (that now configures ``RAGRecommender``, the
shipped default — see ``test_rag_recommender.py``) or reachable via ``from_serve_config`` in
production; constructed directly against the same checkpoint path
``configs/recs_job_eval_offline.json`` already uses for offline two-tower eval.
"""

from __future__ import annotations

import pytest

from steam_review_ml.recommender.retrieve import default_repo_root
from steam_review_ml.recommender.two_tower_recommender import TwoTowerRecommender

_TWO_TOWER_MODEL_PATH = "artifacts/recs/towers/val_dev_12k_v1/updated_user__updated_profile200_item.keras"
_IGDB_ENRICHED_PATH = "artifacts/igdb/igdb_games__enriched.parquet"


def _has_serve_artifacts() -> bool:
    root = default_repo_root()
    tower = root / _TWO_TOWER_MODEL_PATH
    igdb = root / _IGDB_ENRICHED_PATH
    embeddings = (
        root
        / "artifacts"
        / "recs"
        / "embeddings"
        / "game_profile"
        / "default"
        / "game_profile_embeddings.npz"
    )
    legacy = root / "artifacts" / "recs" / "game_profile_embeddings.npz"
    return tower.is_file() and igdb.is_file() and (embeddings.is_file() or legacy.is_file())


@pytest.mark.skipif(not _has_serve_artifacts(), reason="serve stack artifacts not present")
def test_two_tower_recommender_recommend_smoke() -> None:
    pytest.importorskip("tensorflow")
    pytest.importorskip("tensorflow_hub")
    root = default_repo_root()
    rec = TwoTowerRecommender(
        two_tower_model_path=root / _TWO_TOWER_MODEL_PATH,
        repo_root=root,
        igdb_enriched_path=root / _IGDB_ENRICHED_PATH,
    )
    # Civilization V app_id from catalog smoke tests
    df = rec.recommend(
        "Great strategy game with deep mechanics.",
        query_app_id=8930,
    )
    assert len(df) == rec.k_final
    assert "score" in df.columns
    assert "app_id" in df.columns
    assert 8930 not in set(df["app_id"].astype(int).tolist())
    assert rec.method_id == "two_tower_v1_v2a_embed_query_logpop_blend"
