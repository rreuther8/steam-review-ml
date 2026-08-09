"""FastAPI routing tests for shipped v2a default and legacy raw path."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_recommendations_v2a_requires_exclude_app_id() -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from steam_review_ml.api.app import create_app

    client = TestClient(create_app())
    r = client.get("/recommendations", params={"q": "I love tactical RPGs"})
    assert r.status_code == 422
    assert "exclude_app_id" in r.json()["detail"]


def test_recommendations_raw_does_not_require_exclude_app_id() -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("tensorflow")
    pytest.importorskip("tensorflow_hub")
    from fastapi.testclient import TestClient

    from steam_review_ml.api.app import create_app
    from steam_review_ml.recommender.retrieve import default_repo_root

    root = default_repo_root()
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
    if not embeddings.is_file() and not legacy.is_file():
        pytest.skip("recs_002 artifacts not present")

    client = TestClient(create_app())
    r = client.get(
        "/recommendations",
        params={"q": "I enjoy strategy games.", "method": "raw", "k": 3},
    )
    assert r.status_code == 200
    body = r.json()
    assert len(body) == 3
    assert "score" in body[0]


def test_health_reports_stacked_method() -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("tensorflow")
    from fastapi.testclient import TestClient

    from steam_review_ml.api.app import create_app
    from steam_review_ml.recommender.serve_config import load_serve_config

    cfg = load_serve_config()
    tower = Path(cfg["two_tower_model_path"])
    if not tower.is_file():
        pytest.skip("two-tower checkpoint not present")

    with TestClient(create_app()) as client:
        r = client.get("/health")
    assert r.status_code == 200
    payload = r.json()
    assert payload["status"] == "ok"
    assert payload["default_method"] == "v2a"
    assert payload["stacked_method_id"] == "two_tower_v1_v2a_embed_query_logpop_blend"
