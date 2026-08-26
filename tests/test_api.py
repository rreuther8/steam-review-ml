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


def test_health_reports_recommender_method() -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("tensorflow")
    pytest.importorskip("chromadb")
    pytest.importorskip("sentence_transformers")
    from fastapi.testclient import TestClient

    from steam_review_ml.api.app import create_app
    from steam_review_ml.recommender.serve_config import load_serve_config

    cfg = load_serve_config()
    chroma_dir = Path(cfg["rag_chroma_persist_dir"])
    if not chroma_dir.is_dir():
        pytest.skip("RAG Chroma index not present")

    with TestClient(create_app()) as client:
        r = client.get("/health")
    assert r.status_code == 200
    payload = r.json()
    assert payload["status"] == "ok"
    assert payload["default_method"] == "v2a"
    assert payload["recommender_method_id"] == "two_tower_v1_v2a_embed_query_logpop_blend"
    assert payload["rag_variant"] == "any_polarity__log_weighted"


def test_recommendations_v2a_attaches_explanation_to_top_pick_only(monkeypatch) -> None:
    """Uses a fake backend (no local LLM) to check wiring: only row 0 gets an explanation."""
    pytest.importorskip("fastapi")
    pytest.importorskip("tensorflow")
    pytest.importorskip("chromadb")
    pytest.importorskip("sentence_transformers")
    from fastapi.testclient import TestClient

    import steam_review_ml.api.app as app_module
    from steam_review_ml.recommender.serve_config import load_serve_config

    cfg = load_serve_config()
    chroma_dir = Path(cfg["rag_chroma_persist_dir"])
    if not chroma_dir.is_dir():
        pytest.skip("RAG Chroma index not present")

    class _FakeBackend:
        def generate_explanation(self, query_text: str, recommended_text: str) -> str:
            return "fake explanation"

    monkeypatch.setattr(app_module, "_load_explanation_backend", lambda: _FakeBackend())

    with TestClient(app_module.create_app()) as client:
        r = client.get(
            "/recommendations",
            params={"q": "I love tactical RPGs", "exclude_app_id": 8930, "k": 3},
        )
    assert r.status_code == 200
    body = r.json()
    assert body[0]["explanation"] == "fake explanation"
    assert all("explanation" not in row for row in body[1:])


def test_explain_top_pick_caches_by_query_and_rec_pair(monkeypatch) -> None:
    import steam_review_ml.api.app as app_module
    from steam_review_ml.api.app import _explain_top_pick

    monkeypatch.setattr(
        app_module,
        "build_candidate_text_lookup",
        lambda app_ids: {a: f"text for {a}" for a in app_ids},
    )

    call_count = {"n": 0}

    class _FakeBackend:
        def generate_explanation(self, query_text: str, recommended_text: str) -> str:
            call_count["n"] += 1
            return f"explanation #{call_count['n']}"

    cache: dict[tuple[int, int], str] = {}
    backend = _FakeBackend()

    first = _explain_top_pick(backend, cache, query_app_id=8930, rec_app_id=42)
    second = _explain_top_pick(backend, cache, query_app_id=8930, rec_app_id=42)
    different_pair = _explain_top_pick(backend, cache, query_app_id=8930, rec_app_id=99)

    assert first == second == "explanation #1"
    assert different_pair == "explanation #2"
    assert call_count["n"] == 2


def test_explain_top_pick_returns_none_without_backend() -> None:
    from steam_review_ml.api.app import _explain_top_pick

    assert _explain_top_pick(None, {}, query_app_id=1, rec_app_id=2) is None
