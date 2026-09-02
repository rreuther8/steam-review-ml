"""FastAPI routing tests for shipped v2a default and legacy raw path."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _patch_serving_log_path(monkeypatch, app_module, log_path: Path) -> None:
    """Redirect the app's serving log to ``log_path`` by wrapping ``load_serve_config``."""
    original = app_module.load_serve_config

    def _patched(*args, **kwargs):
        cfg = dict(original(*args, **kwargs))
        cfg["serving_log_path"] = str(log_path)
        return cfg

    monkeypatch.setattr(app_module, "load_serve_config", _patched)


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
    pytest.importorskip("tensorflow_hub")
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
    assert payload["recommender_method_id"] == "two_tower_v1_v2a_embed_query_logpop_blend"


def test_recommendations_v2a_does_not_block_on_explanation(monkeypatch) -> None:
    """Explanation generation moved to GET /explain -- /recommendations must not attach it."""
    pytest.importorskip("fastapi")
    pytest.importorskip("tensorflow")
    pytest.importorskip("tensorflow_hub")
    from fastapi.testclient import TestClient

    import steam_review_ml.api.app as app_module
    from steam_review_ml.recommender.serve_config import load_serve_config

    cfg = load_serve_config()
    tower = Path(cfg["two_tower_model_path"])
    if not tower.is_file():
        pytest.skip("two-tower checkpoint not present")

    def _fail_if_called():
        raise AssertionError("/recommendations must not load the explanation backend")

    monkeypatch.setattr(app_module, "_load_explanation_backend", lambda: _fail_if_called())

    with TestClient(app_module.create_app()) as client:
        r = client.get(
            "/recommendations",
            params={"q": "I love tactical RPGs", "exclude_app_id": 8930, "k": 3},
        )
    assert r.status_code == 200
    body = r.json()
    assert len(body) == 3
    assert all("explanation" not in row for row in body)


def test_recommendations_v2a_logs_event(monkeypatch, tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("tensorflow")
    pytest.importorskip("tensorflow_hub")
    from fastapi.testclient import TestClient

    import steam_review_ml.api.app as app_module
    from steam_review_ml.recommender.serve_config import load_serve_config

    cfg = load_serve_config()
    tower = Path(cfg["two_tower_model_path"])
    if not tower.is_file():
        pytest.skip("two-tower checkpoint not present")

    log_path = tmp_path / "events.jsonl"
    _patch_serving_log_path(monkeypatch, app_module, log_path)

    with TestClient(app_module.create_app()) as client:
        r = client.get(
            "/recommendations",
            params={"q": "I love tactical RPGs", "exclude_app_id": 8930, "k": 3},
        )
    assert r.status_code == 200
    body = r.json()

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    event = json.loads(lines[0])
    assert event["event_type"] == "recommendation"
    assert event["query_app_id"] == 8930
    assert event["query_text"] == "I love tactical RPGs"
    assert event["method_id"] == "two_tower_v1_v2a_embed_query_logpop_blend"
    assert event["duration_ms"] >= 0
    assert event["retrieve_ms"] >= 0
    assert event["rerank_ms"] >= 0
    assert [r["app_id"] for r in event["results"]] == [row["app_id"] for row in body]
    assert [r["rank"] for r in event["results"]] == [1, 2, 3]


def test_explain_returns_top_pick_explanation(monkeypatch) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("tensorflow")
    pytest.importorskip("tensorflow_hub")
    from fastapi.testclient import TestClient

    import steam_review_ml.api.app as app_module
    from steam_review_ml.recommender.serve_config import load_serve_config

    cfg = load_serve_config()
    tower = Path(cfg["two_tower_model_path"])
    if not tower.is_file():
        pytest.skip("two-tower checkpoint not present")

    class _FakeBackend:
        def generate_explanation(self, query_text: str, recommended_text: str) -> str:
            return "fake explanation"

    monkeypatch.setattr(app_module, "_load_explanation_backend", lambda: _FakeBackend())
    monkeypatch.setattr(
        app_module,
        "build_candidate_text_lookup",
        lambda app_ids: {a: f"text for {a}" for a in app_ids},
    )

    with TestClient(app_module.create_app()) as client:
        r = client.get("/explain", params={"query_app_id": 8930, "rec_app_id": 42})
    assert r.status_code == 200
    assert r.json() == {"explanation": "fake explanation"}


def test_explain_logs_event_with_cache_hit_and_backend_available(monkeypatch, tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("tensorflow")
    pytest.importorskip("tensorflow_hub")
    from fastapi.testclient import TestClient

    import steam_review_ml.api.app as app_module
    from steam_review_ml.recommender.serve_config import load_serve_config

    cfg = load_serve_config()
    tower = Path(cfg["two_tower_model_path"])
    if not tower.is_file():
        pytest.skip("two-tower checkpoint not present")

    class _FakeBackend:
        def generate_explanation(self, query_text: str, recommended_text: str) -> str:
            return "fake explanation"

    monkeypatch.setattr(app_module, "_load_explanation_backend", lambda: _FakeBackend())
    monkeypatch.setattr(
        app_module,
        "build_candidate_text_lookup",
        lambda app_ids: {a: f"text for {a}" for a in app_ids},
    )
    log_path = tmp_path / "events.jsonl"
    _patch_serving_log_path(monkeypatch, app_module, log_path)

    with TestClient(app_module.create_app()) as client:
        client.get("/explain", params={"query_app_id": 8930, "rec_app_id": 42})
        client.get("/explain", params={"query_app_id": 8930, "rec_app_id": 42})

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    first, second = (json.loads(line) for line in lines)
    for event in (first, second):
        assert event["event_type"] == "explanation"
        assert event["query_app_id"] == 8930
        assert event["rec_app_id"] == 42
        assert event["explanation"] == "fake explanation"
        assert event["backend_available"] is True
        assert event["duration_ms"] >= 0
    assert first["cache_hit"] is False
    assert second["cache_hit"] is True


def test_explain_logs_backend_unavailable(monkeypatch, tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("tensorflow")
    pytest.importorskip("tensorflow_hub")
    from fastapi.testclient import TestClient

    import steam_review_ml.api.app as app_module
    from steam_review_ml.recommender.serve_config import load_serve_config

    cfg = load_serve_config()
    tower = Path(cfg["two_tower_model_path"])
    if not tower.is_file():
        pytest.skip("two-tower checkpoint not present")

    monkeypatch.setattr(app_module, "_load_explanation_backend", lambda: None)
    log_path = tmp_path / "events.jsonl"
    _patch_serving_log_path(monkeypatch, app_module, log_path)

    with TestClient(app_module.create_app()) as client:
        r = client.get("/explain", params={"query_app_id": 8930, "rec_app_id": 42})
    assert r.json() == {"explanation": None}

    event = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])
    assert event["explanation"] is None
    assert event["backend_available"] is False
    assert event["cache_hit"] is False


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
