"""FastAPI routing tests for shipped v2a default and legacy raw path."""

from __future__ import annotations

import asyncio
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


def test_iterate_in_thread_yields_all_items_and_closes_on_early_exit() -> None:
    from steam_review_ml.api.app import _iterate_in_thread

    closed = {"flag": False}

    def _pieces():
        try:
            yield from ["a", "b", "c"]
        finally:
            closed["flag"] = True

    async def _take_all():
        return [item async for item in _iterate_in_thread(_pieces())]

    async def _take_first_then_stop():
        stream = _iterate_in_thread(_pieces())
        first = await stream.__anext__()
        await stream.aclose()  # what a client disconnect does to the SSE body
        return first

    assert asyncio.run(_take_all()) == ["a", "b", "c"]
    closed["flag"] = False
    assert asyncio.run(_take_first_then_stop()) == "a"
    assert closed["flag"] is True


class _FakeStreamingBackend:
    """Stands in for ``LlamaCppBackend``'s streaming methods; records each call."""

    def __init__(self, pieces: tuple[str, ...] = ("fake ", "explanation"), fail_after: int | None = None) -> None:
        self.pieces = pieces
        self.fail_after = fail_after
        self.calls: list[tuple] = []

    def stream_explanation(self, query_game_text: str, rec_game_text: str):
        self.calls.append(("game_to_game", query_game_text, rec_game_text))
        yield from self._pieces()

    def stream_review_explanation(self, review_text: str, query_game_text: str, rec_game_text: str):
        self.calls.append(("review_to_game", review_text, query_game_text, rec_game_text))
        yield from self._pieces()

    def _pieces(self):
        for i, piece in enumerate(self.pieces):
            if self.fail_after is not None and i == self.fail_after:
                raise RuntimeError("generation blew up")
            yield piece


def _parse_sse(body: str) -> list[tuple[str, dict]]:
    """``[(event_name, data_dict), ...]`` from a text/event-stream body."""
    events = []
    for block in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def _explain_app(monkeypatch, tmp_path: Path, backend):
    """App module with the tower checkpoint required, a fake explanation backend, stub IGDB text,
    and the serving log redirected to ``tmp_path``. Returns ``(app_module, log_path)``."""
    pytest.importorskip("fastapi")
    pytest.importorskip("tensorflow")
    pytest.importorskip("tensorflow_hub")
    import steam_review_ml.api.app as app_module
    from steam_review_ml.recommender.serve_config import load_serve_config

    if not Path(load_serve_config()["two_tower_model_path"]).is_file():
        pytest.skip("two-tower checkpoint not present")

    monkeypatch.setattr(app_module, "_load_explanation_backend", lambda: backend)
    monkeypatch.setattr(
        app_module,
        "build_candidate_text_lookup",
        lambda app_ids: {a: f"text for {a}" for a in app_ids},
    )
    log_path = tmp_path / "events.jsonl"
    _patch_serving_log_path(monkeypatch, app_module, log_path)
    return app_module, log_path


def _log_events(log_path: Path) -> list[dict]:
    return [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]


def test_explain_streams_tokens_then_done(monkeypatch, tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    app_module, _ = _explain_app(monkeypatch, tmp_path, _FakeStreamingBackend())

    with TestClient(app_module.create_app()) as client:
        r = client.get("/explain", params={"query_app_id": 8930, "rec_app_id": 42})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    assert _parse_sse(r.text) == [
        ("token", {"text": "fake "}),
        ("token", {"text": "explanation"}),
        ("done", {}),
    ]


def test_explain_caches_pair_and_logs_ttft(monkeypatch, tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    backend = _FakeStreamingBackend()
    app_module, log_path = _explain_app(monkeypatch, tmp_path, backend)

    with TestClient(app_module.create_app()) as client:
        client.get("/explain", params={"query_app_id": 8930, "rec_app_id": 42})
        second = client.get("/explain", params={"query_app_id": 8930, "rec_app_id": 42})

    assert len(backend.calls) == 1
    assert _parse_sse(second.text) == [("token", {"text": "fake explanation"}), ("done", {})]
    first_event, second_event = _log_events(log_path)
    for event in (first_event, second_event):
        assert event["event_type"] == "explanation"
        assert event["kind"] == "game_to_game"
        assert event["explanation"] == "fake explanation"
        assert event["backend_available"] is True
        assert event["completed"] is True
        assert event["query_text"] is None
        assert 0 <= event["ttft_ms"] <= event["duration_ms"]
    assert first_event["cache_hit"] is False
    assert second_event["cache_hit"] is True


def test_explain_backend_unavailable_sends_done_only(monkeypatch, tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    app_module, log_path = _explain_app(monkeypatch, tmp_path, None)

    with TestClient(app_module.create_app()) as client:
        r = client.get("/explain", params={"query_app_id": 8930, "rec_app_id": 42})
    assert _parse_sse(r.text) == [("done", {})]

    (event,) = _log_events(log_path)
    assert event["explanation"] is None
    assert event["backend_available"] is False
    assert event["cache_hit"] is False
    assert event["ttft_ms"] is None


def test_explain_personalized_grounds_in_review_and_is_not_cached(monkeypatch, tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    backend = _FakeStreamingBackend()
    app_module, log_path = _explain_app(monkeypatch, tmp_path, backend)
    params = {"q": "Loved the tactical combat", "query_app_id": 8930, "rec_app_id": 42}

    with TestClient(app_module.create_app()) as client:
        r = client.get("/explain/personalized", params=params)
        client.get("/explain/personalized", params=params)

    assert _parse_sse(r.text)[-1] == ("done", {})
    assert backend.calls == [("review_to_game", "Loved the tactical combat", "text for 8930", "text for 42")] * 2
    for event in _log_events(log_path):
        assert event["kind"] == "review_to_game"
        assert event["query_text"] == "Loved the tactical combat"
        assert event["cache_hit"] is False
        assert event["ttft_ms"] >= 0


def test_explain_failure_mid_stream_sends_failed_and_is_not_cached(monkeypatch, tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    backend = _FakeStreamingBackend(fail_after=1)
    app_module, log_path = _explain_app(monkeypatch, tmp_path, backend)

    with TestClient(app_module.create_app()) as client:
        r = client.get("/explain", params={"query_app_id": 8930, "rec_app_id": 42})
        client.get("/explain", params={"query_app_id": 8930, "rec_app_id": 42})

    assert _parse_sse(r.text) == [("token", {"text": "fake "}), ("failed", {"detail": "generation blew up"})]
    assert len(backend.calls) == 2
    first_event = _log_events(log_path)[0]
    assert first_event["completed"] is False
    assert first_event["explanation"] == "fake"
