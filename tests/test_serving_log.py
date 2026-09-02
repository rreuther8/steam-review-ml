"""Unit tests for steam_review_ml.api.serving_log (no API/model deps)."""

from __future__ import annotations

import json
from pathlib import Path

from steam_review_ml.api.serving_log import (
    ExplanationEvent,
    RecommendationEvent,
    RecommendationResult,
    log_event,
)


def test_log_event_writes_one_json_line_with_timestamp(tmp_path: Path) -> None:
    log_path = tmp_path / "nested" / "events.jsonl"
    log_event(
        log_path,
        RecommendationEvent(
            query_app_id=1,
            query_text="q",
            method_id="m",
            results=[RecommendationResult(app_id=2, app_name="G", score=0.5, rank=1)],
            duration_ms=12.5,
            retrieve_ms=8.0,
            rerank_ms=4.5,
        ),
    )

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert "timestamp" in record
    assert record["event_type"] == "recommendation"
    assert record["query_app_id"] == 1
    assert record["results"] == [{"app_id": 2, "app_name": "G", "score": 0.5, "rank": 1}]
    assert record["duration_ms"] == 12.5
    assert record["retrieve_ms"] == 8.0
    assert record["rerank_ms"] == 4.5


def test_log_event_appends_across_calls(tmp_path: Path) -> None:
    log_path = tmp_path / "events.jsonl"
    log_event(
        log_path,
        ExplanationEvent(
            query_app_id=1, rec_app_id=2, explanation="a", cache_hit=False, backend_available=True, duration_ms=1.0
        ),
    )
    log_event(
        log_path,
        ExplanationEvent(
            query_app_id=1, rec_app_id=3, explanation=None, cache_hit=False, backend_available=False, duration_ms=0.5
        ),
    )

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    first, second = (json.loads(line) for line in lines)
    assert first["event_type"] == "explanation"
    assert first["rec_app_id"] == 2
    assert first["backend_available"] is True
    assert second["rec_app_id"] == 3
    assert second["explanation"] is None
    assert second["backend_available"] is False
