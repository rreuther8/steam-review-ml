"""Append-only JSONL log of live serving requests.

Captures what real callers actually saw -- retrieved/ranked candidates and generated
explanations -- as raw material for a later, separate async LLM-as-judge pass (or any
other offline analysis) over live traffic. Writing an event is a single local file
append, no network call, so it happens synchronously inline without adding meaningful
latency to the request it's logging. The judge step that *does* call a network API is
not built here -- it would run later, as its own process reading this log.

Events are dataclasses (not raw dicts) so the schema is checked at construction time,
not just by convention.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class RecommendationResult:
    app_id: int
    app_name: str
    score: float
    rank: int


@dataclass(frozen=True)
class RecommendationEvent:
    """One live ``/recommendations`` response -- the ranked list a real caller was shown."""

    query_app_id: int
    query_text: str
    method_id: str
    results: list[RecommendationResult]
    duration_ms: float
    retrieve_ms: float
    rerank_ms: float
    event_type: str = field(default="recommendation", init=False)


ExplanationKind = Literal["game_to_game", "review_to_game"]


@dataclass(frozen=True)
class ExplanationEvent:
    """One live streamed ``/explain`` (``game_to_game``) or ``/explain/personalized``
    (``review_to_game``) response for a single ``(query_app_id, rec_app_id)`` pair.

    ``ttft_ms`` is time from request start to the first streamed token (``None`` if no token
    was sent). ``completed`` is ``False`` when the client disconnected or generation failed
    mid-stream -- ``explanation`` then holds only the partial text.
    """

    query_app_id: int
    rec_app_id: int
    kind: ExplanationKind
    explanation: str | None
    cache_hit: bool
    backend_available: bool
    completed: bool
    duration_ms: float
    ttft_ms: float | None
    query_text: str | None = None
    event_type: str = field(default="explanation", init=False)


ServingEvent = RecommendationEvent | ExplanationEvent


def log_event(log_path: Path, event: ServingEvent) -> None:
    """Append one JSON line to ``log_path``, stamped with the current UTC time."""
    record = {"timestamp": datetime.now(timezone.utc).isoformat(), **asdict(event)}
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
