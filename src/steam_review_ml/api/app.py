"""FastAPI app factory for content retrieval.

Run (from repo root, with ``.`` on ``PYTHONPATH`` or editable install)::

    uvicorn steam_review_ml.api.app:create_app --factory --host 0.0.0.0 --port 8000

Or: ``uvicorn steam_review_ml.api:create_app --factory`` (same factory, shorter import path).

Requires ``pip install -e '.[api]'`` and TensorFlow + TF Hub in the environment (conda-forge
recommended; see ``docs/usage_pipeline.md``).

Default recommendations use the shipped stack: ``two_tower_v1`` @100 →
``two_tower_v1_v2a_embed_query_logpop_blend`` @10 (see ``configs/recs_serve.json``).
"""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Iterator, Literal, cast

import pandas as pd

from steam_review_ml.api.serving_log import (
    ExplanationEvent,
    ExplanationKind,
    RecommendationEvent,
    RecommendationResult,
    log_event,
)
from steam_review_ml.evaluation.candidate_text import build_candidate_text_lookup
from steam_review_ml.recommender.retrieve import ContentRetriever, default_repo_root
from steam_review_ml.recommender.serve_config import load_serve_config
from steam_review_ml.recommender.two_tower_recommender import TwoTowerRecommender

_UI_HTML = Path(__file__).resolve().parent / "static" / "index.html"
ServeMethod = Literal["v2a", "raw", "structured"]


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """``DataFrame.to_dict(orient="records")`` typed for JSON response bodies.

    pandas-stubs types this as ``list[dict[Hashable, Any]]``; our columns are
    always strings, so this is a safe narrowing cast, not a runtime check.
    """
    return cast("list[dict[str, Any]]", df.to_dict(orient="records"))


def _load_explanation_backend() -> Any | None:
    """Load the Stage 4 explanation ``LlamaCppBackend``, or ``None`` if unavailable.

    Optional: the GGUF model is a multi-GB local file not everyone running this API
    has (or needs — ``llm-local`` is an optional extra). Missing model/dep degrades
    to no ``explanation`` field on ``/recommendations`` rather than failing startup.
    """
    cfg = load_serve_config()
    gguf_path = cfg.get("explanation_gguf_path")
    if not gguf_path or not Path(gguf_path).is_file():
        return None
    try:
        from steam_review_ml.recommender.llm_backends import LlamaCppBackend
    except ImportError:
        return None
    return LlamaCppBackend(str(gguf_path), n_gpu_layers=-1)


def _sse(event: str, data: dict[str, Any]) -> str:
    """One Server-Sent Events message. ``data`` is JSON so newlines in model text can't break framing."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def _iterate_in_thread(iterator: Iterator[str]) -> AsyncIterator[str]:
    """Pull each item of a blocking iterator (llama-cpp generation) in a worker thread, so the
    event loop stays free for other requests while tokens are produced.

    If the consumer is cancelled (client disconnect), wait for the in-flight ``next()`` to
    finish before closing the iterator -- the caller holds the generation lock, and releasing
    it while the model is still mid-call would let a second generation overlap it.
    """
    done = object()
    try:
        while True:
            step = asyncio.ensure_future(asyncio.to_thread(next, iterator, done))
            try:
                item = await asyncio.shield(step)
            except asyncio.CancelledError:
                await step
                raise
            if item is done:
                return
            yield item
    finally:
        iterator.close()  # type: ignore[attr-defined]  # backend stream_* methods are generators


def create_app() -> Any:
    try:
        from fastapi import FastAPI, HTTPException, Query
        from fastapi.responses import FileResponse, StreamingResponse
    except ImportError as e:
        raise ImportError("Install API deps: pip install -e '.[api]'") from e

    default_serve_method: ServeMethod = "v2a"

    _serving_log_path = Path(
        load_serve_config().get(
            "serving_log_path", str(default_repo_root() / "artifacts" / "recs" / "serving_logs" / "events.jsonl")
        )
    )

    _state: dict[str, Any] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        _state["recommender"] = TwoTowerRecommender.from_serve_config()
        yield
        _state.clear()

    app = FastAPI(title="steam-review-ml recommendations", version="0.2.0", lifespan=lifespan)

    @app.get("/")
    def root() -> dict[str, str]:
        """Avoid a bare 404 when opening the server root in a browser."""
        return {
            "service": app.title,
            "ui": "/ui",
            "docs": "/docs",
            "health": "/health",
            "games": "/games?q=civil&limit=30 (typeahead; omit q for first *limit* names A-Z)",
            "recommendations": (
                "/recommendations?q=your+review+draft&k=10&exclude_app_id=8930"
                " (default method=v2a; use method=raw for legacy ContentRetriever)"
            ),
            "explain": (
                "/explain?query_app_id=8930&rec_app_id=42"
                " (SSE stream of top-pick 'why' text, generated separately -- not blocking /recommendations)"
            ),
            "explain_personalized": (
                "/explain/personalized?q=your+review&query_app_id=8930&rec_app_id=42"
                " (SSE stream; like /explain but also grounded in the review text)"
            ),
        }

    @app.get("/ui")
    def ui() -> Any:
        """Small browser UI: game typeahead + review text → masked recommendations."""
        if not _UI_HTML.is_file():
            raise RuntimeError(f"Missing UI file: {_UI_HTML}")
        return FileResponse(_UI_HTML, media_type="text/html; charset=utf-8")

    _content_retriever: ContentRetriever | None = None

    def content_retriever() -> ContentRetriever:
        nonlocal _content_retriever
        if _content_retriever is None:
            _content_retriever = ContentRetriever()
        return _content_retriever

    def recommender() -> TwoTowerRecommender:
        return _state["recommender"]

    _explanation_backend: Any | None = None
    _explanation_backend_loaded = False
    _explanation_cache: dict[tuple[int, int], str] = {}
    # One loaded llama-cpp model can't serve two generations at once, so generations (and the
    # lazy model load) take turns. An asyncio lock, held by an async generator, so a client
    # disconnect cancels the generator and its ``async with`` releases the lock -- a sync
    # generator isn't closed on disconnect and would hold a threading lock forever.
    _generation_lock = asyncio.Lock()

    def explanation_backend() -> Any | None:
        """Lazy-loaded, like ``content_retriever()`` -- avoids paying the ~6s GGUF load
        (and its GPU memory) for requests/tests that never reach a v2a top-1 result."""
        nonlocal _explanation_backend, _explanation_backend_loaded
        if not _explanation_backend_loaded:
            _explanation_backend = _load_explanation_backend()
            _explanation_backend_loaded = True
        return _explanation_backend

    @app.get("/health")
    def health() -> dict[str, str]:
        rec = recommender()
        return {
            "status": "ok",
            "default_method": default_serve_method,
            "recommender_method_id": rec.method_id,
            "igdb_enriched_path": rec.igdb_enriched_path or "",
            "k_retrieval": str(rec.k_retrieval),
            "k_final": str(rec.k_final),
        }

    @app.get("/games")
    def games(
        q: str | None = Query(
            None,
            max_length=200,
            description="Substring on app name (case-insensitive). Omit to list the first *limit* games sorted A-Z.",
        ),
        limit: int = Query(50, ge=1, le=200),
    ) -> list[dict[str, Any]]:
        """Catalog slice for a searchable game picker. Pair with ``exclude_app_id`` on ``/recommendations``."""
        df: pd.DataFrame = content_retriever().index_frame
        for col in ("app_id", "app_name"):
            if col not in df.columns:
                raise RuntimeError(f"Index frame missing {col!r} — rebuild recs_002 index parquet")
        out = df[["app_id", "app_name"]].copy()
        needle = (q or "").strip()
        if needle:
            mask = out["app_name"].str.contains(needle, case=False, na=False, regex=False)
            out = out.loc[mask]
        out = out.sort_values("app_name", kind="mergesort").head(limit)
        return _records(out)

    @app.get("/recommendations")
    def recommendations(
        q: str = Query(..., min_length=1, description="User draft or query text"),
        k: int = Query(10, ge=1, le=500),
        method: ServeMethod = Query(
            default_serve_method,
            description=(
                "v2a: shipped two_tower_v1 + v2a rerank (requires exclude_app_id); "
                "raw|structured: legacy ContentRetriever ablations"
            ),
        ),
        structured: bool = Query(
            False,
            description="Deprecated when method=raw|structured; use method=structured instead.",
        ),
        history_text: list[str] | None = Query(
            None,
            description="Optional prior review texts (raw/structured methods only).",
        ),
        history_alpha: float = Query(
            0.0,
            ge=0.0,
            le=1.0,
            description="History blend weight for raw/structured retrieval only.",
        ),
        history_top_k: int = Query(
            3,
            ge=1,
            le=20,
            description="Max prior reviews to blend (raw/structured only).",
        ),
        history_min_similarity: float = Query(
            0.2,
            ge=-1.0,
            le=1.0,
            description="Min cosine(query, prior_review) for history blend (raw/structured only).",
        ),
        exclude_app_id: int | None = Query(
            None,
            description=(
                "Steam app_id of the game being reviewed — required for method=v2a "
                "(masks query game and anchors IGDB metadata)"
            ),
        ),
    ) -> list[dict[str, Any]]:
        """JSON list of rows from the retrieval index, each with a ``score`` column."""
        if method == "v2a":
            if exclude_app_id is None:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "exclude_app_id is required for method=v2a "
                        "(select the game being reviewed via GET /games)"
                    ),
                )
            rec = recommender()
            if k > rec.k_final:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"k={k} exceeds method=v2a's fixed result size "
                        f"(k_final={rec.k_final}); request k<={rec.k_final} "
                        "or use method=raw for a full-catalog search"
                    ),
                )
            t0 = time.perf_counter()
            hits = rec.recommend(q, query_app_id=int(exclude_app_id))
            records = _records(hits.head(k))
            duration_ms = (time.perf_counter() - t0) * 1000
            log_event(
                _serving_log_path,
                RecommendationEvent(
                    query_app_id=int(exclude_app_id),
                    query_text=q,
                    method_id=rec.method_id,
                    results=[
                        RecommendationResult(
                            app_id=row["app_id"], app_name=row["app_name"], score=row["score"], rank=i + 1
                        )
                        for i, row in enumerate(records)
                    ],
                    duration_ms=duration_ms,
                    retrieve_ms=hits.attrs.get("retrieve_ms", 0.0),
                    rerank_ms=hits.attrs.get("rerank_ms", 0.0),
                ),
            )
            return records

        use_structured = method == "structured" or structured
        mask = {int(exclude_app_id)} if exclude_app_id is not None else None
        hits = content_retriever().top_k(
            q,
            k=k,
            structured=use_structured,
            exclude_app_ids=mask,
            history_texts=history_text,
            history_blend_alpha=history_alpha,
            history_top_k=history_top_k,
            history_min_similarity=history_min_similarity,
        )
        return _records(hits)

    async def _stream_explanation(
        kind: ExplanationKind, query_app_id: int, rec_app_id: int, review_text: str | None
    ) -> AsyncIterator[str]:
        """SSE body shared by both explain endpoints: ``token`` events, then ``done`` (or ``failed``).

        Backend unavailable -> ``done`` with no tokens (the UI hides the box). Only
        ``game_to_game`` is cached: it depends on the two app_ids alone and runs at
        temperature 0, so it's deterministic per pair; ``review_to_game`` depends on free text.
        The event is logged in ``finally`` so a client disconnect still leaves a record.
        """
        t0 = time.perf_counter()
        ttft_ms: float | None = None
        pieces: list[str] = []
        cache_key = (query_app_id, rec_app_id)
        cache_hit = kind == "game_to_game" and cache_key in _explanation_cache
        backend_available = True
        completed = False
        try:
            if cache_hit:
                pieces.append(_explanation_cache[cache_key])
                ttft_ms = (time.perf_counter() - t0) * 1000
                yield _sse("token", {"text": pieces[0]})
            else:
                async with _generation_lock:
                    backend = await asyncio.to_thread(explanation_backend)
                    backend_available = backend is not None
                    if backend is not None:
                        game_texts = build_candidate_text_lookup([query_app_id, rec_app_id])
                        if kind == "review_to_game":
                            tokens = backend.stream_review_explanation(
                                review_text, game_texts[query_app_id], game_texts[rec_app_id]
                            )
                        else:
                            tokens = backend.stream_explanation(game_texts[query_app_id], game_texts[rec_app_id])
                        async for text in _iterate_in_thread(tokens):
                            if ttft_ms is None:
                                ttft_ms = (time.perf_counter() - t0) * 1000
                            pieces.append(text)
                            yield _sse("token", {"text": text})
                if kind == "game_to_game" and pieces:
                    _explanation_cache[cache_key] = "".join(pieces).strip()
            completed = True
            yield _sse("done", {})
        except Exception as e:
            # Headers (200) are already sent mid-stream, so report failure as an event instead.
            yield _sse("failed", {"detail": str(e)})
        finally:
            log_event(
                _serving_log_path,
                ExplanationEvent(
                    query_app_id=query_app_id,
                    rec_app_id=rec_app_id,
                    kind=kind,
                    explanation="".join(pieces).strip() or None,
                    cache_hit=cache_hit,
                    backend_available=backend_available,
                    completed=completed,
                    duration_ms=(time.perf_counter() - t0) * 1000,
                    ttft_ms=ttft_ms,
                    query_text=review_text,
                ),
            )

    def _sse_response(body: AsyncIterator[str]) -> Any:
        return StreamingResponse(body, media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    @app.get("/explain")
    def explain(
        query_app_id: int = Query(..., description="app_id of the game being reviewed"),
        rec_app_id: int = Query(..., description="app_id of the recommended game to explain"),
    ) -> Any:
        """SSE stream of grounded game-to-game 'why this pick' text for one (query, rec) pair --
        generated separately from ``/recommendations`` so the LLM call doesn't block it."""
        return _sse_response(_stream_explanation("game_to_game", int(query_app_id), int(rec_app_id), None))

    @app.get("/explain/personalized")
    def explain_personalized(
        q: str = Query(..., min_length=1, description="The user's review text"),
        query_app_id: int = Query(..., description="app_id of the game being reviewed"),
        rec_app_id: int = Query(..., description="app_id of the recommended game to explain"),
    ) -> Any:
        """SSE stream like ``/explain``, but also grounded in the user's review (review-to-game)."""
        return _sse_response(_stream_explanation("review_to_game", int(query_app_id), int(rec_app_id), q))

    return app
