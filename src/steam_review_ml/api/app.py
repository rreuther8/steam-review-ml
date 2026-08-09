"""FastAPI app factory for content retrieval.

Run (from repo root, with ``.`` on ``PYTHONPATH`` or editable install)::

    uvicorn steam_review_ml.api.app:create_app --factory --host 0.0.0.0 --port 8000

Or: ``uvicorn steam_review_ml.api:create_app --factory`` (same factory, shorter import path).

Requires TensorFlow + Hub in the environment (conda-forge recommended) and ``pip install -e '.[api]'``.
Pip-only stack: ``pip install -e '.[api,recs-pip]'``.

Default recommendations use the shipped stack: ``two_tower_v1`` @100 →
``two_tower_v1_v2a_embed_query_logpop_blend`` @10 (see ``configs/recs_serve.json``).
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal, cast

import pandas as pd

from steam_review_ml.recommender.retrieve import ContentRetriever
from steam_review_ml.recommender.stacked_recommender import StackedRecommender

_UI_HTML = Path(__file__).resolve().parent / "static" / "index.html"
ServeMethod = Literal["v2a", "raw", "structured"]


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """``DataFrame.to_dict(orient="records")`` typed for JSON response bodies.

    pandas-stubs types this as ``list[dict[Hashable, Any]]``; our columns are
    always strings, so this is a safe narrowing cast, not a runtime check.
    """
    return cast("list[dict[str, Any]]", df.to_dict(orient="records"))


def create_app() -> Any:
    try:
        from fastapi import FastAPI, HTTPException, Query
        from fastapi.responses import FileResponse
    except ImportError as e:
        raise ImportError("Install API deps: pip install -e '.[api]'") from e

    default_serve_method: ServeMethod = "v2a"

    _state: dict[str, Any] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        _state["stacked"] = StackedRecommender.from_serve_config()
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

    def stacked_recommender() -> StackedRecommender:
        return _state["stacked"]

    @app.get("/health")
    def health() -> dict[str, str]:
        stacked = stacked_recommender()
        return {
            "status": "ok",
            "default_method": default_serve_method,
            "stacked_method_id": stacked.method_id,
            "two_tower_model_path": str(stacked.two_tower_model_path),
            "igdb_enriched_path": stacked.igdb_enriched_path or "",
            "k_retrieval": str(stacked.k_retrieval),
            "k_final": str(stacked.k_final),
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
                "v2a: shipped two_tower + v2a rerank (requires exclude_app_id); "
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
            stacked = stacked_recommender()
            if k > stacked.k_final:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"k={k} exceeds method=v2a's fixed result size "
                        f"(k_final={stacked.k_final}); request k<={stacked.k_final} "
                        "or use method=raw for a full-catalog search"
                    ),
                )
            hits = stacked.recommend(q, query_app_id=int(exclude_app_id))
            return _records(hits.head(k))

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

    return app
