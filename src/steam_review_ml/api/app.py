"""FastAPI app factory for content retrieval.

Run (from repo root, with ``.`` on ``PYTHONPATH`` or editable install)::

    uvicorn steam_review_ml.api.app:create_app --factory --host 0.0.0.0 --port 8000

Or: ``uvicorn steam_review_ml.api:create_app --factory`` (same factory, shorter import path).

Requires TensorFlow + Hub in the environment (conda-forge recommended) and ``pip install -e '.[api]'``.
Pip-only stack: ``pip install -e '.[api,recs-pip]'``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

_UI_HTML = Path(__file__).resolve().parent / "static" / "index.html"
DEFAULT_STRUCTURED = False  # Serve raw-query retrieval by default.


def create_app() -> Any:
    try:
        from fastapi import FastAPI, Query
        from fastapi.responses import FileResponse
    except ImportError as e:
        raise ImportError("Install API deps: pip install -e '.[api]'") from e

    from steam_review_ml.recommender.retrieve import ContentRetriever

    app = FastAPI(title="steam-review-ml recommendations", version="0.1.0")

    @app.get("/")
    def root() -> dict[str, str]:
        """Avoid a bare 404 when opening the server root in a browser."""
        return {
            "service": app.title,
            "ui": "/ui",
            "docs": "/docs",
            "health": "/health",
            "games": "/games?q=civil&limit=30 (typeahead; omit q for first *limit* names A-Z)",
            "recommendations": "/recommendations?q=your+review+draft&k=10&exclude_app_id=8930",
        }

    @app.get("/ui")
    def ui() -> Any:
        """Small browser UI: game typeahead + review text → masked recommendations."""
        if not _UI_HTML.is_file():
            raise RuntimeError(f"Missing UI file: {_UI_HTML}")
        return FileResponse(_UI_HTML, media_type="text/html; charset=utf-8")

    # One process-wide retriever: loads TF Hub and the embedding matrix once.
    _singleton: ContentRetriever | None = None

    def retriever() -> ContentRetriever:
        nonlocal _singleton
        if _singleton is None:
            _singleton = ContentRetriever()
        return _singleton

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

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
        df = retriever().index_frame
        for col in ("app_id", "app_name"):
            if col not in df.columns:
                raise RuntimeError(f"Index frame missing {col!r} — rebuild recs_002 index parquet")
        out = df[["app_id", "app_name"]].copy()
        needle = (q or "").strip()
        if needle:
            mask = out["app_name"].str.contains(needle, case=False, na=False, regex=False)
            out = out.loc[mask]
        out = out.sort_values("app_name", kind="mergesort").head(limit)
        return out.to_dict(orient="records")

    @app.get("/recommendations")
    def recommendations(
        q: str = Query(..., min_length=1, description="User draft or query text"),
        k: int = Query(10, ge=1, le=500),
        structured: bool = Query(
            DEFAULT_STRUCTURED,
            description="Experimental path: If true, use extract_preferences → build_embedding_input; default is raw query embedding.",
        ),
        exclude_app_id: int | None = Query(
            None,
            description="Steam app_id of the game being reviewed — exclude from hits (use selected row from GET /games)",
        ),
    ) -> list[dict[str, Any]]:
        """JSON list of rows from the retrieval index, each with a ``score`` column."""
        mask = {int(exclude_app_id)} if exclude_app_id is not None else None
        hits = retriever().top_k(q, k=k, structured=structured, exclude_app_ids=mask)
        return hits.to_dict(orient="records")

    return app
