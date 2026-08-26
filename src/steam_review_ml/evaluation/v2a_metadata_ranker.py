"""V2a metadata-augmented pool rerankers (IGDB taxonomy USE signals)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from steam_review_ml.evaluation.heuristic_ranker import (
    DEFAULT_LOGPOP_BLEND_ALPHA,
    minmax_norm,
    score_logpop_blend,
)
from steam_review_ml.igdb.constants import (
    IGDB_GAMES_ENRICHED_PARQUET,
    TAXONOMY_NAMES_POOLED_USE_SUFFIX,
    TAXONOMY_RESOLVE_FIELDS,
)

METHOD_TWO_TOWER_V1_V2A_EMBED_QUERY_LOGPOP_BLEND = "two_tower_v1_v2a_embed_query_logpop_blend"
# Same rerank formula/params as above, paired with Stage 3 RAG retrieval pools instead of
# two_tower_v1 -- lets the offline ranking-eval job score the actually-shipped RAGRecommender
# combination (retrieval + rerank together), not just each stage in isolation.
METHOD_RAG_CHUNK_V1_VECTOR_BLEND_QUERY_V2A_EMBED_QUERY_LOGPOP_BLEND = (
    "rag_chunk_v1_vector_blend_query_v2a_embed_query_logpop_blend"
)

# Frozen from recs_020 train_tune; do not tune on val.
DEFAULT_V2A_EMBED_W_META = 0.1
V2A_GENRE_THEME_KW_FIELDS: tuple[str, ...] = ("genres", "themes", "keywords")


def _repo_root() -> Path:
    for p in [Path.cwd(), *Path.cwd().parents]:
        if (p / "pyproject.toml").is_file():
            return p
    raise FileNotFoundError("Could not locate repo root (pyproject.toml)")


def _pooled_col(field: str) -> str:
    return f"{field}_names{TAXONOMY_NAMES_POOLED_USE_SUFFIX}"


def _parse_pooled_use(val: Any) -> np.ndarray:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return np.zeros(512, dtype=np.float64)
    arr = np.asarray(val, dtype=np.float64)
    return arr.reshape(512) if arr.size == 512 else np.zeros(512, dtype=np.float64)


@lru_cache(maxsize=4)
def _load_pooled_by_app(enriched_path: str) -> dict[int, dict[str, np.ndarray]]:
    path = Path(enriched_path)
    if not path.is_file():
        raise FileNotFoundError(f"IGDB enriched parquet not found: {path}")
    use_cols = [_pooled_col(f) for f in TAXONOMY_RESOLVE_FIELDS]
    df = pd.read_parquet(path, columns=["app_id", *use_cols])
    pooled_by_app: dict[int, dict[str, np.ndarray]] = {}
    for _, row in df.iterrows():
        app_id = int(row["app_id"])
        pooled_by_app[app_id] = {
            f: _parse_pooled_use(row[_pooled_col(f)]) for f in TAXONOMY_RESOLVE_FIELDS
        }
    return pooled_by_app


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def embed_meta_scores(
    pool_app_ids: list[int],
    *,
    query_app_id: int,
    fields: tuple[str, ...],
    enriched_path: str,
) -> np.ndarray:
    pooled_by_app = _load_pooled_by_app(enriched_path)
    meta = np.empty(len(pool_app_ids), dtype=np.float64)
    for i, app_id in enumerate(pool_app_ids):
        sims = []
        for f in fields:
            anchor = pooled_by_app.get(int(query_app_id), {}).get(f, np.zeros(512))
            cand = pooled_by_app.get(int(app_id), {}).get(f, np.zeros(512))
            sims.append(_cosine_sim(anchor, cand))
        meta[i] = float(np.mean(sims))
    return meta


def score_v2a_embed_query_logpop_blend(
    pool_app_ids: list[int],
    retrieval_scores: list[float] | np.ndarray,
    *,
    w_meta: float,
    query_app_id: int,
    alpha: float = DEFAULT_LOGPOP_BLEND_ALPHA,
    fields: tuple[str, ...] = V2A_GENRE_THEME_KW_FIELDS,
    enriched_path: str | None = None,
    pop_row: np.ndarray,
    app_to_row: dict[int, int],
    **_: Any,
) -> np.ndarray:
    """``(1 - w_meta) * norm(D1) + w_meta * norm(taxonomy USE cosine)``; query anchor."""
    path = enriched_path or str(_repo_root() / "artifacts" / "igdb" / IGDB_GAMES_ENRICHED_PARQUET)
    meta = embed_meta_scores(
        pool_app_ids,
        query_app_id=int(query_app_id),
        fields=fields,
        enriched_path=path,
    )
    d1 = score_logpop_blend(
        pool_app_ids,
        retrieval_scores,
        alpha=alpha,
        pop_row=pop_row,
        app_to_row=app_to_row,
    )
    if w_meta <= 0.0:
        return d1
    if w_meta >= 1.0:
        return minmax_norm(meta)
    return (1.0 - w_meta) * minmax_norm(d1) + w_meta * minmax_norm(meta)
