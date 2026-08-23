"""Candidate-text lookup for Stage 4's LLM reranker prompts.

Builds ``{app_id: text}`` from IGDB metadata so a prompt can show the LLM what a
candidate game *is*, not just its ``app_id``. Raises on a missing app_id rather than
silently degrading: this project's catalog (``game_profile_embedding_index.parquet``,
315 games) has 1:1 IGDB coverage today, so a miss means real data drift worth
investigating, not a case to paper over. See
``docs/plans/rag_stage4_llm_ranker_plan.md`` for the decision.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable

import pandas as pd

from steam_review_ml.igdb.constants import IGDB_GAMES_ENRICHED_PARQUET


def _repo_root() -> Path:
    for p in [Path.cwd(), *Path.cwd().parents]:
        if (p / "pyproject.toml").is_file():
            return p
    raise FileNotFoundError("Could not locate repo root (pyproject.toml)")


@lru_cache(maxsize=4)
def _load_igdb_text_by_app(enriched_path: str) -> dict[int, str]:
    path = Path(enriched_path)
    if not path.is_file():
        raise FileNotFoundError(f"IGDB enriched parquet not found: {path}")
    df = pd.read_parquet(path, columns=["app_id", "app_name", "summary", "storyline"])
    text_by_app: dict[int, str] = {}
    for row in df.itertuples(index=False):
        parts = [str(row.app_name), str(row.summary)]
        if isinstance(row.storyline, str) and row.storyline.strip():
            parts.append(row.storyline)
        text_by_app[int(row.app_id)] = "\n\n".join(parts)
    return text_by_app


def build_candidate_text_lookup(
    app_ids: Iterable[int],
    *,
    enriched_path: str | None = None,
) -> dict[int, str]:
    """Map each requested app_id to prompt-ready text (name + summary [+ storyline]).

    Raises ``ValueError`` listing any app_ids with no IGDB row.
    """
    path = enriched_path or str(_repo_root() / "artifacts" / "igdb" / IGDB_GAMES_ENRICHED_PARQUET)
    text_by_app = _load_igdb_text_by_app(path)

    requested = {int(a) for a in app_ids}
    missing = sorted(requested - text_by_app.keys())
    if missing:
        raise ValueError(f"No IGDB text for app_ids: {missing}")

    return {app_id: text_by_app[app_id] for app_id in requested}
