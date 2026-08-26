"""Reference-free heuristics for Stage 4 explanation quality.

Cheap proxies for the two dimensions the later LLM-judge will score properly
(groundedness, relevance) -- catch obviously broken or hallucinating explanations
before spending real judge-API calls. These are diagnostics for iterating on the
generation prompt, not a promotion bar: no threshold here gates anything.
See ``docs/plans/rag_extension_plan.md`` (Stage 5, Track B).
"""

from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from steam_review_ml.igdb.constants import IGDB_GAMES_ENRICHED_PARQUET

_REFUSAL_PHRASES = (
    "i cannot",
    "i can't",
    "i'm unable",
    "as an ai",
    "i don't have enough information",
)

_WORD_RE = re.compile(r"[a-z0-9']+")

_STOPWORDS = frozenset(
    """
    a an the this that these those and or but if then so of in on at to for with
    from by as is are was were be been being it its it's you your they their he
    she his her we our i my not no do does did will would can could should has
    have had also into about than more most such
    """.split()
)


def is_degenerate_output(text: str) -> bool:
    """Empty, refusal, or repeated-token-loop output -- broken generations, not worth scoring further."""
    cleaned = (text or "").strip()
    if not cleaned:
        return True
    lowered = cleaned.lower()
    if any(phrase in lowered for phrase in _REFUSAL_PHRASES):
        return True
    words = _WORD_RE.findall(lowered)
    if len(words) >= 6:
        most_common_count = max(Counter(words).values())
        if most_common_count / len(words) > 0.5:
            return True
    return False


def content_word_overlap_ratio(explanation: str, source_text: str, *, min_word_len: int = 4) -> float:
    """Fraction of the explanation's content words that also appear in ``source_text``.

    Coarse groundedness proxy: low overlap suggests the explanation is describing
    something not actually present in the candidate's own IGDB text. Returns 1.0 for
    an explanation with no content words to check (nothing to be ungrounded about).
    """
    exp_words = {
        w for w in _WORD_RE.findall(explanation.lower()) if len(w) >= min_word_len and w not in _STOPWORDS
    }
    if not exp_words:
        return 1.0
    source_words = set(_WORD_RE.findall(source_text.lower()))
    matched = sum(1 for w in exp_words if w in source_words)
    return matched / len(exp_words)


def find_ungrounded_tags(
    explanation: str, candidate_tags: Iterable[str], tag_vocabulary: Iterable[str]
) -> list[str]:
    """Tag-vocabulary terms mentioned in ``explanation`` that aren't among the candidate's actual tags.

    ``tag_vocabulary`` is the catalog's closed genre/theme vocabulary (IGDB's controlled
    taxonomy) -- checking against it catches hallucinated genre/theme claims more
    precisely than raw word overlap, e.g. explanation calls the game "an open-world RPG"
    but its actual tags are ``["Platform", "Indie"]``. A coarse proxy, not exact: common
    words that happen to be tag names (e.g. "action") can false-positive-flag.
    """
    lowered = explanation.lower()
    candidate_set = {t.lower() for t in candidate_tags}
    flagged = []
    for tag in tag_vocabulary:
        tag_l = tag.lower()
        if tag_l in candidate_set:
            continue
        if re.search(r"\b" + re.escape(tag_l) + r"\b", lowered):
            flagged.append(tag)
    return flagged


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    a = np.asarray(vec_a, dtype=np.float64).ravel()
    b = np.asarray(vec_b, dtype=np.float64).ravel()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _repo_root() -> Path:
    for p in [Path.cwd(), *Path.cwd().parents]:
        if (p / "pyproject.toml").is_file():
            return p
    raise FileNotFoundError("Could not locate repo root (pyproject.toml)")


@lru_cache(maxsize=4)
def _load_igdb_tags_by_app(enriched_path: str) -> dict[int, set[str]]:
    path = Path(enriched_path)
    if not path.is_file():
        raise FileNotFoundError(f"IGDB enriched parquet not found: {path}")
    df = pd.read_parquet(path, columns=["app_id", "genres_names", "themes_names"])
    tags_by_app: dict[int, set[str]] = {}
    for row in df.itertuples(index=False):
        genres = row.genres_names if row.genres_names is not None else []
        themes = row.themes_names if row.themes_names is not None else []
        tags_by_app[int(row.app_id)] = {str(t) for t in genres} | {str(t) for t in themes}
    return tags_by_app


def load_igdb_tags(app_ids: Iterable[int], *, enriched_path: str | None = None) -> dict[int, set[str]]:
    """Map each requested app_id to its IGDB genre+theme tags (for ``find_ungrounded_tags``)."""
    path = enriched_path or str(_repo_root() / "artifacts" / "igdb" / IGDB_GAMES_ENRICHED_PARQUET)
    tags_by_app = _load_igdb_tags_by_app(path)
    requested = {int(a) for a in app_ids}
    missing = sorted(requested - tags_by_app.keys())
    if missing:
        raise ValueError(f"No IGDB tags for app_ids: {missing}")
    return {app_id: tags_by_app[app_id] for app_id in requested}


def load_catalog_tag_vocabulary(*, enriched_path: str | None = None) -> set[str]:
    """Every distinct genre/theme name used anywhere in the IGDB catalog."""
    path = enriched_path or str(_repo_root() / "artifacts" / "igdb" / IGDB_GAMES_ENRICHED_PARQUET)
    tags_by_app = _load_igdb_tags_by_app(path)
    vocabulary: set[str] = set()
    for tags in tags_by_app.values():
        vocabulary |= tags
    return vocabulary
