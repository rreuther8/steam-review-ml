"""Serve-time configuration (no TensorFlow dependency)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from steam_review_ml.recommender.retrieve import default_repo_root

DEFAULT_SERVE_CONFIG = "configs/recs_serve.json"


def load_serve_config(
    config_path: Path | str | None = None,
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Load serve settings from ``configs/recs_serve.json`` (paths relative to repo root)."""
    root = repo_root or default_repo_root()
    path = Path(config_path) if config_path is not None else root / DEFAULT_SERVE_CONFIG
    if not path.is_file():
        raise FileNotFoundError(f"Serve config not found: {path}")
    cfg = json.loads(path.read_text(encoding="utf-8"))
    resolved = dict(cfg)
    for key in (
        "two_tower_model_path",
        "igdb_enriched_path",
        "rag_chroma_persist_dir",
        "explanation_gguf_path",
    ):
        if key in resolved and resolved[key]:
            p = Path(str(resolved[key]))
            if not p.is_absolute():
                resolved[key] = str((root / p).resolve())
    return resolved
