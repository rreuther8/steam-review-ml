"""USE text embeddings for IGDB game text fields."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from steam_review_ml.igdb.constants import USE_EMBEDDING_FIELD_SUFFIX
from steam_review_ml.igdb.job_config import IgdbGamesJobConfig
from steam_review_ml.recommender.math_utils import l2_normalize

logger = logging.getLogger(__name__)

DEFAULT_TFHub_URL = "https://tfhub.dev/google/universal-sentence-encoder/4"
DEFAULT_GAME_EMBEDDING_META = (
    "artifacts/recs/embeddings/game_profile/default/game_profile_embedding_meta.json"
)
IGDB_FEATURES_META_FILENAME = "igdb_features_meta.json"


def use_embedding_column(field: str) -> str:
    return f"{field}{USE_EMBEDDING_FIELD_SUFFIX}"


def resolve_tfhub_url(
    repo_root: Path,
    *,
    tfhub_url: str | None = None,
    game_embedding_meta_path: Path | None = None,
) -> str:
    if tfhub_url and str(tfhub_url).strip():
        return str(tfhub_url).strip()
    meta_path = game_embedding_meta_path or (repo_root / DEFAULT_GAME_EMBEDDING_META)
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return str(meta["model_name"])
    return DEFAULT_TFHub_URL


def _encode_texts_batched(
    texts: list[str],
    embed_fn: Callable[[list[str]], Any],
    batch_size: int,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        out = embed_fn(batch)
        chunks.append(np.asarray(out, dtype=np.float32))
    emb = np.concatenate(chunks, axis=0)
    if emb.ndim != 2:
        raise RuntimeError(f"Expected 2D embedding array, got shape={emb.shape}")
    return emb


def embed_text_columns(
    df: pd.DataFrame,
    text_fields: Sequence[str],
    *,
    embed_fn: Callable[[list[str]], Any],
    batch_size: int = 64,
    max_chars_per_field: int | None = 8000,
) -> pd.DataFrame:
    """Return a copy of ``df`` with ``{field}__use`` L2-normalized USE vectors added."""
    if not text_fields:
        return df.copy()

    out = df.copy()
    for field in text_fields:
        if field not in out.columns:
            raise KeyError(f"Missing text field for embedding: {field!r}")
        texts = out[field].fillna("").astype(str)
        if max_chars_per_field is not None:
            texts = texts.str[:max_chars_per_field]
        embeddings = _encode_texts_batched(texts.tolist(), embed_fn, batch_size)
        normalized = np.stack([l2_normalize(row) for row in embeddings], axis=0)
        out[use_embedding_column(field)] = list(normalized)
        logger.info("Embedded IGDB field %r -> %r", field, use_embedding_column(field))

    return out


def build_features_meta(
    *,
    config: IgdbGamesJobConfig,
    tfhub_url: str,
    embed_dim: int,
    n_rows: int,
) -> dict[str, Any]:
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": tfhub_url,
        "backend": "tensorflow_hub",
        "method": "encode_each_field_l2_normalize",
        "embed_text_fields": list(config.embed_text_fields),
        "embed_text_field_columns": [use_embedding_column(f) for f in config.embed_text_fields],
        "embed_batch_size": config.embed_batch_size,
        "max_chars_per_field": config.max_chars_per_field,
        "n_rows": n_rows,
        "embed_dim": embed_dim,
        "raw_input_filename": config.output_filename,
        "features_output_filename": config.features_output_filename,
    }


def write_igdb_text_features(
    config: IgdbGamesJobConfig,
    joined: pd.DataFrame,
) -> Path:
    """Embed configured text fields and write features parquet + meta JSON."""
    if not config.embed_text_fields:
        raise ValueError("embed_text_fields is empty; nothing to write.")

    import tensorflow as tf
    import tensorflow_hub as hub

    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

    tfhub_url = resolve_tfhub_url(
        config.repo_root,
        tfhub_url=config.tfhub_url,
        game_embedding_meta_path=config.game_embedding_meta_path,
    )
    embed_fn = hub.load(tfhub_url)

    features = embed_text_columns(
        joined,
        config.embed_text_fields,
        embed_fn=embed_fn,
        batch_size=config.embed_batch_size,
        max_chars_per_field=config.max_chars_per_field,
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = config.output_dir / config.features_output_filename
    features.to_parquet(out_path, index=False)

    sample_col = use_embedding_column(config.embed_text_fields[0])
    embed_dim = int(np.asarray(features[sample_col].iloc[0]).shape[0])
    meta = build_features_meta(
        config=config,
        tfhub_url=tfhub_url,
        embed_dim=embed_dim,
        n_rows=len(features),
    )
    meta_path = config.output_dir / IGDB_FEATURES_META_FILENAME
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("Wrote %s (%s rows)", out_path, len(features))
    return out_path
