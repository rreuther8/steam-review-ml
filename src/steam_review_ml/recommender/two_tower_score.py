"""Score catalog for offline eval using a trained two-tower Keras model."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
from tensorflow import keras

from steam_review_ml.recommender.math_utils import l2_normalize
from steam_review_ml.recommender.retrieve import ContentRetriever
from steam_review_ml.recommender.two_tower_train import clip_text, keras_custom_object_scope, _init_two_tower_model_class


DEFAULT_HUB_URL = "https://tfhub.dev/google/universal-sentence-encoder/4"


def load_two_tower_model(
    model_path: Path,
    *,
    hub_url: str | None = None,
    n_items: int | None = None,
    embed_dim: int | None = None,
):

    if not model_path.is_file():
        raise FileNotFoundError(f"two-tower model not found: {model_path}")
    if n_items is None or embed_dim is None:
        raise ValueError("load_two_tower_model requires n_items and embed_dim (from ContentRetriever).")

    model_cls = _init_two_tower_model_class()
    scope = {
        "TwoTowerModel": model_cls,
        "TwoTower>TwoTowerModel": model_cls,
    }
    with keras_custom_object_scope(keras, scope):
        model = keras.models.load_model(model_path, compile=False)
    if not model.built:
        model.build()
    if hub_url is not None and str(getattr(model, "_hub_url", "")) != str(hub_url):
        raise ValueError(
            f"checkpoint hub_url {model._hub_url!r} does not match requested {hub_url!r}"
        )
    if n_items is not None and int(getattr(model, "_n_items", -1)) != int(n_items):
        raise ValueError(
            f"checkpoint n_items {model._n_items} does not match catalog {n_items}"
        )
    if embed_dim is not None and int(getattr(model, "_embed_dim", -1)) != int(embed_dim):
        raise ValueError(
            f"checkpoint embed_dim {model._embed_dim} does not match catalog {embed_dim}"
        )
    return model


def precompute_catalog_item_vectors(
    model,
    n_items: int,
    *,
    batch_size: int = 256,
) -> np.ndarray:
    import tensorflow as tf

    parts: list[np.ndarray] = []
    for start in range(0, n_items, batch_size):
        stop = min(start + batch_size, n_items)
        item_row_batch = tf.constant(np.arange(start, stop), dtype=tf.int32)
        parts.append(model.encode_item_by_row_ids(item_row_batch).numpy())
    return np.vstack(parts).astype(np.float32, copy=False)


def _l2_normalize_rows(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    mat = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return mat / norms


def encode_query_vector(
    model,
    query_text: str,
    *,
    max_chars: int | None,
) -> np.ndarray:
    import tensorflow as tf

    text_batch = tf.constant([clip_text(query_text, max_chars)], dtype=tf.string)
    return model.encode_user(text_batch).numpy()[0]


def score_catalog(
    user_vector: np.ndarray,
    item_vectors: np.ndarray,
    *,
    mask_row: int | None,
) -> np.ndarray:
    user_unit = l2_normalize(np.asarray(user_vector, dtype=np.float32))
    item_unit = _l2_normalize_rows(item_vectors)
    scores = item_unit @ user_unit
    scores = scores.astype(np.float64, copy=False)
    if mask_row is not None and 0 <= int(mask_row) < scores.shape[0]:
        scores[int(mask_row)] = -np.inf
    return scores


def make_two_tower_score_fn(
    model,
    retriever: ContentRetriever,
    *,
    max_chars: int | None,
    catalog_item_batch: int = 256,
    mask_query_app: bool = True,
) -> Callable[[dict], np.ndarray]:
    item_vectors = precompute_catalog_item_vectors(
        model, len(retriever.app_ids), batch_size=catalog_item_batch
    )
    app_to_row = {int(a): i for i, a in enumerate(np.asarray(retriever.app_ids).tolist())}

    def score_fn(ex: dict) -> np.ndarray:
        user_vector = encode_query_vector(model, str(ex["query_text"]), max_chars=max_chars)
        mask_row = None
        if mask_query_app:
            mask_row = app_to_row.get(int(ex["query_app_id"]))
        return score_catalog(user_vector, item_vectors, mask_row=mask_row)

    return score_fn
