"""D6a: listwise rank head on frozen ``two_tower_v1`` trunk features.

Status: **killed** (D6 spike; below D1 on val). Not in shipped stack; not wired to ``pool_rerank_registry``.
Notebook: ``notebooks/ranking/recs_018_ranker_d6_frozen_trunk_spike.ipynb``.

Retrieve checkpoint is loaded read-only; only the rank head is saved to a separate artifact.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from steam_review_ml.recommender.ranker_d3_listwise import _listnet_loss
from steam_review_ml.recommender.ranker_d6_common import (
    D6ListwiseConfig,
    METHOD_TWO_TOWER_V1_RANKER_D6_RANK_HEAD_V1,
    iter_pool_minibatches,
    write_rank_manifest,
)
from steam_review_ml.recommender.retrieve import ContentRetriever
from steam_review_ml.recommender.two_tower_score import (
    encode_query_vector,
    load_two_tower_model,
    precompute_catalog_item_vectors,
)
from steam_review_ml.recommender.two_tower_train import load_hub_settings


def rank_head_feature_dim(projection_dim: int) -> int:
    """Per pool item: ``[u, v, u*v, retr_score]``."""
    d = int(projection_dim)
    return 3 * d + 1


def build_rank_head_model(*, n_features: int, cfg: D6ListwiseConfig):
    """Shared MLP head: ``(list_len, n_features)`` → list scores."""
    tf = __import__("tensorflow")
    keras = tf.keras
    inp = keras.layers.Input(shape=(int(cfg.max_list_len), int(n_features)), name="rank_head_features")
    x = keras.layers.Dense(cfg.hidden_units, activation="relu")(inp)
    x = keras.layers.Dropout(cfg.dropout)(x)
    scores = keras.layers.Dense(1)(x)
    scores = keras.layers.Reshape((int(cfg.max_list_len),))(scores)
    model = keras.Model(inp, scores, name="rank_head_d6a")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=cfg.learning_rate), loss=_listnet_loss)
    return model


@dataclass
class FrozenTrunkContext:
    """Read-only retrieve tower + precomputed item projections."""

    retrieve_model: Any
    item_vectors: np.ndarray
    projection_dim: int
    max_chars: int | None


def load_frozen_trunk(
    retrieve_model_path: Path,
    retriever: ContentRetriever,
) -> FrozenTrunkContext:
    """Load retrieve checkpoint and freeze all weights."""
    hub_url, max_chars = load_hub_settings(retriever)
    model = load_two_tower_model(
        retrieve_model_path,
        hub_url=hub_url,
        n_items=len(retriever.app_ids),
        embed_dim=int(retriever.embedding_matrix.shape[1]),
    )
    model.trainable = False

    item_vectors = precompute_catalog_item_vectors(model, len(retriever.app_ids))
    # Keras 3 Dense sublayers may lack output_shape until called; model stores dim explicitly.
    proj_dim = int(getattr(model, "_projection_dim", item_vectors.shape[1]))
    return FrozenTrunkContext(
        retrieve_model=model,
        item_vectors=item_vectors,
        projection_dim=proj_dim,
        max_chars=max_chars,
    )


def pool_item_features(
    ctx: FrozenTrunkContext,
    *,
    query_text: str,
    pool_apps: list[int],
    retr_scores: list[float],
    app_to_row: dict[int, int],
) -> np.ndarray:
    """Build ``(n_items, n_features)`` matrix for one pool."""
    user_vec = encode_query_vector(
        ctx.retrieve_model,
        query_text,
        max_chars=ctx.max_chars,
    ).astype(np.float32)
    row_ids = [int(app_to_row[int(a)]) for a in pool_apps]
    item_vecs = ctx.item_vectors[np.asarray(row_ids, dtype=np.int64)]
    n = int(item_vecs.shape[0])
    u_rep = np.tile(user_vec, (n, 1))
    retr = np.asarray(retr_scores, dtype=np.float32).reshape(n, 1)
    return np.concatenate([u_rep, item_vecs, u_rep * item_vecs, retr], axis=1)


def _padded_feature_batch(
    ctx: FrozenTrunkContext,
    pools: list[dict[str, Any]],
    *,
    app_to_row: dict[int, int],
    max_list_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(pools)
    feat_dim = rank_head_feature_dim(ctx.projection_dim)
    x = np.zeros((n, max_list_len, feat_dim), dtype=np.float32)
    y = np.zeros((n, max_list_len), dtype=np.float32)
    for i, row in enumerate(pools):
        pool_apps = [int(a) for a in json.loads(row["retrieved_app_ids_json"])]
        retr = [float(v) for v in json.loads(row["retrieved_scores_json"])]
        positives = {int(v) for v in json.loads(row["validation_positive_app_ids_json"])}
        length = min(len(pool_apps), max_list_len)
        if length <= 0:
            continue
        feats = pool_item_features(
            ctx,
            query_text=str(row["query_text"]),
            pool_apps=pool_apps[:length],
            retr_scores=retr[:length],
            app_to_row=app_to_row,
        )
        x[i, :length, :] = feats
        for j in range(length):
            y[i, j] = 1.0 if int(pool_apps[j]) in positives else 0.0
    return x, y


def score_rank_head(
    head_model,
    ctx: FrozenTrunkContext,
    *,
    pool_apps: list[int],
    retr_scores: list[float],
    query_text: str,
    app_to_row: dict[int, int],
    max_list_len: int,
) -> np.ndarray:
    """Score one pool with trained rank head."""
    if not pool_apps:
        return np.asarray([], dtype=np.float64)
    feats = pool_item_features(
        ctx,
        query_text=query_text,
        pool_apps=pool_apps,
        retr_scores=retr_scores,
        app_to_row=app_to_row,
    )
    n_use = min(int(feats.shape[0]), int(max_list_len))
    feat_dim = int(feats.shape[1])
    padded = np.zeros((1, int(max_list_len), feat_dim), dtype=np.float32)
    padded[0, :n_use, :] = feats[:n_use]
    preds = head_model.predict(padded, verbose=0)
    return np.asarray(preds[0, :n_use], dtype=np.float64)


def train_rank_head_ranker(
    *,
    fit_pools: list[dict[str, Any]],
    tune_pools: list[dict[str, Any]],
    ctx: FrozenTrunkContext,
    app_ids: np.ndarray,
    app_to_row: dict[int, int],
    cfg: D6ListwiseConfig | None = None,
    k_final: int = 10,
    verbose: int = 0,
) -> tuple[Any, pd.DataFrame]:
    """Train rank head only; early-stop on ``train_tune`` mean NDCG@k."""
    cfg = cfg or D6ListwiseConfig()
    tf = __import__("tensorflow")
    tf.random.set_seed(int(cfg.random_seed))

    max_len = int(cfg.max_list_len)
    feat_dim = rank_head_feature_dim(ctx.projection_dim)
    head = build_rank_head_model(n_features=feat_dim, cfg=cfg)

    def _score_pool(pool_apps: list[int], retr: list[float], query_text: str) -> np.ndarray:
        return score_rank_head(
            head,
            ctx,
            pool_apps=pool_apps,
            retr_scores=retr,
            query_text=query_text,
            app_to_row=app_to_row,
            max_list_len=max_len,
        )

    def _mean_ndcg(pools: list[dict[str, Any]]) -> float:
        from steam_review_ml.evaluation.retrieval_offline_eval import _rank_rows, ndcg_at_k

        vals: list[float] = []
        for row in pools:
            positives = {int(x) for x in json.loads(row["validation_positive_app_ids_json"])}
            if not positives:
                continue
            pool_apps = [int(x) for x in json.loads(row["retrieved_app_ids_json"])]
            retr = [float(x) for x in json.loads(row["retrieved_scores_json"])]
            scores = _score_pool(pool_apps, retr, str(row["query_text"]))
            full = np.full(len(app_ids), -np.inf, dtype=np.float64)
            for app_id, score in zip(pool_apps, scores):
                full[int(app_to_row[int(app_id)])] = float(score)
            ranked = _rank_rows(full)[:k_final]
            vals.append(ndcg_at_k(ranked, positives, k_final, app_ids))
        return float(np.mean(vals)) if vals else float("nan")

    rng = np.random.default_rng(int(cfg.random_seed))
    best_ndcg = -np.inf
    best_weights: list[Any] | None = None
    wait = 0
    history_rows: list[dict[str, float | int]] = []

    for epoch in range(int(cfg.epochs)):
        batches = iter_pool_minibatches(fit_pools, batch_size=int(cfg.list_batch_size), rng=rng)
        losses: list[float] = []
        for batch in batches:
            x_batch, y_batch = _padded_feature_batch(
                ctx, batch, app_to_row=app_to_row, max_list_len=max_len
            )
            hist = head.fit(x_batch, y_batch, batch_size=len(batch), epochs=1, verbose=0)
            losses.append(float(hist.history["loss"][0]))
        train_loss = float(np.mean(losses)) if losses else float("nan")
        tune_ndcg = _mean_ndcg(tune_pools)
        history_rows.append({"epoch": epoch + 1, "train_loss": train_loss, "tune_NDCG": tune_ndcg})
        print(
            f"D6a epoch {epoch + 1}/{int(cfg.epochs)}: "
            f"train_loss={train_loss:.4f} tune_NDCG@{k_final}={tune_ndcg:.4f}",
            flush=True,
        )
        if tune_ndcg > best_ndcg:
            best_ndcg = tune_ndcg
            best_weights = head.get_weights()
            wait = 0
        else:
            wait += 1
            if wait >= int(cfg.early_stopping_patience):
                break

    if best_weights is not None:
        head.set_weights(best_weights)

    return head, pd.DataFrame(history_rows)


def load_rank_head(model_path: Path):
    tf = __import__("tensorflow")
    return tf.keras.models.load_model(
        model_path,
        compile=False,
        custom_objects={"_listnet_loss": _listnet_loss},
    )


def save_rank_head_bundle(
    *,
    head_model,
    output_dir: Path,
    retrieve_model_path: Path,
    projection_dim: int,
    cfg: D6ListwiseConfig,
) -> Path:
    """Save rank head + manifest; retrieve path is reference-only."""
    output_dir.mkdir(parents=True, exist_ok=True)
    head_path = output_dir / "rank_head.keras"
    head_model.save(head_path)
    write_rank_manifest(
        output_dir / "manifest.json",
        rank_method=METHOD_TWO_TOWER_V1_RANKER_D6_RANK_HEAD_V1,
        retrieve_model_path=retrieve_model_path,
        rank_artifact_path=head_path,
        rank_kind="rank_head_on_frozen_trunk",
        extra={
            "projection_dim": int(projection_dim),
            "n_features": rank_head_feature_dim(projection_dim),
            "max_list_len": int(cfg.max_list_len),
        },
    )
    return head_path


def make_pool_score_fn(
    head_model,
    ctx: FrozenTrunkContext,
    *,
    app_to_row: dict[int, int],
    max_list_len: int,
) -> Callable[..., np.ndarray]:
    def score(pool_app_ids: list[int], retrieval_scores: list[float] | np.ndarray, query_text: str = "", **_ignored) -> np.ndarray:
        retr = np.asarray(retrieval_scores, dtype=np.float64).tolist()
        return score_rank_head(
            head_model,
            ctx,
            pool_apps=[int(a) for a in pool_app_ids],
            retr_scores=retr,
            query_text=str(query_text),
            app_to_row=app_to_row,
            max_list_len=max_list_len,
        )

    return score
