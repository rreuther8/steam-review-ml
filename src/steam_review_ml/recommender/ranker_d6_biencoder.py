"""D6b: separate rank bi-encoder trained listwise on frozen retrieve pools.

Status: **killed** (D6 spike; below D1 on val). Not in shipped stack; not wired to ``pool_rerank_registry``.
Notebook: ``notebooks/ranking/recs_018_ranker_d6_frozen_trunk_spike.ipynb``.

``two_tower_v1`` retrieve checkpoint is never written; rank model is a separate ``.keras`` file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from steam_review_ml.recommender.ranker_d3_listwise import _listnet_loss
from steam_review_ml.recommender.ranker_d6_common import (
    D6ListwiseConfig,
    METHOD_TWO_TOWER_V1_RANKER_D6_BIENCODER_V1,
    iter_pool_minibatches,
    write_rank_manifest,
)
from steam_review_ml.recommender.retrieve import ContentRetriever
from steam_review_ml.recommender.two_tower_score import encode_query_vector, load_two_tower_model
from steam_review_ml.recommender.two_tower_train import (
    _init_two_tower_model_class,
    keras_custom_object_scope,
    load_hub_settings,
)


def build_rank_biencoder_model(
    retriever: ContentRetriever,
    *,
    init_from_retrieve_path: Path | None = None,
    hub_url: str | None = None,
) -> Any:
    """New bi-encoder; optionally warm-start weights from read-only retrieve checkpoint."""
    if hub_url is None:
        hub_url, _ = load_hub_settings(retriever)
    model_cls = _init_two_tower_model_class()
    rank_model = model_cls(
        hub_url=hub_url,
        item_init_matrix=np.asarray(retriever.embedding_matrix, dtype=np.float32),
        name="rank_biencoder_d6b",
    )
    if not rank_model.built:
        rank_model.build()
    if init_from_retrieve_path is not None:
        retrieve = load_two_tower_model(
            init_from_retrieve_path,
            hub_url=hub_url,
            n_items=len(retriever.app_ids),
            embed_dim=int(retriever.embedding_matrix.shape[1]),
        )
        rank_model.set_weights(retrieve.get_weights())
    return rank_model


def _padded_id_label_batch(
    pools: list[dict[str, Any]],
    *,
    app_to_row: dict[int, int],
    max_list_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(pools)
    max_len = int(max_list_len)
    query_texts = np.asarray([str(row["query_text"]) for row in pools], dtype=object)
    item_row_ids = np.zeros((n, max_len), dtype=np.int32)
    labels = np.zeros((n, max_len), dtype=np.float32)
    for i, row in enumerate(pools):
        pool_apps = [int(a) for a in json.loads(row["retrieved_app_ids_json"])]
        positives = {int(v) for v in json.loads(row["validation_positive_app_ids_json"])}
        length = min(len(pool_apps), max_len)
        for j in range(length):
            app_id = int(pool_apps[j])
            item_row_ids[i, j] = int(app_to_row[app_id])
            labels[i, j] = 1.0 if app_id in positives else 0.0
    return query_texts, item_row_ids, labels


def _build_listwise_trainer(rank_model: Any, *, max_list_len: int, learning_rate: float):
    tf = __import__("tensorflow")
    keras = tf.keras

    class _RankBiencoderListwiseTrainer(keras.Model):
        def __init__(self, inner_model, list_len: int, **kwargs):
            super().__init__(name="rank_biencoder_listwise_trainer", **kwargs)
            self.inner_model = inner_model
            self.list_len = int(list_len)

        def _pool_scores(self, query_texts, item_row_ids):
            user_vecs = self.inner_model.encode_user(query_texts)
            batch_size = tf.shape(item_row_ids)[0]
            list_len = tf.shape(item_row_ids)[1]
            flat_ids = tf.reshape(item_row_ids, (-1,))
            item_vecs = self.inner_model.encode_item_by_row_ids(flat_ids)
            item_vecs = tf.reshape(item_vecs, (batch_size, list_len, -1))
            user_vecs = tf.expand_dims(user_vecs, 1)
            return tf.reduce_sum(user_vecs * item_vecs, axis=-1)

        def train_step(self, data):
            query_texts, item_row_ids, labels = data
            with tf.GradientTape() as tape:
                scores = self._pool_scores(query_texts, item_row_ids)
                loss = _listnet_loss(labels, scores)
            grads = tape.gradient(loss, self.inner_model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.inner_model.trainable_variables))
            return {"loss": loss}

        def test_step(self, data):
            query_texts, item_row_ids, labels = data
            scores = self._pool_scores(query_texts, item_row_ids)
            loss = _listnet_loss(labels, scores)
            return {"loss": loss}

    trainer = _RankBiencoderListwiseTrainer(rank_model, max_list_len)
    trainer.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate))
    return trainer


def score_biencoder_pool(
    rank_model: Any,
    *,
    query_text: str,
    pool_apps: list[int],
    app_to_row: dict[int, int],
    max_chars: int | None,
) -> np.ndarray:
    """Dot-product scores for one pool using the rank bi-encoder only."""
    if not pool_apps:
        return np.asarray([], dtype=np.float64)
    import tensorflow as tf

    user_vec = encode_query_vector(rank_model, query_text, max_chars=max_chars)
    row_ids = tf.constant([int(app_to_row[int(a)]) for a in pool_apps], dtype=tf.int32)
    item_vecs = rank_model.encode_item_by_row_ids(row_ids).numpy()
    return item_vecs @ user_vec.astype(np.float32)


def train_rank_biencoder(
    *,
    fit_pools: list[dict[str, Any]],
    tune_pools: list[dict[str, Any]],
    rank_model: Any,
    app_ids: np.ndarray,
    app_to_row: dict[int, int],
    max_chars: int | None,
    cfg: D6ListwiseConfig | None = None,
    k_final: int = 10,
) -> tuple[Any, pd.DataFrame]:
    """Train separate rank bi-encoder; early-stop on train_tune NDCG@k."""
    cfg = cfg or D6ListwiseConfig()
    tf = __import__("tensorflow")
    tf.random.set_seed(int(cfg.random_seed))
    max_len = int(cfg.max_list_len)
    trainer = _build_listwise_trainer(
        rank_model, max_list_len=max_len, learning_rate=float(cfg.learning_rate)
    )

    def _mean_ndcg(pools: list[dict[str, Any]]) -> float:
        from steam_review_ml.evaluation.retrieval_offline_eval import _rank_rows, ndcg_at_k

        vals: list[float] = []
        for row in pools:
            positives = {int(x) for x in json.loads(row["validation_positive_app_ids_json"])}
            if not positives:
                continue
            pool_apps = [int(x) for x in json.loads(row["retrieved_app_ids_json"])]
            scores = score_biencoder_pool(
                rank_model,
                query_text=str(row["query_text"]),
                pool_apps=pool_apps,
                app_to_row=app_to_row,
                max_chars=max_chars,
            )
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
            q, ids, labels = _padded_id_label_batch(batch, app_to_row=app_to_row, max_list_len=max_len)
            loss_tensor = trainer.train_step(
                (
                    tf.constant(q, dtype=tf.string),
                    tf.constant(ids, dtype=tf.int32),
                    tf.constant(labels, dtype=tf.float32),
                )
            )
            losses.append(float(loss_tensor["loss"].numpy()))
        train_loss = float(np.mean(losses)) if losses else float("nan")
        tune_ndcg = _mean_ndcg(tune_pools)
        history_rows.append({"epoch": epoch + 1, "train_loss": train_loss, "tune_NDCG": tune_ndcg})
        print(
            f"D6b epoch {epoch + 1}/{int(cfg.epochs)}: "
            f"train_loss={train_loss:.4f} tune_NDCG@{k_final}={tune_ndcg:.4f}",
            flush=True,
        )
        if tune_ndcg > best_ndcg:
            best_ndcg = tune_ndcg
            best_weights = rank_model.get_weights()
            wait = 0
        else:
            wait += 1
            if wait >= int(cfg.early_stopping_patience):
                break

    if best_weights is not None:
        rank_model.set_weights(best_weights)

    return rank_model, pd.DataFrame(history_rows)


def load_rank_biencoder(model_path: Path, *, retriever: ContentRetriever):
    from tensorflow import keras

    hub_url, _ = load_hub_settings(retriever)
    model_cls = _init_two_tower_model_class()
    scope = {"TwoTowerModel": model_cls, "TwoTower>TwoTowerModel": model_cls}
    with keras_custom_object_scope(keras, scope):
        return keras.models.load_model(model_path, compile=False)


def save_rank_biencoder_bundle(
    *,
    rank_model: Any,
    output_dir: Path,
    retrieve_model_path: Path,
    cfg: D6ListwiseConfig,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "rank_biencoder.keras"
    rank_model.save(model_path)
    write_rank_manifest(
        output_dir / "manifest.json",
        rank_method=METHOD_TWO_TOWER_V1_RANKER_D6_BIENCODER_V1,
        retrieve_model_path=retrieve_model_path,
        rank_artifact_path=model_path,
        rank_kind="separate_rank_biencoder",
        extra={"max_list_len": int(cfg.max_list_len)},
    )
    return model_path


def make_pool_score_fn(
    rank_model: Any,
    *,
    app_to_row: dict[int, int],
    max_chars: int | None,
) -> Callable[..., np.ndarray]:
    def score(pool_app_ids: list[int], retrieval_scores: list[float] | np.ndarray, query_text: str = "", **_ignored) -> np.ndarray:
        return score_biencoder_pool(
            rank_model,
            query_text=str(query_text),
            pool_apps=[int(a) for a in pool_app_ids],
            app_to_row=app_to_row,
            max_chars=max_chars,
        )

    return score
