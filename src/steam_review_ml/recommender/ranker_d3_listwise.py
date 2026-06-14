"""D3 listwise pool ranker: ListNet-style softmax over each frozen pool.

Status: **killed** (2026-06-07). Not in shipped stack; not wired to ``pool_rerank_registry``.
Notebook: ``notebooks/ranking/recs_014_ranker_d2_d3_train_head_to_head.ipynb``.
See ``docs/ranking_decision_log.md`` § 2026-06-07.

One training example = one retrieval list (≤ ``max_list_len`` candidates).
Shared per-item MLP scores the list; loss is cross-entropy vs a uniform
distribution over within-pool positives (not pairwise).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from steam_review_ml.recommender.ranker_d2_pointwise import (
    FEATURE_COLS,
    mean_pool_ndcg,
)

METHOD_TWO_TOWER_V1_RANKER_D3_LISTWISE_V1 = "two_tower_v1_ranker_d3_listwise_v1"


@dataclass(frozen=True)
class ListwiseRankerConfig:
    """Hyperparameters for ListNet-style listwise training."""

    hidden_units: int = 16
    dropout: float = 0.1
    learning_rate: float = 1e-3
    list_batch_size: int = 256
    max_list_len: int = 100
    epochs: int = 20
    early_stopping_patience: int = 3
    random_seed: int = 2027


def listwise_rows_from_pools(
    pools: list[dict[str, Any]],
    *,
    pop_row: np.ndarray,
    app_to_row: dict[int, int],
    max_list_len: int = 100,
) -> pd.DataFrame:
    """One row per example: JSON lists of ``retr_score``, ``log_pop``, ``label``."""
    rows: list[dict[str, Any]] = []
    for row in pools:
        positives = set(int(x) for x in json.loads(row["validation_positive_app_ids_json"]))
        pool_apps = [int(x) for x in json.loads(row["retrieved_app_ids_json"])]
        retr = [float(x) for x in json.loads(row["retrieved_scores_json"])]
        pops = np.log1p(np.asarray([float(pop_row[app_to_row[int(a)]]) for a in pool_apps], dtype=np.float64))
        n = min(len(pool_apps), int(max_list_len))
        labels = [1.0 if int(pool_apps[i]) in positives else 0.0 for i in range(n)]
        rows.append(
            {
                "ex_idx": int(row["ex_idx"]),
                "retr_scores_json": json.dumps(retr[:n]),
                "log_pops_json": json.dumps(pops[:n].tolist()),
                "labels_json": json.dumps(labels),
                "list_len": n,
            }
        )
    return pd.DataFrame(rows)


def listnet_target_probs(labels: np.ndarray) -> np.ndarray:
    """Uniform distribution over positive labels; zeros if no positive."""
    y = np.asarray(labels, dtype=np.float64)
    pos_sum = float(y.sum())
    if pos_sum <= 0:
        return np.zeros_like(y)
    return y / pos_sum


def _arrays_from_listwise_df(df: pd.DataFrame, *, max_list_len: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(df)
    x = np.zeros((n, max_list_len, len(FEATURE_COLS)), dtype=np.float32)
    y = np.zeros((n, max_list_len), dtype=np.float32)
    for i, row in enumerate(df.to_dict(orient="records")):
        retr = json.loads(row["retr_scores_json"])
        pops = json.loads(row["log_pops_json"])
        lab = json.loads(row["labels_json"])
        length = min(int(row["list_len"]), len(retr), len(pops), len(lab), max_list_len)
        x[i, :length, 0] = retr[:length]
        x[i, :length, 1] = pops[:length]
        y[i, :length] = lab[:length]
    return x, y


def _listnet_loss(y_true, y_pred):
    """ListNet CE: softmax(scores) vs uniform distribution over positives."""
    tf = __import__("tensorflow")
    eps = 1e-10
    pos_sum = tf.reduce_sum(y_true, axis=-1)
    target = y_true / (tf.expand_dims(pos_sum, axis=-1) + eps)
    pred_prob = tf.nn.softmax(y_pred, axis=-1)
    ce = -tf.reduce_sum(target * tf.math.log(pred_prob + eps), axis=-1)
    has_pos = pos_sum > 0.5
    ce_masked = tf.boolean_mask(ce, has_pos)
    return tf.cond(
        tf.size(ce_masked) > 0,
        lambda: tf.reduce_mean(ce_masked),
        lambda: tf.constant(0.0),
    )


def build_listwise_model(*, max_list_len: int, cfg: ListwiseRankerConfig):
    """Shared MLP on each list position → length-``max_list_len`` score vector."""
    tf = __import__("tensorflow")
    keras = tf.keras
    inp = keras.layers.Input(shape=(int(max_list_len), len(FEATURE_COLS)), name="pool_features")
    x = keras.layers.Dense(cfg.hidden_units, activation="relu")(inp)
    x = keras.layers.Dropout(cfg.dropout)(x)
    scores = keras.layers.Dense(1)(x)
    scores = keras.layers.Reshape((int(max_list_len),))(scores)
    model = keras.Model(inp, scores, name="listwise_ranker_d3")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=cfg.learning_rate), loss=_listnet_loss)
    return model


def score_list(model, features: np.ndarray) -> np.ndarray:
    """Score variable-length pool: ``features`` shape ``(n_items, 2)``."""
    feats = np.asarray(features, dtype=np.float32)
    if feats.ndim != 2 or feats.shape[0] == 0:
        return np.asarray([], dtype=np.float64)
    max_len = int(model.input_shape[1])
    n_use = min(int(feats.shape[0]), max_len)
    padded = np.zeros((1, max_len, feats.shape[1]), dtype=np.float32)
    padded[0, :n_use, :] = feats[:n_use]
    preds = model.predict(padded, verbose=0)
    return np.asarray(preds[0, :n_use], dtype=np.float64)


def train_listwise_ranker(
    *,
    fit_df: pd.DataFrame,
    tune_pools: list[dict[str, Any]],
    app_ids: np.ndarray,
    app_to_row: dict[int, int],
    pop_row: np.ndarray,
    tune_df: pd.DataFrame | None = None,
    cfg: ListwiseRankerConfig | None = None,
    k_final: int = 10,
    verbose: int = 0,
) -> tuple[Any, pd.DataFrame]:
    """Train ListNet on padded lists; early-stop on ``train_tune`` mean NDCG@k."""
    cfg = cfg or ListwiseRankerConfig()
    tf = __import__("tensorflow")
    tf.random.set_seed(int(cfg.random_seed))

    max_len = int(cfg.max_list_len)
    x_fit, y_fit = _arrays_from_listwise_df(fit_df, max_list_len=max_len)
    tune_df = tune_df if tune_df is not None else listwise_rows_from_pools(
        tune_pools, pop_row=pop_row, app_to_row=app_to_row, max_list_len=max_len
    )
    x_tune, y_tune = _arrays_from_listwise_df(tune_df, max_list_len=max_len)

    model = build_listwise_model(max_list_len=max_len, cfg=cfg)

    def _score_pool(pool_apps: list[int], retr: list[float]) -> np.ndarray:
        pops = np.log1p(np.asarray([float(pop_row[app_to_row[int(a)]]) for a in pool_apps], dtype=np.float64))
        feats = np.column_stack([np.asarray(retr, dtype=np.float32), pops.astype(np.float32)])
        return score_list(model, feats)

    best_ndcg = -np.inf
    best_weights: list[Any] | None = None
    wait = 0
    history_rows: list[dict[str, float | int]] = []

    for epoch in range(int(cfg.epochs)):
        hist = model.fit(
            x_fit,
            y_fit,
            batch_size=int(cfg.list_batch_size),
            epochs=1,
            verbose=verbose,
        )
        train_loss = float(hist.history["loss"][0])
        tune_loss = float(model.evaluate(x_tune, y_tune, verbose=0))
        tune_ndcg = mean_pool_ndcg(
            tune_pools,
            _score_pool,
            app_ids=app_ids,
            app_to_row=app_to_row,
            k_final=k_final,
        )
        history_rows.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "tune_loss": tune_loss,
                "tune_NDCG": tune_ndcg,
            }
        )
        print(
            f"D3 epoch {epoch + 1}/{int(cfg.epochs)}: "
            f"train_loss={train_loss:.4f} tune_loss={tune_loss:.4f} "
            f"tune_NDCG@{k_final}={tune_ndcg:.4f}",
            flush=True,
        )

        if tune_ndcg > best_ndcg:
            best_ndcg = tune_ndcg
            best_weights = model.get_weights()
            wait = 0
        else:
            wait += 1
            if wait >= int(cfg.early_stopping_patience):
                break

    if best_weights is not None:
        model.set_weights(best_weights)

    return model, pd.DataFrame(history_rows)


def load_listwise_ranker(model_path: Path):
    tf = __import__("tensorflow")
    return tf.keras.models.load_model(
        model_path,
        compile=False,
        custom_objects={"_listnet_loss": _listnet_loss},
    )


def make_pool_score_fn(
    model,
    *,
    pop_row: np.ndarray,
    app_to_row: dict[int, int],
) -> Callable[..., np.ndarray]:
    def score(pool_app_ids: list[int], retrieval_scores: list[float] | np.ndarray, **_ignored) -> np.ndarray:
        retr = np.asarray(retrieval_scores, dtype=np.float64)
        pops = np.log1p(
            np.asarray([float(pop_row[app_to_row[int(a)]]) for a in pool_app_ids], dtype=np.float64)
        )
        feats = np.column_stack([retr.astype(np.float32), pops.astype(np.float32)])
        return score_list(model, feats)

    return score
