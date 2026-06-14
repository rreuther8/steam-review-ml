"""D2 pointwise pool ranker: (retr_score, log_pop) -> relevance score.

Status: **killed** (2026-06-07). Not in shipped stack; not wired to ``pool_rerank_registry``.
Notebook: ``notebooks/ranking/recs_014_ranker_d2_d3_train_head_to_head.ipynb``.
See ``docs/ranking_decision_log.md`` § 2026-06-07.

Used by ``recs_014_ranker_d2_d3_train_head_to_head.ipynb``:
expand frozen ``two_tower_v1`` pools into pair rows, train a small MLP with
class-weighted BCE, early-stop on inner-val (train tune split) mean NDCG@k,
then score val pools via :func:`make_pool_score_fn`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from steam_review_ml.evaluation.retrieval_offline_eval import (
    _rank_rows,
    hit_rate_at_k,
    ndcg_at_k,
)

METHOD_TWO_TOWER_V1_RANKER_D2_POINTWISE_V1 = "two_tower_v1_ranker_d2_pointwise_v1"

FEATURE_COLS = ("retr_score", "log_pop")


@dataclass(frozen=True)
class PointwiseRankerConfig:
    """Hyperparameters for the pointwise MLP and training loop."""

    hidden_units: int = 16
    dropout: float = 0.1
    learning_rate: float = 1e-3
    batch_size: int = 4096
    epochs: int = 20
    early_stopping_patience: int = 3
    random_seed: int = 2027


def pointwise_rows_from_pools(
    pools: list[dict[str, Any]],
    *,
    pop_row: np.ndarray,
    app_to_row: dict[int, int],
) -> pd.DataFrame:
    """Expand pool rows into one training row per (candidate, example).

    Each output row has ``retr_score``, ``log_pop`` (log1p train popularity),
    and ``label`` (1 if the candidate is in ``validation_positive_app_ids_json``).
    Pools come from train parquet or any dict with the same JSON fields as
    ``load_retrieval_pool_rows``.
    """
    rows: list[dict[str, float | int]] = []
    for row in pools:
        positives = set(int(x) for x in json.loads(row["validation_positive_app_ids_json"]))
        pool_apps = [int(x) for x in json.loads(row["retrieved_app_ids_json"])]
        retr = [float(x) for x in json.loads(row["retrieved_scores_json"])]
        pops = np.log1p(np.asarray([float(pop_row[app_to_row[int(a)]]) for a in pool_apps], dtype=np.float64))
        for app_id, r, p in zip(pool_apps, retr, pops.tolist()):
            rows.append(
                {
                    "ex_idx": int(row["ex_idx"]),
                    "retr_score": float(r),
                    "log_pop": float(p),
                    "label": 1.0 if int(app_id) in positives else 0.0,
                }
            )
    return pd.DataFrame(rows)


def _feature_matrix(df: pd.DataFrame) -> np.ndarray:
    return df[list(FEATURE_COLS)].to_numpy(dtype=np.float32)


def _class_weights(labels: np.ndarray) -> dict[int, float]:
    n_pos = float(np.sum(labels > 0.5))
    n_neg = float(len(labels) - n_pos)
    if n_pos <= 0 or n_neg <= 0:
        return {0: 1.0, 1: 1.0}
    return {0: 1.0, 1: n_neg / n_pos}


def _sample_weights(labels: np.ndarray, class_weight: dict[int, float]) -> np.ndarray:
    """Per-row weights matching Keras ``class_weight`` (for ``model.evaluate``)."""
    return np.where(labels > 0.5, float(class_weight[1]), float(class_weight[0])).astype(np.float32)


def build_pointwise_model(*, n_features: int, cfg: PointwiseRankerConfig):
    """Small MLP: Input -> Dense(relu) -> Dropout -> Dense(sigmoid); BCE loss."""
    tf = __import__("tensorflow")
    keras = tf.keras
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(n_features,)),
            keras.layers.Dense(cfg.hidden_units, activation="relu"),
            keras.layers.Dropout(cfg.dropout),
            keras.layers.Dense(1, activation="sigmoid"),
        ],
        name="pointwise_ranker_d2",
    )
    # Early stopping uses tune-pool NDCG, not Keras metrics (Auc vs AUC differs across TF/Keras versions).
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=cfg.learning_rate),
        loss="binary_crossentropy",
    )
    return model


def score_pairs(model, features: np.ndarray) -> np.ndarray:
    """Predict relevance scores for an (N, 2) feature matrix."""
    preds = model.predict(features, verbose=0)
    return np.asarray(preds, dtype=np.float64).reshape(-1)


def mean_pool_ndcg(
    pools: list[dict[str, Any]],
    score_fn: Callable[[list[int], list[float]], np.ndarray],
    *,
    app_ids: np.ndarray,
    app_to_row: dict[int, int],
    k_final: int,
) -> float:
    """Mean NDCG@k over pools: score_fn reranks each pool, then global top-k."""
    vals: list[float] = []
    for row in pools:
        positives = set(int(x) for x in json.loads(row["validation_positive_app_ids_json"]))
        if not positives:
            continue
        pool_apps = [int(x) for x in json.loads(row["retrieved_app_ids_json"])]
        retr = [float(x) for x in json.loads(row["retrieved_scores_json"])]
        scores = score_fn(pool_apps, retr)
        full = np.full(len(app_ids), -np.inf, dtype=np.float64)
        for app_id, score in zip(pool_apps, scores):
            full[int(app_to_row[int(app_id)])] = float(score)
        ranked = _rank_rows(full)[:k_final]
        vals.append(ndcg_at_k(ranked, positives, k_final, app_ids))
    return float(np.mean(vals)) if vals else float("nan")


def train_pointwise_ranker(
    *,
    fit_df: pd.DataFrame,
    tune_pools: list[dict[str, Any]],
    app_ids: np.ndarray,
    app_to_row: dict[int, int],
    pop_row: np.ndarray,
    tune_df: pd.DataFrame | None = None,
    cfg: PointwiseRankerConfig | None = None,
    k_final: int = 10,
    verbose: int = 0,
) -> tuple[Any, pd.DataFrame]:
    """Train on pair rows; early-stop on tune-pool mean NDCG@k.

    ``fit_df`` — pair rows for training (e.g. 90% of train ranker pools).
    ``tune_pools`` — example-level holdout from the **same train cohort** (not
    external val); used for tune BCE and ranking NDCG each epoch.
    ``tune_df`` — optional pair rows for tune; built from ``tune_pools`` if omitted.

    Each epoch: one ``model.fit`` on fit, ``model.evaluate`` on tune (BCE),
    ``mean_pool_ndcg`` on tune pools. Early stopping uses tune NDCG only.
    Prints train_loss, tune_loss, and tune_NDCG per epoch.

    Returns ``(model, history)`` with columns
    ``epoch``, ``train_loss``, ``tune_loss``, ``tune_NDCG``.
    """
    cfg = cfg or PointwiseRankerConfig()
    tf = __import__("tensorflow")
    tf.random.set_seed(int(cfg.random_seed))

    x_fit = _feature_matrix(fit_df)
    y_fit = fit_df["label"].to_numpy(dtype=np.float32)
    tune_df = tune_df if tune_df is not None else pointwise_rows_from_pools(
        tune_pools, pop_row=pop_row, app_to_row=app_to_row
    )
    x_tune = _feature_matrix(tune_df)
    y_tune = tune_df["label"].to_numpy(dtype=np.float32)

    model = build_pointwise_model(n_features=x_fit.shape[1], cfg=cfg)
    class_weight = _class_weights(y_fit)
    tune_sample_weight = _sample_weights(y_tune, class_weight)

    def _score_pool(pool_apps: list[int], retr: list[float]) -> np.ndarray:
        pops = np.log1p(np.asarray([float(pop_row[app_to_row[int(a)]]) for a in pool_apps], dtype=np.float64))
        feats = np.column_stack([np.asarray(retr, dtype=np.float32), pops.astype(np.float32)])
        return score_pairs(model, feats)

    best_ndcg = -np.inf
    best_weights: list[Any] | None = None
    wait = 0
    history_rows: list[dict[str, float | int]] = []

    for epoch in range(int(cfg.epochs)):
        hist = model.fit(
            x_fit,
            y_fit,
            batch_size=int(cfg.batch_size),
            epochs=1,
            verbose=verbose,
            class_weight=class_weight,
        )
        train_loss = float(hist.history["loss"][0])
        # Keras evaluate does not accept class_weight; sample_weight matches fit weighting.
        tune_loss = float(
            model.evaluate(x_tune, y_tune, verbose=0, sample_weight=tune_sample_weight)
        )
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
            f"D2 epoch {epoch + 1}/{int(cfg.epochs)}: "
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


def load_pointwise_ranker(model_path: Path):
    """Load a Keras model saved by recs_014 (``compile=False``)."""
    tf = __import__("tensorflow")
    return tf.keras.models.load_model(model_path, compile=False)


def make_pool_score_fn(
    model,
    *,
    pop_row: np.ndarray,
    app_to_row: dict[int, int],
) -> Callable[..., np.ndarray]:
    """Score adapter for recs_014 val head-to-head (same call pattern as D1 logpop).

    ``(pool_app_ids, retrieval_scores, **kwargs) -> np.ndarray`` of per-candidate scores.
    """

    def score(pool_app_ids: list[int], retrieval_scores: list[float] | np.ndarray, **_ignored) -> np.ndarray:
        retr = np.asarray(retrieval_scores, dtype=np.float64)
        pops = np.log1p(
            np.asarray([float(pop_row[app_to_row[int(a)]]) for a in pool_app_ids], dtype=np.float64)
        )
        feats = np.column_stack([retr.astype(np.float32), pops.astype(np.float32)])
        return score_pairs(model, feats)

    return score
