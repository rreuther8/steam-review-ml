"""Train fine-tuned dual-tower retrieval (trainable USE user + trainable catalog item embeddings)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from steam_review_ml.recommender.contrastive_examples import positive_app_ids_from_example
from steam_review_ml.recommender.retrieve import ContentRetriever

PRODUCTION_SPEC_LABEL = "updated_user__updated_profile200_item"
TrainingMode = Literal["smoke", "full"]


@dataclass
class TwoTowerTrainConfig:
    training_mode: TrainingMode = "full"
    batch_size: int = 64
    proj_dim: int = 64
    temperature: float = 0.07
    epochs: int = 15
    smoke_epochs: int = 5
    early_stopping_patience: int = 3
    tower_seed: int = 2026
    adam_lr: float = 1e-3


def find_repo_root(start: Path | None = None) -> Path:
    here = (start or Path.cwd()).resolve()
    for parent_dir in (here, *here.parents):
        if (parent_dir / "pyproject.toml").is_file():
            return parent_dir
    raise RuntimeError(f"No pyproject.toml above {here}")


def load_hub_settings(retriever: ContentRetriever) -> tuple[str, int | None]:
    _, _, meta_path = retriever._artifact_paths()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return str(meta["model_name"]), meta.get("max_chars_per_review")


def clip_text(text: str, max_chars: int | None) -> str:
    cleaned = (text or "").strip()
    if max_chars is None:
        return cleaned
    return cleaned[: int(max_chars)]


def catalog_app_row_map(retriever: ContentRetriever) -> dict[int, int]:
    return {int(app_id): row_index for row_index, app_id in enumerate(np.asarray(retriever.app_ids).tolist())}


def item_init_matrix_from_catalog(retriever: ContentRetriever) -> np.ndarray:
    return np.asarray(retriever.embedding_matrix, dtype=np.float32)


def build_tower_contrastive_rows(
    examples: list[dict],
    retriever: ContentRetriever,
    *,
    max_examples: int,
    max_chars: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One row per (query, positive app) for updated-user training."""
    app_to_row = catalog_app_row_map(retriever)
    user_texts: list[str] = []
    item_row_ids: list[int] = []
    group_ids: list[int] = []
    group_id = 0
    for ex in examples[: int(max_examples)]:
        positives = [a for a in positive_app_ids_from_example(ex) if a in app_to_row]
        if not positives:
            continue
        query_text = clip_text(str(ex["query_text"]), max_chars)
        for app_id in positives:
            user_texts.append(query_text)
            item_row_ids.append(int(app_to_row[int(app_id)]))
            group_ids.append(group_id)
        group_id += 1
    if len(item_row_ids) < 2:
        raise RuntimeError("Need >= 2 contrastive rows after catalog join.")
    return (
        np.asarray(user_texts, dtype=object),
        np.asarray(item_row_ids, dtype=np.int32),
        np.asarray(group_ids, dtype=np.int32),
    )


def _tf():
    import tensorflow as tf

    return tf


def multi_positive_inbatch_loss(similarity_logits, group_ids):
    tf = _tf()
    group_ids_column = tf.expand_dims(group_ids, 1)
    group_ids_row = tf.expand_dims(group_ids, 0)
    positive_mask = tf.cast(tf.equal(group_ids_column, group_ids_row), tf.float32)
    log_normalizer = tf.reduce_logsumexp(similarity_logits, axis=1)
    masked_logits = similarity_logits + (positive_mask - 1.0) * 1e9
    log_positive_sum = tf.reduce_logsumexp(masked_logits, axis=1)
    return tf.reduce_mean(-(log_positive_sum - log_normalizer))


def contrastive_loss_from_vectors(user_vectors, item_vectors, group_ids, temperature):
    tf = _tf()
    user_norm = tf.math.l2_normalize(user_vectors, axis=1)
    item_norm = tf.math.l2_normalize(item_vectors, axis=1)
    logits = tf.matmul(user_norm, item_norm, transpose_b=True) / temperature
    user_to_item = multi_positive_inbatch_loss(logits, group_ids)
    item_to_user = multi_positive_inbatch_loss(tf.transpose(logits), group_ids)
    total = 0.5 * (user_to_item + item_to_user)
    return total, user_to_item, item_to_user


def make_updated_user_dataset(user_texts, item_row_ids, group_ids, batch_size, shuffle_seed):
    tf = _tf()
    return (
        tf.data.Dataset.from_tensor_slices(
            (
                tf.constant(list(user_texts), dtype=tf.string),
                item_row_ids.astype(np.int32),
                group_ids.astype(np.int32),
            )
        )
        .shuffle(int(len(user_texts)), seed=shuffle_seed)
        .batch(batch_size, drop_remainder=True)
        .map(
            lambda user_batch, item_row_batch, group_batch: {
                "user_text": user_batch,
                "item_row_id": item_row_batch,
                "group_id": group_batch,
            }
        )
    )


def _keras_hub():
    import tensorflow as tf
    import tensorflow_hub as hub
    from tensorflow import keras

    return tf, hub, keras


def _register_keras_serializable(keras: Any, *, package: str, name: str):
    """TF-integrated Keras (no keras.saving) vs Keras 3."""
    saving = getattr(keras, "saving", None)
    if saving is not None and hasattr(saving, "register_keras_serializable"):
        return saving.register_keras_serializable(package=package, name=name)
    return keras.utils.register_keras_serializable(package=package, name=name)


def keras_custom_object_scope(keras: Any, custom_objects: dict[str, Any]):
    saving = getattr(keras, "saving", None)
    if saving is not None and hasattr(saving, "custom_object_scope"):
        return saving.custom_object_scope(custom_objects)
    return keras.utils.custom_object_scope(custom_objects)


TwoTowerModel: type | None = None


def _init_two_tower_model_class() -> type:
    global TwoTowerModel
    if TwoTowerModel is not None:
        return TwoTowerModel

    tf, hub, keras = _keras_hub()
    default_hub = "https://tfhub.dev/google/universal-sentence-encoder/4"

    serializable = _register_keras_serializable(keras, package="TwoTower", name="TwoTowerModel")

    @serializable
    class _TwoTowerModel(keras.Model):
        def __init__(
            self,
            hub_url: str = default_hub,
            projection_dim: int = 64,
            temperature: float = 0.07,
            n_items: int | None = None,
            embed_dim: int | None = None,
            item_init_matrix: np.ndarray | None = None,
            **kwargs: Any,
        ):
            model_name = str(kwargs.pop("name", "two_tower"))
            super().__init__(name=model_name, **kwargs)
            self._hub_url = str(hub_url)
            self._projection_dim = int(projection_dim)
            self._temperature = float(temperature)
            self.user_encoder = hub.KerasLayer(self._hub_url, trainable=True, name="user_use")
            self.user_projection = keras.layers.Dense(self._projection_dim)
            self.item_projection = keras.layers.Dense(self._projection_dim)
            if item_init_matrix is not None:
                init = np.asarray(item_init_matrix, dtype=np.float32)
                n_items, embed_dim = int(init.shape[0]), int(init.shape[1])
                initializer: Any = keras.initializers.Constant(init)
            else:
                if n_items is None or embed_dim is None:
                    raise ValueError("n_items and embed_dim are required when item_init_matrix is omitted.")
                initializer = "zeros"
            self._n_items = int(n_items)
            self._embed_dim = int(embed_dim)
            self.item_base = self.add_weight(
                name="item_base_embeddings",
                shape=(self._n_items, self._embed_dim),
                initializer=initializer,
                trainable=True,
            )

        def get_config(self) -> dict[str, Any]:
            config = super().get_config()
            config.update(
                {
                    "hub_url": self._hub_url,
                    "projection_dim": self._projection_dim,
                    "temperature": self._temperature,
                    "n_items": self._n_items,
                    "embed_dim": self._embed_dim,
                }
            )
            return config

        @classmethod
        def from_config(cls, config: dict[str, Any]):
            config = dict(config)
            config.pop("name", None)
            return cls(**config)

        def build(self, input_shape=None):
            tf = _tf()
            self.encode_user(tf.constant(["."], dtype=tf.string))
            self.encode_item_by_row_ids(tf.constant([0], dtype=tf.int32))
            super().build(input_shape)

        def encode_user(self, texts):
            with tf.device("/CPU:0"):
                encoded = self.user_encoder(texts)
            return self.user_projection(encoded)

        def encode_item_by_row_ids(self, item_row_ids):
            item_base_vectors = tf.gather(self.item_base, item_row_ids)
            return self.item_projection(item_base_vectors)

        def train_step(self, batch):
            with tf.GradientTape() as tape:
                user_vectors = self.encode_user(batch["user_text"])
                item_vectors = self.encode_item_by_row_ids(batch["item_row_id"])
                loss, u2i, i2u = contrastive_loss_from_vectors(
                    user_vectors, item_vectors, batch["group_id"], self._temperature
                )
            grads = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
            return {"loss": loss, "user_to_item_loss": u2i, "item_to_user_loss": i2u}

        def test_step(self, batch):
            user_vectors = self.encode_user(batch["user_text"])
            item_vectors = self.encode_item_by_row_ids(batch["item_row_id"])
            loss, u2i, i2u = contrastive_loss_from_vectors(
                user_vectors, item_vectors, batch["group_id"], self._temperature
            )
            return {"loss": loss, "user_to_item_loss": u2i, "item_to_user_loss": i2u}

    import sys

    TwoTowerModel = _TwoTowerModel
    sys.modules[__name__].TwoTowerModel = _TwoTowerModel
    return TwoTowerModel


def __getattr__(name: str) -> Any:
    if name == "TwoTowerModel":
        return _init_two_tower_model_class()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def tower_fit_kwargs(
    cfg: TwoTowerTrainConfig, n_train_rows: int, n_val_rows: int, batch_size: int
) -> dict[str, int]:
    if cfg.training_mode == "smoke":
        return {
            "epochs": int(cfg.smoke_epochs),
            "steps_per_epoch": 1,
            "validation_steps": 1,
        }
    steps_train = max(1, int(n_train_rows) // int(batch_size))
    steps_val = max(1, int(n_val_rows) // int(batch_size))
    return {
        "epochs": int(cfg.epochs),
        "steps_per_epoch": steps_train,
        "validation_steps": steps_val,
    }


def train_two_tower(
    train_payload: tuple[np.ndarray, np.ndarray, np.ndarray],
    val_payload: tuple[np.ndarray, np.ndarray, np.ndarray],
    cfg: TwoTowerTrainConfig,
    *,
    hub_url: str,
    item_init_matrix: np.ndarray,
):
    import os

    import tensorflow as tf
    from tensorflow import keras

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    tf.config.optimizer.set_jit(False)

    batch_size = int(cfg.batch_size)
    projection_dim = int(cfg.proj_dim)
    temperature = float(cfg.temperature)
    seed = int(cfg.tower_seed)

    model_cls = _init_two_tower_model_class()
    model = model_cls(
        hub_url=hub_url,
        projection_dim=projection_dim,
        temperature=temperature,
        item_init_matrix=item_init_matrix,
    )
    train_ds = make_updated_user_dataset(*train_payload, batch_size, seed)
    val_ds = make_updated_user_dataset(*val_payload, batch_size, seed + 1)

    train_row_count = len(train_payload[0])
    val_row_count = len(val_payload[0])
    if train_row_count < batch_size or val_row_count < batch_size:
        raise RuntimeError(
            f"Need >= {batch_size} train/val rows; got train={train_row_count}, val={val_row_count}."
        )

    fit_kwargs = tower_fit_kwargs(cfg, train_row_count, val_row_count, batch_size)
    print(
        f"{PRODUCTION_SPEC_LABEL}: train rows={train_row_count} val rows={val_row_count} "
        f"user_encoder_trainable=True item_encoder_trainable=True mode={cfg.training_mode} "
        f"steps/epoch={fit_kwargs['steps_per_epoch']} val_steps={fit_kwargs['validation_steps']} "
        f"epochs={fit_kwargs['epochs']}"
    )
    callbacks: list[keras.callbacks.Callback] = []
    if cfg.training_mode == "full":
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=int(cfg.early_stopping_patience),
                restore_best_weights=True,
            )
        )
    model.compile(optimizer=keras.optimizers.Adam(cfg.adam_lr), jit_compile=False)
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=1,
        **fit_kwargs,
    )
    if not model.built:
        model.build()
    return model, history


def history_to_dataframe(history) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    n_epochs = len(history.history.get("loss", []))
    for epoch_idx in range(n_epochs):
        row: dict[str, Any] = {"epoch": epoch_idx + 1}
        for key, values in history.history.items():
            if epoch_idx < len(values):
                row[key] = float(values[epoch_idx])
        rows.append(row)
    return pd.DataFrame(rows)


def print_loss_summary(history) -> None:
    hist = history.history
    if not hist.get("loss"):
        print("No training history to summarize.")
        return
    n_epochs = len(hist["loss"])
    last = n_epochs - 1
    val_losses = hist.get("val_loss", [])
    best_val_epoch = int(np.argmin(val_losses)) if val_losses else last
    print(f"\n=== Training summary ({PRODUCTION_SPEC_LABEL}) ===")
    print(f"epochs completed: {n_epochs}")
    print(
        f"final epoch — loss: {hist['loss'][last]:.4f}  val_loss: {val_losses[last]:.4f}"
        if val_losses
        else f"final epoch — loss: {hist['loss'][last]:.4f}"
    )
    if val_losses:
        print(
            f"best val_loss: {val_losses[best_val_epoch]:.4f} at epoch {best_val_epoch + 1}"
        )
    for key in ("user_to_item_loss", "item_to_user_loss", "val_user_to_item_loss", "val_item_to_user_loss"):
        if key in hist and len(hist[key]) > last:
            print(f"  {key} (final): {hist[key][last]:.4f}")
