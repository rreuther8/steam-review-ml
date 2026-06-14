"""Shared helpers for D6 rank-only models (frozen retrieve trunk + separate rank artifacts).

Status: **killed** (D6 spike; below D1 on val). Not in shipped stack.
Notebook: ``notebooks/ranking/recs_018_ranker_d6_frozen_trunk_spike.ipynb``.
Used by ``ranker_d6_rank_head.py`` and ``ranker_d6_biencoder.py`` (archival / replicability only).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

METHOD_TWO_TOWER_V1_RANKER_D6_RANK_HEAD_V1 = "two_tower_v1_ranker_d6_rank_head_v1"
METHOD_TWO_TOWER_V1_RANKER_D6_BIENCODER_V1 = "two_tower_v1_ranker_d6_biencoder_v1"


@dataclass(frozen=True)
class D6ListwiseConfig:
    """Hyperparameters for D6 listwise pool rankers."""

    hidden_units: int = 32
    dropout: float = 0.1
    learning_rate: float = 1e-3
    list_batch_size: int = 64
    max_list_len: int = 100
    epochs: int = 15
    early_stopping_patience: int = 3
    random_seed: int = 2028
    user_encode_batch: int = 64


def query_text_by_ex_idx(cohort_parquet: Path) -> dict[int, str]:
    """Map ``ex_idx`` → ``query_text`` from a cached example cohort parquet."""
    df = pd.read_parquet(cohort_parquet, columns=["ex_idx", "query_text"])
    return {int(row.ex_idx): str(row.query_text) for row in df.itertuples(index=False)}


def attach_query_text(
    pools: list[dict[str, Any]],
    query_texts: dict[int, str],
) -> list[dict[str, Any]]:
    """Attach ``query_text`` to pool rows (export parquet does not store it)."""
    out: list[dict[str, Any]] = []
    for row in pools:
        ex_idx = int(row["ex_idx"])
        if ex_idx not in query_texts:
            raise KeyError(f"ex_idx={ex_idx} missing from cohort query_text map")
        enriched = dict(row)
        enriched["query_text"] = query_texts[ex_idx]
        out.append(enriched)
    return out


def padded_pool_batch(
    pools: list[dict[str, Any]],
    *,
    app_to_row: dict[int, int],
    max_list_len: int,
) -> tuple[list[str], list[list[int]], list[list[float]], list[list[float]]]:
    """Build variable-length lists per pool (no padding) for one minibatch."""
    query_texts: list[str] = []
    item_row_ids: list[list[int]] = []
    retr_scores: list[list[float]] = []
    labels: list[list[float]] = []
    max_len = int(max_list_len)
    for row in pools:
        query_texts.append(str(row["query_text"]))
        pool_apps = [int(x) for x in json.loads(row["retrieved_app_ids_json"])]
        retr = [float(x) for x in json.loads(row["retrieved_scores_json"])]
        positives = {int(x) for x in json.loads(row["validation_positive_app_ids_json"])}
        length = min(len(pool_apps), max_len)
        rows_i: list[int] = []
        retr_i: list[float] = []
        lab_i: list[float] = []
        for j in range(length):
            app_id = int(pool_apps[j])
            rows_i.append(int(app_to_row[app_id]))
            retr_i.append(float(retr[j]))
            lab_i.append(1.0 if app_id in positives else 0.0)
        item_row_ids.append(rows_i)
        retr_scores.append(retr_i)
        labels.append(lab_i)
    return query_texts, item_row_ids, retr_scores, labels


def iter_pool_minibatches(
    pools: list[dict[str, Any]],
    *,
    batch_size: int,
    rng,
) -> list[list[dict[str, Any]]]:
    """Shuffle and chunk pool rows into minibatches."""
    order = rng.permutation(len(pools))
    shuffled = [pools[int(i)] for i in order]
    bs = max(1, int(batch_size))
    return [shuffled[i : i + bs] for i in range(0, len(shuffled), bs)]


def write_rank_manifest(
    path: Path,
    *,
    rank_method: str,
    retrieve_model_path: Path,
    rank_artifact_path: Path,
    rank_kind: str,
    extra: dict[str, Any] | None = None,
) -> None:
    """Sidecar JSON: rank artifact + read-only retrieve checkpoint path."""
    payload = {
        "rank_method": rank_method,
        "rank_kind": rank_kind,
        "rank_artifact_path": str(rank_artifact_path.resolve()),
        "retrieve_model_path": str(retrieve_model_path.resolve()),
        "retrieve_immutable": True,
    }
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
