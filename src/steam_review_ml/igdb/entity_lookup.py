"""IGDB entity lookup tables and enriched games FK resolution."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from steam_review_ml.igdb.client import IGDBClient
from steam_review_ml.igdb.constants import (
    TAXONOMY_FETCH_SCOPE,
    TAXONOMY_FIELD_ENDPOINTS,
    TAXONOMY_NAMES_POOLED_USE_SUFFIX,
    TAXONOMY_NAMES_SUFFIX,
    USE_EMBEDDING_FIELD_SUFFIX,
)
from steam_review_ml.igdb.text_embeddings import embed_text_columns, use_embedding_column
from steam_review_ml.recommender.math_utils import l2_normalize

logger = logging.getLogger(__name__)

FULL_TABLE_FETCH_LIMIT = 500
TAXONOMY_NAME_COLUMN = "name"


def _as_id_list(value: Any) -> list[int]:
    """Normalize a parquet FK cell (None, scalar, list, or ndarray) to a list of integer ids."""
    if value is None:
        return []
    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple, set)):
        out: list[int] = []
        for item in value:
            if item is None:
                continue
            if isinstance(item, float) and np.isnan(item):
                continue
            out.append(int(item))
        return out
    return [int(value)]


def collect_unique_ids(df: pd.DataFrame, field: str) -> set[int]:
    if field not in df.columns:
        return set()
    ids: set[int] = set()
    for value in df[field]:
        ids.update(_as_id_list(value))
    return ids


def taxonomy_lookup_path(lookups_dir: Path, field: str) -> Path:
    endpoint = TAXONOMY_FIELD_ENDPOINTS[field]
    return lookups_dir / f"{endpoint}.parquet"


def taxonomy_names_column(field: str) -> str:
    return f"{field}{TAXONOMY_NAMES_SUFFIX}"


def taxonomy_names_embedding_column(field: str) -> str:
    return f"{field}{TAXONOMY_NAMES_SUFFIX}{USE_EMBEDDING_FIELD_SUFFIX}"


def taxonomy_names_embedding_pooled_column(field: str) -> str:
    return f"{field}{TAXONOMY_NAMES_SUFFIX}{TAXONOMY_NAMES_POOLED_USE_SUFFIX}"


def mean_pool_entity_embeddings(vecs: Sequence[Any]) -> list[Any] | np.ndarray:
    """Mean-pool per-entity USE vectors to one L2-normalized vector; ``[]`` when empty."""
    if not vecs:
        return []
    stacked = np.stack([np.asarray(v, dtype=np.float32).ravel() for v in vecs], axis=0)
    return l2_normalize(stacked.mean(axis=0))


def fetch_entity_table(
    client: IGDBClient,
    endpoint: str,
    ids: list[int] | None,
    *,
    batch_size: int = 100,
    full_table_limit: int = FULL_TABLE_FETCH_LIMIT,
) -> pd.DataFrame:
    """Fetch id and name from IGDB. Paginate the full table when ids is None; otherwise batch by id list."""
    rows: list[dict[str, Any]] = []
    if ids is not None and not ids:
        return pd.DataFrame(columns=["id", "name"])

    if ids is None:
        offset = 0
        while True:
            body = f"fields id,name; limit {full_table_limit}; offset {offset};"
            payload = client.post(endpoint, body)
            if not payload:
                break
            rows.extend(payload)
            if len(payload) < full_table_limit:
                break
            offset += full_table_limit
            logger.info("entity %s full fetch offset=%s +%s", endpoint, offset, len(payload))
    else:
        sorted_ids = sorted(set(ids))
        for start in range(0, len(sorted_ids), batch_size):
            batch = sorted_ids[start : start + batch_size]
            id_list = ",".join(str(i) for i in batch)
            body = f"fields id,name; where id = ({id_list}); limit {len(batch)};"
            payload = client.post(endpoint, body)
            rows.extend(payload)
            logger.info(
                "entity %s batch %s: +%s",
                endpoint,
                start // batch_size + 1,
                len(payload),
            )

    if not rows:
        return pd.DataFrame(columns=["id", "name"])

    out = pd.DataFrame(rows).drop_duplicates(subset=["id"], keep="first")
    out["id"] = out["id"].astype(int)
    out["name"] = out["name"].fillna("").astype(str)
    return out[["id", "name"]].sort_values("id").reset_index(drop=True)


def embed_lookup_names(
    lookup_df: pd.DataFrame,
    *,
    embed_fn: Callable[[list[str]], Any],
    batch_size: int = 64,
    max_chars_per_field: int | None = 8000,
    text_column: str = TAXONOMY_NAME_COLUMN,
) -> pd.DataFrame:
    """L2-normalize USE embeddings for the lookup name column; empty tables get an empty name__use series."""
    if lookup_df.empty:
        out = lookup_df.copy()
        out[use_embedding_column(text_column)] = pd.Series(dtype=object)
        return out
    return embed_text_columns(
        lookup_df,
        [text_column],
        embed_fn=embed_fn,
        batch_size=batch_size,
        max_chars_per_field=max_chars_per_field,
    )


def build_lookup_index(lookup_df: pd.DataFrame) -> dict[int, dict[str, Any]]:
    index: dict[int, dict[str, Any]] = {}
    for _, row in lookup_df.iterrows():
        index[int(row["id"])] = row.to_dict()
    return index


def resolve_fk_arrays(
    id_list: list[int],
    lookup: dict[int, dict[str, Any]],
    *,
    name_col: str = TAXONOMY_NAME_COLUMN,
) -> tuple[list[str], list[Any]]:
    """Map FK ids to parallel name and embedding arrays; skip missing lookup ids."""
    names: list[str] = []
    vecs: list[Any] = []
    vec_col = use_embedding_column(name_col)
    for entity_id in id_list:
        row = lookup.get(entity_id)
        if row is None:
            continue
        names.append(str(row[name_col]))
        if vec_col in row and row[vec_col] is not None:
            vecs.append(row[vec_col])
    return names, vecs


def add_taxonomy_resolved_columns(
    games_df: pd.DataFrame,
    lookups: dict[str, pd.DataFrame],
    fields: Sequence[str],
) -> pd.DataFrame:
    """For each taxonomy field, map FK ids to names, per-entity ``*_names__use``, and pooled ``*_names__use_pooled``."""
    out = games_df.copy()
    for field in fields:
        if field not in out.columns:
            raise KeyError(f"Games dataframe missing FK field {field!r}")
        lookup_df = lookups[field]
        lookup_index = build_lookup_index(lookup_df)
        names_col = taxonomy_names_column(field)
        vecs_col = taxonomy_names_embedding_column(field)
        pooled_col = taxonomy_names_embedding_pooled_column(field)

        def _resolve(value: Any) -> tuple[list[str], list[Any]]:
            return resolve_fk_arrays(_as_id_list(value), lookup_index)

        resolved = out[field].map(_resolve)
        out[names_col] = resolved.map(lambda pair: pair[0])
        out[vecs_col] = resolved.map(lambda pair: pair[1])
        out[pooled_col] = out[vecs_col].map(mean_pool_entity_embeddings)
    return out


def build_enriched_games(
    games_df: pd.DataFrame,
    lookups: dict[str, pd.DataFrame],
    taxonomy_fields: Sequence[str],
) -> pd.DataFrame:
    return add_taxonomy_resolved_columns(games_df, lookups, taxonomy_fields)


def load_taxonomy_lookups(lookups_dir: Path, fields: Sequence[str]) -> dict[str, pd.DataFrame]:
    lookups: dict[str, pd.DataFrame] = {}
    for field in fields:
        path = taxonomy_lookup_path(lookups_dir, field)
        if not path.is_file():
            raise FileNotFoundError(f"Missing taxonomy lookup: {path}")
        lookups[field] = pd.read_parquet(path)
    return lookups


def write_lookup_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info("Wrote lookup %s (%s rows)", path, len(df))


def build_lookup_meta(
    *,
    entity_counts: dict[str, int],
    model_name: str | None,
    taxonomy_fields: Sequence[str],
    embed_game_text_fields: Sequence[str],
    embed_taxonomy_names: bool,
) -> dict[str, Any]:
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "taxonomy_fields": list(taxonomy_fields),
        "embed_game_text_fields": list(embed_game_text_fields),
        "embed_taxonomy_names": embed_taxonomy_names,
        "entity_row_counts": entity_counts,
    }


def write_lookup_meta(path: Path, meta: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _ids_for_taxonomy_field(games_df: pd.DataFrame, field: str) -> list[int] | None:
    scope = TAXONOMY_FETCH_SCOPE.get(field, "from_games")
    if scope == "full":
        return None
    return sorted(collect_unique_ids(games_df, field))


def fetch_taxonomy_lookups(
    client: IGDBClient | None,
    games_df: pd.DataFrame,
    fields: Sequence[str],
    lookups_dir: Path,
    *,
    skip_fetch: bool,
    entity_batch_size: int,
    embed_fn: Callable[[list[str]], Any] | None,
    embed_taxonomy_names: bool,
    embed_batch_size: int,
    max_chars_per_field: int | None,
) -> dict[str, int]:
    """Fetch or reload per-field taxonomy parquets, optionally embed names, write lookups, return row counts."""
    entity_counts: dict[str, int] = {}
    for field in fields:
        path = taxonomy_lookup_path(lookups_dir, field)
        endpoint = TAXONOMY_FIELD_ENDPOINTS[field]
        if client is not None and not skip_fetch:
            ids = _ids_for_taxonomy_field(games_df, field)
            table = fetch_entity_table(
                client,
                endpoint,
                ids,
                batch_size=entity_batch_size,
            )
        else:
            if not path.is_file():
                raise FileNotFoundError(
                    f"skip_fetch=True but taxonomy lookup missing: {path}"
                )
            table = pd.read_parquet(path)
            name_use = use_embedding_column(TAXONOMY_NAME_COLUMN)
            if name_use in table.columns:
                table = table.drop(columns=[name_use])

        if embed_taxonomy_names:
            if embed_fn is None:
                raise ValueError("embed_taxonomy_names=True requires embed_fn.")
            table = embed_lookup_names(
                table,
                embed_fn=embed_fn,
                batch_size=embed_batch_size,
                max_chars_per_field=max_chars_per_field,
            )

        write_lookup_parquet(table, path)
        entity_counts[field] = len(table)
    return entity_counts
