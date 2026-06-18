"""Fetch IGDB game metadata and join to the Steam recommender catalog."""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from dotenv import load_dotenv
from pathlib import Path
from typing import Any
import pandas as pd

from steam_review_ml.igdb.client import IGDBClient
from steam_review_ml.igdb.constants import (
    IGDB_GAMES_PARQUET,
    STEAM_EXTERNAL_GAME_SOURCE,
    STEAM_JOIN_COLS,
    V2_CORE_FIELDS,
    resolve_game_fields,
)

from steam_review_ml.igdb.job_config import IgdbGamesJobConfig
from steam_review_ml.igdb.mocks import apply_mock_rows, load_mock_rows

logger = logging.getLogger(__name__)


def load_twitch_credentials(*, repo_root: Path | None = None) -> tuple[str, str]:
    """Load Twitch app credentials from environment (optional repo-root .env)."""
    if repo_root is not None:
        try:
            load_dotenv(repo_root / ".env")
        except ImportError:
            pass

    client_id = os.getenv("TWITCH_CLIENT_ID", "").strip()
    client_secret = os.getenv("TWITCH_CLIENT_SECRET", "").strip()
    if not client_id or not client_secret:
        raise RuntimeError(
            "Missing TWITCH_CLIENT_ID or TWITCH_CLIENT_SECRET. "
            "Set them in the environment or repo-root .env."
        )
    return client_id, client_secret


def normalize_title(text: str) -> str:
    s = str(text).lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())


def resolve_game_index_path(repo_root: Path) -> Path:
    candidates = [
        repo_root
        / "artifacts/recs/embeddings/game_profile/default/game_profile_embedding_index.parquet",
        repo_root / "artifacts/recs/game_profile_embedding_index.parquet",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "Missing game index parquet. Run recs_job_game_embeddings first."
    )


def load_steam_catalog(index_path: Path) -> pd.DataFrame:
    idx_df = pd.read_parquet(index_path)
    name_col = (
        "app_name"
        if "app_name" in idx_df.columns
        else ("name" if "name" in idx_df.columns else None)
    )
    if name_col is None:
        raise KeyError(f"Index missing app name column. Columns={list(idx_df.columns)}")

    catalog = (
        idx_df[[ "app_id", name_col]]
        .dropna(subset=["app_id", name_col])
        .drop_duplicates(subset=["app_id"])
        .rename(columns={name_col: "app_name"})
        .assign(
            app_id=lambda d: d["app_id"].astype(int),
            app_name=lambda d: d["app_name"].astype(str),
        )
        .sort_values("app_id")
        .reset_index(drop=True)
    )
    catalog["app_name_norm"] = catalog["app_name"].map(normalize_title)
    return catalog


def load_eval_query_app_ids(eval_parquet_path: Path | None) -> set[int]:
    if eval_parquet_path is None or not eval_parquet_path.is_file():
        return set()
    eval_df = pd.read_parquet(eval_parquet_path, columns=["query_app_id"])
    return set(eval_df["query_app_id"].dropna().astype(int).tolist())


def lookup_external_games_by_steam_uids(
    client: IGDBClient,
    uids: list[str],
    *,
    steam_external_game_source: int = STEAM_EXTERNAL_GAME_SOURCE,
) -> pd.DataFrame:
    if not uids:
        return pd.DataFrame(columns=["app_id", "igdb_game_id", "join_method"])

    uid_list = ",".join(f'"{uid}"' for uid in uids)
    body = (
        f"fields game, uid, external_game_source; "
        f"where external_game_source = {steam_external_game_source} & uid = ({uid_list}); "
        f"limit {len(uids)};"
    )
    rows = client.post("external_games", body)

    out_rows: list[dict[str, Any]] = []
    for row in rows:
        uid = str(row.get("uid", "")).strip()
        if not uid.isdigit():
            continue
        out_rows.append(
            {
                "app_id": int(uid),
                "igdb_game_id": int(row["game"]),
                "join_method": "external_games",
            }
        )
    return pd.DataFrame(out_rows).drop_duplicates(subset=["app_id"], keep="first")


def lookup_igdb_id_by_name(client: IGDBClient, app_name: str) -> int | None:
    target = normalize_title(app_name)
    if not target:
        return None

    safe_name = app_name.replace('"', "'")
    body = f'search "{safe_name}"; fields id,name; limit 8;'
    hits = client.post("games", body)
    for hit in hits:
        if normalize_title(str(hit.get("name", ""))) == target:
            return int(hit["id"])
    return None


def build_join_map(
    client: IGDBClient,
    steam_catalog: pd.DataFrame,
    *,
    external_batch_size: int,
    max_name_lookups: int,
    steam_external_game_source: int = STEAM_EXTERNAL_GAME_SOURCE,
) -> pd.DataFrame:
    """
    Build a join map between Steam app IDs and IGDB game IDs.

    This function attempts to match Steam games to IGDB games, preferring a robust match
    via IGDB's `external_games` table (using Steam app IDs), optionally falling back to
    name-based lookup for unresolved games.

    Args:
        client (IGDBClient): Authenticated IGDB API client.
        steam_catalog (pd.DataFrame): Steam games catalog; must include 'app_id' and 'app_name' columns.
        external_batch_size (int): Batch size for querying external_games lookup by app_id.
        max_name_lookups (int): Maximum number of unresolved games to attempt via name lookup.
        steam_external_game_source (int): External game source ID for Steam in IGDB (default is 1).

    Returns:
        pd.DataFrame: DataFrame with columns ['app_id', 'igdb_game_id', 'join_method'],
                      where 'join_method' is either 'external_games' or 'name_search'.
                      Rows are unique on 'app_id'.
    """
    steam_uids = steam_catalog["app_id"].astype(str).tolist()
    external_parts: list[pd.DataFrame] = []
    for start in range(0, len(steam_uids), external_batch_size):
        batch = steam_uids[start : start + external_batch_size]
        part = lookup_external_games_by_steam_uids(
            client,
            batch,
            steam_external_game_source=steam_external_game_source,
        )
        external_parts.append(part)
        logger.info(
            "external_games batch %s: +%s matches",
            start // external_batch_size + 1,
            len(part),
        )

    external_join = (
        pd.concat(external_parts, ignore_index=True) if external_parts else pd.DataFrame()
    )
    matched_external_ids = (
        set(external_join["app_id"].tolist()) if not external_join.empty else set()
    )

    unresolved = steam_catalog.loc[~steam_catalog["app_id"].isin(matched_external_ids)]
    name_rows: list[dict[str, Any]] = []
    for _, row in unresolved.head(max_name_lookups).iterrows():
        igdb_id = lookup_igdb_id_by_name(client, row["app_name"])
        if igdb_id is not None:
            name_rows.append(
                {
                    "app_id": int(row["app_id"]),
                    "igdb_game_id": igdb_id,
                    "join_method": "name_search",
                }
            )

    name_join = pd.DataFrame(name_rows)
    join_map = pd.concat([external_join, name_join], ignore_index=True).drop_duplicates(
        subset=["app_id"], keep="first"
    )

    logger.info(
        "join map: external=%s name_fallback=%s total=%s / %s",
        len(external_join),
        len(name_join),
        len(join_map),
        len(steam_catalog),
    )
    return join_map


def fetch_game_details(
    client: IGDBClient,
    igdb_ids: list[int],
    *,
    game_fields: str,
    game_detail_batch_size: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for start in range(0, len(igdb_ids), game_detail_batch_size):
        batch = igdb_ids[start : start + game_detail_batch_size]
        id_list = ",".join(str(i) for i in batch)
        body = f"fields {game_fields}; where id = ({id_list}); limit {len(batch)};"
        payload = client.post("games", body)
        rows.extend(payload)
        logger.info(
            "game details batch %s: +%s",
            start // game_detail_batch_size + 1,
            len(payload),
        )
    return pd.DataFrame(rows)


def merge_joined_games(
    join_map: pd.DataFrame,
    steam_catalog: pd.DataFrame,
    igdb_games_raw: pd.DataFrame,
) -> pd.DataFrame:
    if igdb_games_raw.empty:
        raise RuntimeError("No IGDB game rows fetched.")

    igdb_games = igdb_games_raw.copy()
    igdb_games["igdb_game_id"] = igdb_games["id"].astype(int)

    joined = join_map.merge(steam_catalog[["app_id", "app_name"]], on="app_id", how="left").merge(
        igdb_games,
        on="igdb_game_id",
        how="left",
        suffixes=("", "_igdb"),
    )
    if "id" in joined.columns:
        joined = joined.drop(columns=["id"])
    if "name_igdb" in joined.columns:
        joined = joined.rename(columns={"name_igdb": "igdb_name"})
    elif "name" in joined.columns:
        joined = joined.rename(columns={"name": "igdb_name"})
    return joined


def _field_populated(value: Any) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict, set, tuple)):
        return len(value) > 0
    return True


def build_join_report(
    *,
    steam_catalog: pd.DataFrame,
    joined: pd.DataFrame,
    eval_app_ids: set[int],
    index_path: Path,
    max_name_lookups: int,
) -> dict[str, Any]:
    catalog_n = len(steam_catalog)
    joined_n = len(joined)
    unresolved_n = catalog_n - joined_n

    igdb_field_cols = [c for c in joined.columns if c not in STEAM_JOIN_COLS]
    field_coverage = {
        f"{col}_pct": float(joined[col].map(_field_populated).mean()) if joined_n else 0.0
        for col in igdb_field_cols
    }
    v2_core_coverage = {
        k: field_coverage.get(f"{k}_pct", 0.0)
        for k in V2_CORE_FIELDS
        if f"{k}_pct" in field_coverage
    }

    eval_join_rate = None
    if eval_app_ids:
        eval_joined = len(set(joined["app_id"].tolist()) & eval_app_ids)
        eval_join_rate = eval_joined / len(eval_app_ids)

    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "steam_catalog_n": catalog_n,
        "joined_n": joined_n,
        "unresolved_n": unresolved_n,
        "match_rate": joined_n / catalog_n if catalog_n else 0.0,
        "join_method_counts": joined["join_method"].value_counts().to_dict(),
        "field_coverage_on_joined": field_coverage,
        "v2_core_field_coverage": v2_core_coverage,
        "igdb_field_count": len(igdb_field_cols),
        "eval_cohort_unique_query_app_ids": len(eval_app_ids),
        "eval_cohort_join_rate": eval_join_rate,
        "name_fallback_cap": max_name_lookups,
        "index_path": str(index_path),
    }


def build_meta(
    *,
    game_fields: str,
    game_fields_preset: str = "pipeline",
    mock_rows_path: str | None = None,
    output_dir: str | None = None,
    output_filename: str | None = None,
    external_batch_size: int,
    game_detail_batch_size: int,
    max_name_lookups: int,
    steam_external_game_source: int = STEAM_EXTERNAL_GAME_SOURCE,
) -> dict[str, Any]:
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "igdb_api_base": IGDBClient.API_BASE,
        "steam_external_game_source": steam_external_game_source,
        "external_batch_size": external_batch_size,
        "game_detail_batch_size": game_detail_batch_size,
        "max_name_lookups": max_name_lookups,
        "game_fields_preset": game_fields_preset,
        "game_fields": game_fields,
        "mock_rows_path": mock_rows_path,
        "output_dir": output_dir,
        "output_filename": output_filename,
    }


def write_igdb_artifacts(
    output_dir: Path,
    *,
    joined: pd.DataFrame,
    join_report: dict[str, Any],
    meta: dict[str, Any],
    output_filename: str = IGDB_GAMES_PARQUET,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    joined.to_parquet(output_dir / output_filename, index=False)
    (output_dir / "igdb_join_report.json").write_text(
        json.dumps(join_report, indent=2), encoding="utf-8"
    )
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def fetch_and_join_igdb_games(
    config: IgdbGamesJobConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Join Steam catalog to IGDB, fetch game rows, optionally write artifacts."""
    index_path = config.resolved_game_index_path()
    out_dir = config.output_dir
    resolved_game_fields = config.resolved_game_fields()

    client_id, client_secret = load_twitch_credentials(repo_root=config.repo_root)
    client = IGDBClient(
        client_id,
        client_secret,
        request_min_interval_s=config.request_min_interval_s,
    )

    steam_catalog = load_steam_catalog(index_path)
    eval_app_ids = load_eval_query_app_ids(config.eval_parquet_path)

    join_map = build_join_map(
        client,
        steam_catalog,
        external_batch_size=config.external_batch_size,
        max_name_lookups=config.max_name_lookups,
    )
    if join_map.empty:
        raise RuntimeError("No Steam app_ids matched in IGDB.")

    igdb_ids = sorted(set(join_map["igdb_game_id"].astype(int).tolist()))
    igdb_games_raw = fetch_game_details(
        client,
        igdb_ids,
        game_fields=resolved_game_fields,
        game_detail_batch_size=config.game_detail_batch_size,
    )
    joined = merge_joined_games(join_map, steam_catalog, igdb_games_raw)
    mock_rows = load_mock_rows(config.mock_rows_path)
    joined = apply_mock_rows(joined, steam_catalog, mock_rows)

    join_report = build_join_report(
        steam_catalog=steam_catalog,
        join_map=join_map,
        joined=joined,
        eval_app_ids=eval_app_ids,
        index_path=index_path,
        max_name_lookups=config.max_name_lookups,
    )
    meta = build_meta(
        game_fields=resolved_game_fields,
        game_fields_preset=config.game_fields_preset,
        mock_rows_path=str(config.mock_rows_path) if config.mock_rows_path else None,
        output_dir=str(out_dir),
        output_filename=config.output_filename,
        external_batch_size=config.external_batch_size,
        game_detail_batch_size=config.game_detail_batch_size,
        max_name_lookups=config.max_name_lookups,
    )

    if config.write_artifacts:
        write_igdb_artifacts(
            out_dir,
            joined=joined,
            join_report=join_report,
            meta=meta,
            output_filename=config.output_filename,
        )
        logger.info("Wrote %s (%s rows)", out_dir / config.output_filename, len(joined))

    return steam_catalog, joined, join_report, meta
