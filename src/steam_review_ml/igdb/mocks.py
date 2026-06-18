"""Manual IGDB row mocks for catalog games that fail API join."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from steam_review_ml.igdb.constants import STEAM_JOIN_COLS

logger = logging.getLogger(__name__)

MOCK_META_KEYS = frozenset({"_note", "override"})


def load_mock_rows(path: Path | None) -> pd.DataFrame:
    if path is None or not path.is_file():
        return pd.DataFrame()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Mock rows file must be a JSON list: {path}")
    rows = [row for row in payload if isinstance(row, dict)]
    return pd.DataFrame(rows)


def _normalize_mock_value(value: Any) -> Any:
    if isinstance(value, list):
        if not value:
            return np.array([], dtype=np.int64)
        if all(isinstance(x, (int, np.integer)) and not isinstance(x, bool) for x in value):
            return np.array(value, dtype=np.int64)
        return value
    return value


def _mock_override(raw: pd.Series) -> bool:
    if "override" not in raw.index or pd.isna(raw["override"]):
        return False
    return bool(raw["override"])


def _is_missing_value(value: Any) -> bool:
    if isinstance(value, (list, np.ndarray)):
        return False
    return bool(pd.isna(value))


def apply_mock_rows(
    joined: pd.DataFrame,
    steam_catalog: pd.DataFrame,
    mock_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Append or replace rows from manual mock definitions."""
    if mock_rows.empty:
        return joined

    out = joined.copy()
    catalog = steam_catalog.set_index("app_id", drop=False)
    present_app_ids = set(out["app_id"].astype(int).tolist())

    for _, raw in mock_rows.iterrows():
        app_id = int(raw["app_id"])
        if app_id not in catalog.index:
            logger.warning("Skipping mock app_id=%s — not in Steam catalog", app_id)
            continue

        override = _mock_override(raw)
        if app_id in present_app_ids and not override:
            logger.info("Skipping mock app_id=%s — already joined (set override=true to replace)", app_id)
            continue

        catalog_row = catalog.loc[app_id]
        igdb_game_id = int(raw["igdb_game_id"]) if pd.notna(raw.get("igdb_game_id")) else -app_id
        join_method = str(raw["join_method"]) if pd.notna(raw.get("join_method")) else "manual_mock"
        igdb_name = (
            str(raw["igdb_name"]) if pd.notna(raw.get("igdb_name")) else str(catalog_row["app_name"])
        )
        row: dict[str, Any] = {
            "app_id": app_id,
            "app_name": str(catalog_row["app_name"]),
            "igdb_game_id": igdb_game_id,
            "join_method": join_method,
            "igdb_name": igdb_name,
        }

        for key in raw.index:
            if key in STEAM_JOIN_COLS or key in MOCK_META_KEYS or key == "app_id":
                continue
            value = raw[key]
            if _is_missing_value(value):
                continue
            row[key] = _normalize_mock_value(value)

        mock_df = pd.DataFrame([row])
        if app_id in present_app_ids and override:
            out = out.loc[out["app_id"].astype(int) != app_id]
            present_app_ids.discard(app_id)

        out = pd.concat([out, mock_df], ignore_index=True)
        present_app_ids.add(app_id)
        logger.info("Applied manual mock for app_id=%s (%s)", app_id, row["app_name"])

    return out
