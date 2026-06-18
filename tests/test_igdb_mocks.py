"""Tests for IGDB manual mock rows."""

from __future__ import annotations

import pandas as pd

from steam_review_ml.igdb.mocks import apply_mock_rows


def test_apply_mock_rows_appends_missing_catalog_game() -> None:
    joined = pd.DataFrame(
        [
            {
                "app_id": 730,
                "app_name": "Counter-Strike",
                "igdb_game_id": 1,
                "join_method": "external_games",
                "igdb_name": "Counter-Strike",
                "summary": "FPS",
            }
        ]
    )
    catalog = pd.DataFrame(
        [
            {"app_id": 730, "app_name": "Counter-Strike", "app_name_norm": "counter strike"},
            {"app_id": 431960, "app_name": "Wallpaper Engine", "app_name_norm": "wallpaper engine"},
        ]
    )
    mock_rows = pd.DataFrame(
        [
            {
                "app_id": 431960,
                "igdb_game_id": -431960,
                "join_method": "manual_mock",
                "igdb_name": "Wallpaper Engine",
                "summary": "Live wallpapers on desktop.",
                "genres": [],
            }
        ]
    )

    out = apply_mock_rows(joined, catalog, mock_rows)

    assert len(out) == 2
    mock = out.loc[out["app_id"] == 431960].iloc[0]
    assert mock["join_method"] == "manual_mock"
    assert mock["summary"] == "Live wallpapers on desktop."
    assert mock["app_name"] == "Wallpaper Engine"
