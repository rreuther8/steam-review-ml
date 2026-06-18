"""Tests for IGDB join helpers (no live API)."""

from __future__ import annotations

import pandas as pd

from steam_review_ml.igdb.constants import (
    PIPELINE_DEFAULT_GAME_FIELD_NAMES,
    resolve_game_fields,
)
from steam_review_ml.igdb.fetch import merge_joined_games, normalize_title


def test_normalize_title_strips_punctuation() -> None:
    assert normalize_title("Counter-Strike: Source") == "counter strike source"


def test_resolve_game_fields_pipeline_preset_includes_id_and_name() -> None:
    fields = resolve_game_fields(preset="pipeline").split(",")
    assert fields[0] == "id"
    assert fields[1] == "name"
    assert set(PIPELINE_DEFAULT_GAME_FIELD_NAMES).issubset(set(fields))


def test_resolve_game_fields_custom_list() -> None:
    assert resolve_game_fields(["genres", "summary"]) == "id,name,genres,summary"


def test_merge_joined_games_renames_igdb_name() -> None:
    join_map = pd.DataFrame(
        [{"app_id": 730, "igdb_game_id": 1, "join_method": "external_games"}]
    )
    steam_catalog = pd.DataFrame([{"app_id": 730, "app_name": "Counter-Strike"}])
    igdb_raw = pd.DataFrame([{"id": 1, "name": "Counter-Strike", "summary": "FPS"}])

    joined = merge_joined_games(join_map, steam_catalog, igdb_raw)

    assert joined.loc[0, "igdb_name"] == "Counter-Strike"
    assert joined.loc[0, "app_name"] == "Counter-Strike"
    assert "id" not in joined.columns
