"""IGDB metadata fetch and Steam catalog join."""

from steam_review_ml.igdb.constants import (
    ALL_GAME_FIELDS,
    DEFAULT_GAME_FIELDS,
    PIPELINE_DEFAULT_GAME_FIELD_NAMES,
    V2_CORE_FIELDS,
    resolve_game_fields,
)
from steam_review_ml.igdb.field_docs import GAME_FIELD_DOCS, V2_FIELD_NOTES
from steam_review_ml.igdb.fetch import fetch_and_join_igdb_games, load_steam_catalog
from steam_review_ml.igdb.job_config import IgdbGamesJobConfig

__all__ = [
    "ALL_GAME_FIELDS",
    "DEFAULT_GAME_FIELDS",
    "GAME_FIELD_DOCS",
    "IgdbGamesJobConfig",
    "PIPELINE_DEFAULT_GAME_FIELD_NAMES",
    "V2_CORE_FIELDS",
    "V2_FIELD_NOTES",
    "fetch_and_join_igdb_games",
    "load_steam_catalog",
    "resolve_game_fields",
]
