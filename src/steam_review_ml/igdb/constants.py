"""IGDB API constants for Steam catalog join jobs."""

from __future__ import annotations

from collections.abc import Iterable

STEAM_EXTERNAL_GAME_SOURCE = 1  # IGDB external_game_sources.id for Steam

# Full /v4/games field list (EDA archive pulls).
ALL_GAME_FIELDS = (
    "id,age_ratings,aggregated_rating,aggregated_rating_count,alternative_names,artworks,bundles,"
    "category,checksum,collection,collections,cover,created_at,dlcs,expanded_games,expansions,"
    "external_games,first_release_date,follows,forks,franchise,franchises,game_engines,"
    "game_localizations,game_modes,game_status,game_type,genres,hypes,involved_companies,"
    "keywords,language_supports,multiplayer_modes,name,parent_game,platforms,player_perspectives,"
    "ports,rating,rating_count,release_dates,remakes,remasters,screenshots,similar_games,slug,"
    "standalone_expansions,status,storyline,summary,tags,themes,total_rating,total_rating_count,"
    "updated_at,url,version_parent,version_title,videos,websites"
)

# Backward-compatible alias.
DEFAULT_GAME_FIELDS = ALL_GAME_FIELDS

# Always included in game detail fetches (join + display).
JOIN_GAME_FIELD_NAMES: tuple[str, ...] = ("id", "name")

# Default pipeline pull: v2 ranker + metadata EDA subset.
PIPELINE_DEFAULT_GAME_FIELD_NAMES: tuple[str, ...] = (
    "age_ratings",
    "collections",
    "franchises",
    "game_engines",
    "game_modes",
    "game_type",
    "genres",
    "involved_companies",
    "keywords",
    "multiplayer_modes",
    "player_perspectives",
    "storyline",
    "summary",
    "tags",
    "themes",
)

V2_CORE_FIELDS = (
    "summary",
    "genres",
    "themes",
    "keywords",
    "game_modes",
    "player_perspectives",
    "franchises",
)

STEAM_JOIN_COLS = frozenset({"app_id", "app_name", "join_method", "igdb_game_id", "igdb_name"})

# Artifact filenames under artifacts/recs/igdb/
IGDB_GAMES_PARQUET = "igdb_games__raw.parquet"
IGDB_GAMES_FEATURES_PARQUET = "igdb_games__features.parquet"
USE_EMBEDDING_FIELD_SUFFIX = "__use"


def format_game_fields_query(field_names: Iterable[str]) -> str:
    """Build Apicalypse `fields` value; always includes id + name for joins."""
    names: list[str] = []
    seen: set[str] = set()
    for field in (*JOIN_GAME_FIELD_NAMES, *field_names):
        key = str(field).strip()
        if key and key not in seen:
            seen.add(key)
            names.append(key)
    return ",".join(names)


def resolve_game_fields(
    game_fields: str | list[str] | None = None,
    *,
    preset: str = "pipeline",
) -> str:
    """Resolve config `game_fields` / preset to an IGDB Apicalypse fields string.

    - ``game_fields`` str: use as-is (comma-separated API field list)
    - ``game_fields`` list: join with ``id`` + ``name``
    - ``None`` + ``preset=\"pipeline\"``: ``PIPELINE_DEFAULT_GAME_FIELD_NAMES``
    - ``None`` + ``preset=\"full\"``: ``ALL_GAME_FIELDS``
    """
    if isinstance(game_fields, str) and game_fields.strip():
        return game_fields.strip()
    if isinstance(game_fields, list) and game_fields:
        return format_game_fields_query(game_fields)
    if preset == "pipeline":
        return format_game_fields_query(PIPELINE_DEFAULT_GAME_FIELD_NAMES)
    if preset == "full":
        return ALL_GAME_FIELDS
    raise ValueError(f"Unknown game_fields_preset: {preset!r} (use 'pipeline' or 'full')")
