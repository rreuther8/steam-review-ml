"""Official IGDB /v4/games field metadata (https://api-docs.igdb.com/#game)."""

from __future__ import annotations

from typing import TypedDict


class GameFieldDoc(TypedDict, total=False):
    type: str
    description: str
    deprecated: bool


# Source: IGDB API docs — GET https://api.igdb.com/v4/games
GAME_FIELD_DOCS: dict[str, GameFieldDoc] = {
    "age_ratings": {
        "type": "Array of Age Rating IDs",
        "description": "The PEGI rating",
    },
    "aggregated_rating": {
        "type": "Double",
        "description": "Rating based on external critic scores",
    },
    "aggregated_rating_count": {
        "type": "Integer",
        "description": "Number of external critic scores",
    },
    "alternative_names": {
        "type": "Array of Alternative Name IDs",
        "description": "Alternative names for this game",
    },
    "artworks": {
        "type": "Array of Artwork IDs",
        "description": "Artworks of this game",
    },
    "bundles": {
        "type": "Array of Game IDs",
        "description": "The bundles this game is a part of",
    },
    "category": {
        "type": "Category Enum",
        "description": "DEPRECATED — use game_type instead",
        "deprecated": True,
    },
    "checksum": {
        "type": "uuid",
        "description": "Hash of the object",
    },
    "collection": {
        "type": "Reference ID for Collection",
        "description": "DEPRECATED — use collections instead",
        "deprecated": True,
    },
    "collections": {
        "type": "Array of Collection IDs",
        "description": "The collections that this game is in",
    },
    "cover": {
        "type": "Reference ID for Cover",
        "description": "The cover of this game",
    },
    "created_at": {
        "type": "datetime",
        "description": "Date this was initially added to the IGDB database",
    },
    "dlcs": {
        "type": "Array of Game IDs",
        "description": "DLCs for this game",
    },
    "expanded_games": {
        "type": "Array of Game IDs",
        "description": "Expanded games of this game",
    },
    "expansions": {
        "type": "Array of Game IDs",
        "description": "Expansions of this game",
    },
    "external_games": {
        "type": "Array of External Game IDs",
        "description": "External IDs this game has on other services",
    },
    "first_release_date": {
        "type": "Unix Time Stamp",
        "description": "The first release date for this game",
    },
    "follows": {
        "type": "Integer",
        "description": "DEPRECATED — to be removed",
        "deprecated": True,
    },
    "forks": {
        "type": "Array of Game IDs",
        "description": "Forks of this game",
    },
    "franchise": {
        "type": "Reference ID for Franchise",
        "description": "The main franchise",
    },
    "franchises": {
        "type": "Array of Franchise IDs",
        "description": "Other franchises the game belongs to",
    },
    "game_engines": {
        "type": "Array of Game Engine IDs",
        "description": "The game engine used in this game",
    },
    "game_localizations": {
        "type": "Array of Game Localization IDs",
        "description": (
            "Supported game localizations for this game. "
            "A region can have at most one game localization for a given game"
        ),
    },
    "game_modes": {
        "type": "Array of Game Mode IDs",
        "description": "Modes of gameplay",
    },
    "game_status": {
        "type": "Reference ID for Game Status",
        "description": "The status of the game's release",
    },
    "game_type": {
        "type": "Reference ID for Game Type",
        "description": "The type of game",
    },
    "genres": {
        "type": "Array of Genre IDs",
        "description": "Genres of the game",
    },
    "hypes": {
        "type": "Integer",
        "description": "Number of follows a game gets before release",
    },
    "involved_companies": {
        "type": "Array of Involved Company IDs",
        "description": "Companies who developed this game",
    },
    "keywords": {
        "type": "Array of Keyword IDs",
        "description": "Associated keywords",
    },
    "language_supports": {
        "type": "Array of Language Support IDs",
        "description": "Supported languages for this game",
    },
    "multiplayer_modes": {
        "type": "Array of Multiplayer Mode IDs",
        "description": "Multiplayer modes for this game",
    },
    "name": {
        "type": "String",
        "description": "Game title on IGDB (stored as igdb_name in our parquet)",
    },
    "parent_game": {
        "type": "Reference ID for Game",
        "description": "If a DLC, expansion or part of a bundle, this is the main game or bundle",
    },
    "platforms": {
        "type": "Array of Platform IDs",
        "description": "Platforms this game was released on",
    },
    "player_perspectives": {
        "type": "Array of Player Perspective IDs",
        "description": "The main perspective of the player",
    },
    "ports": {
        "type": "Array of Game IDs",
        "description": "Ports of this game",
    },
    "rating": {
        "type": "Double",
        "description": "Average IGDB user rating",
    },
    "rating_count": {
        "type": "Integer",
        "description": "Total number of IGDB user ratings",
    },
    "release_dates": {
        "type": "Array of Release Date IDs",
        "description": "Release dates of this game",
    },
    "remakes": {
        "type": "Array of Game IDs",
        "description": "Remakes of this game",
    },
    "remasters": {
        "type": "Array of Game IDs",
        "description": "Remasters of this game",
    },
    "screenshots": {
        "type": "Array of Screenshot IDs",
        "description": "Screenshots of this game",
    },
    "similar_games": {
        "type": "Array of Game IDs",
        "description": "Similar games",
    },
    "slug": {
        "type": "String",
        "description": "A url-safe, unique, lower-case version of the name",
    },
    "standalone_expansions": {
        "type": "Array of Game IDs",
        "description": "Standalone expansions of this game",
    },
    "status": {
        "type": "Status Enum",
        "description": "DEPRECATED — use game_status instead",
        "deprecated": True,
    },
    "storyline": {
        "type": "String",
        "description": "A short description of a game's story",
    },
    "summary": {
        "type": "String",
        "description": "A description of the game",
    },
    "tags": {
        "type": "Array of Tag Numbers",
        "description": "Related entities in the IGDB API",
    },
    "themes": {
        "type": "Array of Theme IDs",
        "description": "Themes of the game",
    },
    "total_rating": {
        "type": "Double",
        "description": "Average rating based on both IGDB user and external critic scores",
    },
    "total_rating_count": {
        "type": "Integer",
        "description": "Total number of user and external critic scores",
    },
    "updated_at": {
        "type": "datetime",
        "description": "The last date this entry was updated in the IGDB database",
    },
    "url": {
        "type": "String",
        "description": "The website address (URL) of the item",
    },
    "version_parent": {
        "type": "Reference ID for Game",
        "description": "If a version, this is the main game",
    },
    "version_title": {
        "type": "String",
        "description": "Title of this version (i.e. Gold edition)",
    },
    "videos": {
        "type": "Array of Game Video IDs",
        "description": "Videos of this game",
    },
    "websites": {
        "type": "Array of Website IDs",
        "description": "Websites associated with this game",
    },
}

# Project-specific v2 ranker notes (not from IGDB API).
V2_FIELD_NOTES: dict[str, str] = {
    "summary": "V2b — USE dot(query_review, summary). Ready without entity lookup.",
    "genres": "V2a — Jaccard on genre ID sets (entity lookup optional for human QA).",
    "themes": "V2a — Jaccard on theme ID sets.",
    "keywords": "V2a — Jaccard on keyword ID sets; ~91% coverage on our catalog.",
    "game_modes": "V2a — Jaccard on game mode ID sets.",
    "player_perspectives": "V2a — Jaccard on perspective ID sets.",
    "franchises": "V2a — Jaccard on franchise ID sets; sparse (~30% coverage) on our catalog.",
    "tags": "V2a candidate — related-entity tag numbers; noisier than genres/themes.",
    "storyline": "Extra text field (not summary); ~50% coverage; optional V2b supplement.",
    "similar_games": "Graph prior — IGDB game IDs; check overlap with our catalog index.",
    "rating": "Popularity proxy; likely redundant with D1 log-pop blend.",
    "rating_count": "Support count for rating.",
    "total_rating": "Blended IGDB + critic rating.",
    "total_rating_count": "Support count for total_rating.",
    "first_release_date": "Recency feature; unix timestamp.",
    "game_type": "Filter DLC/expansion/bundle noise before ranker features.",
    "platforms": "Weak ranker signal; mostly PC on our Steam catalog.",
    "slug": "Audit / join debug only.",
    "url": "Audit only.",
    "checksum": "Audit / reproducibility only.",
    "cover": "Media reference — not for text ranker.",
    "artworks": "Media reference.",
    "screenshots": "Media reference.",
    "videos": "Media reference.",
    "websites": "Media / link reference.",
    "release_dates": "Platform/region-specific release metadata.",
    "involved_companies": "Developer/publisher IDs; optional metadata signal.",
    "language_supports": "Localization IDs.",
    "external_games": "Store mapping IDs; join cross-check (Steam uid in pass 1).",
    "created_at": "IGDB record metadata.",
    "updated_at": "IGDB record metadata.",
}
