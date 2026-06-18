"""Tests for IGDB entity lookup and enriched games join."""

from __future__ import annotations

import numpy as np
import pandas as pd

from steam_review_ml.igdb.entity_lookup import (
    _as_id_list,
    add_taxonomy_resolved_columns,
    build_enriched_games,
    collect_unique_ids,
    fetch_entity_table,
    mean_pool_entity_embeddings,
    resolve_fk_arrays,
    taxonomy_names_column,
    taxonomy_names_embedding_column,
    taxonomy_names_embedding_pooled_column,
)


class _MockClient:
    def __init__(self, responses: dict[str, list[dict]]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, str]] = []

    def post(self, endpoint: str, body: str) -> list[dict]:
        self.calls.append((endpoint, body))
        if endpoint in self.responses:
            return self.responses[endpoint]
        return []


def test_as_id_list_handles_numpy_and_none() -> None:
    assert _as_id_list(None) == []
    assert _as_id_list(np.array([8, 31, 32])) == [8, 31, 32]
    assert _as_id_list([8, 31]) == [8, 31]


def test_collect_unique_ids() -> None:
    df = pd.DataFrame(
        {
            "genres": [np.array([8, 31]), np.array([32])],
            "themes": [None, np.array([1])],
        }
    )
    assert collect_unique_ids(df, "genres") == {8, 31, 32}
    assert collect_unique_ids(df, "themes") == {1}


def test_fetch_entity_table_by_ids() -> None:
    client = _MockClient(
        {
            "genres": [
                {"id": 8, "name": "Platformer"},
                {"id": 31, "name": "Adventure"},
            ]
        }
    )
    out = fetch_entity_table(client, "genres", [8, 31], batch_size=50)
    assert list(out["id"]) == [8, 31]
    assert list(out["name"]) == ["Platformer", "Adventure"]
    assert client.calls[0][0] == "genres"
    assert "where id = (8,31)" in client.calls[0][1]


def test_resolve_fk_arrays_skips_missing_ids() -> None:
    vec = np.array([1.0, 0.0], dtype=np.float32)
    lookup = {
        8: {"name": "Platformer", "name__use": vec},
        31: {"name": "Adventure", "name__use": vec},
    }
    names, vecs = resolve_fk_arrays([8, 99, 31], lookup)
    assert names == ["Platformer", "Adventure"]
    assert len(vecs) == 2


def test_mean_pool_entity_embeddings_empty() -> None:
    assert mean_pool_entity_embeddings([]) == []


def test_mean_pool_entity_embeddings_mean_and_l2_norm() -> None:
    v1 = np.array([1.0, 0.0], dtype=np.float32)
    v2 = np.array([1.0, 0.0], dtype=np.float32)
    pooled = np.asarray(mean_pool_entity_embeddings([v1, v2]))
    np.testing.assert_allclose(pooled, np.array([1.0, 0.0], dtype=np.float32))


def test_add_taxonomy_resolved_columns_aligns_with_fk_order() -> None:
    vec8 = np.array([1.0, 0.0], dtype=np.float32)
    vec31 = np.array([0.0, 1.0], dtype=np.float32)
    games = pd.DataFrame(
        [
            {"app_id": 753420, "genres": np.array([8, 31, 32])},
        ]
    )
    genres_lookup = pd.DataFrame(
        [
            {"id": 8, "name": "Platformer", "name__use": vec8},
            {"id": 31, "name": "Adventure", "name__use": vec31},
        ]
    )
    out = add_taxonomy_resolved_columns(games, {"genres": genres_lookup}, ["genres"])
    assert list(out.loc[0, "genres"]) == [8, 31, 32]
    assert out.loc[0, taxonomy_names_column("genres")] == ["Platformer", "Adventure"]
    assert len(out.loc[0, taxonomy_names_embedding_column("genres")]) == 2
    pooled = np.asarray(out.loc[0, taxonomy_names_embedding_pooled_column("genres")])
    assert pooled.shape == (2,)
    np.testing.assert_allclose(pooled, np.array([1.0, 1.0], dtype=np.float32) / np.sqrt(2.0))


def test_build_enriched_games_preserves_fk_column() -> None:
    games = pd.DataFrame([{"app_id": 1, "themes": np.array([1])}])
    themes_lookup = pd.DataFrame(
        [{"id": 1, "name": "Action", "name__use": np.array([0.5, 0.5], dtype=np.float32)}]
    )
    enriched = build_enriched_games(games, {"themes": themes_lookup}, ["themes"])
    assert _as_id_list(enriched.loc[0, "themes"]) == [1]
    assert enriched.loc[0, taxonomy_names_column("themes")] == ["Action"]
    assert len(enriched.loc[0, taxonomy_names_embedding_column("themes")]) == 1
    pooled = np.asarray(enriched.loc[0, taxonomy_names_embedding_pooled_column("themes")])
    np.testing.assert_allclose(pooled, np.array([0.5, 0.5], dtype=np.float32) / np.sqrt(0.5))


def test_enriched_job_config_paths(tmp_path: Path) -> None:
    from steam_review_ml.igdb.job_config import IgdbGamesEnrichedJobConfig

    cfg = IgdbGamesEnrichedJobConfig.from_json(
        tmp_path,
        {"output_dir": "artifacts/igdb"},
    )
    assert cfg.games_lookup_path == tmp_path / "artifacts/igdb/lookups/games.parquet"
    assert cfg.lookups_dir == tmp_path / "artifacts/igdb/lookups"
    assert cfg.enriched_parquet_path() == tmp_path / "artifacts/igdb/igdb_games__enriched.parquet"
