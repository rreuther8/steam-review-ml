"""Tests for IgdbGamesJobConfig."""

from __future__ import annotations

from pathlib import Path

import pytest

from steam_review_ml.igdb.constants import IGDB_GAMES_PARQUET, PIPELINE_DEFAULT_GAME_FIELD_NAMES
from steam_review_ml.igdb.job_config import IgdbGamesJobConfig


def test_from_json_uses_defaults_and_resolves_paths(tmp_path: Path) -> None:
    repo_root = tmp_path
    cfg = {
        "output_dir": "artifacts/recs/igdb",
        "game_fields_preset": "pipeline",
    }

    job_config = IgdbGamesJobConfig.from_json(repo_root, cfg)

    assert job_config.output_dir == repo_root / "artifacts/recs/igdb"
    assert job_config.output_filename == IGDB_GAMES_PARQUET
    assert job_config.external_batch_size == 500
    assert job_config.game_fields_preset == "pipeline"


def test_from_json_requires_output_dir(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="output_dir"):
        IgdbGamesJobConfig.from_json(tmp_path, {})


def test_from_json_parses_embed_text_fields(tmp_path: Path) -> None:
    job_config = IgdbGamesJobConfig.from_json(
        tmp_path,
        {"output_dir": "artifacts/recs/igdb", "embed_text_fields": ["summary", "storyline"]},
    )
    assert job_config.embed_text_fields == ("summary", "storyline")


def test_resolved_game_fields_pipeline_preset() -> None:
    job_config = IgdbGamesJobConfig(
        repo_root=Path("/repo"),
        output_dir=Path("/repo/artifacts/recs/igdb"),
    )
    fields = job_config.resolved_game_fields().split(",")
    assert fields[0] == "id"
    assert fields[1] == "name"
    assert set(PIPELINE_DEFAULT_GAME_FIELD_NAMES).issubset(set(fields))
