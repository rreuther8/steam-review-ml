"""Job config for IGDB games fetch + join pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from steam_review_ml.igdb.constants import (
    IGDB_GAMES_FEATURES_PARQUET,
    IGDB_GAMES_PARQUET,
    resolve_game_fields,
)


def _require_str(cfg: dict[str, Any], key: str) -> str:
    value = cfg.get(key)
    if value is None or not str(value).strip():
        raise ValueError(f"Config must include non-empty '{key}'.")
    return str(value).strip()


def _optional_repo_path(repo_root: Path, rel: str | None) -> Path | None:
    if rel is None or not str(rel).strip():
        return None
    return repo_root / str(rel).strip()


def _parse_embed_text_fields(raw: Any) -> tuple[str, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ValueError("embed_text_fields must be a JSON list of field names or null.")
    fields = tuple(str(field).strip() for field in raw if str(field).strip())
    return fields


@dataclass(frozen=True)
class IgdbGamesJobConfig:
    repo_root: Path
    output_dir: Path
    game_index_path: Path | None = None
    eval_parquet_path: Path | None = None
    game_fields: str | None = None
    game_fields_preset: str = "pipeline"
    external_batch_size: int = 500
    game_detail_batch_size: int = 50
    max_name_lookups: int = 200
    request_min_interval_s: float = 0.26
    mock_rows_path: Path | None = None
    output_filename: str = IGDB_GAMES_PARQUET
    write_artifacts: bool = True
    skip_fetch: bool = False
    embed_text_fields: tuple[str, ...] = ()
    features_output_filename: str = IGDB_GAMES_FEATURES_PARQUET
    tfhub_url: str | None = None
    game_embedding_meta_path: Path | None = None
    embed_batch_size: int = 64
    max_chars_per_field: int | None = 8000

    @classmethod
    def from_json(cls, repo_root: Path, cfg: dict[str, Any]) -> IgdbGamesJobConfig:
        max_chars = cfg.get("max_chars_per_field", 8000)
        return cls(
            repo_root=repo_root,
            output_dir=repo_root / _require_str(cfg, "output_dir"),
            game_index_path=_optional_repo_path(repo_root, cfg.get("game_index_path")),
            eval_parquet_path=_optional_repo_path(repo_root, cfg.get("eval_parquet_path")),
            game_fields=cfg.get("game_fields"),
            game_fields_preset=str(cfg.get("game_fields_preset", "pipeline")),
            external_batch_size=int(cfg.get("external_batch_size", 500)),
            game_detail_batch_size=int(cfg.get("game_detail_batch_size", 50)),
            max_name_lookups=int(cfg.get("max_name_lookups", 200)),
            request_min_interval_s=float(cfg.get("request_min_interval_s", 0.26)),
            mock_rows_path=_optional_repo_path(repo_root, cfg.get("mock_rows_path")),
            output_filename=str(cfg.get("output_filename") or IGDB_GAMES_PARQUET).strip(),
            write_artifacts=True,
            skip_fetch=bool(cfg.get("skip_fetch", False)),
            embed_text_fields=_parse_embed_text_fields(cfg.get("embed_text_fields")),
            features_output_filename=str(
                cfg.get("features_output_filename") or IGDB_GAMES_FEATURES_PARQUET
            ).strip(),
            tfhub_url=cfg.get("tfhub_url"),
            game_embedding_meta_path=_optional_repo_path(
                repo_root, cfg.get("game_embedding_meta_path")
            ),
            embed_batch_size=int(cfg.get("embed_batch_size", 64)),
            max_chars_per_field=None if max_chars is None else int(max_chars),
        )

    def resolved_game_index_path(self) -> Path:
        if self.game_index_path is not None:
            return self.game_index_path
        from steam_review_ml.igdb.fetch import resolve_game_index_path

        return resolve_game_index_path(self.repo_root)

    def resolved_game_fields(self) -> str:
        return resolve_game_fields(self.game_fields, preset=self.game_fields_preset)

    def raw_parquet_path(self) -> Path:
        return self.output_dir / self.output_filename

    def features_parquet_path(self) -> Path:
        return self.output_dir / self.features_output_filename
