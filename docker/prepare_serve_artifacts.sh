#!/usr/bin/env bash
# Stages the small set of files the served API actually reads (~25MB) into docker/artifacts/,
# since the real artifacts/ and data/ dirs (16GB combined) are excluded from the Docker build
# context. Run before `docker compose build` whenever the source artifacts change.
set -euo pipefail
cd "$(dirname "$0")/.."

dest=docker/artifacts
rm -rf "$dest"
mkdir -p \
  "$dest/recs/towers/val_dev_12k_v1" \
  "$dest/recs/embeddings/game_profile/default" \
  "$dest/igdb"

cp artifacts/recs/towers/val_dev_12k_v1/updated_user__updated_profile200_item.keras \
  "$dest/recs/towers/val_dev_12k_v1/"
cp artifacts/recs/embeddings/game_profile/default/game_profile_embeddings.npz \
  artifacts/recs/embeddings/game_profile/default/game_profile_embedding_index.parquet \
  artifacts/recs/embeddings/game_profile/default/game_profile_embedding_meta.json \
  "$dest/recs/embeddings/game_profile/default/"
cp artifacts/igdb/igdb_games__enriched.parquet "$dest/igdb/"

echo "Staged serve artifacts into $dest:"
du -sh "$dest"
