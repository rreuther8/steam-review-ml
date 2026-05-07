"""Build recommender game embedding artifacts from per-review profile rows.

This script moves the core recs_002 artifact logic into a pipeline job:
  - load game_profile_reviews.parquet
  - encode each review with TF Hub USE
  - mean-pool per app_id and L2-normalize vectors
  - write NPZ matrix, index parquet, and meta JSON
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from tqdm.auto import tqdm

from steam_review_ml.recommender.math_utils import l2_normalize
from steam_review_ml.recommender.preferences import build_embedding_input, extract_preferences
from steam_review_ml.utils import load_config


def _build_and_write_game_embeddings(
    *,
    input_path: Path,
    out_npz: Path,
    out_index: Path,
    out_meta: Path,
    tfhub_url: str,
    batch_size: int,
    max_chars_per_review: int | None,
    text_mode: str,
    structured_max_context_chars: int,
) -> tuple[int, int, int]:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

    embed_fn = hub.load(tfhub_url)
    df = pd.read_parquet(input_path)
    if len(df) == 0:
        raise RuntimeError("Input profile rows parquet is empty.")

    req_cols = {"app_id", "app_name", "review"}
    missing = req_cols.difference(df.columns)
    if missing:
        raise KeyError(f"Missing required columns in input parquet: {sorted(missing)}")

    df = df.copy()
    df["review"] = df["review"].fillna("").astype(str)
    if text_mode not in {"raw", "structured"}:
        raise ValueError(f"text_mode must be 'raw' or 'structured', got: {text_mode!r}")

    if text_mode == "raw":
        if max_chars_per_review is not None:
            df["_text"] = df["review"].str[:max_chars_per_review]
        else:
            df["_text"] = df["review"]
    else:
        structured_texts: list[str] = []
        for raw in tqdm(df["review"].tolist(), desc="build structured text"):
            prefs = extract_preferences(raw)
            structured = build_embedding_input(
                prefs,
                raw,
                max_context_chars=structured_max_context_chars,
            )
            if max_chars_per_review is not None:
                structured = structured[:max_chars_per_review]
            structured_texts.append(structured)
        df["_text"] = structured_texts
    df = df.reset_index(drop=True)

    texts = df["_text"].tolist()
    chunks: list[np.ndarray] = []
    for start in tqdm(range(0, len(texts), batch_size), desc="encode batches"):
        batch = texts[start : start + batch_size]
        out = embed_fn(batch)
        chunks.append(np.asarray(out, dtype=np.float32))
    emb = np.concatenate(chunks, axis=0)
    if emb.ndim != 2:
        raise RuntimeError(f"Expected 2D embedding array, got shape={emb.shape}")

    mean_vecs: list[np.ndarray] = []
    meta_app: list[int] = []
    meta_name: list[str] = []

    for app_id, g in df.groupby("app_id", sort=True):
        idx = g.index.to_numpy()
        v = emb[idx].mean(axis=0)
        mean_vecs.append(l2_normalize(v))
        meta_app.append(int(app_id))
        meta_name.append(str(g["app_name"].iloc[0]))

    X = np.stack(mean_vecs, axis=0).astype(np.float32)
    app_ids_arr = np.array(meta_app, dtype=np.int64)

    np.savez_compressed(out_npz, embeddings=X, app_id=app_ids_arr)
    pd.DataFrame({"app_id": meta_app, "app_name": meta_name}).to_parquet(out_index, index=False)

    meta = {
        "model_name": tfhub_url,
        "backend": "tensorflow",
        "method": f"encode_each_{text_mode}_review_mean_pool_per_app_id_l2_normalize",
        "text_mode": text_mode,
        "max_chars_per_review": max_chars_per_review,
        "structured_max_context_chars": structured_max_context_chars if text_mode == "structured" else None,
        "n_games": int(X.shape[0]),
        "dim": int(X.shape[1]),
        "n_review_rows_encoded": int(len(df)),
    }
    out_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return int(X.shape[0]), int(X.shape[1]), int(len(df))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build game profile embedding artifacts via JSON config."
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to JSON config for game-embeddings job.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    repo_root = Path(__file__).resolve().parents[1]
    input_path = repo_root / cfg["input_path"]
    out_npz = repo_root / cfg["output_npz_path"]
    out_index = repo_root / cfg["output_index_path"]
    out_meta = repo_root / cfg["output_meta_path"]
    tfhub_url = str(cfg.get("tfhub_url", "https://tfhub.dev/google/universal-sentence-encoder/4"))
    batch_size = int(cfg.get("batch_size", 64))
    max_chars_per_review = cfg.get("max_chars_per_review", 8000)
    if max_chars_per_review is not None:
        max_chars_per_review = int(max_chars_per_review)
    text_mode = str(cfg.get("text_mode", "raw")).strip().lower()
    structured_max_context_chars = int(cfg.get("structured_max_context_chars", 220))

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Missing required input artifact: {input_path}. "
            "Run scripts/recs_job_game_profiles.py first."
        )

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    out_index.parent.mkdir(parents=True, exist_ok=True)
    out_meta.parent.mkdir(parents=True, exist_ok=True)

    print(f"Input profile rows: {input_path}")
    print(
        f"Outputs: npz={out_npz} index={out_index} meta={out_meta} "
        f"batch_size={batch_size} max_chars_per_review={max_chars_per_review} "
        f"text_mode={text_mode}"
    )

    n_games, dim, n_rows = _build_and_write_game_embeddings(
        input_path=input_path,
        out_npz=out_npz,
        out_index=out_index,
        out_meta=out_meta,
        tfhub_url=tfhub_url,
        batch_size=batch_size,
        max_chars_per_review=max_chars_per_review,
        text_mode=text_mode,
        structured_max_context_chars=structured_max_context_chars,
    )
    print(f"Wrote {out_npz}")
    print(f"Wrote {out_index}")
    print(f"Wrote {out_meta}")
    print(f"Summary: n_games={n_games} dim={dim} n_review_rows_encoded={n_rows}")


if __name__ == "__main__":
    main()
