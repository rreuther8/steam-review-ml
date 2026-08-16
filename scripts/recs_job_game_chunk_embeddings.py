"""Embed Stage 1 chunk table and load it into Chroma (Stage 2 of the RAG extension plan).

This is the Stage 2 job of docs/plans/rag_extension_plan.md:
  - read game_review_chunks.parquet (Stage 1 output)
  - encode every chunk's text once with TF Hub USE (the one embedding pass)
  - write the fine-grain `game_review_chunks` Chroma collection (one row per chunk)
  - compute 4 pooling variants (any_polarity/recommended_only x flat/log1p_weighted)
    from those same chunk embeddings -- cheap re-aggregation, not re-embedding --
    blend each game's pooled-review vector with its description vector, and write
    the coarse `game_profiles` Chroma collection (one row per app_id x variant).
    This is the collection Stage 3 queries.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import chromadb
import numpy as np
import pandas as pd
from chromadb.api import ClientAPI

from steam_review_ml.recommender.math_utils import l2_normalize
from steam_review_ml.utils import load_config

CHROMA_ADD_BATCH = 5000  # stays under chromadb's client.get_max_batch_size() across backends

POLARITY_ARMS = ("any_polarity", "recommended_only")
WEIGHTING_ARMS = ("flat", "log_weighted")


def _encode_texts(texts: list[str], embed_fn, *, batch_size: int, max_chars: int) -> np.ndarray:
    clipped = [t[:max_chars] for t in texts]
    chunks: list[np.ndarray] = []
    for start in range(0, len(clipped), batch_size):
        batch = clipped[start : start + batch_size]
        out = embed_fn(batch)
        chunks.append(np.asarray(out, dtype=np.float32))
    emb = np.concatenate(chunks, axis=0)
    if emb.ndim != 2:
        raise RuntimeError(f"Expected 2D embedding array, got shape={emb.shape}")
    return np.stack([l2_normalize(row) for row in emb], axis=0)


def _add_in_batches(
    collection,
    *,
    ids: list[str],
    embeddings: np.ndarray,
    metadatas: list[dict],
    documents: list[str] | None = None,
    batch_size: int = CHROMA_ADD_BATCH,
) -> None:
    for start in range(0, len(ids), batch_size):
        end = start + batch_size
        collection.add(
            ids=ids[start:end],
            embeddings=embeddings[start:end].tolist(),
            metadatas=metadatas[start:end],
            documents=documents[start:end] if documents is not None else None,
        )


def _fresh_collection(client: ClientAPI, name: str):
    """Drop and recreate the collection so reruns fully overwrite prior contents."""
    if name in {c.name for c in client.list_collections()}:
        client.delete_collection(name)
    return client.create_collection(name, metadata={"hnsw:space": "cosine"})


def _write_review_chunks_collection(
    client: ClientAPI,
    *,
    collection_name: str,
    df: pd.DataFrame,
    embeddings: np.ndarray,
) -> int:
    collection = _fresh_collection(client, collection_name)

    ids: list[str] = []
    metadatas: list[dict] = []
    documents: list[str] = []
    for row in df.to_dict("records"):
        app_id = int(row["app_id"])
        if row["chunk_type"] == "review":
            ids.append(f"{app_id}_{int(row['review_id'])}")
            metadatas.append(
                {
                    "app_id": app_id,
                    "chunk_type": "review",
                    "review_id": int(row["review_id"]),
                    "votes_helpful": int(row["votes_helpful"]),
                    "recommended": bool(int(row["recommended"])),
                }
            )
        else:
            ids.append(f"{app_id}_description")
            metadatas.append({"app_id": app_id, "chunk_type": "description"})
        documents.append(row["text"])

    _add_in_batches(collection, ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)
    return len(ids)


def _pool_review_variant(
    review_df: pd.DataFrame,
    review_embeddings: np.ndarray,
    *,
    polarity: str,
    weighting: str,
) -> dict[int, dict]:
    """Weighted mean-pool review chunk embeddings per app_id for one (polarity, weighting) arm."""
    if polarity == "recommended_only":
        sub = review_df[review_df["recommended"] == 1]
    else:
        sub = review_df

    out: dict[int, dict] = {}
    for app_id, g in sub.groupby("app_id"):
        idx = g.index.to_numpy()
        vecs = review_embeddings[idx]
        if weighting == "log_weighted":
            w = np.log1p(g["votes_helpful"].to_numpy(dtype=np.float64))
            if w.sum() <= 0:
                w = np.ones(len(g), dtype=np.float64)
        else:
            w = np.ones(len(g), dtype=np.float64)
        w = w / w.sum()
        pooled = l2_normalize((vecs * w[:, None]).sum(axis=0))
        out[int(app_id)] = {
            "vector": pooled,
            "n_reviews_pooled": int(len(g)),
            "recommended_rate": float(g["recommended"].mean()),
        }
    return out


def _blend_with_description(
    pooled_vector: np.ndarray, description_vector: np.ndarray, blend_weight: float
) -> np.ndarray:
    return l2_normalize((1.0 - blend_weight) * pooled_vector + blend_weight * description_vector)


def _write_game_profiles_collection(
    client: ClientAPI,
    *,
    collection_name: str,
    review_df: pd.DataFrame,
    review_embeddings: np.ndarray,
    description_vectors: dict[int, np.ndarray],
    blend_weight: float,
) -> tuple[int, dict[str, int]]:
    collection = _fresh_collection(client, collection_name)

    ids: list[str] = []
    vectors: list[np.ndarray] = []
    metadatas: list[dict] = []
    skipped_per_variant: dict[str, int] = {}

    for polarity in POLARITY_ARMS:
        for weighting in WEIGHTING_ARMS:
            variant = f"{polarity}__{weighting}"
            pooled_by_app = _pool_review_variant(
                review_df, review_embeddings, polarity=polarity, weighting=weighting
            )
            n_skipped = 0
            for app_id, description_vector in description_vectors.items():
                pooled = pooled_by_app.get(app_id)
                if pooled is None:
                    n_skipped += 1
                    continue
                blended = _blend_with_description(pooled["vector"], description_vector, blend_weight)
                ids.append(f"{app_id}::{variant}")
                vectors.append(blended)
                metadatas.append(
                    {
                        "app_id": app_id,
                        "variant": variant,
                        "recommended_rate": pooled["recommended_rate"],
                        "n_reviews_pooled": pooled["n_reviews_pooled"],
                        "blend_weight": float(blend_weight),
                    }
                )
            skipped_per_variant[variant] = n_skipped

    embeddings = np.stack(vectors, axis=0) if vectors else np.zeros((0, 512), dtype=np.float32)
    _add_in_batches(collection, ids=ids, embeddings=embeddings, metadatas=metadatas)
    return len(ids), skipped_per_variant


def _load_tfhub_embed_fn(tfhub_url: str):
    import tensorflow as tf
    import tensorflow_hub as hub

    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass
    return hub.load(tfhub_url)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Embed game_review_chunks.parquet and load Chroma collections via JSON config."
    )
    parser.add_argument("config", type=str, help="Path to JSON config for game-chunk-embeddings job.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    repo_root = Path(__file__).resolve().parents[1]
    input_path = repo_root / cfg["input_path"]
    chroma_persist_dir = repo_root / cfg["chroma_persist_dir"]
    review_chunks_collection = str(cfg.get("review_chunks_collection", "game_review_chunks"))
    game_profiles_collection = str(cfg.get("game_profiles_collection", "game_profiles"))
    tfhub_url = str(cfg.get("tfhub_url", "https://tfhub.dev/google/universal-sentence-encoder/4"))
    batch_size = int(cfg.get("batch_size", 64))
    max_chars_per_chunk = int(cfg.get("max_chars_per_chunk", 8000))
    description_blend_weight = float(cfg.get("description_blend_weight", 0.1))

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Missing required input artifact: {input_path}. "
            "Run scripts/recs_job_game_chunks.py first."
        )

    chroma_persist_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(input_path)
    if len(df) == 0:
        raise RuntimeError("Input chunk parquet is empty.")
    df = df.reset_index(drop=True)

    print(f"Input chunk rows: {input_path} (n={len(df):,})")
    print(f"Chroma persist dir: {chroma_persist_dir}")
    print(
        f"Params: batch_size={batch_size} max_chars_per_chunk={max_chars_per_chunk} "
        f"description_blend_weight={description_blend_weight}"
    )

    embed_fn = _load_tfhub_embed_fn(tfhub_url)
    embeddings = _encode_texts(
        df["text"].tolist(), embed_fn, batch_size=batch_size, max_chars=max_chars_per_chunk
    )
    print(f"Encoded {len(df):,} chunks -> dim={embeddings.shape[1]}")

    client = chromadb.PersistentClient(path=str(chroma_persist_dir))

    n_chunk_rows = _write_review_chunks_collection(
        client, collection_name=review_chunks_collection, df=df, embeddings=embeddings
    )
    print(f"Wrote {review_chunks_collection}: rows={n_chunk_rows:,}")

    review_df = df[df["chunk_type"] == "review"].reset_index(drop=True)
    review_embeddings = embeddings[df["chunk_type"].to_numpy() == "review"]
    description_df = df[df["chunk_type"] == "description"]
    description_embeddings = embeddings[df["chunk_type"].to_numpy() == "description"]
    description_vectors = {
        int(app_id): description_embeddings[i]
        for i, app_id in enumerate(description_df["app_id"].tolist())
    }

    n_profile_rows, skipped_per_variant = _write_game_profiles_collection(
        client,
        collection_name=game_profiles_collection,
        review_df=review_df,
        review_embeddings=review_embeddings,
        description_vectors=description_vectors,
        blend_weight=description_blend_weight,
    )
    print(f"Wrote {game_profiles_collection}: rows={n_profile_rows:,}")
    for variant, n_skipped in skipped_per_variant.items():
        if n_skipped:
            print(f"  variant={variant}: skipped {n_skipped} app_id(s) with zero eligible reviews")


if __name__ == "__main__":
    main()
