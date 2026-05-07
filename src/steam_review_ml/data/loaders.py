"""Data loading for Steam reviews (raw and cleaned streams)."""

import logging
import math
import heapq
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence, Union, Dict, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

from steam_review_ml.data.preprocess import (
    filter_reviews,
    select_features,
    convert_to_int,
    add_stratify_group,
    stable_split_u,
)

logger = logging.getLogger(__name__)
SPLIT_TRAIN = "train"
SPLIT_VAL = "val"
SPLIT_TEST = "test"
NORMALIZED_SPLIT_FILENAMES = {
    SPLIT_TRAIN: "steam_reviews_cleaned_english_train_norm.parquet",
    SPLIT_VAL: "steam_reviews_cleaned_english_val_norm.parquet",
    SPLIT_TEST: "steam_reviews_cleaned_english_test_norm.parquet",
}


def resolve_normalized_split_parquet(processed_dir: Path, split_name: str) -> tuple[Path, str]:
    """Resolve normalized split parquet path with val->train fallback when val is missing."""
    split = split_name.strip().lower()
    if split not in {SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST}:
        raise ValueError(f"split must be one of val/train/test, got: {split_name!r}")

    if split == SPLIT_VAL:
        val_path = processed_dir / NORMALIZED_SPLIT_FILENAMES[SPLIT_VAL]
        if val_path.is_file():
            return val_path, SPLIT_VAL
        train_path = processed_dir / NORMALIZED_SPLIT_FILENAMES[SPLIT_TRAIN]
        return train_path, SPLIT_TRAIN

    return processed_dir / NORMALIZED_SPLIT_FILENAMES[split], split


def load_normalized_split_df(
    path: Path,
    *,
    indexed_apps: set[int],
    min_review_chars: int,
    user_col: str = "author.steamid",
    time_col: str = "timestamp_created",
) -> pd.DataFrame:
    """Load and lightly filter a normalized split parquet for recommender eval."""
    usecols = [user_col, "app_id", "review", "recommended", "review_id", time_col]
    d = pd.read_parquet(path, columns=usecols).copy()
    d[user_col] = d[user_col].astype(str)
    d["review"] = d["review"].fillna("").astype(str)
    d["ts"] = pd.to_numeric(d[time_col], errors="coerce")
    d = d.dropna(subset=["ts"])
    d["ts"] = d["ts"].astype(np.float64)
    d = d[d["review"].str.len() >= int(min_review_chars)]
    d = d[d["app_id"].isin(indexed_apps)]
    return d


@dataclass(frozen=True)
class _HybridSplitMetadata:
    sparse_users: set[int]
    eval_assignments: Dict[int, Dict[int, str]]
    quantiles: Dict[str, float]


@dataclass(frozen=True)
class _SupportAwareSplitMetadata:
    singleton_assignments: Dict[int, str]
    eval_assignments: Dict[int, Dict[int, str]]


def _compute_random_split_labels(
    df: pd.DataFrame,
    *,
    random_state: int,
    cut_train: float,
    cut_val: float,
) -> pd.Series:
    """
    Deterministic hash-stratified split labels for one chunk.
    Returns a Series over df.index with values in {"train","val","test"}.
    """
    with_group = add_stratify_group(df)
    u = with_group.apply(
        lambda r: stable_split_u(
            seed=random_state,
            stratify_group=r["stratify_group"],
            review_id=r["review_id"],
        ),
        axis=1,
    )
    labels = pd.Series(SPLIT_TRAIN, index=df.index)
    labels.loc[(u >= cut_train) & (u < cut_val)] = SPLIT_VAL
    labels.loc[u >= cut_val] = SPLIT_TEST
    return labels


def _split_chunk_by_labels(
    df: pd.DataFrame,
    labels: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Slice one chunk into train/val/test dataframes from split labels."""
    return (
        df.loc[labels == SPLIT_TRAIN],
        df.loc[labels == SPLIT_VAL],
        df.loc[labels == SPLIT_TEST],
    )


def _prepare_hybrid_metadata(
    *,
    pf: pq.ParquetFile,
    chunksize: int,
    user_col: str,
    timestamp_col: str,
    sparse_user_threshold: int,
    per_user_val_n: int,
    per_user_test_n: int,
    train_size: float,
    val_size: float,
) -> _HybridSplitMetadata:
    """
    Hybrid pass A: collect sparse/dense users and per-dense-user eval assignments.
    """
    logger.info("Hybrid split pass A: collecting user interaction counts and recency candidates")

    user_counts: Dict[int, int] = defaultdict(int)
    topk_heap: Dict[int, list[Tuple[float, int]]] = defaultdict(list)
    ts_values: list[np.ndarray] = []
    top_k = per_user_val_n + per_user_test_n

    for chunk in tqdm(
        pf.iter_batches(batch_size=chunksize, columns=[user_col, timestamp_col, "review_id"]),
        desc="Hybrid split pass A",
        unit="batch",
    ):
        df_a = chunk.to_pandas()
        if len(df_a) == 0:
            continue

        uids = pd.to_numeric(df_a[user_col], errors="coerce")
        ts_series = pd.to_numeric(df_a[timestamp_col], errors="coerce").fillna(-1.0)
        rid_series = pd.to_numeric(df_a["review_id"], errors="coerce")

        ts_arr = ts_series.to_numpy(dtype=np.float64, copy=False)
        ts_values.append(ts_arr[np.isfinite(ts_arr)])

        for uid in uids.dropna().astype(np.int64).tolist():
            user_counts[int(uid)] += 1

        if top_k <= 0:
            continue
        for uid, ts, rid in zip(uids.tolist(), ts_series.tolist(), rid_series.tolist()):
            if pd.isna(uid) or pd.isna(rid):
                continue
            uid_i = int(uid)
            rid_i = int(rid)
            ts_f = float(ts)
            if not math.isfinite(ts_f):
                ts_f = -1.0
            heap = topk_heap[uid_i]
            item = (ts_f, rid_i)
            if len(heap) < top_k:
                heapq.heappush(heap, item)
            elif item > heap[0]:
                heapq.heapreplace(heap, item)

    sparse_users = {uid for uid, n in user_counts.items() if n < sparse_user_threshold}
    dense_users = {uid for uid, n in user_counts.items() if n >= sparse_user_threshold}

    quantiles: Dict[str, float] = {}
    if ts_values:
        all_ts = np.concatenate(ts_values)
        all_ts = all_ts[np.isfinite(all_ts)]
        if all_ts.size > 0:
            quantiles = {
                "q_train_end": float(np.quantile(all_ts, train_size)),
                "q_val_end": float(np.quantile(all_ts, train_size + val_size)),
            }

    eval_assignments: Dict[int, Dict[int, str]] = {}
    for uid in dense_users:
        heap = topk_heap.get(uid, [])
        if not heap:
            continue
        ordered = sorted(heap, key=lambda t: (t[0], t[1]), reverse=True)
        mapping: Dict[int, str] = {}
        for idx, (_, rid) in enumerate(ordered):
            if idx < per_user_test_n:
                mapping[rid] = SPLIT_TEST
            elif idx < (per_user_test_n + per_user_val_n):
                mapping[rid] = SPLIT_VAL
            else:
                break
        if mapping:
            eval_assignments[uid] = mapping

    logger.info(
        "Hybrid split pass A summary: users_total=%d sparse_users=%d dense_users=%d "
        "random_train/val/test=%.3f/%.3f/%.3f per_user_val_n=%d per_user_test_n=%d "
        "q_train_end=%s q_val_end=%s",
        len(user_counts),
        len(sparse_users),
        len(dense_users),
        train_size,
        val_size,
        1.0 - train_size - val_size,
        per_user_val_n,
        per_user_test_n,
        f"{quantiles.get('q_train_end'):.2f}" if "q_train_end" in quantiles else "NA",
        f"{quantiles.get('q_val_end'):.2f}" if "q_val_end" in quantiles else "NA",
    )

    return _HybridSplitMetadata(
        sparse_users=sparse_users,
        eval_assignments=eval_assignments,
        quantiles=quantiles,
    )


def _split_chunk_hybrid(
    df: pd.DataFrame,
    *,
    metadata: _HybridSplitMetadata,
    user_col: str,
    random_state: int,
    cut_train: float,
    cut_val: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Hybrid split for one chunk:
      - sparse users: deterministic random labels
      - dense users: pass-A temporal assignments (default to train)
    """
    uids = pd.to_numeric(df[user_col], errors="coerce")
    review_ids = pd.to_numeric(df["review_id"], errors="coerce")
    sparse_mask = uids.apply(lambda x: (not pd.isna(x)) and int(x) in metadata.sparse_users)

    labels = pd.Series(SPLIT_TRAIN, index=df.index)

    dense_rows = df.index[~sparse_mask]
    for idx in dense_rows:
        uid_raw = uids.loc[idx]
        rid_raw = review_ids.loc[idx]
        if pd.isna(uid_raw) or pd.isna(rid_raw):
            continue
        uid_i = int(uid_raw)
        rid_i = int(rid_raw)
        split_name = metadata.eval_assignments.get(uid_i, {}).get(rid_i)
        if split_name is not None:
            labels.loc[idx] = split_name

    sparse_df = df.loc[sparse_mask]
    if len(sparse_df) > 0:
        sparse_labels = _compute_random_split_labels(
            sparse_df,
            random_state=random_state,
            cut_train=cut_train,
            cut_val=cut_val,
        )
        labels.loc[sparse_labels.index] = sparse_labels

    return _split_chunk_by_labels(df, labels)


def _choose_eval_split(*, val_size: float, test_size: float, u: float) -> str:
    eval_total = val_size + test_size
    if eval_total <= 0:
        return SPLIT_VAL
    p_val = val_size / eval_total
    return SPLIT_VAL if u < p_val else SPLIT_TEST


def _prepare_support_aware_metadata(
    *,
    pf: pq.ParquetFile,
    chunksize: int,
    user_col: str,
    timestamp_col: str,
    random_state: int,
    train_size: float,
    val_size: float,
    test_size: float,
) -> _SupportAwareSplitMetadata:
    """
    Support-aware metadata pass:
      - n=1 user: assign whole user to train/val/test by global ratios
      - n=2 user: 1 train + 1 eval (val/test chosen per-user)
      - n>=3 user: >=1 train + >=2 eval rows in one eval split (val/test chosen per-user),
                   with eval rows selected as the most-recent rows by timestamp.
    """
    logger.info("Support-aware split pass A: collecting user interaction counts")
    user_counts: Dict[int, int] = defaultdict(int)

    for chunk in tqdm(
        pf.iter_batches(batch_size=chunksize, columns=[user_col]),
        desc="Support-aware split pass A",
        unit="batch",
    ):
        df_a = chunk.to_pandas()
        if len(df_a) == 0:
            continue
        uids = pd.to_numeric(df_a[user_col], errors="coerce")
        for uid in uids.dropna().astype(np.int64).tolist():
            user_counts[int(uid)] += 1

    singleton_assignments: Dict[int, str] = {}
    eval_split_by_user: Dict[int, str] = {}
    eval_count_by_user: Dict[int, int] = {}

    for uid, n in user_counts.items():
        u_main = stable_split_u(seed=random_state, stratify_group="support-aware-main", review_id=uid)
        u_eval = stable_split_u(seed=random_state, stratify_group="support-aware-eval", review_id=uid)

        if n == 1:
            if u_main < train_size:
                singleton_assignments[uid] = SPLIT_TRAIN
            elif u_main < (train_size + val_size):
                singleton_assignments[uid] = SPLIT_VAL
            else:
                singleton_assignments[uid] = SPLIT_TEST
            continue

        eval_split = _choose_eval_split(val_size=val_size, test_size=test_size, u=u_eval)
        eval_split_by_user[uid] = eval_split

        if n == 2:
            eval_count_by_user[uid] = 1
            continue

        eval_ratio = val_size if eval_split == SPLIT_VAL else test_size
        n_eval = int(round(n * eval_ratio))
        n_eval = max(2, n_eval)
        n_eval = min(n - 1, n_eval)  # keep >=1 row in train
        eval_count_by_user[uid] = int(n_eval)

    logger.info(
        "Support-aware split pass A summary: users_total=%d singleton_users=%d users_with_eval=%d",
        len(user_counts),
        len(singleton_assignments),
        len(eval_count_by_user),
    )

    logger.info("Support-aware split pass B: collecting most-recent eval rows per user")
    topk_heap: Dict[int, list[Tuple[float, int]]] = defaultdict(list)
    for chunk in tqdm(
        pf.iter_batches(batch_size=chunksize, columns=[user_col, timestamp_col, "review_id"]),
        desc="Support-aware split pass B",
        unit="batch",
    ):
        df_b = chunk.to_pandas()
        if len(df_b) == 0:
            continue
        uids = pd.to_numeric(df_b[user_col], errors="coerce")
        ts_series = pd.to_numeric(df_b[timestamp_col], errors="coerce").fillna(-1.0)
        rid_series = pd.to_numeric(df_b["review_id"], errors="coerce")

        for uid, ts, rid in zip(uids.tolist(), ts_series.tolist(), rid_series.tolist()):
            if pd.isna(uid) or pd.isna(rid):
                continue
            uid_i = int(uid)
            rid_i = int(rid)
            k = eval_count_by_user.get(uid_i, 0)
            if k <= 0:
                continue
            ts_f = float(ts)
            if not math.isfinite(ts_f):
                ts_f = -1.0
            heap = topk_heap[uid_i]
            item = (ts_f, rid_i)
            if len(heap) < k:
                heapq.heappush(heap, item)
            elif item > heap[0]:
                heapq.heapreplace(heap, item)

    eval_assignments: Dict[int, Dict[int, str]] = {}
    for uid, heap in topk_heap.items():
        split_name = eval_split_by_user.get(uid)
        if split_name is None:
            continue
        mapping: Dict[int, str] = {}
        for _, rid in heap:
            mapping[int(rid)] = split_name
        if mapping:
            eval_assignments[uid] = mapping

    return _SupportAwareSplitMetadata(
        singleton_assignments=singleton_assignments,
        eval_assignments=eval_assignments,
    )


def _split_chunk_support_aware(
    df: pd.DataFrame,
    *,
    metadata: _SupportAwareSplitMetadata,
    user_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    uids = pd.to_numeric(df[user_col], errors="coerce")
    review_ids = pd.to_numeric(df["review_id"], errors="coerce")
    labels = pd.Series(SPLIT_TRAIN, index=df.index)

    for idx in df.index:
        uid_raw = uids.loc[idx]
        rid_raw = review_ids.loc[idx]
        if pd.isna(uid_raw):
            continue
        uid_i = int(uid_raw)

        singleton_split = metadata.singleton_assignments.get(uid_i)
        if singleton_split is not None:
            labels.loc[idx] = singleton_split
            continue

        if pd.isna(rid_raw):
            continue
        rid_i = int(rid_raw)
        split_name = metadata.eval_assignments.get(uid_i, {}).get(rid_i)
        if split_name is not None:
            labels.loc[idx] = split_name

    return _split_chunk_by_labels(df, labels)


def load_raw_reviews(
    path: Union[str, Path],
    nrows: Optional[int] = None,
    usecols: Optional[Sequence[str]] = None,
    **read_csv_kwargs,
) -> pd.DataFrame:
    """
    Load the raw Steam reviews CSV into a DataFrame.

    Parameters
    ----------
    path:
        Path to the raw CSV file.
    nrows:
        Optional limit on number of rows to read (for quick experiments).
    usecols:
        Optional subset of columns to read.
    read_csv_kwargs:
        Extra keyword arguments forwarded to ``pd.read_csv``.
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Raw reviews file not found at: {csv_path}")

    return pd.read_csv(csv_path, nrows=nrows, usecols=usecols, **read_csv_kwargs)


def _dedupe_keep_first(
    df: pd.DataFrame,
    seen_ids: set,
    id_col: str = "review_id",
) -> pd.DataFrame:
    """
    Keep rows whose id_col is not in seen_ids; update seen_ids in place.
    Returns the deduped DataFrame (first occurrence per id kept).
    """
    mask = ~df[id_col].isin(seen_ids)
    out = df[mask]
    seen_ids.update(out[id_col].tolist())
    return out


def iter_clean_chunks(
    input_path: Union[str, Path],
    chunksize: int,
    language: str = "english",
    columns: Optional[Sequence[str]] = None,
) -> Iterator[pd.DataFrame]:
    """
    Stream raw CSV in chunks, apply filter → dedupe → select_features, yield each chunk.

    Row-level feature engineering (``review_word_count``, ``review_length_chars``) runs **after**
    the train/val/test split (see ``scripts/split_reviews.py``), not here.

    Pipeline only; no file output. Use write_parquet_chunked to write the cleaned Parquet.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    logger.info("iter_clean_chunks: input=%s chunksize=%s language=%s", input_path, chunksize, language)

    read_kwargs: dict = {"chunksize": chunksize}
    if columns is not None:
        read_kwargs["usecols"] = columns
        logger.debug("  usecols=%s", columns)

    seen_review_ids: set = set()
    chunk_iter = pd.read_csv(input_path, **read_kwargs)

    for chunk in tqdm(chunk_iter, desc="Reading & cleaning chunks", unit="chunk"):
        filtered = filter_reviews(chunk, language=language)
        if len(filtered) == 0:
            continue

        filtered = _dedupe_keep_first(filtered, seen_review_ids)
        if len(filtered) == 0:
            continue

        featured = select_features(filtered)
        bool_cols = [
            "recommended",
            "is_helpful",
            "steam_purchase",
            "received_for_free",
            "written_during_early_access",
        ]
        featured = convert_to_int(
            featured, [col for col in bool_cols if col in featured.columns]
        )
        yield featured

def iter_split_chunks(
    input_path: Union[str, Path],
    chunksize: int,
    val_size: float,
    test_size: float,
    random_state: int,
    split_mode: str = "random_stratified",
    sparse_user_threshold: int = 3,
    user_col: str = "author.steamid",
    timestamp_col: str = "timestamp_created",
    per_user_val_n: int = 1,
    per_user_test_n: int = 1,
) -> Iterable[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """
    Stream cleaned reviews in chunks, split into train, validation, and test sets.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input Parquet not found: {input_path}")

    if val_size < 0 or test_size < 0 or (val_size + test_size) >= 1:
        raise ValueError("val_size and test_size must be >=0 and sum to < 1")

    train_size = 1.0 - val_size - test_size
    cut_train = train_size
    cut_val = train_size + val_size

    # pandas read_parquet does not support 'chunksize'; use pyarrow to iterate Parquet row groups
    pf = pq.ParquetFile(input_path)

    logger.info(
        "iter_split_chunks: input=%s batch_size=%s split_mode=%s train/val/test=%.3f/%.3f/%.3f sparse_threshold=%d seed=%s",
        input_path,
        chunksize,
        split_mode,
        train_size,
        val_size,
        test_size,
        sparse_user_threshold,
        random_state,
    )

    split_mode = split_mode.strip().lower()
    if split_mode not in {"random_stratified", "hybrid_user_temporal", "support_aware_user_temporal"}:
        raise ValueError(
            "split_mode must be one of {'random_stratified', 'hybrid_user_temporal', 'support_aware_user_temporal'}"
        )

    hybrid_meta: Optional[_HybridSplitMetadata] = None
    support_meta: Optional[_SupportAwareSplitMetadata] = None
    if split_mode == "hybrid_user_temporal":
        if per_user_val_n < 0 or per_user_test_n < 0:
            raise ValueError("per_user_val_n and per_user_test_n must be >= 0")
        if user_col not in pf.schema_arrow.names:
            raise KeyError(f"Missing required user column for hybrid split: {user_col!r}")
        if timestamp_col not in pf.schema_arrow.names:
            raise KeyError(f"Missing required timestamp column for hybrid split: {timestamp_col!r}")

        hybrid_meta = _prepare_hybrid_metadata(
            pf=pf,
            chunksize=chunksize,
            user_col=user_col,
            timestamp_col=timestamp_col,
            sparse_user_threshold=sparse_user_threshold,
            per_user_val_n=per_user_val_n,
            per_user_test_n=per_user_test_n,
            train_size=train_size,
            val_size=val_size,
        )

        # Re-open file iterator for pass B.
        pf = pq.ParquetFile(input_path)
    elif split_mode == "support_aware_user_temporal":
        if user_col not in pf.schema_arrow.names:
            raise KeyError(f"Missing required user column for support-aware split: {user_col!r}")
        if timestamp_col not in pf.schema_arrow.names:
            raise KeyError(f"Missing required timestamp column for support-aware split: {timestamp_col!r}")
        support_meta = _prepare_support_aware_metadata(
            pf=pf,
            chunksize=chunksize,
            user_col=user_col,
            timestamp_col=timestamp_col,
            random_state=random_state,
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
        )
        # Re-open file iterator for pass C.
        pf = pq.ParquetFile(input_path)

    for batch_idx, chunk in enumerate(
        tqdm(
            pf.iter_batches(batch_size=chunksize),
            desc="Reading & splitting chunks",
            unit="batch",
        ),
        start=1,
    ):
        df = chunk.to_pandas()
        if len(df) == 0:
            continue

        if split_mode == "random_stratified":
            labels = _compute_random_split_labels(
                df,
                random_state=random_state,
                cut_train=cut_train,
                cut_val=cut_val,
            )
            train_df, val_df, test_df = _split_chunk_by_labels(df, labels)
        elif split_mode == "hybrid_user_temporal":
            if hybrid_meta is None:
                raise RuntimeError("hybrid metadata missing; pass-A preparation should run before pass-B")
            train_df, val_df, test_df = _split_chunk_hybrid(
                df,
                metadata=hybrid_meta,
                user_col=user_col,
                random_state=random_state,
                cut_train=cut_train,
                cut_val=cut_val,
            )
        else:
            if support_meta is None:
                raise RuntimeError("support-aware metadata missing; preparation should run before split pass")
            train_df, val_df, test_df = _split_chunk_support_aware(
                df,
                metadata=support_meta,
                user_col=user_col,
            )

        logger.debug(
            "split batch %d: raw=%d -> train=%d val=%d test=%d",
            batch_idx,
            len(df),
            len(train_df),
            len(val_df),
            len(test_df),
        )
        yield train_df, val_df, test_df
