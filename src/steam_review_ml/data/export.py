"""Export cleaned Steam reviews to a single file (e.g. CSV)."""

from pathlib import Path
from typing import Optional, Sequence, Union

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from steam_review_ml.data.preprocess import filter_reviews, select_features

def export_cleaned_reviews(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    chunksize: int = 500_000,
    language: str = "english",
    columns: Optional[Sequence[str]] = None,
) -> None:
    """
    Stream raw CSV, apply filter → dedupe → select_features, write single CSV.

    Matches docs/data_filtering.md §7: stream → filter → dedupe (by review_id,
    keep first) → select_features → write. Order is filter-then-dedupe so the
    seen-ID set only contains IDs from kept rows (e.g. English only).

    Output is one CSV file (pandas only; no pyarrow). You can add Parquet
    later if you want (e.g. with pyarrow or fastparquet).
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    read_kwargs: dict = {"chunksize": chunksize}
    if columns is not None:
        read_kwargs["usecols"] = columns

    seen_review_ids: set = set()
    writer = None

    for chunk in pd.read_csv(input_path, **read_kwargs):
        filtered = filter_reviews(chunk, language=language)
        if len(filtered) == 0:
            continue

        # Dedupe: keep first occurrence per review_id (filter-then-dedupe order)
        mask = ~filtered["review_id"].isin(seen_review_ids)
        filtered = filtered[mask]
        seen_review_ids.update(filtered["review_id"].tolist())
        if len(filtered) == 0:
            continue

        featured = select_features(filtered)
        
        table = pa.Table.from_pandas(featured)
        if writer is None:
            writer = pq.ParquetWriter(output_path, table.schema)
        writer.write_table(table)
    
    if writer:
        writer.close()
