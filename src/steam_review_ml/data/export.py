"""Export cleaned Steam reviews to a single file (Parquet)."""

from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from steam_review_ml.data.loaders import iter_cleaned_chunks


def write_cleaned_to_parquet(
    chunks_iter: Iterable[pd.DataFrame],
    output_path: Union[str, Path],
) -> None:
    """
    Write a single Parquet file from an iterator of DataFrames (e.g. from iter_cleaned_chunks).

    Schema is taken from the first chunk. Low memory: one chunk in memory at a time.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = None
    for featured in chunks_iter:
        table = pa.Table.from_pandas(featured, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(output_path, table.schema)
        writer.write_table(table)

    if writer is not None:
        writer.close()
