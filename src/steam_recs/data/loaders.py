import os
from typing import Optional, Sequence

import pandas as pd


def chunk_csv(
    input_file: str,
    output_dir: str,
    chunksize: int = 100_000,
    columns: Optional[Sequence[str]] = None,
    language: str = "english",
) -> None:
    """
    Split a large CSV into smaller chunks, optionally filtering columns and language.
    """
    os.makedirs(output_dir, exist_ok=True)

    for i, chunk in enumerate(pd.read_csv(input_file, chunksize=chunksize)):
        if columns:
            # Ensure we only select existing columns
            valid_columns = [c for c in columns if c in chunk.columns]
            chunk = chunk[valid_columns]
        if "language" in chunk.columns and language is not None:
            chunk = chunk[chunk["language"] == language]
        chunk_file = os.path.join(output_dir, f"chunk_{i + 1}.csv")
        chunk.to_csv(chunk_file, index=False)
        print(f"Saved {chunk_file}")

