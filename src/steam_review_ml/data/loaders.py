"""Data loading for Steam reviews (raw and cleaned streams)."""
from pathlib import Path
from typing import Optional, Sequence, Union

import pandas as pd


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
