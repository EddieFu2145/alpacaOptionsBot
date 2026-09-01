"""On-disk parquet cache for historical data pulls.

Backtests replay the same requests over and over; without this, every run
re-hits the Alpaca API for data that never changes once the historical
window is in the past.
"""
from pathlib import Path
from typing import Callable

import pandas as pd

CACHE_DIR = Path(__file__).resolve().parent.parent / "data_cache"
CACHE_DIR.mkdir(exist_ok=True)


def cached_fetch(cache_key: str, fetch_fn: Callable[[], pd.DataFrame]) -> pd.DataFrame:
    path = CACHE_DIR / f"{cache_key}.parquet"
    if path.exists():
        return pd.read_parquet(path)

    df = fetch_fn()
    if not df.empty:
        df.to_parquet(path)
    return df
