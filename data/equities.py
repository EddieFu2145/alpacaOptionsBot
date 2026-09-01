"""Historical stock bar data."""
from datetime import datetime

import pandas as pd

from alpaca.data.enums import DataFeed
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from .cache import cached_fetch
from .clients import stock_client


def get_stock_bars(
    symbol: str,
    start: datetime,
    end: datetime,
    timeframe: TimeFrame = TimeFrame.Day,
) -> pd.DataFrame:
    """OHLCV bars for a single symbol, indexed by (symbol, timestamp).

    Explicitly pinned to the IEX feed - this account's subscription doesn't
    permit recent SIP data, and the SDK defaults to SIP when `feed` is
    omitted. Older date ranges (backtests) happen not to trip the recency
    restriction, but any request touching the last ~15 minutes (e.g. the
    pre-market check) does, so pin it unconditionally rather than only
    where it's been observed to matter.
    """
    cache_key = f"stock_bars_{symbol}_{timeframe}_{start:%Y%m%d}_{end:%Y%m%d}"

    def fetch() -> pd.DataFrame:
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            start=start,
            end=end,
            timeframe=timeframe,
            feed=DataFeed.IEX,
        )
        return stock_client().get_stock_bars(request).df

    return cached_fetch(cache_key, fetch)
