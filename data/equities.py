"""Historical and live stock price data."""
import hashlib
from datetime import datetime

import pandas as pd

from alpaca.data.enums import DataFeed
from alpaca.data.requests import StockBarsRequest, StockLatestTradeRequest
from alpaca.data.timeframe import TimeFrame

from .cache import cached_fetch
from .clients import stock_client


def latest_spot(symbol: str) -> float:
    trade = stock_client().get_stock_latest_trade(StockLatestTradeRequest(symbol_or_symbols=symbol))
    return trade[symbol].price


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


def get_bulk_stock_bars(
    symbols: list[str],
    start: datetime,
    end: datetime,
    timeframe: TimeFrame = TimeFrame.Day,
) -> pd.DataFrame:
    """OHLCV bars for MANY symbols in one request, indexed by (symbol,
    timestamp) - same shape as get_stock_bars, just for a whole universe
    at once instead of one call per symbol.

    This is what actually makes screening hundreds of names at once
    possible: Alpaca's own StockBarsRequest already accepts a list via
    `symbol_or_symbols` (confirmed in the SDK - get_stock_bars above just
    never passed more than one), so the wide screen's data cost is ONE
    round trip regardless of universe size, and the mean-reversion/vol-rank
    math on top of it is vectorized pandas, not a per-symbol Python loop.
    """
    # Python's builtin hash() is randomized per-process for strings
    # (PYTHONHASHSEED) - using it here would give this cache a different
    # key on every single process restart, silently never hitting across
    # runs despite looking like a working cache within one. hashlib is
    # stable across processes, which is the entire point of an on-disk
    # cache that's meant to survive the bot (and now the dashboard, which
    # calls this same function independently) restarting.
    key_symbols = "-".join(sorted(symbols))
    symbols_hash = hashlib.md5(key_symbols.encode()).hexdigest()[:12]
    cache_key = f"stock_bars_bulk_{symbols_hash}_{len(symbols)}_{timeframe}_{start:%Y%m%d}_{end:%Y%m%d}"

    def fetch() -> pd.DataFrame:
        request = StockBarsRequest(
            symbol_or_symbols=symbols,
            start=start,
            end=end,
            timeframe=timeframe,
            feed=DataFeed.IEX,
        )
        return stock_client().get_stock_bars(request).df

    return cached_fetch(cache_key, fetch)
