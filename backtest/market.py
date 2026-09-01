"""Price lookups for the backtest engine.

Fetches each underlying's full price history once per run (rather than once
per week) and serves it from memory; option data is still fetched lazily
since the set of contracts touched depends on what the strategy trades.
"""
import math
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta

import pandas as pd

from data.equities import get_stock_bars
from data.options import get_expired_contracts, get_option_bars


def _to_datetime(day: date, end_of_day: bool = False) -> datetime:
    return datetime.combine(day, time(23, 59) if end_of_day else time())


@dataclass
class MarketContext:
    underlyings: list[str]
    start: date
    end: date
    history_buffer_days: int = 60  # extra lookback so indicators (e.g. realized vol) work from day one
    _underlying_close: dict[str, dict[date, float]] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        fetch_start = self.start - timedelta(days=self.history_buffer_days)
        for symbol in self.underlyings:
            bars = get_stock_bars(symbol, _to_datetime(fetch_start), _to_datetime(self.end, True))
            self._underlying_close[symbol] = {
                ts.date(): close for ts, close in bars.loc[symbol]["close"].items()
            }

    def underlying_close(self, symbol: str, day: date) -> float:
        return self._underlying_close[symbol][day]

    def trailing_volatility(self, symbol: str, as_of: date, window: int = 20) -> float:
        """Annualized realized volatility of daily log returns over the
        `window` trading days strictly before `as_of`. Used as a stand-in for
        implied volatility, which Alpaca doesn't provide historically.
        """
        closes = self._underlying_close[symbol]
        days = sorted(d for d in closes if d < as_of)
        window_days = days[-(window + 1):]
        if len(window_days) < 2:
            raise ValueError(f"Not enough price history before {as_of} to compute volatility for {symbol}")
        prices = [closes[d] for d in window_days]
        log_returns = [math.log(prices[i] / prices[i - 1]) for i in range(1, len(prices))]
        mean = sum(log_returns) / len(log_returns)
        variance = sum((r - mean) ** 2 for r in log_returns) / (len(log_returns) - 1)
        return math.sqrt(variance) * math.sqrt(252)

    def contracts_expiring(self, underlying: str, expiration: date) -> pd.DataFrame:
        return get_expired_contracts(underlying, expiration, expiration)

    def option_closes(self, symbols: list[str], start: date, end: date) -> dict[str, dict[date, float]]:
        """{symbol: {date: close}} for the given contracts over [start, end]. A
        symbol/date with no trades that day is simply absent from the dict -
        many contracts don't trade every day, which is expected, not an error.
        """
        if not symbols:
            return {}
        bars = get_option_bars(symbols, _to_datetime(start), _to_datetime(end, True))
        if bars.empty:
            return {symbol: {} for symbol in symbols}
        result: dict[str, dict[date, float]] = {symbol: {} for symbol in symbols}
        for (symbol, ts), close in bars["close"].items():
            result[symbol][ts.date()] = close
        return result
