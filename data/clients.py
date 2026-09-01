"""Shared, authenticated Alpaca API clients.

Every data-layer module pulls its client from here instead of reading
ALPACA_API_KEY / ALPACA_SECRET_KEY itself, so credentials are loaded once.
"""
from functools import lru_cache
import os

from dotenv import load_dotenv

from alpaca.data.historical.news import NewsClient
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.trading.client import TradingClient

load_dotenv()

_API_KEY = os.getenv("ALPACA_API_KEY")
_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
_PAPER = os.getenv("ALPACA_PAPER", "true").lower() == "true"


@lru_cache(maxsize=1)
def stock_client() -> StockHistoricalDataClient:
    return StockHistoricalDataClient(_API_KEY, _SECRET_KEY)


@lru_cache(maxsize=1)
def trading_client() -> TradingClient:
    return TradingClient(_API_KEY, _SECRET_KEY, paper=_PAPER)


@lru_cache(maxsize=1)
def option_client() -> OptionHistoricalDataClient:
    return OptionHistoricalDataClient(_API_KEY, _SECRET_KEY)


@lru_cache(maxsize=1)
def news_client() -> NewsClient:
    return NewsClient(_API_KEY, _SECRET_KEY)
