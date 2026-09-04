"""Historical news headlines from Alpaca's built-in News API (Benzinga feed,
included with a standard Alpaca account, no separate subscription needed).
"""
from datetime import datetime

import pandas as pd

from alpaca.data.requests import NewsRequest

from .cache import cached_fetch
from .clients import news_client


def get_news(
    symbols: list[str] | str,
    start: datetime,
    end: datetime,
    limit: int = 50,
    include_content: bool = False,
) -> pd.DataFrame:
    """News articles mentioning the given symbol(s) in [start, end]."""
    key_symbols = symbols if isinstance(symbols, str) else "-".join(sorted(symbols))
    cache_key = f"news_{key_symbols}_{start:%Y%m%d}_{end:%Y%m%d}_{limit}_{include_content}"

    def fetch() -> pd.DataFrame:
        # NewsRequest's `symbols` field is actually a single comma-joined
        # string, despite this function's own list[str] | str type hint -
        # confirmed live: passing a real list raises a pydantic
        # ValidationError ("Input should be a valid string") instead of
        # querying multiple tickers in one call.
        request = NewsRequest(
            symbols=",".join(symbols) if isinstance(symbols, list) else symbols,
            start=start,
            end=end,
            limit=limit,
            include_content=include_content,
            exclude_contentless=True,
        )
        response = news_client().get_news(request)
        # alpaca-py's own NewsSet.df property crashes with
        # KeyError("None of ['id'] are in the columns") when the result set
        # is genuinely empty - confirmed live, reproduced directly against
        # a real symbol with zero recent news (LOW). A real SDK bug, not
        # ours, but it was crashing every single live session that happened
        # to check a thinly-covered name - which, with hundreds of names
        # now in the wide screen, is most sessions. "No news" is itself a
        # completely normal, expected result and should return quietly.
        if not response.data.get("news"):
            return pd.DataFrame()
        return response.df

    return cached_fetch(cache_key, fetch)
