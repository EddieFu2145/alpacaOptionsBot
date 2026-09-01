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
        request = NewsRequest(
            symbols=symbols,
            start=start,
            end=end,
            limit=limit,
            include_content=include_content,
            exclude_contentless=True,
        )
        return news_client().get_news(request).df

    return cached_fetch(cache_key, fetch)
