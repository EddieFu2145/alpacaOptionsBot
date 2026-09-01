"""Company fundamentals via Finnhub - Alpaca has none of this. Confirmed
directly: neither the screener response (get_most_actives/get_market_movers)
nor TradingClient.get_asset exposes market cap or shares outstanding: Alpaca
is a broker/data API, not a fundamentals provider.

Reuses FINNHUB_API_KEY - the same key data/earnings_calendar.py needs.

Retries + same-day caching added after observing a real ~50% transient
failure rate on individual calls during testing (likely rate-limit
pressure) - without this, data/screener.py's output was non-deterministic
run to run: a genuinely large-cap, liquid name could randomly drop in or
out of the candidate list purely from an API hiccup, not a real decision.
"""
import json
import os
import time
from datetime import date
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()

_FINNHUB_KEY = os.getenv("FINNHUB_API_KEY")
_PROFILE_URL = "https://finnhub.io/api/v1/stock/profile2"
_CACHE_PATH = Path(__file__).resolve().parent.parent / "data_cache" / "market_cap_cache.json"


def _load_cache() -> dict:
    if not _CACHE_PATH.exists():
        return {}
    return json.loads(_CACHE_PATH.read_text())


def _save_cache(cache: dict) -> None:
    _CACHE_PATH.parent.mkdir(exist_ok=True)
    _CACHE_PATH.write_text(json.dumps(cache))


def market_cap(symbol: str) -> Optional[float]:
    """Market cap in dollars, or None if unavailable - no key, Finnhub has
    no profile for this symbol, or all retries were exhausted on a
    transient failure. None is already the safe outcome here:
    data/screener.py treats an unverified cap as "fall back to the
    curated large-cap list", not "assume it's fine" - so failing to None
    doesn't quietly wave anything through.
    """
    if not _FINNHUB_KEY:
        return None

    cache_key = f"{symbol}_{date.today().isoformat()}"
    cache = _load_cache()
    if cache_key in cache:
        return cache[cache_key]

    result = None
    for attempt in range(3):
        try:
            resp = requests.get(_PROFILE_URL, params={"symbol": symbol, "token": _FINNHUB_KEY}, timeout=10)
            resp.raise_for_status()
            cap_millions = resp.json().get("marketCapitalization")
            result = cap_millions * 1_000_000 if cap_millions else None
            break
        except requests.exceptions.RequestException:
            if attempt < 2:
                time.sleep(0.5)

    cache[cache_key] = result
    _save_cache(cache)
    return result
