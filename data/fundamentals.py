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
_LOCK_PATH = _CACHE_PATH.with_suffix(".lock")


class _FileLock:
    """Same mutex as live/trade_log.py - confirmed necessary here live: the
    wide screen's concurrent per-symbol lookups (data/screener.py's
    ThreadPoolExecutor) call market_cap() from up to 10 threads at once,
    and this cache is one shared JSON file read-modify-written by every
    call. Without a lock, two threads racing __save_cache__ mid-write
    truncates the file out from under a third thread's read - confirmed:
    this crashed with a JSONDecodeError on an empty file the first time
    the wide screen actually ran concurrently.
    """

    def __enter__(self):
        for _ in range(100):
            try:
                self._fd = open(_LOCK_PATH, "x")
                return self
            except (FileExistsError, PermissionError):
                # Confirmed live under real concurrent load (10 threads
                # hammering this same lock file): a create racing another
                # thread's delete can surface as PermissionError instead of
                # FileExistsError on Windows - a documented NTFS quirk during
                # a tight create/unlink race on the same path, not a real
                # permissions problem. Same "someone else has it, retry"
                # condition either way.
                time.sleep(0.05)
        raise TimeoutError("Could not acquire the market-cap cache lock after 5s")

    def __exit__(self, *exc_info):
        self._fd.close()
        _LOCK_PATH.unlink(missing_ok=True)


def _load_cache() -> dict:
    if not _CACHE_PATH.exists():
        return {}
    try:
        return json.loads(_CACHE_PATH.read_text())
    except json.JSONDecodeError:
        return {}  # torn read despite the lock (e.g. a leftover file from before locking existed) - treat as empty rather than crash


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
    with _FileLock():
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

    with _FileLock():
        cache = _load_cache()  # reload inside the lock - another thread may have written since our read above
        cache[cache_key] = result
        _save_cache(cache)
    return result
