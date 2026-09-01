"""Earnings dates via Finnhub's free-tier calendar API - Alpaca has no
earnings data at all (its corporate-actions API covers splits/dividends/
mergers only). Needs FINNHUB_API_KEY in .env; sign up free at finnhub.io.

Verified against a real key: the free tier only returns data from roughly
the past 25-30 days forward, not deeper history - confirmed empirically
(a window ending >30 days ago returns empty; anything more recent or
forward returns real rows). That means `earnings_this_week` and
`earnings_before_expiration` (both forward-looking, within that window)
work reliably; `recent_earnings_dates` (needs up to ~12 months back for 4
quarterly reports) does not and will typically return an empty list on
this tier.
"""
import os
from datetime import date, timedelta
from typing import Optional

import pandas as pd
import requests
from dotenv import load_dotenv

from .cache import cached_fetch

load_dotenv()

_FINNHUB_KEY = os.getenv("FINNHUB_API_KEY")
_BASE_URL = "https://finnhub.io/api/v1/calendar/earnings"


def _get(params: dict) -> list[dict]:
    if not _FINNHUB_KEY:
        return []
    resp = requests.get(_BASE_URL, params={**params, "token": _FINNHUB_KEY}, timeout=10)
    resp.raise_for_status()
    return resp.json().get("earningsCalendar", [])


def earnings_in_range(start: date, end: date, symbol: Optional[str] = None) -> pd.DataFrame:
    """All earnings announcements in [start, end], optionally for one symbol.
    Columns: symbol, date, hour ('bmo' | 'amc' | 'dmh'). Empty (not an
    error) if FINNHUB_API_KEY isn't set.
    """
    if not _FINNHUB_KEY:
        return pd.DataFrame(columns=["symbol", "date", "hour"])

    cache_key = f"earnings_{symbol or 'ALL'}_{start:%Y%m%d}_{end:%Y%m%d}"

    def fetch() -> pd.DataFrame:
        params = {"from": start.isoformat(), "to": end.isoformat()}
        if symbol:
            params["symbol"] = symbol
        rows = _get(params)
        return pd.DataFrame(
            [{"symbol": r["symbol"], "date": date.fromisoformat(r["date"]), "hour": r.get("hour", "")} for r in rows],
            columns=["symbol", "date", "hour"],
        )

    return cached_fetch(cache_key, fetch)


def earnings_this_week(as_of: Optional[date] = None) -> pd.DataFrame:
    """Everyone reporting in the trading week containing `as_of` (default
    today) - the actual live-screening entry point for the AvgEA-Implied
    strategy."""
    as_of = as_of or date.today()
    monday = as_of - timedelta(days=as_of.weekday())
    friday = monday + timedelta(days=4)
    return earnings_in_range(monday, friday)


def earnings_before_expiration(underlying: str, expiration: date) -> dict:
    """Does `underlying` report earnings before an option on it expires?
    This is the forward-looking check (today -> expiration), which is
    exactly the window Finnhub's free tier actually covers - unlike
    `recent_earnings_dates`, this one is reliable on this tier.

    `checked=False` means a transient failure (a real Finnhub 503 showed
    up during testing) prevented verifying either way - this is reported
    as unknown, not silently treated as "no risk". Failing open here would
    mean an API hiccup quietly waves through exactly the trades this check
    exists to catch.
    """
    today = date.today()
    if expiration <= today:
        return {"has_earnings_risk": False, "earnings_dates": [], "checked": True}

    try:
        rows = earnings_in_range(today, expiration, symbol=underlying)
    except requests.exceptions.RequestException as exc:
        return {"has_earnings_risk": None, "earnings_dates": [], "checked": False, "error": str(exc)}

    if rows.empty:
        return {"has_earnings_risk": False, "earnings_dates": [], "checked": True}

    dates = [{"date": d.isoformat(), "hour": h} for d, h in zip(rows["date"], rows["hour"])]
    return {"has_earnings_risk": True, "earnings_dates": dates, "checked": True}


def recent_earnings_dates(symbol: str, before: date, n: int = 4) -> list[tuple[date, str]]:
    """A symbol's last `n` earnings dates strictly before `before`, each as
    (date, hour). Looks back 3 years, which should comfortably cover 4
    quarterly reports even accounting for schedule slippage."""
    df = earnings_in_range(before - timedelta(days=3 * 365), before - timedelta(days=1), symbol=symbol)
    if df.empty:
        return []
    df = df.sort_values("date", ascending=False).head(n).sort_values("date")
    return list(zip(df["date"], df["hour"]))
