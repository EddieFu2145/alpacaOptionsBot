"""Near-term macro/economic event calendar - NFP, CPI, FOMC, PPI, and
similar high-impact scheduled releases. Alpaca and Finnhub (used for
earnings) have neither; this covers a completely separate risk the
earnings-risk check doesn't touch at all.

Only needs near-term coverage: this system trades weekly credit spreads
expiring a handful of days out, so there's no need for a deep historical
archive - just "does anything high-impact land on/before this Friday's
expiration". Source is the free, no-key ForexFactory-style weekly feed
(confirmed live: returns real, current data - a real NFP + Unemployment
Rate + Average Hourly Earnings trio showed up on the exact date checked
during development). Covers this week and next week only; an expiration
further out than that returns checked=False (unknown), not a false "clear".
"""
import json
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import requests

from .cache import CACHE_DIR

_THIS_WEEK_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
_NEXT_WEEK_URL = "https://nfs.faireconomy.media/ff_calendar_nextweek.json"
_CACHE_PATH = CACHE_DIR / "macro_calendar_cache.json"
_CACHE_TTL_SECONDS = 20 * 3600  # confirmed live: this free feed rate-limits (429) aggressively - a handful of requests in a few minutes was enough to get blocked. Refreshing roughly once/day keeps real usage well under whatever threshold that is; the feed's own content only changes week to week anyway.


class MacroFeedUnavailable(Exception):
    pass


def _fetch_raw() -> list[dict]:
    """Raises MacroFeedUnavailable if EVERY source URL failed - confirmed
    live that this free feed is genuinely flaky (worked once, then failed
    twice within minutes in the same session with an empty/invalid body).
    A silent `except RequestException: continue` on every URL used to mean
    "0 events found" and "couldn't reach the source at all" looked
    identical to the caller - which turned "unknown" into a false "clear",
    the exact fail-open danger earnings_before_expiration is careful to
    avoid. One URL failing while the other succeeds is still reported as
    partial data, not a hard failure.
    """
    cache = {}
    if _CACHE_PATH.exists():
        try:
            cache = json.loads(_CACHE_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            cache = {}

    if cache.get("fetched_at", 0) > time.time() - _CACHE_TTL_SECONDS:
        return cache["events"]

    events: list[dict] = []
    successes = 0
    for url in (_THIS_WEEK_URL, _NEXT_WEEK_URL):
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            events.extend(resp.json())
            successes += 1
        except requests.exceptions.RequestException:
            continue

    if successes == 0:
        raise MacroFeedUnavailable("both this-week and next-week calendar fetches failed")

    CACHE_DIR.mkdir(exist_ok=True)
    _CACHE_PATH.write_text(json.dumps({"fetched_at": time.time(), "events": events}))
    return events


def high_impact_events(start: date, end: date, country: str = "USD") -> list[dict]:
    """High-impact events for `country` in [start, end]. Only reliable for
    dates within roughly the next two weeks (the feed's own coverage) -
    call `macro_risk_before_expiration` for the checked/unknown framing
    rather than trusting an empty list from this to mean "nothing"."""
    raw = _fetch_raw()
    results = []
    for e in raw:
        if e.get("impact") != "High" or e.get("country") != country:
            continue
        try:
            event_date = datetime.fromisoformat(e["date"]).date()
        except (KeyError, ValueError):
            continue
        if start <= event_date <= end:
            results.append({"date": event_date.isoformat(), "title": e.get("title", ""), "country": e.get("country", "")})
    return results


def macro_risk_before_expiration(expiration: date, underlying_country: str = "USD") -> dict:
    """Does a high-impact macro event (NFP, CPI, FOMC, PPI, etc.) land
    on or before `expiration`? Mirrors earnings_before_expiration's shape
    and fail-open-as-unknown behavior: `checked=False` means the feed
    couldn't be reached or the expiration is too far out for this feed's
    coverage - reported as unknown, not silently treated as "no risk".
    """
    today = date.today()
    if expiration <= today:
        return {"has_macro_risk": False, "events": [], "checked": True}

    # The feed only realistically covers ~this week + next week - treat
    # anything further out as beyond what this check can verify, rather
    # than returning an empty list that looks identical to "checked and
    # clear".
    if (expiration - today).days > 13:
        return {"has_macro_risk": None, "events": [], "checked": False, "error": "expiration beyond this feed's ~2-week coverage"}

    try:
        events = high_impact_events(today, expiration, country=underlying_country)
    except Exception as exc:
        return {"has_macro_risk": None, "events": [], "checked": False, "error": str(exc)}

    return {"has_macro_risk": bool(events), "events": events, "checked": True}
