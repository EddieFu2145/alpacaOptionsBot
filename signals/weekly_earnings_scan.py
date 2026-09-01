"""Ranks this week's earnings reporters by AvgEA-Implied.

The paper forms quintiles across ~40-200 firms per quarter; a single
week's earnings cohort is usually far smaller (a handful to a few dozen
names), so quintile buckets don't mean much here - this just returns
every name ranked, and leaves picking "top N / bottom N" to the caller
rather than pretending a 6-name week has real quintiles.
"""
from datetime import date, timedelta
from typing import Optional

import pandas as pd

from data.earnings_calendar import earnings_this_week, recent_earnings_dates
from data.options import has_liquid_weekly_options
from .avgea_implied import avgea_implied

_HOUR_MAP = {"bmo": "bmo", "amc": "amc", "dmh": "amc"}  # dmh (during market hours) approximated as amc


def _week_friday(as_of: date) -> date:
    monday = as_of - timedelta(days=as_of.weekday())
    return monday + timedelta(days=4)


def scan_this_week(as_of: Optional[date] = None) -> pd.DataFrame:
    as_of = as_of or date.today()
    expiration = _week_friday(as_of)

    reporters = earnings_this_week(as_of)
    if reporters.empty:
        return pd.DataFrame(columns=["symbol", "avg_ea", "implied", "avgea_implied"])

    rows = []
    for _, r in reporters.iterrows():
        symbol = r["symbol"]
        if not has_liquid_weekly_options(symbol, expiration):
            continue

        past = recent_earnings_dates(symbol, before=as_of)
        past = [(d, _HOUR_MAP.get(h, "amc")) for d, h in past]
        if len(past) < 4:
            continue  # not enough earnings history for a stable AvgEA

        result = avgea_implied(symbol, expiration, past)
        if result:
            rows.append(result)

    if not rows:
        return pd.DataFrame(columns=["symbol", "avg_ea", "implied", "avgea_implied"])

    return pd.DataFrame(rows).sort_values("avgea_implied", ascending=False).reset_index(drop=True)
