"""Candidate universe screener: Alpaca's own most-active/movers lists,
filtered down to names actually worth running the strategy on.

Alpaca has no market cap data (see data/fundamentals.py - confirmed
directly against both the screener response and get_asset), so the
market-cap filter uses Finnhub; without a key, this falls back to a
curated list of known large-cap, liquid-options names rather than letting
an unverified small-cap through unchecked.
"""
from datetime import date, timedelta
from typing import Optional

import pandas as pd

from alpaca.data.enums import MostActivesBy
from alpaca.data.historical.screener import ScreenerClient
from alpaca.data.requests import MarketMoversRequest, MostActivesRequest

from data.clients import _API_KEY, _SECRET_KEY
from data.fundamentals import market_cap
from data.options import get_option_chain, has_liquid_weekly_options
from signals.avgea_implied import latest_spot
from signals.options_quality import realized_vol_rank, realized_volatility, variance_risk_premium

# Known large-cap, reliably-liquid-weekly-options names - used when no
# FINNHUB_API_KEY is set (an arbitrary screener hit's market cap can't be
# verified then), and always checked alongside Alpaca's daily movers.
DEFAULT_LARGE_CAP_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO",
    "JPM", "V", "UNH", "XOM", "WMT", "COST", "NFLX", "AMD",
]

_screener_client = ScreenerClient(_API_KEY, _SECRET_KEY)


def _raw_screener_candidates(top: int = 20) -> list[str]:
    actives = _screener_client.get_most_actives(MostActivesRequest(by=MostActivesBy.VOLUME, top=top))
    movers = _screener_client.get_market_movers(MarketMoversRequest(top=max(top // 2, 1)))
    symbols = [a.symbol for a in actives.most_actives]
    symbols += [m.symbol for m in movers.gainers] + [m.symbol for m in movers.losers]
    return list(dict.fromkeys(symbols))  # dedupe, preserve order


def _week_friday(as_of: date) -> date:
    monday = as_of - timedelta(days=as_of.weekday())
    return monday + timedelta(days=4)


def candidate_universe(min_market_cap: float = 10_000_000_000, top: int = 20) -> list[str]:
    """Alpaca's daily movers, filtered to a market-cap floor and liquid
    weekly options. Without FINNHUB_API_KEY, market cap can't be verified
    for arbitrary tickers, so anything not on DEFAULT_LARGE_CAP_UNIVERSE
    is skipped rather than let through unchecked.
    """
    raw = _raw_screener_candidates(top)
    expiration = _week_friday(date.today())

    kept = []
    for symbol in raw:
        cap = market_cap(symbol)
        if cap is not None:
            if cap < min_market_cap:
                continue
        elif symbol not in DEFAULT_LARGE_CAP_UNIVERSE:
            continue  # can't verify cap and it's not on the known-safe list

        if has_liquid_weekly_options(symbol, expiration):
            kept.append(symbol)

    # Always consider the curated universe too, even if it didn't happen
    # to show up in today's movers/actives.
    for symbol in DEFAULT_LARGE_CAP_UNIVERSE:
        if symbol not in kept and has_liquid_weekly_options(symbol, expiration):
            kept.append(symbol)

    return kept


def _atm_implied_vol(symbol: str, expiration: date, spot: float) -> Optional[float]:
    chain = get_option_chain(symbol, expiration_date=expiration, strike_price_gte=spot * 0.95, strike_price_lte=spot * 1.05)
    chain = chain.dropna(subset=["implied_volatility"])
    if chain.empty:
        return None
    row = chain.iloc[(chain["strike"] - spot).abs().argsort()[:1]].iloc[0]
    return float(row["implied_volatility"])


def rank_by_vol_signal(symbols: list[str], as_of: Optional[date] = None) -> pd.DataFrame:
    """For each candidate: realized-vol rank (the IV-rank proxy from
    signals/options_quality.py) and VRP/NVRP off the nearest weekly ATM
    contract - ranked by NVRP descending, the direct "how rich is premium
    right now" read.
    """
    as_of = as_of or date.today()
    expiration = _week_friday(as_of)
    rows = []
    for symbol in symbols:
        try:
            spot = latest_spot(symbol)
            iv = _atm_implied_vol(symbol, expiration, spot)
            rv = realized_volatility(symbol, as_of)
            rank = realized_vol_rank(symbol, as_of)
        except (ValueError, KeyError):
            continue
        if iv is None:
            continue
        row = {"symbol": symbol, "spot": spot, "vol_rank_pct": rank["rank_pct"]}
        row.update(variance_risk_premium(iv, rv))
        rows.append(row)

    columns = ["symbol", "spot", "implied_vol", "realized_vol", "vrp", "nvrp", "vrp_elevated", "nvrp_high", "vol_rank_pct"]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows)[columns].sort_values("nvrp", ascending=False).reset_index(drop=True)
