"""Option-quality signals for evaluating a candidate credit spread:
gamma structure, VRP, NVRP, IV rank (proxy), expected-move characteristics,
liquidity, and earnings risk. Most are computed from Alpaca's own live
chain snapshot - implied volatility and Greeks come directly from Alpaca
(no Black-Scholes inversion needed for live data, unlike the historical
backtest path) - plus historical stock prices for realized-vol context.
Earnings risk uses Finnhub (data/earnings_calendar.py) since Alpaca has
none; everything else needs no new API key.

Thresholds below (what counts as "elevated"/"high"/"solid") are reasonable
starting heuristics, not statistically calibrated breakpoints - there
isn't enough historical IV data available to calibrate them properly yet.
Treat them as a first pass to tune once real usage data exists.
"""
from datetime import date, datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from data.earnings_calendar import earnings_before_expiration
from data.macro_calendar import macro_risk_before_expiration
from data.equities import get_stock_bars, latest_spot
from live import strategy_config
from data.options import get_option_bars, get_option_chain, parse_occ_symbol

TRADING_DAYS_PER_YEAR = 252


def _daily_log_returns(underlying: str, start: date, end: date) -> pd.Series:
    bars = get_stock_bars(underlying, datetime.combine(start, datetime.min.time()), datetime.combine(end, datetime.min.time()))
    closes = bars.loc[underlying]["close"]
    return np.log(closes / closes.shift(1)).dropna()


def realized_volatility(underlying: str, as_of: date, window: int = 20) -> float:
    """Annualized realized vol over the trailing `window` trading days."""
    returns = _daily_log_returns(underlying, as_of - timedelta(days=window * 2 + 10), as_of)
    recent = returns.tail(window)
    if len(recent) < 2:
        raise ValueError(f"Not enough price history before {as_of} to compute realized vol for {underlying}")
    return float(recent.std() * (TRADING_DAYS_PER_YEAR**0.5))


def realized_vol_rank(underlying: str, as_of: date, window: int = 20, lookback_days: int = 252) -> dict:
    """Where today's trailing realized vol sits within its own trailing
    ~1-year distribution of realized-vol readings. Used as a stand-in for
    true IV rank, since Alpaca has no historical implied-vol series to
    rank against directly.
    """
    returns = _daily_log_returns(underlying, as_of - timedelta(days=int(lookback_days * 1.6)), as_of)
    if len(returns) < window + 20:
        raise ValueError(f"Not enough price history to build a vol-rank distribution for {underlying}")

    rolling_vol = (returns.rolling(window).std() * (TRADING_DAYS_PER_YEAR**0.5)).dropna()
    current = float(rolling_vol.iloc[-1])
    history = rolling_vol.iloc[:-1]
    rank_pct = float((history < current).mean()) * 100
    return {"current_realized_vol": current, "rank_pct": rank_pct, "n_observations": len(history), "high": rank_pct >= 70}


def variance_risk_premium(implied_vol: float, realized_vol: float) -> dict:
    cfg = strategy_config.load()
    vrp = implied_vol - realized_vol
    nvrp = vrp / realized_vol if realized_vol > 0 else float("nan")
    return {
        "implied_vol": implied_vol,
        "realized_vol": realized_vol,
        "vrp": vrp,
        "nvrp": nvrp,
        "vrp_elevated": vrp > cfg["vrp_threshold"],
        "nvrp_high": nvrp > cfg["nvrp_threshold"],
    }


def gamma_structure(short_gamma: float, long_gamma: float, spot: float, contracts: int) -> dict:
    """Net gamma of the short/long pair. A short option carries negative
    gamma exposure, a long option positive - a 'strong' (contained)
    structure is one where the two mostly offset rather than leaving a
    large net short-gamma position that whipsaws on small underlying moves.
    """
    net_gamma_per_share = long_gamma - short_gamma
    net_gamma_position = net_gamma_per_share * 100 * contracts
    dollar_gamma_1pct = 0.5 * net_gamma_position * (spot * 0.01) ** 2
    return {
        "short_gamma": short_gamma,
        "long_gamma": long_gamma,
        "net_gamma_position": net_gamma_position,
        "dollar_gamma_per_1pct_move": dollar_gamma_1pct,
    }


def expected_move_profile(
    underlying: str, as_of: date, spot: float, implied_vol: float, days_to_expiry: float,
    window: int = 20, lookback_days: int = 252,
) -> dict:
    """A fixed 1.5%/8% cutoff treats every underlying identically - a 5%
    weekly expected move is routine for a name like TSLA but extreme for
    something like WMT, so the same absolute number means very different
    things depending on the underlying's own vol regime.

    Alpaca has no historical implied-vol series to rank the CURRENT
    expected move against directly (same constraint realized_vol_rank
    already works around) - so instead this builds the underlying's own
    trailing distribution of what an equivalent-duration move would have
    looked like using its historical REALIZED vol at each point over the
    past year, and ranks today's actual (IV-based) expected move against
    that. It's an honest proxy, not a true apples-to-apples IV percentile:
    mixing today's IV against a realized-vol-based reference distribution
    is exactly the same kind of approximation VRP itself already makes.
    Falls back to the old fixed thresholds if there isn't enough price
    history yet to build the distribution.
    """
    years = max(days_to_expiry, 0) / 365
    move_pct = implied_vol * (years**0.5)
    move_dollars = spot * move_pct

    try:
        returns = _daily_log_returns(underlying, as_of - timedelta(days=int(lookback_days * 1.6)), as_of)
    except Exception:
        returns = pd.Series(dtype=float)

    if len(returns) < window + 20:
        if move_pct < 0.015:
            assessment = "narrow - limited premium available"
        elif move_pct > 0.08:
            assessment = "wide - elevated event/gap risk"
        else:
            assessment = "workable"
        return {
            "expected_move_pct": move_pct, "expected_move_dollars": move_dollars,
            "assessment": assessment, "percentile_vs_own_history": None,
        }

    rolling_vol = (returns.rolling(window).std() * (TRADING_DAYS_PER_YEAR**0.5)).dropna()
    historical_moves = rolling_vol * (years**0.5)  # same duration, RV-implied moves at each past point
    rank_pct = float((historical_moves < move_pct).mean()) * 100

    if rank_pct < 20:
        assessment = f"narrow relative to {underlying}'s own history ({rank_pct:.0f}th pct) - limited premium available"
    elif rank_pct > 80:
        assessment = f"wide relative to {underlying}'s own history ({rank_pct:.0f}th pct) - elevated event/gap risk"
    else:
        assessment = f"workable for {underlying} ({rank_pct:.0f}th pct of its own history)"

    return {
        "expected_move_pct": move_pct, "expected_move_dollars": move_dollars,
        "assessment": assessment, "percentile_vs_own_history": round(rank_pct, 1),
    }


def liquidity_check(symbol: str, bid_price: float, ask_price: float, bid_size: float, ask_size: float, lookback_days: int = 5) -> dict:
    end = date.today()
    start = end - timedelta(days=lookback_days * 3)  # calendar buffer for weekends/holidays
    bars = get_option_bars(symbol, datetime.combine(start, datetime.min.time()), datetime.combine(end, datetime.min.time()))
    avg_volume = float(bars["volume"].tail(lookback_days).mean()) if not bars.empty and "volume" in bars.columns else 0.0
    bid_size, ask_size = float(bid_size), float(ask_size)
    has_two_sided_quote = bool(pd.notna(bid_price) and pd.notna(ask_price) and bid_price > 0 and ask_price > 0)
    solid = bool(has_two_sided_quote and avg_volume >= 25 and bid_size >= 1 and ask_size >= 1)
    return {
        "avg_daily_volume": avg_volume,
        "bid_size": bid_size,
        "ask_size": ask_size,
        "has_two_sided_quote": has_two_sided_quote,
        "solid": solid,
    }


def spread_quality_report(underlying: str, short_symbol: str, long_symbol: str, contracts: int, credit_per_spread: float) -> dict:
    """Everything above, combined into one report for a specific candidate
    spread - the single tool call an agent should make before proposing a
    trade, rather than pulling the raw chain itself and reasoning over it
    unaided.
    """
    today = date.today()
    short_info = parse_occ_symbol(short_symbol)
    long_info = parse_occ_symbol(long_symbol)

    chain = get_option_chain(underlying, expiration_date=short_info["expiration"])
    short_rows = chain[chain["symbol"] == short_symbol]
    long_rows = chain[chain["symbol"] == long_symbol]
    if short_rows.empty or long_rows.empty:
        raise ValueError(f"'{short_symbol}' or '{long_symbol}' not found in the current chain for {underlying}")
    short_row, long_row = short_rows.iloc[0], long_rows.iloc[0]

    spot = latest_spot(underlying)
    days_to_expiry = (short_info["expiration"] - today).days
    iv = short_row["implied_volatility"]
    rv = realized_volatility(underlying, today)

    report: dict = {
        "underlying": underlying,
        "spot": spot,
        "days_to_expiry": days_to_expiry,
        "implied_volatility": None if pd.isna(iv) else float(iv),
        "realized_volatility": rv,
        "expected_move": expected_move_profile(underlying, today, spot, iv if pd.notna(iv) else rv, days_to_expiry),
        "vol_rank_proxy": realized_vol_rank(underlying, today),
        "liquidity_short_leg": liquidity_check(
            short_symbol, short_row["bid_price"], short_row["ask_price"], short_row["bid_size"], short_row["ask_size"]
        ),
        "liquidity_long_leg": liquidity_check(
            long_symbol, long_row["bid_price"], long_row["ask_price"], long_row["bid_size"], long_row["ask_size"]
        ),
        "earnings_risk": earnings_before_expiration(underlying, short_info["expiration"]),
        "macro_risk": macro_risk_before_expiration(short_info["expiration"]),
    }

    report["vrp"] = variance_risk_premium(float(iv), rv) if pd.notna(iv) else None

    if pd.notna(short_row["gamma"]) and pd.notna(long_row["gamma"]):
        gamma = gamma_structure(float(short_row["gamma"]), float(long_row["gamma"]), spot, contracts)
        credit_total = credit_per_spread * 100 * contracts
        gamma["contained_relative_to_credit"] = abs(gamma["dollar_gamma_per_1pct_move"]) < 0.5 * credit_total
        report["gamma"] = gamma
    else:
        report["gamma"] = None

    return report
