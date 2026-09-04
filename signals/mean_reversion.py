"""Extreme mean-reversion signal: how far the current price sits from its
own trailing simple moving average, in standard deviations.

Built to replace what was previously a pure narrative requirement: the
directional trade path (propose_and_execute_credit_spread) only ever
required the LLM to assert "a real, stated directional thesis" in its own
rationale text, with zero code-level verification - the same class of gap
VRP/NVRP had until it became a hard gate. This gives the directional path
an actual, checkable condition: only a genuinely extreme statistical
deviation (default 2 standard deviations from a 20-day mean) counts as a
real setup, not any story the LLM comes up with.
"""
from datetime import date, datetime, timedelta
from typing import Optional

from data.equities import get_stock_bars
from live import strategy_config

Z_THRESHOLD = 2.0  # fallback default only - live value comes from strategy_config, see get_z_threshold()
WINDOW = 20
CHOP_THRESHOLD = 0.3  # Kaufman's Efficiency Ratio - below this is range-bound/choppy, at/above is trending; fallback default only


def get_z_threshold() -> float:
    return strategy_config.load()["z_threshold"]


def get_chop_threshold() -> float:
    return strategy_config.load()["chop_threshold"]


def _efficiency_ratio(prices: list[float]) -> float:
    """Kaufman's Efficiency Ratio: net directional movement over the
    window divided by the total absolute movement (sum of every day's
    move, in either direction). Bounded [0, 1] - close to 1 means the
    price moved efficiently in one direction (a real trend); close to 0
    means lots of back-and-forth with little net progress (choppy,
    range-bound). Mean reversion is a bet that price snaps back toward
    its average - that only tends to hold in the choppy regime. In a
    real trend, an "extreme" z-score is often just the trend continuing,
    not a reversion setup - the exact opposite read.
    """
    net_change = abs(prices[-1] - prices[0])
    total_movement = sum(abs(prices[i] - prices[i - 1]) for i in range(1, len(prices)))
    return net_change / total_movement if total_movement > 0 else 0.0


def mean_reversion_signal(underlying: str, as_of: Optional[date] = None, window: int = WINDOW) -> dict:
    """Args:
        underlying: Underlying ticker, e.g. AAPL.
        window: Trailing SMA/std-dev window in trading days (default 20).
    """
    as_of = as_of or date.today()
    bars = get_stock_bars(
        underlying,
        datetime.combine(as_of - timedelta(days=window * 2 + 10), datetime.min.time()),
        datetime.combine(as_of, datetime.min.time()),
    )
    closes = bars.loc[underlying]["close"]
    recent = closes.tail(window + 1)
    if len(recent) < window + 1:
        raise ValueError(f"Not enough price history to compute a {window}-day mean-reversion signal for {underlying}")

    # The window BEFORE today's close, so today's own price isn't part of
    # the baseline it's being measured against - otherwise an extreme move
    # partially drags its own mean/std toward itself, understating how
    # extreme it really is.
    trailing = recent.iloc[:-1]
    current_price = float(recent.iloc[-1])
    sma = float(trailing.mean())
    std = float(trailing.std())
    if std == 0:
        raise ValueError(f"Zero price variance in the trailing window for {underlying} - can't compute a z-score")

    z_threshold = get_z_threshold()
    chop_threshold = get_chop_threshold()

    z_score = (current_price - sma) / std
    if z_score >= z_threshold:
        direction = "overbought"  # extreme ABOVE its own mean - reversion thesis is DOWN
    elif z_score <= -z_threshold:
        direction = "oversold"  # extreme BELOW its own mean - reversion thesis is UP
    else:
        direction = "normal"

    efficiency_ratio = _efficiency_ratio(list(trailing))
    regime = "trending" if efficiency_ratio >= chop_threshold else "choppy"

    return {
        "underlying": underlying,
        "current_price": current_price,
        "sma": round(sma, 2),
        "std": round(std, 4),
        "z_score": round(z_score, 2),
        "z_threshold": z_threshold,
        "direction": direction,
        "is_extreme": abs(z_score) >= z_threshold,
        "efficiency_ratio": round(efficiency_ratio, 3),
        "chop_threshold": chop_threshold,
        "regime": regime,
        "favorable_for_reversion": regime == "choppy",
    }
