"""Black-Scholes pricing, Greeks, and implied-volatility inversion.

Alpaca has no historical implied volatility or Greeks - only OHLC bars, live
snapshot Greeks only. This recovers both from real historical option and
underlying prices via Black-Scholes, which is what makes an IV-aware
strategy (rather than a realized-vol proxy) testable against real history
at all.
"""
import math
from typing import Optional

from scipy.optimize import brentq
from scipy.stats import norm

DEFAULT_RISK_FREE_RATE = 0.04


def _d1_d2(S: float, K: float, T: float, r: float, sigma: float) -> tuple[float, float]:
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return d1, d1 - sigma * math.sqrt(T)


def price(S: float, K: float, T: float, sigma: float, option_type: str, r: float = DEFAULT_RISK_FREE_RATE) -> float:
    if T <= 0 or sigma <= 0:
        return max(0.0, S - K) if option_type == "call" else max(0.0, K - S)
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    if option_type == "call":
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def greeks(S: float, K: float, T: float, sigma: float, option_type: str, r: float = DEFAULT_RISK_FREE_RATE) -> dict:
    """Per-contract (1 share basis) delta/gamma/theta/vega. Multiply by 100 x
    quantity for a position's dollar exposure - the caller's job, not this
    function's, since that depends on whether it's long or short."""
    if T <= 0 or sigma <= 0:
        return {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0}
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    pdf_d1 = norm.pdf(d1)
    gamma = pdf_d1 / (S * sigma * math.sqrt(T))
    vega = S * pdf_d1 * math.sqrt(T) / 100  # per 1 vol point (e.g. 0.20 -> 0.21)
    if option_type == "call":
        delta = norm.cdf(d1)
        theta = (-S * pdf_d1 * sigma / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * norm.cdf(d2)) / 365
    else:
        delta = norm.cdf(d1) - 1
        theta = (-S * pdf_d1 * sigma / (2 * math.sqrt(T)) + r * K * math.exp(-r * T) * norm.cdf(-d2)) / 365
    return {"delta": delta, "gamma": gamma, "theta": theta, "vega": vega}


def implied_volatility(
    option_price: float, S: float, K: float, T: float, option_type: str, r: float = DEFAULT_RISK_FREE_RATE
) -> Optional[float]:
    """None if the price is at/below intrinsic value (no time value to
    invert) or if no volatility in [0.01%, 500%] reproduces it (stale/bad
    print) - both real, expected outcomes with historical data, not errors.
    """
    if T <= 0:
        return None
    intrinsic = max(0.0, S - K) if option_type == "call" else max(0.0, K - S)
    if option_price <= intrinsic + 1e-6:
        return None

    def objective(sigma: float) -> float:
        return price(S, K, T, sigma, option_type, r) - option_price

    try:
        return brentq(objective, 1e-4, 5.0, xtol=1e-6)
    except ValueError:
        return None
