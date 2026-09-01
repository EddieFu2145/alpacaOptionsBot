"""Execution-cost simulation.

Alpaca's historical option bars are OHLC only - no historical bid/ask - so
there's no direct historical spread to cross. `apply_slippage` approximates
the cost of actually crossing a spread by haircutting the fill price against
the trade direction, which is a simplification, not observed market data.
"""


def apply_slippage(quote_price: float, quantity: float, slippage_bps: float = 5.0) -> float:
    haircut = slippage_bps / 10_000
    return quote_price * (1 - haircut) if quantity < 0 else quote_price * (1 + haircut)
