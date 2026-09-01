"""AvgEA-Implied: the earnings-announcement signal from Milian (2023),
"The Efficiency of Weekly Option Prices around Earnings Announcements"
(J. Risk Financial Manag. 16(5):270).

AvgEA = a firm's average |1-day return| on its last 4 earnings dates.
Implied = ATM straddle price / spot, on the option expiring in the
          earnings week - sampled live, since we only ever care about
          companies reporting THIS week. No historical options data is
          needed for this signal at all, which sidesteps Alpaca's
          ~January 2024 options-history floor entirely.
AvgEA-Implied = AvgEA - Implied. Positive means the market's current
          pricing looks cheap relative to this firm's own earnings-day
          history; negative means it looks rich.

This is a live/forward screening tool, not a historical backtest - the
paper's own effect size is small (quarterly hedge-portfolio t~2.7, raw
return correlation of just 0.05), so treat any result here as a candidate
worth risk-gating, not a proven edge.
"""
from datetime import date, datetime, timedelta
from typing import Literal, Optional

from data.equities import get_stock_bars
from data.options import get_option_chain
from data.clients import stock_client
from alpaca.data.requests import StockLatestTradeRequest

Timing = Literal["bmo", "amc"]


def _close_on_or_before(closes: dict, day: date) -> Optional[float]:
    for offset in range(5):  # walk back over a weekend/holiday
        candidate = day - timedelta(days=offset)
        if candidate in closes:
            return closes[candidate]
    return None


def _close_on_or_after(closes: dict, day: date) -> Optional[float]:
    for offset in range(5):
        candidate = day + timedelta(days=offset)
        if candidate in closes:
            return closes[candidate]
    return None


def _daily_closes(symbol: str, start: date, end: date) -> dict:
    bars = get_stock_bars(symbol, datetime.combine(start, datetime.min.time()), datetime.combine(end, datetime.min.time()))
    if bars.empty:
        return {}
    return {ts.date(): close for ts, close in bars.loc[symbol]["close"].items()}


def earnings_day_move(symbol: str, earnings_date: date, timing: Timing) -> Optional[float]:
    """Absolute 1-day return capturing the market's reaction to one past
    earnings announcement. BMO (before market open) reacts same-day;
    AMC (after market close) reacts the next trading day.
    """
    closes = _daily_closes(symbol, earnings_date - timedelta(days=7), earnings_date + timedelta(days=7))
    if not closes:
        return None

    if timing == "bmo":
        pre = _close_on_or_before(closes, earnings_date - timedelta(days=1))
        post = _close_on_or_after(closes, earnings_date)
    else:
        pre = _close_on_or_before(closes, earnings_date)
        post = _close_on_or_after(closes, earnings_date + timedelta(days=1))

    if pre is None or post is None or pre == 0:
        return None
    return abs(post - pre) / pre


def avg_earnings_move(symbol: str, past_earnings: list[tuple[date, Timing]]) -> Optional[float]:
    """AvgEA: mean absolute earnings-day move over up to the last 4 dates
    supplied. Caller is responsible for sourcing those dates (an external
    earnings calendar - not something Alpaca provides)."""
    moves = [m for d, t in past_earnings[-4:] if (m := earnings_day_move(symbol, d, t)) is not None]
    if not moves:
        return None
    return sum(moves) / len(moves)


def latest_spot(symbol: str) -> float:
    trade = stock_client().get_stock_latest_trade(StockLatestTradeRequest(symbol_or_symbols=symbol))
    return trade[symbol].price


def implied_move(symbol: str, expiration: date, atm_tolerance_pct: float = 0.05) -> Optional[float]:
    """Implied: today's ATM straddle price / spot, for the option expiring
    on `expiration`. Live data only - this is what makes the signal usable
    without any historical options depth. Matches the paper's ATM
    definition: a strike within 5% of spot, same strike for both legs.
    """
    spot = latest_spot(symbol)
    chain = get_option_chain(
        symbol,
        expiration_date=expiration,
        strike_price_gte=spot * (1 - atm_tolerance_pct),
        strike_price_lte=spot * (1 + atm_tolerance_pct),
    )
    if chain.empty or "last_trade_price" not in chain.columns:
        return None

    calls = chain[(chain["option_type"] == "call") & chain["last_trade_price"].notna()]
    puts = chain[(chain["option_type"] == "put") & chain["last_trade_price"].notna()]
    shared_strikes = set(calls["strike"]) & set(puts["strike"])
    if not shared_strikes:
        return None

    atm_strike = min(shared_strikes, key=lambda k: abs(k - spot))
    call_price = calls[calls["strike"] == atm_strike].iloc[0]["last_trade_price"]
    put_price = puts[puts["strike"] == atm_strike].iloc[0]["last_trade_price"]
    return (call_price + put_price) / spot


def avgea_implied(symbol: str, expiration: date, past_earnings: list[tuple[date, Timing]]) -> Optional[dict]:
    avg_ea = avg_earnings_move(symbol, past_earnings)
    implied = implied_move(symbol, expiration)
    if avg_ea is None or implied is None:
        return None
    return {"symbol": symbol, "avg_ea": avg_ea, "implied": implied, "avgea_implied": avg_ea - implied}
