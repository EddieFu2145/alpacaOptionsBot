"""Pre-market move detection: checks whether an underlying has moved
materially since Friday's close using whatever quote is actually
available, ahead of the options market opening and its own quotes
catching up.

Real, confirmed limitation: this account's data subscription is IEX-only -
SIP access is explicitly rejected ("subscription does not permit querying
recent SIP data", confirmed live). IEX has materially less pre-market
participation than the consolidated tape: checked directly, as late as
~04:50 ET on a trading day, AAPL's "latest trade" was still Friday's 16:04
close, not a real pre-market print. A "stale" result here means "nothing
seen on IEX", not "nothing happened" - treat it as a genuine data gap, not
a green light.
"""
from datetime import datetime, timedelta, timezone
from typing import Optional

from alpaca.data.requests import StockLatestTradeRequest

from data.clients import stock_client
from data.equities import get_stock_bars

STALE_THRESHOLD_MINUTES = 30


def premarket_move(symbol: str) -> dict:
    """Compares the latest available trade's timestamp against now. If
    it's fresh, computes the % move from the prior session's close. If
    it's stale (no real pre-market print reached IEX), says so explicitly
    rather than reporting a false "no move".
    """
    trade = stock_client().get_stock_latest_trade(StockLatestTradeRequest(symbol_or_symbols=symbol))[symbol]
    age_minutes = (datetime.now(timezone.utc) - trade.timestamp).total_seconds() / 60

    if age_minutes > STALE_THRESHOLD_MINUTES:
        return {
            "symbol": symbol,
            "fresh": False,
            "age_minutes": round(age_minutes, 1),
            "note": "No recent print on IEX (our data feed) - can't assess pre-market move. This does not mean nothing happened.",
        }

    bars = get_stock_bars(symbol, trade.timestamp - timedelta(days=5), trade.timestamp)
    closes = bars.loc[symbol]["close"] if symbol in bars.index.get_level_values(0) else None
    if closes is None or len(closes) < 2:
        return {"symbol": symbol, "fresh": True, "spot": trade.price, "note": "couldn't find a prior close to compare against"}

    prior_close = float(closes.iloc[-2])
    move_pct = (trade.price - prior_close) / prior_close
    return {
        "symbol": symbol,
        "fresh": True,
        "spot": trade.price,
        "prior_close": prior_close,
        "move_pct": round(move_pct, 4),
        "age_minutes": round(age_minutes, 1),
    }


def premarket_briefing(symbols: list[str]) -> list[dict]:
    """premarket_move for a batch of symbols, sorted so anything showing a
    real fresh move surfaces first - meant to run once, right before
    market open, ahead of the normal research flow."""
    results = [premarket_move(s) for s in symbols]
    return sorted(results, key=lambda r: (not r["fresh"], -abs(r.get("move_pct", 0.0))))
