"""Real-time position monitoring via Alpaca's option data websocket.

MCP itself has no push/streaming primitive - its tools are pull-based
request/response, so this deliberately steps outside the MCP layer and
uses alpaca-py's OptionDataStream directly. This is NOT an LLM call - it's
a cheap, always-on deterministic watcher using the same take-profit/
disproportionate-swing math as live/position_management.py, computed
continuously from live quotes instead of only when the agent happens to
poll.

The moment a logged position crosses a threshold, this automatically
invokes the LLM agent (no human step in between) with a prompt describing
what triggered it. The agent still decides what to do using its existing
tools, including close_position - a breach here means "go look at this
now", not a direct, unmediated close bypassing the agent entirely.

Re-subscribes dynamically: a background thread re-reads open_trades()
every RESUB_INTERVAL_SECONDS and subscribes to any new symbols. Confirmed
via alpaca-py's own source that this is safe - DataStream._subscribe
explicitly handles the "already running" case with
asyncio.run_coroutine_threadsafe, which is exactly what's needed to push
a live subscribe message onto the stream's event loop from a separate
thread while stream.run() blocks the main one. Previously this only
subscribed at startup, so a position opened after the watcher started got
zero real-time monitoring until a restart - partially masked by
run_trading_day.py's supervisor loop re-entering watch_positions() after
every research session, but with a real gap in between.
"""
import asyncio
import os
import threading
import time
from datetime import date
from typing import Optional

from dotenv import load_dotenv

from alpaca.data.live.option import OptionDataStream
from data.options import parse_occ_symbol
from live.activity_log import log_event
from live.trade_log import open_trades

load_dotenv()

TAKE_PROFIT_THRESHOLD = 0.65
PACE_RATIO_THRESHOLD = 3.0
TRIGGER_COOLDOWN_SECONDS = 30 * 60  # re-arm 30 min after a review, in case the agent held rather than closed
RESUB_INTERVAL_SECONDS = 60
STREAM_STALE_SECONDS = 180  # no quote at all in 3 minutes during market hours means the connection is wedged, not that the market is quiet

_last_mid: dict[str, float] = {}
_last_triggered_at: dict[str, float] = {}  # short_symbol -> monotonic time of last escalation
_last_quote_at: Optional[float] = None


def _mid_price(quote) -> Optional[float]:
    bid = quote.bid_price or 0.0
    ask = quote.ask_price or 0.0
    if bid and ask:
        return (bid + ask) / 2
    return bid or ask or None


async def _check_and_maybe_trigger(agent_module) -> None:
    today = date.today()
    for trade in open_trades():
        short_symbol, long_symbol = trade["short_symbol"], trade["long_symbol"]
        last_triggered = _last_triggered_at.get(short_symbol)
        if last_triggered is not None and time.monotonic() - last_triggered < TRIGGER_COOLDOWN_SECONDS:
            continue  # reviewed recently - avoid re-spawning an agent session every tick

        short_mid = _last_mid.get(short_symbol)
        long_mid = _last_mid.get(long_symbol)
        if short_mid is None or long_mid is None:
            continue  # haven't received a quote for both legs yet

        entry_credit = trade["entry_credit"]
        contracts = trade["contracts"]
        current_cost_to_close = short_mid - long_mid
        unrealized_pl_total = (entry_credit - current_cost_to_close) * 100 * contracts

        short_leg = parse_occ_symbol(short_symbol)
        long_leg = parse_occ_symbol(long_symbol)
        width = abs(short_leg["strike"] - long_leg["strike"])
        entry_credit_total = entry_credit * 100 * contracts
        max_loss_total = max(width * 100 - entry_credit * 100, 0.0) * contracts

        if unrealized_pl_total >= 0:
            pct_of_max_outcome = unrealized_pl_total / entry_credit_total if entry_credit_total else 0.0
        else:
            pct_of_max_outcome = unrealized_pl_total / max_loss_total if max_loss_total else 0.0

        entry_date = date.fromisoformat(trade["entry_date"])
        expiration = date.fromisoformat(trade["expiration"])
        days_held = max((today - entry_date).days, 0)
        planned_days = max((expiration - entry_date).days, 1)
        pct_of_life_elapsed = min(days_held / planned_days, 1.0)
        pace_ratio = abs(pct_of_max_outcome) / max(pct_of_life_elapsed, 0.05) if days_held >= 1 else None

        take_profit = pct_of_max_outcome >= TAKE_PROFIT_THRESHOLD
        disproportionate = pace_ratio is not None and pace_ratio >= PACE_RATIO_THRESHOLD
        if not (take_profit or disproportionate):
            continue

        _last_triggered_at[short_symbol] = time.monotonic()
        reason = "TAKE PROFIT" if take_profit else "DISPROPORTIONATE SWING"
        print(
            f"[TRIGGER] {reason} on {trade['underlying']} ({short_symbol}/{long_symbol}): "
            f"{pct_of_max_outcome:.1%} of max outcome after {days_held}/{planned_days} days held."
        )
        log_event(
            "trigger",
            underlying=trade["underlying"],
            reason=reason,
            pct_of_max_outcome=pct_of_max_outcome,
            days_held=days_held,
            planned_days=planned_days,
        )
        prompt = (
            f"A live monitor just flagged {reason} on your {trade['underlying']} position "
            f"(short {short_symbol}, long {long_symbol}, {contracts} contracts): "
            f"{pct_of_max_outcome:.1%} of max outcome captured after {days_held}/{planned_days} days held. "
            f"Call evaluate_positions to confirm current state, then decide whether to close this "
            f"position now."
        )
        asyncio.create_task(_run_agent_safely(agent_module, prompt))


async def _run_agent_safely(agent_module, prompt: str) -> None:
    try:
        result = await agent_module.run(prompt)
        print(f"[agent response] {result}")
    except Exception as exc:
        print(f"[agent error] {exc}")
        log_event("error", text=str(exc), source="watcher")


class _ThrottledOptionDataStream(OptionDataStream):
    """alpaca-py's own reconnect loop (DataStream._run_forever) has no
    backoff at all - on any exception it logs and immediately loops back
    around (`finally: await asyncio.sleep(0)`, which just yields, it
    doesn't pause). Confirmed live: a single connection hiccup turned into
    340,000+ rapid-fire reconnect attempts in under 4 minutes, pegging the
    process and very plausibly getting IP-throttled by Alpaca's own
    streaming endpoint on top of it - a self-inflicted retry storm, not a
    persistent outage (a standalone connection attempt succeeded
    immediately once the storm was killed). Since that loop's internals
    aren't ours to edit, throttling the one thing it always calls on every
    attempt - `_connect` - is the only hook available: a real sleep here
    before re-raising a failure means the outer loop can still spin as
    fast as it wants, but each cycle is now rate-limited regardless.
    """

    async def _connect(self) -> None:
        try:
            await super()._connect()
        except Exception:
            await asyncio.sleep(10)
            raise


def _watchdog_loop(stream: OptionDataStream, started_at: float) -> None:
    """alpaca-py's own reconnect loop (_run_forever) has no escape hatch when
    a connection gets stuck failing to (re)connect - confirmed live: a
    "connection limit exceeded" / DNS-resolution failure loop ran for 4.5+
    hours straight, with zero quotes ever getting through, silently leaving
    every open position completely unmonitored the whole time. The only
    known fix (confirmed by the same failure once before) is a full process
    restart - a fresh stream connects immediately once the old one is gone.
    Forcing stream.stop() after a stretch with no quotes at all breaks
    watch_positions() out of its blocking stream.run() call and back to
    run_trading_day.py's retry loop, which builds a brand new stream (and,
    if positions closed in the meantime, a fresh research session first) -
    a real chance to reconnect cleanly instead of staying wedged forever.
    """
    while True:
        time.sleep(30)
        last = _last_quote_at or started_at
        if time.monotonic() - last > STREAM_STALE_SECONDS:
            print(
                f"[watchdog] no quotes received in over {STREAM_STALE_SECONDS}s - "
                "the stream looks wedged, forcing a reconnect."
            )
            try:
                stream.stop()
            except Exception as exc:
                print(f"[watchdog] stream.stop() failed: {exc!r}")
            return


def _resubscribe_loop(stream: OptionDataStream, on_quote, known_symbols: set[str]) -> None:
    """Runs in a background thread for the life of the process. Best-effort:
    a symbol added between a check and the next one just gets picked up
    next cycle rather than needing any locking - not closing the gap
    entirely, but shrinking it from "until the next restart" to at most
    RESUB_INTERVAL_SECONDS.
    """
    while True:
        time.sleep(RESUB_INTERVAL_SECONDS)
        current: set[str] = set()
        for trade in open_trades():
            current.add(trade["short_symbol"])
            current.add(trade["long_symbol"])
        new_symbols = current - known_symbols
        if new_symbols:
            print(f"[re-subscribe] {len(new_symbols)} new symbol(s) found: {sorted(new_symbols)}")
            try:
                stream.subscribe_quotes(on_quote, *new_symbols)
                known_symbols.update(new_symbols)
            except Exception as exc:
                print(f"[re-subscribe] failed, will retry next cycle: {exc!r}")


def watch_positions(agent_module=None) -> None:
    """Blocking - runs the websocket's own event loop internally (alpaca-py's
    stream.run() calls asyncio.run itself). Ctrl+C to stop."""
    if agent_module is None:
        import live.agent_deepseek as agent_module

    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    stream = _ThrottledOptionDataStream(api_key, secret_key)

    symbols: set[str] = set()
    for trade in open_trades():
        symbols.add(trade["short_symbol"])
        symbols.add(trade["long_symbol"])

    if not symbols:
        print("No open positions to watch - nothing to subscribe to.")
        return

    global _last_quote_at
    _last_quote_at = None  # reset so a stale timestamp from a prior call can't trip the watchdog instantly

    async def on_quote(quote) -> None:
        global _last_quote_at
        _last_quote_at = time.monotonic()
        price = _mid_price(quote)
        if price is not None:
            _last_mid[quote.symbol] = price
        await _check_and_maybe_trigger(agent_module)

    stream.subscribe_quotes(on_quote, *symbols)
    print(f"Watching {len(symbols)} option legs across {len(open_trades())} logged position(s)...")

    threading.Thread(target=_resubscribe_loop, args=(stream, on_quote, symbols), daemon=True).start()
    threading.Thread(target=_watchdog_loop, args=(stream, time.monotonic()), daemon=True).start()
    stream.run()


if __name__ == "__main__":
    watch_positions()
