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

Known limitation: subscribes to symbols from currently-open logged
positions at startup only. If the agent opens a new position while this
is already running, restart the watcher to pick it up - there's no
dynamic re-subscription.
"""
import asyncio
import os
import time
from datetime import date
from typing import Optional

from dotenv import load_dotenv

from alpaca.data.live.option import OptionDataStream
from data.options import parse_occ_symbol
from live.trade_log import open_trades

load_dotenv()

TAKE_PROFIT_THRESHOLD = 0.65
PACE_RATIO_THRESHOLD = 3.0
TRIGGER_COOLDOWN_SECONDS = 30 * 60  # re-arm 30 min after a review, in case the agent held rather than closed

_last_mid: dict[str, float] = {}
_last_triggered_at: dict[str, float] = {}  # short_symbol -> monotonic time of last escalation


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

    async def on_quote(quote) -> None:
        price = _mid_price(quote)
        if price is not None:
            _last_mid[quote.symbol] = price
        await _check_and_maybe_trigger(agent_module)

    stream.subscribe_quotes(on_quote, *symbols)
    print(f"Watching {len(symbols)} option legs across {len(open_trades())} logged position(s)...")
    stream.run()


if __name__ == "__main__":
    watch_positions()
