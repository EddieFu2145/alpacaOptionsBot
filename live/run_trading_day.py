"""Entry point for autonomous end-to-end operation: waits for market open,
runs one agent session to research and possibly open a trade, then hands
off to the websocket watcher for continuous exit monitoring for the rest
of the session.

Uses alpaca-py directly (not the MCP client) for the open-market wait loop
- AlpacaMCPClient launches a fresh alpaca-mcp-server subprocess per
instantiation, which is fine for one-shot tool calls but far too heavy to
poll every minute for hours. The MCP-based agent is only invoked once the
market is actually open.
"""
import asyncio
import sys
import time
from datetime import datetime, timezone

# Confirmed live crash: a DeepSeek response containing a plain arrow
# character (U+2192) crashed print() outright, because this process's
# stdout defaults to Windows' cp1252 codec, which can't encode it - and
# since that happened *inside* the retry loop's own error-logging print(),
# it took down the whole process instead of just skipping one message.
# Forcing UTF-8 here (errors="replace" as a last-resort safety net) means
# an odd character in a response never gets to crash the process again.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import live.agent_deepseek as agent
from data.clients import trading_client
from data.screener import DEFAULT_LARGE_CAP_UNIVERSE
from live.position_stream import watch_positions
from live.premarket_check import premarket_briefing
from live.trade_log import open_trades

POLL_SECONDS = 60
SESSION_RETRY_SECONDS = 90  # backoff between initial-session attempts during an LLM API outage
RECHECK_SECONDS = 300  # how often to re-run the research session if it decided not to trade


def _wait_for_market_open() -> None:
    while True:
        now = datetime.now(timezone.utc).isoformat()
        try:
            clock = trading_client().get_clock()
        except Exception as exc:
            print(f"[{now}] Clock check failed ({exc!r}), retrying in {POLL_SECONDS}s.")
            time.sleep(POLL_SECONDS)
            continue
        if clock.is_open:
            print(f"[{now}] Market is open.")
            return
        print(f"[{now}] Market closed. Next open: {clock.next_open.isoformat()}. Checking again in {POLL_SECONDS}s.")
        time.sleep(POLL_SECONDS)


def _market_is_open() -> bool:
    try:
        return bool(trading_client().get_clock().is_open)
    except Exception as exc:
        print(f"[{datetime.now(timezone.utc).isoformat()}] Clock check failed ({exc!r}); assuming still open.")
        return True


def _run_research_session() -> None:
    watched_symbols = sorted({t["underlying"] for t in open_trades()} | set(DEFAULT_LARGE_CAP_UNIVERSE))
    print(f"Checking pre-market moves for {len(watched_symbols)} symbols before the session...")
    briefing = premarket_briefing(watched_symbols)
    fresh_moves = [r for r in briefing if r["fresh"]]
    if fresh_moves:
        print(f"[pre-market] {len(fresh_moves)} fresh reading(s): {fresh_moves}")
        briefing_text = (
            f"Pre-market check found {len(fresh_moves)} name(s) with a fresh reading: {fresh_moves}. "
            "The rest showed no recent print on our data feed (IEX only, no SIP) - that means no data, "
            "not necessarily no move, so don't treat silence on a name as confirmation it's quiet."
        )
    else:
        print("[pre-market] No fresh pre-market prints on IEX for any watched symbol.")
        briefing_text = (
            "Pre-market check found no fresh prints on our data feed (IEX only, no SIP) for any watched "
            "symbol - that means no data was available, not that nothing happened overnight."
        )

    print("Running agent session...")
    message = (
        f"Market is open. {briefing_text} Check existing positions, screen for candidates, "
        "and decide whether to open a new trade this week. Reminder: this is the first live day "
        "for this pipeline - size any new trade at 1 contract."
    )
    # Indefinite retry, not capped: a Gemini API outage (confirmed to happen -
    # sustained 503 "high demand" errors observed for 10+ minutes on this
    # exact first live day) must not be allowed to end the research session
    # for the entire rest of the trading day. Every attempt is cheap (a
    # single API call); there's no real downside to patience here.
    result = _run_with_retries(
        "agent session", lambda: asyncio.run(agent.run(message)), max_attempts=None, backoff_seconds=SESSION_RETRY_SECONDS
    )
    print(f"[agent session result]\n{result}")


def main() -> None:
    _wait_for_market_open()

    while _market_is_open():
        _run_research_session()

        if open_trades():
            print("Starting continuous position monitoring (auto-triggers the agent on a threshold breach)...")
            _run_with_retries("position monitoring", lambda: watch_positions(agent_module=agent), max_attempts=None)
            # watch_positions() only returns if every watched position closed
            # (or it errored out of retries) - loop back to re-run research
            # rather than ending the process for the day.
        else:
            print(f"No open positions after this session. Re-checking again in {RECHECK_SECONDS}s.")
            time.sleep(RECHECK_SECONDS)

    print(f"[{datetime.now(timezone.utc).isoformat()}] Market closed. Ending for the day.")


def _run_with_retries(label: str, fn, max_attempts: int | None = 5, backoff_seconds: float = 30.0):
    """Run `fn` and retry on any exception instead of letting a transient
    failure (a 503 from the LLM API, a dropped websocket, a network blip -
    all confirmed to happen in practice on the first real day of running
    this unattended) take the whole process down for the rest of the
    session. `max_attempts=None` retries forever, for the long-running
    position-monitoring loop which is meant to survive indefinitely.
    """
    attempt = 0
    while True:
        attempt += 1
        try:
            return fn()
        except Exception as exc:
            now = datetime.now(timezone.utc).isoformat()
            print(f"[{now}] [{label}] attempt {attempt} failed: {exc!r}")
            if max_attempts is not None and attempt >= max_attempts:
                print(f"[{now}] [{label}] giving up after {attempt} attempts.")
                return None
            time.sleep(backoff_seconds)


if __name__ == "__main__":
    main()
