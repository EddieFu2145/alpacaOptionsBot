"""Shared MCP-session helpers for order execution - collateral checks and
fill confirmation - used identically by both live/agent.py and
live/agent_deepseek.py so this logic can't drift between the two providers.

Fill confirmation matters because paper trading does NOT fill instantly on
submission: per Alpaca's own paper-engine behavior, a limit order only
fills once it's marketable (price touches your level), and does so
"generously" with no queue position or market impact the moment it is -
but an order that isn't marketable yet just sits as 'new'/'accepted'
indefinitely. Logging a trade as open immediately after submission, before
checking status, would log positions that were never actually filled.
"""
import asyncio
import json
from typing import Optional

from mcp import ClientSession

from live.mcp_client import unwrap
from live.trade_log import open_trades


async def _call_json(mcp_session: ClientSession, tool_name: str, args: dict) -> dict:
    result = await mcp_session.call_tool(tool_name, args)
    texts = [block.text for block in result.content if hasattr(block, "text")]
    if not texts:
        return {}
    payload = json.loads(texts[0])
    # unwrap() peels off however many "data"/"result" layers are actually
    # present - confirmed live that different Alpaca MCP tools nest
    # differently (some single-level {"data": {...}}, get_all_positions
    # double-nested {"data": {"result": [...]}}) and a one-shot
    # payload.get("data", payload.get("result", payload)) silently returns
    # the wrong (still-wrapped) object for the double-nested shape instead
    # of raising, which is exactly what broke position_management.py.
    return unwrap(payload)


async def live_options_buying_power(mcp_session: ClientSession) -> float:
    """Real, current collateral available for options trades - reflects
    whatever margin existing positions have already consumed, not a
    static number blind to them."""
    data = await _call_json(mcp_session, "get_account_info", {})
    return float(data["options_buying_power"])


async def live_account_equity(mcp_session: ClientSession) -> float:
    """Real, current account equity - the basis for a percentage-of-
    equity per-trade risk cap. Deliberately `equity`, not
    `options_buying_power`: buying power reflects available margin
    capacity (leverage-dependent, shrinks as positions are opened), not
    the account's actual value - a risk cap stated as "% of the account"
    should track the account, not how much room is left to trade."""
    data = await _call_json(mcp_session, "get_account_info", {})
    return float(data["equity"])


async def live_quotes(mcp_session: ClientSession, symbols: list[str]) -> dict[str, dict]:
    """Fresh (uncached) bid/ask for each symbol, fetched right now.

    Deliberately NOT data.options.get_option_chain, which caches per
    calendar day - reusing it here would silently price a new order off
    quotes from earlier in the session instead of the live book right
    before submission, defeating the entire point of pricing dynamically.
    """
    result = await mcp_session.call_tool("get_option_latest_quote", {"symbols": ",".join(symbols)})
    texts = [block.text for block in result.content if hasattr(block, "text")]
    payload = json.loads(texts[0]) if texts else {}
    quotes = unwrap(payload).get("quotes", {})
    out = {}
    for sym in symbols:
        q = quotes.get(sym) or {}
        bid, ask = float(q.get("bp") or 0), float(q.get("ap") or 0)
        if bid > 0 and ask > 0 and ask >= bid:
            out[sym] = {"bid": bid, "ask": ask, "mid": (bid + ask) / 2}
    return out


async def compute_marketable_credit(
    mcp_session: ClientSession, short_symbols: list[str], long_symbols: list[str]
) -> Optional[dict]:
    """A limit credit grounded in the live book right now, instead of a
    static number the model chose earlier in the session (possibly minutes
    ago) and hoped was still marketable.

    Confirmed live why this matters twice today: orders priced from stale
    research sat unfilled, and the fix each time was the model manually
    resubmitting at a lower, more marketable credit without cancelling the
    first - which is exactly what caused the HD double-fill. Pricing off
    fresh quotes at submission time is the other half of that fix: fewer
    orders miss marketable in the first place, so there's less reason to
    ever resubmit at all.

    guaranteed = crossing the full spread on every leg (sell at bid, buy at
    ask) - the worst realistic price, but always immediately marketable per
    Alpaca's own paper-fill behavior. fair_mid = the honest mid-price value,
    not guaranteed to be marketable. target sits halfway between them: a
    real, current, disciplined price that should fill promptly without
    giving away the full spread on every leg.
    """
    quotes = await live_quotes(mcp_session, short_symbols + long_symbols)
    if any(sym not in quotes for sym in short_symbols + long_symbols):
        return None  # a leg has no usable live quote right now - can't safely price this

    guaranteed = sum(quotes[s]["bid"] for s in short_symbols) - sum(quotes[s]["ask"] for s in long_symbols)
    fair_mid = sum(quotes[s]["mid"] for s in short_symbols) - sum(quotes[s]["mid"] for s in long_symbols)
    target = guaranteed + 0.5 * (fair_mid - guaranteed)
    return {"guaranteed": round(guaranteed, 2), "fair_mid": round(fair_mid, 2), "target": round(target, 2)}


async def confirm_fill(
    mcp_session: ClientSession, order_id: str, max_attempts: int = 40, poll_seconds: float = 3.0
) -> dict:
    """Poll an order until it fills, partially fills, or terminates - and if
    it's STILL not filled when the budget runs out, actively cancel it
    rather than leaving it live.

    Confirmed live TWICE now: the original budget (8 attempts x 1.5s = 12s)
    gave up on a day order that filled minutes later, and the first fix
    (20 x 2s = 40s) still wasn't enough for a real 4-leg iron condor that
    took 51s to fill on Alpaca's paper engine (submitted 16:00:42, filled
    16:01:33) - both times the fill then went completely untracked (no
    trade-log entry, zero exit monitoring) until caught by hand. 40 x 3s =
    120s gives real multi-leg fills comfortable headroom above that
    observed worst case.

    The cancel-on-timeout is the more important fix, added after a real
    double-fill: the agent, seeing "not yet filled", submitted a SECOND
    order for the same structure at a more marketable price instead of
    waiting - and with no cancel tool available to it, the FIRST order was
    left live. Both later filled, silently doubling the position (HD:
    intended 5 contracts, actual 10). Cancelling here closes the gap at the
    root - a tool call that returns "not filled" now GUARANTEES there is no
    live order left behind to duplicate, so a resubmission at a new price
    can never stack with a still-pending earlier one. The untracked-
    position reconciliation check in position_management.py stays in place
    as a second-layer backstop regardless.
    """
    for attempt in range(max_attempts):
        order = await _call_json(mcp_session, "get_order_by_id", {"order_id": order_id})
        status = order.get("status")

        if status == "filled":
            return {"status": "filled", "filled_qty": order.get("filled_qty"), "filled_avg_price": order.get("filled_avg_price")}
        if status in ("canceled", "rejected", "expired"):
            return {"status": status, "order": order}
        if status == "partially_filled" and attempt == max_attempts - 1:
            return {
                "status": "partially_filled",
                "filled_qty": order.get("filled_qty"),
                "filled_avg_price": order.get("filled_avg_price"),
            }

        await asyncio.sleep(poll_seconds)

    try:
        await mcp_session.call_tool("cancel_order_by_id", {"order_id": order_id})
    except Exception as exc:
        # Still report pending rather than filled/dead - a failed cancel
        # attempt (e.g. it filled in the instant between our last poll and
        # this call) means the order's true status is unknown, not that
        # it's safely gone. The reconciliation backstop covers this.
        return {"status": "pending", "order_id": order_id, "cancel_error": str(exc)}

    final = await _call_json(mcp_session, "get_order_by_id", {"order_id": order_id})
    final_status = final.get("status")
    if final_status == "filled":
        # Rare race: it filled in the gap between our last poll and the
        # cancel request landing - Alpaca can't cancel a filled order, and
        # that's a real fill, not a failure.
        return {"status": "filled", "filled_qty": final.get("filled_qty"), "filled_avg_price": final.get("filled_avg_price")}
    return {"status": "canceled_after_timeout", "order_id": order_id, "order": final}


async def close_both_legs(mcp_session: ClientSession, symbol: str) -> dict:
    """Close `symbol` and every other leg logged as part of the same
    structure - all in one call.

    Real bug this closes: trade_log.record_close marks a logged 2-leg
    spread closed the moment either symbol matches, but the actual Alpaca
    call only ever closed the one symbol it was given. An agent closing
    just the short leg (e.g. to take profit) would mark the whole trade
    "closed" in the log while the long leg stayed open on the real account
    - silently dropping a real position out of all further monitoring
    (position_stream's watcher and evaluate_positions both filter on
    open_trades(), which the log now falsely excludes it from). Closing
    every leg together whenever any one is requested keeps the log's
    "closed" state actually true.

    A 4-leg iron condor is logged as TWO linked 2-leg records (a put spread
    and a call spread opened together) sharing a `group_id` -
    trade_log.py's schema is deliberately kept at the proven, tested 2-leg
    shape rather than reworked for a variable leg count. When the trade
    containing `symbol` has a group_id, every other record sharing that
    same id is pulled in too, closing all 4 real legs together instead of
    just the 2 from whichever record `symbol` happens to be in. A plain
    2-leg spread (group_id is None - true for every record logged before
    this existed, and any single spread opened since) behaves exactly as
    before: just its own 2 legs.
    """
    trades = open_trades()
    targets = {symbol}
    involved_trades = []
    owning_trade = next((t for t in trades if symbol in (t["short_symbol"], t["long_symbol"])), None)
    if owning_trade is not None:
        targets.add(owning_trade["short_symbol"])
        targets.add(owning_trade["long_symbol"])
        involved_trades.append(owning_trade)
        group_id = owning_trade.get("group_id")
        if group_id is not None:
            for trade in trades:
                if trade.get("group_id") == group_id and trade is not owning_trade:
                    targets.add(trade["short_symbol"])
                    targets.add(trade["long_symbol"])
                    involved_trades.append(trade)

    results: dict[str, list] = {}
    errors: dict[str, str] = {}
    exit_fills: dict[str, float] = {}
    for sym in targets:
        try:
            result = await mcp_session.call_tool("close_position", {"symbol_or_asset_id": sym})
            texts = [block.text for block in result.content if hasattr(block, "text")]
            results[sym] = texts
            order = unwrap(json.loads(texts[0])) if texts else {}
            order_id = order.get("id") if isinstance(order, dict) else None
            if order_id:
                price = await _poll_close_fill_price(mcp_session, order_id)
                if price is not None:
                    exit_fills[sym] = price
        except Exception as exc:
            # Don't let one leg's failure (e.g. already closed manually)
            # stop the other legs from being closed - report it instead.
            errors[sym] = str(exc)

    # Realized P&L per logged record, best-effort: only computed when BOTH
    # of that record's legs got a confirmed exit fill above. Keyed under
    # both the short and long symbol so record_close (called once per
    # symbol in closed_symbols, in arbitrary order) finds it regardless of
    # which leg it's called with first.
    realized_pnl_by_symbol: dict[str, float] = {}
    for t in involved_trades:
        short_fill, long_fill = exit_fills.get(t["short_symbol"]), exit_fills.get(t["long_symbol"])
        if short_fill is not None and long_fill is not None:
            per_spread = t["entry_credit"] - (short_fill - long_fill)
            pnl = round(per_spread * 100 * t["contracts"], 2)
            realized_pnl_by_symbol[t["short_symbol"]] = pnl
            realized_pnl_by_symbol[t["long_symbol"]] = pnl

    return {
        "closed_symbols": sorted(targets),
        "results": results,
        "errors": errors,
        "realized_pnl_by_symbol": realized_pnl_by_symbol,
    }


async def _poll_close_fill_price(
    mcp_session: ClientSession, order_id: str, max_attempts: int = 5, poll_seconds: float = 2.0
) -> Optional[float]:
    """Best-effort fill-price lookup for a just-submitted close order, purely
    for realized-P&L display. Deliberately NOT confirm_fill: that function
    cancels the order if it times out, which is the right safety behavior
    for a NEW position (never leave a stray order that could double-fill
    later) but the wrong one here - cancelling a close order would leave a
    position the agent just asked to exit sitting open. If this order
    hasn't filled within the short budget below, it's left alone and this
    just returns None (the caller skips realized-P&L for that leg rather
    than guessing).
    """
    for _ in range(max_attempts):
        order = await _call_json(mcp_session, "get_order_by_id", {"order_id": order_id})
        if order.get("status") == "filled" and order.get("filled_avg_price") is not None:
            return float(order["filled_avg_price"])
        await asyncio.sleep(poll_seconds)
    return None


def resolved_credit(fill: dict, limit_credit: float) -> float:
    """The real per-spread credit to log, preferring the confirmed fill
    price over the originally requested limit.

    Confirmed live: a marketable limit order can fill with price
    improvement (JPM was requested at a $0.60 limit, filled at $0.62) - and
    the trade log used to record the requested limit unconditionally,
    understating the real credit captured and slightly skewing every
    downstream take-profit/pace calculation for that trade. Alpaca's
    multi-leg order `filled_avg_price` is the net price across both legs
    (negative for a credit), so this only needs abs() to match
    `entry_credit`'s dollars-per-share-of-credit convention.
    """
    price = fill.get("filled_avg_price")
    if price is None:
        return limit_credit
    try:
        return abs(float(price))
    except (TypeError, ValueError):
        return limit_credit
