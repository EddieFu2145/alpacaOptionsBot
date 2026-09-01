"""Exit-worthiness evaluation for open positions: take profit at a large
% of max available gain, or exit defensively if the P&L swing so far is
disproportionate to how long the position has actually been held (moving
much faster than a normal decay pace implies something unusual happened,
in either direction).

Shared between live/agent.py and live/agent_gemini.py - both providers
get identical exit math, not two copies that can drift apart.
"""
from datetime import date

from mcp import ClientSession

from data.options import parse_occ_symbol
from live.mcp_client import unwrap
from live.trade_log import open_trades

TAKE_PROFIT_THRESHOLD = 0.65  # close once this much of max profit is captured
PACE_RATIO_THRESHOLD = 3.0  # swing this many times faster than a linear pace implies


async def _mcp_positions_by_symbol(mcp_session: ClientSession) -> dict[str, dict]:
    import json

    result = await mcp_session.call_tool("get_all_positions", {})
    texts = [block.text for block in result.content if hasattr(block, "text")]
    payload = json.loads(texts[0]) if texts else {}

    # Confirmed live: get_all_positions' real envelope is the doubly-nested
    # {"data": {"result": [...]}} - NOT the single-level {"data": [...]}
    # some other Alpaca MCP tools use. A one-shot
    # payload.get("data", payload.get("result", payload)) grabbed
    # payload["data"] (that key exists) and stopped there, landing on
    # {"result": [...]} - a dict, not a list - which silently reset to []
    # below. That meant this function returned "no positions" for every
    # real open position since it was written, making the whole
    # take-profit/defensive-exit check a permanent no-op. unwrap() peels
    # off as many "data"/"result" layers as are actually present instead
    # of assuming a fixed depth.
    positions = unwrap(payload)
    if not isinstance(positions, list):
        positions = []
    return {p["symbol"]: p for p in positions}


async def evaluate_open_positions(mcp_session: ClientSession) -> list[dict]:
    # Deliberately fetches live positions unconditionally, even if `logged`
    # is empty - confirmed live that a real fill can land on the account
    # with nothing in the trade log at all (a day order that filled after
    # confirm_fill's polling window gave up and moved on without ever
    # calling record_open). An early `if not logged: return []` here would
    # mean an entirely untracked position never gets flagged, precisely
    # the case that actually happened and left a real open position with
    # zero exit monitoring for over two hours before it was caught by hand.
    logged = open_trades()
    live_positions = await _mcp_positions_by_symbol(mcp_session)
    today = date.today()
    reports = []

    logged_symbols = {s for t in logged for s in (t["short_symbol"], t["long_symbol"])}
    for symbol, pos in live_positions.items():
        if symbol not in logged_symbols and pos.get("asset_class") == "us_option":
            try:
                underlying = parse_occ_symbol(symbol)["underlying"]
            except ValueError:
                underlying = symbol
            reports.append(
                {
                    "underlying": underlying,
                    "symbol": symbol,
                    "unrealized_pl": pos.get("unrealized_pl"),
                    "recommendation": (
                        "UNTRACKED POSITION - this option leg is live on the account but isn't in the "
                        "trade log (likely a fill that landed after order-submission polling gave up). "
                        "It's getting NO automated take-profit/defensive-exit monitoring. Investigate "
                        "immediately: check get_order_by_id history to find the real entry credit and "
                        "expiration, then treat it as a real position when deciding whether to close it."
                    ),
                }
            )

    for trade in logged:
        short_pos = live_positions.get(trade["short_symbol"])
        long_pos = live_positions.get(trade["long_symbol"])
        if short_pos is None and long_pos is None:
            continue  # already closed or expired/settled outside this tool - nothing to evaluate

        combined_unrealized_pl = float(short_pos.get("unrealized_pl", 0.0) if short_pos else 0.0) + float(
            long_pos.get("unrealized_pl", 0.0) if long_pos else 0.0
        )

        entry_date = date.fromisoformat(trade["entry_date"])
        expiration = date.fromisoformat(trade["expiration"])
        days_held = max((today - entry_date).days, 0)
        planned_days = max((expiration - entry_date).days, 1)
        pct_of_life_elapsed = min(days_held / planned_days, 1.0)

        entry_credit_total = trade["entry_credit"] * 100 * trade["contracts"]
        short_leg = parse_occ_symbol(trade["short_symbol"])
        long_leg = parse_occ_symbol(trade["long_symbol"])
        width = abs(short_leg["strike"] - long_leg["strike"])
        max_loss_total = max(width * 100 - trade["entry_credit"] * 100, 0.0) * trade["contracts"]

        if combined_unrealized_pl >= 0:
            pct_of_max_outcome = combined_unrealized_pl / entry_credit_total if entry_credit_total else 0.0
        else:
            pct_of_max_outcome = combined_unrealized_pl / max_loss_total if max_loss_total else 0.0

        pace_ratio = abs(pct_of_max_outcome) / max(pct_of_life_elapsed, 0.05) if days_held >= 1 else None

        take_profit = pct_of_max_outcome >= TAKE_PROFIT_THRESHOLD
        disproportionate_swing = pace_ratio is not None and pace_ratio >= PACE_RATIO_THRESHOLD

        if take_profit:
            recommendation = "TAKE PROFIT - a large share of max profit is already captured; holding for the remainder risks giving it back for little extra gain."
        elif disproportionate_swing and pct_of_max_outcome < 0:
            recommendation = "DEFENSIVE EXIT - loss is moving much faster than the time held would predict; consider cutting it rather than waiting it out."
        elif disproportionate_swing:
            recommendation = "CONSIDER CLOSING - profit is moving unusually fast for how little time has passed; may be worth locking in rather than assuming the pace continues."
        else:
            recommendation = "HOLD - within a normal pace for time held."

        reports.append(
            {
                "underlying": trade["underlying"],
                "short_symbol": trade["short_symbol"],
                "long_symbol": trade["long_symbol"],
                "days_held": days_held,
                "planned_days": planned_days,
                "pct_of_life_elapsed": round(pct_of_life_elapsed, 3),
                "unrealized_pl": round(combined_unrealized_pl, 2),
                "pct_of_max_outcome": round(pct_of_max_outcome, 3),
                "pace_ratio": round(pace_ratio, 2) if pace_ratio is not None else None,
                "take_profit": take_profit,
                "disproportionate_swing": disproportionate_swing,
                "recommendation": recommendation,
            }
        )

    return reports
