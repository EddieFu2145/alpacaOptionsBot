"""Shared MCP-session helpers for order execution - collateral checks and
fill confirmation - used identically by both live/agent.py and
live/agent_gemini.py so this logic can't drift between the two providers.

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

from mcp import ClientSession

from live.mcp_client import unwrap


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


async def confirm_fill(
    mcp_session: ClientSession, order_id: str, max_attempts: int = 8, poll_seconds: float = 1.5
) -> dict:
    """Poll an order until it fills, partially fills, or terminates.
    Paper fills that are going to happen at all should resolve within a
    poll or two; an order still 'new'/'accepted' after the full budget is
    reported as pending, not assumed filled.
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

    return {"status": "pending", "order_id": order_id}
