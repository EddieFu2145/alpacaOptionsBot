"""Order placement through the Alpaca MCP server - the real execution path
for live/paper trading, as opposed to the backtest engine's simulated
`Portfolio`. Every call here is a real (paper, unless ALPACA_PAPER_TRADE=
false) order against the account - nothing here is a dry run.
"""
from typing import Optional

from .mcp_client import AlpacaMCPClient


def place_credit_spread(
    client: AlpacaMCPClient,
    short_symbol: str,
    long_symbol: str,
    contracts: int,
    limit_credit: float,
    client_order_id: Optional[str] = None,
) -> dict:
    """One atomic multi-leg order: sell `short_symbol`, buy `long_symbol`,
    for a net credit of `limit_credit` per spread (must be positive) - the
    same defined-risk structure `vol_credit_spread.py` builds in the
    backtest, placed as a single fill instead of two independent legs.
    """
    kwargs = dict(
        qty=str(contracts),
        type="limit",
        limit_price=str(-abs(limit_credit)),  # the tool's convention: negative limit_price = net credit
        legs=[
            {"symbol": short_symbol, "ratio_qty": "1", "side": "sell", "position_intent": "sell_to_open"},
            {"symbol": long_symbol, "ratio_qty": "1", "side": "buy", "position_intent": "buy_to_open"},
        ],
    )
    if client_order_id:
        kwargs["client_order_id"] = client_order_id
    return client.call("place_option_order", **kwargs)


def close_position(client: AlpacaMCPClient, symbol: str) -> dict:
    return client.call("close_position", symbol_or_asset_id=symbol)


def cancel_order(client: AlpacaMCPClient, order_id: str) -> dict:
    return client.call("cancel_order_by_id", order_id=order_id)
