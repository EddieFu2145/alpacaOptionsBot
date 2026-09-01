"""Live market data through the Alpaca MCP server - the MCP-native
counterpart to `data/` (which stays on direct alpaca-py for historical
backtesting). Used by the live half of the AvgEA-Implied scanner and by
the execution layer for pre-trade price checks.
"""
from datetime import date
from typing import Optional

from .mcp_client import AlpacaMCPClient, unwrap


def latest_spot(client: AlpacaMCPClient, symbol: str) -> float:
    data = unwrap(client.call("get_stock_latest_trade", symbols=symbol))
    return float(data["trades"][symbol]["p"])


def option_chain(
    client: AlpacaMCPClient,
    underlying_symbol: str,
    expiration_date: Optional[date] = None,
    strike_price_gte: Optional[float] = None,
    strike_price_lte: Optional[float] = None,
) -> dict:
    kwargs = {"underlying_symbol": underlying_symbol}
    if expiration_date:
        kwargs["expiration_date"] = expiration_date.isoformat()
    if strike_price_gte is not None:
        kwargs["strike_price_gte"] = strike_price_gte
    if strike_price_lte is not None:
        kwargs["strike_price_lte"] = strike_price_lte
    return unwrap(client.call("get_option_chain", **kwargs))
