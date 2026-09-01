"""Account and position state, read through the Alpaca MCP server."""
from .mcp_client import AlpacaMCPClient, unwrap


def account_summary(client: AlpacaMCPClient) -> dict:
    return unwrap(client.call("get_account_info"))


def positions(client: AlpacaMCPClient) -> list[dict]:
    return unwrap(client.call("get_all_positions"))


def open_orders(client: AlpacaMCPClient) -> list[dict]:
    return unwrap(client.call("get_orders", status="open"))
