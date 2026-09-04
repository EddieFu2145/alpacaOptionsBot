"""Shared helpers for talking to Alpaca's official MCP server.

This is what makes MCP central rather than decorative: everything on the
live/execution side of the bot (account, positions, order placement) goes
through the Alpaca MCP server's tools, not alpaca-py's TradingClient
directly - agent.py and agent_deepseek.py each open their own async
`ClientSession`/`stdio_client` (one per research session) and use
`_default_server_path`/`unwrap` from here rather than a shared client
object, since MCP tool calls only ever happen from within that one async
session's lifetime.

The historical backtest engine deliberately still uses alpaca-py directly.
The MCP server's data tools are themselves OpenAPI-generated wrappers
around the identical REST endpoints alpaca-py already calls - routing
hundreds of cached historical-bar lookups per backtest run through an
extra MCP round trip would add real latency for no different data. MCP
earns its place where the project is actually acting or reasoning live,
not where it's replaying history.
"""
import os
from pathlib import Path
from typing import Any

_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


def _default_server_path() -> str:
    override = os.environ.get("ALPACA_MCP_SERVER_PATH")
    if override:
        return override
    appdata = os.environ.get("APPDATA", "")
    return str(Path(appdata) / "Python" / "Python314" / "Scripts" / "alpaca-mcp-server.exe")


def unwrap(response: Any) -> Any:
    """Every tool wraps its payload differently, sometimes in more than one
    layer ("data" containing "result", or vice versa), alongside an
    `_alpaca_mcp_security` advisory tag - this strips all of it down to the
    actual API payload."""
    while isinstance(response, dict):
        keys = set(response.keys()) - {"_alpaca_mcp_security"}
        if keys in ({"data"}, {"result"}):
            response = response[next(iter(keys))]
        else:
            break
    return response
