"""A synchronous Python client for Alpaca's official MCP server.

This is what makes MCP central rather than decorative: everything on the
live/execution side of the bot (account, positions, order placement, and
the live half of the AvgEA-Implied scanner) is meant to go through this
client and the Alpaca MCP server's tools, not alpaca-py's TradingClient
directly.

The historical backtest engine deliberately still uses alpaca-py directly.
The MCP server's data tools are themselves OpenAPI-generated wrappers
around the identical REST endpoints alpaca-py already calls - routing
hundreds of cached historical-bar lookups per backtest run through an
extra MCP round trip would add real latency for no different data. MCP
earns its place where the project is actually acting or reasoning live,
not where it's replaying history.
"""
import asyncio
import json
import os
import threading
from pathlib import Path
from typing import Any, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

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


class AlpacaMCPClient:
    """Keeps one alpaca-mcp-server subprocess + MCP session alive on a
    background thread, exposing plain synchronous `.call()` / `.list_tools()`
    so the rest of the codebase doesn't need to be async.
    """

    def __init__(self, server_path: Optional[str] = None, startup_timeout: float = 20.0):
        self._server_path = server_path or _default_server_path()
        if not Path(self._server_path).exists():
            raise FileNotFoundError(
                f"alpaca-mcp-server executable not found at '{self._server_path}'. "
                "Set ALPACA_MCP_SERVER_PATH if it's installed somewhere else."
            )

        self._loop = asyncio.new_event_loop()
        self._session: Optional[ClientSession] = None
        self._stop_event: Optional[asyncio.Event] = None
        self._startup_error: Optional[BaseException] = None
        self._ready = threading.Event()

        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=startup_timeout):
            raise TimeoutError("Timed out waiting for the Alpaca MCP server to start")
        if self._startup_error is not None:
            raise self._startup_error

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._main())
        except BaseException as exc:  # surfaced to the constructor via _startup_error
            self._startup_error = exc
            self._ready.set()

    async def _main(self) -> None:
        params = StdioServerParameters(command=self._server_path, args=["--env-file", str(_ENV_FILE)])
        self._stop_event = asyncio.Event()
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                self._session = session
                self._ready.set()
                await self._stop_event.wait()

    def list_tools(self) -> list[str]:
        future = asyncio.run_coroutine_threadsafe(self._session.list_tools(), self._loop)
        return [t.name for t in future.result(timeout=20).tools]

    def call(self, tool_name: str, timeout: float = 30.0, **arguments: Any) -> Any:
        future = asyncio.run_coroutine_threadsafe(self._session.call_tool(tool_name, arguments), self._loop)
        result = future.result(timeout=timeout)
        if result.isError:
            raise RuntimeError(f"MCP tool '{tool_name}' failed: {result.content}")

        texts = [block.text for block in result.content if hasattr(block, "text")]
        if len(texts) == 1:
            try:
                return json.loads(texts[0])
            except json.JSONDecodeError:
                return texts[0]
        return texts

    def close(self, timeout: float = 10.0) -> None:
        if self._stop_event is not None:
            self._loop.call_soon_threadsafe(self._stop_event.set)
        self._thread.join(timeout=timeout)

    def __enter__(self) -> "AlpacaMCPClient":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()
