"""DeepSeek port of the "AI assistant talks to Alpaca" agent - same
business logic and safety architecture as live/agent_gemini.py and
live/agent.py, rebuilt on DeepSeek's OpenAI-compatible chat completions API.

Built the night all three Gemini free-tier fallback models (3.7-flash,
3.6-flash, 2.5-flash) hit their 20-requests/day cap in under an hour of
real use - DeepSeek doesn't share that capacity pool, so this is a genuine
third provider, not another fallback rung on the same ladder.

Kept as its own file rather than sharing code with the other two agent
modules - same reasoning as their split: each provider implementation is
independently auditable for exactly what tools it exposes and withholds,
without a shared abstraction hiding a provider-specific gap.

DeepSeek uses OpenAI-style function calling (tools=[{"type":"function",
"function":{...}}], tool_calls on the response message, role="tool"
messages to answer them - NOT the "user"-role-with-function-response shape
Gemini uses). MCP tool schemas convert directly since MCP's inputSchema is
already plain JSON Schema - no import to strip like Gemini's
`additionalProperties` issue, but sanitized the same way anyway for
consistency and because it's a no-op if unneeded.
"""
import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import AsyncOpenAI

from data.earnings_calendar import earnings_before_expiration
from data.options import parse_occ_symbol
from data.screener import candidate_universe, rank_by_vol_signal
from live.mcp_client import _default_server_path
from live.order_helpers import confirm_fill, live_options_buying_power
from live.position_management import evaluate_open_positions
from live.premarket_check import premarket_briefing
from live.trade_log import record_close, record_open
from signals.options_quality import spread_quality_report

load_dotenv()

_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
_BASE_URL = "https://api.deepseek.com"

# Same set as live/agent_gemini.py / live/agent.py - kept in sync manually,
# not shared, for the same "independently auditable" reason as those two.
_MUTATING_TOOLS = {
    "place_stock_order",
    "place_option_order",
    "place_crypto_order",
    "close_position",
    "close_all_positions",
    "cancel_order_by_id",
    "cancel_all_orders",
    "replace_order_by_id",
    "exercise_options_position",
    "do_not_exercise_options_position",
    "create_locate",
    "create_watchlist",
    "delete_watchlist_by_id",
    "update_watchlist_by_id",
    "add_asset_to_watchlist_by_id",
    "remove_asset_from_watchlist_by_id",
    "update_account_config",
}

MAX_SPREAD_MARGIN = 2_000.0  # per-trade cap - deliberately tight for the first live day; raise once the pipeline has a real track record
MAX_LOOP_TURNS = 12  # hard cap against a runaway tool-calling loop

# This is a real paid API (unlike the free-tier Gemini models) - a single
# research session was observed making 7+ calls in under a minute, and the
# supervisor loop in run_trading_day.py re-runs sessions all day. A $5
# balance needs to survive a week, not an afternoon, so calls are throttled
# rather than left to fire as fast as the loop can go. Persisted to disk
# (not just an in-process deque) since the window needs to hold across the
# supervisor loop's periodic process relaunches, not just within one.
RATE_LIMIT_PER_MINUTE = 4
_RATE_LIMIT_LOG_PATH = Path(__file__).resolve().parent.parent / "data_cache" / "deepseek_call_log.json"


def _load_call_timestamps() -> list[float]:
    if not _RATE_LIMIT_LOG_PATH.exists():
        return []
    try:
        return json.loads(_RATE_LIMIT_LOG_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return []


def _save_call_timestamps(timestamps: list[float]) -> None:
    _RATE_LIMIT_LOG_PATH.parent.mkdir(exist_ok=True)
    _RATE_LIMIT_LOG_PATH.write_text(json.dumps(timestamps))


async def _throttle() -> None:
    """Block until fewer than RATE_LIMIT_PER_MINUTE calls have happened in
    the trailing 60 seconds, then record this call. A sliding window, not a
    fixed per-minute bucket, so it can't be burst-then-reset gamed."""
    while True:
        now = time.time()
        recent = [t for t in _load_call_timestamps() if now - t < 60]
        if len(recent) < RATE_LIMIT_PER_MINUTE:
            recent.append(now)
            _save_call_timestamps(recent)
            return
        wait_for = 60 - (now - min(recent))
        print(f"[rate limit] {len(recent)}/{RATE_LIMIT_PER_MINUTE} calls in the last minute, waiting {wait_for:.0f}s")
        await asyncio.sleep(max(wait_for, 1.0))

SYSTEM_PROMPT = """You are the research and execution agent for a paper-trading options \
account running in a week-long P&L race: $100,000 cash, $400,000 buying power, scored on \
each week's mark-to-market change on its own.

This is the first live day this exact pipeline has ever traded. Use 1 contract per trade \
regardless of how attractive a setup looks - this is a deliberate constraint for validating \
the system end-to-end, not a judgment call. The per-trade margin cap is also set tight for \
today and will reject anything larger.

Standards you must follow, and that are enforced in code, not just by this instruction:
- Start every session by calling `evaluate_positions` before anything else - not raw \
`get_all_positions`. It flags two concrete exit conditions computed against this agent's own \
trade log (Alpaca's position data has no entry-date field, so this can't be derived any other \
way): TAKE PROFIT once 65%+ of max available profit is captured, and a disproportionate-swing \
flag when the P&L move so far is 3x+ faster than the fraction of the position's planned life \
that has actually elapsed - in either direction. Treat TAKE PROFIT and DEFENSIVE EXIT \
recommendations as strong signals to act on with `close_position`, not just information to \
note. A position left open by default is a decision, not a non-action.
- Then call `screen_candidates` rather than defaulting to a name you already know - it \
surfaces large-cap, liquid-options underlyings ranked by how rich implied vol is running \
relative to their own realized vol right now. Research the top few, not just the single \
best-ranked one.
- The ONLY way to open a new trade is the `propose_and_execute_credit_spread` tool. It only \
accepts a short leg plus a further-OTM long leg on the same underlying/expiration/type \
(a vertical credit spread) - there is no way to submit a naked position through it. It checks \
your REAL current options buying power before approving anything - collateral already tied \
up in other open positions reduces what's available for a new one, on top of the flat \
per-trade cap.
- `close_position` is the only way to exit a trade before expiration. It has no margin gate \
(closing only ever reduces risk), but always give a rationale - it's part of the trade log.
- Before proposing any trade, call `analyze_spread_quality` on it. Favor spreads that show:
  - Strong gamma structure: net gamma from the two legs mostly offsetting, not a large \
uncovered short-gamma position (check `gamma.contained_relative_to_credit`).
  - Elevated VRP and high NVRP: implied volatility trading rich relative to the \
underlying's own realized volatility (`vrp.vrp_elevated`, `vrp.nvrp_high`) - this is the \
core edge for selling premium.
  - High IV rank: `vol_rank_proxy` is a realized-volatility-rank stand-in for true IV rank \
(Alpaca has no historical implied-vol series) - prefer `rank_pct` on the higher end.
  - Good expected-move characteristics: `expected_move.assessment` of "workable", not \
"narrow" (not enough premium) or "wide" (elevated event/gap risk).
  - Solid options volume: both legs' `liquidity_*.solid` should be true - don't trade into \
thin, one-sided markets.
  - `earnings_risk`: if `has_earnings_risk` is true, the underlying reports earnings before \
this option expires. This one IS effectively a hard gate - `propose_and_execute_credit_spread` \
will reject the trade unless you explicitly say in the rationale that this is a deliberate \
earnings play, since an earnings move can blow through what the other five signals assumed \
about normal-day volatility.
  None of the first five is individually a hard gate - weigh them together and use judgment - \
but a spread that fails most of them is a weak candidate regardless of what the raw premium \
looks like.
- Every other Alpaca tool available to you is read-only: account state, positions, option \
chains, quotes, and news. Use them to research before proposing a trade.
- A proposed trade will be rejected in code (not by you) if it isn't a real net credit, if its \
margin requirement exceeds the per-trade cap or your actual current buying power, or if it has \
unacknowledged earnings risk. If rejected, explain why to the user rather than retrying blindly.
- Paper trading fills a marketable limit order almost immediately, but not instantly - \
`propose_and_execute_credit_spread` submits the order and polls for a real fill before telling \
you it succeeded. A "SUBMITTED but not yet filled" result means the order is genuinely pending, \
not a bug - it has NOT been logged as an open position. Don't treat submission alone as success.

Research using the real tools available to you before proposing anything. If nothing looks \
like a reasonable defined-risk opportunity this week, say so explicitly rather than forcing \
a trade."""


def check_premarket_moves(symbols: list[str]) -> list[dict]:
    """Checks whether each symbol has a fresh pre-/after-hours price move,
    ahead of what option quotes may reflect yet. Real limitation: this
    account's data feed is IEX only (no SIP), which has much less
    off-hours participation than the consolidated tape - a result showing
    no fresh data means no data was available, not that nothing happened.
    """
    return premarket_briefing(symbols)


def screen_candidates(min_market_cap: float = 10_000_000_000, top: int = 20) -> list[dict]:
    """Find underlyings worth researching this week: pulls Alpaca's own
    most-active/movers lists, filters to a market-cap floor and liquid
    weekly options, then ranks the survivors by NVRP (how rich implied vol
    is running relative to the name's own realized vol) - highest first.
    """
    candidates = candidate_universe(min_market_cap=min_market_cap, top=top)
    return rank_by_vol_signal(candidates).to_dict(orient="records")


def analyze_spread_quality(
    underlying: str, short_symbol: str, long_symbol: str, contracts: int, credit_per_spread: float
) -> dict:
    """Run the six-factor quality check on a candidate credit spread before
    proposing it: gamma structure, VRP, NVRP, IV-rank proxy, expected-move
    characteristics, and liquidity.
    """
    return spread_quality_report(underlying, short_symbol, long_symbol, contracts, credit_per_spread)


def make_propose_and_execute(mcp_session: ClientSession):
    async def propose_and_execute_credit_spread(
        short_symbol: str,
        long_symbol: str,
        contracts: int,
        limit_credit: float,
        rationale: str,
    ) -> str:
        try:
            short_leg = parse_occ_symbol(short_symbol)
            long_leg = parse_occ_symbol(long_symbol)
        except ValueError as exc:
            return f"REJECTED: {exc}"

        if short_leg["underlying"] != long_leg["underlying"] or short_leg["option_type"] != long_leg["option_type"]:
            return "REJECTED: both legs must share the same underlying and option type."
        if short_leg["expiration"] != long_leg["expiration"]:
            return "REJECTED: both legs must share the same expiration."
        if limit_credit <= 0:
            return "REJECTED: limit_credit must be positive - this tool only places credit spreads."

        is_call = short_leg["option_type"] == "call"
        protective_direction_ok = (
            long_leg["strike"] > short_leg["strike"] if is_call else long_leg["strike"] < short_leg["strike"]
        )
        if not protective_direction_ok:
            return "REJECTED: long_symbol is not further out-of-the-money than short_symbol - this isn't a defined-risk spread."

        width = abs(short_leg["strike"] - long_leg["strike"])
        margin = max(width * 100 - limit_credit * 100, 0.0) * contracts
        if margin > MAX_SPREAD_MARGIN:
            return f"REJECTED: margin ${margin:,.0f} exceeds the per-trade cap of ${MAX_SPREAD_MARGIN:,.0f}."

        available = await live_options_buying_power(mcp_session)
        if margin > available:
            return (
                f"REJECTED: margin ${margin:,.0f} exceeds current available options buying power "
                f"${available:,.0f} - collateral already committed to other open positions leaves "
                f"less room than the static per-trade cap alone would suggest."
            )

        earnings = earnings_before_expiration(short_leg["underlying"], short_leg["expiration"])
        acknowledges_earnings = "earnings" in rationale.lower()

        if earnings["has_earnings_risk"] and not acknowledges_earnings:
            return (
                f"REJECTED: {short_leg['underlying']} reports earnings before this option expires "
                f"({earnings['earnings_dates']}) - an earnings move can blow through defined-risk "
                f"assumptions built on normal-day volatility. If this is a deliberate earnings play, "
                f"say so explicitly in the rationale (mention 'earnings') and resubmit."
            )
        if earnings["has_earnings_risk"]:
            earnings_note = f" [ACKNOWLEDGED EARNINGS RISK: {earnings['earnings_dates']}]"
        elif not earnings["checked"]:
            earnings_note = " [EARNINGS RISK UNKNOWN - Finnhub couldn't be reached to verify; proceed with caution]"
        else:
            earnings_note = ""

        submit_result = await mcp_session.call_tool(
            "place_option_order",
            {
                "qty": str(contracts),
                "type": "limit",
                "limit_price": str(-abs(limit_credit)),
                "legs": [
                    {"symbol": short_symbol, "ratio_qty": "1", "side": "sell", "position_intent": "sell_to_open"},
                    {"symbol": long_symbol, "ratio_qty": "1", "side": "buy", "position_intent": "buy_to_open"},
                ],
            },
        )
        submit_texts = [block.text for block in submit_result.content if hasattr(block, "text")]
        submit_payload = json.loads(submit_texts[0]) if submit_texts else {}
        order = submit_payload.get("data", submit_payload.get("result", submit_payload))
        order_id = order.get("id") if isinstance(order, dict) else None

        if not order_id:
            return f"SUBMITTED but couldn't parse an order id to confirm the fill - not logged as open. Raw response: {submit_texts}"

        fill = await confirm_fill(mcp_session, order_id)

        if fill["status"] == "filled":
            record_open(short_symbol, long_symbol, short_leg["underlying"], contracts, limit_credit, short_leg["expiration"])
            return f"APPROVED AND FILLED (margin ${margin:,.0f}).{earnings_note} {fill}"
        if fill["status"] == "partially_filled":
            record_open(short_symbol, long_symbol, short_leg["underlying"], contracts, limit_credit, short_leg["expiration"])
            return (
                f"PARTIALLY FILLED (margin ${margin:,.0f}).{earnings_note} logged, but confirm the actual filled size "
                f"via get_order_by_id before assuming the full {contracts} contracts are on. {fill}"
            )
        if fill["status"] in ("canceled", "rejected", "expired"):
            return f"NOT FILLED - order {fill['status']}. Nothing logged as open. {fill}"

        return (
            f"SUBMITTED but not yet filled after polling (order_id={order_id}, status={fill['status']}). "
            f"NOT logged as open - paper trading fills a marketable limit order almost immediately, so "
            f"this likely means the limit price isn't marketable yet. Check get_order_by_id('{order_id}') "
            f"later, or reconsider the price."
        )

    return propose_and_execute_credit_spread


def make_close_position(mcp_session: ClientSession):
    async def close_position(symbol: str, rationale: str) -> str:
        result = await mcp_session.call_tool("close_position", {"symbol_or_asset_id": symbol})
        texts = [block.text for block in result.content if hasattr(block, "text")]
        record_close(symbol)
        return f"Closed {symbol} ({rationale}). Order response: {texts}"

    return close_position


def make_evaluate_positions(mcp_session: ClientSession):
    async def evaluate_positions() -> list[dict]:
        return await evaluate_open_positions(mcp_session)

    return evaluate_positions


def _strip_additional_properties(schema):
    if isinstance(schema, dict):
        return {k: _strip_additional_properties(v) for k, v in schema.items() if k != "additionalProperties"}
    if isinstance(schema, list):
        return [_strip_additional_properties(v) for v in schema]
    return schema


# Hand-written OpenAI-style function schemas for the custom (non-MCP) tools -
# unlike google-genai, the openai SDK doesn't auto-generate a schema from a
# plain Python callable's type hints/docstring, so these are declared explicitly.
_CUSTOM_TOOL_SCHEMAS = {
    "check_premarket_moves": {
        "type": "function",
        "function": {
            "name": "check_premarket_moves",
            "description": check_premarket_moves.__doc__,
            "parameters": {
                "type": "object",
                "properties": {"symbols": {"type": "array", "items": {"type": "string"}, "description": "Tickers to check."}},
                "required": ["symbols"],
            },
        },
    },
    "screen_candidates": {
        "type": "function",
        "function": {
            "name": "screen_candidates",
            "description": screen_candidates.__doc__,
            "parameters": {
                "type": "object",
                "properties": {
                    "min_market_cap": {"type": "number", "description": "Minimum market cap in dollars."},
                    "top": {"type": "integer", "description": "How many names to pull from Alpaca's screener before filtering."},
                },
                "required": [],
            },
        },
    },
    "analyze_spread_quality": {
        "type": "function",
        "function": {
            "name": "analyze_spread_quality",
            "description": analyze_spread_quality.__doc__,
            "parameters": {
                "type": "object",
                "properties": {
                    "underlying": {"type": "string", "description": "Underlying ticker, e.g. AAPL."},
                    "short_symbol": {"type": "string", "description": "OCC symbol of the leg you'd sell."},
                    "long_symbol": {"type": "string", "description": "OCC symbol of the protective leg you'd buy."},
                    "contracts": {"type": "integer", "description": "Number of spreads you're considering."},
                    "credit_per_spread": {"type": "number", "description": "Net credit per spread, in dollars per share."},
                },
                "required": ["underlying", "short_symbol", "long_symbol", "contracts", "credit_per_spread"],
            },
        },
    },
    "propose_and_execute_credit_spread": {
        "type": "function",
        "function": {
            "name": "propose_and_execute_credit_spread",
            "description": (
                "Place a defined-risk vertical credit spread: sell short_symbol, buy long_symbol as "
                "protection, for a net credit of limit_credit per spread. This is the only tool that "
                "can touch the account."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "short_symbol": {"type": "string", "description": "OCC option symbol to sell (e.g. AAPL260918P00185000)."},
                    "long_symbol": {
                        "type": "string",
                        "description": "OCC option symbol to buy as protection - must be further out-of-the-money than short_symbol, same underlying, expiration, and option type.",
                    },
                    "contracts": {"type": "integer", "description": "Number of spreads."},
                    "limit_credit": {
                        "type": "number",
                        "description": "Net credit per spread, in dollars per share (must be positive - this tool refuses debit trades).",
                    },
                    "rationale": {"type": "string", "description": "One or two sentences on why this trade makes sense right now, for the trade log."},
                },
                "required": ["short_symbol", "long_symbol", "contracts", "limit_credit", "rationale"],
            },
        },
    },
    "close_position": {
        "type": "function",
        "function": {
            "name": "close_position",
            "description": "Close (or reduce) an existing position - the only way to exit a trade before expiration rather than holding it to settlement.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "OCC option symbol, or underlying ticker if this is an assigned/exercised stock position, to close."},
                    "rationale": {"type": "string", "description": "One or two sentences on why you're closing this now."},
                },
                "required": ["symbol", "rationale"],
            },
        },
    },
    "evaluate_positions": {
        "type": "function",
        "function": {
            "name": "evaluate_positions",
            "description": (
                "Check every open position this agent has logged for exit signals: TAKE PROFIT once "
                "65%+ of max profit is captured, or a DEFENSIVE/CONSIDER-CLOSING flag if the P&L swing "
                "is moving 3x+ faster than the fraction of the position's planned life elapsed. Call "
                "this instead of reasoning from raw get_all_positions output alone."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
}


def _mcp_tool_to_openai(tool) -> dict:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": _strip_additional_properties(tool.inputSchema),
        },
    }


async def run(user_message: str, model: str = "deepseek-v4-flash") -> str:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    client = AsyncOpenAI(api_key=api_key, base_url=_BASE_URL)
    server_params = StdioServerParameters(command=_default_server_path(), args=["--env-file", str(_ENV_FILE)])

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as mcp_session:
            await mcp_session.initialize()

            tools_result = await mcp_session.list_tools()
            read_only_mcp_tools = [t for t in tools_result.tools if t.name not in _MUTATING_TOOLS]
            read_only_names = {t.name for t in read_only_mcp_tools}

            gated_execution_tool = make_propose_and_execute(mcp_session)
            close_tool = make_close_position(mcp_session)
            evaluate_tool = make_evaluate_positions(mcp_session)
            dispatch = {
                "screen_candidates": screen_candidates,
                "analyze_spread_quality": analyze_spread_quality,
                "propose_and_execute_credit_spread": gated_execution_tool,
                "close_position": close_tool,
                "evaluate_positions": evaluate_tool,
                "check_premarket_moves": check_premarket_moves,
            }

            tools = [_CUSTOM_TOOL_SCHEMAS[name] for name in dispatch] + [
                _mcp_tool_to_openai(t) for t in read_only_mcp_tools
            ]

            messages: list[dict] = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ]
            final_text = ""

            for _ in range(MAX_LOOP_TURNS):
                await _throttle()
                response = await client.chat.completions.create(model=model, messages=messages, tools=tools)
                message = response.choices[0].message

                if not message.tool_calls:
                    final_text = message.content or final_text
                    print(f"[assistant] {final_text}")
                    break

                messages.append(message.model_dump(exclude_none=True))

                for call in message.tool_calls:
                    name = call.function.name
                    try:
                        args = json.loads(call.function.arguments) if call.function.arguments else {}
                    except json.JSONDecodeError:
                        args = {}
                    print(f"[tool call] {name}({args})")

                    if name in dispatch:
                        func = dispatch[name]
                        result: Any = await func(**args) if asyncio.iscoroutinefunction(func) else func(**args)
                    elif name in read_only_names:
                        mcp_result = await mcp_session.call_tool(name, args)
                        result = [block.text for block in mcp_result.content if hasattr(block, "text")]
                    else:
                        result = f"ERROR: '{name}' is not an authorized tool."

                    messages.append(
                        {"role": "tool", "tool_call_id": call.id, "content": json.dumps(result, default=str)}
                    )
            else:
                print("[warning] hit the max loop-turn cap without a final answer")

            return final_text


if __name__ == "__main__":
    import sys

    prompt = " ".join(sys.argv[1:]) or (
        "Check the account, then screen for candidate underlyings this week. "
        "If there's a reasonable defined-risk credit spread opportunity, "
        "propose and execute it. Otherwise, explain why you're passing."
    )
    print(asyncio.run(run(prompt)))
