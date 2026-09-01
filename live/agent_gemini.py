"""Gemini 3.7 Flash port of the "AI assistant talks to Alpaca" agent - a
from-scratch rebuild of live/agent.py on Google's Gen AI SDK, sharing the
same business logic (signals/, data/screener.py) and the same safety
architecture.

Verified locally against the installed google-genai==2.8.0 before writing
this loop (no API key needed for this part): GenerateContentConfig.tools
accepts a per-element mix of Tool / plain Callable / raw mcp.types.Tool /
mcp.ClientSession, and a bare mcp.types.Tool (no session attached) builds
and passes schema-only - it cannot be auto-executed because nothing ties
it to a session. response.function_calls returns FunctionCall objects
directly (.name, .args) - confirmed by reading the SDK source, not
guessed. What is NOT verified: the actual generate_content network
behavior (whether Gemini's tool-selection and argument formatting work
as expected against this real tool set) - that needs GEMINI_API_KEY,
which isn't set yet.

Safety split, same as the Claude version: automatic_function_calling is
disabled entirely, so every tool call - MCP-sourced or our own - is
executed by this file's manual loop, never by SDK auto-execution. Only
read-only Alpaca tools are declared to the model; every mutating tool is
withheld and reachable only through propose_and_execute_credit_spread,
which runs the same code-enforced checks as the Claude version before
ever calling Alpaca for real.
"""
import asyncio
import json
import os
from datetime import date
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

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

# Same set as live/agent.py - kept in sync manually since this is a
# separate provider implementation, not a shared import, to keep each
# agent file independently auditable for what it withholds.
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

# Confirmed live on this first trading day: gemini-3.7-flash (and the
# gemini-flash-latest alias, which points at it) returned a sustained
# "503 UNAVAILABLE - high demand" for 10+ minutes, while gemini-3.6-flash
# and gemini-2.5-flash both answered normally on the same API key at the
# same time. Falls through in order on a ServerError; stays on the primary
# otherwise. Not a permanent downgrade - just resilience against exactly
# this kind of single-model capacity outage.
MODEL_FALLBACKS = ["gemini-3.6-flash", "gemini-2.5-flash"]

# Reasoning-depth calibration: cheap for rapid single-value lookups, more
# for comparing multiple candidates against each other or verifying a
# trade's risk parameters before it's allowed to execute.
_ALLOCATION_TOOLS = {"screen_candidates", "get_all_positions"}
_RISK_VERIFICATION_TOOLS = {
    "analyze_spread_quality",
    "propose_and_execute_credit_spread",
    "close_position",
    "evaluate_positions",
}


def _thinking_level_for(last_tool_names: set[str]) -> types.ThinkingLevel:
    if last_tool_names & _RISK_VERIFICATION_TOOLS:
        return types.ThinkingLevel.HIGH
    if last_tool_names & _ALLOCATION_TOOLS:
        return types.ThinkingLevel.MEDIUM
    return types.ThinkingLevel.LOW  # rapid price/quote/news/account checks

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

    Args:
        symbols: Tickers to check.
    """
    return premarket_briefing(symbols)


def screen_candidates(min_market_cap: float = 10_000_000_000, top: int = 20) -> list[dict]:
    """Find underlyings worth researching this week: pulls Alpaca's own
    most-active/movers lists, filters to a market-cap floor (Finnhub, if
    configured - otherwise restricted to a curated large-cap list, since
    Alpaca itself has no market-cap data) and liquid weekly options, then
    ranks the survivors by NVRP (how rich implied vol is running relative
    to the name's own realized vol) - highest first. Call this before
    picking a specific underlying, rather than defaulting to a name you
    already know.

    Args:
        min_market_cap: Minimum market cap in dollars.
        top: How many names to pull from Alpaca's screener before filtering.
    """
    candidates = candidate_universe(min_market_cap=min_market_cap, top=top)
    return rank_by_vol_signal(candidates).to_dict(orient="records")


def analyze_spread_quality(
    underlying: str, short_symbol: str, long_symbol: str, contracts: int, credit_per_spread: float
) -> dict:
    """Run the six-factor quality check on a candidate credit spread before
    proposing it: gamma structure, VRP, NVRP, IV-rank proxy, expected-move
    characteristics, and liquidity. Always call this before
    propose_and_execute_credit_spread - do not evaluate a spread from the
    raw option chain alone.

    Args:
        underlying: Underlying ticker, e.g. AAPL.
        short_symbol: OCC symbol of the leg you'd sell.
        long_symbol: OCC symbol of the protective leg you'd buy.
        contracts: Number of spreads you're considering.
        credit_per_spread: Net credit per spread you'd expect to collect,
            in dollars per share.
    """
    return spread_quality_report(underlying, short_symbol, long_symbol, contracts, credit_per_spread)


def make_propose_and_execute(mcp_session: ClientSession):
    """Factory kept separate from the tool body so the MCP session can be
    closed over without becoming part of the function's declared schema."""

    async def propose_and_execute_credit_spread(
        short_symbol: str,
        long_symbol: str,
        contracts: int,
        limit_credit: float,
        rationale: str,
    ) -> str:
        """Place a defined-risk vertical credit spread: sell `short_symbol`,
        buy `long_symbol` as protection, for a net credit of `limit_credit`
        per spread. This is the only tool that can touch the account.

        Args:
            short_symbol: OCC option symbol to sell (e.g. AAPL260918P00185000).
            long_symbol: OCC option symbol to buy as protection - must be
                further out-of-the-money than short_symbol, same underlying,
                expiration, and option type.
            contracts: Number of spreads.
            limit_credit: Net credit per spread, in dollars per share (must
                be positive - this tool refuses debit trades).
            rationale: One or two sentences on why this trade makes sense
                right now, for the trade log.
        """
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
    """The only way this agent can exit a position before expiration -
    without this, every trade would simply ride to expiration with no
    ability to take profit early, cut a loss, or free up collateral.
    Closing is risk-reducing by nature, so this carries no margin/credit
    gate the way opening a trade does - just a real Alpaca call and a
    logged reason.
    """

    async def close_position(symbol: str, rationale: str) -> str:
        """Close (or reduce) an existing position - the only way to exit a
        trade before expiration rather than holding it to settlement.

        Args:
            symbol: OCC option symbol, or underlying ticker if this is an
                assigned/exercised stock position, to close.
            rationale: One or two sentences on why you're closing this now
                (e.g. most of the credit captured, thesis invalidated, the
                underlying has moved against the short strike).
        """
        result = await mcp_session.call_tool("close_position", {"symbol_or_asset_id": symbol})
        texts = [block.text for block in result.content if hasattr(block, "text")]
        record_close(symbol)
        return f"Closed {symbol} ({rationale}). Order response: {texts}"

    return close_position


def make_evaluate_positions(mcp_session: ClientSession):
    async def evaluate_positions() -> list[dict]:
        """Check every open position this agent has logged for exit
        signals: TAKE PROFIT once a large share of max profit is captured
        (65%+), or a DEFENSIVE/CONSIDER-CLOSING flag if the P&L swing so
        far is moving much faster (3x+) than the fraction of the position's
        planned life that has actually elapsed - in either direction. Call
        this instead of reasoning from raw get_all_positions output alone;
        Alpaca's position data has no entry-date field, so pace-of-swing
        can only be computed here, against this agent's own trade log.
        """
        return await evaluate_open_positions(mcp_session)

    return evaluate_positions


def _strip_additional_properties(schema):
    """Drop every `additionalProperties` key from a JSON schema, recursively.

    Confirmed live crash: several Alpaca MCP tool schemas (e.g. get_stock_bars,
    place_stock_order) set `additionalProperties: false` (or `true` on a nested
    anyOf branch) at multiple levels. google-genai's own MCP-to-Gemini schema
    converter (_filter_to_supported_schema) always recurses into
    `additionalProperties` as if it were a nested object schema, and blows up
    with `AttributeError: 'bool' object has no attribute 'items'` the moment it
    isn't one. Gemini function-calling has no use for the keyword anyway, so
    dropping it is lossless here.
    """
    if isinstance(schema, dict):
        return {k: _strip_additional_properties(v) for k, v in schema.items() if k != "additionalProperties"}
    if isinstance(schema, list):
        return [_strip_additional_properties(v) for v in schema]
    return schema


def _sanitize_mcp_tool(tool):
    return tool.model_copy(update={"inputSchema": _strip_additional_properties(tool.inputSchema)})


def _is_fallback_worthy(exc: Exception) -> bool:
    """A 503 (model overloaded) or a 429 RESOURCE_EXHAUSTED (this specific
    model's daily/rate quota is used up - confirmed live: the free-tier key
    in use here has a hard 20-requests/day cap per model) both mean "this
    model is unusable right now, try the next one". Any other 4xx (e.g. a
    malformed-request 400, which is a real code bug, not a capacity issue)
    should propagate instead of being silently masked by a model switch.
    """
    if isinstance(exc, genai_errors.ServerError):
        return True
    if isinstance(exc, genai_errors.ClientError):
        return exc.code == 429
    return False


_EXHAUSTED_CACHE_PATH = Path(__file__).resolve().parent.parent / "data_cache" / "gemini_quota_exhausted.json"


def _load_exhausted_models() -> set[str]:
    """Models confirmed maxed out on their free-tier RPD (requests-per-day)
    cap, persisted to disk and keyed by date.

    Confirmed the hard way tonight: this in-process cache used to live only
    in a module-level set, which reset on every relaunch - and this pipeline
    got relaunched ~7 times in under an hour chasing unrelated bugs, each
    relaunch wasting one more real API call re-probing a model Google had
    already told us was maxed out an hour earlier. Persisting to disk means
    a relaunch doesn't repeat that mistake. Google's free-tier RPD reset
    time isn't published as UTC-aligned, so this is approximate, not exact -
    but "sometimes clears the cache a bit early" costs one wasted probe,
    while never persisting cost dozens tonight.
    """
    if not _EXHAUSTED_CACHE_PATH.exists():
        return set()
    try:
        data = json.loads(_EXHAUSTED_CACHE_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return set()
    if data.get("date") != date.today().isoformat():
        return set()
    return set(data.get("models", []))


def _save_exhausted_models(models: set[str]) -> None:
    _EXHAUSTED_CACHE_PATH.parent.mkdir(exist_ok=True)
    _EXHAUSTED_CACHE_PATH.write_text(json.dumps({"date": date.today().isoformat(), "models": sorted(models)}))


_exhausted_models: set[str] = _load_exhausted_models()  # models that hit a 429 quota today - a daily cap, no point retrying


_no_thinking_support: set[str] = set()  # models confirmed to reject thinking_config outright


async def _generate_with_fallback(client: genai.Client, model: str, contents, config):
    """generate_content against `model`, falling back through
    MODEL_FALLBACKS in order when a model is unusable (overloaded or its
    quota is exhausted) rather than propagating and killing the whole
    session over what's usually a single-model problem.

    Also handles a confirmed real incompatibility: gemini-2.5-flash (one of
    the fallbacks) rejects `thinking_config` outright with a 400 - it's not
    a "this model is down" error like the others, so it gets its own retry:
    same candidate, config stripped of thinking_config, instead of wasting
    the attempt and moving on.
    """
    last_exc: Exception | None = None
    candidates = [c for c in [model, *MODEL_FALLBACKS] if c not in _exhausted_models] or [model]
    for candidate in candidates:
        candidate_config = config
        if candidate in _no_thinking_support and getattr(config, "thinking_config", None) is not None:
            candidate_config = config.model_copy(update={"thinking_config": None})
        try:
            if candidate != model:
                print(f"[model fallback] {model} unavailable, trying {candidate}")
            return await client.aio.models.generate_content(model=candidate, contents=contents, config=candidate_config)
        except Exception as exc:
            if (
                isinstance(exc, genai_errors.ClientError)
                and exc.code == 400
                and "Thinking level is not supported" in str(exc)
                and candidate not in _no_thinking_support
            ):
                print(f"[model fallback] {candidate} doesn't support thinking_config, retrying without it")
                _no_thinking_support.add(candidate)
                try:
                    return await client.aio.models.generate_content(
                        model=candidate, contents=contents, config=config.model_copy(update={"thinking_config": None})
                    )
                except Exception as retry_exc:
                    exc = retry_exc
            if not _is_fallback_worthy(exc):
                raise
            if isinstance(exc, genai_errors.ClientError) and exc.code == 429:
                print(f"[model fallback] {candidate} quota exhausted for today, no longer trying it today")
                _exhausted_models.add(candidate)
                _save_exhausted_models(_exhausted_models)
            last_exc = exc
            continue
    raise last_exc


async def run(user_message: str, model: str = "gemini-3.7-flash") -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    server_params = StdioServerParameters(command=_default_server_path(), args=["--env-file", str(_ENV_FILE)])

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as mcp_session:
            await mcp_session.initialize()

            tools_result = await mcp_session.list_tools()
            read_only_mcp_tools = [
                _sanitize_mcp_tool(t) for t in tools_result.tools if t.name not in _MUTATING_TOOLS
            ]
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
            }

            base_tools = [
                *read_only_mcp_tools,
                check_premarket_moves,
                screen_candidates,
                analyze_spread_quality,
                gated_execution_tool,
                close_tool,
                evaluate_tool,
            ]

            contents = [types.Content(role="user", parts=[types.Part.from_text(text=user_message)])]
            final_text = ""
            last_tool_names: set[str] = set()

            for _ in range(MAX_LOOP_TURNS):
                level = _thinking_level_for(last_tool_names)
                config = types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    tools=base_tools,
                    # Disabled everywhere, not just for our own tools: an MCP
                    # tool passed with no attached session can't auto-execute
                    # anyway, but keeping this explicit means every call in
                    # this loop - MCP or custom - is dispatched by our own
                    # code, never by SDK-side automatic execution.
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                    thinking_config=types.ThinkingConfig(thinking_level=level),
                )
                print(f"[thinking level: {level.value}]")
                response = await _generate_with_fallback(client, model, contents, config)

                calls = response.function_calls
                if not calls:
                    final_text = getattr(response, "text", None) or final_text
                    print(f"[assistant] {final_text}")
                    break

                contents.append(response.candidates[0].content)

                response_parts = []
                for fc in calls:
                    args = dict(fc.args) if fc.args else {}
                    print(f"[tool call] {fc.name}({args})")

                    if fc.name in dispatch:
                        func = dispatch[fc.name]
                        result: Any = await func(**args) if asyncio.iscoroutinefunction(func) else func(**args)
                    elif fc.name in read_only_names:
                        mcp_result = await mcp_session.call_tool(fc.name, args)
                        result = [block.text for block in mcp_result.content if hasattr(block, "text")]
                    else:
                        result = f"ERROR: '{fc.name}' is not an authorized tool."

                    response_parts.append(types.Part.from_function_response(name=fc.name, response={"result": result}))

                # "user" is the spec-correct role for a function response
                # (confirmed via the API's own error message, which lists
                # valid roles and "tool" isn't one of them) - gemini-3.7-flash
                # tolerated "tool" leniently, but the fallback models reject
                # it outright with a 400.
                contents.append(types.Content(role="user", parts=response_parts))
                last_tool_names = {fc.name for fc in calls}
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
