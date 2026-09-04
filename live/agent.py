"""The actual "AI assistant talks to Alpaca" agent, matching the hackathon's
framing of the MCP server rather than the deterministic live/ modules.

Architecture note: Anthropic's native `mcp_servers` connector requires the
MCP server to be publicly reachable over HTTPS (Anthropic's own servers
call out to it) - a local stdio process, which is what alpaca-mcp-server
is here, cannot be used that way. Instead this uses Anthropic's
client-side MCP helpers (`anthropic[mcp]`, `async_mcp_tool`) to wrap our
own local MCP session's tools for the SDK's tool_runner - Claude still
calls Alpaca's real MCP tools, just through a locally-held connection
instead of a server-to-server one.

Safety split: every *read* tool from the Alpaca MCP server (account,
positions, market data, news, chains) is handed to Claude directly - it
can research freely. Every *mutating* tool (place/cancel/close/exercise/
watchlist writes) is withheld from Claude entirely; the only way this
agent can touch the account is `propose_and_execute_credit_spread`, a
custom tool defined in this file, which is deliberately spread-only by
its own schema (no naked positions are even expressible) and enforces the
per-trade margin cap in code before it ever calls Alpaca. The model's own
adherence to instructions is not the safety mechanism here - the tool
surface is.
"""
import asyncio
import json
from pathlib import Path

from anthropic import AsyncAnthropic, beta_async_tool, beta_tool
from anthropic.lib.tools.mcp import async_mcp_tool
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from data.earnings_calendar import earnings_before_expiration
from data.options import parse_occ_symbol
from data.screener import candidate_universe, rank_by_vol_signal
from live.mcp_client import _default_server_path
from live.order_helpers import close_both_legs, confirm_fill, live_options_buying_power, resolved_credit
from live.position_management import evaluate_open_positions
from live.premarket_check import premarket_briefing
from live.trade_log import record_close, record_open
from signals.options_quality import spread_quality_report

load_dotenv()

_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"

# Every order-placing / account-mutating tool the Alpaca MCP server exposes.
# Claude never sees these directly - only through the gated tool below.
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


@beta_tool
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


@beta_tool
def screen_candidates(min_market_cap: float = 2_000_000_000, top: int = 20) -> list[dict]:
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


@beta_tool
def analyze_spread_quality(underlying: str, short_symbol: str, long_symbol: str, contracts: int, credit_per_spread: float) -> dict:
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
about normal-day volatility. Earnings-driven IV isn't an edge this pipeline trades - it's a risk \
to avoid, not a setup to seek out.
  - `macro_risk`: a high-impact scheduled macro event (NFP, CPI, FOMC, PPI, etc.) landing on or \
before expiration. This is a soft flag, not a hard gate - unlike earnings, it isn't blocked in \
code - but weigh it seriously: a broad market-moving print on expiration day is correlated risk \
across every position you hold, not just this one, and it settles same-day with no time to \
react if it gaps against you. `checked: false` means the calendar source couldn't be reached \
(it's a free feed and does rate-limit) - treat unknown the same as a real risk, not as "clear". \
`has_macro_risk: true` with named events should push you toward a smaller size or passing \
outright, not just noting it in the rationale.
  None of these is individually a hard gate except earnings_risk - weigh them together and use \
judgment - but a spread that fails most of them is a weak candidate regardless of what the raw \
premium looks like.
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


def make_propose_and_execute_tool(mcp_session: ClientSession):
    """Factory kept separate from the tool body so the MCP session can be
    closed over without becoming part of the tool's declared schema."""

    @beta_async_tool
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
            record_open(short_symbol, long_symbol, short_leg["underlying"], contracts, resolved_credit(fill, limit_credit), short_leg["expiration"])
            return f"APPROVED AND FILLED (margin ${margin:,.0f}).{earnings_note} {fill}"
        if fill["status"] == "partially_filled":
            record_open(short_symbol, long_symbol, short_leg["underlying"], contracts, resolved_credit(fill, limit_credit), short_leg["expiration"])
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


def make_close_position_tool(mcp_session: ClientSession):
    """The only way this agent can exit a position before expiration -
    without this, every trade would simply ride to expiration with no
    ability to take profit early, cut a loss, or free up collateral.
    Closing is risk-reducing by nature, so this carries no margin/credit
    gate the way opening a trade does - just a real Alpaca call and a
    logged reason.
    """

    @beta_async_tool
    async def close_position(symbol: str, rationale: str) -> str:
        """Close (or reduce) an existing position - the only way to exit a
        trade before expiration rather than holding it to settlement.

        Args:
            symbol: OCC option symbol, or underlying ticker if this is an
                assigned/exercised stock position, to close.
            rationale: One or two sentences on why you're closing this now
                (e.g. most of the credit captured, thesis invalidated, the
                underlying has moved against the short strike).
        Closing either leg of a logged spread automatically closes its
        paired leg too, so a spread never gets left half-open.
        """
        outcome = await close_both_legs(mcp_session, symbol)
        for sym in outcome["closed_symbols"]:
            record_close(sym, realized_pnl=outcome["realized_pnl_by_symbol"].get(sym))
        summary = f"Closed {outcome['closed_symbols']} ({rationale}). Results: {outcome['results']}"
        if outcome["errors"]:
            summary += f" ERRORS on some legs (may already be closed): {outcome['errors']}"
        return summary

    return close_position


def make_evaluate_positions_tool(mcp_session: ClientSession):
    @beta_async_tool
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


async def run(user_message: str, model: str = "claude-opus-5", max_tokens: int = 16000) -> str:
    client = AsyncAnthropic()
    server_params = StdioServerParameters(command=_default_server_path(), args=["--env-file", str(_ENV_FILE)])

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as mcp_session:
            await mcp_session.initialize()

            tools_result = await mcp_session.list_tools()
            read_only_tools = [
                async_mcp_tool(t, mcp_session) for t in tools_result.tools if t.name not in _MUTATING_TOOLS
            ]
            gated_execution_tool = make_propose_and_execute_tool(mcp_session)
            close_tool = make_close_position_tool(mcp_session)
            evaluate_tool = make_evaluate_positions_tool(mcp_session)

            runner = client.beta.messages.tool_runner(
                model=model,
                max_tokens=max_tokens,
                system=SYSTEM_PROMPT,
                # Tools + system prompt are ~55 tool schemas and don't change between
                # turns; without this, every turn in the loop re-bills them at full
                # price. cache_control caches everything up to and including the
                # last cacheable block (here, the tool list), so turn 2+ reads it
                # back at ~10% of the cost instead of paying full price again.
                cache_control={"type": "ephemeral"},
                tools=[
                    *read_only_tools,
                    check_premarket_moves,
                    screen_candidates,
                    analyze_spread_quality,
                    gated_execution_tool,
                    close_tool,
                    evaluate_tool,
                ],
                messages=[{"role": "user", "content": user_message}],
            )

            final_text = ""
            async for message in runner:
                for block in message.content:
                    if block.type == "text":
                        final_text = block.text
                        print(f"[assistant] {block.text}")
                    elif block.type == "tool_use":
                        print(f"[tool call] {block.name}({block.input})")
            return final_text


if __name__ == "__main__":
    import sys

    prompt = " ".join(sys.argv[1:]) or (
        "Check the account, then screen for candidate underlyings this week. "
        "If there's a reasonable defined-risk credit spread opportunity, "
        "propose and execute it. Otherwise, explain why you're passing."
    )
    print(asyncio.run(run(prompt)))
