"""DeepSeek port of the "AI assistant talks to Alpaca" agent - same
business logic and safety architecture as live/agent.py, rebuilt on
DeepSeek's OpenAI-compatible chat completions API.

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
import uuid
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import AsyncOpenAI

from data.earnings_calendar import earnings_before_expiration
from data.options import parse_occ_symbol
from data.screener import wide_vectorized_screen
from live.mcp_client import _default_server_path, unwrap
from live.order_helpers import (
    close_both_legs,
    compute_marketable_credit,
    confirm_fill,
    live_account_equity,
    live_options_buying_power,
    resolved_credit,
)
from live.position_management import evaluate_open_positions
from live.premarket_check import premarket_briefing
from live.activity_log import _truncate, log_event
from live import strategy_config
from live.token_usage import record as record_token_usage
from live.trade_log import record_close, record_open
from signals import disqualification_cache
from signals.material_news import material_news_check
from signals.mean_reversion import mean_reversion_signal
from signals.options_quality import spread_quality_report

load_dotenv()

_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
_BASE_URL = "https://api.deepseek.com"

# Same set as live/agent.py - kept in sync manually, not shared, for the
# same "independently auditable" reason.
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

MAX_SPREAD_MARGIN_PCT_OF_EQUITY = 0.06  # per-trade realistic-max-loss cap, stated as a fraction of the account, not a frozen dollar figure - raised 2%->4% after a two-day track record (JPM/AAPL/TSLA), then 4%->6% after a full week live
MAX_LOOP_TURNS = 30  # hard cap against a runaway tool-calling loop - raised from 12 after confirming live that a single session chasing one directional idea through the wide screen (checking 5-6 alternates, each needing several check_mean_reversion/check_material_news/analyze_spread_quality calls) burned through 76 tool calls and hit the 12-turn cap TWICE without ever producing a final answer, let alone room left to also evaluate a condor candidate in the same session

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

This pipeline now has a real week-long track record, so the earlier 1-contract-per-trade \
validation constraint is lifted - size contracts deliberately based on conviction and liquidity, \
not habit. Per-trade max loss is capped at 6% of current account equity (the account's stated \
realistic-max-loss risk tolerance per trade) and will reject anything larger - use enough \
contracts to make meaningful use of that room on a genuinely good setup, rather than defaulting \
to the smallest size out of old habit.

Standards you must follow, and that are enforced in code, not just by this instruction:
- Start every session by calling `evaluate_positions` before anything else - not raw \
`get_all_positions`. It flags two concrete exit conditions computed against this agent's own \
trade log (Alpaca's position data has no entry-date field, so this can't be derived any other \
way): TAKE PROFIT once 65%+ of max available profit is captured, and a disproportionate-swing \
flag when the P&L move so far is 3x+ faster than the fraction of the position's planned life \
that has actually elapsed - in either direction. Treat TAKE PROFIT and DEFENSIVE EXIT \
recommendations as strong signals to act on with `close_position`, not just information to \
note. A position left open by default is a decision, not a non-action.
- Then call `screen_candidates` rather than defaulting to a name you already know. This screens \
hundreds of liquid names in one call, not a handful - one bulk price pull ranks the whole \
universe by realized-vol/mean-reversion with pure vectorized math (no per-symbol API call, no \
LLM reasoning involved), then only the ~40 that actually look interesting get the real \
per-underlying data that can't be bulked (implied vol off the live chain, market cap, liquid-\
weekly-options). What comes back to you has ALREADY been narrowed to names with a real VRP/NVRP \
edge, with `z_score`/`is_extreme`/`regime` (mean-reversion) pre-computed on every row - that part \
is still plain deterministic code, not an LLM judgment call, just run automatically now instead \
of waiting on you to call `check_mean_reversion` one underlying at a time. Material news is \
deliberately NOT pre-checked here, even for names that pass every numeric filter - whether a \
headline actually invalidates a setup isn't arithmetic, it needs the real text read and weighed, \
so only call `check_material_news` yourself on the one or two names you're seriously about to \
trade, not the whole table. Research the top few by NVRP, not just the single best-ranked one.
- If a name you remember from a recent session (e.g. one that failed on real earnings or material \
news) is missing from this table, that's deliberate, not a bug - a name you or a prior session \
already rejected for a hard, time-stable reason (earnings date, real news) stays filtered out of \
the screen for about an hour so you're not re-deriving the same rejection and re-explaining it \
every session instead of looking at something new. This ONLY affects what gets surfaced here - if \
you deliberately re-check a filtered name yourself, or try to execute on it, every real gate still \
runs fresh with no memory of the earlier rejection.
- There are two ways to open a new trade - pick deliberately, don't default to the first one \
out of habit. Both are equally legitimate; most candidates on any given day will only qualify \
for one of them, and that's fine - don't force a directional thesis onto a name that's just a \
clean VRP/NVRP condor, and don't skip a condor because a more "interesting" directional setup is \
still being chased. If a name doesn't clear the mean-reversion bar, that's your answer for that \
name - move to the next candidate or default to the condor, rather than spending many tool calls \
trying to build a case for a structure that keeps getting rejected. The table you get back from \
`screen_candidates` typically has room for more than one trade in a session - don't burn the \
whole session's tool-call budget re-analyzing one candidate from every angle before considering \
any other.
  - `propose_and_execute_credit_spread`: a single vertical spread (short leg + further-OTM \
long leg, same underlying/expiration/type). This is a DIRECTIONAL bet - a put spread wins \
unless the stock falls through your strike, a call spread wins unless it rises through yours. \
The ONLY thing that can justify picking a direction is an extreme mean-reversion setup, and ALL \
THREE of the following are enforced in code, not just advisory:
  1. Call `check_mean_reversion` - `is_extreme` (price is 2+ standard deviations from its own \
20-day mean) must be true, and the spread type must match the reversion direction (overbought -> \
call spread betting on reversion down; oversold -> put spread betting on reversion up) - the \
opposite pairing is rejected as a momentum bet dressed as a reversion thesis.
  2. Same call: `favorable_for_reversion` must be true (the market must be choppy/range-bound, \
via Kaufman's Efficiency Ratio - not trending). Mean reversion assumes price snaps back toward \
its average; in a real trend an extreme z-score is usually the trend continuing, not about to \
revert, and this is checked, not assumed.
  3. Call `check_material_news` - `has_material_news` must be false. A real catalyst (leadership \
change, M&A, legal/regulatory action, guidance surprise) means the move may be a justified \
repricing, not a statistical fluke, and blocks the trade outright regardless of how extreme the \
z-score looks.
This tool still requires the same VRP/NVRP edge as the condor (elevated IV is also enforced in \
code) - mean reversion justifies the direction, VRP/NVRP justifies selling \
premium at all; both are required together, not either/or.
  - `propose_and_execute_iron_condor`: sells a put spread AND a call spread together as one \
4-leg structure, profiting as long as the stock stays between both short strikes. This is the \
tool that actually matches what "IV is rich relative to realized vol" means - it requires no \
directional view at all. Default to this one when your only real signal is elevated VRP/NVRP \
and you don't have a separate, stated directional thesis. Also hard-gated on material news, same \
as the directional tool - a real catalyst (leadership change, M&A, legal action, guidance \
surprise) can move a stock through a wing sized for normal-day volatility regardless of \
structure, and unlike the earnings gate there is NO override for this one - it always rejects.
Both tools check your REAL current options buying power before approving anything - collateral \
already tied up in other open positions reduces what's available for a new one, on top of the \
6%-of-equity per-trade max-loss cap.
- `close_position` is the only way to exit a trade before expiration. It has no margin gate \
(closing only ever reduces risk), but always give a rationale - it's part of the trade log.
- Before proposing any trade, call `analyze_spread_quality` on it - for an iron condor, call it \
once for the put side and once for the call side, since it only evaluates one 2-leg spread at a \
time. Favor spreads that show:
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
- Paper trading fills a marketable limit order almost immediately, but not instantly - both \
execution tools submit the order and poll for a real fill (up to 120s) before telling you it \
succeeded. If it still hasn't filled by then, the order is now actively CANCELED for you, not \
left pending - "NOT FILLED" means there is no live order left on that structure, and it's \
completely safe to resubmit at a different price. Never submit a second order for the same \
structure while treating the first as still maybe-live - always wait for this tool's own \
response before deciding what to do next, since a genuinely-unknown status (a failed cancel, \
called out explicitly when it happens) is the ONLY case where resubmitting risks a double-fill.
- `limit_credit` is a MINIMUM, not the exact price used - both execution tools re-price off live \
quotes right before submitting and will use a better credit than you asked for if the live book \
supports it, but REJECT outright rather than submit below your floor. This means a non-fill \
should be rare now (you're no longer guessing a stale price) - if you do get REJECTED for a \
live-book reason, that's real, current information that the setup's edge has moved, not just a \
price to shave down and retry.

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


def screen_candidates(min_market_cap: float = 2_000_000_000, top_movers: int = 100, narrow_to: int = 40) -> list[dict]:
    """Find underlyings worth researching this week - a tiered funnel, not
    a single flat pass: hundreds of liquid names get ranked by realized-vol
    and mean-reversion z-score from one bulk price pull (pure vectorized
    math, no per-symbol API call), then only the top `narrow_to` get real
    per-underlying options data (implied vol, VRP/NVRP, market cap,
    liquidity). Returns whatever clears the VRP/NVRP bar, NVRP-ranked.
    Material news is NOT checked here - call check_material_news yourself
    on the specific name(s) you're actually about to trade.
    """
    return wide_vectorized_screen(min_market_cap=min_market_cap, top_movers=top_movers, narrow_to=narrow_to).to_dict(orient="records")


def analyze_spread_quality(
    underlying: str, short_symbol: str, long_symbol: str, contracts: int, credit_per_spread: float
) -> dict:
    """Run the six-factor quality check on a candidate credit spread before
    proposing it: gamma structure, VRP, NVRP, IV-rank proxy, expected-move
    characteristics, and liquidity.
    """
    return spread_quality_report(underlying, short_symbol, long_symbol, contracts, credit_per_spread)


def check_mean_reversion(underlying: str) -> dict:
    """How far the stock's current price sits from its own trailing 20-day
    mean, in standard deviations - one of three things ALL required (in
    code, not just advisory) to justify a directional trade
    (propose_and_execute_credit_spread):
    1. `is_extreme` (|z_score| >= 3) - a genuinely large statistical
       deviation, not a routine move.
    2. `favorable_for_reversion` (regime == "choppy", via Kaufman's
       Efficiency Ratio) - mean reversion only tends to hold in a
       range-bound market; in a real trend, an extreme z-score is more
       often the trend continuing, not about to snap back.
    3. No material news (checked separately via `check_material_news`) -
       a real catalyst means the move may be a justified repricing, not
       a fluke likely to revert.
    `direction` tells you which spread type is consistent with the
    reversion thesis: "overbought" means the reversion bet is DOWN (sell
    a call spread), "oversold" means UP (sell a put spread) - the
    opposite pairing is rejected as a momentum bet.
    """
    return mean_reversion_signal(underlying)


def check_material_news(underlying: str) -> dict:
    """Keyword-based scan of recent news for genuinely high-materiality
    events (leadership changes, M&A, legal/regulatory action, guidance
    surprises, etc.) - required to be clear before trusting an extreme
    mean-reversion reading (`check_mean_reversion`) as a real setup rather
    than a justified repricing. `has_material_news: true` blocks
    propose_and_execute_credit_spread outright, enforced in code.
    """
    return material_news_check(underlying)


def _verify_vrp_edge(underlying: str, short_symbol: str, long_symbol: str, contracts: int, credit: float) -> Optional[str]:
    """Hard gate, shared by both execution tools: returns a REJECTED
    message if the actual edge this strategy exists to harvest - implied
    vol running rich relative to realized vol - isn't actually present,
    or None if it's confirmed and the caller should proceed.

    Previously VRP/NVRP was only ever advisory (the system prompt said to
    "favor" spreads showing it, via analyze_spread_quality), which meant
    every other check could pass - real credit, margin, buying power, no
    earnings risk - and the trade would still execute even with IV
    trading BELOW realized vol, i.e. the opposite of the edge this
    strategy is supposed to be capturing. Recomputed live here rather
    than trusting whatever analyze_spread_quality returned earlier in the
    session - same reasoning as the earnings-risk gate: an earlier
    reading can go stale, and a hard gate should verify for itself, not
    trust a self-report.
    """
    try:
        quality = spread_quality_report(underlying, short_symbol, long_symbol, contracts, credit)
    except ValueError as exc:
        # spread_quality_report raises (not returns None) when it can't
        # compute realized vol at all (not enough trailing price history)
        # or when a symbol isn't found in the current chain snapshot -
        # distinct from the "vrp is None" case below (a NaN implied_vol on
        # an otherwise-valid contract). Both are real, both mean the same
        # thing here: the edge can't be confirmed, so fail closed with a
        # clean rejection instead of letting this crash the whole call.
        return f"REJECTED: couldn't verify implied/realized volatility for {underlying} ({exc}) - can't confirm the VRP/NVRP edge this strategy requires."

    vrp = quality.get("vrp")
    if vrp is None:
        return (
            f"REJECTED: couldn't verify implied volatility for {short_symbol} to confirm VRP/NVRP - "
            f"this strategy exists to harvest implied vol running rich relative to realized vol, and "
            f"a trade can't be justified as that edge without being able to confirm it's actually there."
        )
    if not (vrp.get("vrp_elevated") or vrp.get("nvrp_high")):
        return (
            f"REJECTED: implied vol isn't elevated relative to realized vol for {underlying} "
            f"(vrp={vrp.get('vrp'):.3f}, nvrp={vrp.get('nvrp'):.3f}) - this strategy only harvests "
            f"rich premium/IV crush, and this setup doesn't show that edge regardless of how the "
            f"other signals look."
        )
    return None


def _verify_reversion_thesis(underlying: str, is_call: bool) -> Optional[str]:
    """Hard gate on the directional tool only: the ONLY thing that can
    justify a one-sided directional bet is a genuinely extreme (|z| >= 2,
    a 20-day SMA basis) statistical deviation, with the spread's direction
    actually consistent with reverting it - not any narrative the LLM
    supplies. Previously "a real, stated directional thesis" was pure
    prose with zero verification (confirmed: this pipeline picked put
    spreads 100% of the time with no analytical framework backing it up
    at all). Recomputed live here rather than trusting whatever
    check_mean_reversion returned earlier in the session, same reasoning
    as the other two hard gates.
    """
    try:
        signal = mean_reversion_signal(underlying)
    except ValueError as exc:
        return f"REJECTED: couldn't compute a mean-reversion z-score for {underlying} ({exc}) - can't confirm a real directional thesis without it."

    if not signal["is_extreme"]:
        return (
            f"REJECTED: {underlying}'s price is only {signal['z_score']} standard deviations from its "
            f"20-day mean (threshold is +/-{signal['z_threshold']}) - not extreme enough to justify a directional "
            f"bet. Use propose_and_execute_iron_condor instead if VRP/NVRP is the only real signal."
        )

    # A call spread is a bet the stock stays BELOW the short strike - only
    # consistent with an "overbought, expect reversion down" reading. A
    # put spread is the mirror: only consistent with "oversold, expect
    # reversion up". The other pairing would be betting WITH the extreme
    # move continuing, not against it reverting - a momentum bet dressed
    # up as a reversion thesis.
    expected_direction = "overbought" if is_call else "oversold"
    if signal["direction"] != expected_direction:
        return (
            f"REJECTED: {underlying} is {signal['direction']} (z={signal['z_score']}), but "
            f"{'a call spread' if is_call else 'a put spread'} bets on the opposite direction reverting - "
            f"that's a momentum bet, not a mean-reversion thesis."
        )

    # Mean reversion assumes price snaps back toward its average - that's
    # a choppy/range-bound-market assumption. In a real trend (high
    # Kaufman's Efficiency Ratio - net movement close to total movement,
    # not a lot of back-and-forth), an "extreme" z-score is often just the
    # trend continuing, not a reversion setup - the opposite read.
    if not signal["favorable_for_reversion"]:
        return (
            f"REJECTED: {underlying} is in a trending regime (efficiency ratio {signal['efficiency_ratio']}, "
            f">= {signal['chop_threshold']} threshold), not choppy/range-bound - an extreme z-score in a real trend "
            f"is more likely trend continuation than reversion, and this strategy only trades the reversion case."
        )

    if strategy_config.load()["news_gate_enabled"]:
        news = material_news_check(underlying)
        if news["has_material_news"]:
            headlines = [m["headline"] for m in news["matches"][:3]]
            disqualification_cache.record(underlying, f"material news: {headlines}")
            return (
                f"REJECTED: {underlying} has recent material news ({headlines}) that could explain this extreme "
                f"move as a real repricing rather than a statistical fluke - mean reversion assumes there's no "
                f"fundamental reason for the move, and this doesn't confirm that."
            )

    return None


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
        equity = await live_account_equity(mcp_session)
        max_margin = MAX_SPREAD_MARGIN_PCT_OF_EQUITY * equity
        if margin > max_margin:
            return (
                f"REJECTED: margin ${margin:,.0f} exceeds the per-trade cap of "
                f"{MAX_SPREAD_MARGIN_PCT_OF_EQUITY:.0%} of current equity (${max_margin:,.0f} of ${equity:,.0f})."
            )

        available = await live_options_buying_power(mcp_session)
        if margin > available:
            return (
                f"REJECTED: margin ${margin:,.0f} exceeds current available options buying power "
                f"${available:,.0f} - collateral already committed to other open positions leaves "
                f"less room than the equity-based per-trade cap alone would suggest."
            )

        vrp_rejection = _verify_vrp_edge(short_leg["underlying"], short_symbol, long_symbol, contracts, limit_credit)
        if vrp_rejection:
            return vrp_rejection

        # Directional-only: this tool is a one-sided bet, so on top of the
        # VRP edge (still selling rich premium either way) it also needs a
        # real, verified reason to lean one direction over the other -
        # otherwise there's nothing distinguishing this from just picking
        # a side arbitrarily.
        reversion_rejection = _verify_reversion_thesis(short_leg["underlying"], is_call)
        if reversion_rejection:
            return reversion_rejection

        earnings_gate_enabled = strategy_config.load()["earnings_gate_enabled"]
        earnings = earnings_before_expiration(short_leg["underlying"], short_leg["expiration"])
        acknowledges_earnings = "earnings" in rationale.lower()

        if earnings_gate_enabled and earnings["has_earnings_risk"] and not acknowledges_earnings:
            disqualification_cache.record(short_leg["underlying"], f"earnings before expiration: {earnings['earnings_dates']}")
            return (
                f"REJECTED: {short_leg['underlying']} reports earnings before this option expires "
                f"({earnings['earnings_dates']}) - an earnings move can blow through defined-risk "
                f"assumptions built on normal-day volatility. If this is a deliberate earnings play, "
                f"say so explicitly in the rationale (mention 'earnings') and resubmit."
            )
        if earnings["has_earnings_risk"] and not earnings_gate_enabled:
            earnings_note = f" [EARNINGS RISK PRESENT, GATE DISABLED: {earnings['earnings_dates']}]"
        elif earnings["has_earnings_risk"]:
            earnings_note = f" [ACKNOWLEDGED EARNINGS RISK: {earnings['earnings_dates']}]"
        elif not earnings["checked"]:
            earnings_note = " [EARNINGS RISK UNKNOWN - Finnhub couldn't be reached to verify; proceed with caution]"
        else:
            earnings_note = ""

        # Price off the live book right now rather than trusting limit_credit
        # as-is - it may reflect research from minutes earlier. See
        # compute_marketable_credit's docstring: this is what actually
        # reduces how often an order misses marketable and tempts a risky
        # manual resubmit, rather than just cleaning up after the fact.
        pricing = await compute_marketable_credit(mcp_session, [short_symbol], [long_symbol])
        if pricing is None:
            submit_credit = limit_credit
            pricing_note = " [live quote unavailable - used the requested limit as-is]"
        elif pricing["target"] < limit_credit:
            return (
                f"REJECTED: the live book won't support a ${limit_credit:.2f} credit right now - the best "
                f"currently-achievable price is ~${pricing['target']:.2f} (guaranteed-fillable "
                f"${pricing['guaranteed']:.2f}, fair mid ${pricing['fair_mid']:.2f}). The market has moved "
                f"since this was researched. Re-propose at ${pricing['target']:.2f} or lower if the setup "
                f"still holds, rather than submitting at a stale price."
            )
        else:
            submit_credit = pricing["target"]
            pricing_note = f" [priced off live book: guaranteed ${pricing['guaranteed']:.2f}, fair mid ${pricing['fair_mid']:.2f}, submitted ${submit_credit:.2f}]"

        submit_result = await mcp_session.call_tool(
            "place_option_order",
            {
                "qty": str(contracts),
                "type": "limit",
                "limit_price": str(-abs(submit_credit)),
                "legs": [
                    {"symbol": short_symbol, "ratio_qty": "1", "side": "sell", "position_intent": "sell_to_open"},
                    {"symbol": long_symbol, "ratio_qty": "1", "side": "buy", "position_intent": "buy_to_open"},
                ],
            },
        )
        submit_texts = [block.text for block in submit_result.content if hasattr(block, "text")]
        submit_payload = json.loads(submit_texts[0]) if submit_texts else {}
        order = unwrap(submit_payload)
        order_id = order.get("id") if isinstance(order, dict) else None

        if not order_id:
            return f"SUBMITTED but couldn't parse an order id to confirm the fill - not logged as open. Raw response: {submit_texts}"

        fill = await confirm_fill(mcp_session, order_id)

        if fill["status"] == "filled":
            record_open(short_symbol, long_symbol, short_leg["underlying"], contracts, resolved_credit(fill, submit_credit), short_leg["expiration"])
            return f"APPROVED AND FILLED (margin ${margin:,.0f}).{earnings_note}{pricing_note} {fill}"
        if fill["status"] == "partially_filled":
            record_open(short_symbol, long_symbol, short_leg["underlying"], contracts, resolved_credit(fill, submit_credit), short_leg["expiration"])
            return (
                f"PARTIALLY FILLED (margin ${margin:,.0f}).{earnings_note}{pricing_note} logged, but confirm the actual filled size "
                f"via get_order_by_id before assuming the full {contracts} contracts are on. {fill}"
            )
        if fill["status"] in ("canceled", "rejected", "expired"):
            return f"NOT FILLED - order {fill['status']}. Nothing logged as open. {fill}"
        if fill["status"] == "canceled_after_timeout":
            return (
                f"NOT FILLED - the limit price never became marketable within {120}s, so this order has "
                f"been actively CANCELED (not left pending). Nothing logged as open, and there is no live "
                f"order left on this structure - it is now completely safe to resubmit at a different "
                f"price without any risk of double-filling. {fill}"
            )

        return (
            f"SUBMITTED but its true status is unknown after polling and a failed cancel attempt "
            f"(order_id={order_id}, status={fill['status']}). NOT logged as open. Do NOT resubmit this "
            f"exact structure - check get_order_by_id('{order_id}') first to see whether it filled, since "
            f"a duplicate could double the position. {fill}"
        )

    return propose_and_execute_credit_spread


def make_propose_and_execute_condor(mcp_session: ClientSession):
    """A genuinely direction-neutral alternative to propose_and_execute_credit_spread.

    That tool only ever sells ONE side (a put spread or a call spread) -
    which requires guessing a direction, even though the actual edge being
    screened for (VRP/NVRP: implied vol rich vs realized vol) says nothing
    about direction at all. This is the structure that actually matches
    that signal: sell a put spread AND a call spread together as one
    4-leg iron condor, profiting as long as the stock stays between BOTH
    strikes - direction-agnostic by construction, not by hoping the LLM
    guesses the safer side.

    Confirmed via Alpaca's own multi-leg docs before building this: 4-leg
    "mleg" orders are fully supported, real margin uses the "universal
    spread rule" (worst case across ALL legs together, not each 2-leg
    spread's margin summed) - since the stock can't finish both below the
    put wing and above the call wing at once, real margin is
    max(put_width, call_width) minus the TOTAL credit from both sides, not
    the sum of two independent spreads' margins.
    """

    async def propose_and_execute_iron_condor(
        long_put_symbol: str,
        short_put_symbol: str,
        short_call_symbol: str,
        long_call_symbol: str,
        contracts: int,
        limit_credit: float,
        rationale: str,
    ) -> str:
        try:
            long_put = parse_occ_symbol(long_put_symbol)
            short_put = parse_occ_symbol(short_put_symbol)
            short_call = parse_occ_symbol(short_call_symbol)
            long_call = parse_occ_symbol(long_call_symbol)
        except ValueError as exc:
            return f"REJECTED: {exc}"

        legs = [long_put, short_put, short_call, long_call]
        underlying = long_put["underlying"]
        expiration = long_put["expiration"]
        if any(leg["underlying"] != underlying for leg in legs):
            return "REJECTED: all four legs must share the same underlying."
        if any(leg["expiration"] != expiration for leg in legs):
            return "REJECTED: all four legs must share the same expiration."
        if long_put["option_type"] != "put" or short_put["option_type"] != "put":
            return "REJECTED: long_put_symbol and short_put_symbol must both be puts."
        if short_call["option_type"] != "call" or long_call["option_type"] != "call":
            return "REJECTED: short_call_symbol and long_call_symbol must both be calls."
        if not (long_put["strike"] < short_put["strike"] < short_call["strike"] < long_call["strike"]):
            return (
                "REJECTED: strikes must be strictly ordered long_put < short_put < short_call < long_call "
                "- this isn't a valid iron condor shape."
            )
        if limit_credit <= 0:
            return "REJECTED: limit_credit must be positive - this tool only places net-credit condors."

        put_width = short_put["strike"] - long_put["strike"]
        call_width = long_call["strike"] - short_call["strike"]
        margin = max(0.0, max(put_width, call_width) * 100 * contracts - limit_credit * 100 * contracts)
        equity = await live_account_equity(mcp_session)
        max_margin = MAX_SPREAD_MARGIN_PCT_OF_EQUITY * equity
        if margin > max_margin:
            return (
                f"REJECTED: margin ${margin:,.0f} exceeds the per-trade cap of "
                f"{MAX_SPREAD_MARGIN_PCT_OF_EQUITY:.0%} of current equity (${max_margin:,.0f} of ${equity:,.0f})."
            )

        available = await live_options_buying_power(mcp_session)
        if margin > available:
            return (
                f"REJECTED: margin ${margin:,.0f} exceeds current available options buying power "
                f"${available:,.0f} - collateral already committed to other open positions leaves "
                f"less room than the equity-based per-trade cap alone would suggest."
            )

        # Checked once, on the put side - both sides share the same
        # underlying/expiration, so this reflects the same IV regime
        # either wing would show; no need to compute it twice.
        vrp_rejection = _verify_vrp_edge(underlying, short_put_symbol, long_put_symbol, contracts, limit_credit / 2)
        if vrp_rejection:
            return vrp_rejection

        # Previously only the directional path (_verify_reversion_thesis)
        # checked this - a condor has no directional thesis for news to
        # invalidate, but a real catalyst (M&A, leadership change, guidance
        # surprise, legal action) can move a stock clean through a wing
        # sized for normal-day volatility just as easily as it can for a
        # single-leg spread. No override via the rationale, unlike the
        # earnings gate below - earnings is a known, scheduled date that
        # can be deliberately traded around; material news is unscheduled
        # and reactive, so letting the model talk its way past a real
        # leadership/legal/M&A event with an acknowledgment is a real gap,
        # not a reasonable exception.
        gate_cfg = strategy_config.load()
        if gate_cfg["news_gate_enabled"]:
            news = material_news_check(underlying)
            if news["has_material_news"]:
                headlines = [m["headline"] for m in news["matches"][:3]]
                disqualification_cache.record(underlying, f"material news: {headlines}")
                return (
                    f"REJECTED: {underlying} has recent material news ({headlines}) - a real catalyst can move "
                    f"a stock through a wing sized for normal-day volatility. Selling premium into an active "
                    f"catalyst is a gamble regardless of structure; pick a different underlying."
                )

        earnings = earnings_before_expiration(underlying, expiration)
        acknowledges_earnings = "earnings" in rationale.lower()
        if gate_cfg["earnings_gate_enabled"] and earnings["has_earnings_risk"] and not acknowledges_earnings:
            disqualification_cache.record(underlying, f"earnings before expiration: {earnings['earnings_dates']}")
            return (
                f"REJECTED: {underlying} reports earnings before this option expires "
                f"({earnings['earnings_dates']}) - an earnings move can blow through both wings of a "
                f"defined-risk structure sized for normal-day volatility. If this is a deliberate "
                f"earnings play, say so explicitly in the rationale (mention 'earnings') and resubmit."
            )
        if earnings["has_earnings_risk"] and not gate_cfg["earnings_gate_enabled"]:
            earnings_note = f" [EARNINGS RISK PRESENT, GATE DISABLED: {earnings['earnings_dates']}]"
        elif earnings["has_earnings_risk"]:
            earnings_note = f" [ACKNOWLEDGED EARNINGS RISK: {earnings['earnings_dates']}]"
        elif not earnings["checked"]:
            earnings_note = " [EARNINGS RISK UNKNOWN - Finnhub couldn't be reached to verify; proceed with caution]"
        else:
            earnings_note = ""

        # Price off the live book right now, same reasoning as the
        # directional tool - see compute_marketable_credit's docstring.
        pricing = await compute_marketable_credit(
            mcp_session, [short_put_symbol, short_call_symbol], [long_put_symbol, long_call_symbol]
        )
        if pricing is None:
            submit_credit = limit_credit
            pricing_note = " [live quote unavailable - used the requested limit as-is]"
        elif pricing["target"] < limit_credit:
            return (
                f"REJECTED: the live book won't support a ${limit_credit:.2f} combined credit right now - "
                f"the best currently-achievable price is ~${pricing['target']:.2f} (guaranteed-fillable "
                f"${pricing['guaranteed']:.2f}, fair mid ${pricing['fair_mid']:.2f}). The market has moved "
                f"since this was researched. Re-propose at ${pricing['target']:.2f} or lower if the setup "
                f"still holds, rather than submitting at a stale price."
            )
        else:
            submit_credit = pricing["target"]
            pricing_note = f" [priced off live book: guaranteed ${pricing['guaranteed']:.2f}, fair mid ${pricing['fair_mid']:.2f}, submitted ${submit_credit:.2f}]"

        submit_result = await mcp_session.call_tool(
            "place_option_order",
            {
                "qty": str(contracts),
                "type": "limit",
                "limit_price": str(-abs(submit_credit)),
                "order_class": "mleg",
                "legs": [
                    {"symbol": long_put_symbol, "ratio_qty": "1", "side": "buy", "position_intent": "buy_to_open"},
                    {"symbol": short_put_symbol, "ratio_qty": "1", "side": "sell", "position_intent": "sell_to_open"},
                    {"symbol": short_call_symbol, "ratio_qty": "1", "side": "sell", "position_intent": "sell_to_open"},
                    {"symbol": long_call_symbol, "ratio_qty": "1", "side": "buy", "position_intent": "buy_to_open"},
                ],
            },
        )
        submit_texts = [block.text for block in submit_result.content if hasattr(block, "text")]
        submit_payload = json.loads(submit_texts[0]) if submit_texts else {}
        order = unwrap(submit_payload)
        order_id = order.get("id") if isinstance(order, dict) else None

        if not order_id:
            return f"SUBMITTED but couldn't parse an order id to confirm the fill - not logged as open. Raw response: {submit_texts}"

        fill = await confirm_fill(mcp_session, order_id)

        if fill["status"] in ("filled", "partially_filled"):
            # Split the combined credit into a real per-side amount using
            # each leg's own filled_avg_price (confirmed live this is
            # present on a real filled multi-leg order) rather than
            # assuming an even 50/50 split between the put and call sides,
            # which real markets essentially never actually give you.
            detail_result = await mcp_session.call_tool("get_order_by_id", {"order_id": order_id})
            detail_texts = [block.text for block in detail_result.content if hasattr(block, "text")]
            order_detail = unwrap(json.loads(detail_texts[0])) if detail_texts else {}
            leg_prices = {leg.get("symbol"): leg.get("filled_avg_price") for leg in (order_detail.get("legs") or [])}

            total_credit = resolved_credit(fill, submit_credit)

            def _leg_price(sym: str, fallback: float) -> float:
                raw = leg_prices.get(sym)
                try:
                    return float(raw) if raw is not None else fallback
                except (TypeError, ValueError):
                    return fallback

            long_put_px = _leg_price(long_put_symbol, total_credit / 4)
            short_put_px = _leg_price(short_put_symbol, total_credit / 4)
            short_call_px = _leg_price(short_call_symbol, total_credit / 4)
            long_call_px = _leg_price(long_call_symbol, total_credit / 4)
            put_credit = short_put_px - long_put_px
            call_credit = short_call_px - long_call_px

            condor_id = str(uuid.uuid4())
            record_open(short_put_symbol, long_put_symbol, underlying, contracts, put_credit, expiration, group_id=condor_id)
            record_open(short_call_symbol, long_call_symbol, underlying, contracts, call_credit, expiration, group_id=condor_id)

            status_label = "APPROVED AND FILLED" if fill["status"] == "filled" else "PARTIALLY FILLED"
            note = "" if fill["status"] == "filled" else " - confirm actual filled size via get_order_by_id before assuming the full size is on."
            return (
                f"{status_label} (margin ${margin:,.0f}, put side credit ${put_credit:.3f}, "
                f"call side credit ${call_credit:.3f}).{earnings_note}{pricing_note}{note} {fill}"
            )
        if fill["status"] in ("canceled", "rejected", "expired"):
            return f"NOT FILLED - order {fill['status']}. Nothing logged as open. {fill}"
        if fill["status"] == "canceled_after_timeout":
            return (
                f"NOT FILLED - the limit price never became marketable within {120}s, so this order has "
                f"been actively CANCELED (not left pending). Nothing logged as open, and there is no live "
                f"order left on this structure - it is now completely safe to resubmit at a different "
                f"price without any risk of double-filling. {fill}"
            )

        return (
            f"SUBMITTED but its true status is unknown after polling and a failed cancel attempt "
            f"(order_id={order_id}, status={fill['status']}). NOT logged as open. Do NOT resubmit this "
            f"exact structure - check get_order_by_id('{order_id}') first to see whether it filled, since "
            f"a duplicate could double the position. {fill}"
        )

    return propose_and_execute_iron_condor


def make_close_position(mcp_session: ClientSession):
    async def close_position(symbol: str, rationale: str) -> str:
        outcome = await close_both_legs(mcp_session, symbol)
        # record_close on every closed symbol, not just the one originally
        # requested - for a plain 2-leg spread this just calls it twice on
        # the same trade record (harmless, record_close is idempotent), but
        # for a 4-leg condor (two linked trade_log records) this is what
        # marks BOTH records closed instead of leaving the sibling record
        # looking open when its real legs have already been closed.
        for sym in outcome["closed_symbols"]:
            record_close(sym, realized_pnl=outcome["realized_pnl_by_symbol"].get(sym))
        summary = f"Closed {outcome['closed_symbols']} ({rationale}). Results: {outcome['results']}"
        if outcome["errors"]:
            summary += f" ERRORS on some legs (may already be closed): {outcome['errors']}"
        return summary

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
                    "top_movers": {"type": "integer", "description": "How many of Alpaca's daily most-actives/gainers/losers to fold into the wide universe, on top of the ~116-name curated liquid-options list that's always included."},
                    "narrow_to": {"type": "integer", "description": "How many top names (by vol-rank proxy) get real per-underlying options data fetched - anything with an extreme mean-reversion z-score is always included in this group too, regardless of vol rank."},
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
                        "description": "Minimum acceptable net credit per spread, in dollars per share (must be positive). The tool re-prices off live quotes right before submitting - it may submit at a BETTER credit than this if the live book supports it, but REJECTS instead of submitting worse than this floor.",
                    },
                    "rationale": {"type": "string", "description": "One or two sentences on why this trade makes sense right now, for the trade log."},
                },
                "required": ["short_symbol", "long_symbol", "contracts", "limit_credit", "rationale"],
            },
        },
    },
    "propose_and_execute_iron_condor": {
        "type": "function",
        "function": {
            "name": "propose_and_execute_iron_condor",
            "description": (
                "Place a genuinely direction-neutral 4-leg iron condor: sell short_put_symbol and "
                "short_call_symbol, each protected by long_put_symbol/long_call_symbol further out of "
                "the money, for a combined net credit of limit_credit. Profits as long as the stock "
                "stays between the two short strikes at expiration - unlike "
                "propose_and_execute_credit_spread, this doesn't require any directional judgment, "
                "since it collects premium on both sides at once. Submitted as a single atomic 4-leg "
                "order (fills completely or not at all). Hard-gated on VRP/NVRP, market cap, buying "
                "power, and material news (no override) - REJECTED if any fail, nothing submitted."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "long_put_symbol": {"type": "string", "description": "OCC put symbol to buy - the lowest strike, furthest OTM downside protection."},
                    "short_put_symbol": {"type": "string", "description": "OCC put symbol to sell - strike must be above long_put_symbol's."},
                    "short_call_symbol": {"type": "string", "description": "OCC call symbol to sell - strike must be above short_put_symbol's."},
                    "long_call_symbol": {"type": "string", "description": "OCC call symbol to buy - the highest strike, furthest OTM upside protection."},
                    "contracts": {"type": "integer", "description": "Number of condors."},
                    "limit_credit": {"type": "number", "description": "Minimum acceptable combined net credit for all 4 legs together, in dollars per share (must be positive). The tool re-prices off live quotes right before submitting - it may submit at a BETTER credit than this if the live book supports it, but REJECTS instead of submitting worse than this floor."},
                    "rationale": {"type": "string", "description": "One or two sentences on why this trade makes sense right now, for the trade log."},
                },
                "required": ["long_put_symbol", "short_put_symbol", "short_call_symbol", "long_call_symbol", "contracts", "limit_credit", "rationale"],
            },
        },
    },
    "close_position": {
        "type": "function",
        "function": {
            "name": "close_position",
            "description": "Close (or reduce) an existing position - the only way to exit a trade before expiration rather than holding it to settlement. Closing either leg of a logged spread automatically closes its paired leg too.",
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
    "check_mean_reversion": {
        "type": "function",
        "function": {
            "name": "check_mean_reversion",
            "description": (
                "How far the stock's current price sits from its own trailing 20-day mean, in standard "
                "deviations - one of three things ALL required in code to justify "
                "propose_and_execute_credit_spread: is_extreme (|z| >= 2), favorable_for_reversion "
                "(regime=='choppy', not trending - mean reversion only holds in a range-bound market), "
                "and no material news (check separately via check_material_news). direction='overbought' "
                "means the reversion bet is DOWN (sell a call spread); 'oversold' means UP (sell a put "
                "spread) - the opposite pairing is rejected."
            ),
            "parameters": {
                "type": "object",
                "properties": {"underlying": {"type": "string", "description": "Underlying ticker, e.g. AAPL."}},
                "required": ["underlying"],
            },
        },
    },
    "check_material_news": {
        "type": "function",
        "function": {
            "name": "check_material_news",
            "description": (
                "Keyword scan of recent news for genuinely high-materiality events (leadership changes, "
                "M&A, legal/regulatory action, guidance surprises, etc). Required before trusting an "
                "extreme mean-reversion reading as a real setup - has_material_news: true blocks "
                "propose_and_execute_credit_spread outright, enforced in code."
            ),
            "parameters": {
                "type": "object",
                "properties": {"underlying": {"type": "string", "description": "Underlying ticker, e.g. AAPL."}},
                "required": ["underlying"],
            },
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
    log_event("session", text=user_message)
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
            condor_tool = make_propose_and_execute_condor(mcp_session)
            close_tool = make_close_position(mcp_session)
            evaluate_tool = make_evaluate_positions(mcp_session)
            dispatch = {
                "screen_candidates": screen_candidates,
                "analyze_spread_quality": analyze_spread_quality,
                "check_mean_reversion": check_mean_reversion,
                "check_material_news": check_material_news,
                "propose_and_execute_credit_spread": gated_execution_tool,
                "propose_and_execute_iron_condor": condor_tool,
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
                if response.usage is not None:
                    try:
                        record_token_usage(model, response.usage.model_dump())
                    except Exception as exc:
                        print(f"[warning] failed to record token usage: {exc!r}")
                message = response.choices[0].message

                if not message.tool_calls:
                    final_text = message.content or final_text
                    print(f"[assistant] {final_text}")
                    log_event("assistant", text=final_text)
                    break

                messages.append(message.model_dump(exclude_none=True))

                for call in message.tool_calls:
                    name = call.function.name
                    try:
                        args = json.loads(call.function.arguments) if call.function.arguments else {}
                    except json.JSONDecodeError:
                        args = {}
                    print(f"[tool call] {name}({args})")
                    log_event("tool_call", name=name, args=args)

                    if name in dispatch:
                        func = dispatch[name]
                        result: Any = await func(**args) if asyncio.iscoroutinefunction(func) else func(**args)
                    elif name in read_only_names:
                        mcp_result = await mcp_session.call_tool(name, args)
                        result = [block.text for block in mcp_result.content if hasattr(block, "text")]
                    else:
                        result = f"ERROR: '{name}' is not an authorized tool."

                    log_event("tool_result", name=name, result=_truncate(result))
                    messages.append(
                        {"role": "tool", "tool_call_id": call.id, "content": json.dumps(result, default=str)}
                    )
            else:
                print("[warning] hit the max loop-turn cap without a final answer")
                log_event("error", text=f"Hit the {MAX_LOOP_TURNS}-turn cap without a final answer - session ended mid-research.", source="agent")

            return final_text


if __name__ == "__main__":
    import sys

    prompt = " ".join(sys.argv[1:]) or (
        "Check the account, then screen for candidate underlyings this week. "
        "If there's a reasonable defined-risk credit spread opportunity, "
        "propose and execute it. Otherwise, explain why you're passing."
    )
    print(asyncio.run(run(prompt)))
