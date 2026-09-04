"""Candidate universe screener: Alpaca's own most-active/movers lists,
filtered down to names actually worth running the strategy on.

Alpaca has no market cap data (see data/fundamentals.py - confirmed
directly against both the screener response and get_asset), so the
market-cap filter uses Finnhub; without a key, this falls back to a
curated list of known large-cap, liquid-options names rather than letting
an unverified small-cap through unchecked.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from alpaca.data.enums import MostActivesBy
from alpaca.data.historical.screener import ScreenerClient
from alpaca.data.requests import MarketMoversRequest, MostActivesRequest

from data.clients import _API_KEY, _SECRET_KEY
from data.equities import get_bulk_stock_bars, latest_spot
from data.fundamentals import market_cap
from data.options import get_option_chain, has_liquid_weekly_options
from signals.disqualification_cache import recent as recently_disqualified
from signals.mean_reversion import get_chop_threshold, get_z_threshold
from signals.options_quality import realized_vol_rank, realized_volatility, variance_risk_premium

# Known large-cap, reliably-liquid-weekly-options names - used when no
# FINNHUB_API_KEY is set (an arbitrary screener hit's market cap can't be
# verified then), and always checked alongside Alpaca's daily movers.
DEFAULT_LARGE_CAP_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO",
    "JPM", "V", "UNH", "XOM", "WMT", "COST", "NFLX", "AMD",
]

# A much wider curated set of liquid, large/mid-cap, reliably-optionable
# names, spanning sectors - this is what actually lets wide_vectorized_screen
# cover hundreds of names cheaply: every member here skips the per-symbol
# Finnhub market-cap lookup entirely (same reasoning as the smaller list
# above), so widening THIS list is what turns "hundreds of bulk-fetched
# bars" into "hundreds of candidates screened" without hundreds of
# rate-limited fundamentals calls. Alpaca's own movers/actives lists still
# supplement this for catching real breakout-vol names outside it.
WIDE_LIQUID_UNIVERSE = sorted(set(DEFAULT_LARGE_CAP_UNIVERSE + [
    # Mega-cap tech / growth
    "GOOG", "CRM", "ORCL", "ADBE", "INTC", "QCOM", "TXN", "MU", "INTU",
    "NOW", "PANW", "SNOW", "PLTR", "SHOP", "UBER", "ABNB", "NET", "CRWD",
    "DDOG", "MDB", "TEAM", "WDAY", "PYPL", "SQ", "COIN", "MSTR", "SMCI",
    "ARM", "DELL", "IBM", "CSCO", "HPQ", "ZM", "DOCU", "ROKU", "SNAP", "PINS",
    # Financials
    "BAC", "WFC", "GS", "MS", "C", "SCHW", "AXP", "BLK",
    # Healthcare
    "LLY", "JNJ", "ABBV", "MRK", "PFE", "TMO", "ABT", "DHR", "BMY", "GILD",
    "AMGN", "CVS", "CI", "HUM", "ISRG", "VRTX", "REGN", "MRNA",
    # Consumer
    "HD", "LOW", "TGT", "NKE", "SBUX", "MCD", "DIS", "CMCSA", "T", "VZ", "TMUS",
    # Industrials / energy
    "CVX", "COP", "OXY", "SLB", "BA", "CAT", "DE", "GE", "HON", "UPS", "RTX",
    "LMT", "NOC",
    # Media / other liquid optionable names (incl. high-options-volume
    # meme-adjacent and EV/crypto-linked names - liquid weekly options
    # regardless of view on the underlying business)
    "WBD", "PARA", "GME", "AMC", "RIVN", "LCID", "F", "GM", "SOFI",
    "RIOT", "MARA", "CVNA", "DKNG",
]))

_screener_client = ScreenerClient(_API_KEY, _SECRET_KEY)

# Broad-market/sector/leveraged ETFs and bond funds - reliably among
# Alpaca's daily most-actives (they're some of the highest-volume tickers
# on the market on any given day) but out of scope for this strategy:
# material-news and earnings-date gates are built around single-company
# events and don't mean anything for an index fund. Not exhaustive - a
# denylist for the specific names that actually showed up polluting real
# screen output, not a general ETF classifier (Alpaca's asset data doesn't
# cleanly distinguish ETFs from equities to filter this programmatically).
_ETF_EXCLUDE = {
    "SPY", "SPYM", "SPDN", "SPXU", "SPXL", "SPXS", "VOO", "VTI", "IVV", "DIA",
    "QQQ", "QQQM", "TQQQ", "SQQQ", "IWM", "TNA", "TZA",
    "SOXL", "SOXS", "SOXX", "UVXY", "VXX", "SVXY",
    "HYG", "LQD", "TLT", "TMF", "TMV", "SGOV", "BIL", "SHY", "AGG", "BND",
    "XLF", "XLE", "XLK", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLC", "XLRE",
    "GLD", "SLV", "USO", "UNG",
}


def _raw_screener_candidates(top: int = 20) -> list[str]:
    actives = _screener_client.get_most_actives(MostActivesRequest(by=MostActivesBy.VOLUME, top=top))
    movers = _screener_client.get_market_movers(MarketMoversRequest(top=max(top // 2, 1)))
    symbols = [a.symbol for a in actives.most_actives]
    symbols += [m.symbol for m in movers.gainers] + [m.symbol for m in movers.losers]
    deduped = list(dict.fromkeys(symbols))  # dedupe, preserve order
    return [s for s in deduped if s not in _ETF_EXCLUDE]


def _week_friday(as_of: date) -> date:
    monday = as_of - timedelta(days=as_of.weekday())
    return monday + timedelta(days=4)


def candidate_universe(min_market_cap: float = 2_000_000_000, top: int = 20) -> list[str]:
    """Alpaca's daily movers, filtered to a market-cap floor and liquid
    weekly options. Without FINNHUB_API_KEY, market cap can't be verified
    for arbitrary tickers, so anything not on DEFAULT_LARGE_CAP_UNIVERSE
    is skipped rather than let through unchecked.
    """
    raw = _raw_screener_candidates(top)
    expiration = _week_friday(date.today())

    kept = []
    for symbol in raw:
        cap = market_cap(symbol)
        if cap is not None:
            if cap < min_market_cap:
                continue
        elif symbol not in DEFAULT_LARGE_CAP_UNIVERSE:
            continue  # can't verify cap and it's not on the known-safe list

        if has_liquid_weekly_options(symbol, expiration):
            kept.append(symbol)

    # Always consider the curated universe too, even if it didn't happen
    # to show up in today's movers/actives.
    for symbol in DEFAULT_LARGE_CAP_UNIVERSE:
        if symbol not in kept and has_liquid_weekly_options(symbol, expiration):
            kept.append(symbol)

    return kept


def _atm_implied_vol(symbol: str, expiration: date, spot: float) -> Optional[float]:
    chain = get_option_chain(symbol, expiration_date=expiration, strike_price_gte=spot * 0.95, strike_price_lte=spot * 1.05)
    chain = chain.dropna(subset=["implied_volatility"])
    if chain.empty:
        return None
    row = chain.iloc[(chain["strike"] - spot).abs().argsort()[:1]].iloc[0]
    return float(row["implied_volatility"])


def rank_by_vol_signal(symbols: list[str], as_of: Optional[date] = None) -> pd.DataFrame:
    """For each candidate: realized-vol rank (the IV-rank proxy from
    signals/options_quality.py) and VRP/NVRP off the nearest weekly ATM
    contract - ranked by NVRP descending, the direct "how rich is premium
    right now" read.
    """
    as_of = as_of or date.today()
    expiration = _week_friday(as_of)
    rows = []
    for symbol in symbols:
        try:
            spot = latest_spot(symbol)
            iv = _atm_implied_vol(symbol, expiration, spot)
            rv = realized_volatility(symbol, as_of)
            rank = realized_vol_rank(symbol, as_of)
        except (ValueError, KeyError):
            continue
        if iv is None:
            continue
        row = {"symbol": symbol, "spot": spot, "vol_rank_pct": rank["rank_pct"]}
        row.update(variance_risk_premium(iv, rv))
        rows.append(row)

    columns = ["symbol", "spot", "implied_vol", "realized_vol", "vrp", "nvrp", "vrp_elevated", "nvrp_high", "vol_rank_pct"]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows)[columns].sort_values("nvrp", ascending=False).reset_index(drop=True)


_MIN_DOLLAR_VOLUME = 3_000_000  # cheap junk filter, not the real liquidity gate


def _vectorized_vol_and_reversion(bars: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Pure math over an already-bulk-fetched price panel - zero API calls,
    zero LLM calls, one pandas pass over however many symbols were pulled.

    Deliberately reimplements the same z-score/efficiency-ratio definitions
    as signals/mean_reversion.py rather than importing its per-symbol
    function - that module's version is a live hard-gate recomputed fresh
    per-symbol at trade time (safety-critical, independently tested) and
    should stay untouched by this wide, approximate prefilter. Only the
    THRESHOLDS (Z_THRESHOLD, CHOP_THRESHOLD) are shared, so a future "change
    the bar" edit can't silently drift between the two.

    realized_vol_rank_pct here is a cheaper cousin of
    signals/options_quality.realized_vol_rank: ranked against its own
    trailing history WITHIN the pulled window (~1 year), not a separate
    252-day pull per symbol - good enough to shortlist candidates for the
    real, per-symbol IV/VRP check that follows, not a replacement for it.

    Confirmed live why a liquidity floor has to happen HERE, not just
    downstream in _enrich's has_liquid_weekly_options check: Alpaca's raw
    movers/actives lists are full of dead warrants (ACONW, CCGWW, and a
    dozen others - confirmed zero trading volume for days, a flat stale
    price) that produce degenerate z-scores and "100% vol decay" purely
    from near-zero-variance noise. _enrich would eventually reject these,
    but not before they'd already crowded real candidates out of the
    capped shortlist slots (head(narrow_to), head(20) for decay) - garbage
    ranked ahead of signal is worse than garbage rejected late.
    """
    z_threshold = get_z_threshold()
    chop_threshold = get_chop_threshold()

    rows = []
    for symbol, group in bars.groupby(level=0):
        closes = group["close"].reset_index(level=0, drop=True)
        volumes = group["volume"].reset_index(level=0, drop=True)
        if len(closes) < window + 25:
            continue  # not enough history in this window to rank vol reliably

        avg_dollar_volume = float((closes.tail(window) * volumes.tail(window)).mean())
        if avg_dollar_volume < _MIN_DOLLAR_VOLUME:
            continue  # not really trading - any z-score/vol reading on this is noise, not signal

        recent = closes.tail(window + 1)
        trailing = recent.iloc[:-1]
        current_price = float(recent.iloc[-1])
        sma, std = float(trailing.mean()), float(trailing.std())
        if std == 0:
            continue
        z_score = (current_price - sma) / std

        net_change = abs(float(trailing.iloc[-1]) - float(trailing.iloc[0]))
        total_movement = float(trailing.diff().abs().sum())
        efficiency_ratio = net_change / total_movement if total_movement > 0 else 0.0
        regime = "trending" if efficiency_ratio >= chop_threshold else "choppy"

        log_returns = np.log(closes / closes.shift(1)).dropna()
        rolling_vol = (log_returns.rolling(window).std() * (252**0.5)).dropna()
        if len(rolling_vol) < 20:
            continue
        current_vol = float(rolling_vol.iloc[-1])
        vol_rank_pct = float((rolling_vol.iloc[:-1] < current_vol).mean()) * 100

        # A separate signal from "high CURRENT vol": how much realized vol
        # has recently FALLEN from a genuinely elevated level. IV tends to
        # lag realized vol on the way down (the same lag that makes
        # post-earnings IV crush work) - a name whose vol just came down
        # can carry excellent VRP/NVRP right now precisely because IV
        # hasn't caught down yet, even though its CURRENT vol-rank looks
        # unremarkable. Filtering the shortlist by current vol-rank alone
        # would systematically miss this - the classic VRP setup, not an
        # edge case.
        #
        # Two guards, both added after this blew up against real data on
        # the first pass: (1) a >=30% annualized floor on the "before"
        # reading - without it, an illiquid near-zero-volume warrant going
        # from "barely trades" to "still barely trades" computed as a
        # meaningless 100% "decay" (confirmed live: ACONW, CCGWW, SQFTW and
        # a dozen other thinly-traded warrants dominated the raw ranking);
        # (2) averaging a small window centered ~15 trading days ago rather
        # than one single day, since one extreme day rolling out of the
        # 20-day window can itself cause a big mechanical drop unrelated to
        # any real change in vol regime.
        vol_baseline = float(rolling_vol.iloc[-20:-12].mean()) if len(rolling_vol) >= 20 else None
        vol_decay_pct = (
            (vol_baseline - current_vol) / vol_baseline if vol_baseline is not None and vol_baseline >= 0.30 else None
        )

        rows.append(
            {
                "symbol": symbol,
                "current_price": current_price,
                "z_score": round(z_score, 2),
                "is_extreme": abs(z_score) >= z_threshold,
                "direction": "overbought" if z_score >= z_threshold else "oversold" if z_score <= -z_threshold else "normal",
                "efficiency_ratio": round(efficiency_ratio, 3),
                "regime": regime,
                "favorable_for_reversion": regime == "choppy",
                "realized_vol_20d": round(current_vol, 4),
                "vol_rank_proxy_pct": round(vol_rank_pct, 1),
                "vol_decay_pct": round(vol_decay_pct, 3) if vol_decay_pct is not None else None,
            }
        )

    return pd.DataFrame(rows)


def wide_vectorized_screen(top_movers: int = 100, narrow_to: int = 40, min_market_cap: float = 2_000_000_000) -> pd.DataFrame:
    """The actual "check hundreds of stocks at once" entry point - a tiered
    funnel, cheap-and-wide to expensive-and-narrow:

    Stage 0 (hundreds of names, ONE bulk API call, vectorized pandas math,
    zero LLM involvement): every name in WIDE_LIQUID_UNIVERSE plus today's
    real movers/actives gets a realized-vol-rank and mean-reversion
    z-score/regime from one bulk price-history pull.

    Stage 1 (at least the top `narrow_to` by vol-rank, plus anything with
    an extreme z-score, plus anything whose vol just fell sharply off a
    recent high - three different, all legitimate, reasons to look closer;
    the shortlist can run a bit past `narrow_to` on a day when a lot of
    names qualify on the latter two): real per-underlying data that
    genuinely can't be bulked - implied vol off the actual options chain,
    market cap for anything not already on the curated safe list, liquid-
    weekly-options - fetched CONCURRENTLY, since one underlying's chain
    lookup has no dependency on any other's.

    Returns only names that actually clear the VRP/NVRP bar, ranked by
    NVRP. Deliberately stops there - material news is NOT checked here.
    Whether a headline is genuinely disqualifying isn't arithmetic, it's a
    judgment call that needs the actual text read and weighed, so it stays
    a call the agent makes itself (`check_material_news`) on the one or two
    names it's actually about to trade, not a boolean pre-computed here for
    every numeric survivor.
    """
    universe = sorted(set(WIDE_LIQUID_UNIVERSE) | set(_raw_screener_candidates(top_movers)))

    today = date.today()
    bars = get_bulk_stock_bars(universe, datetime.combine(today - timedelta(days=400), datetime.min.time()), datetime.combine(today, datetime.min.time()))
    ranked = _vectorized_vol_and_reversion(bars)
    if ranked.empty:
        return ranked

    by_vol_rank = ranked.sort_values("vol_rank_proxy_pct", ascending=False)
    decaying_vol = ranked[ranked["vol_decay_pct"].fillna(0) >= 0.25].sort_values("vol_decay_pct", ascending=False).head(20)
    shortlist = (
        set(by_vol_rank.head(narrow_to)["symbol"])
        | set(ranked.loc[ranked["is_extreme"], "symbol"])
        | set(decaying_vol["symbol"])
    )

    # Skip names that just failed a hard, time-stable gate (material news,
    # earnings) within the cooldown window - a scheduled earnings date or a
    # real news event doesn't change minutes later, so re-running the full
    # per-symbol enrichment (market cap, options chain, liquidity - real API
    # calls) on a name already known to be disqualified is pure waste, and
    # surfacing it again just tempts the agent into re-deriving a
    # conclusion it already reached last session instead of looking at
    # something new. Only ever filters what gets SHOWN here - the actual
    # execution-time hard gates in propose_and_execute_* always re-check
    # fresh regardless, no exceptions, since that's the last line of
    # defense before a real order goes out.
    shortlist -= set(recently_disqualified())

    expiration = _week_friday(today)

    def _enrich(symbol: str) -> Optional[dict]:
        cap = market_cap(symbol)
        if cap is not None and cap < min_market_cap:
            return None
        if cap is None and symbol not in WIDE_LIQUID_UNIVERSE:
            return None  # can't verify cap and it's not on the known-safe list
        if not has_liquid_weekly_options(symbol, expiration):
            return None
        try:
            spot = latest_spot(symbol)
            iv = _atm_implied_vol(symbol, expiration, spot)
            rv = realized_volatility(symbol, today)
        except (ValueError, KeyError):
            return None
        if iv is None:
            return None
        row = {"symbol": symbol, "spot": spot}
        row.update(variance_risk_premium(iv, rv))
        return row

    with ThreadPoolExecutor(max_workers=10) as pool:
        enriched = [r for r in pool.map(_enrich, shortlist) if r is not None]

    columns = [
        "symbol", "spot", "implied_vol", "realized_vol", "vrp", "nvrp", "vrp_elevated", "nvrp_high",
        "z_score", "is_extreme", "direction", "regime", "favorable_for_reversion", "vol_rank_proxy_pct",
    ]
    if not enriched:
        return pd.DataFrame(columns=columns)

    detail = pd.DataFrame(enriched).merge(ranked, on="symbol", how="left")
    passing = detail[detail["vrp_elevated"] | detail["nvrp_high"]].copy()
    if passing.empty:
        return pd.DataFrame(columns=columns)

    # Deliberately NOT running material_news_check here for every survivor.
    # Materiality isn't arithmetic - it's "does this specific headline mean
    # this move is a real repricing," which needs the actual headline text
    # read and weighed, not a keyword-match boolean collapsed at screen time
    # for names nobody may end up looking at closely. check_material_news
    # stays a tool the agent calls itself, on the one or two names it's
    # actually about to trade - not an automatic column on ~20 candidates.
    return passing[columns].sort_values("nvrp", ascending=False).reset_index(drop=True)
