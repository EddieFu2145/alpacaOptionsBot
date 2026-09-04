"""Local dashboard: serves a page showing the live agent's positions,
candidate screen, earnings/news calendars, and tunable strategy rules.

Stdlib-only (http.server) - this is a local observability tool, not part
of the trading path, so it doesn't need a real web framework dependency.
Run separately from the trading process: `python -m live.dashboard`. It
never touches MCP or the agent directly - the one write path it does have
(POST /api/strategy_config) only ever writes strategy_config.json, which
the live agent's signal modules read fresh on every check; the dashboard
itself never places, modifies, or cancels an order.
"""
import json
import threading
import time
from datetime import date, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional

from data.clients import trading_client
from data.earnings_calendar import earnings_in_range
from data.news import get_news
from data.options import parse_occ_symbol
from data.screener import DEFAULT_LARGE_CAP_UNIVERSE, wide_vectorized_screen
from live import strategy_config
from live.activity_log import recent_events
from live.token_usage import all_records as all_token_records
from live.trade_log import all_trades, open_trades
from signals.material_news import _HIGH_IMPACT_KEYWORDS

PORT = 8787
_HTML_PATH = Path(__file__).resolve().parent / "dashboard.html"

# wide_vectorized_screen() takes 15-30s and hits real, rate-limited APIs
# (Finnhub especially - already shows real rate-limit stress at current
# volumes, confirmed in data/fundamentals.py's own docstring). Computing it
# on every /api/state poll (every 4s) would hammer those APIs constantly
# and make every poll as slow as the screen itself. Instead it runs on its
# own background timer and every poll just reads the cached result -
# staleness is shown in the UI via computed_at rather than hidden.
CANDIDATES_REFRESH_SECONDS = 900  # 15 min - independent of, and in addition to, the live bot's own screen_candidates calls, so this stays conservative
_candidates_lock = threading.Lock()
_candidates_cache: dict = {"computed_at": None, "rows": [], "error": None, "computing": False}


def _leg_detail(symbol: str, role: str, live_positions: dict) -> dict:
    pos = live_positions.get(symbol)
    try:
        parsed = parse_occ_symbol(symbol)
        strike, option_type = parsed["strike"], parsed["option_type"]
    except ValueError:
        strike, option_type = None, None
    return {
        "symbol": symbol,
        "role": role,
        "strike": strike,
        "option_type": option_type,
        "entry_price": float(pos.avg_entry_price) if pos is not None else None,
        "current_price": float(pos.current_price) if pos is not None and pos.current_price is not None else None,
        "unrealized_pl": float(pos.unrealized_pl) if pos is not None else None,
    }


# Same thresholds and formula as live/position_management.py's
# evaluate_open_positions - this is a read-only mirror for display, so it's
# duplicated rather than imported (that module needs a live MCP session,
# which the dashboard deliberately never touches). Kept numerically
# identical on purpose: showing a different % here than what actually
# drives the agent's real take-profit/defensive decisions would be worse
# than not showing it at all.
TAKE_PROFIT_THRESHOLD = 0.65
PACE_RATIO_THRESHOLD = 3.0


def _risk_metrics(trade: dict, live_positions: dict) -> Optional[dict]:
    short_pos = live_positions.get(trade["short_symbol"])
    long_pos = live_positions.get(trade["long_symbol"])
    if short_pos is None and long_pos is None:
        return None

    combined_pl = float(short_pos.unrealized_pl if short_pos else 0.0) + float(long_pos.unrealized_pl if long_pos else 0.0)

    today = date.today()
    entry_date = date.fromisoformat(trade["entry_date"])
    expiration = date.fromisoformat(trade["expiration"])
    days_held = max((today - entry_date).days, 0)
    planned_days = max((expiration - entry_date).days, 1)
    pct_of_life_elapsed = min(days_held / planned_days, 1.0)

    entry_credit_total = trade["entry_credit"] * 100 * trade["contracts"]
    short_leg = parse_occ_symbol(trade["short_symbol"])
    long_leg = parse_occ_symbol(trade["long_symbol"])
    width = abs(short_leg["strike"] - long_leg["strike"])
    max_loss_total = max(width * 100 - trade["entry_credit"] * 100, 0.0) * trade["contracts"]

    if combined_pl >= 0:
        pct_of_max_outcome = combined_pl / entry_credit_total if entry_credit_total else 0.0
    else:
        pct_of_max_outcome = combined_pl / max_loss_total if max_loss_total else 0.0

    pace_ratio = abs(pct_of_max_outcome) / max(pct_of_life_elapsed, 0.05) if days_held >= 1 else None
    take_profit = pct_of_max_outcome >= TAKE_PROFIT_THRESHOLD
    disproportionate = pace_ratio is not None and pace_ratio >= PACE_RATIO_THRESHOLD

    return {
        "option_type": short_leg["option_type"],
        "pct_of_max_outcome": round(pct_of_max_outcome, 3),
        "pace_ratio": round(pace_ratio, 2) if pace_ratio is not None else None,
        "take_profit": take_profit,
        "defensive_exit": disproportionate and pct_of_max_outcome < 0,
        "consider_closing": disproportionate and pct_of_max_outcome >= 0,
    }


def _collateral(legs: list[dict]) -> float:
    """The real dollar collateral/buying-power this position ties up - the
    exact same formula checked against the risk cap at entry time, not a
    naive re-sum of each leg's own isolated max loss.

    For a condor this matters: summing each 2-leg record's OWN width-based
    max loss independently (what the per-record risk metrics above use,
    deliberately, to mirror evaluate_positions) roughly DOUBLE-COUNTS the
    real risk - the stock can't finish both below the put wing and above
    the call wing at once, so the true combined collateral is
    max(put_width, call_width) minus the TOTAL credit from both sides, the
    same "universal spread rule" math propose_and_execute_iron_condor
    already uses when actually placing the order. Showing the doubled
    number here would misrepresent how much buying power is really held.
    """
    contracts = legs[0]["contracts"]
    if len(legs) == 1:
        t = legs[0]
        short_leg, long_leg = parse_occ_symbol(t["short_symbol"]), parse_occ_symbol(t["long_symbol"])
        width = abs(short_leg["strike"] - long_leg["strike"])
        return max(width * 100 - t["entry_credit"] * 100, 0.0) * contracts

    widths = []
    total_credit = 0.0
    for t in legs:
        short_leg, long_leg = parse_occ_symbol(t["short_symbol"]), parse_occ_symbol(t["long_symbol"])
        widths.append(abs(short_leg["strike"] - long_leg["strike"]))
        total_credit += t["entry_credit"]
    return max(0.0, max(widths) * 100 * contracts - total_credit * 100 * contracts)


def _build_positions(open_records: list[dict], live_positions: dict) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for t in open_records:
        key = t.get("group_id") or f"single-{t['short_symbol']}"
        groups.setdefault(key, []).append(t)

    positions = []
    for legs in groups.values():
        leg_details = []
        total_unrealized = 0.0
        legs_with_marks = 0
        risk_metrics = []
        for t in legs:
            for symbol, role in ((t["short_symbol"], "short"), (t["long_symbol"], "long")):
                detail = _leg_detail(symbol, role, live_positions)
                if detail["unrealized_pl"] is not None:
                    total_unrealized += detail["unrealized_pl"]
                    legs_with_marks += 1
                leg_details.append(detail)
            metrics = _risk_metrics(t, live_positions)
            if metrics:
                risk_metrics.append(metrics)
        first = legs[0]
        is_condor = len(legs) == 2
        if is_condor:
            bias = "neutral"
        else:
            # A single 2-leg record is either all puts or all calls - a put
            # credit spread profits if the stock holds above the short
            # strike (bullish/neutral-to-up), a call credit spread profits
            # if it stays below (bearish/neutral-to-down).
            option_type = leg_details[0]["option_type"] if leg_details else None
            bias = "bullish" if option_type == "put" else "bearish" if option_type == "call" else None
        positions.append(
            {
                "underlying": first["underlying"],
                "is_condor": is_condor,
                "bias": bias,
                "contracts": first["contracts"],
                "entry_date": first["entry_date"],
                "expiration": first["expiration"],
                "entry_credit_total": round(sum(t["entry_credit"] for t in legs), 3),
                "collateral": round(_collateral(legs), 2),
                "legs": leg_details,
                "risk": risk_metrics,
                "unrealized_pl": round(total_unrealized, 2) if legs_with_marks else None,
                # False if even one leg is missing a live mark - confirmed
                # this can genuinely happen (a leg closed on the broker but
                # still logged open, or a fresh fill the stream hasn't
                # subscribed to yet) and silently summing only what's
                # available would show a P&L number that looks complete
                # but isn't.
                "partial_marks": 0 < legs_with_marks < len(leg_details),
            }
        )
    positions.sort(key=lambda p: p["underlying"])
    return positions


RESEARCHING_STALE_SECONDS = 90  # a couple of DeepSeek-throttled turns (4/min) - past this, treat the session as idle rather than "active"

_TICKER_FIELDS_DIRECT = ("underlying", "underlying_symbol")
_TICKER_FIELDS_OCC = (
    "symbol", "short_symbol", "long_symbol",
    "short_put_symbol", "long_put_symbol", "short_call_symbol", "long_call_symbol",
)


def _extract_ticker(args: dict) -> Optional[str]:
    """Best-effort: pull the underlying ticker a tool call was actually
    about, out of whichever argument shape that particular tool uses -
    there's no single consistent field name across the ~15 tools the agent
    calls, so this tries the plain-ticker fields first, then OCC option
    symbols (parsed down to their underlying), then a symbols= list.
    """
    for field in _TICKER_FIELDS_DIRECT:
        if args.get(field):
            return args[field]
    for field in _TICKER_FIELDS_OCC:
        value = args.get(field)
        if value:
            try:
                return parse_occ_symbol(value)["underlying"]
            except ValueError:
                return value
    symbols = args.get("symbols")
    if symbols:
        first = symbols.split(",")[0].strip() if isinstance(symbols, str) else (symbols[0] if symbols else None)
        if first:
            try:
                return parse_occ_symbol(first)["underlying"]
            except ValueError:
                return first
    return None


def _build_researching(limit: int = 150) -> dict:
    events = recent_events(limit)  # chronological, oldest first
    if not events:
        return {"active": False, "ticker": None, "action": None, "seconds_ago": None, "session_tickers": []}

    seconds_ago = time.time() - events[-1]["ts"]
    ticker = action = None
    session_tickers: list[str] = []
    seen: set[str] = set()

    for ev in reversed(events):  # newest to oldest
        if ev["type"] == "session":
            break  # don't reach back into a previous session's tickers
        if ev["type"] == "tool_call":
            found = _extract_ticker(ev.get("args") or {})
            if found:
                if ticker is None:
                    ticker, action = found, ev.get("name")
                if found not in seen:
                    seen.add(found)
                    session_tickers.append(found)

    return {
        "active": seconds_ago < RESEARCHING_STALE_SECONDS,
        "ticker": ticker,
        "action": action,
        "seconds_ago": round(seconds_ago),
        "session_tickers": session_tickers[:15],
    }


def _closed_with_pct(t: dict) -> dict:
    """Adds realized_pnl_pct: return on the capital this record actually put
    at risk (its own max loss, same formula as _risk_metrics/_collateral for
    a plain 2-leg spread), not on the credit collected - collateral is the
    number that was actually locked up, so it's the honest basis for a %
    return figure. None whenever realized_pnl itself is unavailable (either
    this record predates realized-P&L tracking, or the closing fill wasn't
    confirmed in time) - never guessed.
    """
    out = dict(t)
    pnl = t.get("realized_pnl")
    if pnl is None:
        out["realized_pnl_pct"] = None
        return out
    short_leg, long_leg = parse_occ_symbol(t["short_symbol"]), parse_occ_symbol(t["long_symbol"])
    width = abs(short_leg["strike"] - long_leg["strike"])
    max_loss_total = max(width * 100 - t["entry_credit"] * 100, 0.0) * t["contracts"]
    out["realized_pnl_pct"] = round(pnl / max_loss_total * 100, 1) if max_loss_total else None
    return out


def _build_state() -> dict:
    trades = all_trades()
    open_records = [t for t in trades if not t["closed"]]
    closed_records = [_closed_with_pct(t) for t in trades if t["closed"]][-10:][::-1]

    account_info = None
    live_positions: dict = {}
    broker_error = None
    positions: list[dict] = []
    try:
        tc = trading_client()
        account = tc.get_account()
        live_positions = {p.symbol: p for p in tc.get_all_positions()}
        positions = _build_positions(open_records, live_positions)
        equity, last_equity = float(account.equity), float(account.last_equity)
        options_bp = float(account.options_buying_power)
        total_collateral = round(sum(p["collateral"] for p in positions), 2)
        account_info = {
            "equity": equity,
            "day_pl": round(equity - last_equity, 2),
            "day_pl_pct": round((equity - last_equity) / last_equity * 100, 3) if last_equity else 0.0,
            "options_buying_power": options_bp,
            # Collateral already committed by open positions - options_bp
            # already reflects this (it's Alpaca's remaining, post-collateral
            # figure), so this is shown alongside it for context, not summed
            # with it.
            "total_collateral": total_collateral,
            "collateral_pct_of_equity": round(total_collateral / equity * 100, 2) if equity else 0.0,
        }
    except Exception as exc:
        broker_error = str(exc)

    return {
        "generated_at": time.time(),
        "account": account_info,
        "broker_error": broker_error,
        "positions": positions,
        "closed_recent": closed_records,
        "researching": _build_researching(),
    }


def _token_usage_summary(recent_limit: int = 50) -> dict:
    records = all_token_records()  # chronological, oldest first
    today_iso = date.today().isoformat()

    def _sum(rows: list[dict], field: str) -> int:
        return sum(r.get(field, 0) or 0 for r in rows)

    today_records = [r for r in records if date.fromtimestamp(r["ts"]).isoformat() == today_iso]
    cache_hit = _sum(records, "prompt_cache_hit_tokens")
    cache_miss = _sum(records, "prompt_cache_miss_tokens")

    return {
        "total_calls": len(records),
        "total_prompt_tokens": _sum(records, "prompt_tokens"),
        "total_completion_tokens": _sum(records, "completion_tokens"),
        "total_tokens": _sum(records, "total_tokens"),
        "cache_hit_tokens": cache_hit,
        "cache_miss_tokens": cache_miss,
        # None (not 0) when the API never reported cache fields at all -
        # distinct from "reported, but 0% hit rate."
        "cache_hit_rate_pct": round(cache_hit / (cache_hit + cache_miss) * 100, 1) if (cache_hit + cache_miss) else None,
        "today_calls": len(today_records),
        "today_tokens": _sum(today_records, "total_tokens"),
        "recent": list(reversed(records[-recent_limit:])),
    }


def _refresh_candidates_once() -> None:
    with _candidates_lock:
        if _candidates_cache["computing"]:
            return
        _candidates_cache["computing"] = True
    try:
        rows = wide_vectorized_screen().to_dict(orient="records")
        with _candidates_lock:
            _candidates_cache["rows"] = rows
            _candidates_cache["computed_at"] = time.time()
            _candidates_cache["error"] = None
    except Exception as exc:
        with _candidates_lock:
            _candidates_cache["error"] = str(exc)
    finally:
        with _candidates_lock:
            _candidates_cache["computing"] = False


def _market_is_open() -> bool:
    try:
        return bool(trading_client().get_clock().is_open)
    except Exception:
        return False  # don't burn the rate-limited screen budget on an uncertain clock check


def _candidates_refresh_loop() -> None:
    while True:
        if _market_is_open():
            _refresh_candidates_once()
        time.sleep(CANDIDATES_REFRESH_SECONDS)


def _watched_universe() -> list[str]:
    """Same scope run_trading_day.py's own pre-market check uses: whatever
    is currently open plus the default large-cap watchlist - not the full
    market, which would make an earnings/news calendar mostly irrelevant
    noise for a book this small.
    """
    return sorted({t["underlying"] for t in open_trades()} | set(DEFAULT_LARGE_CAP_UNIVERSE))


EARNINGS_REFRESH_SECONDS = 1800  # 30 min - scheduled earnings dates don't move intraday
EARNINGS_LOOKAHEAD_DAYS = 21  # matches the widest expiration window this pipeline actually trades
_earnings_lock = threading.Lock()
_earnings_cache: dict = {"computed_at": None, "rows": [], "error": None}


def _refresh_earnings_once() -> None:
    try:
        watched = set(_watched_universe())
        today = date.today()
        df = earnings_in_range(today, today + timedelta(days=EARNINGS_LOOKAHEAD_DAYS))
        rows = []
        if not df.empty:
            scoped = df[df["symbol"].isin(watched)].sort_values("date")
            rows = [{"symbol": r.symbol, "date": r.date.isoformat(), "hour": r.hour} for r in scoped.itertuples()]
        with _earnings_lock:
            _earnings_cache["rows"] = rows
            _earnings_cache["computed_at"] = time.time()
            _earnings_cache["error"] = None
    except Exception as exc:
        with _earnings_lock:
            _earnings_cache["error"] = str(exc)


def _earnings_refresh_loop() -> None:
    while True:
        _refresh_earnings_once()
        time.sleep(EARNINGS_REFRESH_SECONDS)


NEWS_REFRESH_SECONDS = 900  # 15 min
NEWS_LOOKBACK_DAYS = 5
_news_lock = threading.Lock()
_news_cache: dict = {"computed_at": None, "rows": [], "error": None}


def _refresh_news_once() -> None:
    try:
        watched = set(_watched_universe())
        end = datetime.now()
        start = end - timedelta(days=NEWS_LOOKBACK_DAYS)
        df = get_news(sorted(watched), start, end, limit=200)
        rows = []
        if not df.empty:
            for _, row in df.iterrows():
                # Round-tripping through data/cache.py's parquet cache turns
                # this column's Python lists into numpy arrays - `arr or []`
                # then raises ("truth value of an array... is ambiguous")
                # instead of falling back, so the None-check has to come
                # first and the conversion to a plain list has to be
                # explicit rather than relying on truthiness.
                raw_symbols = row.get("symbols")
                symbols_list = list(raw_symbols) if raw_symbols is not None else []
                matched = sorted(set(symbols_list) & watched)
                if not matched:
                    continue
                text = f"{row.get('headline', '')} {row.get('summary', '')}".lower()
                # Same keyword list signals/material_news.py's real gate
                # uses - what's flagged here is exactly what would block a
                # trade, not a separately-tuned "looks newsworthy" heuristic.
                hit_keywords = [kw for kw in _HIGH_IMPACT_KEYWORDS if kw in text]
                created_at = row.get("created_at")
                rows.append(
                    {
                        "symbols": matched,
                        "headline": row.get("headline"),
                        "source": row.get("source"),
                        "url": row.get("url"),
                        "created_at": created_at.isoformat() if created_at is not None else None,
                        "is_material": bool(hit_keywords),
                        "matched_keywords": hit_keywords,
                    }
                )
        rows.sort(key=lambda r: r["created_at"] or "", reverse=True)
        with _news_lock:
            _news_cache["rows"] = rows[:150]
            _news_cache["computed_at"] = time.time()
            _news_cache["error"] = None
    except Exception as exc:
        with _news_lock:
            _news_cache["error"] = str(exc)


def _news_refresh_loop() -> None:
    while True:
        _refresh_news_once()
        time.sleep(NEWS_REFRESH_SECONDS)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:
        pass  # keep the console quiet - this is polled every few seconds

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self._serve_html()
        elif self.path == "/api/state":
            self._serve_json(_build_state())
        elif self.path == "/api/candidates":
            with _candidates_lock:
                self._serve_json(dict(_candidates_cache))
        elif self.path == "/api/candidates/refresh":
            threading.Thread(target=_refresh_candidates_once, daemon=True).start()
            self._serve_json({"status": "refreshing"})
        elif self.path == "/api/token_usage":
            self._serve_json(_token_usage_summary())
        elif self.path == "/api/earnings_calendar":
            with _earnings_lock:
                self._serve_json(dict(_earnings_cache))
        elif self.path == "/api/news_calendar":
            with _news_lock:
                self._serve_json(dict(_news_cache))
        elif self.path == "/api/strategy_config":
            self._serve_json(strategy_config.load())
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self) -> None:
        if self.path == "/api/strategy_config":
            length = int(self.headers.get("Content-Length", 0) or 0)
            body = self.rfile.read(length) if length else b"{}"
            try:
                updates = json.loads(body)
            except json.JSONDecodeError:
                updates = {}
            self._serve_json(strategy_config.save(updates))
        else:
            self.send_response(404)
            self.end_headers()

    def _serve_html(self) -> None:
        body = _HTML_PATH.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_json(self, data: dict) -> None:
        body = json.dumps(data, default=str).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    threading.Thread(target=_refresh_candidates_once, daemon=True).start()  # don't make the first page load wait 15 min for data
    threading.Thread(target=_candidates_refresh_loop, daemon=True).start()
    threading.Thread(target=_refresh_earnings_once, daemon=True).start()
    threading.Thread(target=_earnings_refresh_loop, daemon=True).start()
    threading.Thread(target=_refresh_news_once, daemon=True).start()
    threading.Thread(target=_news_refresh_loop, daemon=True).start()

    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"Dashboard running at http://127.0.0.1:{PORT} (Ctrl+C to stop)")
    server.serve_forever()


if __name__ == "__main__":
    main()
