"""The weekly backtest loop.

Each week is scored independently: `weekly_pnl = equity_at_week_end -
equity_at_week_start`, where equity = cash + mark-to-market value of open
positions. Because a fair-value trade leaves equity unchanged at the instant
it happens (the cash paid/received exactly offsets the position opened),
this formula automatically counts only the *change* in value during the
week for any position - whether it was opened this week or ten weeks ago -
with no separate "premium collected" accounting needed.

The margin ceiling resets to WEEKLY_MARGIN_CEILING every week regardless of
prior weeks' results (a competition-specific rule, not real broker behavior).
"""
from dataclasses import dataclass, field
from datetime import date
from typing import Callable

import pandas as pd

from .calendar import TradingWeek, trading_weeks
from .market import MarketContext
from .portfolio import Portfolio
from .risk import RiskGate, RiskLimits

Strategy = Callable[[TradingWeek, Portfolio, MarketContext], None]


class RiskLimitBreachedError(Exception):
    pass


@dataclass
class WeekResult:
    week: TradingWeek
    start_equity: float
    end_equity: float
    margin_used: float

    @property
    def pnl(self) -> float:
        return self.end_equity - self.start_equity


@dataclass
class BacktestResult:
    weeks: list[WeekResult] = field(default_factory=list)

    @property
    def weekly_pnl(self) -> pd.Series:
        return pd.Series(
            {w.week.end: w.pnl for w in self.weeks},
            name="weekly_pnl",
        )


def _portfolio_prices_on(
    portfolio: Portfolio, market: MarketContext, day: date
) -> tuple[dict[str, float], dict[str, float]]:
    """Current mark for every held position, and for each position's
    underlying, as of `day`. A contract with no trade that day falls back to
    its own entry price (best available estimate of fair value) rather than
    treating a quiet trading day as an error.
    """
    option_symbols = [s for s, p in portfolio.positions.items() if p.is_option]
    option_closes = market.option_closes(option_symbols, day, day)

    prices: dict[str, float] = {}
    underlying_prices: dict[str, float] = {}
    for symbol, position in portfolio.positions.items():
        if position.is_option:
            prices[symbol] = option_closes.get(symbol, {}).get(day, position.entry_price)
        else:
            prices[symbol] = market.underlying_close(symbol, day)
        if position.underlying:
            underlying_prices.setdefault(
                position.underlying, market.underlying_close(position.underlying, day)
            )
    return prices, underlying_prices


def _settle_expirations(portfolio: Portfolio, market: MarketContext, week: TradingWeek) -> None:
    expiring = [
        (symbol, pos)
        for symbol, pos in portfolio.positions.items()
        if pos.is_option and pos.expiration is not None and week.start <= pos.expiration <= week.end
    ]
    for symbol, position in expiring:
        underlying_price = market.underlying_close(position.underlying, position.expiration)
        if position.option_type == "call":
            in_the_money = underlying_price > position.strike
        else:
            in_the_money = underlying_price < position.strike

        # Capture before close() - it mutates position.quantity to 0 in place.
        original_quantity = position.quantity

        # Cash-settle directly at intrinsic value rather than physically
        # creating an assigned/exercised stock position: a naked, unmanaged
        # stock position left in the portfolio afterward - which this
        # strategy has no logic to ever close - would sit there and swing
        # with the stock's price in every subsequent week, contaminating
        # PnL for weeks that have nothing to do with the original trade.
        # Intrinsic-value settlement is mathematically identical to physical
        # settlement immediately liquidated (both are value-neutral at the
        # instant of expiration), without that risk.
        if in_the_money:
            if position.option_type == "call":
                intrinsic = max(0.0, underlying_price - position.strike)
            else:
                intrinsic = max(0.0, position.strike - underlying_price)
        else:
            intrinsic = 0.0

        portfolio.close(symbol, original_quantity, price=intrinsic, exit_date=position.expiration)


def run_weekly_backtest(
    strategy: Strategy,
    underlyings: list[str],
    start: date,
    end: date,
    risk_limits: RiskLimits | None = None,
) -> BacktestResult:
    market = MarketContext(underlyings=underlyings, start=start, end=end)
    portfolio = Portfolio()
    risk_gate = RiskGate(risk_limits)
    result = BacktestResult()

    prior_prices: dict[str, float] = {}
    for week in trading_weeks(start, end):
        start_equity = portfolio.equity(prior_prices) if prior_prices else portfolio.cash

        strategy(week, portfolio, market)

        start_prices, start_underlying_prices = _portfolio_prices_on(portfolio, market, week.start)
        decision = risk_gate.check(portfolio, start_prices, start_underlying_prices)
        if not decision.approved:
            raise RiskLimitBreachedError(f"Week of {week.start}: " + "; ".join(decision.reasons))
        margin_used = portfolio.margin_used(start_prices, start_underlying_prices)

        _settle_expirations(portfolio, market, week)
        end_prices, _ = _portfolio_prices_on(portfolio, market, week.end)
        end_equity = portfolio.equity(end_prices)

        result.weeks.append(
            WeekResult(week=week, start_equity=start_equity, end_equity=end_equity, margin_used=margin_used)
        )
        prior_prices = end_prices

    return result
