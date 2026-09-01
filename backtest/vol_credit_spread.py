"""A realized-volatility-driven weekly put credit spread.

Still a starting point, not a tuned strategy - but unlike the earlier
ATM-put placeholder this one is defined-risk (a paired short/long leg, so it
satisfies the risk gate's require_defined_risk rule) and picks its short
strike off the underlying's own trailing realized volatility rather than
"whatever's closest to spot", since Alpaca has no historical IV/delta to
target directly.
"""
from .calendar import TradingWeek
from .execution import apply_slippage
from .market import MarketContext
from .portfolio import Portfolio, Position


def weekly_put_credit_spread(
    underlying: str,
    short_std_devs: float = 1.0,
    spread_width: float = 5.0,
    contracts: int = 1,
    slippage_bps: float = 5.0,
):
    """Sell a put `short_std_devs` trailing-vol standard deviations OTM, buy
    protection `spread_width` dollars further OTM, both expiring at week end.
    """

    def strategy(week: TradingWeek, portfolio: Portfolio, market: MarketContext) -> None:
        if any(p.underlying == underlying and p.is_option for p in portfolio.positions.values()):
            return  # one spread per name at a time - no rolling/adding logic yet

        spot = market.underlying_close(underlying, week.start)
        try:
            vol = market.trailing_volatility(underlying, week.start)
        except ValueError:
            return  # not enough trailing history yet (start of the backtest window)

        years_to_expiry = (week.end - week.start).days / 365
        expected_move = spot * vol * (years_to_expiry**0.5)
        short_target = spot - short_std_devs * expected_move
        long_target = short_target - spread_width

        contracts_df = market.contracts_expiring(underlying, week.end)
        puts = contracts_df[contracts_df["option_type"] == "put"]
        if puts.empty:
            return

        short_row = puts.iloc[(puts["strike"] - short_target).abs().argsort()[:1]].iloc[0]
        long_candidates = puts[puts["strike"] < short_row["strike"]]
        if long_candidates.empty:
            return
        long_row = long_candidates.iloc[(long_candidates["strike"] - long_target).abs().argsort()[:1]].iloc[0]

        symbols = [short_row["symbol"], long_row["symbol"]]
        entry_prices = market.option_closes(symbols, week.start, week.start)
        short_price = entry_prices.get(short_row["symbol"], {}).get(week.start)
        long_price = entry_prices.get(long_row["symbol"], {}).get(week.start)
        if short_price is None or long_price is None or short_price <= long_price:
            return  # no trade/thin quotes this week rather than force a bad structure

        portfolio.open_or_add(
            Position(
                symbol=short_row["symbol"],
                quantity=-contracts,
                entry_price=apply_slippage(short_price, -contracts, slippage_bps),
                entry_date=week.start,
                is_option=True,
                multiplier=100,
                underlying=underlying,
                option_type="put",
                strike=float(short_row["strike"]),
                expiration=week.end,
            )
        )
        portfolio.open_or_add(
            Position(
                symbol=long_row["symbol"],
                quantity=contracts,
                entry_price=apply_slippage(long_price, contracts, slippage_bps),
                entry_date=week.start,
                is_option=True,
                multiplier=100,
                underlying=underlying,
                option_type="put",
                strike=float(long_row["strike"]),
                expiration=week.end,
            )
        )

    return strategy
