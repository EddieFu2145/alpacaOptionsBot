"""A minimal example strategy - NOT a real trading strategy - included only
to prove the engine works end-to-end against real historical data: sell one
at-the-money weekly put on the underlying, every week, and hold it to
expiration/assignment.
"""
from .calendar import TradingWeek
from .market import MarketContext
from .portfolio import Portfolio, Position


def sell_weekly_atm_put(underlying: str):
    def strategy(week: TradingWeek, portfolio: Portfolio, market: MarketContext) -> None:
        if any(p.underlying == underlying and p.is_option for p in portfolio.positions.values()):
            return  # already holding a position on this name - example only, no rolling logic

        contracts = market.contracts_expiring(underlying, week.end)
        puts = contracts[contracts["option_type"] == "put"]
        if puts.empty:
            return

        spot = market.underlying_close(underlying, week.start)
        target = puts.iloc[(puts["strike"] - spot).abs().argsort()[:1]]
        symbol = target.iloc[0]["symbol"]
        strike = float(target.iloc[0]["strike"])

        price = market.option_closes([symbol], week.start, week.start).get(symbol, {}).get(week.start)
        if price is None:
            return

        portfolio.open_or_add(
            Position(
                symbol=symbol,
                quantity=-1,
                entry_price=price,
                entry_date=week.start,
                is_option=True,
                multiplier=100,
                underlying=underlying,
                option_type="put",
                strike=strike,
                expiration=week.end,
            )
        )

    return strategy
