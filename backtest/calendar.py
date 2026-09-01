"""Trading-week boundaries, respecting market holidays.

The competition scores week by week, so every other module in this package
operates on the trading week (first trading day -> last trading day) as its
fundamental unit, not the calendar week.
"""
from dataclasses import dataclass
from datetime import date
from itertools import groupby

from alpaca.trading.requests import GetCalendarRequest

from data.clients import trading_client


@dataclass(frozen=True)
class TradingWeek:
    days: tuple[date, ...]

    @property
    def start(self) -> date:
        return self.days[0]

    @property
    def end(self) -> date:
        return self.days[-1]


def trading_days(start: date, end: date) -> list[date]:
    calendar = trading_client().get_calendar(GetCalendarRequest(start=start, end=end))
    return [c.date for c in calendar]


def trading_weeks(start: date, end: date) -> list[TradingWeek]:
    """Trading days in [start, end] grouped into Mon-Fri trading weeks."""
    days = trading_days(start, end)
    weeks = []
    for _, group in groupby(days, key=lambda d: d.isocalendar()[:2]):
        weeks.append(TradingWeek(days=tuple(group)))
    return weeks
