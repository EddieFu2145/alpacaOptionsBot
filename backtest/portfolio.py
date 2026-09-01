"""Position tracking, mark-to-market valuation, and margin usage.

Quantity is signed (negative = short) throughout, which makes mark-to-market
P&L work identically for longs and shorts with no special-casing: the value
of a position is always `quantity * price * multiplier`, so a short position's
value rises toward zero (i.e. profits) as the option's price falls.
"""
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

STARTING_CAPITAL = 100_000.0
WEEKLY_MARGIN_CEILING = 400_000.0


@dataclass
class Position:
    symbol: str
    quantity: float
    entry_price: float
    entry_date: date
    is_option: bool
    multiplier: float = 1.0
    underlying: Optional[str] = None
    option_type: Optional[str] = None  # "call" | "put"
    strike: Optional[float] = None
    expiration: Optional[date] = None

    def value(self, price: float) -> float:
        return self.quantity * price * self.multiplier

    def naked_margin_per_contract(self, price: float, underlying_price: Optional[float] = None) -> float:
        """Simplified Reg-T naked-option/short-equity requirement, per single
        contract/share. An approximation of real broker formulas, not an
        exact reproduction. Only meaningful for short positions."""
        if not self.is_option:
            return 1.5 * price  # short equity: proceeds + 50% margin

        underlying_price = underlying_price if underlying_price is not None else price
        if self.option_type == "call":
            otm_amount = max(0.0, self.strike - underlying_price)
        else:
            otm_amount = max(0.0, underlying_price - self.strike)

        return max(0.20 * underlying_price - otm_amount, 0.10 * underlying_price) * 100 + price * 100

    def margin_requirement(self, price: float, underlying_price: Optional[float] = None) -> float:
        if self.quantity >= 0:
            return 0.0
        return self.naked_margin_per_contract(price, underlying_price) * abs(self.quantity)


@dataclass
class Portfolio:
    cash: float = STARTING_CAPITAL
    positions: dict[str, Position] = field(default_factory=dict)

    def open_or_add(self, position: Position) -> None:
        self.cash -= position.quantity * position.entry_price * position.multiplier
        existing = self.positions.get(position.symbol)
        if existing is None:
            self.positions[position.symbol] = position
            return

        total_quantity = existing.quantity + position.quantity
        if total_quantity == 0:
            del self.positions[position.symbol]
            return
        existing.entry_price = (
            existing.entry_price * existing.quantity + position.entry_price * position.quantity
        ) / total_quantity
        existing.quantity = total_quantity

    def close(self, symbol: str, quantity: float, price: float, exit_date: date) -> None:
        """Close (or reduce) a position. `quantity` is the signed amount to
        remove - i.e. positive to close a short, negative to close a long."""
        position = self.positions[symbol]
        self.cash += quantity * price * position.multiplier
        position.quantity -= quantity
        if abs(position.quantity) < 1e-9:
            del self.positions[symbol]

    def mark_to_market(self, prices: dict[str, float]) -> float:
        return sum(pos.value(prices[symbol]) for symbol, pos in self.positions.items())

    def equity(self, prices: dict[str, float]) -> float:
        return self.cash + self.mark_to_market(prices)

    def margin_used(self, prices: dict[str, float], underlying_prices: dict[str, float]) -> float:
        return sum(self.margin_by_underlying(prices, underlying_prices).values())

    def margin_by_underlying(self, prices: dict[str, float], underlying_prices: dict[str, float]) -> dict[str, float]:
        """Margin requirement grouped by underlying. Short options paired
        with a further-OTM long option on the same underlying/expiration/
        type are priced as a defined-risk vertical spread (strike width
        minus the credit locked in at entry, fixed regardless of current
        price) rather than each leg's full naked requirement - otherwise a
        protective long leg would count for nothing, which isn't how real
        margin works.
        """
        breakdown: dict[str, float] = {}
        for pos in self.positions.values():
            if not pos.is_option:
                breakdown[pos.symbol] = breakdown.get(pos.symbol, 0.0) + pos.margin_requirement(
                    prices[pos.symbol], None
                )

        groups: dict[tuple, list[Position]] = {}
        for pos in self.positions.values():
            if pos.is_option:
                groups.setdefault((pos.underlying, pos.expiration, pos.option_type), []).append(pos)

        for (underlying, _, _), legs in groups.items():
            margin, _ = _option_group_margin(legs, prices, underlying_prices.get(underlying))
            breakdown[underlying] = breakdown.get(underlying, 0.0) + margin

        return breakdown

    def naked_option_legs(self) -> list[tuple[Position, float]]:
        """(position, unprotected quantity) for every short option leg whose
        quantity isn't fully offset by a further-OTM long on the same
        underlying/expiration/type."""
        groups: dict[tuple, list[Position]] = {}
        for pos in self.positions.values():
            if pos.is_option:
                groups.setdefault((pos.underlying, pos.expiration, pos.option_type), []).append(pos)

        naked = []
        for legs in groups.values():
            _, leftovers = _option_group_margin(legs, prices=None, underlying_price=None)
            naked.extend(leftovers)
        return naked


def _option_group_margin(
    legs: list[Position], prices: Optional[dict[str, float]], underlying_price: Optional[float]
) -> tuple[float, list[tuple[Position, float]]]:
    shorts = sorted([[leg, abs(leg.quantity)] for leg in legs if leg.quantity < 0], key=lambda e: e[0].strike)
    longs = sorted([[leg, leg.quantity] for leg in legs if leg.quantity > 0], key=lambda e: e[0].strike)

    total = 0.0
    naked: list[tuple[Position, float]] = []
    long_index = 0
    for short_leg, short_remaining in shorts:
        while short_remaining > 1e-9 and long_index < len(longs):
            long_leg, long_remaining = longs[long_index]
            paired = min(short_remaining, long_remaining)

            width = abs(short_leg.strike - long_leg.strike)
            net_credit_per_contract = short_leg.entry_price - long_leg.entry_price
            margin_per_contract = max(width * 100 - net_credit_per_contract * 100, 0.0)
            total += margin_per_contract * paired

            short_remaining -= paired
            longs[long_index][1] -= paired
            if longs[long_index][1] <= 1e-9:
                long_index += 1

        if short_remaining > 1e-9:
            naked.append((short_leg, short_remaining))
            if prices is not None:
                total += short_leg.naked_margin_per_contract(prices[short_leg.symbol], underlying_price) * short_remaining

    return total, naked
