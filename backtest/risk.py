"""The hard standards layer: a deterministic, code-enforced gate between
whatever proposes a trade (today, a plain strategy function; later, an LLM
research agent) and the portfolio. Nothing here is advisory - a rejection is
a rejection regardless of what proposed the trade, which is the whole point
of keeping this separate from any discretionary/LLM-driven logic.
"""
from dataclasses import dataclass

from .portfolio import WEEKLY_MARGIN_CEILING, Portfolio


@dataclass
class RiskLimits:
    max_margin: float = WEEKLY_MARGIN_CEILING
    max_margin_per_underlying: float = WEEKLY_MARGIN_CEILING * 0.4  # no single name eats the whole ceiling
    require_defined_risk: bool = True  # every short option must be paired with a further-OTM long
    max_positions_per_underlying: int = 6


@dataclass
class RiskDecision:
    approved: bool
    reasons: list[str]


class RiskGate:
    """Call `.check(...)` AFTER a candidate trade has been applied to the
    portfolio (the engine itself does this every week) to get a pass/fail
    verdict with reasons. This gate does not undo trades - a strategy that
    gets rejected is expected to check before proceeding, the same way the
    engine's margin check already halts the backtest with a clear error
    rather than silently reinterpreting what the strategy meant to do.
    """

    def __init__(self, limits: RiskLimits | None = None):
        self.limits = limits or RiskLimits()

    def check(self, portfolio: Portfolio, prices: dict, underlying_prices: dict) -> RiskDecision:
        reasons = []

        margin_by_underlying = portfolio.margin_by_underlying(prices, underlying_prices)
        total_margin = sum(margin_by_underlying.values())
        if total_margin > self.limits.max_margin:
            reasons.append(f"total margin ${total_margin:,.0f} exceeds ceiling ${self.limits.max_margin:,.0f}")

        for underlying, margin in margin_by_underlying.items():
            if margin > self.limits.max_margin_per_underlying:
                reasons.append(
                    f"{underlying} margin ${margin:,.0f} exceeds per-underlying cap "
                    f"${self.limits.max_margin_per_underlying:,.0f}"
                )

        if self.limits.require_defined_risk:
            naked = portfolio.naked_option_legs()
            for position, unprotected_qty in naked:
                reasons.append(
                    f"{position.symbol} has {unprotected_qty:g} unprotected (naked) short contracts"
                )

        positions_per_underlying: dict[str, int] = {}
        for pos in portfolio.positions.values():
            if pos.underlying:
                positions_per_underlying[pos.underlying] = positions_per_underlying.get(pos.underlying, 0) + 1
        for underlying, count in positions_per_underlying.items():
            if count > self.limits.max_positions_per_underlying:
                reasons.append(
                    f"{underlying} has {count} open positions, exceeding limit of "
                    f"{self.limits.max_positions_per_underlying}"
                )

        return RiskDecision(approved=not reasons, reasons=reasons)
