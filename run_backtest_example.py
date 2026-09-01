"""Smoke test for the weekly backtest engine using the placeholder strategy.
Not a real trading strategy - just proves the pipeline works end-to-end.
"""
from datetime import date

from backtest.engine import run_weekly_backtest
from backtest.example_strategy import sell_weekly_atm_put
from backtest.stats import bootstrap_ci, deflated_sharpe_ratio, permutation_test

UNDERLYING = "AAPL"

result = run_weekly_backtest(
    strategy=sell_weekly_atm_put(UNDERLYING),
    underlyings=[UNDERLYING],
    start=date(2024, 2, 1),
    end=date(2024, 6, 1),
)

weekly_pnl = result.weekly_pnl
print("Weekly PnL:")
print(weekly_pnl)
print(f"\nTotal weeks: {len(weekly_pnl)}")
print(f"Total PnL: ${weekly_pnl.sum():,.2f}")
print(f"Winning weeks: {(weekly_pnl > 0).sum()}/{len(weekly_pnl)}")
print(f"Max single-week margin used: ${max(w.margin_used for w in result.weeks):,.2f}")

print("\n--- Bootstrap CI ---")
print(bootstrap_ci(weekly_pnl))

print("\n--- Permutation test ---")
print(permutation_test(weekly_pnl))

print("\n--- Deflated Sharpe ---")
print(deflated_sharpe_ratio(weekly_pnl))
