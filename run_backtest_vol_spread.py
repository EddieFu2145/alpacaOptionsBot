"""End-to-end test of the defined-risk vol-driven credit spread strategy,
run through the risk gate and full statistical validation battery.
"""
from datetime import date

from backtest.engine import run_weekly_backtest
from backtest.risk import RiskLimits
from backtest.stats import bootstrap_ci, deflated_sharpe_ratio, out_of_sample_split, permutation_test
from backtest.vol_credit_spread import weekly_put_credit_spread

UNDERLYING = "AAPL"

result = run_weekly_backtest(
    strategy=weekly_put_credit_spread(UNDERLYING, short_std_devs=1.0, spread_width=5.0, contracts=5),
    underlyings=[UNDERLYING],
    start=date(2024, 3, 1),
    end=date(2025, 12, 1),
    risk_limits=RiskLimits(),
)

weekly_pnl = result.weekly_pnl
print(f"Total weeks traded/scored: {len(weekly_pnl)}")
print(f"Total PnL: ${weekly_pnl.sum():,.2f}")
print(f"Winning weeks: {(weekly_pnl > 0).sum()}/{len(weekly_pnl)}")
print(f"Mean weekly PnL: ${weekly_pnl.mean():,.2f}")
print(f"Max single-week margin used: ${max(w.margin_used for w in result.weeks):,.2f}")
print(f"Worst week: ${weekly_pnl.min():,.2f}   Best week: ${weekly_pnl.max():,.2f}")

print("\n--- Bootstrap CI (weeks as independent trials) ---")
print(bootstrap_ci(weekly_pnl))

print("\n--- Permutation test ---")
print(permutation_test(weekly_pnl))

print("\n--- Deflated Sharpe (n_trials=1) ---")
print(deflated_sharpe_ratio(weekly_pnl))

print("\n--- Out-of-sample split (80/20) ---")
train, holdout = out_of_sample_split(weekly_pnl)
print(f"Train weeks: {len(train)}, mean ${train.mean():,.2f}")
print(f"Holdout weeks: {len(holdout)}, mean ${holdout.mean():,.2f}")
