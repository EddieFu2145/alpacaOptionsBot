"""Per-trade breakdown of the vol-driven put credit spread strategy across a
basket of underlyings, to find what losing weeks have in common.

The engine (backtest/engine.py) only scores weekly portfolio equity deltas -
it doesn't tag P&L back to an individual trade. To get trade-level detail we
run each underlying through its own single-name portfolio (mirroring how
run_backtest_vol_spread.py already does this for AAPL alone), and manually
recompute each week's move/breach stats from the same MarketContext the
strategy used.
"""
from datetime import date, timedelta

import pandas as pd

from backtest.engine import run_weekly_backtest
from backtest.market import MarketContext
from backtest.risk import RiskLimits
from backtest.vol_credit_spread import weekly_put_credit_spread
from data.screener import DEFAULT_LARGE_CAP_UNIVERSE

START = date(2026, 5, 31)
END = date(2026, 8, 29)
SHORT_STD_DEVS = 1.0
SPREAD_WIDTH = 5.0

rows = []

for underlying in DEFAULT_LARGE_CAP_UNIVERSE:
    try:
        result = run_weekly_backtest(
            strategy=weekly_put_credit_spread(underlying, short_std_devs=SHORT_STD_DEVS, spread_width=SPREAD_WIDTH, contracts=1),
            underlyings=[underlying],
            start=START,
            end=END,
            risk_limits=RiskLimits(),
        )
    except Exception as exc:
        print(f"{underlying}: skipped ({exc})")
        continue

    market = MarketContext(underlyings=[underlying], start=START, end=END)

    for wr in result.weeks:
        week = wr.week
        try:
            spot_start = market.underlying_close(underlying, week.start)
            spot_end = market.underlying_close(underlying, week.end)
            vol = market.trailing_volatility(underlying, week.start)
        except (KeyError, ValueError):
            continue

        move_pct = (spot_end - spot_start) / spot_start
        years = (week.end - week.start).days / 365
        expected_move = spot_start * vol * (years ** 0.5)
        short_target = spot_start - SHORT_STD_DEVS * expected_move
        breach = max(0.0, short_target - spot_end)  # how far below the short strike target it finished, if at all

        rows.append(
            {
                "underlying": underlying,
                "week_start": week.start,
                "week_end": week.end,
                "pnl": wr.pnl,
                "spot_start": spot_start,
                "spot_end": spot_end,
                "move_pct": move_pct,
                "trailing_vol": vol,
                "short_target": short_target,
                "breached_short": spot_end < short_target,
                "breach_amount": breach,
            }
        )

df = pd.DataFrame(rows)
df.to_csv("losing_trade_analysis.csv", index=False)

print(f"Total weeks analyzed: {len(df)} across {df['underlying'].nunique()} underlyings")
print(f"Losing weeks: {(df['pnl'] < 0).sum()}   Winning weeks: {(df['pnl'] > 0).sum()}   Flat: {(df['pnl'] == 0).sum()}")

losers = df[df["pnl"] < 0]
winners = df[df["pnl"] >= 0]

print("\n--- Losing weeks vs winning weeks: mean stats ---")
compare = pd.DataFrame(
    {
        "losers": [
            losers["move_pct"].mean(),
            losers["move_pct"].abs().mean(),
            (losers["move_pct"] < 0).mean(),
            losers["trailing_vol"].mean(),
            losers["breached_short"].mean(),
            losers["pnl"].mean(),
        ],
        "winners": [
            winners["move_pct"].mean(),
            winners["move_pct"].abs().mean(),
            (winners["move_pct"] < 0).mean(),
            winners["trailing_vol"].mean(),
            winners["breached_short"].mean(),
            winners["pnl"].mean(),
        ],
    },
    index=["mean_move_pct", "mean_abs_move_pct", "pct_weeks_down", "mean_trailing_vol", "pct_breached_short", "mean_pnl"],
)
print(compare)

print("\n--- Worst 10 weeks ---")
print(losers.sort_values("pnl").head(10)[["underlying", "week_start", "pnl", "move_pct", "trailing_vol", "breach_amount"]].to_string(index=False))

print("\n--- Losses by underlying ---")
print(df.groupby("underlying").agg(weeks=("pnl", "size"), losing_weeks=("pnl", lambda s: (s < 0).sum()), total_pnl=("pnl", "sum")).sort_values("total_pnl"))
