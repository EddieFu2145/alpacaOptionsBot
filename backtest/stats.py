"""Statistical validation of a weekly PnL series.

Each week is treated as one independent trial - the unit the competition
actually scores on - rather than smoothing everything into a single
cumulative equity curve. Every function here takes a `pd.Series` of
per-week PnL (as produced by `BacktestResult.weekly_pnl`) and answers a
different flavor of "is this edge real, or noise?".
"""
import numpy as np
import pandas as pd
from scipy import stats


def bootstrap_ci(
    weekly_pnl: pd.Series,
    n_resamples: int = 10_000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict:
    """Resample weeks with replacement to get a confidence interval on mean
    weekly PnL and on the weekly Sharpe ratio, instead of trusting the single
    historical sequence of weeks we happened to observe.
    """
    rng = np.random.default_rng(seed)
    values = weekly_pnl.to_numpy()
    n = len(values)

    resampled_means = np.empty(n_resamples)
    resampled_sharpes = np.empty(n_resamples)
    for i in range(n_resamples):
        sample = rng.choice(values, size=n, replace=True)
        resampled_means[i] = sample.mean()
        std = sample.std(ddof=1)
        resampled_sharpes[i] = sample.mean() / std if std > 0 else 0.0

    alpha = (1 - ci) / 2
    return {
        "n_weeks": n,
        "observed_mean_weekly_pnl": values.mean(),
        "mean_ci": (
            np.quantile(resampled_means, alpha),
            np.quantile(resampled_means, 1 - alpha),
        ),
        "observed_weekly_sharpe": values.mean() / values.std(ddof=1) if values.std(ddof=1) > 0 else 0.0,
        "sharpe_ci": (
            np.quantile(resampled_sharpes, alpha),
            np.quantile(resampled_sharpes, 1 - alpha),
        ),
    }


def permutation_test(
    weekly_pnl: pd.Series,
    n_permutations: int = 10_000,
    seed: int = 0,
) -> dict:
    """Tests H0: the strategy has no real edge, i.e. each week's PnL sign is
    as likely to have been a coin flip. Randomly flips each week's sign many
    times to build a null distribution, then checks where the observed mean
    falls in it.
    """
    rng = np.random.default_rng(seed)
    values = weekly_pnl.to_numpy()
    observed_mean = values.mean()

    null_means = np.empty(n_permutations)
    for i in range(n_permutations):
        signs = rng.choice([-1, 1], size=len(values))
        null_means[i] = (values * signs).mean()

    p_value = np.mean(np.abs(null_means) >= abs(observed_mean))
    return {
        "observed_mean_weekly_pnl": observed_mean,
        "p_value": p_value,
        "significant_at_5pct": p_value < 0.05,
    }


def out_of_sample_split(weekly_pnl: pd.Series, holdout_fraction: float = 0.2) -> tuple[pd.Series, pd.Series]:
    """Chronological split - the holdout weeks must never influence anything
    upstream (strategy parameters, thresholds) that was tuned on the rest."""
    split_index = int(len(weekly_pnl) * (1 - holdout_fraction))
    return weekly_pnl.iloc[:split_index], weekly_pnl.iloc[split_index:]


def walk_forward_splits(
    weekly_pnl: pd.Series, n_folds: int = 5, min_train_weeks: int = 10
) -> list[tuple[pd.Series, pd.Series]]:
    """Expanding-window train/test folds, for checking whether a strategy's
    edge (or its tuned parameters) holds up out of the window it was found
    in, rather than being an artifact of one period.
    """
    n = len(weekly_pnl)
    test_size = (n - min_train_weeks) // n_folds
    if test_size < 1:
        raise ValueError(f"Not enough weeks ({n}) for {n_folds} folds with a {min_train_weeks}-week minimum train set")

    folds = []
    for fold in range(n_folds):
        train_end = min_train_weeks + fold * test_size
        test_end = train_end + test_size
        folds.append((weekly_pnl.iloc[:train_end], weekly_pnl.iloc[train_end:test_end]))
    return folds


def deflated_sharpe_ratio(weekly_pnl: pd.Series, n_trials: int = 1) -> dict:
    """Approximate implementation of Bailey & Lopez de Prado's deflated
    Sharpe ratio: corrects the Sharpe ratio for non-normal returns (skew/
    kurtosis) and for having tried `n_trials` parameter combinations, since
    the best of many random configurations will look good by chance alone.
    Pass n_trials=1 for "just tell me the significance of this one config".
    """
    values = weekly_pnl.to_numpy()
    n = len(values)
    sr = values.mean() / values.std(ddof=1)
    skew = stats.skew(values)
    kurt = stats.kurtosis(values, fisher=False)  # non-excess; normal = 3

    variance_of_sr = (1 - skew * sr + (kurt - 1) / 4 * sr**2) / (n - 1)

    if n_trials > 1:
        euler_gamma = 0.5772156649
        benchmark_sr = np.sqrt(variance_of_sr) * (
            (1 - euler_gamma) * stats.norm.ppf(1 - 1.0 / n_trials)
            + euler_gamma * stats.norm.ppf(1 - 1.0 / (n_trials * np.e))
        )
    else:
        benchmark_sr = 0.0

    dsr_statistic = (sr - benchmark_sr) / np.sqrt(variance_of_sr)
    deflated_probability = stats.norm.cdf(dsr_statistic)

    return {
        "n_weeks": n,
        "weekly_sharpe": sr,
        "skew": skew,
        "kurtosis": kurt,
        "n_trials_corrected_for": n_trials,
        "benchmark_sharpe_under_null": benchmark_sr,
        "deflated_sharpe_probability": deflated_probability,
    }
