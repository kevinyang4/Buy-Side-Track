"""
backtest.py
Member B - Backtest Engine
Vectorized Portfolio Accounting

Interface contract:
    run_backtest(prices: pd.Series, positions: pd.Series) -> pd.DataFrame
    compute_metrics(results: pd.DataFrame) -> dict
"""

import pandas as pd
import numpy as np


def run_backtest(prices: pd.Series, positions: pd.Series) -> pd.DataFrame:
    """Returns DataFrame with asset_returns, strat_returns, and cumulative wealth."""

    asset_returns = prices.pct_change()

    strat_returns = positions * asset_returns
    strat_returns = strat_returns.fillna(0.0)

    cumulative_wealth = (1 + strat_returns).cumprod()

    results = pd.DataFrame({
        "asset_returns": asset_returns,
        "strat_returns": strat_returns,
        "cumulative_wealth": cumulative_wealth
    })

    return results


def compute_metrics(results: pd.DataFrame) -> dict:
    """Computes Total Return, Annualized Sharpe Ratio, and Maximum Drawdown."""

    strat_returns = results["strat_returns"]
    wealth = results["cumulative_wealth"]

    total_return = wealth.iloc[-1] - 1

    mean_ret = strat_returns.mean()
    std_ret = strat_returns.std()
    sharpe_ratio = np.sqrt(252) * mean_ret / std_ret if std_ret != 0 else 0.0

    running_max = wealth.cummax()
    drawdown = (running_max - wealth) / running_max
    max_drawdown = drawdown.max()

    metrics = {
        "total_return": total_return,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown
    }

    return metrics


if __name__ == "__main__":

    np.random.seed(0)
    dummy_prices = pd.Series(100 * (1 + np.random.normal(0, 0.01, 300)).cumprod())
    dummy_positions = pd.Series(np.random.choice([0.0, 1.0], size=300))

    results = run_backtest(dummy_prices, dummy_positions)
    metrics = compute_metrics(results)

    print(results.tail())
    print("=" * 70)
    print(f"Total Return   : {metrics['total_return']:.2%}")
    print(f"Sharpe Ratio   : {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown   : {metrics['max_drawdown']:.2%}")
    print("=" * 70)