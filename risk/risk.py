import pandas as pd
import numpy as np
from scipy.stats import norm


def compute_risk_summary(strat_returns: pd.Series, alpha: float = 0.05) -> dict:
    """
    Calcule le VaR historique, le VaR paramétrique (Gaussien) et
    l'Expected Shortfall (CVaR) sur les rendements de la stratégie.

    Parameters
    ----------
    strat_returns : pd.Series
        Série des rendements quotidiens de la stratégie (ex: results["strat_returns"]).
    alpha : float
        Niveau de risque (0.05 -> VaR/CVaR à 95%).

    Returns
    -------
    dict avec les clés : var_hist_95, var_gaussian_95, cvar_hist_95
    """

    # On enlève les NaN (souvent présents au début à cause du .shift(1) / rolling)
    returns = strat_returns.dropna()

    # --- 1. VaR historique ---
    # Qα = quantile empirique à alpha (ex: 5e percentile)
    # VaR = -Qα -> on l'exprime comme une perte positive
    q_alpha = np.percentile(returns, alpha * 100)
    var_hist = -q_alpha

    # --- 2. VaR paramétrique (Gaussien) ---
    mu = returns.mean()
    sigma = returns.std()
    z_alpha = norm.ppf(alpha)  # ex: -1.645 pour alpha=0.05
    var_gaussian = -(mu + z_alpha * sigma)

    # --- 3. Expected Shortfall (CVaR historique) ---
    # Moyenne des rendements qui sont pires que le VaR historique (donc <= q_alpha)
    tail_losses = returns[returns <= q_alpha]
    cvar_hist = -tail_losses.mean()

    return {
        "var_hist_95": var_hist,
        "var_gaussian_95": var_gaussian,
        "cvar_hist_95": cvar_hist,
    }


if __name__ == "__main__":
    # --- Test local avec données mockées (indépendant de backtest.py) ---
    np.random.seed(42)
    fake_returns = pd.Series(np.random.normal(loc=0.0005, scale=0.012, size=1000))

    summary = compute_risk_summary(fake_returns, alpha=0.05)

    print("=== Test risk.py (mocked data) ===")
    print(f"Historical VaR (95%)     : {summary['var_hist_95']:.4%}")
    print(f"Gaussian VaR (95%)       : {summary['var_gaussian_95']:.4%}")
    print(f"Expected Shortfall (95%) : {summary['cvar_hist_95']:.4%}")