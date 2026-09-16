"""
risk.py
Member C - Risk Analytics
Historical VaR, Parametric (Gaussian) VaR, and Expected Shortfall (CVaR)

Interface contract:
    compute_risk_summary(strat_returns: pd.Series, alpha: float = 0.05) -> dict
"""

import pandas as pd
import numpy as np
from scipy.stats import norm

def compute_risk_summary(strat_returns: pd.Series, alpha: float = 0.05)->dict:
    """Returns historical VaR, parametric Gaussian VaR, and Expected Shortfall."""
    mu = strat_returns.mean()
    std = strat_returns.std()
    historical_var = np.percentile(a=strat_returns, q=alpha*100)
    parametric_gaussian_var = norm.ppf(1-alpha, loc=mu, scale=std)
    es = -mu * + std *(norm.pdf(norm.ppf(alpha))/alpha)

    risk_summary = {"var_hist_95":historical_var, 
                    "parametric_gaussien":parametric_gaussian_var, 
                    "cvar_hist_95":es}

    return risk_summary

if __name__ == "__main__":

    df = pd.DataFrame(data=np.arange(stop=100))
    risk = compute_risk_summary(df)
    print(df)
    print("="*70)
    print(f"Var historique 95: {risk["var_hist_95"]}")
    print(f"parametric gaussien: {risk["parametric_gaussien"]}")
    print(f"Expected Shortfall: {risk["cvar_hist_95"]}")
    print("="*70)