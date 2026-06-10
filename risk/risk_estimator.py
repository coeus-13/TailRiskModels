"""
risk/risk_estimator.py
======================
Risk Estimation Module
-----------------------
Provides Value at Risk (VaR) and Expected Shortfall (ES) estimation
from simulated return distributions produced by the three model classes.

Mathematical Background
-----------------------
Let  F(r)  be the distribution of one-period log-returns.

Value at Risk (VaR) — Basel III definition
    VaR_α  is the loss level exceeded with probability  (1 − α):

        VaR_α  = −inf{ r : F(r) ≥ 1 − α }
               = −Q_{1−α}(F)

    where Q_p denotes the p-th quantile.  The minus sign converts the
    quantile of a loss-tailed distribution into a positive loss figure.

    Convention used throughout this project:
        VaR is expressed as a POSITIVE number.
        A return of  r_t < −VaR_α  constitutes a VIOLATION (exceedance).

Expected Shortfall (ES / CVaR) — the "tail mean"
    ES_α  is the expected loss conditional on exceeding VaR_α:

        ES_α  = −E[r  |  r < −VaR_α]
              = −(1/(1−α)) ∫_{−∞}^{−VaR_α} r · f(r) dr

    From a Monte-Carlo sample { r_i }_{i=1}^{N}:

        ES_α  = −mean( r_i  |  r_i ≤ q_{1−α} )

    where  q_{1−α}  is the empirical (1−α)-quantile of the sample.

    ES is a coherent risk measure (Artzner et al., 1999) and has replaced
    VaR at the 97.5% level as the primary regulatory capital metric under
    FRTB (Fundamental Review of the Trading Book).

Historical Simulation (Benchmark)
    A non-parametric baseline: VaR and ES are read directly from the
    empirical quantiles of the calibration-window log-returns.
    No model is assumed; the past distribution IS the forecast.

Notes on Sign Convention
    Internally, all return arrays are signed (negative = loss).
    VaR and ES are returned as POSITIVE scalars (loss magnitudes).
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CONFIDENCE_LEVELS, N_SIMULATIONS


# ===========================================================================
# CORE FUNCTIONS (stateless, operate on numpy arrays)
# ===========================================================================

def compute_var(
    simulated_returns: np.ndarray,
    confidence_level:  float,
) -> float:
    """
    Compute Value at Risk from a sample of simulated returns.

    VaR_α = −Q_{1−α}(simulated_returns)

    Parameters
    ----------
    simulated_returns : np.ndarray  Shape (N,)  — signed log-returns
    confidence_level  : float       e.g. 0.95 for 95% VaR

    Returns
    -------
    float : VaR expressed as a POSITIVE loss magnitude
    """
    alpha     = 1.0 - confidence_level          # tail probability
    quantile  = np.quantile(simulated_returns, alpha)
    return -quantile                            # negate → positive loss


def compute_es(
    simulated_returns: np.ndarray,
    confidence_level:  float,
) -> float:
    """
    Compute Expected Shortfall (CVaR) from a sample of simulated returns.

    ES_α = −E[ r | r ≤ Q_{1−α}(r) ]
         = −mean( r_i  for all r_i ≤ q_{1−α} )

    Parameters
    ----------
    simulated_returns : np.ndarray  Shape (N,)  — signed log-returns
    confidence_level  : float       e.g. 0.95 for 95% ES

    Returns
    -------
    float : ES expressed as a POSITIVE loss magnitude (ES ≥ VaR always)
    """
    alpha       = 1.0 - confidence_level
    threshold   = np.quantile(simulated_returns, alpha)  # = −VaR_α
    tail_losses = simulated_returns[simulated_returns <= threshold]

    if len(tail_losses) == 0:
        # Degenerate case: return the VaR itself
        return -threshold

    return -np.mean(tail_losses)


def compute_var_es(
    simulated_returns: np.ndarray,
    confidence_levels: List[float] = None,
) -> Dict[str, float]:
    """
    Compute both VaR and ES at multiple confidence levels in one call.

    Parameters
    ----------
    simulated_returns : np.ndarray
    confidence_levels : list of floats  (default: [0.95, 0.99])

    Returns
    -------
    dict with keys like:
        'VaR_0.95', 'ES_0.95', 'VaR_0.99', 'ES_0.99'
    """
    if confidence_levels is None:
        confidence_levels = CONFIDENCE_LEVELS

    results = {}
    for cl in confidence_levels:
        key = f"{int(cl * 100)}"
        results[f"VaR_{key}"] = compute_var(simulated_returns, cl)
        results[f"ES_{key}"]  = compute_es(simulated_returns, cl)

    return results


def historical_var_es(
    historical_returns: np.ndarray,
    confidence_levels:  List[float] = None,
) -> Dict[str, float]:
    """
    Historical Simulation (HS) VaR and ES — non-parametric benchmark.

    Treats the empirical distribution of the calibration window as the
    forecast distribution.  No model assumption; past IS future.

    Limitation: assumes i.i.d. returns and ignores volatility clustering.

    Parameters
    ----------
    historical_returns : np.ndarray  Calibration-window log-returns
    confidence_levels  : list of floats

    Returns
    -------
    dict  (same key format as compute_var_es)
    """
    return compute_var_es(historical_returns, confidence_levels)


# ===========================================================================
# CLASS: RiskEstimator
# ===========================================================================

class RiskEstimator:
    """
    Orchestrates risk estimation across all three models (+ HS baseline)
    for a single calibration window.

    Usage
    -----
    >>> est = RiskEstimator(confidence_levels=[0.95, 0.99])
    >>> results = est.estimate_all(
    ...     cal_returns  = train_returns,
    ...     garch_model  = fitted_garch,
    ...     heston_model = fitted_heston,
    ...     merton_model = fitted_merton,
    ...     n_sims       = 10_000,
    ...     seed         = 42,
    ... )
    >>> results['GARCH']['VaR_95']   # float

    The returned dict has the structure:
        {
          'GARCH':  {'VaR_95': ..., 'ES_95': ..., 'VaR_99': ..., 'ES_99': ...},
          'Heston': { ... },
          'Merton': { ... },
          'HistSim':{ ... },
        }
    """

    def __init__(self, confidence_levels: List[float] = None):
        self.confidence_levels = confidence_levels or CONFIDENCE_LEVELS

    # -----------------------------------------------------------------------
    def estimate_all(
        self,
        cal_returns:   pd.Series,
        garch_model,
        heston_model,
        merton_model,
        n_sims: int = N_SIMULATIONS,
        seed:   int = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Run all four risk estimators and return a unified results dict.

        Parameters
        ----------
        cal_returns  : pd.Series    Calibration-window log-returns
        garch_model  : GARCHModel   (already fitted)
        heston_model : HestonModel  (already fitted)
        merton_model : MertonJumpDiffusion  (already fitted)
        n_sims       : int          MC paths per model
        seed         : int          Base RNG seed (offset per model)

        Returns
        -------
        dict[model_name -> dict[metric_name -> float]]
        """
        results = {}

        # --- GARCH Monte-Carlo ---
        garch_sims        = garch_model.simulate_returns(n_sims, seed=seed)
        results["GARCH"]  = compute_var_es(garch_sims, self.confidence_levels)

        # --- Heston Monte-Carlo ---
        heston_sims        = heston_model.simulate_returns(n_sims, seed=(seed or 0) + 1)
        results["Heston"]  = compute_var_es(heston_sims, self.confidence_levels)

        # --- Merton Monte-Carlo ---
        merton_sims        = merton_model.simulate_returns(n_sims, seed=(seed or 0) + 2)
        results["Merton"]  = compute_var_es(merton_sims, self.confidence_levels)

        # --- Historical Simulation (no model needed) ---
        results["HistSim"] = historical_var_es(
            cal_returns.values, self.confidence_levels
        )

        return results

    # -----------------------------------------------------------------------
    def results_to_dataframe(
        self,
        results: Dict[str, Dict[str, float]],
        test_date=None,
    ) -> pd.DataFrame:
        """
        Convert a single-window results dict to a tidy DataFrame row.

        Columns: model, date, VaR_95, ES_95, VaR_99, ES_99
        """
        rows = []
        for model_name, metrics in results.items():
            row = {"model": model_name, "date": test_date}
            row.update(metrics)
            rows.append(row)
        return pd.DataFrame(rows)


# ===========================================================================
# UTILITY: Aggregate backtest forecasts into a summary DataFrame
# ===========================================================================

def build_forecast_dataframe(forecast_records: list) -> pd.DataFrame:
    """
    Combine a list of per-step forecast dicts into a tidy long-format
    DataFrame suitable for backtesting and visualisation.

    Each record in forecast_records should be a dict with fields:
        date         : pd.Timestamp
        realised_r   : float           (actual log-return on that date)
        model        : str             ('GARCH', 'Heston', 'Merton', 'HistSim')
        VaR_95       : float
        ES_95        : float
        VaR_99       : float
        ES_99        : float

    Returns
    -------
    pd.DataFrame  sorted by (model, date)
    """
    df = pd.DataFrame(forecast_records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["model", "date"]).reset_index(drop=True)
    return df


# ===========================================================================
# STANDALONE TEST
# ===========================================================================

if __name__ == "__main__":
    # Quick functional test using synthetic Gaussian returns
    np.random.seed(0)
    fake_sims = np.random.normal(loc=0.0, scale=0.01, size=100_000)

    print("=== VaR / ES on N(0, 0.01²) synthetic returns ===")
    print(f"  Theoretical 95% VaR = {0.01 * 1.6449:.6f}")   # z_{0.05} = 1.6449
    print(f"  Estimated   95% VaR = {compute_var(fake_sims, 0.95):.6f}")

    # ES for normal: E[Z | Z < -z_α] = -φ(z_α)/α
    from scipy.stats import norm
    z95   = norm.ppf(0.05)
    es_th = 0.01 * norm.pdf(z95) / 0.05
    print(f"  Theoretical 95% ES  = {es_th:.6f}")
    print(f"  Estimated   95% ES  = {compute_es(fake_sims, 0.95):.6f}")

    multi = compute_var_es(fake_sims, [0.95, 0.99])
    print(f"\n  Full results: {multi}")
