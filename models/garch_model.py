"""
models/garch_model.py
=====================
GARCH(1,1) Model Module
-----------------------
Uses the `arch` library (Hansen & Lunde, 2005) for maximum-likelihood
estimation and provides a Monte-Carlo simulation interface that is
consistent with the Heston and Merton modules.

Mathematical Background
-----------------------
The GARCH(1,1) model (Bollerslev, 1986) decomposes the return process as:

    r_t = μ + ε_t                                        (mean equation)
    ε_t = σ_t · z_t,   z_t ~ i.i.d. N(0,1)             (innovation)
    σ²_t = ω + α · ε²_{t-1} + β · σ²_{t-1}             (variance equation)

Parameters
    ω > 0                       (intercept — unconditional variance floor)
    α ≥ 0                       (ARCH coefficient — impact of past shocks)
    β ≥ 0                       (GARCH coefficient — variance persistence)
    α + β < 1                   (stationarity / mean-reversion condition)

Unconditional (Long-Run) Variance
    σ²_∞ = ω / (1 − α − β)

One-Step-Ahead Variance Forecast
    σ²_{t+1|t} = ω + α · ε²_t + β · σ²_t

Monte-Carlo VaR Simulation
    For the 1-day horizon used here, the simulation simply draws
    N_SIM standard normal shocks z_i and computes:
        r_i = μ̂ + σ_{t+1|t} · z_i
    The α-quantile of {r_i} gives the VaR forecast.
"""

import numpy as np
import pandas as pd
from arch import arch_model
from arch.__future__ import reindexing        # suppress future warning
import warnings
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import GARCH_P, GARCH_Q, GARCH_DIST, N_SIMULATIONS, TRADING_DAYS_YEAR


# ===========================================================================
# CLASS: GARCHModel
# ===========================================================================

class GARCHModel:
    """
    Wraps the `arch` library GARCH(p,q) model with a clean interface for
    rolling-window calibration and Monte-Carlo risk forecasting.

    Parameters
    ----------
    p    : int   GARCH lag order (default 1)
    q    : int   ARCH  lag order (default 1)
    dist : str   Innovation distribution — 'normal' or 'studentst'
    """

    def __init__(
        self,
        p:    int = GARCH_P,
        q:    int = GARCH_Q,
        dist: str = GARCH_DIST,
    ):
        self.p    = p
        self.q    = q
        self.dist = dist

        # Populated after .fit()
        self.result    = None      # arch ModelResult object
        self.params    = None      # dict of fitted parameter values
        self.sigma2_t  = None      # In-sample conditional variance series
        self.fitted_mu = None      # Fitted mean (μ̂)

    # -----------------------------------------------------------------------
    def fit(self, returns: pd.Series, disp: str = "off") -> "GARCHModel":
        """
        Fit GARCH(p,q) to the supplied log-return series via MLE.

        The `arch` library maximises the log-likelihood:
            L(θ) = Σ_t [ -½ ln(2π) - ½ ln(σ²_t) - ε²_t / (2σ²_t) ]

        Parameters
        ----------
        returns : pd.Series  Log-returns (×100 scaling applied internally
                             to improve numerical conditioning).
        disp    : str        'off' suppresses optimiser output.

        Returns
        -------
        self  (fluent)
        """
        # Scale returns by 100 — improves numerical stability of MLE
        # (arch library works with percentage returns by convention)
        scaled = returns * 100.0

        am = arch_model(
            scaled,
            vol="Garch",
            p=self.p,
            q=self.q,
            dist=self.dist,
            mean="Constant",   # μ = constant
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.result = am.fit(disp=disp, show_warning=False)

        p = self.result.params

        # -------------------------------------------------------------------
        # Extract and store parameters
        # Note: ω, α, β are on the *scaled* (×100) return space.
        # For risk calculations we rescale back to the original space.
        # -------------------------------------------------------------------
        self.params = {
            "mu":    p["mu"]    / 100.0,     # mean  (rescaled)
            "omega": p["omega"] / 1e4,       # ω     (rescaled: /100² )
            "alpha": p["alpha[1]"],          # α     (dimensionless)
            "beta":  p["beta[1]"],           # β     (dimensionless)
        }

        # In-sample conditional variances (rescaled)
        self.sigma2_t  = self.result.conditional_volatility.values ** 2 / 1e4
        self.fitted_mu = self.params["mu"]

        return self  # fluent

    # -----------------------------------------------------------------------
    def forecast_variance(self) -> float:
        """
        One-step-ahead conditional variance forecast.

        σ²_{t+1|t} = ω + α · ε²_t + β · σ²_t

        Uses the arch library's built-in analytic forecasting.

        Returns
        -------
        float : σ²_{t+1|t}  (in original return units, not percentage)
        """
        if self.result is None:
            raise RuntimeError("Call .fit() before .forecast_variance().")

        # arch forecast returns 1-step variance in scaled space
        fc = self.result.forecast(horizon=1, reindex=False)
        # variance in (100×return)² units → convert back
        var_scaled = fc.variance.values[-1, 0]
        return var_scaled / 1e4

    # -----------------------------------------------------------------------
    def simulate_returns(
        self,
        n_sims: int = N_SIMULATIONS,
        seed:   int = None,
    ) -> np.ndarray:
        """
        Simulate N one-day-ahead returns using the GARCH forecast variance.

        r_i = μ̂  +  σ_{t+1|t} · z_i,    z_i ~ N(0,1)

        This is a parametric bootstrap:  we use the point estimate of the
        next-period volatility and draw innovation shocks.

        Parameters
        ----------
        n_sims : int   Number of simulation paths
        seed   : int   Random seed for reproducibility

        Returns
        -------
        np.ndarray  shape (n_sims,)  — simulated 1-day log-returns
        """
        if self.result is None:
            raise RuntimeError("Call .fit() before .simulate_returns().")

        rng = np.random.default_rng(seed)

        sigma2_next = self.forecast_variance()          # σ²_{t+1|t}
        sigma_next  = np.sqrt(sigma2_next)              # σ_{t+1|t}
        mu          = self.fitted_mu

        # Draw standard normal innovations
        z = rng.standard_normal(n_sims)

        # Simulated returns:  r = μ + σ·z
        simulated_returns = mu + sigma_next * z

        return simulated_returns

    # -----------------------------------------------------------------------
    def get_params_dict(self) -> dict:
        """Return the fitted parameters as a plain dictionary."""
        if self.params is None:
            raise RuntimeError("Call .fit() first.")
        return self.params.copy()

    # -----------------------------------------------------------------------
    def persistence(self) -> float:
        """
        Variance persistence  α + β.

        Values close to 1 indicate highly persistent volatility clustering
        (shocks decay slowly), which is characteristic of equity markets.
        """
        p = self.get_params_dict()
        return p["alpha"] + p["beta"]

    # -----------------------------------------------------------------------
    def unconditional_variance(self) -> float:
        """
        Long-run (unconditional) variance:  σ²_∞ = ω / (1 − α − β)

        Only defined when α + β < 1 (covariance stationarity).
        """
        p   = self.get_params_dict()
        den = 1.0 - p["alpha"] - p["beta"]
        if den <= 0:
            raise ValueError(
                "Model is non-stationary (α+β ≥ 1). "
                "Unconditional variance is undefined."
            )
        return p["omega"] / den

    # -----------------------------------------------------------------------
    def __repr__(self) -> str:
        if self.params is None:
            return "GARCHModel(unfitted)"
        p = self.params
        return (
            f"GARCHModel(p={self.p}, q={self.q}, dist={self.dist}) | "
            f"ω={p['omega']:.2e}, α={p['alpha']:.4f}, β={p['beta']:.4f}, "
            f"persistence={self.persistence():.4f}"
        )


# ===========================================================================
# STANDALONE TEST
# ===========================================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")

    from data.data_loader import MarketDataLoader

    loader = MarketDataLoader().load()
    train, _ = loader.train_test_split()

    garch = GARCHModel()
    garch.fit(train)

    print(garch)
    print(f"\nFitted parameters  : {garch.get_params_dict()}")
    print(f"Persistence (α+β)  : {garch.persistence():.6f}")
    print(f"Unconditional σ²   : {garch.unconditional_variance():.6f}")
    print(f"Unconditional σ (ann): "
          f"{np.sqrt(garch.unconditional_variance() * TRADING_DAYS_YEAR):.4f}")
    print(f"\nOne-step σ² forecast: {garch.forecast_variance():.8f}")

    sims = garch.simulate_returns(n_sims=10_000, seed=42)
    print(f"\nSimulated return distribution (N=10,000):")
    print(f"  Mean : {sims.mean():.6f}")
    print(f"  Std  : {sims.std():.6f}")
    print(f"  5th  : {np.percentile(sims, 5):.6f}")
    print(f"  1st  : {np.percentile(sims, 1):.6f}")
