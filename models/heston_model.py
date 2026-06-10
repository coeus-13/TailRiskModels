"""
models/heston_model.py
======================
Heston Stochastic Volatility Model Module
------------------------------------------
Implements parameter calibration via MLE on log-returns and path simulation
via the Euler-Maruyama discretisation scheme.

Mathematical Background
-----------------------
The Heston (1993) model describes joint dynamics of the asset price S_t
and its instantaneous variance V_t under the physical (real-world) measure P:

    dS_t / S_t  =  μ dt  +  √V_t dW^S_t                    (price SDE)
    dV_t        =  κ(θ − V_t) dt  +  σ_v √V_t dW^V_t       (variance SDE)
    dW^S_t · dW^V_t = ρ dt                                  (correlation)

Parameters
    μ      : drift of log-price (estimated as sample mean here)
    κ      : mean-reversion speed of variance (kappa)
    θ      : long-run mean of variance         (theta)
    σ_v    : volatility of variance, "vol-of-vol" (sigma)
    ρ      : correlation between price and variance Brownian motions (rho)
    V_0    : initial variance

Feller Condition
    2κθ > σ_v²   ↔   variance stays strictly positive
    (if violated, variance can hit zero and the discretisation may
     produce negative values; we apply  max(V, 0) as a floor)

Euler-Maruyama Discretisation  (Δt = 1/252)
    ln S_{t+Δt} = ln S_t  +  (μ − ½ V_t) Δt  +  √(V_t Δt) Z^S
    V_{t+Δt}    = V_t  +  κ(θ − V_t) Δt  +  σ_v √(V_t Δt) Z^V
                  V_{t+Δt} = max(V_{t+Δt}, 0)   (reflection at zero)

    where  Z^S, Z^V ~ N(0,1)  with  Corr(Z^S, Z^V) = ρ:
        Z^S = z_1
        Z^V = ρ·z_1 + √(1−ρ²)·z_2,    z_1,z_2 ~ i.i.d. N(0,1)

Calibration Strategy
    We maximise the approximate log-likelihood of the observed log-returns
    under the model. The 1-period log-return is:
        r_t ≈ (μ − ½ V_{t-1}) Δt + √(V_{t-1} Δt) · ε_t,  ε_t ~ N(0,1)
    This treats V_{t-1} as the current (observable-like) variance, which we
    proxy with the squared return on day t-1 (or the rolling realised variance).
    A proper particle-filter approach exists but is computationally prohibitive
    for a rolling window; this approximation is standard in the literature.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Tuple
import warnings
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    HESTON_INIT_PARAMS, HESTON_BOUNDS,
    N_SIMULATIONS, DT, TRADING_DAYS_YEAR
)


# ===========================================================================
# CLASS: HestonModel
# ===========================================================================

class HestonModel:
    """
    Heston (1993) stochastic volatility model.

    Calibration: Approximate MLE on realised log-returns, treating the
    variance process as observed (via rolling realised variance proxy).

    Simulation: Euler-Maruyama discretisation of the two-factor SDE.
    """

    def __init__(self):
        self.params = None   # dict: kappa, theta, sigma, rho, v0, mu
        self.is_fitted = False

    # -----------------------------------------------------------------------
    # PRIVATE: Log-Likelihood under Heston (approximate, for calibration)
    # -----------------------------------------------------------------------
    @staticmethod
    def _log_likelihood(
        theta_vec: np.ndarray,
        returns:   np.ndarray,
        dt:        float,
    ) -> float:
        """
        Approximate Gaussian log-likelihood for Heston log-returns.

        Given parameter vector θ = (κ, θ, σ_v, ρ, V_0), we propagate the
        variance process forward with Euler-Maruyama and evaluate the
        conditional density of each log-return.

        Conditional distribution of r_t given V_{t-1}:
            r_t | V_{t-1}  ~  N( (μ − ½ V_{t-1}) Δt,  V_{t-1} Δt )

        Log-likelihood (ignoring the μ contribution absorbed into the mean):
            ℓ = Σ_t  [ −½ ln(2π V_{t-1} Δt)
                       − (r_t − (μ_hat − ½ V_{t-1}) Δt)² / (2 V_{t-1} Δt) ]

        Parameters
        ----------
        theta_vec : array [kappa, theta_lr, sigma_v, rho, v0]
        returns   : array of log-returns
        dt        : time step in years

        Returns
        -------
        Negative log-likelihood (for minimisation)
        """
        kappa, theta_lr, sigma_v, rho, v0 = theta_vec

        # Quick feasibility checks — penalise infeasible regions
        if (kappa <= 0 or theta_lr <= 0 or sigma_v <= 0
                or v0 <= 0 or not (-1 < rho < 1)):
            return 1e10

        T   = len(returns)
        mu_hat = returns.mean() / dt   # annualised drift estimate

        # Propagate variance filter via Euler-Maruyama (deterministic part only
        # for speed — a common approximation in likelihood-based calibration)
        V = np.empty(T + 1)
        V[0] = v0
        for t in range(T):
            # Use the expected update (no noise in the filter step)
            V[t + 1] = max(
                V[t] + kappa * (theta_lr - V[t]) * dt,
                1e-8   # numerical floor
            )

        # Evaluate conditional log-likelihood
        V_prev    = V[:T]                         # V_{t-1} for each t
        mean_t    = (mu_hat - 0.5 * V_prev) * dt  # conditional mean
        var_t     = V_prev * dt                   # conditional variance

        # Guard against non-positive variances
        var_t = np.maximum(var_t, 1e-10)

        ll = -0.5 * (
            np.log(2 * np.pi * var_t)
            + (returns - mean_t) ** 2 / var_t
        )

        return -np.sum(ll)   # return NEGATIVE log-likelihood for minimiser

    # -----------------------------------------------------------------------
    def fit(
        self,
        returns:  pd.Series,
        dt:       float = DT,
        n_restarts: int = 3,
    ) -> "HestonModel":
        """
        Calibrate Heston parameters via MLE using L-BFGS-B.

        Multiple restarts are used to mitigate local-optima problems.

        Parameters
        ----------
        returns    : pd.Series   Log-returns series
        dt         : float       Time step (1/252 for daily)
        n_restarts : int         Number of random restarts

        Returns
        -------
        self (fluent)
        """
        r = returns.values.astype(float)

        # Initial parameter vector: [kappa, theta, sigma_v, rho, v0]
        x0 = [
            HESTON_INIT_PARAMS["kappa"],
            HESTON_INIT_PARAMS["theta"],
            HESTON_INIT_PARAMS["sigma"],
            HESTON_INIT_PARAMS["rho"],
            HESTON_INIT_PARAMS["v0"],
        ]

        best_result  = None
        best_nll     = np.inf
        rng          = np.random.default_rng(42)

        for attempt in range(n_restarts):
            if attempt > 0:
                # Perturb initial guess for subsequent restarts.
                # Draw uniformly inside each parameter's valid bounds.
                x0_perturbed = [
                    rng.uniform(b[0] + 1e-4, b[1] - 1e-4)
                    for b in HESTON_BOUNDS
                ]
                x_start = x0_perturbed
            else:
                x_start = x0

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = minimize(
                    fun=self._log_likelihood,
                    x0=x_start,
                    args=(r, dt),
                    method="L-BFGS-B",
                    bounds=HESTON_BOUNDS,
                    options={"maxiter": 500, "ftol": 1e-9},
                )

            if res.fun < best_nll:
                best_nll    = res.fun
                best_result = res

        kappa, theta_lr, sigma_v, rho, v0 = best_result.x

        self.params = {
            "kappa":   kappa,
            "theta":   theta_lr,
            "sigma":   sigma_v,
            "rho":     rho,
            "v0":      v0,
            "mu":      float(np.mean(r) / dt),   # annualised drift
            "neg_ll":  best_nll,
        }
        self.is_fitted = True

        return self  # fluent

    # -----------------------------------------------------------------------
    def simulate_returns(
        self,
        n_sims:   int   = N_SIMULATIONS,
        n_steps:  int   = 1,
        dt:       float = DT,
        seed:     int   = None,
    ) -> np.ndarray:
        """
        Simulate log-returns using Euler-Maruyama discretisation.

        For each of the n_sims paths we evolve the two-factor SDE for
        n_steps time steps and record the total log-return.

        Joint normal draws are constructed via Cholesky decomposition:
            [Z^S]   [1     0     ] [z_1]
            [Z^V] = [ρ  √(1-ρ²)] [z_2]

        Parameters
        ----------
        n_sims  : int   Number of independent simulation paths
        n_steps : int   Horizon in days (1 for 1-day VaR)
        dt      : float Time increment (years per step)
        seed    : int   RNG seed

        Returns
        -------
        np.ndarray, shape (n_sims,)  — simulated 1-period log-returns
        """
        if not self.is_fitted:
            raise RuntimeError("Call .fit() before .simulate_returns().")

        p       = self.params
        kappa   = p["kappa"]
        theta   = p["theta"]
        sigma_v = p["sigma"]
        rho     = p["rho"]
        v0      = p["v0"]
        mu      = p["mu"]

        rng = np.random.default_rng(seed)

        # Initialise paths
        log_price = np.zeros(n_sims)     # cumulative log-return per path
        V         = np.full(n_sims, v0)  # current variance for each path

        for _ in range(n_steps):
            # --- Draw correlated Brownian increments ----------------------
            # z1, z2 ~ i.i.d. N(0,1)
            z1 = rng.standard_normal(n_sims)
            z2 = rng.standard_normal(n_sims)

            # Correlated draws:
            #   W^S ~ z1
            #   W^V ~ ρ·z1 + √(1−ρ²)·z2
            dW_S = z1
            dW_V = rho * z1 + np.sqrt(1.0 - rho ** 2) * z2

            # --- Euler-Maruyama step for log-price ------------------------
            # d(ln S) = (μ − ½ V) dt + √V dW^S
            sqrt_V_dt = np.sqrt(np.maximum(V, 0.0) * dt)
            d_log_S   = (mu - 0.5 * V) * dt + sqrt_V_dt * dW_S
            log_price += d_log_S

            # --- Euler-Maruyama step for variance -------------------------
            # dV = κ(θ − V) dt + σ_v √V dW^V
            dV = (kappa * (theta - V) * dt
                  + sigma_v * sqrt_V_dt * dW_V)
            V  = np.maximum(V + dV, 0.0)   # full truncation scheme

        return log_price   # shape (n_sims,)

    # -----------------------------------------------------------------------
    def feller_condition_satisfied(self) -> bool:
        """
        Checks the Feller condition:  2κθ > σ_v²
        If True, the variance process remains strictly positive a.s.
        """
        p = self.params
        return 2 * p["kappa"] * p["theta"] > p["sigma"] ** 2

    # -----------------------------------------------------------------------
    def get_params_dict(self) -> dict:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")
        return {k: v for k, v in self.params.items() if k != "neg_ll"}

    # -----------------------------------------------------------------------
    def __repr__(self) -> str:
        if not self.is_fitted:
            return "HestonModel(unfitted)"
        p = self.params
        feller = "✓" if self.feller_condition_satisfied() else "✗"
        return (
            f"HestonModel | κ={p['kappa']:.4f}, θ={p['theta']:.4f}, "
            f"σ_v={p['sigma']:.4f}, ρ={p['rho']:.4f}, V₀={p['v0']:.4f} | "
            f"Feller={feller}"
        )


# ===========================================================================
# STANDALONE TEST
# ===========================================================================

if __name__ == "__main__":
    from data.data_loader import MarketDataLoader

    loader = MarketDataLoader().load()
    train, _ = loader.train_test_split()

    print("Fitting Heston model (this may take ~10s) ...")
    heston = HestonModel()
    heston.fit(train)

    print(heston)
    print(f"\nFeller condition satisfied: {heston.feller_condition_satisfied()}")

    sims = heston.simulate_returns(n_sims=10_000, seed=42)
    print(f"\nSimulated return distribution (N=10,000):")
    print(f"  Mean : {sims.mean():.6f}")
    print(f"  Std  : {sims.std():.6f}")
    print(f"  5th  : {np.percentile(sims, 5):.6f}")
    print(f"  1st  : {np.percentile(sims, 1):.6f}")
