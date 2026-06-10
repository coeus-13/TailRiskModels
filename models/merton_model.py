"""
models/merton_model.py
======================
Merton (1976) Jump-Diffusion Model Module
------------------------------------------
Calibrates jump parameters from log-returns via MLE and simulates
one-period return paths using compound Poisson jumps.

Mathematical Background
-----------------------
The Merton (1976) model augments the Black-Scholes GBM with a
compound Poisson jump component:

    dS_t / S_t  =  (μ − λ·k̄) dt  +  σ dW_t  +  (J_t − 1) dN_t

where:
    W_t  : standard Brownian motion
    N_t  : Poisson process with intensity λ  (jumps/year)
    J_t  : log-normal jump size:  ln(J_t) ~ N(μ_J, σ_J²)
    k̄    : E[J − 1] = exp(μ_J + ½σ_J²) − 1  (compensator)

The compensator term (λ·k̄) in the drift ensures the process is a
martingale under the risk-neutral measure (or equivalently, that we
account for the expected jump contribution to the drift).

Discrete-Time Log-Return (Δt = 1/252)
--------------------------------------
Over a small interval Δt:

    r_t = (μ − λ·k̄ − ½σ²) Δt  +  σ √Δt · Z  +  Σ_{j=1}^{N_t} Y_j

where:
    Z      ~ N(0,1)
    N_t    ~ Poisson(λ Δt)         (number of jumps in Δt)
    Y_j    ~ N(μ_J, σ_J²)          (individual log-jump sizes)

Conditional on N_t = n jumps, the log-return is Gaussian:
    r_t | N_t = n  ~  N( m_n,  s²_n )

    m_n = (μ − λ·k̄ − ½σ²) Δt + n · μ_J
    s²_n = σ² Δt + n · σ_J²

This leads to the Merton log-likelihood (finite mixture of Gaussians):

    p(r) = Σ_{n=0}^{N_max}  [ e^{-λΔt} (λΔt)^n / n! ]  ·  φ(r; m_n, s²_n)

where φ(·; μ, σ²) is the normal PDF.  We truncate at N_max = 10 jumps
(the Poisson tail beyond that is negligible for typical equity markets).

Calibration
-----------
Five parameters are estimated via MLE:
    θ = (μ, σ, λ, μ_J, σ_J)

We use scipy.optimize.minimize with L-BFGS-B and multiple restarts
to avoid local optima in this non-convex likelihood surface.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import factorial
import warnings
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MERTON_INIT_PARAMS, MERTON_BOUNDS, N_SIMULATIONS, DT


# ===========================================================================
# CLASS: MertonJumpDiffusion
# ===========================================================================

class MertonJumpDiffusion:
    """
    Merton (1976) Jump-Diffusion model.

    Calibration: MLE via Merton's finite-mixture Gaussian log-likelihood.
    Simulation : Compound Poisson + GBM Euler step.

    Parameters
    ----------
    n_max_jumps : int
        Truncation limit for Poisson sum in the likelihood (default 10).
    """

    def __init__(self, n_max_jumps: int = 10):
        self.n_max_jumps = n_max_jumps
        self.params      = None
        self.is_fitted   = False

    # -----------------------------------------------------------------------
    # PRIVATE: Merton Mixture Log-Likelihood
    # -----------------------------------------------------------------------
    def _log_likelihood(
        self,
        theta_vec: np.ndarray,
        returns:   np.ndarray,
        dt:        float,
    ) -> float:
        """
        Evaluate the negative Merton log-likelihood.

        p(r | θ) = Σ_{n=0}^{N_max}  w_n · φ(r; m_n, s_n²)

        where:
            w_n  = e^{-λΔt} (λΔt)^n / n!         (Poisson weights)
            m_n  = (μ − λ·k̄ − ½σ²) Δt + n·μ_J   (conditional mean)
            s²_n = σ²Δt + n·σ_J²                  (conditional variance)
            k̄    = exp(μ_J + ½σ_J²) − 1           (compensator)

        Parameters
        ----------
        theta_vec : [mu, sigma, lam, mu_j, sigma_j]
        returns   : array of log-returns
        dt        : time step in years

        Returns
        -------
        Negative log-likelihood (scalar)
        """
        mu, sigma, lam, mu_j, sigma_j = theta_vec

        # Feasibility guards
        if sigma <= 0 or sigma_j <= 0 or lam <= 0:
            return 1e10

        # Compensator:  k̄ = E[J−1] = exp(μ_J + ½σ_J²) − 1
        k_bar = np.exp(mu_j + 0.5 * sigma_j ** 2) - 1.0

        # Drift of log-price (including compensator correction)
        # ν = μ − λ·k̄ − ½σ²   (per unit time)
        nu = mu - lam * k_bar - 0.5 * sigma ** 2

        # Pre-compute Poisson weights  w_n = e^{-λΔt} (λΔt)^n / n!
        lam_dt    = lam * dt
        poisson_w = np.array([
            np.exp(-lam_dt) * (lam_dt ** n) / factorial(n, exact=False)
            for n in range(self.n_max_jumps + 1)
        ])
        # Normalise weights (avoids numerical underflow accumulation)
        poisson_w = np.maximum(poisson_w, 1e-300)

        # Accumulate mixture density over the n-jump components
        # For each observation r, we compute:
        #   p(r) = Σ_n w_n · N(r; m_n, s_n²)
        total_density = np.zeros(len(returns))

        for n in range(self.n_max_jumps + 1):
            m_n  = nu * dt + n * mu_j           # conditional mean
            s2_n = sigma ** 2 * dt + n * sigma_j ** 2   # conditional var
            s_n  = np.sqrt(max(s2_n, 1e-10))

            # Normal PDF:  φ(r; m_n, s_n) = 1/(s_n √2π) exp(−½((r−m_n)/s_n)²)
            z         = (returns - m_n) / s_n
            phi_n     = np.exp(-0.5 * z ** 2) / (s_n * np.sqrt(2 * np.pi))
            total_density += poisson_w[n] * phi_n

        # Guard against log(0)
        total_density = np.maximum(total_density, 1e-300)

        log_likelihood = np.sum(np.log(total_density))
        return -log_likelihood   # negate for minimisation

    # -----------------------------------------------------------------------
    def fit(
        self,
        returns:    pd.Series,
        dt:         float = DT,
        n_restarts: int   = 4,
    ) -> "MertonJumpDiffusion":
        """
        Fit Merton parameters via MLE using L-BFGS-B.

        Parameters
        ----------
        returns    : pd.Series  Log-returns
        dt         : float      Time step in years
        n_restarts : int        Number of random restarts

        Returns
        -------
        self (fluent)
        """
        r   = returns.values.astype(float)
        rng = np.random.default_rng(123)

        x0 = [
            MERTON_INIT_PARAMS["mu"],
            MERTON_INIT_PARAMS["sigma"],
            MERTON_INIT_PARAMS["lam"],
            MERTON_INIT_PARAMS["mu_j"],
            MERTON_INIT_PARAMS["sigma_j"],
        ]

        best_result = None
        best_nll    = np.inf

        for attempt in range(n_restarts):
            if attempt > 0:
                # Random restart: uniformly sample within bounds
                x_start = [
                    rng.uniform(b[0] + 1e-4, b[1])
                    for b in MERTON_BOUNDS
                ]
            else:
                x_start = x0

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = minimize(
                    fun=self._log_likelihood,
                    x0=x_start,
                    args=(r, dt),
                    method="L-BFGS-B",
                    bounds=MERTON_BOUNDS,
                    options={"maxiter": 1000, "ftol": 1e-10},
                )

            if res.fun < best_nll:
                best_nll    = res.fun
                best_result = res

        mu, sigma, lam, mu_j, sigma_j = best_result.x

        # Derived quantities
        k_bar = np.exp(mu_j + 0.5 * sigma_j ** 2) - 1.0
        expected_jump_return = lam * k_bar  # annualised expected jump contribution

        self.params = {
            "mu":       mu,
            "sigma":    sigma,
            "lam":      lam,
            "mu_j":     mu_j,
            "sigma_j":  sigma_j,
            "k_bar":    k_bar,
            "neg_ll":   best_nll,
        }
        self.is_fitted = True

        return self  # fluent

    # -----------------------------------------------------------------------
    def simulate_returns(
        self,
        n_sims:  int   = N_SIMULATIONS,
        n_steps: int   = 1,
        dt:      float = DT,
        seed:    int   = None,
    ) -> np.ndarray:
        """
        Simulate log-returns from the Merton Jump-Diffusion.

        Per time step Δt, the log-return increment is:
            r = (μ − λ·k̄ − ½σ²)·Δt  +  σ√Δt·Z  +  Σ_{j=1}^{N} Y_j

            N ~ Poisson(λΔt)          (number of jumps)
            Z ~ N(0,1)                (diffusion shock)
            Y_j ~ N(μ_J, σ_J²)       (jump sizes)

        Parameters
        ----------
        n_sims  : int   Paths to simulate
        n_steps : int   Time steps per path (1 for 1-day VaR)
        dt      : float Time increment in years
        seed    : int   RNG seed

        Returns
        -------
        np.ndarray, shape (n_sims,)
        """
        if not self.is_fitted:
            raise RuntimeError("Call .fit() before .simulate_returns().")

        p      = self.params
        mu     = p["mu"]
        sigma  = p["sigma"]
        lam    = p["lam"]
        mu_j   = p["mu_j"]
        sig_j  = p["sigma_j"]
        k_bar  = p["k_bar"]

        rng = np.random.default_rng(seed)

        # Drift term (per dt, including compensator)
        drift = (mu - lam * k_bar - 0.5 * sigma ** 2) * dt

        log_returns = np.zeros(n_sims)

        for _ in range(n_steps):
            # --- Diffusion component ---
            diffusion = sigma * np.sqrt(dt) * rng.standard_normal(n_sims)

            # --- Jump component ---
            # 1. Draw number of jumps per path:  N ~ Poisson(λΔt)
            n_jumps = rng.poisson(lam * dt, size=n_sims)

            # 2. For paths with jumps, draw aggregate jump sizes
            # Σ_{j=1}^{N} Y_j ~ N(N·μ_J, N·σ_J²)  (by additivity of normals)
            jump_component = np.where(
                n_jumps > 0,
                (n_jumps * mu_j
                 + np.sqrt(n_jumps * sig_j ** 2) * rng.standard_normal(n_sims)),
                0.0
            )

            log_returns += drift + diffusion + jump_component

        return log_returns   # shape (n_sims,)

    # -----------------------------------------------------------------------
    def jump_probability_per_day(self) -> float:
        """
        Expected probability of at least one jump occurring in a single day.

        P(N_Δt ≥ 1) = 1 − e^{−λΔt}
        """
        return 1.0 - np.exp(-self.params["lam"] * DT)

    # -----------------------------------------------------------------------
    def expected_jumps_per_year(self) -> float:
        """Returns λ — the annualised jump intensity."""
        return self.params["lam"]

    # -----------------------------------------------------------------------
    def get_params_dict(self) -> dict:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")
        return {k: v for k, v in self.params.items() if k != "neg_ll"}

    # -----------------------------------------------------------------------
    def __repr__(self) -> str:
        if not self.is_fitted:
            return "MertonJumpDiffusion(unfitted)"
        p = self.params
        return (
            f"MertonJumpDiffusion | σ={p['sigma']:.4f}, λ={p['lam']:.2f}/yr, "
            f"μ_J={p['mu_j']:.4f}, σ_J={p['sigma_j']:.4f}, "
            f"k̄={p['k_bar']:.4f}"
        )


# ===========================================================================
# STANDALONE TEST
# ===========================================================================

if __name__ == "__main__":
    from data.data_loader import MarketDataLoader

    loader = MarketDataLoader().load()
    train, _ = loader.train_test_split()

    print("Fitting Merton Jump-Diffusion model (this may take ~15s) ...")
    merton = MertonJumpDiffusion()
    merton.fit(train)

    print(merton)
    print(f"\nJump probability per day: {merton.jump_probability_per_day():.4f}")
    print(f"Expected jumps per year : {merton.expected_jumps_per_year():.2f}")

    sims = merton.simulate_returns(n_sims=10_000, seed=42)
    print(f"\nSimulated return distribution (N=10,000):")
    print(f"  Mean  : {sims.mean():.6f}")
    print(f"  Std   : {sims.std():.6f}")
    print(f"  5th   : {np.percentile(sims, 5):.6f}")
    print(f"  1st   : {np.percentile(sims, 1):.6f}")
    print(f"  Kurt  : {pd.Series(sims).kurt():.4f}   (excess; > 0 = fat tails)")
