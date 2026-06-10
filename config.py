"""
config.py
=========
Central configuration file for the Tail Risk Backtesting project.
All hyperparameters, ticker symbols, window sizes, and confidence levels
are defined here so that every module imports from a single source of truth.
"""

# ---------------------------------------------------------------------------
# DATA SETTINGS
# ---------------------------------------------------------------------------
TICKER          = "^NSEI"          # NIFTY 50 index  (use "^GSPC" for S&P 500)
START_DATE      = "2010-01-01"     # Full history start
END_DATE        = "2024-12-31"     # Full history end

# Rolling-window backtesting parameters
CALIBRATION_WINDOW = 500           # Trading days used to fit each model
TEST_STEP          = 1             # Days to step forward (1 = daily re-fit)

# ---------------------------------------------------------------------------
# RISK SETTINGS
# ---------------------------------------------------------------------------
CONFIDENCE_LEVELS = [0.95, 0.99]   # VaR / ES confidence levels
TRADING_DAYS_YEAR = 252            # Annualisation constant

# ---------------------------------------------------------------------------
# MODEL SIMULATION SETTINGS
# ---------------------------------------------------------------------------
N_SIMULATIONS = 10_000             # Monte-Carlo paths per forecast step
N_STEPS       = 1                  # Horizon in days (1-day VaR)
DT            = 1 / TRADING_DAYS_YEAR   # Daily time increment (in years)

# ---------------------------------------------------------------------------
# GARCH SETTINGS
# ---------------------------------------------------------------------------
GARCH_P = 1                        # GARCH lag order
GARCH_Q = 1                        # ARCH lag order
GARCH_DIST = "normal"              # Innovation distribution: 'normal' | 'studentst'

# ---------------------------------------------------------------------------
# HESTON MODEL — initial parameter guesses for optimiser
# ---------------------------------------------------------------------------
HESTON_INIT_PARAMS = {
    "kappa": 2.0,    # Mean-reversion speed  (κ)
    "theta": 0.04,   # Long-run variance      (θ)
    "sigma": 0.3,    # Vol-of-vol             (σ_v)
    "rho":  -0.7,    # Corr(S,V)              (ρ)
    "v0":   0.04,    # Initial variance       (V_0)
}

# Bounds for Heston MLE optimiser  (lower, upper) per parameter
HESTON_BOUNDS = [
    (1e-3, 20.0),   # kappa
    (1e-4, 1.0),    # theta
    (1e-4, 2.0),    # sigma
    (-0.99, 0.99),  # rho
    (1e-4, 2.0),    # v0
]

# ---------------------------------------------------------------------------
# MERTON JUMP-DIFFUSION — initial parameter guesses
# ---------------------------------------------------------------------------
MERTON_INIT_PARAMS = {
    "mu":      0.0,    # Drift of continuous part
    "sigma":   0.15,   # Diffusion vol (σ)
    "lam":     5.0,    # Jump intensity  (λ jumps/year)
    "mu_j":   -0.02,   # Mean jump size  (μ_J)
    "sigma_j": 0.05,   # Jump size std   (σ_J)
}

MERTON_BOUNDS = [
    (-1.0,  1.0),    # mu
    (1e-4,  2.0),    # sigma
    (1e-2, 50.0),    # lam
    (-1.0,  1.0),    # mu_j
    (1e-4,  1.0),    # sigma_j
]

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
RESULTS_DIR = "results/"
