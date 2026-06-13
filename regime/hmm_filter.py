"""
regime/hmm_filter.py
====================
Volatility Regime Filter — 2-State Gaussian HMM
-------------------------------------------------
Fits a Hidden Markov Model on a log-transformed rolling realised variance
proxy to classify each trading day as either a Calm (low-vol) or Stress
(high-vol) regime.  The decoded regime sequence is then used by
Backtester.run_conditional_tests() to split the backtest results.

Mathematical Background
-----------------------
Observable feature:
    x_t = ln( RV₅_t + ε )

    where  RV₅_t = (1/5) Σ_{i=0}^{4} r²_{t-i}   (5-day rolling RV proxy)
    and    ε = 1e-8  (floor to avoid log(0))

We take the log of RV because log-variance is more symmetric and
approximately Gaussian — better suited to the Gaussian emission model.

HMM Specification:
    State space:   S ∈ {0, 1}  (Calm, Stress)
    Emissions:     x_t | S_t = k  ~  N( μ_k, σ²_k )
    Transitions:   P(S_t | S_{t-1}) = A   (2×2 row-stochastic matrix)
    Initial dist:  π₀

Training (Baum-Welch EM):
    Maximises  E_q[log P(X, S | θ)]  over θ = (π₀, A, μ, σ²)

Decoding (Viterbi):
    S* = argmax_{S₁…S_T}  P(S₁…S_T | X, θ*)
    Returns the globally most probable state path.

State labelling convention (enforced post-fit):
    State 0 always = Calm   (μ₀ < μ₁)
    State 1 always = Stress (μ₁ > μ₀)
    Labels are swapped if the HMM assigns them in reverse order.

Expected regime characteristics on NIFTY50 / S&P500:
    Calm   : μ_logRV ≈ −9 to −8,  low persistence of violations
    Stress : μ_logRV ≈ −7 to −6,  high persistence, volatility clustering
"""

import warnings
import numpy as np
import pandas as pd
from typing import Tuple, Optional

try:
    from hmmlearn.hmm import GaussianHMM
except ImportError:
    raise ImportError(
        "hmmlearn is required for the RegimeFilter. "
        "Install with:  pip install hmmlearn"
    )

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class RegimeFilter:
    """
    2-state Gaussian HMM volatility regime classifier.

    Workflow
    --------
    >>> rf = RegimeFilter(n_iter=200, random_state=42)
    >>> rf.fit(loader.log_returns)
    >>> regimes = rf.predict_regimes(loader.log_returns)
    >>> print(rf.regime_summary())

    Attributes (available after fit)
    -----------
    model_        : GaussianHMM    The fitted hmmlearn model
    _calm_state   : int            Index (0 or 1) of the calm state
    _stress_state : int            Index (0 or 1) of the stress state
    """

    def __init__(
        self,
        rv_window:    int  = 5,
        n_iter:       int  = 200,
        random_state: int  = 42,
        n_restarts:   int  = 5,
    ):
        """
        Parameters
        ----------
        rv_window    : int  Rolling window for realised variance (days)
        n_iter       : int  Maximum Baum-Welch EM iterations
        random_state : int  RNG seed for reproducibility
        n_restarts   : int  HMM random restarts (keep best log-likelihood)
        """
        self.rv_window    = rv_window
        self.n_iter       = n_iter
        self.random_state = random_state
        self.n_restarts   = n_restarts

        self.model_       = None
        self._calm_state  = 0
        self._stress_state= 1
        self._feature_index: Optional[pd.DatetimeIndex] = None

    # -----------------------------------------------------------------------
    def _build_features(self, returns: pd.Series) -> Tuple[np.ndarray, pd.DatetimeIndex]:
        """
        Compute log rolling realised variance feature vector.

        x_t = ln( (1/W) Σ_{i=0}^{W-1} r²_{t-i}  + ε )

        Parameters
        ----------
        returns : pd.Series  Log-returns with DatetimeIndex

        Returns
        -------
        X     : np.ndarray   shape (T_valid, 1)  — feature matrix for hmmlearn
        index : DatetimeIndex  — corresponding dates (NaN-leading rows dropped)
        """
        rv   = (returns ** 2).rolling(self.rv_window).mean()
        log_rv = np.log(rv + 1e-8)
        log_rv = log_rv.dropna()

        X     = log_rv.values.reshape(-1, 1).astype(float)
        index = log_rv.index
        return X, index

    # -----------------------------------------------------------------------
    def fit(self, returns: pd.Series) -> "RegimeFilter":
        """
        Fit the 2-state HMM via Baum-Welch EM with multiple restarts.

        The best model (highest log-likelihood across restarts) is retained.
        State labels are then normalised so that State 0 = Calm.

        Parameters
        ----------
        returns : pd.Series  Full log-return series (DatetimeIndex)

        Returns
        -------
        self (fluent interface)
        """
        X, self._feature_index = self._build_features(returns)

        best_model  = None
        best_score  = -np.inf
        rng         = np.random.default_rng(self.random_state)

        for attempt in range(self.n_restarts):
            seed = int(rng.integers(0, 99999))
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    hmm = GaussianHMM(
                        n_components=2,
                        covariance_type="diag",
                        n_iter=self.n_iter,
                        random_state=seed,
                        tol=1e-4,
                    )
                    hmm.fit(X)
                    score = hmm.score(X)

                if score > best_score:
                    best_score = score
                    best_model = hmm

            except Exception as e:
                warnings.warn(f"[RegimeFilter] Restart {attempt} failed: {e}")
                continue

        if best_model is None:
            raise RuntimeError(
                "All HMM restarts failed. Try reducing n_restarts or "
                "check that your return series has no NaN values."
            )

        self.model_ = best_model

        # ── Enforce labelling convention: State 0 = Calm (lower log-RV) ───
        means = self.model_.means_.flatten()  # [μ_state0, μ_state1]
        if means[0] > means[1]:
            # State 0 has HIGHER variance → it's actually the stress state
            # Swap internal indices so that self._calm_state always has lower μ
            self._calm_state   = 1
            self._stress_state = 0
        else:
            self._calm_state   = 0
            self._stress_state = 1

        return self

    # -----------------------------------------------------------------------
    def predict_regimes(self, returns: pd.Series) -> pd.Series:
        """
        Decode the most probable regime sequence via the Viterbi algorithm.

        Remaps HMM internal state indices so the output always uses the
        convention:  0 = Calm,  1 = Stress  (independent of fit randomness).

        Parameters
        ----------
        returns : pd.Series  Log-returns (may be the same series used for fit)

        Returns
        -------
        pd.Series[int]
            Values in {0, 1},  index = DatetimeIndex matching the valid
            (non-NaN) portion of the feature series.
            0 = Calm,   1 = Stress
        """
        if self.model_ is None:
            raise RuntimeError("Call .fit() before .predict_regimes().")

        X, index = self._build_features(returns)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw_states = self.model_.predict(X)  # HMM internal state ids

        # Remap to canonical labelling (0=Calm, 1=Stress)
        canonical = np.where(raw_states == self._calm_state, 0, 1)

        return pd.Series(
            canonical.astype(int),
            index=index,
            name="regime_code",
        )

    # -----------------------------------------------------------------------
    def state_means(self) -> np.ndarray:
        """
        Return HMM emission means in canonical order [μ_calm, μ_stress].

        These are means of log(RV₅), so more negative = lower variance.
        Used by Backtester.print_conditional_report() for the header block.
        """
        if self.model_ is None:
            raise RuntimeError("Call .fit() first.")
        means = self.model_.means_.flatten()
        # Return in canonical order regardless of internal indexing
        return np.array([means[self._calm_state], means[self._stress_state]])

    # -----------------------------------------------------------------------
    def transition_matrix(self) -> np.ndarray:
        """
        Return the transition probability matrix in canonical order.

        Output shape (2, 2):
            A[i, j] = P(next state = j | current state = i)
            Row 0 → from Calm,   Row 1 → from Stress
            Col 0 → to Calm,     Col 1 → to Stress

        This canonical reordering is applied so that the report always
        prints Calm→Calm, Calm→Stress, Stress→Calm, Stress→Stress
        regardless of which internal state index was assigned to each regime.
        """
        if self.model_ is None:
            raise RuntimeError("Call .fit() first.")

        A_raw = self.model_.transmat_   # shape (2, 2) in raw HMM indexing

        # Build reorder index: canonical state k maps to raw state _idx[k]
        _idx = [self._calm_state, self._stress_state]

        A_canonical = np.array([
            [A_raw[_idx[i], _idx[j]] for j in range(2)]
            for i in range(2)
        ])
        return A_canonical

    # -----------------------------------------------------------------------
    def expected_regime_durations(self) -> Tuple[float, float]:
        """
        Expected number of consecutive days in each regime.

        For a two-state Markov chain:
            E[duration | state k] = 1 / (1 − A_{kk})

        Returns
        -------
        (expected_calm_days, expected_stress_days)
        """
        A = self.transition_matrix()
        calm_dur   = 1.0 / (1.0 - A[0, 0] + 1e-10)
        stress_dur = 1.0 / (1.0 - A[1, 1] + 1e-10)
        return calm_dur, stress_dur

    # -----------------------------------------------------------------------
    def stationary_distribution(self) -> np.ndarray:
        """
        Unconditional (stationary) probability of each regime.

        Computed as the left eigenvector of A corresponding to eigenvalue 1:
            π A = π,   Σπ_k = 1

        Or equivalently for a 2-state chain:
            π_calm   = p_{10} / (p_{01} + p_{10})
            π_stress = p_{01} / (p_{01} + p_{10})

        where p_{01} = A[0,1],  p_{10} = A[1,0].

        Returns
        -------
        np.ndarray [π_calm, π_stress]
        """
        A       = self.transition_matrix()
        p01     = A[0, 1]   # calm → stress
        p10     = A[1, 0]   # stress → calm
        denom   = p01 + p10 + 1e-10
        pi_calm   = p10 / denom
        pi_stress = p01 / denom
        return np.array([pi_calm, pi_stress])

    # -----------------------------------------------------------------------
    def regime_summary(self) -> str:
        """
        Return a formatted string summary of the fitted HMM.
        Includes emission parameters, transition matrix, and regime statistics.
        """
        if self.model_ is None:
            return "RegimeFilter(unfitted)"

        means  = self.state_means()
        covars = np.array([
            self.model_.covars_.flatten()[self._calm_state],
            self.model_.covars_.flatten()[self._stress_state],
        ])
        A      = self.transition_matrix()
        stationary  = self.stationary_distribution()
        calm_dur, stress_dur = self.expected_regime_durations()

        lines = [
            "=" * 62,
            "  HMM Volatility Regime Filter — Summary",
            "=" * 62,
            f"  Feature       : log(RV₅),  window = {self.rv_window} days",
            f"  Restarts      : {self.n_restarts}  "
            f"(best log-lik = {self.model_.score(self.model_.means_):+.2f})",
            "",
            f"  {'State':10s}  {'log-RV mean':>14s}  {'log-RV std':>12s}  "
            f"{'Stationary π':>14s}  {'Exp. Duration':>14s}",
            f"  {'─' * 68}",
            f"  {'Calm (0)':10s}  {means[0]:>14.4f}  {np.sqrt(covars[0]):>12.4f}  "
            f"{stationary[0]:>14.4f}  {calm_dur:>12.1f} days",
            f"  {'Stress (1)':10s}  {means[1]:>14.4f}  {np.sqrt(covars[1]):>12.4f}  "
            f"{stationary[1]:>14.4f}  {stress_dur:>12.1f} days",
            "",
            f"  Transition Matrix A  (row = from, col = to):",
            f"    Calm   → [Calm: {A[0,0]:.4f},  Stress: {A[0,1]:.4f}]",
            f"    Stress → [Calm: {A[1,0]:.4f},  Stress: {A[1,1]:.4f}]",
            "=" * 62,
        ]
        return "\n".join(lines)

    # -----------------------------------------------------------------------
    def __repr__(self) -> str:
        if self.model_ is None:
            return "RegimeFilter(unfitted)"
        means = self.state_means()
        A     = self.transition_matrix()
        return (
            f"RegimeFilter(fitted) | "
            f"μ_calm={means[0]:.3f}, μ_stress={means[1]:.3f} | "
            f"P(S→S)={A[1,1]:.3f}"
        )
