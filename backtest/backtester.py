"""
backtest/backtester.py
======================
Backtesting Module — Core Statistical Contribution
----------------------------------------------------
Implements the two industry-standard VaR backtesting frameworks AND a
HMM-driven conditional (regime-split) backtesting layer.

Module Structure
----------------
  PART I  — Core statistical test functions (stateless)
      kupiec_pof_test()
      christoffersen_independence_test()
      christoffersen_cc_test()

  PART II — Backtester class
      run_all_tests()            Unconditional backtest (original)
      run_conditional_tests()    NEW — regime-split backtest via HMM
      violation_series()
      print_report()
      print_conditional_report() NEW

References
----------
  Kupiec, P. (1995). "Techniques for Verifying the Accuracy of Risk
    Measurement Models." Journal of Derivatives, 3(2), 73–84.

  Christoffersen, P. (1998). "Evaluating Interval Forecasts."
    International Economic Review, 39(4), 841–862.

  Hamilton, J.D. (1989). "A New Approach to the Economic Analysis of
    Nonstationary Time Series." Econometrica, 57(2), 357–384.

Mathematical Background — Statistical Tests
--------------------------------------------

--- Kupiec POF Test (Unconditional Coverage) ---

Null hypothesis H₀: p̂ = p = 1 − α
  (the empirical violation rate equals the theoretical tail probability)

Let:
    T   = total number of out-of-sample observations
    V   = number of VaR violations  (r_t < −VaR_{α,t})
    p̂   = V / T                    (empirical failure rate)
    p   = 1 − α                    (theoretical failure rate)

Log-Likelihood Ratio:
    LR_uc = −2 [ V·ln(p) + (T−V)·ln(1−p)
                 − V·ln(p̂) − (T−V)·ln(1−p̂) ]

    Under H₀:  LR_uc ~ χ²(1)

--- Christoffersen Independence Test ---

Violations I_t ∈ {0,1} follow a 1st-order Markov chain under H₁.
Transition counts:
    n₀₀, n₀₁, n₁₀, n₁₁   (I_{t-1}→I_t transitions)

Transition probabilities under H₁:
    π₀₁ = n₀₁ / (n₀₀ + n₀₁)
    π₁₁ = n₁₁ / (n₁₀ + n₁₁)
    π    = (n₀₁ + n₁₁) / T̃    (pooled rate under H₀)

Independence LR statistic:
    LR_ind = −2 [ (n₀₀+n₁₀)·ln(1−π) + (n₀₁+n₁₁)·ln(π)
                  − n₀₀·ln(1−π₀₁) − n₀₁·ln(π₀₁)
                  − n₁₀·ln(1−π₁₁) − n₁₁·ln(π₁₁) ]

    Under H₀:  LR_ind ~ χ²(1)

--- Christoffersen Conditional Coverage (CC) Test ---

    LR_cc = LR_uc + LR_ind  ~  χ²(2)   under H₀

Mathematical Background — HMM Regime Classification
-----------------------------------------------------
A 2-state Gaussian HMM is fit on the log of 5-day rolling realised
variance (RV₅):

    Feature:  x_t = ln( RV₅_t + ε ),   RV₅_t = (1/5) Σ_{i=0}^{4} r²_{t-i}

The HMM emission model per state k:
    x_t | S_t = k  ~  N( μ_k, σ²_k )

with first-order Markov transitions:
    P( S_t = j | S_{t-1} = i ) = A_{ij}

State labelling convention:
    State 0 = Calm   (lower emission mean μ_0 < μ_1)
    State 1 = Stress (higher emission mean μ_1)

If the fitted HMM assigns μ_0 > μ_1, labels are swapped to enforce
this convention.  This makes all downstream reporting stable regardless
of HMM initialisation randomness.

The Viterbi algorithm (called internally by GaussianHMM.predict()) then
decodes the most probable state sequence S₁, …, S_T given the full
observation sequence — this is the globally optimal path, not a
greedy step-by-step assignment.

Regime-Conditional Testing Interpretation
------------------------------------------
For each (model, confidence level) pair we compute:

    Calm regime:
        T_calm, V_calm, p̂_calm = V_calm / T_calm
        → run Kupiec + CC on this sub-sequence

    Stress regime:
        T_stress, V_stress, p̂_stress = V_stress / T_stress
        → run Kupiec + CC on this sub-sequence

Key diagnostic:

    If π₁₁(stress) >> π₁₁(calm):
        Violation clustering is concentrated in the stress regime.
        GARCH is expected to show this (vol clustering → violation
        clustering).  Merton jump-diffusion may show lower stress
        π₁₁ because jumps are i.i.d. by construction.

    If p̂_stress >> p̂_calm:
        The model systematically under-estimates tail risk in stress
        periods — this is the structural failure of GARCH identified
        in the unconditional backtest, now localised to its regime.

Small-Sample Warning
--------------------
The χ²(1) approximation for LR_uc and LR_ind requires sufficient
observations per cell.  For regime slices with T < MIN_OBS_FOR_TEST
(default 30) the test statistics are computed but flagged with a
warning, as the asymptotic distribution may not apply.  The raw
violation counts are always valid regardless of sample size.
"""

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import chi2

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CONFIDENCE_LEVELS

# Minimum observations for χ² approximation to be considered reliable
MIN_OBS_FOR_TEST = 30


# ============================================================================
# PART I — CORE STATISTICAL TEST FUNCTIONS  (stateless)
# ============================================================================

def kupiec_pof_test(
    violations:       np.ndarray,
    confidence_level: float,
) -> Dict:
    """
    Kupiec (1995) Proportion of Failures (POF) test.

    Unconditional coverage LR test:
        LR_uc = −2 [ V·ln(p) + (T−V)·ln(1−p) − V·ln(p̂) − (T−V)·ln(1−p̂) ]
        LR_uc ~ χ²(1)  under H₀: p̂ = p

    Parameters
    ----------
    violations        : np.ndarray  Binary (0/1); 1 = VaR breach
    confidence_level  : float       e.g. 0.95 for 95% VaR

    Returns
    -------
    dict
        T, V, p_hat, p_expected, LR_uc, p_value, reject_H0,
        small_sample (bool — True if T < MIN_OBS_FOR_TEST)
    """
    T     = len(violations)
    V     = int(np.sum(violations))
    p     = 1.0 - confidence_level          # theoretical tail probability
    p_hat = V / T if T > 0 else 0.0

    small_sample = T < MIN_OBS_FOR_TEST

    eps       = 1e-10
    p_hat_s   = np.clip(p_hat, eps, 1 - eps)

    ll_H0 = V * np.log(p + eps) + (T - V) * np.log(1 - p + eps)
    ll_H1 = V * np.log(p_hat_s) + (T - V) * np.log(1 - p_hat_s)

    LR_uc     = max(-2.0 * (ll_H0 - ll_H1), 0.0)
    p_value   = 1.0 - chi2.cdf(LR_uc, df=1)
    reject_H0 = p_value < 0.05

    return {
        "T":            T,
        "V":            V,
        "p_hat":        round(p_hat,  6),
        "p_expected":   round(p,      6),
        "LR_uc":        round(LR_uc,  4),
        "p_value":      round(p_value, 4),
        "reject_H0":    reject_H0,
        "small_sample": small_sample,
    }


def christoffersen_independence_test(
    violations: np.ndarray,
) -> Dict:
    """
    Christoffersen (1998) independence component.

    Tests whether violations form an i.i.d. Bernoulli sequence (H₀)
    vs. a first-order Markov chain (H₁).

    LR_ind ~ χ²(1)  under H₀

    Parameters
    ----------
    violations : np.ndarray   Binary sequence

    Returns
    -------
    dict
        n00, n01, n10, n11, pi01, pi11, pi,
        LR_ind, p_value, reject_H0, small_sample
    """
    I      = violations.astype(int)
    I_prev = I[:-1]
    I_curr = I[1:]

    n00 = int(np.sum((I_prev == 0) & (I_curr == 0)))
    n01 = int(np.sum((I_prev == 0) & (I_curr == 1)))
    n10 = int(np.sum((I_prev == 1) & (I_curr == 0)))
    n11 = int(np.sum((I_prev == 1) & (I_curr == 1)))

    T_trans      = n00 + n01 + n10 + n11
    small_sample = T_trans < MIN_OBS_FOR_TEST

    eps   = 1e-10
    pi01  = n01 / (n00 + n01 + eps)
    pi11  = n11 / (n10 + n11 + eps)
    pi    = (n01 + n11) / (T_trans + eps)

    pi_s   = np.clip(pi,   eps, 1 - eps)
    pi01_s = np.clip(pi01, eps, 1 - eps)
    pi11_s = np.clip(pi11, eps, 1 - eps)

    ll_H0 = ((n00 + n10) * np.log(1 - pi_s)
             + (n01 + n11) * np.log(pi_s))

    ll_H1 = (n00 * np.log(1 - pi01_s) + n01 * np.log(pi01_s)
             + n10 * np.log(1 - pi11_s) + n11 * np.log(pi11_s))

    LR_ind    = max(-2.0 * (ll_H0 - ll_H1), 0.0)
    p_value   = 1.0 - chi2.cdf(LR_ind, df=1)
    reject_H0 = p_value < 0.05

    return {
        "n00": n00, "n01": n01, "n10": n10, "n11": n11,
        "pi01":        round(pi01,    6),
        "pi11":        round(pi11,    6),
        "pi":          round(pi,      6),
        "LR_ind":      round(LR_ind,  4),
        "p_value":     round(p_value, 4),
        "reject_H0":   reject_H0,
        "small_sample": small_sample,
    }


def christoffersen_cc_test(
    violations:       np.ndarray,
    confidence_level: float,
) -> Dict:
    """
    Christoffersen (1998) Conditional Coverage (CC) test.

        LR_cc = LR_uc + LR_ind  ~  χ²(2)

    Returns a merged dict of POF + independence results plus:
        LR_cc, cc_pvalue, cc_reject
    """
    pof = kupiec_pof_test(violations, confidence_level)
    ind = christoffersen_independence_test(violations)

    LR_cc     = pof["LR_uc"] + ind["LR_ind"]
    cc_pvalue = 1.0 - chi2.cdf(LR_cc, df=2)
    cc_reject = cc_pvalue < 0.05

    # Prefix independence keys to avoid name collision when merging
    combined = {**pof, **{f"ind_{k}": v for k, v in ind.items()}}
    combined["LR_cc"]     = round(LR_cc,     4)
    combined["cc_pvalue"] = round(cc_pvalue, 4)
    combined["cc_reject"] = cc_reject
    return combined


# ============================================================================
# PART II — BACKTESTER CLASS
# ============================================================================

class Backtester:
    """
    Runs unconditional and HMM regime-conditional VaR backtests.

    The forecast DataFrame produced by main.py's rolling loop must have:
        date, model, realised_r, VaR_95, ES_95, VaR_99, ES_99

    Two public test runners:
        run_all_tests()           — unconditional (original behaviour)
        run_conditional_tests()   — regime-split via HMM filter

    Typical usage
    -------------
    >>> bt = Backtester(forecast_df)
    >>> uncond = bt.run_all_tests()
    >>> bt.print_report(uncond)

    >>> from regime.hmm_filter import RegimeFilter
    >>> rf = RegimeFilter().fit(loader.log_returns)
    >>> cond = bt.run_conditional_tests(rf, loader.log_returns)
    >>> bt.print_conditional_report(cond)
    """

    def __init__(
        self,
        forecast_df:       pd.DataFrame,
        confidence_levels: List[float] = None,
    ):
        self.df  = forecast_df.copy()
        self.df["date"] = pd.to_datetime(self.df["date"])
        self.cls = confidence_levels or CONFIDENCE_LEVELS
        self._validate()

    # -----------------------------------------------------------------------
    def _validate(self) -> None:
        required = {"date", "model", "realised_r"}
        for cl in self.cls:
            pct = int(cl * 100)
            required |= {f"VaR_{pct}", f"ES_{pct}"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"forecast_df is missing columns: {missing}")

    # -----------------------------------------------------------------------
    def _get_violations(
        self,
        model_df:         pd.DataFrame,
        confidence_level: float,
    ) -> np.ndarray:
        """
        Binary violation indicator:
            I_t = 1  iff  realised_r_t < −VaR_{α,t}

        Returns np.ndarray  dtype=int, shape (T,)
        """
        var_col    = f"VaR_{int(confidence_level * 100)}"
        violations = (
            model_df["realised_r"].values < -model_df[var_col].values
        ).astype(int)
        return violations

    # -----------------------------------------------------------------------
    def _build_result_row(
        self,
        model_name:       str,
        confidence_level: float,
        model_df:         pd.DataFrame,
        regime_label:     str = "All",
    ) -> Dict:
        """
        Compute all statistics for one (model, CL, regime) combination
        and return a flat dict ready for DataFrame assembly.
        """
        pct     = int(confidence_level * 100)
        var_col = f"VaR_{pct}"
        es_col  = f"ES_{pct}"

        violations = self._get_violations(model_df, confidence_level)
        cc         = christoffersen_cc_test(violations, confidence_level)

        mean_var = model_df[var_col].mean()
        mean_es  = model_df[es_col].mean()
        es_var_r = mean_es / mean_var if mean_var > 0 else np.nan

        return {
            # Identification
            "model":            model_name,
            "regime":           regime_label,
            "confidence_level": confidence_level,
            # Violation counts
            "T":                cc["T"],
            "V":                cc["V"],
            "p_hat":            cc["p_hat"],
            "p_expected":       cc["p_expected"],
            # Kupiec POF
            "LR_uc":            cc["LR_uc"],
            "uc_pvalue":        cc["p_value"],
            "uc_reject":        cc["reject_H0"],
            # Independence (transition counts + probabilities)
            "n11":              cc["ind_n11"],
            "pi11":             cc["ind_pi11"],
            "LR_ind":           cc["ind_LR_ind"],
            "ind_pvalue":       cc["ind_p_value"],
            "ind_reject":       cc["ind_reject_H0"],
            # Conditional Coverage
            "LR_cc":            cc["LR_cc"],
            "cc_pvalue":        cc["cc_pvalue"],
            "cc_reject":        cc["cc_reject"],
            # Forecast quality
            "mean_VaR":               round(mean_var, 6),
            "mean_ES":                round(mean_es,  6),
            "mean_ES_to_VaR_ratio":   round(es_var_r, 4),
            # Small-sample flag
            "small_sample":     cc["small_sample"],
        }

    # -----------------------------------------------------------------------
    # ── UNCONDITIONAL BACKTEST (original) ────────────────────────────────

    def run_all_tests(self) -> pd.DataFrame:
        """
        Kupiec POF + Christoffersen CC for every (model, CL) pair
        over the full test window (no regime split).

        Returns
        -------
        pd.DataFrame  — one row per (model, confidence_level)
        """
        rows = []
        for model_name in sorted(self.df["model"].unique()):
            sub = self.df[self.df["model"] == model_name].sort_values("date")
            for cl in self.cls:
                rows.append(
                    self._build_result_row(model_name, cl, sub, regime_label="All")
                )
        return pd.DataFrame(rows)

    # -----------------------------------------------------------------------
    # ── REGIME-CONDITIONAL BACKTEST  (new) ───────────────────────────────

    def run_conditional_tests(
        self,
        regime_filter,
        full_returns: pd.Series,
        regime_labels: Tuple[str, str] = ("Calm", "Stress"),
    ) -> pd.DataFrame:
        """
        Split the backtest by HMM-detected regime and run Kupiec POF +
        Christoffersen CC tests separately for each regime.

        Workflow
        --------
        1.  Call regime_filter.predict_regimes(full_returns) to obtain a
            dated pd.Series of regime labels (0 = Calm, 1 = Stress) aligned
            to the full return history.

        2.  Merge the regime label onto the forecast DataFrame by date.
            Only test dates present in both DataFrames are used (inner join),
            so a few NaN boundary dates are silently dropped.

        3.  For each (model, CL, regime) triple, run the standard
            kupiec + cc pipeline on the regime-filtered sub-DataFrame.

        4.  Return a unified DataFrame with an extra 'regime' column that
            identifies "Calm", "Stress", or (for the full-sample reference
            row) "All".

        Parameters
        ----------
        regime_filter  : RegimeFilter  (already .fit()-ted)
            Must expose a .predict_regimes(returns) method returning a
            pd.Series[int] with values in {0, 1} and a DatetimeIndex.

        full_returns   : pd.Series
            The complete log-return series from MarketDataLoader — used
            to compute realised variance and run the Viterbi decoder.
            Must cover all dates in forecast_df["date"].

        regime_labels  : tuple of str
            Human-readable names for state 0 and state 1.
            Default: ("Calm", "Stress")

        Returns
        -------
        pd.DataFrame  — one row per (model, regime, confidence_level)
            regime ∈ {"All", "Calm", "Stress"}
            Rows are ordered: All → Calm → Stress for each (model, CL).
        """
        # ── Step 1: Predict full regime series ─────────────────────────────
        regime_series = regime_filter.predict_regimes(full_returns)
        # regime_series: pd.Series[int], DatetimeIndex, values in {0, 1}

        if not isinstance(regime_series.index, pd.DatetimeIndex):
            regime_series.index = pd.to_datetime(regime_series.index)

        regime_series.name = "regime_code"
        label_0, label_1 = regime_labels

        # ── Step 2: Attach regime code to forecast_df by date ──────────────
        # Build the lookup DataFrame explicitly so we are never dependent on
        # what pandas names the index column after reset_index() — that name
        # varies by pandas version and by whether the Series index itself has
        # a name, which is the root cause of the KeyError: 'date' crash.
        regime_df = pd.DataFrame({
            "date":        regime_series.index,
            "regime_code": regime_series.values,
        })
        regime_df["date"] = pd.to_datetime(regime_df["date"])

        # Left-merge: every forecast row keeps its data even when a regime
        # label is missing (rare boundary NaNs from the rolling-RV window).
        df_work = self.df.copy()
        df_work["date"] = pd.to_datetime(df_work["date"])
        df_work = df_work.merge(regime_df, on="date", how="left")

        # Fill any unmatched dates with the majority regime (conservative)
        n_missing = df_work["regime_code"].isna().sum()
        if n_missing > 0:
            majority = int(df_work["regime_code"].mode().iloc[0])
            df_work["regime_code"] = df_work["regime_code"].fillna(majority)
            warnings.warn(
                f"[Backtester] {n_missing} forecast dates had no matching "
                f"regime label (boundary NaNs from rolling RV window). "
                f"Filled with majority regime ({majority})."
            )

        df_work["regime_code"] = df_work["regime_code"].astype(int)

        # Regime-0 (Calm) and Regime-1 (Stress) boolean masks
        mask_calm   = df_work["regime_code"] == 0
        mask_stress = df_work["regime_code"] == 1

        # ── Step 3: Build rows for every (model, CL, regime) ───────────────
        rows = []
        for model_name in sorted(df_work["model"].unique()):
            model_mask = df_work["model"] == model_name
            sub_all    = df_work[model_mask].sort_values("date")
            sub_calm   = df_work[model_mask & mask_calm].sort_values("date")
            sub_stress = df_work[model_mask & mask_stress].sort_values("date")

            for cl in self.cls:
                # Full sample reference row (identical to run_all_tests)
                rows.append(
                    self._build_result_row(model_name, cl, sub_all, "All")
                )

                # Calm regime
                if len(sub_calm) > 0:
                    rows.append(
                        self._build_result_row(
                            model_name, cl, sub_calm, label_0
                        )
                    )
                else:
                    rows.append(self._empty_row(model_name, cl, label_0))

                # Stress regime
                if len(sub_stress) > 0:
                    rows.append(
                        self._build_result_row(
                            model_name, cl, sub_stress, label_1
                        )
                    )
                else:
                    rows.append(self._empty_row(model_name, cl, label_1))

        result_df = pd.DataFrame(rows)

        # ── Step 4: Attach HMM metadata for reporting ──────────────────────
        result_df.attrs["regime_obs_calm"]   = int(mask_calm.sum() //
                                                    df_work["model"].nunique())
        result_df.attrs["regime_obs_stress"] = int(mask_stress.sum() //
                                                    df_work["model"].nunique())
        result_df.attrs["regime_labels"]     = regime_labels

        # Emit HMM state means if accessible (for report header)
        try:
            result_df.attrs["hmm_means"] = regime_filter.state_means()
        except AttributeError:
            result_df.attrs["hmm_means"] = None

        try:
            result_df.attrs["hmm_transmat"] = regime_filter.transition_matrix()
        except AttributeError:
            result_df.attrs["hmm_transmat"] = None

        return result_df

    # -----------------------------------------------------------------------
    def _empty_row(
        self,
        model_name:       str,
        confidence_level: float,
        regime_label:     str,
    ) -> Dict:
        """Return a NaN-filled placeholder for an empty regime slice."""
        row = {
            "model": model_name, "regime": regime_label,
            "confidence_level": confidence_level,
            "T": 0, "V": 0, "p_hat": np.nan, "p_expected": 1 - confidence_level,
            "LR_uc": np.nan, "uc_pvalue": np.nan, "uc_reject": np.nan,
            "n11": 0, "pi11": np.nan, "LR_ind": np.nan,
            "ind_pvalue": np.nan, "ind_reject": np.nan,
            "LR_cc": np.nan, "cc_pvalue": np.nan, "cc_reject": np.nan,
            "mean_VaR": np.nan, "mean_ES": np.nan,
            "mean_ES_to_VaR_ratio": np.nan, "small_sample": True,
        }
        return row

    # -----------------------------------------------------------------------
    def violation_series(
        self,
        model_name:       str,
        confidence_level: float,
    ) -> pd.Series:
        """
        Dated binary violation Series for a specific model and CL.

        Returns
        -------
        pd.Series  index=DatetimeIndex, values ∈ {0, 1}
        """
        sub        = self.df[self.df["model"] == model_name].sort_values("date")
        violations = self._get_violations(sub, confidence_level)
        return pd.Series(
            violations,
            index=pd.to_datetime(sub["date"].values),
            name=f"violation_{model_name}_{int(confidence_level * 100)}",
        )

    # -----------------------------------------------------------------------
    # ── REPORT PRINTERS ─────────────────────────────────────────────────

    @staticmethod
    def print_report(summary_df: pd.DataFrame) -> None:
        """
        Formatted unconditional backtest report.
        Prints one block per (model, confidence level).
        """
        PASS = "  PASS ✓"
        FAIL = "  FAIL ✗"

        print("\n" + "═" * 92)
        print("  UNCONDITIONAL BACKTESTING REPORT")
        print("  Kupiec POF (LR_uc ~ χ²(1))  +  "
              "Christoffersen CC (LR_cc ~ χ²(2))")
        print("═" * 92)

        for _, row in summary_df.iterrows():
            cl_pct = int(row["confidence_level"] * 100)
            _print_single_block(row, cl_pct, PASS, FAIL, regime_label=None)

        print("\n" + "═" * 92)
        print("  ✗ = H₀ rejected at 5% significance. "
              "Flagged rows marked [!] if small-sample (T < 30).\n")

    # -----------------------------------------------------------------------
    @staticmethod
    def print_conditional_report(cond_df: pd.DataFrame) -> None:
        """
        Formatted regime-conditional backtest report.

        Prints a nested layout:
            ┌─ Model  ─────────────────────────────────────────────────────┐
            │   CL = 95%  |  All Obs  |  Calm Regime  |  Stress Regime    │
            │   ...        |  ...       |  ...           |  ...              │
            └──────────────────────────────────────────────────────────────┘

        Also prints the HMM summary (state means, transition matrix) as a
        header block, since those figures are essential for interpreting
        the regime split.

        Parameters
        ----------
        cond_df : pd.DataFrame  Output of Backtester.run_conditional_tests()
        """
        PASS = "PASS ✓"
        FAIL = "FAIL ✗"

        # ── HMM metadata header ────────────────────────────────────────────
        regime_labels = cond_df.attrs.get("regime_labels", ("Calm", "Stress"))
        obs_calm      = cond_df.attrs.get("regime_obs_calm",   "?")
        obs_stress    = cond_df.attrs.get("regime_obs_stress", "?")
        hmm_means     = cond_df.attrs.get("hmm_means",   None)
        hmm_transmat  = cond_df.attrs.get("hmm_transmat", None)

        label_0, label_1 = regime_labels

        print("\n" + "═" * 100)
        print("  REGIME-CONDITIONAL BACKTESTING REPORT")
        print("  HMM Volatility Regime Filter  +  "
              "Kupiec POF / Christoffersen CC")
        print("═" * 100)

        print(f"\n  ── HMM Summary ──")
        print(f"  State 0 [{label_0:7s}] :  {obs_calm:>5} test-day observations", end="")
        if hmm_means is not None:
            print(f"   |  log-RV mean = {hmm_means[0]:.4f}")
        else:
            print()
        print(f"  State 1 [{label_1:7s}] :  {obs_stress:>5} test-day observations", end="")
        if hmm_means is not None:
            print(f"   |  log-RV mean = {hmm_means[1]:.4f}")
        else:
            print()

        if hmm_transmat is not None:
            A = hmm_transmat
            print(f"\n  Transition matrix A:")
            print(f"    P(Calm   → Calm  ) = {A[0,0]:.4f}   "
                  f"P(Calm   → Stress) = {A[0,1]:.4f}")
            print(f"    P(Stress → Calm  ) = {A[1,0]:.4f}   "
                  f"P(Stress → Stress) = {A[1,1]:.4f}")
            calm_persist   = A[0, 0]
            stress_persist = A[1, 1]
            exp_calm_dur   = 1.0 / (1 - calm_persist   + 1e-10)
            exp_stress_dur = 1.0 / (1 - stress_persist + 1e-10)
            print(f"\n  Expected Calm   regime duration : {exp_calm_dur:.1f}  days")
            print(f"  Expected Stress regime duration : {exp_stress_dur:.1f}  days")

        # ── Per-model conditional tables ───────────────────────────────────
        models = sorted(cond_df["model"].unique())

        for model_name in models:
            model_df = cond_df[cond_df["model"] == model_name]

            print(f"\n\n  {'─' * 96}")
            print(f"  MODEL: {model_name}")
            print(f"  {'─' * 96}")

            # Column header
            W = 28   # width per regime column
            print(
                f"  {'':30s}"
                f"{'ALL OBS':>{W}s}"
                f"{label_0.upper() + ' REGIME':>{W}s}"
                f"{label_1.upper() + ' REGIME':>{W}s}"
            )

            for cl in sorted(cond_df["confidence_level"].unique()):
                cl_pct = int(cl * 100)
                # Extract the three rows for this (model, CL)
                rows_cl = model_df[model_df["confidence_level"] == cl]
                row_all    = _safe_row(rows_cl, "All")
                row_calm   = _safe_row(rows_cl, label_0)
                row_stress = _safe_row(rows_cl, label_1)

                print(f"\n  ┌─ {cl_pct}% VaR {'─' * 87}")
                _print_regime_row("Observations",
                    _fmt_int(row_all, "T"),
                    _fmt_int(row_calm, "T"),
                    _fmt_int(row_stress, "T"), W)
                _print_regime_row("Violations (V)",
                    _fmt_int(row_all, "V"),
                    _fmt_int(row_calm, "V"),
                    _fmt_int(row_stress, "V"), W)
                _print_regime_row("Hit rate p̂",
                    _fmt_pct(row_all, "p_hat"),
                    _fmt_pct(row_calm, "p_hat"),
                    _fmt_pct(row_stress, "p_hat"), W)
                _print_regime_row(f"Expected rate ({100*(1-cl):.0f}%)",
                    _fmt_pct(row_all, "p_expected"),
                    _fmt_pct(row_calm, "p_expected"),
                    _fmt_pct(row_stress, "p_expected"), W)
                _print_regime_row("Mean VaR forecast",
                    _fmt_f5(row_all, "mean_VaR"),
                    _fmt_f5(row_calm, "mean_VaR"),
                    _fmt_f5(row_stress, "mean_VaR"), W)
                _print_regime_row("Mean ES forecast",
                    _fmt_f5(row_all, "mean_ES"),
                    _fmt_f5(row_calm, "mean_ES"),
                    _fmt_f5(row_stress, "mean_ES"), W)

                print(f"  │")
                _print_regime_row("π₁₁  (consec. viol.)",
                    _fmt_pct(row_all, "pi11"),
                    _fmt_pct(row_calm, "pi11"),
                    _fmt_pct(row_stress, "pi11"), W)
                _print_regime_row("n₁₁  (consec. count)",
                    _fmt_int(row_all, "n11"),
                    _fmt_int(row_calm, "n11"),
                    _fmt_int(row_stress, "n11"), W)

                print(f"  │")
                _print_regime_row("LR_uc  (Kupiec POF)",
                    _fmt_stat_pval(row_all,    "LR_uc", "uc_pvalue", "uc_reject", PASS, FAIL),
                    _fmt_stat_pval(row_calm,   "LR_uc", "uc_pvalue", "uc_reject", PASS, FAIL),
                    _fmt_stat_pval(row_stress, "LR_uc", "uc_pvalue", "uc_reject", PASS, FAIL),
                    W, wide=True)
                _print_regime_row("LR_ind (Independence)",
                    _fmt_stat_pval(row_all,    "LR_ind", "ind_pvalue", "ind_reject", PASS, FAIL),
                    _fmt_stat_pval(row_calm,   "LR_ind", "ind_pvalue", "ind_reject", PASS, FAIL),
                    _fmt_stat_pval(row_stress, "LR_ind", "ind_pvalue", "ind_reject", PASS, FAIL),
                    W, wide=True)
                _print_regime_row("LR_cc  (Cond. Cover.)",
                    _fmt_stat_pval(row_all,    "LR_cc", "cc_pvalue", "cc_reject", PASS, FAIL),
                    _fmt_stat_pval(row_calm,   "LR_cc", "cc_pvalue", "cc_reject", PASS, FAIL),
                    _fmt_stat_pval(row_stress, "LR_cc", "cc_pvalue", "cc_reject", PASS, FAIL),
                    W, wide=True)

                # Small-sample warning for any regime
                for lbl, rw in [(label_0, row_calm), (label_1, row_stress)]:
                    if rw is not None and rw.get("small_sample", False):
                        print(f"  │  [!] {lbl} regime T={rw.get('T',0)} < {MIN_OBS_FOR_TEST}"
                              f" — χ² approximation may be unreliable.")
                print(f"  └{'─' * 92}")

        # ── Cross-model pi11 comparison table ─────────────────────────────
        print("\n\n" + "═" * 100)
        print(f"  VIOLATION CLUSTERING SUMMARY  —  π₁₁  "
              f"[P(violation | prior violation)]")
        print(f"  (Higher π₁₁ in {label_1} regime than {label_0} → "
              f"clustering concentrated in stress)\n")

        header = f"  {'Model':12s}  {'CL':>6s}  "
        header += f"{'π₁₁ All':>12s}  {'π₁₁ ' + label_0:>14s}  {'π₁₁ ' + label_1:>14s}  "
        header += f"{'Δπ₁₁ (S−C)':>12s}  {'Clustering?':>12s}"
        print(header)
        print("  " + "─" * 96)

        for model_name in models:
            model_df = cond_df[cond_df["model"] == model_name]
            for cl in sorted(cond_df["confidence_level"].unique()):
                cl_pct = int(cl * 100)
                rows_cl  = model_df[model_df["confidence_level"] == cl]
                r_all    = _safe_row(rows_cl, "All")
                r_calm   = _safe_row(rows_cl, label_0)
                r_stress = _safe_row(rows_cl, label_1)

                pi11_all    = r_all.get("pi11",    np.nan) if r_all    else np.nan
                pi11_calm   = r_calm.get("pi11",   np.nan) if r_calm   else np.nan
                pi11_stress = r_stress.get("pi11", np.nan) if r_stress else np.nan

                delta = (pi11_stress - pi11_calm
                         if not (np.isnan(pi11_stress) or np.isnan(pi11_calm))
                         else np.nan)
                cluster_flag = (
                    "YES ⚠" if (not np.isnan(delta) and delta > 0.05)
                    else ("—" if np.isnan(delta) else "no")
                )

                print(
                    f"  {model_name:12s}  {cl_pct:>4d}%  "
                    f"  {pi11_all:>10.4f}  "
                    f"  {pi11_calm:>12.4f}  "
                    f"  {pi11_stress:>12.4f}  "
                    f"  {delta:>10.4f}  "
                    f"  {cluster_flag:>12s}"
                )

        print("\n" + "═" * 100)
        print("  ✗ = H₀ rejected at 5% significance. "
              "Δπ₁₁ > 0.05 flagged as stress-clustered.\n")


# ============================================================================
# PRIVATE FORMATTING HELPERS  (module-level, not polluting the class)
# ============================================================================

def _safe_row(df: pd.DataFrame, regime: str) -> Optional[Dict]:
    """Extract a single row as a dict, or None if not present."""
    sub = df[df["regime"] == regime]
    if len(sub) == 0:
        return None
    return sub.iloc[0].to_dict()


def _fmt_int(row: Optional[Dict], key: str) -> str:
    if row is None or key not in row or pd.isna(row[key]):
        return "—"
    return str(int(row[key]))


def _fmt_pct(row: Optional[Dict], key: str) -> str:
    if row is None or key not in row or pd.isna(row[key]):
        return "—"
    return f"{row[key] * 100:.2f}%"


def _fmt_f5(row: Optional[Dict], key: str) -> str:
    if row is None or key not in row or pd.isna(row[key]):
        return "—"
    return f"{row[key]:.5f}"


def _fmt_stat_pval(
    row:       Optional[Dict],
    stat_key:  str,
    pval_key:  str,
    rej_key:   str,
    PASS:      str,
    FAIL:      str,
) -> str:
    if row is None or stat_key not in row or pd.isna(row.get(stat_key, np.nan)):
        return "—"
    stat = row[stat_key]
    pval = row.get(pval_key, np.nan)
    rej  = row.get(rej_key,  False)
    verdict = FAIL if rej else PASS
    return f"{stat:.3f} / p={pval:.3f} {verdict}"


def _print_regime_row(
    label: str,
    val_all:    str,
    val_calm:   str,
    val_stress: str,
    W:    int,
    wide: bool = False,
) -> None:
    """Print one data row across the All / Calm / Stress columns."""
    if wide:
        W2 = W + 8
    else:
        W2 = W
    print(f"  │  {label:28s}{val_all:>{W2}s}{val_calm:>{W2}s}{val_stress:>{W2}s}")


def _print_single_block(
    row:          pd.Series,
    cl_pct:       int,
    PASS:         str,
    FAIL:         str,
    regime_label: Optional[str],
) -> None:
    """Render one block in the unconditional report."""
    regime_str = f"  |  Regime: {regime_label}" if regime_label else ""
    print(f"\n  Model: {row['model']:10s}  |  Confidence Level: {cl_pct}%{regime_str}")
    print(f"  {'─' * 60}")
    print(f"  Observations        : {row['T']}")
    print(f"  Violations          : {row['V']}  "
          f"(expected ≈ {int(row['p_expected'] * row['T'])})")
    print(f"  Empirical VaR hit % : {row['p_hat']*100:.2f}%  "
          f"(theoretical: {row['p_expected']*100:.1f}%)")
    print(f"  Mean VaR forecast   : {row['mean_VaR']:.5f}")
    print(f"  Mean ES  forecast   : {row['mean_ES']:.5f}")
    print(f"  ES / VaR ratio      : {row['mean_ES_to_VaR_ratio']:.3f}")
    if row.get("small_sample", False):
        print(f"  [!] Small sample (T={row['T']} < {MIN_OBS_FOR_TEST})"
              f" — χ² approximation may be unreliable")
    print(f"\n  ── Kupiec POF (LR_uc ~ χ²(1)) ──")
    print(f"  LR_uc  = {row['LR_uc']:.4f}   p-value = {row['uc_pvalue']:.4f}  "
          + (PASS if not row['uc_reject'] else FAIL))
    print(f"\n  ── Christoffersen Independence (LR_ind ~ χ²(1)) ──")
    print(f"  n₁₁    = {row['n11']}  (consecutive violations)")
    print(f"  π₁₁    = {row['pi11']:.4f}  P(violation | prior violation)")
    print(f"  LR_ind = {row['LR_ind']:.4f}   p-value = {row['ind_pvalue']:.4f}  "
          + (PASS if not row['ind_reject'] else FAIL))
    print(f"\n  ── Christoffersen CC (LR_cc ~ χ²(2)) ──")
    print(f"  LR_cc  = {row['LR_cc']:.4f}   p-value = {row['cc_pvalue']:.4f}  "
          + (PASS if not row['cc_reject'] else FAIL))


# ============================================================================
# STANDALONE TEST
# ============================================================================

if __name__ == "__main__":
    np.random.seed(42)
    T = 600

    # Simulate i.i.d. violations at the right rate
    viols_iid = np.random.binomial(1, 0.05, T)
    print("=== Kupiec POF (i.i.d. at p=0.05) ===")
    pof = kupiec_pof_test(viols_iid, 0.95)
    for k, v in pof.items():
        print(f"  {k:20s}: {v}")

    # Simulate CLUSTERED violations
    clustered = np.zeros(T, dtype=int)
    clustered[50:70]   = 1
    clustered[200:218] = 1
    clustered[410:425] = 1
    print("\n=== Christoffersen CC (clustered) ===")
    cc = christoffersen_cc_test(clustered, 0.95)
    print(f"  π₁₁ = {cc['ind_pi11']:.4f}  LR_ind = {cc['ind_LR_ind']:.4f}"
          f"  p = {cc['ind_p_value']:.4f}  reject = {cc['ind_reject_H0']}")
    print(f"  LR_cc = {cc['LR_cc']:.4f}  p = {cc['cc_pvalue']:.4f}")

    # Minimal Backtester smoke-test (synthetic forecast_df)
    import pandas as pd
    dates  = pd.date_range("2018-01-01", periods=T, freq="B")
    np.random.seed(7)

    # Build a regime-like return series (calm first, stress second half)
    ret    = np.concatenate([
        np.random.normal(0, 0.007, T // 2),
        np.random.normal(0, 0.020, T // 2),
    ])
    var95  = np.full(T, 0.013)
    var99  = np.full(T, 0.019)

    records = []
    for m in ["GARCH", "Heston", "Merton", "HistSim"]:
        for i, d in enumerate(dates):
            records.append({
                "date": d, "model": m, "realised_r": ret[i],
                "VaR_95": var95[i], "ES_95": var95[i] * 1.25,
                "VaR_99": var99[i], "ES_99": var99[i] * 1.20,
            })

    fdf = pd.DataFrame(records)
    bt  = Backtester(fdf)

    print("\n=== Unconditional Backtest (smoke test) ===")
    unc = bt.run_all_tests()
    Backtester.print_report(unc[unc["model"] == "GARCH"])

    # Minimal RegimeFilter mock for smoke-test (no hmmlearn needed)
    class MockRegimeFilter:
        def predict_regimes(self, returns):
            n      = len(returns)
            states = np.concatenate([np.zeros(n // 2), np.ones(n // 2)]).astype(int)
            return pd.Series(states, index=returns.index, name="regime_code")
        def state_means(self):
            return np.array([-9.8, -7.5])
        def transition_matrix(self):
            return np.array([[0.97, 0.03], [0.05, 0.95]])

    full_rets = pd.Series(ret, index=dates, name="log_return")
    cond = bt.run_conditional_tests(MockRegimeFilter(), full_rets)

    print("\n=== Conditional Report (smoke test, mock HMM) ===")
    Backtester.print_conditional_report(cond)