"""
backtest/backtester.py
======================
Backtesting Module — Core Statistical Contribution
----------------------------------------------------
Implements the two industry-standard VaR backtesting frameworks:

  1. Kupiec (1995) Proportion of Failures (POF) Test
     — Unconditional Coverage: tests whether the total number of VaR
       violations matches the expected proportion.

  2. Christoffersen (1998) Conditional Coverage Test
     — Tests BOTH unconditional coverage AND independence of violations
       (violations should not cluster in time).

Both tests use likelihood ratio statistics that are asymptotically
chi-squared distributed, enabling rigorous hypothesis testing.
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2
from typing import Dict, Tuple, List
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CONFIDENCE_LEVELS


# ===========================================================================
# CORE STATISTICAL TESTS (stateless functions)
# ===========================================================================

def kupiec_pof_test(
    violations:        np.ndarray,
    confidence_level:  float,
) -> Dict[str, float]:
    """Kupiec (1995) Proportion of Failures (POF) test."""
    T   = len(violations)
    V   = int(np.sum(violations))
    p   = 1.0 - confidence_level
    p_hat = V / T if T > 0 else 0.0

    eps   = 1e-10
    p_hat_safe = np.clip(p_hat, eps, 1 - eps)

    ll_H0 = V * np.log(p + eps) + (T - V) * np.log(1 - p + eps)
    ll_H1 = V * np.log(p_hat_safe) + (T - V) * np.log(1 - p_hat_safe)

    LR_uc  = -2.0 * (ll_H0 - ll_H1)
    LR_uc  = max(LR_uc, 0.0)

    p_value    = 1.0 - chi2.cdf(LR_uc, df=1)
    reject_H0  = p_value < 0.05

    return {
        "T":           T,
        "V":           V,
        "p_hat":       round(p_hat, 6),
        "p_expected":  round(p, 6),
        "LR_uc":       round(LR_uc, 4),
        "p_value":     round(p_value, 4),
        "reject_H0":   reject_H0,
    }


def christoffersen_independence_test(
    violations: np.ndarray,
) -> Dict[str, float]:
    """Christoffersen (1998) independence component of the CC test."""
    I      = violations.astype(int)
    I_prev = I[:-1]
    I_curr = I[1:]

    n00 = int(np.sum((I_prev == 0) & (I_curr == 0)))
    n01 = int(np.sum((I_prev == 0) & (I_curr == 1)))
    n10 = int(np.sum((I_prev == 1) & (I_curr == 0)))
    n11 = int(np.sum((I_prev == 1) & (I_curr == 1)))

    eps = 1e-10

    pi01 = n01 / (n00 + n01 + eps)
    pi11 = n11 / (n10 + n11 + eps)

    T_trans = n00 + n01 + n10 + n11
    pi      = (n01 + n11) / (T_trans + eps)

    pi_s    = np.clip(pi,   eps, 1 - eps)
    pi01_s  = np.clip(pi01, eps, 1 - eps)
    pi11_s  = np.clip(pi11, eps, 1 - eps)

    ll_H0 = ((n00 + n10) * np.log(1 - pi_s) + (n01 + n11) * np.log(pi_s))
    ll_H1 = (n00 * np.log(1 - pi01_s) + n01 * np.log(pi01_s) +
             n10 * np.log(1 - pi11_s) + n11 * np.log(pi11_s))

    LR_ind    = -2.0 * (ll_H0 - ll_H1)
    LR_ind    = max(LR_ind, 0.0)

    p_value   = 1.0 - chi2.cdf(LR_ind, df=1)
    reject_H0 = p_value < 0.05

    return {
        "n00": n00, "n01": n01, "n10": n10, "n11": n11,
        "pi01":      round(pi01, 6),
        "pi11":      round(pi11, 6),
        "pi":        round(pi,   6),
        "LR_ind":    round(LR_ind, 4),
        "p_value":   round(p_value, 4),
        "reject_H0": reject_H0,
    }


def christoffersen_cc_test(
    violations:       np.ndarray,
    confidence_level: float,
) -> Dict[str, float]:
    """Christoffersen (1998) Conditional Coverage (CC) test."""
    pof  = kupiec_pof_test(violations, confidence_level)
    ind  = christoffersen_independence_test(violations)

    LR_cc     = pof["LR_uc"] + ind["LR_ind"]
    cc_pvalue = 1.0 - chi2.cdf(LR_cc, df=2)
    cc_reject = cc_pvalue < 0.05

    combined = {**pof, **{f"ind_{k}": v for k, v in ind.items()}}
    combined["LR_cc"]     = round(LR_cc, 4)
    combined["cc_pvalue"] = round(cc_pvalue, 4)
    combined["cc_reject"] = cc_reject
    return combined


# ===========================================================================
# CLASS: Backtester
# ===========================================================================

class Backtester:
    """Runs full backtesting evaluations and regime conditional metrics."""

    def __init__(
        self,
        forecast_df:       pd.DataFrame,
        confidence_levels: List[float] = None,
    ):
        self.df  = forecast_df.copy()
        self.cls = confidence_levels or CONFIDENCE_LEVELS
        self._validate()

    def _validate(self):
        required = {"date", "model", "realised_r"}
        for cl in self.cls:
            pct = int(cl * 100)
            required |= {f"VaR_{pct}", f"ES_{pct}"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"forecast_df is missing columns: {missing}")

    def _get_violations(self, model_df: pd.DataFrame, confidence_level: float) -> np.ndarray:
        pct    = int(confidence_level * 100)
        var_col = f"VaR_{pct}"
        return (model_df["realised_r"].values < -model_df[var_col].values).astype(int)

    def run_all_tests(self) -> pd.DataFrame:
        """Run standard baseline Kupiec POF + Christoffersen CC tests."""
        rows = []
        for model_name in self.df["model"].unique():
            sub = self.df[self.df["model"] == model_name].sort_values("date")

            for cl in self.cls:
                pct         = int(cl * 100)
                violations  = self._get_violations(sub, cl)
                cc_results  = christoffersen_cc_test(violations, cl)

                var_col = f"VaR_{pct}"
                es_col  = f"ES_{pct}"
                mean_var = sub[var_col].mean()
                mean_es  = sub[es_col].mean()
                ratio    = mean_es / mean_var if mean_var > 0 else np.nan

                rows.append({
                    "model":            model_name,
                    "confidence_level": cl,
                    "T":                cc_results["T"],
                    "V":                cc_results["V"],
                    "p_hat":            cc_results["p_hat"],
                    "p_expected":       cc_results["p_expected"],
                    "LR_uc":            cc_results["LR_uc"],
                    "uc_pvalue":        cc_results["p_value"],
                    "uc_reject":        cc_results["reject_H0"],
                    "n11":              cc_results["ind_n11"],
                    "pi11":             cc_results["ind_pi11"],
                    "LR_ind":           cc_results["ind_LR_ind"],
                    "ind_pvalue":       cc_results["ind_p_value"],
                    "ind_reject":       cc_results["ind_reject_H0"],
                    "LR_cc":            cc_results["LR_cc"],
                    "cc_pvalue":        cc_results["cc_pvalue"],
                    "cc_reject":        cc_results["cc_reject"],
                    "mean_VaR":              round(mean_var, 6),
                    "mean_ES":               round(mean_es, 6),
                    "mean_ES_to_VaR_ratio":  round(ratio, 4),
                })
        return pd.DataFrame(rows)

    def violation_series(self, model_name: str, confidence_level: float) -> pd.Series:
        sub        = self.df[self.df["model"] == model_name].sort_values("date")
        violations = self._get_violations(sub, confidence_level)
        return pd.Series(
            violations,
            index=pd.to_datetime(sub["date"].values),
            name=f"violation_{model_name}_{int(confidence_level*100)}",
        )

    # -----------------------------------------------------------------------
    # REGIME FILTER ANALYSIS INTEGRATION
    # -----------------------------------------------------------------------
    def run_conditional_tests(self) -> None:
        """
        Fits an HMM on realized variance to separate out calculations
        by Calm vs. Stressed market regimes natively.
        """
        from regime.hmm_filter import RegimeFilter
        import pandas as pd

        print("\n" + "=" * 90)
        print("  VOLATILITY REGIME CONDITIONAL BACKTEST (HMM)")
        print("=" * 90)

        # 1. Isolate a single continuous timeline of index returns to safely train the HMM
        any_model = self.df["model"].unique()[0]
        daily_series = self.df[self.df["model"] == any_model].sort_values("date").copy()

        daily_series["date"] = pd.to_datetime(daily_series["date"])
        daily_returns = daily_series.set_index("date")["realised_r"]

        # 2. Extract and match states cleanly across matching historical indices
        hmm_filter = RegimeFilter()
        hmm_filter.fit(daily_returns)
        all_states = hmm_filter.predict_regimes(daily_returns)

        # 3. Create map coordinates to append safely to internal datasets
        regime_map = dict(zip(daily_returns.index[4:], all_states))
        self.df["date_parsed"] = pd.to_datetime(self.df["date"])
        self.df["Regime"] = self.df["date_parsed"].map(regime_map)

        regime_labels = {0: "Calm Regime (Low Vol)", 1: "Stress Regime (High Vol)"}

        for regime_id, label in regime_labels.items():
            print(f"\n>>> {label} <<<")
            print(f"{'─' * 85}")

            regime_chunk = self.df[self.df["Regime"] == regime_id]

            for model_name in regime_chunk["model"].unique():
                sub = regime_chunk[regime_chunk["model"] == model_name].sort_values("date")

                for cl in self.cls:
                    pct = int(cl * 100)
                    violations = self._get_violations(sub, cl)

                    T_sub = len(violations)
                    V_sub = int(np.sum(violations))
                    p_hat_sub = V_sub / T_sub if T_sub > 0 else 0.0
                    p_exp = 1.0 - cl

                    if T_sub > 5:
                        cc_results = christoffersen_cc_test(violations, cl)
                        status = "PASS ✓" if not cc_results["cc_reject"] else "FAIL ✗"
                        p_val = str(cc_results["cc_pvalue"])
                    else:
                        status = "INSUFFICIENT DATA"
                        p_val = "N/A"

                    print(f"  Model: {model_name:<8} | CL: {pct}% | Days: {T_sub:<4} | Breaches: {V_sub:<2} | Hit Rate: {p_hat_sub:.2%} (Target: {p_exp:.1%}) | CC p-val: {p_val:<6} -> {status}")
        print("\n" + "=" * 90)

    # -----------------------------------------------------------------------
    @staticmethod
    def print_report(summary_df: pd.DataFrame) -> None:
        print("\n" + "=" * 90)
        print("  BACKTESTING REPORT")
        print("  Kupiec POF (Unconditional Coverage) + Christoffersen CC (Conditional Coverage)")
        print("=" * 90)

        PASS = "  PASS ✓"
        FAIL = "  FAIL ✗"

        for _, row in summary_df.iterrows():
            cl_pct = int(row["confidence_level"] * 100)
            print(f"\n  Model: {row['model']:10s}  |  Confidence Level: {cl_pct}%")
            print(f"  {'─'*60}")
            print(f"  Observations        : {row['T']}")
            print(f"  Violations          : {row['V']}  "
                  f"(expected ≈ {int(row['p_expected'] * row['T'])})")
            print(f"  Empirical VaR hit % : {row['p_hat']*100:.2f}%  "
                  f"(theoretical: {row['p_expected']*100:.1f}%)")
            print(f"  Mean VaR forecast   : {row['mean_VaR']:.5f}")
            print(f"  Mean ES  forecast   : {row['mean_ES']:.5f}")
            print(f"  ES / VaR ratio      : {row['mean_ES_to_VaR_ratio']:.3f}")
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
        print("\n" + "=" * 90)
        print("  H₀ rejected at 5% significance level.\n")