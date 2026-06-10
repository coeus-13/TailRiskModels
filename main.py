"""
main.py
=======
Rolling Backtest Orchestrator
------------------------------
End-to-end pipeline that ties together all five modules:

    Data → Calibration → Risk Estimation → Backtesting → Visualization

Architecture: Rolling-Window Scheme
-------------------------------------
For each step t in the test period:

  1. Slice the calibration window  W = [t − N_cal, t − 1]
  2. Fit all three models (GARCH, Heston, Merton) on W
  3. Simulate N_SIM return paths from each fitted model
  4. Compute VaR_95, ES_95, VaR_99, ES_99 from the simulated distributions
  5. Record the actual realised log-return r_t
  6. Store the forecast row: (date, model, realised_r, VaR_95, ES_95, ...)
  7. Advance the window by STEP days

After the loop:
  8. Run Kupiec POF + Christoffersen CC tests on the aggregated forecast table
  9. Generate and save all visualisation charts
 10. Export the results to CSV

Computational Note
------------------
The full rolling window with daily re-fitting across 3 models can take
30–90 minutes depending on hardware and series length. The STEP parameter
in config.py can be set to 5 (weekly) or 21 (monthly) to drastically
reduce runtime while preserving the statistical validity of the backtest.

Progress is printed every `LOG_EVERY` steps and an intermediate
checkpoint CSV is saved every `CHECKPOINT_EVERY` steps so that
a crash does not lose all work.

Usage
-----
    python main.py                        # use config.py defaults
    python main.py --ticker ^GSPC         # override ticker
    python main.py --step 5               # weekly re-fitting (faster)
    python main.py --window 252           # 1-year calibration window
    python main.py --no-plots             # skip visualization
    python main.py --checkpoint results/  # load existing checkpoint

Command-line arguments override config.py values.
"""

import argparse
import os
import sys
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Local module imports
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    TICKER, START_DATE, END_DATE,
    CALIBRATION_WINDOW, TEST_STEP,
    N_SIMULATIONS, CONFIDENCE_LEVELS, RESULTS_DIR
)
from data.data_loader     import MarketDataLoader
from models.garch_model   import GARCHModel
from models.heston_model  import HestonModel
from models.merton_model  import MertonJumpDiffusion
from risk.risk_estimator import RiskEstimator, build_forecast_dataframe
from backtest.backtester import Backtester
from visualization.plotter import save_all_plots


# ===========================================================================
# CONSTANTS
# ===========================================================================
LOG_EVERY        = 25      # Print progress every N steps
CHECKPOINT_EVERY = 100     # Save intermediate CSV every N steps


# ===========================================================================
# HELPER: Parse CLI Arguments
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Rolling Backtest: GARCH / Heston / Merton — VaR & ES"
    )
    p.add_argument("--ticker",     default=TICKER,              help="Yahoo Finance ticker")
    p.add_argument("--start",      default=START_DATE,          help="History start YYYY-MM-DD")
    p.add_argument("--end",        default=END_DATE,            help="History end   YYYY-MM-DD")
    p.add_argument("--window",     default=CALIBRATION_WINDOW,  type=int, help="Calibration window (days)")
    p.add_argument("--step",       default=TEST_STEP,           type=int, help="Rolling step size (days)")
    p.add_argument("--nsims",      default=N_SIMULATIONS,       type=int, help="MC paths per step")
    p.add_argument("--no-plots",   action="store_true",                   help="Skip visualization")
    p.add_argument("--checkpoint", default=None,                          help="Path to existing forecast CSV to resume from")
    p.add_argument("--out",        default=RESULTS_DIR,                   help="Output directory")
    return p.parse_args()


# ===========================================================================
# CORE: Single-Step Forecast
# ===========================================================================

def run_single_step(
    cal_returns: pd.Series,
    test_date:   pd.Timestamp,
    test_return: float,
    estimator:   RiskEstimator,
    n_sims:      int,
    step_seed:   int,
    verbose:     bool = False,
) -> list:
    """
    Fit all three models on `cal_returns`, compute VaR/ES, and return a
    list of forecast records (one per model + one for HistSim).

    Parameters
    ----------
    cal_returns : pd.Series   Calibration-window log-returns (length W)
    test_date   : timestamp   The date we are forecasting
    test_return : float       Realised log-return on test_date
    estimator   : RiskEstimator
    n_sims      : int         Monte-Carlo paths
    step_seed   : int         Base RNG seed (ensures reproducibility)
    verbose     : bool        Print per-step debug info

    Returns
    -------
    list of dicts, one per model name, each containing:
        date, model, realised_r, VaR_95, ES_95, VaR_99, ES_99
    """
    records = []

    try:
        # ── Fit all three models ──────────────────────────────────────────
        garch  = GARCHModel().fit(cal_returns, disp="off")
        heston = HestonModel().fit(cal_returns, n_restarts=2)
        merton = MertonJumpDiffusion().fit(cal_returns, n_restarts=2)

        # ── Compute risk estimates ────────────────────────────────────────
        risk_dict = estimator.estimate_all(
            cal_returns=cal_returns,
            garch_model=garch,
            heston_model=heston,
            merton_model=merton,
            n_sims=n_sims,
            seed=step_seed,
        )

        # ── Build one record per model ────────────────────────────────────
        for model_name, metrics in risk_dict.items():
            record = {
                "date":       test_date,
                "model":      model_name,
                "realised_r": test_return,
            }
            record.update(metrics)
            records.append(record)

        if verbose:
            g_v95 = risk_dict["GARCH"].get("VaR_95", np.nan)
            h_v95 = risk_dict["Heston"].get("VaR_95", np.nan)
            m_v95 = risk_dict["Merton"].get("VaR_95", np.nan)
            print(f"    GARCH VaR95={g_v95:.5f}  "
                  f"Heston VaR95={h_v95:.5f}  "
                  f"Merton VaR95={m_v95:.5f}")

    except Exception as exc:
        # If a step fails (e.g. optimiser diverged), record NaNs so the
        # loop can continue — do NOT crash the entire backtest.
        warnings.warn(f"[WARNING] Step {test_date.date()} failed: {exc}")
        for model_name in ["GARCH", "Heston", "Merton", "HistSim"]:
            record = {
                "date":       test_date,
                "model":      model_name,
                "realised_r": test_return,
            }
            for cl in CONFIDENCE_LEVELS:
                pct = int(cl * 100)
                record[f"VaR_{pct}"] = np.nan
                record[f"ES_{pct}"]  = np.nan
            records.append(record)

    return records


# ===========================================================================
# CORE: Rolling Backtest Loop
# ===========================================================================

def run_rolling_backtest(
    loader:      MarketDataLoader,
    window:      int,
    step:        int,
    n_sims:      int,
    output_dir:  str,
    resume_df:   pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Main rolling-window backtesting loop.

    Parameters
    ----------
    loader     : MarketDataLoader  (already .load()-ed)
    window     : int  Calibration window size in days
    step       : int  Days to advance per iteration
    n_sims     : int  MC paths per model per step
    output_dir : str  Directory for checkpoint CSVs
    resume_df  : pd.DataFrame  If provided, skip dates already computed

    Returns
    -------
    pd.DataFrame  Full forecast table (all models, all dates)
    """
    os.makedirs(output_dir, exist_ok=True)
    estimator = RiskEstimator(confidence_levels=CONFIDENCE_LEVELS)

    # Pre-compute the set of already-done dates (for resuming)
    done_dates = set()
    if resume_df is not None and not resume_df.empty:
        done_dates = set(pd.to_datetime(resume_df["date"]).unique())
        print(f"[Resume] Skipping {len(done_dates)} already-computed dates.")

    all_records   = list(resume_df.to_dict("records")) if resume_df is not None else []
    total_windows = sum(
        1 for _ in loader.rolling_windows(calibration_window=window, step=step)
    )
    print(f"\n[Backtest] Starting rolling window loop.")
    print(f"  Total steps    : {total_windows}")
    print(f"  Cal. window    : {window} days")
    print(f"  Step size      : {step}  day(s)")
    print(f"  MC paths/step  : {n_sims}")
    print(f"  Models         : GARCH, Heston, Merton + HistSim baseline")
    print(f"  Confidence lvls: {[f'{int(c*100)}%' for c in CONFIDENCE_LEVELS]}")
    print(f"  Estimated time : ~{total_windows * 0.35 / 60:.1f} min  "
          f"(varies by machine)\n")

    t_start = time.time()

    for i, (cal_returns, test_date, test_return) in enumerate(
        loader.rolling_windows(calibration_window=window, step=step)
    ):
        # Skip already computed steps when resuming
        if test_date in done_dates:
            continue

        # Progress reporting
        if i % LOG_EVERY == 0:
            elapsed    = time.time() - t_start
            pct_done   = (i + 1) / total_windows * 100
            eta_s      = (elapsed / (i + 1)) * (total_windows - i - 1) if i > 0 else 0
            eta_str    = f"{eta_s/60:.1f} min" if eta_s > 60 else f"{eta_s:.0f}s"
            print(f"  Step {i+1:>5d}/{total_windows}  "
                  f"[{pct_done:5.1f}%]  "
                  f"date={test_date.date()}  "
                  f"elapsed={elapsed:.0f}s  ETA={eta_str}")

        # Run single-step calibration + risk estimation
        records = run_single_step(
            cal_returns=cal_returns,
            test_date=test_date,
            test_return=test_return,
            estimator=estimator,
            n_sims=n_sims,
            step_seed=i,             # deterministic per step
            verbose=(i < 3),         # verbose on first 3 steps
        )
        all_records.extend(records)

        # Checkpoint save
        if (i + 1) % CHECKPOINT_EVERY == 0:
            ckpt_path = os.path.join(output_dir, "checkpoint_forecast.csv")
            pd.DataFrame(all_records).to_csv(ckpt_path, index=False)
            print(f"    [Checkpoint saved → {ckpt_path}]")

    total_elapsed = time.time() - t_start
    print(f"\n[Backtest] Completed {total_windows} steps in "
          f"{total_elapsed/60:.1f} minutes.")

    return build_forecast_dataframe(all_records)


# ===========================================================================
# MAIN ENTRY POINT
# ===========================================================================

def main():
    args = parse_args()

    print("=" * 70)
    print("  TAIL RISK BACKTEST — GARCH / Heston / Merton Jump-Diffusion")
    print(f"  Run started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # ── 1. Load Data ───────────────────────────────────────────────────────
    loader = MarketDataLoader(
        ticker=args.ticker,
        start_date=args.start,
        end_date=args.end,
    ).load()

    print("\n=== Summary Statistics ===")
    print(loader.summary_statistics().to_string())

    # ── 2. Resume or Start Fresh ───────────────────────────────────────────
    resume_df = None
    if args.checkpoint and os.path.exists(args.checkpoint):
        resume_df = pd.read_csv(args.checkpoint, parse_dates=["date"])
        print(f"\n[Resume] Loaded checkpoint with {len(resume_df)} records "
              f"from '{args.checkpoint}'.")

    # ── 3. Rolling Backtest Loop ───────────────────────────────────────────
    forecast_df = run_rolling_backtest(
        loader=loader,
        window=args.window,
        step=args.step,
        n_sims=args.nsims,
        output_dir=args.out,
        resume_df=resume_df,
    )

    # ── 4. Save Full Forecast Table ────────────────────────────────────────
    os.makedirs(args.out, exist_ok=True)
    forecast_path = os.path.join(args.out, "forecast_results.csv")
    forecast_df.to_csv(forecast_path, index=False)
    print(f"\n[Output] Forecast table saved → {forecast_path}")
    print(f"         Shape: {forecast_df.shape}  "
          f"({forecast_df['model'].nunique()} models × "
          f"{forecast_df.groupby('model').size().max()} test dates)")

    # ── 5. Statistical Backtesting ────────────────────────────────────────
    # Drop NaN rows (failed steps) before testing
    clean_df = forecast_df.dropna(subset=[
        f"VaR_{int(cl*100)}" for cl in CONFIDENCE_LEVELS
    ])
    print(f"\n[Backtest] Running statistical tests on {len(clean_df)} "
          f"non-NaN forecast rows ...")

    backtester     = Backtester(clean_df, confidence_levels=CONFIDENCE_LEVELS)
    summary_df     = backtester.run_all_tests()

    # Print formatted report
    Backtester.print_report(summary_df)

    backtester.run_conditional_tests()

    # Save summary
    summary_path = os.path.join(args.out, "backtest_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"[Output] Backtest summary saved → {summary_path}")

    # ── 6. Quick Model Ranking ─────────────────────────────────────────────
    print("\n=== MODEL RANKING (by number of tests passed at 5% sig.) ===")
    ranking = (
        summary_df
        .assign(
            uc_pass  = ~summary_df["uc_reject"],
            ind_pass = ~summary_df["ind_reject"],
            cc_pass  = ~summary_df["cc_reject"],
        )
        .groupby("model")[["uc_pass", "ind_pass", "cc_pass"]]
        .sum()
        .assign(total_passed=lambda x: x.sum(axis=1))
        .sort_values("total_passed", ascending=False)
    )
    print(ranking.to_string())

    # ── 7. Visualization ──────────────────────────────────────────────────
    if not args.no_plots:
        print("\n[Visualization] Generating all charts ...")
        try:
            import matplotlib
            matplotlib.use("Agg")   # non-interactive backend (no display needed)
            saved_paths = save_all_plots(
                forecast_df=clean_df,
                backtest_summary=summary_df,
                empirical_returns=loader.log_returns,
                output_dir=args.out,
            )
            print(f"[Visualization] {len(saved_paths)} charts saved to '{args.out}/'.")
        except Exception as e:
            print(f"[Visualization] WARNING: Chart generation failed — {e}")
            print("  (This is non-fatal; results CSV files are still complete.)")
    else:
        print("\n[Visualization] Skipped (--no-plots flag set).")

    # ── 8. Final Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  RUN COMPLETE")
    print(f"  Output directory : {os.path.abspath(args.out)}/")
    print(f"  Forecast CSV     : forecast_results.csv")
    print(f"  Summary CSV      : backtest_summary.csv")
    if not args.no_plots:
        print(f"  Charts           : see results/*.png")
    print("=" * 70)


# ===========================================================================
# QUICK DEMO (fast, synthetic, no yfinance needed)
# ===========================================================================

def run_demo(n_steps: int = 80, window: int = 250, n_sims: int = 2000):
    """
    Smoke-test the full pipeline with synthetic Gaussian returns.
    Completes in under 2 minutes.  Call from a notebook or REPL.

    >>> from main import run_demo
    >>> forecast_df, summary_df = run_demo()
    """
    print("[DEMO] Generating synthetic return series ...")
    np.random.seed(42)
    T        = window + n_steps
    dates    = pd.date_range("2015-01-01", periods=T, freq="B")
    # Regime-switching vol: calm then turbulent
    vol      = np.concatenate([
        np.full(T // 2, 0.008),
        np.full(T - T // 2, 0.018),
    ])
    ret_vals = np.random.normal(0, vol)
    returns  = pd.Series(ret_vals, index=dates, name="log_return")

    estimator   = RiskEstimator(CONFIDENCE_LEVELS)
    all_records = []

    print(f"[DEMO] Rolling {n_steps} steps (window={window}, nsims={n_sims}) ...")
    for i in range(n_steps):
        cal = returns.iloc[i: i + window]
        test_date   = dates[i + window]
        test_return = ret_vals[i + window]

        records = run_single_step(
            cal_returns=cal,
            test_date=test_date,
            test_return=test_return,
            estimator=estimator,
            n_sims=n_sims,
            step_seed=i,
            verbose=False,
        )
        all_records.extend(records)

        if (i + 1) % 20 == 0:
            print(f"  Step {i+1}/{n_steps}")

    forecast_df = build_forecast_dataframe(all_records)
    clean_df    = forecast_df.dropna()

    backtester  = Backtester(clean_df, CONFIDENCE_LEVELS)
    summary_df  = backtester.run_all_tests()
    Backtester.print_report(summary_df)

    print("\n[DEMO] Done.  Returns: (forecast_df, summary_df)")
    return forecast_df, summary_df


# ===========================================================================
if __name__ == "__main__":
    # If called with no arguments in a Jupyter context, run the demo.
    # In a terminal, parse CLI arguments normally.
    import sys
    if "ipykernel" in sys.modules:
        forecast_df, summary_df = run_demo()
    else:
        main()
