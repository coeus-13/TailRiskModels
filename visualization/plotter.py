"""
visualization/plotter.py
========================
Visualization Module
---------------------
Provides all publication-quality plots for the backtest project.

Plot Inventory
--------------
1. plot_returns_with_var()
   Time-series of realised returns overlaid with ±VaR bands for one model.
   Violation dates are marked with red scatter points.

2. plot_all_models_var()
   2×2 subplot grid: one panel per model (GARCH, Heston, Merton, HistSim),
   each showing returns vs. 95% and 99% VaR bands.

3. plot_violation_comparison()
   Bar chart comparing observed vs. expected violation counts across all
   models and confidence levels.

4. plot_backtest_summary_heatmap()
   Heatmap of p-values from Kupiec and Christoffersen tests; green = pass,
   red = reject.

5. plot_var_es_over_time()
   Time-series of VaR and ES forecasts for a chosen model, showing how
   risk estimates evolve through calm and turbulent periods.

6. plot_return_distribution()
   Histogram + KDE of simulated returns from each model, with empirical
   return histogram overlaid — illustrates tail behaviour differences.

7. plot_violation_clustering()
   Raster / event plot of violation dates, enabling visual detection of
   clustering.

8. save_all_plots()
   Convenience function that renders and saves every chart to disk.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
from typing import List, Optional, Dict
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CONFIDENCE_LEVELS, RESULTS_DIR

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------
sns.set_theme(style="darkgrid", palette="muted", font_scale=1.05)
COLORS = {
    "GARCH":   "#2196F3",   # blue
    "Heston":  "#4CAF50",   # green
    "Merton":  "#FF9800",   # orange
    "HistSim": "#9C27B0",   # purple
}
VaR95_COLOR  = "#E53935"   # red (95% VaR)
VaR99_COLOR  = "#B71C1C"   # dark red (99% VaR)
VIOLATION_COLOR = "#FF1744"


# ===========================================================================
# 1. Single-Model Return vs VaR Plot
# ===========================================================================

def plot_returns_with_var(
    forecast_df:      pd.DataFrame,
    model_name:       str,
    confidence_level: float    = 0.95,
    ax:               plt.Axes = None,
    title_suffix:     str      = "",
) -> plt.Axes:
    """
    Plot realised log-returns against the forecasted VaR boundary for one model.

    Shading below −VaR highlights the "danger zone". Violations (actual
    returns breaching VaR) are marked with red dots.

    Parameters
    ----------
    forecast_df      : pd.DataFrame  Full results from rolling backtest
    model_name       : str           e.g. 'GARCH', 'Heston', 'Merton', 'HistSim'
    confidence_level : float         0.95 or 0.99
    ax               : plt.Axes      Optionally pass an existing axes
    title_suffix     : str           Appended to plot title

    Returns
    -------
    plt.Axes
    """
    pct = int(confidence_level * 100)
    sub = forecast_df[forecast_df["model"] == model_name].sort_values("date").copy()
    sub["date"] = pd.to_datetime(sub["date"])

    var_col = f"VaR_{pct}"
    es_col  = f"ES_{pct}"

    violations = sub["realised_r"] < -sub[var_col]

    if ax is None:
        _, ax = plt.subplots(figsize=(14, 5))

    color = COLORS.get(model_name, "#333333")

    # Realised returns
    ax.plot(sub["date"], sub["realised_r"],
            color="steelblue", lw=0.7, alpha=0.8, label="Realised Return", zorder=2)

    # Negative VaR boundary
    ax.plot(sub["date"], -sub[var_col],
            color=VaR95_COLOR, lw=1.4, linestyle="--",
            label=f"−VaR {pct}%", zorder=3)

    # Negative ES boundary
    ax.plot(sub["date"], -sub[es_col],
            color=VaR99_COLOR, lw=1.0, linestyle=":",
            label=f"−ES {pct}%", zorder=3)

    # Shaded danger zone
    ax.fill_between(
        sub["date"],
        -sub[var_col], sub["realised_r"].min() * 1.1,
        where=np.ones(len(sub), dtype=bool),
        alpha=0.06, color=VaR95_COLOR, label="_nolegend_"
    )

    # Violation markers
    viol_dates   = sub.loc[violations, "date"]
    viol_returns = sub.loc[violations, "realised_r"]
    ax.scatter(viol_dates, viol_returns,
               color=VIOLATION_COLOR, s=25, zorder=5,
               label=f"Violations (n={violations.sum()})", alpha=0.9)

    ax.axhline(0, color="black", lw=0.5, alpha=0.4)

    # Annotations
    n_viol = violations.sum()
    rate   = n_viol / len(sub) * 100
    ax.set_title(
        f"{model_name} — Realised Returns vs {pct}% VaR  "
        f"[violations: {n_viol} / {len(sub)} = {rate:.1f}%]{title_suffix}",
        fontsize=11, fontweight="bold"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Log-Return")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax.legend(fontsize=8, loc="upper left")

    return ax


# ===========================================================================
# 2. All-Models 2×2 Grid
# ===========================================================================

def plot_all_models_var(
    forecast_df:      pd.DataFrame,
    confidence_level: float = 0.95,
    figsize:          tuple = (16, 10),
) -> plt.Figure:
    """
    2×2 subplot grid: one panel per model.
    Shows realised returns overlaid with VaR and ES bands.

    Returns
    -------
    plt.Figure
    """
    models = ["GARCH", "Heston", "Merton", "HistSim"]
    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=False)
    axes_flat = axes.flatten()

    for ax, model in zip(axes_flat, models):
        if model not in forecast_df["model"].unique():
            ax.set_visible(False)
            continue
        plot_returns_with_var(forecast_df, model, confidence_level, ax=ax)

    pct = int(confidence_level * 100)
    fig.suptitle(
        f"Backtesting Comparison — {pct}% VaR Across All Models",
        fontsize=14, fontweight="bold", y=1.01
    )
    fig.tight_layout()
    return fig


# ===========================================================================
# 3. Violation Count Bar Chart
# ===========================================================================

def plot_violation_comparison(
    backtest_summary:  pd.DataFrame,
    figsize:           tuple = (12, 6),
) -> plt.Figure:
    """
    Grouped bar chart: observed violations vs. expected violations for
    each (model, confidence level) combination.

    A bar reaching the expected line = well-calibrated model.
    Bars significantly above = model under-estimates risk.
    Bars significantly below = model over-estimates risk.

    Parameters
    ----------
    backtest_summary : pd.DataFrame  Output of Backtester.run_all_tests()

    Returns
    -------
    plt.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=False)
    cls = sorted(backtest_summary["confidence_level"].unique())

    for ax, cl in zip(axes, cls):
        pct   = int(cl * 100)
        sub   = backtest_summary[backtest_summary["confidence_level"] == cl].copy()
        sub   = sub.sort_values("model")

        models   = sub["model"].tolist()
        observed = sub["V"].tolist()
        expected = (sub["p_expected"] * sub["T"]).round(1).tolist()

        x  = np.arange(len(models))
        w  = 0.35

        bars_obs = ax.bar(x - w/2, observed, w, label="Observed", color="#E53935", alpha=0.85)
        bars_exp = ax.bar(x + w/2, expected, w, label="Expected", color="#1E88E5", alpha=0.85)

        # Colour bars by reject status
        for i, (_, row) in enumerate(sub.iterrows()):
            c = "#B71C1C" if row["uc_reject"] else "#E53935"
            bars_obs[i].set_color(c)

        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15)
        ax.set_title(f"{pct}% VaR — Violation Counts", fontweight="bold")
        ax.set_ylabel("Number of Violations")
        ax.legend()

        # Annotate with p-values
        for i, (_, row) in enumerate(sub.iterrows()):
            status = "✗" if row["uc_reject"] else "✓"
            ax.text(x[i] - w/2, observed[i] + 0.5, status,
                    ha="center", va="bottom", fontsize=10,
                    color="#B71C1C" if row["uc_reject"] else "#2E7D32")

    fig.suptitle("Observed vs. Expected VaR Violations by Model",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


# ===========================================================================
# 4. P-value Heatmap
# ===========================================================================

def plot_backtest_summary_heatmap(
    backtest_summary: pd.DataFrame,
    figsize:          tuple = (12, 6),
) -> plt.Figure:
    """
    Heatmap of test p-values across models and confidence levels.
    Green = model passes (p ≥ 0.05); Red = model rejected.

    Rows = models, Columns = (test, confidence level) combinations.

    Returns
    -------
    plt.Figure
    """
    # Build pivot-ready records
    records = []
    for _, row in backtest_summary.iterrows():
        pct = int(row["confidence_level"] * 100)
        records += [
            {"model": row["model"], "test": f"Kupiec {pct}%",   "pvalue": row["uc_pvalue"]},
            {"model": row["model"], "test": f"Ind. {pct}%",     "pvalue": row["ind_pvalue"]},
            {"model": row["model"], "test": f"CC {pct}%",       "pvalue": row["cc_pvalue"]},
        ]
    pv_df  = pd.DataFrame(records)
    pivot  = pv_df.pivot(index="model", columns="test", values="pvalue")

    # Reorder columns logically
    col_order = []
    for pct in [95, 99]:
        col_order += [f"Kupiec {pct}%", f"Ind. {pct}%", f"CC {pct}%"]
    pivot = pivot.reindex(columns=[c for c in col_order if c in pivot.columns])

    fig, ax = plt.subplots(figsize=figsize)

    # Use a diverging palette: red below 0.05, green above
    cmap = sns.diverging_palette(10, 130, as_cmap=True)
    sns.heatmap(
        pivot,
        ax=ax,
        annot=True,
        fmt=".3f",
        cmap=cmap,
        vmin=0.0,
        vmax=0.5,
        center=0.05,          # 0.05 significance threshold at the midpoint
        linewidths=0.5,
        cbar_kws={"label": "p-value"},
    )

    # Draw significance threshold line annotation
    ax.axvline(x=3, color="white", lw=2.5, ls="--", alpha=0.7)

    ax.set_title(
        "Backtest p-values Heatmap\n"
        "(Green ≥ 0.05 = PASS,  Red < 0.05 = REJECT H₀)",
        fontsize=12, fontweight="bold"
    )
    ax.set_xlabel("Test")
    ax.set_ylabel("Model")
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    return fig


# ===========================================================================
# 5. VaR & ES Over Time
# ===========================================================================

def plot_var_es_over_time(
    forecast_df:      pd.DataFrame,
    model_name:       str,
    confidence_level: float = 0.95,
    figsize:          tuple = (14, 5),
) -> plt.Figure:
    """
    Line chart showing how VaR and ES forecasts evolve over the test window.
    Useful for illustrating model responsiveness to volatility regimes.

    Returns
    -------
    plt.Figure
    """
    pct = int(confidence_level * 100)
    sub = forecast_df[forecast_df["model"] == model_name].sort_values("date").copy()
    sub["date"] = pd.to_datetime(sub["date"])

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(sub["date"], sub[f"VaR_{pct}"],
            color=VaR95_COLOR, lw=1.5, label=f"VaR {pct}%")
    ax.plot(sub["date"], sub[f"ES_{pct}"],
            color=VaR99_COLOR, lw=1.5, linestyle="--", label=f"ES {pct}%")
    ax.fill_between(sub["date"], sub[f"VaR_{pct}"], sub[f"ES_{pct}"],
                    alpha=0.15, color=VaR99_COLOR, label="ES–VaR spread")

    ax.set_title(f"{model_name} — {pct}% VaR and ES Over Time",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Risk Estimate (log-return units)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax.legend()
    fig.tight_layout()
    return fig


# ===========================================================================
# 6. Return Distribution Comparison
# ===========================================================================

def plot_return_distributions(
    forecast_df:       pd.DataFrame,
    empirical_returns: pd.Series,
    confidence_level:  float = 0.95,
    figsize:           tuple = (14, 6),
) -> plt.Figure:
    """
    Overlay histogram + KDE of each model's VaR forecasts against the
    empirical return distribution.

    Fatter left tails → more conservative (better?) VaR estimates.
    Shows how jump-diffusion and SV models differ from GARCH in the tails.

    Returns
    -------
    plt.Figure
    """
    pct     = int(confidence_level * 100)
    var_col = f"VaR_{pct}"
    models  = forecast_df["model"].unique()

    fig, ax = plt.subplots(figsize=figsize)

    # Empirical return distribution
    sns.histplot(empirical_returns, bins=80, stat="density", alpha=0.25,
                 color="steelblue", label="Empirical returns", ax=ax, kde=True)

    # Each model's negative-VaR values (approximation of their tail threshold)
    for model in models:
        sub   = forecast_df[forecast_df["model"] == model]
        var_vals = -sub[var_col]   # expressed as negative numbers (loss side)
        color = COLORS.get(model, "grey")
        ax.axvline(var_vals.mean(), color=color, lw=2,
                   linestyle="--", label=f"{model} mean −VaR{pct}%")
        # Shade the region to the left of the mean VaR
        ax.axvspan(var_vals.min(), var_vals.mean(),
                   alpha=0.04, color=color)

    ax.set_title(
        f"Empirical Return Distribution with Mean −VaR {pct}% by Model\n"
        f"(Dashed lines = average VaR threshold on the loss axis)",
        fontsize=11, fontweight="bold"
    )
    ax.set_xlabel("Log-Return")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# ===========================================================================
# 7. Violation Clustering Raster
# ===========================================================================

def plot_violation_clustering(
    forecast_df:       pd.DataFrame,
    confidence_level:  float = 0.95,
    figsize:           tuple = (14, 5),
) -> plt.Figure:
    """
    Event-raster (rug) plot of violation dates across all models.
    Each model gets a horizontal row; a tick at date t means a VaR breach.

    Horizontal clusters → violations are not independent → Christoffersen
    independence test should reject H₀.

    Returns
    -------
    plt.Figure
    """
    pct     = int(confidence_level * 100)
    models  = list(forecast_df["model"].unique())
    n       = len(models)

    fig, ax = plt.subplots(figsize=figsize)

    for i, model in enumerate(models):
        sub  = forecast_df[forecast_df["model"] == model].sort_values("date").copy()
        sub["date"] = pd.to_datetime(sub["date"])
        mask = sub["realised_r"] < -sub[f"VaR_{pct}"]
        dates = sub.loc[mask, "date"]

        ax.eventplot(
            [d.toordinal() for d in dates],
            lineoffsets=i + 1,
            linelengths=0.7,
            colors=COLORS.get(model, "grey"),
            label=model,
        )

    # Convert ordinal ticks back to year labels
    all_dates = pd.to_datetime(forecast_df["date"])
    years     = sorted(set(d.year for d in all_dates))
    tick_pos  = [pd.Timestamp(str(y)).toordinal() for y in years]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels([str(y) for y in years], rotation=30)

    ax.set_yticks(range(1, n + 1))
    ax.set_yticklabels(models)
    ax.set_title(
        f"VaR {pct}% Violation Dates — Raster Plot\n"
        "(Dense columns = clustered violations → independence rejected)",
        fontsize=11, fontweight="bold"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Model")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


# ===========================================================================
# 8. Convenience: Save All Plots
# ===========================================================================

def save_all_plots(
    forecast_df:       pd.DataFrame,
    backtest_summary:  pd.DataFrame,
    empirical_returns: pd.Series,
    output_dir:        str  = RESULTS_DIR,
    dpi:               int  = 150,
) -> List[str]:
    """
    Render and save all seven chart types to `output_dir`.

    Parameters
    ----------
    forecast_df       : full rolling backtest forecast DataFrame
    backtest_summary  : output of Backtester.run_all_tests()
    empirical_returns : full log-return series from MarketDataLoader
    output_dir        : directory to save PNG files
    dpi               : image resolution

    Returns
    -------
    List of saved file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    saved = []

    def _save(fig, name):
        path = os.path.join(output_dir, name)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)
        print(f"  [Saved] {path}")

    print("\n[Plotter] Generating and saving all charts ...")

    # 1. Individual model panels at 95%
    for model in forecast_df["model"].unique():
        fig, ax = plt.subplots(figsize=(14, 5))
        plot_returns_with_var(forecast_df, model, 0.95, ax=ax)
        fig.tight_layout()
        _save(fig, f"01_returns_var95_{model}.png")

    # 2. All-models 2×2 grid at 95%
    fig = plot_all_models_var(forecast_df, confidence_level=0.95)
    _save(fig, "02_all_models_var95_grid.png")

    # 3. All-models 2×2 grid at 99%
    fig = plot_all_models_var(forecast_df, confidence_level=0.99)
    _save(fig, "03_all_models_var99_grid.png")

    # 4. Violation bar chart
    fig = plot_violation_comparison(backtest_summary)
    _save(fig, "04_violation_comparison_bars.png")

    # 5. P-value heatmap
    fig = plot_backtest_summary_heatmap(backtest_summary)
    _save(fig, "05_pvalue_heatmap.png")

    # 6. VaR + ES over time per model
    for model in forecast_df["model"].unique():
        for cl in CONFIDENCE_LEVELS:
            pct = int(cl * 100)
            fig = plot_var_es_over_time(forecast_df, model, cl)
            _save(fig, f"06_var_es_overtime_{model}_{pct}.png")

    # 7. Return distribution comparison
    fig = plot_return_distributions(forecast_df, empirical_returns, 0.95)
    _save(fig, "07_return_distribution_comparison.png")

    # 8. Violation clustering rasters
    for cl in CONFIDENCE_LEVELS:
        pct = int(cl * 100)
        fig = plot_violation_clustering(forecast_df, cl)
        _save(fig, f"08_violation_clustering_{pct}.png")

    print(f"[Plotter] {len(saved)} charts saved to '{output_dir}/'.")
    return saved
