"""
data/data_loader.py
===================
Data Module — responsible for:
  1. Fetching raw OHLCV price data from Yahoo Finance via `yfinance`.
  2. Computing daily log-returns.
  3. Providing a rolling-window generator that yields (calibration, test)
     slices used by the backtesting loop.

Mathematical Background
-----------------------
Log-Return Definition
    r_t = ln(S_t / S_{t-1})

where S_t is the adjusted closing price on day t.

Log-returns are preferred over simple returns because:
  • They are additive over time: r_{0→T} = Σ r_t
  • They are more symmetric and approximately Gaussian for short horizons
  • They directly map to the log-price process assumed by GBM and its extensions

Rolling Window
--------------
Given a total series of length T, calibration window W, and step size h:
  • Window  k uses observations  [k·h, k·h + W)  for fitting.
  • The out-of-sample test observation is at index  k·h + W.
  • The loop runs for  floor((T - W) / h)  iterations.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from typing import Generator, Tuple
import sys
import os

# Allow imports from parent directory when running module directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    TICKER, START_DATE, END_DATE,
    CALIBRATION_WINDOW, TEST_STEP
)


# ===========================================================================
# CLASS: MarketDataLoader
# ===========================================================================

class MarketDataLoader:
    """
    Downloads and pre-processes market price data for a single index/equity.

    Parameters
    ----------
    ticker : str
        Yahoo Finance ticker symbol (e.g. "^NSEI", "^GSPC").
    start_date : str
        Start of the history in "YYYY-MM-DD" format.
    end_date : str
        End of the history in "YYYY-MM-DD" format.

    Attributes
    ----------
    prices : pd.Series
        Adjusted closing prices indexed by date.
    log_returns : pd.Series
        Daily log-returns  r_t = ln(S_t / S_{t-1}).
    """

    def __init__(
        self,
        ticker: str     = TICKER,
        start_date: str = START_DATE,
        end_date: str   = END_DATE,
    ):
        self.ticker     = ticker
        self.start_date = start_date
        self.end_date   = end_date

        # Will be populated by load()
        self.prices: pd.Series      = None
        self.log_returns: pd.Series = None

    # -----------------------------------------------------------------------
    def load(self) -> "MarketDataLoader":
        """
        Fetch data from Yahoo Finance, compute log-returns, and store them.

        Returns
        -------
        self  (fluent interface — enables  loader.load().get_returns())
        """
        print(f"[DataLoader] Downloading '{self.ticker}' "
              f"from {self.start_date} to {self.end_date} ...")

        raw = yf.download(
            self.ticker,
            start=self.start_date,
            end=self.end_date,
            auto_adjust=True,   # Use split/dividend-adjusted closes
            progress=False,
        )

        if raw.empty:
            raise ValueError(
                f"No data returned for ticker '{self.ticker}'. "
                "Check the symbol and date range."
            )

        # Extract adjusted close; flatten MultiIndex columns if present
        if isinstance(raw.columns, pd.MultiIndex):
            close = raw["Close"][self.ticker]
        else:
            close = raw["Close"]

        close = close.dropna()
        close.name = self.ticker

        # -------------------------------------------------------------------
        # Log-Return Calculation
        # r_t = ln(S_t) - ln(S_{t-1}) = ln(S_t / S_{t-1})
        # np.log(close).diff() implements this efficiently.
        # The first value is NaN (no prior price) and is dropped.
        # -------------------------------------------------------------------
        log_ret = np.log(close).diff().dropna()
        log_ret.name = "log_return"

        self.prices      = close
        self.log_returns = log_ret

        print(f"[DataLoader] {len(self.prices)} price observations loaded.")
        print(f"[DataLoader] {len(self.log_returns)} log-return observations computed.")
        print(f"[DataLoader] Date range: "
              f"{self.log_returns.index[0].date()} → "
              f"{self.log_returns.index[-1].date()}")

        return self  # fluent

    # -----------------------------------------------------------------------
    def summary_statistics(self) -> pd.DataFrame:
        """
        Return descriptive statistics for the log-return series.

        Includes annualised mean and volatility for quick sanity-checking.
        """
        if self.log_returns is None:
            raise RuntimeError("Call .load() before .summary_statistics().")

        r = self.log_returns
        from config import TRADING_DAYS_YEAR as TDY

        stats = {
            "N observations":          len(r),
            "Mean (daily)":            r.mean(),
            "Std  (daily)":            r.std(),
            "Mean (annualised)":       r.mean() * TDY,
            "Vol  (annualised)":       r.std()  * np.sqrt(TDY),
            "Skewness":                r.skew(),
            "Excess Kurtosis":         r.kurt(),   # pandas returns excess kurtosis
            "Min":                     r.min(),
            "Max":                     r.max(),
            "5th Percentile":          r.quantile(0.05),
            "1st Percentile":          r.quantile(0.01),
        }

        df = pd.DataFrame.from_dict(stats, orient="index", columns=["Value"])
        df["Value"] = df["Value"].round(6)
        return df

    # -----------------------------------------------------------------------
    def rolling_windows(
        self,
        calibration_window: int = CALIBRATION_WINDOW,
        step: int               = TEST_STEP,
    ) -> Generator[Tuple[pd.Series, pd.Timestamp, float], None, None]:
        """
        Generator that yields successive (calibration_slice, test_date, test_return)
        tuples for the rolling backtesting loop.

        Parameters
        ----------
        calibration_window : int
            Number of past observations used to calibrate each model.
        step : int
            Number of days to advance before the next re-calibration.
            step=1  → daily re-calibration (computationally expensive but
                       maximally responsive to regime changes).
            step=5  → weekly re-calibration (faster for prototyping).

        Yields
        ------
        cal_returns  : pd.Series
            The W most recent log-returns available on the forecast date.
        test_date    : pd.Timestamp
            The date for which we are forecasting risk.
        test_return  : float
            The realised log-return on test_date (used to count VaR violations).

        Example
        -------
        >>> loader = MarketDataLoader().load()
        >>> for cal, date, r_actual in loader.rolling_windows():
        ...     var_95 = some_model.forecast_var(cal, alpha=0.05)
        ...     violation = (r_actual < -var_95)
        """
        if self.log_returns is None:
            raise RuntimeError("Call .load() before .rolling_windows().")

        returns = self.log_returns.values          # numpy array for speed
        dates   = self.log_returns.index
        T       = len(returns)

        if calibration_window >= T:
            raise ValueError(
                f"calibration_window ({calibration_window}) must be "
                f"smaller than total observations ({T})."
            )

        # -------------------------------------------------------------------
        # Rolling window loop
        #   k  = window index (0-based)
        #   i  = index of the LAST calibration observation
        #   i+1= index of the test (out-of-sample) observation
        # -------------------------------------------------------------------
        k = 0
        while True:
            start_idx = k * step                   # first calibration obs
            end_idx   = start_idx + calibration_window  # exclusive end
            test_idx  = end_idx                    # out-of-sample index

            if test_idx >= T:
                break  # exhausted the series

            cal_returns = pd.Series(
                returns[start_idx:end_idx],
                index=dates[start_idx:end_idx],
                name="log_return",
            )
            test_date   = dates[test_idx]
            test_return = returns[test_idx]

            yield cal_returns, test_date, test_return

            k += 1

    # -----------------------------------------------------------------------
    def train_test_split(
        self,
        calibration_window: int = CALIBRATION_WINDOW,
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Simple one-time split: first `calibration_window` observations for
        training, the remainder for testing.

        Useful for single-fit model comparison (not rolling).

        Returns
        -------
        train_returns : pd.Series
        test_returns  : pd.Series
        """
        if self.log_returns is None:
            raise RuntimeError("Call .load() before .train_test_split().")

        train = self.log_returns.iloc[:calibration_window]
        test  = self.log_returns.iloc[calibration_window:]
        return train, test


# ===========================================================================
# STANDALONE UTILITY FUNCTIONS
# ===========================================================================

def compute_realised_variance(
    log_returns: pd.Series,
    window: int = 21,
) -> pd.Series:
    """
    Realised Variance (RV) over a rolling window.

    RV_t = Σ_{i=t-w+1}^{t} r_i^2

    Realised variance is a model-free estimator of integrated variance and
    provides a benchmark against which model-implied variances can be compared.

    Parameters
    ----------
    log_returns : pd.Series
    window      : int   Rolling window length in days (default 21 ≈ 1 month)

    Returns
    -------
    pd.Series of realised variance estimates
    """
    return (log_returns ** 2).rolling(window).sum()


def annualise_vol(daily_std: float, trading_days: int = 252) -> float:
    """
    Annualise a daily standard deviation using the square-root-of-time rule.

    σ_annual = σ_daily × √(trading_days)

    Note: this rule assumes i.i.d. returns; it understates vol in the
    presence of autocorrelation or volatility clustering.
    """
    return daily_std * np.sqrt(trading_days)


# ===========================================================================
# QUICK TEST  (run:  python -m data.data_loader)
# ===========================================================================

if __name__ == "__main__":
    loader = MarketDataLoader().load()

    print("\n=== Summary Statistics ===")
    print(loader.summary_statistics().to_string())

    print("\n=== First 3 Rolling Windows ===")
    for i, (cal, date, r) in enumerate(loader.rolling_windows(step=5)):
        print(f"  Window {i+1}: cal_end={cal.index[-1].date()}, "
              f"test_date={date.date()}, realised_r={r:.5f}")
        if i == 2:
            break
