# Tail Risk Estimation: GARCH, Heston & Merton Jump-Diffusion Backtesting

This project evaluates the performance of different volatility and stochastic frameworks—GARCH(1,1), Heston Stochastic Volatility, and Merton Jump-Diffusion—for 1-day Value at Risk (VaR) and Expected Shortfall (ES) forecasting on the NIFTY50 index.

## Key Findings & Model Comparison

The backtest was conducted over 636 trading days. Models were evaluated using the **Kupiec POF (Unconditional Coverage)** test to verify the total violation count, and the **Christoffersen (Conditional Coverage)** test to check for violation clustering.

| Model | Confidence Level | Hit Rate ($p_{hat}$) | Expected Rate | Kupiec POF Result | Christoffersen CC Result | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Merton Jump-Diffusion** | 95% | 5.66% | 5.0% | PASSED (p=0.45) | PASSED (p=0.50) | **Optimal** |
| **Merton Jump-Diffusion** | 99% | 1.41% | 1.0% | PASSED (p=0.32) | PASSED (p=0.11) | **Optimal** |
| **Historical Simulation** | 95% | 5.50% | 5.0% | PASSED (p=0.56) | PASSED (p=0.95) | Robust Baseline |
| **Historical Simulation** | 99% | 1.25% | 1.0% | PASSED (p=0.52) | PASSED (p=0.08) | Robust Baseline |
| **Heston Volatility** | 95% | 6.44% | 5.0% | PASSED (p=0.10) | FAILED (p=0.04) | Violation Clustering |
| **Heston Volatility** | 99% | 3.45% | 1.0% | FAILED (p=0.00) | FAILED (p=0.00) | Severe Risk Underestimation |
| **GARCH(1,1)** | 95% | 7.07% | 5.0% | FAILED (p=0.02) | PASSED (p=0.90) | Inadequate |
| **GARCH(1,1)** | 99% | 2.83% | 1.0% | FAILED (p=0.00) | FAILED (p=0.00) | Severe Risk Underestimation |

### **Core Analytical Insights**

1. **The Failure of Continuous Diffusion (GARCH & Heston at 99%)**
   At the extreme tail (99% confidence), both GARCH ($p_{hat} = 2.83\%$) and Heston ($p_{hat} = 3.45\%$) severely underestimated the true risk of the NIFTY50. Because both frameworks model asset prices as a continuous Brownian path, they generate thin-tailed future distributions that fail to account for sudden, discontinuous macro shocks characteristic of Indian equities.
   
2. **Heston's Vulnerability: Violation Clustering**
   While Heston at 95% technically passed overall coverage, it failed the Christoffersen test due to severe violation clustering ($\pi_{11} = 14.63\%$). If a VaR breach occurs, there is an alarmingly high probability of a consecutive breach the next day, indicating the model's calibration window does not adapt fast enough to abrupt regime changes.

3. **Why Merton Won**
   By incorporating a compound Poisson process to explicitly model jump dynamics, the Merton Jump-Diffusion model successfully captured the heavy-tailed behavior of the NIFTY50. It was the only parametric model to successfully navigate both the 95% and 99% horizons without triggering statistical rejections.