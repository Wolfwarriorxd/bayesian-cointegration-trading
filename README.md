# Bayesian Optimization for Cointegration-Based Basket Trading

**Author:** Anish Deshmukh
**Date:** 2025-12-12

---

## Project summary

This repository implements a cointegration-based, mean-reversion basket trading pipeline in which key trading parameters (hedge weights, entry/exit thresholds, leverage and volatility targeting) are optimized with **Bayesian Optimization (BO)**. The project evaluates out-of-sample performance using a rolling walk-forward protocol and compares BO-optimized rules to a Johansen-test baseline.

The goal is to maximize genuine out-of-sample performance (Sharpe ratio) while accounting for turnover and transaction costs.

---

## Key numerical results (rolling out-of-sample averages)

> Results are reported as averages across rolling test windows (typical configuration: 252-day train / 63-day test / 30-day step). See `results/` for full per-window tables.

**BO-optimized strategy (mean across OOS windows):**

- **Sharpe (mean):** **1.073463**  
- **Total return (mean):** **0.054657**  
- **Annualized volatility (mean):** **0.195072**  
- **Max drawdown (mean):** **-0.060086**

**Johansen baseline (mean across OOS windows):**

- **Sharpe (mean):** 0.441129  
- **Total return (mean):** 0.000801  
- **Annualized volatility (mean):** 0.006169  
- **Max drawdown (mean):** -0.002174

**Statistical tests (paired across windows):**

- Paired t-test (BO vs Johansen): **t ≈ 3.247**, **p ≈ 0.0017**  
- Wilcoxon signed-rank p-value: **≈ 0.0008**

**Transaction-cost sensitivity (mean Sharpe):**

| Transaction cost per unit turnover | Mean Sharpe |
|-----------------------------------:|------------:|
| 0.0000                             | 1.0735      |
| 0.0005                             | 0.6873      |
| 0.0010                             | 0.3436      |
| 0.0025                             | -0.4411     |
| 0.0050                             | -1.2204     |

Interpretation: BO produces statistically significant improvement in out-of-sample Sharpe for zero-to-low transaction costs in these experiments; strategy performance degrades materially with larger execution costs.

---

## Methodology (concise)

1. **Data:** daily adjusted close prices for a 3-asset basket (configurable).  
2. **Baseline:** compute Johansen cointegration vector on log prices; normalize by `sum(abs(weights))`.  
3. **Objective:** BO maximizes a validation Sharpe metric penalized by average turnover (`Sharpe - λ·Turnover`). BO searches over:
   - static weights `w1, w2, w3` (continuous bounds),  
   - `entry_z` and `exit_z` thresholds,  
   - `leverage` scaling (0.5–2.0).  
4. **Evaluation:** rolling walk-forward:
   - For each train window run BO (train or train/val),
   - Select the best candidate,
   - Evaluate that candidate on the next test window,
   - Aggregate OOS metrics across windows.  
5. **Diagnostics:** compute trade-level stats, turnover, and a TC sweep. Perform paired statistical tests across per-window Sharpe.  
6. **Robustness options:** ensemble averaging of top-K candidates, turnover filtering, and regularization on weights.

Visualization & presentation

plots/sharpe_histogram.png — distribution of per-window Sharpe (BO vs baseline).

plots/tc_sensitivity.png — transaction-cost sensitivity curve.

plots/cumulative_example.png — cumulative returns for an example OOS window.

Citation / Acknowledgements

This project uses open-source libraries including numpy, pandas, matplotlib, statsmodels, scipy, and bayesian-optimization. Data in the notebook was sourced from publicly available APIs (e.g. / Tiingo). Please respect data-provider terms when using third-party data.
