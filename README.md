# bayesian-cointegration-trading
Bayesian Optimization for Cointegration-Based Basket Trading with Rolling Out-of-Sample Evaluation
# Bayesian Optimization for Cointegration-Based Basket Trading

This project investigates whether classical cointegration-based mean-reversion strategies can be improved using Bayesian Optimization (BO).  
Instead of relying only on Johansen test hedge ratios, the system searches for weight vectors and trading thresholds that maximize out-of-sample Sharpe ratio.

## 🔍 Motivation
Cointegration is widely used for statistical arbitrage, but traditional hedge-ratio estimation often fails out-of-sample.  
The idea explored here is:

> “Treat strategy performance as a black-box function and search the parameter space directly.”

## 📐 Methodology Overview

### 1. Basket Selection  
Three sector ETFs are chosen: XLF, XLY, XLB.

### 2. Baseline Cointegration  
- Johansen test applied on log prices  
- Extract cointegrating vector as reference weights  

### 3. Bayesian Optimization  
BO searches over:
- Hedge weights  
- Entry/exit Z-score thresholds  
- Leverage  
- Volatility target  

Objective = out-of-sample Sharpe − turnover_penalty − λ × volatility

### 4. Rolling Walk-Forward Evaluation
- 252-day training
- 63-day testing  
- Evaluated across multiple windows

### 5. Statistical Significance Tests  
- Paired t-test  
- Wilcoxon signed-rank test  

## 📈 Key Results

| Metric | Johansen | BO-Optimized |
|--------|----------|---------------|
| Sharpe | 0.44 | **1.07** |
| Total Return | 0.0008 | **0.0546** |
| p-value (t-test) | — | **0.0016** |

The BO-optimized strategy delivers a statistically significant improvement in out-of-sample Sharpe ratio.

## 📦 Repository Structure
