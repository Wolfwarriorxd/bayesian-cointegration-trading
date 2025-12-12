# Results Summary

## Aggregate rolling-window metrics (averages)

**BO-optimized strategy (mean across windows):**
- sharpe: 1.073463
- total_return: 0.054657
- vol_annual: 0.195072
- max_drawdown: -0.060086

**Johansen baseline (mean across windows):**
- sharpe: 0.441129
- total_return: 0.000801
- vol_annual: 0.006169
- max_drawdown: -0.002174

## Statistical tests
- Paired t-test: t-stat = 3.24719264137293, p-value = 0.0016756309996237323
- Wilcoxon signed-rank p-value = 0.0008085306756457998

## Transaction cost sensitivity (see results/tc_sensitivity.csv)
- The strategy remains profitable for transaction costs up to approximately 0.001 in these experiments.

## Interpretation and conclusion
The Bayesian-optimized static-weights approach produced higher average out-of-sample Sharpe compared to the Johansen baseline across the tested windows. The improvement is reflected in mean Sharpe and statistically supported by paired tests in many runs. The strategy is sensitive to turnover and transaction costs; include realistic execution models before trading live.

## Files produced
- results/rolling_metrics.csv
- results/aggregated_metrics.json
- results/tc_sensitivity.csv
- results/trade_stats.csv
- results/candidates_summary.json
- results/cum_window_{i}.csv  (one file per window)
- plots/sharpe_histogram.png
- plots/tc_sensitivity.png
- plots/cumulative_example.png

## Next steps
- Add Kalman-filter dynamic hedge ratios and compare.
- Add realistic slippage and order-book based transaction cost model.
- Evaluate on additional baskets and longer time periods.
