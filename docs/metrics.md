# Metrics

All metrics are computed on the test engines of one subset, for one trained
model and one input condition. Let `y` be the true RUL and `yhat` the
prediction, both in cycles and capped at 125.

## Accuracy

| Metric | Definition |
|---|---|
| RMSE | square root of the mean squared error |
| MAE | mean absolute error |
| R2 | coefficient of determination |
| PHM08 score | sum over engines of `exp(-d/13) - 1` if `d < 0`, else `exp(d/10) - 1`, with `d = yhat - y` |

The PHM08 score penalises late predictions (`d > 0`) more than early ones.

## Prediction intervals

The predictive mean and standard deviation come from 30 Monte Carlo dropout
passes. Split-conformal calibration scales the standard deviation by a factor
`q` fitted on the validation windows, so that the interval
`[mean - q * sd, mean + q * sd]` has nominal coverage 0.95.

| Metric | Definition |
|---|---|
| PICP | share of engines whose true RUL lies inside the interval |
| MPIW | mean width of the interval, in cycles |

## Maintenance decisions

A threshold policy triggers maintenance when the predicted RUL is 30 cycles or
less. An engine is due for maintenance when its true RUL is 30 cycles or less.

| Metric | Definition |
|---|---|
| False-alarm rate (FAR) | flagged healthy engines / all healthy engines |
| Missed-alarm rate (MR) | unflagged due engines / all due engines |
| Maintenance cost | `10 * false alarms + 50 * missed alarms` |

## Corruption Robustness Index

For a fault `k`, one architecture and one subset `j`:

```
ratio(j, k) = mean RMSE under fault k (over severities and seeds) / mean clean RMSE (over seeds)
CRI(k)      = mean of ratio(j, k) over the four subsets
```

A value of 1 means that the fault leaves RMSE unchanged. `dqtb.cri()` returns
the ratios at the finest level; average them over `subset`, and over `model`
for pooled values.

## Statistical tests

Architectures are compared with two-sided Wilcoxon signed-rank tests on
severity-3 RMSE. Observations are paired by training seed and injection seed
(50 pairs per test). There are 144 tests (4 subsets x 12 faults x 3 pairs);
p-values are reported raw and after Holm correction across all 144.
