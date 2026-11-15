# Results

## Files

| File | Rows | Content |
|---|---|---|
| `sweep_v2_FD001.csv` ... `sweep_v2_FD004.csv` | 9,000 each | one row per corrupted evaluation |
| `baseline_clean_v2.csv` | 60 | clean-input reference, one row per trained model |
| `significance_sev3_v2.csv` | 144 | Wilcoxon tests written by the benchmark run |
| `original_tables/` | | LaTeX tables written by the benchmark run, kept for provenance |
| `tables/` | | tables rebuilt by `analysis/make_tables.py` |

The suffix `v2` marks the final protocol of the benchmark, which is the one
reported in the paper.

Each subset contributes 3 models x 5 training seeds x 12 faults x 5 severities
x 10 injection seeds = 9,000 rows.

## Columns

| Column | Meaning |
|---|---|
| `subset` | C-MAPSS subset: FD001, FD002, FD003 or FD004 |
| `model` | architecture: LSTM, CNN or Transformer |
| `train_seed` | training seed of the model, 42 to 46 |
| `corruption` | fault code, see `docs/fault_taxonomy.md` |
| `severity` | fault severity, 1 (incipient) to 5 (gross) |
| `inj_seed` | seed of the fault injection, 0 to 9 |
| `RMSE` | root mean square error of the RUL prediction, in cycles |
| `MAE` | mean absolute error, in cycles |
| `R2` | coefficient of determination |
| `Score` | PHM08 score, summed over the test engines (late predictions cost more) |
| `PICP` | share of test engines whose true RUL lies inside the nominal 95% interval |
| `MPIW` | mean width of the 95% interval, in cycles |
| `FalseAlarmRate` | share of healthy engines (true RUL above 30) flagged for maintenance |
| `MissRate` | share of engines due for maintenance (true RUL of 30 or less) that were not flagged |
| `MaintCost` | 10 x false alarms + 50 x missed alarms, summed over the test engines |

`baseline_clean_v2.csv` has the same metric columns without `corruption`,
`severity` and `inj_seed`.

Definitions of all metrics are in `docs/metrics.md`.

## Notes for reuse

- `MaintCost` and `Score` are sums over test engines, and the subsets have
  different numbers of test engines (100, 259, 100, 248). Divide by the clean
  value of the same model before comparing across subsets.
- Monte Carlo dropout makes each evaluation stochastic. Small differences
  between identical settings are expected.
- At severity 1, sampling reduction uses a factor of 1 and leaves the input
  unchanged, so those rows are a second clean reference.
