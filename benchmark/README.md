# Benchmark code

This folder is reserved for the code that produced the files in `results/`:
data preparation, model training, conformal calibration, the fault injector
and the evaluation sweep.

That code is being prepared for release and is not included yet. Everything
else in the repository works without it, because the analysis starts from the
released results.

The settings of the run are recorded in `configs/experiment_config.json`
(seeds, window length, RUL cap, sensors kept, training hyperparameters, number
of dropout passes, decision threshold, cost weights, library versions and
dataset checksums). The fault definitions are summarised in
`docs/fault_taxonomy.md`.

## Planned layout

```
benchmark/
├── prepare_data.py     raw C-MAPSS files -> data/processed/*.npz
├── models.py           LSTM, 1D-CNN and Transformer definitions
├── train.py            training with early stopping, five seeds per model
├── conformal.py        Monte Carlo dropout and split-conformal calibration
├── faults.py           the twelve fault operators and their severity grids
├── metrics.py          accuracy, interval and decision metrics
└── run_sweep.py        36,000 evaluations -> results/sweep_v2_*.csv
```
