"""Shared helpers for the DQ-Twin-Bench analysis scripts.

Everything here works on the released result files in ``results/``. No model
training or fault injection happens in this module.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
FIGURES = ROOT / "figures"
CONFIG = ROOT / "configs" / "experiment_config.json"
PROCESSED = ROOT / "data" / "processed"

SUBSETS = ["FD001", "FD002", "FD003", "FD004"]
MODELS = ["LSTM", "CNN", "Transformer"]

# Fault codes used in the CSV files, ordered by pooled robustness index (most damaging first).
ORDER = [
    "drift", "missing_mnar", "concept_drift", "missing_mar", "bias_offset", "sensor_dropout",
    "nongaussian_noise", "sampling_reduction", "gaussian_noise", "missing_mcar",
    "channel_desync", "timestamp_jitter",
]

# Names used in the paper.
NAME = {
    "drift": "Calibration drift",
    "missing_mnar": "MNAR missingness",
    "concept_drift": "Regime shift",
    "missing_mar": "MAR missingness",
    "bias_offset": "Calibration bias",
    "sensor_dropout": "Sensor dropout",
    "nongaussian_noise": "Heavy-tailed noise",
    "sampling_reduction": "Sampling reduction",
    "gaussian_noise": "Gaussian noise",
    "missing_mcar": "MCAR missingness",
    "channel_desync": "Channel desynchronisation",
    "timestamp_jitter": "Timestamp jitter",
}

METRICS = ["RMSE", "MAE", "R2", "Score", "PICP", "MPIW", "FalseAlarmRate", "MissRate", "MaintCost"]
KEYS = ["subset", "model", "train_seed"]


def load_config() -> dict:
    """Return the experiment configuration written by the benchmark run."""
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def load_baseline() -> pd.DataFrame:
    """Clean-input results: one row per trained model (4 subsets x 3 models x 5 seeds)."""
    return pd.read_csv(RESULTS / "baseline_clean_v2.csv")


def load_sweep() -> pd.DataFrame:
    """Corrupted-input results: 36,000 rows, one per evaluation."""
    frames = [pd.read_csv(RESULTS / f"sweep_v2_{s}.csv") for s in SUBSETS]
    return pd.concat(frames, ignore_index=True)


def load_all() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (sweep, baseline, merged).

    ``merged`` is the sweep with the matching clean-input metrics of the same
    trained model attached under the suffix ``_c``, plus two convenience
    columns: ``ratio`` (RMSE / clean RMSE) and ``width_ratio`` (MPIW / clean MPIW).
    """
    sweep, baseline = load_sweep(), load_baseline()
    merged = sweep.merge(baseline, on=KEYS, suffixes=("", "_c"), validate="many_to_one")
    merged["ratio"] = merged["RMSE"] / merged["RMSE_c"]
    merged["width_ratio"] = merged["MPIW"] / merged["MPIW_c"]
    return sweep, baseline, merged


def cri(merged: pd.DataFrame, keys: list[str]) -> pd.Series:
    """Corruption Robustness Index at the finest level.

    For every combination of ``keys`` + model + subset this returns the mean
    corrupted RMSE divided by the mean clean RMSE. Average the result over
    ``subset`` (and over ``model`` for pooled values) to obtain the figures
    reported in the paper (Eq. 10).
    """
    k = keys + ["model", "subset"]
    g = merged.groupby(k)
    return g["RMSE"].mean() / g["RMSE_c"].mean()


def holm(pvalues) -> np.ndarray:
    """Holm step-down adjusted p-values."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    n = len(p)
    adj = np.empty(n)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (n - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def wilcoxon_tests(sweep: pd.DataFrame, severity: int = 3) -> pd.DataFrame:
    """Paired two-sided Wilcoxon signed-rank tests between architectures.

    Observations are paired by (training seed, injection seed), which gives 50
    pairs per test. Returns one row per subset, fault and architecture pair,
    with raw and Holm-adjusted p-values and the model with the lower mean RMSE.
    """
    from scipy.stats import wilcoxon

    rows = []
    sev = sweep[sweep["severity"] == severity]
    for (subset, fault), g in sev.groupby(["subset", "corruption"]):
        wide = g.pivot_table(index=["train_seed", "inj_seed"], columns="model", values="RMSE")
        for a, b in [("LSTM", "CNN"), ("LSTM", "Transformer"), ("CNN", "Transformer")]:
            _, pval = wilcoxon(wide[a], wide[b])
            rows.append({
                "subset": subset, "corruption": fault, "pair": f"{a} vs {b}", "first": a, "second": b,
                "n": len(wide), "p_value": pval,
                "lower_rmse": a if wide[a].mean() < wide[b].mean() else b,
            })
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p_value"].to_numpy())
    return out


def injection_seed_cv(sweep: pd.DataFrame, severity: int = 3) -> pd.DataFrame:
    """Coefficient of variation of RMSE across the ten injection seeds, per trained model and fault."""
    sev = sweep[sweep["severity"] == severity]
    cv = sev.groupby(KEYS + ["corruption"])["RMSE"].agg(lambda x: x.std(ddof=1) / x.mean())
    return cv.rename("cv").reset_index()
