"""Check every number quoted in the paper against the released result files.

Usage:
    python analysis/verify_claims.py           # print all checks
    python analysis/verify_claims.py --quiet   # print failures and the summary only

The script exits with status 1 if any check fails, so it can be used in CI.
Each check recomputes a statistic from ``results/`` and ``configs/`` and
compares it with the value printed in the manuscript, within rounding.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import dqtb

CHECKS: list[tuple[str, str, float, float, bool]] = []


def check(section: str, name: str, value: float, expected: float, tol: float) -> None:
    CHECKS.append((section, name, float(value), expected, abs(float(value) - expected) <= tol))


def run() -> None:
    sweep, base, m = dqtb.load_all()
    cfg = dqtb.load_config()

    pooled = dqtb.cri(m, ["corruption"]).groupby(["corruption", "model"]).mean().unstack()
    sev5 = dqtb.cri(m[m.severity == 5], ["corruption"]).groupby(["corruption", "model"]).mean().unstack()
    cell = dqtb.cri(m, ["corruption", "severity"]).groupby(["corruption", "severity"]).mean()
    g = m.groupby(["corruption", "severity"])
    picp, miss, far, width = g.PICP.mean(), g.MissRate.mean(), g.FalseAlarmRate.mean(), g.width_ratio.mean()
    cost = g.MaintCost.mean() / base.MaintCost.mean()

    s = "Design"
    check(s, "corrupted evaluations", len(sweep), 36000, 0)
    check(s, "trained models", len(base), 60, 0)
    check(s, "rows per fault and severity", sweep.groupby(["corruption", "severity"]).size().nunique(), 1, 0)
    meta = cfg["training_meta"]
    q = [v["conformal_q95"] for v in meta.values()]
    ep = [v["epochs_run"] for v in meta.values()]
    check(s, "conformal factor, min", min(q), 2.26, 0.005)
    check(s, "conformal factor, max", max(q), 10.52, 0.005)
    check(s, "conformal factor, mean", np.mean(q), 3.59, 0.005)
    check(s, "epochs, min", min(ep), 17, 0)
    check(s, "epochs, max", max(ep), 48, 0)
    check(s, "epochs, mean", np.mean(ep), 26, 0.5)
    check(s, "training time (hours)", sum(v["train_seconds"] for v in meta.values()) / 3600, 0.5, 0.02)

    s = "4.1 Clean baselines"
    gb = base.groupby(["subset", "model"])
    check(s, "FD001 LSTM RMSE", gb.RMSE.mean()["FD001", "LSTM"], 13.99, 0.005)
    check(s, "FD001 CNN RMSE", gb.RMSE.mean()["FD001", "CNN"], 14.19, 0.005)
    check(s, "FD001 Transformer RMSE", gb.RMSE.mean()["FD001", "Transformer"], 16.31, 0.005)
    check(s, "RMSE SD across seeds, min", gb.RMSE.std().min(), 0.34, 0.005)
    check(s, "RMSE SD across seeds, max", gb.RMSE.std().max(), 1.17, 0.005)
    check(s, "clean coverage, min", gb.PICP.mean().min(), 0.910, 0.0005)
    check(s, "clean coverage, max", gb.PICP.mean().max(), 0.956, 0.0005)
    check(s, "clean coverage, grand mean", base.PICP.mean(), 0.939, 0.0005)
    check(s, "clean false-alarm rate", base.FalseAlarmRate.mean(), 0.023, 0.0005)
    check(s, "clean missed-alarm rate", base.MissRate.mean(), 0.078, 0.0005)
    check(s, "clean reference cost", base.MaintCost.mean(), 172.5, 0.01)

    s = "4.2 Sensitivity ranking"
    for k, lo, hi in [("drift", 2.43, 2.94), ("missing_mnar", 2.19, 2.55), ("concept_drift", 1.75, 1.97),
                      ("missing_mar", 1.51, 1.69), ("gaussian_noise", 1.19, 1.33), ("nongaussian_noise", 1.18, 1.46),
                      ("missing_mcar", 1.05, 1.11), ("sampling_reduction", 1.23, 1.33)]:
        check(s, f"{dqtb.NAME[k]} CRI, lowest architecture", pooled.loc[k].min(), lo, 0.005)
        check(s, f"{dqtb.NAME[k]} CRI, highest architecture", pooled.loc[k].max(), hi, 0.005)
    check(s, "drift CRI at severity 5, min", sev5.loc["drift"].min(), 3.89, 0.005)
    check(s, "drift CRI at severity 5, max", sev5.loc["drift"].max(), 5.17, 0.005)
    check(s, "Gaussian noise CRI at severity 5, max", sev5.loc["gaussian_noise"].max(), 2.13, 0.005)
    rank = list(pooled.mean(axis=1).sort_values(ascending=False).index)
    check(s, "rank of Gaussian noise", rank.index("gaussian_noise") + 1, 9, 0)
    per_pair = dqtb.cri(m, ["corruption"]).unstack(["subset", "model"])
    check(s, "combinations where drift ranks first", int((per_pair.idxmax() == "drift").sum()), 12, 0)
    top2 = sum(set(per_pair[c].nlargest(2).index) == {"drift", "missing_mnar"} for c in per_pair.columns)
    check(s, "combinations with drift and MNAR as top two", top2, 11, 0)
    ranks = per_pair.rank(axis=0)
    k_, n_ = ranks.shape[1], ranks.shape[0]
    w = 12 * ((ranks.sum(axis=1) - ranks.sum(axis=1).mean()) ** 2).sum() / (k_ ** 2 * (n_ ** 3 - n_))
    check(s, "Kendall's W of fault rankings", w, 0.96, 0.005)
    by_subset = dqtb.cri(m, ["corruption"]).groupby(["corruption", "subset"]).mean().unstack()
    check(s, "lowest Spearman correlation between subsets", by_subset.corr(method="spearman").to_numpy().min(), 0.98, 0.005)
    check(s, "FD003 drift CRI", by_subset.loc["drift", "FD003"], 3.34, 0.005)
    r2 = m[m.severity == 5].groupby("corruption").R2.mean()
    check(s, "faults with negative mean R2 at severity 5", int((r2 < 0).sum()), 5, 0)

    s = "4.3 Missingness"
    check(s, "MCAR severity 5, lowest RMSE increase", sev5.loc["missing_mcar"].min(), 1.14, 0.005)
    check(s, "MCAR severity 5, highest RMSE increase", sev5.loc["missing_mcar"].max(), 1.27, 0.005)
    check(s, "MNAR severity 1 missed-alarm rate", miss["missing_mnar", 1], 0.440, 0.0006)
    check(s, "MCAR severity 5 missed-alarm rate", miss["missing_mcar", 5], 0.137, 0.0006)
    check(s, "MAR severity 5 missed-alarm rate", miss["missing_mar", 5], 0.857, 0.0006)
    check(s, "MNAR s2 exceeds MCAR s5 on CRI", float(cell["missing_mnar", 2] > cell["missing_mcar", 5]), 1, 0)
    check(s, "MNAR s2 exceeds MCAR s5 on missed alarms", float(miss["missing_mnar", 2] > miss["missing_mcar", 5]), 1, 0)
    worst = m[(m.corruption == "missing_mnar") & (m.severity >= 3)].groupby(["severity", "subset", "model"]).MissRate.mean().min()
    check(s, "MNAR severity 3+, lowest missed-alarm rate", worst, 0.978, 0.0006)
    check(s, "MNAR severity 3 false-alarm rate", far["missing_mnar", 3], 0.0, 0.0005)
    score = (m.Score / m.Score_c)[(m.corruption == "missing_mnar") & (m.severity == 3)].median()
    check(s, "MNAR severity 3, median PHM08 score ratio", score, 128, 0.6)

    s = "4.4 Coverage"
    for k, sv, v in [("drift", 5, 0.059), ("drift", 3, 0.38), ("concept_drift", 5, 0.28), ("bias_offset", 5, 0.46),
                     ("missing_mnar", 5, 0.63), ("gaussian_noise", 5, 0.73), ("sampling_reduction", 5, 0.81),
                     ("missing_mcar", 5, 0.89), ("channel_desync", 5, 0.92), ("timestamp_jitter", 5, 0.94)]:
        check(s, f"coverage, {dqtb.NAME[k]} severity {sv}", picp[k, sv], v, 0.006)
    check(s, "Spearman correlation, coverage vs CRI (60 cells)", spearmanr(cell.sort_index(), picp.sort_index())[0], -0.98, 0.005)
    check(s, "width ratio, drift severity 5", width["drift", 5], 0.24, 0.005)
    check(s, "width ratio, MNAR severity 5", width["missing_mnar", 5], 1.43, 0.005)
    check(s, "width ratio, sensor dropout severity 5", width["sensor_dropout", 5], 1.20, 0.005)

    s = "4.5 Decisions"
    for k, sv, v in [("missing_mnar", 5, 0.999), ("sensor_dropout", 5, 0.804), ("missing_mnar", 3, 0.993)]:
        check(s, f"missed-alarm rate, {dqtb.NAME[k]} severity {sv}", miss[k, sv], v, 0.0006)
    for k, sv, v in [("drift", 5, 0.808), ("concept_drift", 5, 0.394), ("bias_offset", 5, 0.230)]:
        check(s, f"false-alarm rate, {dqtb.NAME[k]} severity {sv}", far[k, sv], v, 0.0006)
    for k, sv, v in [("missing_mnar", 5, 11.5), ("missing_mar", 5, 9.8), ("sensor_dropout", 5, 9.4), ("drift", 5, 6.4),
                     ("drift", 1, 0.65), ("drift", 2, 0.65), ("drift", 3, 1.4), ("missing_mnar", 3, 11.4)]:
        check(s, f"cost ratio, {dqtb.NAME[k]} severity {sv}", cost[k, sv], v, 0.06)
    check(s, "CRI, drift severity 3", cell["drift", 3], 2.58, 0.005)
    check(s, "CRI, MNAR severity 3", cell["missing_mnar", 3], 2.55, 0.005)
    clean_cost = base.groupby("subset").MaintCost.mean()
    sub_cost = m[m.severity == 5].groupby(["corruption", "subset"]).MaintCost.mean().unstack() / clean_cost
    check(s, "cost ratio, MNAR severity 5, FD001", sub_cost.loc["missing_mnar", "FD001"], 7.6, 0.06)
    check(s, "cost ratio, MNAR severity 5, FD002", sub_cost.loc["missing_mnar", "FD002"], 23.1, 0.06)
    check(s, "cost ratio, drift severity 5, FD001", sub_cost.loc["drift", "FD001"], 3.7, 0.06)
    check(s, "cost ratio, drift severity 5, FD002", sub_cost.loc["drift", "FD002"], 11.4, 0.06)

    s = "4.6 Architectures"
    check(s, "mean CRI, Transformer", pooled.Transformer.mean(), 1.49, 0.005)
    check(s, "mean CRI, LSTM", pooled.LSTM.mean(), 1.54, 0.005)
    check(s, "mean CRI, CNN", pooled.CNN.mean(), 1.60, 0.005)
    best = sweep[sweep.severity == 3].groupby(["subset", "corruption", "model"]).RMSE.mean().unstack().idxmin(axis=1).value_counts()
    check(s, "severity-3 conditions won by LSTM (of 48)", best["LSTM"], 33, 0)
    check(s, "severity-3 conditions won by CNN", best["CNN"], 14, 0)
    check(s, "severity-3 conditions won by Transformer", best["Transformer"], 1, 0)
    tests = dqtb.wilcoxon_tests(sweep, severity=3)
    check(s, "Wilcoxon tests", len(tests), 144, 0)
    check(s, "significant, uncorrected", int((tests.p_value < 0.05).sum()), 109, 0)
    check(s, "significant after Holm", int((tests.p_holm < 0.05).sum()), 99, 0)
    check(s, "Holm-significant tests with LSTM lower", int((tests[tests.p_holm < 0.05].lower_rmse == "LSTM").sum()), 59, 0)
    released = pd.read_csv(dqtb.RESULTS / "significance_sev3_v2.csv")
    both = tests.merge(released, on=["subset", "corruption", "pair"], suffixes=("", "_released"))
    check(s, "largest gap to released p-values", (both.p_value - both.p_value_released).abs().max(), 0.0, 1e-9)

    s = "4.7 Timing and channel identity"
    check(s, "jitter CRI, max", pooled.loc["timestamp_jitter"].max(), 1.01, 0.005)
    check(s, "desynchronisation CRI, max", pooled.loc["channel_desync"].max(), 1.06, 0.005)
    check(s, "desynchronisation coverage, min", picp["channel_desync"].min(), 0.916, 0.001)
    cv = dqtb.injection_seed_cv(sweep, severity=3).groupby("corruption").cv.mean()
    check(s, "injection-seed CV, sensor dropout", cv["sensor_dropout"], 0.156, 0.0006)
    check(s, "injection-seed CV, calibration bias", cv["bias_offset"], 0.128, 0.0006)
    check(s, "injection-seed CV, largest of the other ten", cv.drop(["sensor_dropout", "bias_offset"]).max(), 0.047, 0.003)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quiet", action="store_true", help="print failures and the summary only")
    args = ap.parse_args()
    run()
    section = None
    for sec, name, value, expected, ok in CHECKS:
        if args.quiet and ok:
            continue
        if sec != section:
            print(f"\n{sec}")
            section = sec
        print(f"  {'ok  ' if ok else 'FAIL'}  {name:<52} {value:>12.4f}   paper: {expected:g}")
    failed = sum(not c[4] for c in CHECKS)
    print(f"\n{len(CHECKS) - failed} of {len(CHECKS)} checks passed.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
