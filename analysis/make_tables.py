"""Rebuild the data tables of the paper from the released result files.

Usage:
    python analysis/make_tables.py

Writes one CSV per table to ``results/tables/`` and a combined, readable
``results/tables/tables.md``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import dqtb


def pm(x: pd.Series, digits: int) -> str:
    return f"{x.mean():.{digits}f} ± {x.std(ddof=1):.{digits}f}"


def build() -> dict[str, tuple[str, pd.DataFrame]]:
    sweep, base, m = dqtb.load_all()
    tables: dict[str, tuple[str, pd.DataFrame]] = {}

    # Table 2. Dataset summary (from the processed arrays and the configuration)
    info = {"FD001": (1, 1, 100, 100), "FD002": (6, 1, 260, 259), "FD003": (1, 2, 100, 100), "FD004": (6, 2, 249, 248)}
    rows = []
    for s in dqtb.SUBSETS:
        d = np.load(dqtb.PROCESSED / f"{s}_w30_cap125.npz")
        rows.append([s, *info[s], d["Xtr"].shape[0], d["Xva"].shape[0], int((d["yte"] <= 30).sum())])
    tables["table02_datasets"] = ("Table 2. C-MAPSS subsets after preprocessing (window 30, RUL cap 125).", pd.DataFrame(
        rows, columns=["Subset", "Operating conditions", "Fault modes", "Training engines", "Test engines",
                       "Training windows", "Validation windows", "Test engines with RUL <= 30"]))

    # Tables 4 and 5. Clean-input performance
    acc, uq = [], []
    for s in dqtb.SUBSETS:
        for mod in dqtb.MODELS:
            g = base[(base.subset == s) & (base.model == mod)]
            acc.append([s, mod, pm(g.RMSE, 2), pm(g.MAE, 2), pm(g.R2, 2), pm(g.Score, 0)])
            uq.append([s, mod, pm(g.PICP, 3), pm(g.MPIW, 1), pm(g.FalseAlarmRate, 3), pm(g.MissRate, 3), pm(g.MaintCost, 0)])
    tables["table04_clean_accuracy"] = ("Table 4. Clean-input accuracy (mean ± SD over five training seeds).",
                                        pd.DataFrame(acc, columns=["Subset", "Model", "RMSE", "MAE", "R2", "PHM08 score"]))
    tables["table05_clean_intervals_decisions"] = (
        "Table 5. Clean-input interval quality and decision metrics (mean ± SD over five training seeds).",
        pd.DataFrame(uq, columns=["Subset", "Model", "PICP", "MPIW", "FAR", "MR", "Cost"]))

    # Table 6. Corruption Robustness Index
    pooled = dqtb.cri(m, ["corruption"]).groupby(["corruption", "model"]).mean().unstack()[dqtb.MODELS]
    sev5 = dqtb.cri(m[m.severity == 5], ["corruption"]).groupby(["corruption", "model"]).mean().unstack()[dqtb.MODELS]
    t6 = pd.DataFrame({"Fault": [dqtb.NAME[k] for k in dqtb.ORDER]})
    for mod in dqtb.MODELS:
        t6[f"{mod} (all)"] = pooled.loc[dqtb.ORDER, mod].round(2).to_numpy()
    t6["Mean (all)"] = pooled.loc[dqtb.ORDER].mean(axis=1).round(2).to_numpy()
    for mod in dqtb.MODELS:
        t6[f"{mod} (s5)"] = sev5.loc[dqtb.ORDER, mod].round(2).to_numpy()
    t6["Mean (s5)"] = sev5.loc[dqtb.ORDER].mean(axis=1).round(2).to_numpy()
    tables["table06_robustness_index"] = ("Table 6. Corruption Robustness Index (RMSE ratio to clean input).", t6)

    # Cell-level summaries used by several tables
    cs = dqtb.cri(m, ["corruption", "severity"]).groupby(["corruption", "severity"]).mean()
    g = m.groupby(["corruption", "severity"])
    picp, miss, far = g.PICP.mean(), g.MissRate.mean(), g.FalseAlarmRate.mean()
    cost = g.MaintCost.mean() / base.MaintCost.mean()
    width = g.width_ratio.mean()

    # Table 7. Missingness mechanisms
    rows = []
    for s in range(1, 6):
        row = [s]
        for k in ("missing_mcar", "missing_mar", "missing_mnar"):
            row += [round(cs[k, s], 2), round(picp[k, s], 3), round(miss[k, s], 3)]
        rows.append(row)
    cols = ["Severity"] + [f"{mech} {q}" for mech in ("MCAR", "MAR", "MNAR") for q in ("CRI", "PICP", "MR")]
    tables["table07_missingness"] = ("Table 7. Effect of the missingness mechanism by severity.", pd.DataFrame(rows, columns=cols))

    # Table 8. Coverage and interval width
    rows = [["Clean input", round(base.PICP.mean(), 3), round(base.PICP.mean(), 3), round(base.PICP.mean(), 3), 1.0, 1.0, 1.0]]
    for k in dqtb.ORDER:
        rows.append([dqtb.NAME[k]] + [round(picp[k, s], 3) for s in (1, 3, 5)] + [round(width[k, s], 2) for s in (1, 3, 5)])
    tables["table08_coverage_width"] = ("Table 8. Coverage of nominal 95% intervals and interval width ratio.", pd.DataFrame(
        rows, columns=["Fault", "PICP s1", "PICP s3", "PICP s5", "Width ratio s1", "Width ratio s3", "Width ratio s5"]))

    # Table 9. Decision outcomes
    rows = [["Clean input", round(base.FalseAlarmRate.mean(), 3), round(base.MissRate.mean(), 3), 1.0,
             round(base.FalseAlarmRate.mean(), 3), round(base.MissRate.mean(), 3), 1.0]]
    for k in dqtb.ORDER:
        rows.append([dqtb.NAME[k]] + [x for s in (3, 5) for x in (round(far[k, s], 3), round(miss[k, s], 3), round(cost[k, s], 2))])
    tables["table09_decisions"] = ("Table 9. Maintenance decision outcomes at severities 3 and 5.", pd.DataFrame(
        rows, columns=["Fault", "FAR s3", "MR s3", "Cost ratio s3", "FAR s5", "MR s5", "Cost ratio s5"]))

    # Table 10. Wilcoxon tests
    w = dqtb.wilcoxon_tests(sweep, severity=3)
    rows = []
    for pair, gp in w.groupby("pair", sort=False):
        sig = gp[gp.p_holm < 0.05]
        rows.append([pair, int((gp.p_value < 0.05).sum()), len(sig), int((sig.lower_rmse == gp["first"].iloc[0]).sum()),
                     int((sig.lower_rmse == gp["second"].iloc[0]).sum()), len(gp)])
    rows.append(["All pairs", int((w.p_value < 0.05).sum()), int((w.p_holm < 0.05).sum()), "", "", len(w)])
    tables["table10_wilcoxon"] = ("Table 10. Paired Wilcoxon signed-rank tests at severity 3 (alpha = 0.05, Holm over 144 tests).",
                                  pd.DataFrame(rows, columns=["Pair", "Significant (raw)", "Significant (Holm)",
                                                              "First model lower", "Second model lower", "Tests"]))
    tables["wilcoxon_all_tests"] = ("All 144 Wilcoxon tests with raw and Holm-adjusted p-values.",
                                    w[["subset", "corruption", "pair", "n", "p_value", "p_holm", "lower_rmse"]])

    # Table A.1. Robustness index by subset
    by_subset = dqtb.cri(m, ["corruption"]).groupby(["corruption", "subset"]).mean().unstack()[dqtb.SUBSETS]
    ta = by_subset.loc[dqtb.ORDER].round(2).reset_index()
    ta["corruption"] = ta["corruption"].map(dqtb.NAME)
    tables["tableA1_robustness_by_subset"] = ("Table A.1. Corruption Robustness Index by subset.", ta.rename(columns={"corruption": "Fault"}))

    # Full fault x severity grid (source of Fig. 4)
    grid = pd.DataFrame({"CRI": cs, "PICP": picp, "width_ratio": width, "FAR": far, "MR": miss, "cost_ratio": cost}).round(4).reset_index()
    grid.insert(1, "fault", grid["corruption"].map(dqtb.NAME))
    tables["grid_fault_by_severity"] = ("All pooled metrics for each fault and severity (60 cells).", grid)
    return tables


def to_markdown(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in row) + " |")
    return "\n".join(lines)


def main() -> None:
    dqtb.TABLES.mkdir(parents=True, exist_ok=True)
    tables = build()
    md = ["# Tables rebuilt from `results/`", "", "Generated by `analysis/make_tables.py`. Do not edit by hand.", ""]
    for name, (caption, df) in tables.items():
        df.to_csv(dqtb.TABLES / f"{name}.csv", index=False)
        if name not in ("wilcoxon_all_tests", "grid_fault_by_severity"):
            md += [f"## {caption}", "", to_markdown(df), ""]
        print(f"wrote results/tables/{name}.csv  ({len(df)} rows)")
    (dqtb.TABLES / "tables.md").write_text("\n".join(md), encoding="utf-8")
    print("wrote results/tables/tables.md")


if __name__ == "__main__":
    main()
