"""Integrity tests for the released results.

Run with:  pytest -q
"""
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))

import dqtb  # noqa: E402
import verify_claims  # noqa: E402


@pytest.fixture(scope="module")
def data():
    return dqtb.load_all()


def test_design_is_complete(data):
    sweep, base, _ = data
    assert len(sweep) == 36000
    assert len(base) == 60
    keys = ["subset", "model", "train_seed", "corruption", "severity", "inj_seed"]
    assert not sweep.duplicated(keys).any()
    assert set(sweep.subset) == set(dqtb.SUBSETS)
    assert set(sweep.model) == set(dqtb.MODELS)
    assert set(sweep.corruption) == set(dqtb.ORDER)
    assert sorted(sweep.severity.unique()) == [1, 2, 3, 4, 5]
    assert sorted(sweep.inj_seed.unique()) == list(range(10))
    assert sorted(sweep.train_seed.unique()) == [42, 43, 44, 45, 46]
    # fully crossed: 50 rows per subset, model, fault and severity
    assert (sweep.groupby(["subset", "model", "corruption", "severity"]).size() == 50).all()


def test_no_missing_values_and_valid_ranges(data):
    sweep, base, _ = data
    for df in (sweep, base):
        assert not df[dqtb.METRICS].isna().any().any()
        assert (df.RMSE > 0).all() and (df.MAE > 0).all() and (df.MPIW > 0).all()
        for col in ("PICP", "FalseAlarmRate", "MissRate"):
            assert df[col].between(0, 1).all()
        assert (df.MaintCost >= 0).all()


def test_configuration_matches_results(data):
    sweep, base, _ = data
    cfg = dqtb.load_config()
    assert sorted(cfg["corruptions"]) == sorted(dqtb.ORDER)
    assert cfg["train_seeds"] == [42, 43, 44, 45, 46]
    assert cfg["window"] == 30 and cfg["rul_cap"] == 125 and cfg["mc_passes"] == 30
    assert len(cfg["training_meta"]) == len(base)
    assert len(cfg["data_sha256"]) == 12


def test_processed_arrays():
    expected = {"FD001": (15332, 2399, 100), "FD002": (39424, 6795, 259), "FD003": (18749, 3071, 100), "FD004": (45594, 8434, 248)}
    for subset, (n_train, n_val, n_test) in expected.items():
        d = np.load(dqtb.PROCESSED / f"{subset}_w30_cap125.npz")
        assert d["Xtr"].shape == (n_train, 30, 14) and d["ytr"].shape == (n_train,)
        assert d["Xva"].shape == (n_val, 30, 14) and d["yva"].shape == (n_val,)
        assert d["Xte"].shape == (n_test, 30, 14) and d["yte"].shape == (n_test,)
        assert d["yte"].max() <= 125 and d["ytr"].max() <= 125
        assert not np.isnan(d["Xte"]).any()


def _tex_rows(name):
    """Parse the body rows of a LaTeX table written by the benchmark run."""
    text = (dqtb.RESULTS / "original_tables" / name).read_text(encoding="utf-8")
    body = text.split("\\midrule")[1].split("\\bottomrule")[0]
    return [[c.strip().replace("\\_", "_") for c in line.rstrip("\\ ").split("&")] for line in body.strip().splitlines()]


def test_robustness_index_matches_original_table(data):
    _, _, m = data
    pooled = dqtb.cri(m, ["corruption"]).groupby(["corruption", "model"]).mean().unstack()
    rows = _tex_rows("table3_robustness_index.tex")
    assert len(rows) == 12
    for fault, lstm, cnn, transformer in rows:
        assert pooled.loc[fault, "LSTM"] == pytest.approx(float(lstm), abs=0.006)
        assert pooled.loc[fault, "CNN"] == pytest.approx(float(cnn), abs=0.006)
        assert pooled.loc[fault, "Transformer"] == pytest.approx(float(transformer), abs=0.006)


def test_clean_accuracy_matches_original_table(data):
    _, base, _ = data
    g = base.groupby(["subset", "model"]).RMSE
    for subset, model, rmse, *_ in _tex_rows("table1a_accuracy_v2.tex"):
        mean, sd = (float(x) for x in re.findall(r"[\d.]+", rmse)[:2])
        assert g.mean()[subset, model] == pytest.approx(mean, abs=0.006)
        assert g.std()[subset, model] == pytest.approx(sd, abs=0.006)


def test_holm_adjustment():
    adj = dqtb.holm([0.01, 0.04, 0.03, 0.005])
    assert adj == pytest.approx([0.03, 0.06, 0.06, 0.02])


def test_numbers_quoted_in_the_paper():
    verify_claims.CHECKS.clear()
    verify_claims.run()
    failed = [c for c in verify_claims.CHECKS if not c[4]]
    assert len(verify_claims.CHECKS) > 100
    assert not failed, failed
