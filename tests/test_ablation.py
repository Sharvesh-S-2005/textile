"""Tests for ml/ablation.py."""

import numpy as np
import pandas as pd
import pytest

from ml.ablation import (ENVIRONMENTAL, FEATURE_SETS, HISTORY, MACHINE_HISTORY, MACHINE_WORKER_NL, NL_FEATURE_SETS,
                         NS_FEATURE_SETS, SCHEDULE, WORKER, add_history_features, check_nl_sets, target_derived,
                         block_groups, bootstrap_interval, check_feature_sets, tune_per_set, verdict)
from ml.features import feature_columns
from ml.forecast import MIN_MS

T0 = 1_792_540_800_000


# ---------------------------------------------------------------- feature sets


def test_feature_sets_are_strictly_nested() -> None:
    m1, m2, m3 = (set(FEATURE_SETS[k]) for k in ("model_1", "model_2", "model_3"))
    assert m1 < m2 < m3
    assert m2 - m1 == set(WORKER)
    assert m3 - m2 == set(MACHINE_HISTORY)
    check_feature_sets()


def test_model_1_holds_no_worker_or_machine_column() -> None:
    forbidden = {"workers_present_count", "shift_code", "is_break", "machines_running_count", "mean_current_a",
                 "max_vib_rms_g"} | set(HISTORY)
    assert not forbidden & set(ENVIRONMENTAL)


def test_model_3_uses_every_feature_table_column() -> None:
    assert set(FEATURE_SETS["model_3"]) == set(feature_columns()) | set(HISTORY)


def test_no_lag_sets_contain_no_target_derived_column() -> None:
    """No value, lag, rolling statistic or history of noise, PM2.5 or WBGT in any NL/NS set."""
    for sets in (NL_FEATURE_SETS, NS_FEATURE_SETS):
        for name, cols in sets.items():
            assert not [c for c in cols if target_derived(c)], name
            assert not set(cols) & set(HISTORY), name
    # the guard itself recognises every target-derived column of the feature table
    derived = [c for c in feature_columns() if c.startswith(("noise_dba", "pm25_ugm3", "wbgt_c"))]
    assert len(derived) == 45 and all(target_derived(c) for c in derived)


def test_no_lag_sets_are_strictly_nested_and_differ_only_by_machine_and_worker() -> None:
    for sets, lo, hi in ((NL_FEATURE_SETS, "model_1_nl", "model_3_nl"), (NS_FEATURE_SETS, "model_1_ns", "model_3_ns")):
        assert set(sets[lo]) < set(sets[hi])
        assert set(sets[hi]) - set(sets[lo]) == set(MACHINE_WORKER_NL)
        check_nl_sets(sets)
    assert set(NS_FEATURE_SETS["model_1_ns"]) == set(NL_FEATURE_SETS["model_1_nl"]) - set(SCHEDULE)
    assert "zone_code" in NL_FEATURE_SETS["model_1_nl"] and "zone_code" in NS_FEATURE_SETS["model_1_ns"]


def test_nl_check_rejects_a_lagged_column() -> None:
    with pytest.raises(ValueError):
        check_nl_sets({"model_1_nl": ["temp_c"], "model_3_nl": ["temp_c", "noise_dba_lag5"]})


def test_check_rejects_a_set_that_is_not_nested() -> None:
    bad = {"model_1": ["a", "b"], "model_2": ["a", "c"], "model_3": ["a", "b", "c"]}
    with pytest.raises(ValueError):
        check_feature_sets(bad)


# ---------------------------------------------------------------- history features


def test_history_features_use_only_the_past_and_energy_average_noise() -> None:
    n = 20
    df = pd.DataFrame({"zone_id": "Z1", "t_ms": T0 + np.arange(n) * MIN_MS,
                       "noise_dba": [80.0] * 10 + [100.0] * 10, "pm25_ugm3": np.arange(n, dtype=float),
                       "machines_running_count": [0.0] * 10 + [2.0] * 10})
    out = add_history_features(df, long_min=4, machine_min=4, min_coverage=0.5)
    # Window of 4 ending at row 11: 80, 80, 100, 100 -> 10 log10(mean(10^8, 10^8, 10^10, 10^10)) = 97.03
    assert out.loc[11, "noise_dba_laeq_long"] == pytest.approx(10 * np.log10((2e8 + 2e10) / 4))
    assert out.loc[11, "pm25_ugm3_rmean_long"] == pytest.approx((8 + 9 + 10 + 11) / 4)
    assert out.loc[11, "machines_running_frac"] == pytest.approx(0.5)
    # Changing the future does not change the value at row 11.
    later = df.copy()
    later.loc[12:, "noise_dba"] = 40.0
    assert add_history_features(later, 4, 4, 0.5).loc[11, "noise_dba_laeq_long"] == out.loc[11, "noise_dba_laeq_long"]
    assert np.isnan(out.loc[0, "noise_dba_laeq_long"])          # 1 of 4 minutes < 50 % coverage


# ---------------------------------------------------------------- tuning runs independently per set


def test_each_set_is_tuned_independently_on_its_own_columns() -> None:
    rng = np.random.default_rng(0)
    n = 600
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=["a", "b", "c", "d"])
    y = (X["a"] + 2 * X["d"] + rng.normal(0, 0.1, n)).to_numpy()
    sets = {"model_1": ["a"], "model_2": ["a", "b"], "model_3": ["a", "b", "c", "d"]}
    grid = {"n_estimators": [50], "max_depth": [2, 4], "learning_rate": [0.1, 0.3]}
    out = tune_per_set("regression", sets, grid, X[:400], y[:400], X[400:], y[400:], seed=0, early_stopping=10)
    assert list(out) == list(sets)
    for name, cols in sets.items():
        assert out[name]["features"] == cols
        assert out[name]["model"].n_features_in_ == len(cols)        # fitted on exactly its own columns
        assert len(out[name]["search"]) == 4                         # the full grid, searched for this set
    # Model 3 sees the informative column d, so its validation error must be lower.
    best = {k: min(s["val_mae"] for s in v["search"]) for k, v in out.items()}
    assert best["model_3"] < best["model_1"]


# ---------------------------------------------------------------- bootstrap


def test_bootstrap_is_wider_on_a_smaller_sample() -> None:
    rng = np.random.default_rng(3)
    diff = rng.normal(0.1, 1.0, 4000)
    stat = lambda d: (lambda i: float(np.mean(d[i])))              # noqa: E731
    big = bootstrap_interval(np.arange(4000) // 10, stat(diff), 400, seed=1, ci=0.95)
    small = bootstrap_interval(np.arange(400) // 10, stat(diff[:400]), 400, seed=1, ci=0.95)
    assert (small["hi"] - small["lo"]) > (big["hi"] - big["lo"])
    assert big["lo"] < big["point"] < big["hi"]


def test_block_groups_split_by_zone_and_block() -> None:
    zone = np.array(["Z1", "Z1", "Z1", "Z2"])
    t = np.array([0, 59, 60, 0]) * MIN_MS
    g = block_groups(zone, t, block_min=60)
    assert g[0] == g[1] != g[2] and g[3] not in (g[0], g[2])


def test_verdict_labels_intervals_containing_zero_as_not_distinguishable() -> None:
    assert verdict({"lo": 0.01, "hi": 0.05}) == "better"
    assert verdict({"lo": -0.05, "hi": -0.01}) == "worse"
    assert verdict({"lo": -0.01, "hi": 0.05}) == "not distinguishable"
    assert verdict({"lo": None, "hi": None}) == "undefined"
