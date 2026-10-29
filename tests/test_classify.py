"""Tests for ml/classify_exceedance.py."""

import numpy as np
import pandas as pd
import pytest

from ml.classify_exceedance import (choose_threshold, classification_metrics, expand_grid, future_value, make_labels,
                                    persistence_scores, prevalence)
from ml.forecast import MIN_MS

T0 = 1_792_540_800_000


def zone_frame(levels: dict[str, list[float]]) -> pd.DataFrame:
    """One row per zone per minute, in zone order (like the feature table)."""
    rows = [{"zone_id": z, "t_ms": T0 + i * MIN_MS, "noise_dba": v} for z, vs in levels.items()
            for i, v in enumerate(vs)]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- labels come only from the future


def test_future_value_reads_t_plus_h_in_the_same_zone_only() -> None:
    df = zone_frame({"Z1": [60, 61, 62, 63, 64], "Z2": [90, 91, 92, 93, 94]})
    fut = future_value(df, "noise_dba", 2)
    np.testing.assert_array_equal(fut[:3], [62, 63, 64])
    assert np.isnan(fut[3:5]).all()                         # Z1 has no t+2 for its last two rows
    np.testing.assert_array_equal(fut[5:8], [92, 93, 94])  # Z2 never reads Z1's values


def test_above_label_does_not_depend_on_data_at_or_before_t() -> None:
    """Perturbing every value at or before t leaves the label at t unchanged;
    perturbing the value at t+h changes it."""
    h, t = 3, 4
    base = [60.0, 70, 80, 85, 88, 89, 91, 95, 85, 80]
    df = zone_frame({"Z1": base})
    y0, _ = make_labels(df["noise_dba"].to_numpy(), future_value(df, "noise_dba", h), 90.0, "above")

    past = list(base)
    past[: t + 1] = [120.0] * (t + 1)                       # everything at or before t now far above 90
    dfp = zone_frame({"Z1": past})
    yp, _ = make_labels(dfp["noise_dba"].to_numpy(), future_value(dfp, "noise_dba", h), 90.0, "above")
    assert yp[t] == y0[t] == 1                              # base[t+3] = 95

    fut = list(base)
    fut[t + h] = 50.0
    dff = zone_frame({"Z1": fut})
    yf, _ = make_labels(dff["noise_dba"].to_numpy(), future_value(dff, "noise_dba", h), 90.0, "above")
    assert yf[t] == 0


def test_crossing_uses_now_only_to_select_the_population() -> None:
    now = np.array([85.0, 92.0, 85.0, np.nan, 89.9])
    fut = np.array([91.0, 95.0, 80.0, 95.0, np.nan])
    y, pop = make_labels(now, fut, 90.0, "crossing")
    assert pop.tolist() == [True, False, True, False, False]   # above now / unknown now / unknown future excluded
    assert y.tolist() == [1, 0, 0, 0, 0]
    y_a, pop_a = make_labels(now, fut, 90.0, "above")
    assert pop_a.tolist() == [True, True, True, True, False]
    assert y_a.tolist() == [1, 1, 0, 1, 0]
    with pytest.raises(ValueError):
        make_labels(now, fut, 90.0, "sideways")


def test_exactly_at_the_criterion_counts_as_exceeding() -> None:
    y, _ = make_labels(np.array([80.0]), np.array([90.0]), 90.0, "above")
    assert y.tolist() == [1]


# ---------------------------------------------------------------- prevalence and persistence


def test_prevalence_on_a_fixture() -> None:
    """Labels 1,0,0,1,1,0,0,0 -> 3 / 8 = 0.375."""
    assert prevalence(np.array([1, 0, 0, 1, 1, 0, 0, 0])) == pytest.approx(0.375)
    assert prevalence(np.array([], dtype=int)) is None


def test_persistence_classifier_f1_by_hand() -> None:
    """Levels at t:   95 85 92 80 91 70 ;  above now -> predictions 1 0 1 0 1 0
    Labels at t+h:    1  1  0  0  1  0
    TP = 2 (rows 0, 4), FP = 1 (row 2), FN = 1 (row 1), TN = 2.
    Precision 2/3, recall 2/3, F1 = 2*2 / (2*2 + 1 + 1) = 4/6 = 0.6667; accuracy 4/6."""
    now = np.array([95.0, 85, 92, 80, 91, 70])
    y = np.array([1, 1, 0, 0, 1, 0])
    m = classification_metrics(y, persistence_scores(now, 90.0), 0.5)
    assert m["confusion"] == {"tn": 2, "fp": 1, "fn": 1, "tp": 2}
    assert m["f1"] == pytest.approx(4 / 6)
    assert m["precision"] == pytest.approx(2 / 3) and m["recall"] == pytest.approx(2 / 3)
    assert m["accuracy"] == pytest.approx(4 / 6) and m["prevalence"] == pytest.approx(0.5)
    assert m["fpr"] == pytest.approx(1 / 3)


def test_metrics_are_undefined_not_invented_for_a_single_class() -> None:
    m = classification_metrics(np.zeros(5, dtype=int), np.linspace(0, 1, 5), 0.5)
    assert m["pr_auc"] is None and m["roc_auc"] is None and m["recall"] is None


# ---------------------------------------------------------------- threshold selection


def test_threshold_maximises_recall_within_the_false_alarm_ceiling() -> None:
    rng = np.random.default_rng(1)
    y = np.r_[np.ones(200, dtype=int), np.zeros(800, dtype=int)]
    scores = np.r_[rng.normal(1.0, 1.0, 200), rng.normal(0.0, 1.0, 800)]
    for ceiling in (0.01, 0.05, 0.10, 0.30):
        thr = choose_threshold(y, scores, ceiling)
        m = classification_metrics(y, scores, thr)
        assert m["fpr"] <= ceiling
        # no lower threshold gains recall while staying within the ceiling
        for lower in np.sort(scores[scores < thr])[::-1][:50]:
            m_low = classification_metrics(y, scores, lower)
            assert m_low["fpr"] > ceiling or m_low["recall"] <= m["recall"]


def test_threshold_by_hand() -> None:
    """Scores 0.9(1) 0.8(0) 0.7(1) 0.4(0) 0.3(1) 0.1(0). Negatives: 3.
    Ceiling 1/3 allows one false positive: threshold 0.7 -> TP 2, FP 1, recall 2/3.
    Threshold 0.3 would give FP 2 (FPR 2/3), over the ceiling."""
    y = np.array([1, 0, 1, 0, 1, 0])
    s = np.array([0.9, 0.8, 0.7, 0.4, 0.3, 0.1])
    assert choose_threshold(y, s, 1 / 3) == pytest.approx(0.7)
    assert choose_threshold(y, s, 0.0) == pytest.approx(0.9)       # no false alarms allowed


def test_threshold_needs_both_classes() -> None:
    with pytest.raises(ValueError):
        choose_threshold(np.zeros(4, dtype=int), np.arange(4.0), 0.1)


def test_expand_grid_covers_every_combination() -> None:
    g = expand_grid({"a": [1, 2], "b": [None, "x"], "c": [0.5]})
    assert len(g) == 4 and {"a": 2, "b": "x", "c": 0.5} in g
