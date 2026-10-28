"""Tests for ml/forecast.py."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from ml.forecast import (DAY_MS, MIN_MS, chronological_splits, crossing_events, lead_time_analysis, load_model,
                         regression_metrics, save_model, schedule_mask, skill_score, split_mask, summarise_lead)

D0 = 1_792_540_800_000          # 2026-10-21T00:00:00Z
DAYS = {"train": 5, "validation": 1, "test": 1}
FALLBACK = {"train": 0.6, "validation": 0.2, "test": 0.2}


def minutes(n_days: float, start: int = D0) -> pd.Series:
    return pd.Series(np.arange(start, start + int(n_days * DAY_MS), MIN_MS))


# ---------------------------------------------------------------- split

def test_split_is_chronological_and_non_overlapping() -> None:
    t = minutes(7.2)                                   # 7 whole days + a partial one
    splits, rule = chronological_splits(t, DAYS, FALLBACK)
    names = [s.name for s in splits]
    assert names == ["train", "validation", "test"]
    assert splits[0].start_ms == D0
    assert splits[0].end_ms == splits[1].start_ms == D0 + 5 * DAY_MS
    assert splits[1].end_ms == splits[2].start_ms == D0 + 6 * DAY_MS
    assert splits[2].end_ms == D0 + 7 * DAY_MS          # partial 8th day left out
    assert "partial" in rule
    masks = [(t >= s.start_ms) & (t < s.end_ms) for s in splits]
    assert not ((masks[0] & masks[1]) | (masks[1] & masks[2]) | (masks[0] & masks[2])).any()
    assert t[masks[0]].max() < t[masks[1]].min() < t[masks[2]].min()


def test_split_falls_back_to_60_20_20_with_fewer_days() -> None:
    t = minutes(3)
    splits, rule = chronological_splits(t, DAYS, FALLBACK)
    assert "60%/20%/20%" in rule
    span = 3 * DAY_MS
    assert splits[0].end_ms - splits[0].start_ms == int(span * 0.6)
    assert splits[0].end_ms <= splits[1].start_ms and splits[1].end_ms <= splits[2].start_ms


def test_boundary_purge_drops_rows_whose_target_crosses_the_split_end() -> None:
    t = minutes(7)
    train = chronological_splits(t, DAYS, FALLBACK)[0][0]
    m = split_mask(t, train, horizon_min=60)
    assert (t[m] + 60 * MIN_MS < train.end_ms).all()
    assert m.sum() == 5 * 1440 - 60                     # the last 60 minutes of day 5 are purged


# ---------------------------------------------------------------- baselines and skill

def test_persistence_on_a_fixture_with_known_answer() -> None:
    """Series 10, 12, 15, 11 (one per minute). Persistence at h=1 predicts 10, 12, 15
    for targets 12, 15, 11: errors 2, 3, 4 -> MAE 3.0."""
    y = pd.Series([10.0, 12.0, 15.0, 11.0])
    target, persistence = y.shift(-1)[:-1].to_numpy(), y[:-1].to_numpy()
    m = regression_metrics(target, persistence, mae_persistence=3.0)
    assert m["mae"] == pytest.approx(3.0) and m["skill_vs_persistence"] == pytest.approx(0.0)


def test_skill_is_zero_when_equal_to_persistence_and_positive_when_better() -> None:
    assert skill_score(3.0, 3.0) == 0.0
    assert skill_score(1.5, 3.0) == pytest.approx(0.5)
    assert skill_score(4.5, 3.0) == pytest.approx(-0.5)     # worse than persistence is negative
    with pytest.raises(ValueError):
        skill_score(1.0, 0.0)


def test_model_round_trips_through_save_and_load(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(200, 4)), columns=list("abcd"))
    y = X["a"] * 2 + rng.normal(size=200) * 0.1
    model = RandomForestRegressor(n_estimators=10, random_state=0).fit(X, y)
    path = tmp_path / "forecast_noise_dba_30.pkl"
    save_model(model, path, {"target": "noise_dba", "horizon_min": 30, "features": list("abcd")})
    loaded, meta = load_model(path)
    assert meta["horizon_min"] == 30 and meta["features"] == list("abcd")
    np.testing.assert_array_equal(loaded.predict(X), model.predict(X))


# ---------------------------------------------------------------- exceedance lead time

def test_crossing_events_require_a_sustained_change() -> None:
    quiet, loud = [60.0] * 10, [95.0] * 10
    flicker = [60.0] * 10 + [95.0, 60.0, 95.0] + [60.0] * 10
    assert crossing_events(pd.Series(quiet + loud), 90.0, sustain=5) == [10]
    assert crossing_events(pd.Series(flicker), 90.0, sustain=5) == []        # not sustained


def test_lead_time_hit_and_false_alarm_by_hand() -> None:
    """Event at minute 40. Forecast >= 90 from minute 25 onward (and once at minute 5).
    With h = 30: warnings while still below 90 at minutes 5 and 25..39. The first
    warning inside [10, 40) is 25, so lead = 15 min (hit). The warning at minute 5 has
    no event within (5, 35] -> 1 false alarm out of 16 warnings."""
    measured = pd.Series([60.0] * 40 + [95.0] * 40)
    forecast = pd.Series([60.0] * 80)
    forecast[5] = 92.0
    forecast[25:] = 93.0
    r = lead_time_analysis(measured, forecast, 90.0, horizon_min=30, sustain=5)
    assert r["events"] == 1 and r["hits"] == 1 and r["lead_times_min"] == [15]
    assert r["warnings"] == 16 and r["false_alarms"] == 1
    s = summarise_lead([r, {"events": 1, "hits": 0, "lead_times_min": [], "warnings": 0, "false_alarms": 0}])
    assert s["hit_rate"] == 0.5 and s["median_lead_min"] == 15.0
    assert s["false_alarm_ratio"] == pytest.approx(1 / 16)


def test_persistence_never_warns_ahead() -> None:
    measured = pd.Series([60.0] * 40 + [95.0] * 40)
    r = lead_time_analysis(measured, measured, 90.0, horizon_min=30)
    assert r["events"] == 1 and r["hits"] == 0 and r["warnings"] == 0


# ---------------------------------------------------------------- schedule subsets

def test_schedule_mask_marks_horizons_that_span_a_break() -> None:
    t = np.arange(0, 100) * MIN_MS
    sched = pd.DataFrame({"zone_id": "Z1", "t_ms": t, "is_break": (t >= 50 * MIN_MS) & (t < 60 * MIN_MS),
                          "shift_code": 0}).set_index(["zone_id", "t_ms"])
    pred = pd.DataFrame({"zone_id": "Z1", "t_ms": [10 * MIN_MS, 45 * MIN_MS, 61 * MIN_MS]})
    # h = 10: 10 -> 20 steady; 45 -> 55 enters the break; 61 -> 71 steady (after the break).
    assert schedule_mask(pred, sched, 10).tolist() == [True, False, True]
