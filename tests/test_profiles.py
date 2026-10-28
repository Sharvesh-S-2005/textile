"""Tests for simulator/profiles.py (zone hazard model)."""

import copy
import math
from datetime import datetime, timedelta, timezone

import numpy as np

import config
from simulator.profiles import ZoneProfile

START = datetime(2026, 10, 28, 0, 30, tzinfo=timezone.utc)  # 06:00 Asia/Kolkata
DT = timedelta(seconds=config.get_simulation_config()["env_interval_s"])


def _profile(zone_id: str, seed: int = 42, **dynamics_overrides: object) -> ZoneProfile:
    sim = config.get_simulation_config()
    dynamics = copy.deepcopy(sim["zone_dynamics"])
    dynamics.update(dynamics_overrides)
    return ZoneProfile(
        zone_cfg=config.get_zone(zone_id),
        dynamics=dynamics,
        valid_ranges=config.get_validation_config()["fields"],
        n_machines=sum(1 for m in config.get_machines() if m["zone_id"] == zone_id),
        timezone=config.get_shifts()["timezone"],
        default_dt_s=DT.total_seconds(),
        rng=np.random.default_rng(seed),
        start_ts=START,
    )


def _series(profile: ZoneProfile, n: int, factors: list[float], start: datetime = START) -> list[dict]:
    return [profile.reading(start + i * DT, factors) for i in range(n)]


def _energy_mean(levels: list[float]) -> float:
    return 10 * math.log10(np.mean([10 ** (x / 10) for x in levels]))


def _lag1(values: list[float | None]) -> float:
    pairs = [(a, b) for a, b in zip(values, values[1:]) if a is not None and b is not None]
    a, b = np.array(pairs).T
    return float(np.corrcoef(a, b)[0, 1])


def test_readings_within_contract_ranges() -> None:
    ranges = config.get_validation_config()["fields"]
    rng = np.random.default_rng(0)
    for zone in config.get_zones():
        profile = _profile(zone["zone_id"])
        n_machines = sum(1 for m in config.get_machines() if m["zone_id"] == zone["zone_id"])
        for i in range(1500):
            running = [1.0] * int(rng.integers(0, n_machines + 1))
            r = profile.reading(START + i * DT, running)
            for field, value in r.items():
                if field == "sensor_flags" or value is None:
                    continue
                assert ranges[field]["min"] <= value <= ranges[field]["max"], (field, value)


def test_consecutive_readings_are_autocorrelated() -> None:
    rows = _series(_profile("Z5"), 2000, [1.0, 1.0])
    assert _lag1([r["noise_dba"] for r in rows]) > 0.4
    assert _lag1([r["pm25_ugm3"] for r in rows]) > 0.9
    assert _lag1([r["temp_c"] for r in rows]) > 0.95
    assert _lag1([r["co2_ppm"] for r in rows]) > 0.95


def test_noise_drops_to_background_when_machines_stop() -> None:
    on = [r["noise_dba"] for r in _series(_profile("Z5"), 500, [1.0, 1.0])]
    off = [r["noise_dba"] for r in _series(_profile("Z5"), 500, [])]
    on_leq = _energy_mean([x for x in on if x is not None])
    off_leq = _energy_mean([x for x in off if x is not None])
    assert 95.0 <= on_leq <= 103.0
    assert 55.0 <= off_leq <= 66.0
    assert on_leq - off_leq > 25.0


def test_pm_decays_toward_floor_after_shutdown() -> None:
    profile = _profile("Z2")
    running = _series(profile, 360, [1.0])                       # 1 h running
    stopped = _series(profile, 360, [], start=START + 360 * DT)  # 1 h stopped
    pm_on = np.nanmean([r["pm25_ugm3"] or np.nan for r in running[-60:]])
    pm_off = np.nanmean([r["pm25_ugm3"] or np.nan for r in stopped[-60:]])
    assert pm_off < 0.3 * pm_on


def test_degrading_machine_raises_zone_noise() -> None:
    nominal = _series(_profile("Z5", seed=1), 300, [1.0, 1.0])
    degraded = _series(_profile("Z5", seed=1), 300, [1.0, 2.5])
    n_leq = _energy_mean([r["noise_dba"] for r in nominal if r["noise_dba"] is not None])
    d_leq = _energy_mean([r["noise_dba"] for r in degraded if r["noise_dba"] is not None])
    assert d_leq - n_leq > 1.5


def test_same_seed_reproduces_sequence_exactly() -> None:
    a = _series(_profile("Z4", seed=7), 300, [1.0, 1.0])
    b = _series(_profile("Z4", seed=7), 300, [1.0, 1.0])
    c = _series(_profile("Z4", seed=8), 300, [1.0, 1.0])
    assert a == b
    assert a != c


def test_sensor_fault_emits_null_with_flag() -> None:
    faults = {"rate_per_sensor_per_h": 50.0, "duration_min": {"min": 2, "max": 5},
              "kinds": ["stale", "error"]}
    rows = _series(_profile("Z1", sensor_faults=faults), 500, [1.0])
    faulted = [r for r in rows if r["sensor_flags"]["noise"] in ("stale", "error")]
    assert faulted
    for r in faulted:
        assert r["noise_dba"] is None and r["noise_peak_dba"] is None
    for r in rows:
        if r["sensor_flags"]["noise"] == "ok":
            assert r["noise_dba"] is not None


def test_pm_sensor_drifts_then_needs_clean_then_resets() -> None:
    drift = {"gain_per_day": 0.015, "needs_clean_gain": 1.25, "clean_after_h": 8,
             "initial_days_since_clean_max": 0.0}
    no_faults = {"rate_per_sensor_per_h": 0.0, "duration_min": {"min": 2, "max": 5},
                 "kinds": ["stale"]}
    profile = _profile("Z6", pm_drift=drift, sensor_faults=no_faults)
    step = timedelta(minutes=10)
    flags = [profile.reading(START + i * step, [1.0])["sensor_flags"]["pm"]
             for i in range(6 * 24 * 25)]  # 25 simulated days
    first = flags.index("needs_clean")
    days_to_flag = first * step.total_seconds() / 86400
    assert 16.0 <= days_to_flag <= 17.5          # (1.25 - 1) / 0.015 = 16.7 days
    assert "ok" in flags[first:]                 # cleaned afterwards
    # Flagged from t0 until cleaned at exactly t0 + 8 h: 8 h / 10 min = 48 readings.
    assert flags[first:].count("needs_clean") == 8 * 6
