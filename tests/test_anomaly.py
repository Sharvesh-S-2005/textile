"""Tests for ml/anomaly.py."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import config
from ml.anomaly import (DAY_MS, add_rolling, bin_vibration, cross_correlation, detect, ewma_chart, first_alarm,
                        hourly_energy_mean, peak_lag)

BIN_MS = 10 * 60_000
SETTINGS = {"ewma": {"lambda": 0.1, "limit_sigma": 3.5}, "isolation_forest": {"n_estimators": 100,
            "reference_quantile": 0.99}, "alarm_consecutive_bins": 3}


def synthetic_bins(rms: np.ndarray, seed: int = 0) -> pd.DataFrame:
    """Bins with a crest factor of ~2.6 and the rolling features."""
    rng = np.random.default_rng(seed)
    peak = rms * rng.uniform(2.4, 2.8, len(rms))
    b = pd.DataFrame({"rms": rms, "peak": peak, "n": 60}, index=np.arange(len(rms)) * BIN_MS)
    b["crest"] = b["peak"] / b["rms"]
    return add_rolling(b, 6)


def stationary(n: int, seed: int) -> np.ndarray:
    return 0.40 * (1 + 0.01 * np.random.default_rng(seed).standard_normal(n))


N_REF, N_MON = 144, 720                         # one reference day, five monitored days of 10-min bins


def run_detect(rms: np.ndarray, seed: int = 0) -> dict:
    bins = synthetic_bins(rms, seed)
    return detect(bins, np.arange(len(bins)) < N_REF, SETTINGS, seed=42)


# ---------------------------------------------------------------- the three required cases


@pytest.mark.parametrize("detector", ["ewma", "isolation_forest"])
def test_flags_a_step_change(detector: str) -> None:
    rms = stationary(N_REF + N_MON, 1)
    rms[N_REF + 200:] *= 1.15                   # +15 % at bin 200 of monitoring
    d = run_detect(rms)[detector]
    assert d["alarm_ms"] is not None
    assert (N_REF + 200) * BIN_MS <= d["alarm_ms"] <= (N_REF + 210) * BIN_MS


@pytest.mark.parametrize("detector", ["ewma", "isolation_forest"])
def test_flags_a_ramp(detector: str) -> None:
    rms = stationary(N_REF + N_MON, 2)
    rms[N_REF:] *= 1 + np.linspace(0, 0.20, N_MON)      # slow wear: +20 % over five days
    d = run_detect(rms)[detector]
    assert d["alarm_ms"] is not None
    assert d["alarm_ms"] < (N_REF + N_MON // 2) * BIN_MS  # well before the end


@pytest.mark.parametrize("seed", [3, 4, 5, 6])
def test_ewma_does_not_alarm_on_stationary_noise(seed: int) -> None:
    d = run_detect(stationary(N_REF + N_MON, seed), seed)
    assert d["ewma"]["alarm_ms"] is None
    assert d["ewma"]["flag_rate"] < 0.01


@pytest.mark.parametrize("seed", [3, 4, 5, 6])
def test_isolation_forest_does_not_alarm_on_stationary_noise(seed: int) -> None:
    d = run_detect(stationary(N_REF + N_MON, seed), seed)
    assert d["isolation_forest"]["alarm_ms"] is None
    assert d["isolation_forest"]["flag_rate"] < 0.05     # per-bin rate near the 1 % reference quantile


# ---------------------------------------------------------------- building blocks


def test_ewma_limits_by_hand() -> None:
    """lambda 0.5, L 3, standardised input 2, 2: z1 = 1, z2 = 1.5.
    Half-widths: 3 sqrt(0.5/1.5 (1 - 0.5^2)) = 1.5 and 3 sqrt(1/3 (1 - 0.5^4)) = 1.6771."""
    c = ewma_chart(np.array([12.0, 12.0]), mu0=10.0, sigma0=1.0, lam=0.5, limit_sigma=3.0)
    np.testing.assert_allclose(c["z"], [1.0, 1.5])
    np.testing.assert_allclose(c["limit"], [1.5, 3 * np.sqrt((1 / 3) * (1 - 0.5 ** 4))])
    assert c["flag"].tolist() == [False, False]


def test_first_alarm_needs_consecutive_flags() -> None:
    assert first_alarm(np.array([1, 1, 0, 1, 1, 1, 0], dtype=bool), 3) == 5
    assert first_alarm(np.array([1, 0, 1, 0], dtype=bool), 2) is None


def test_binning_keeps_only_running_usable_readings() -> None:
    ts = np.arange(0, 1200_000, 10_000)                  # two 10-min bins of 10 s readings
    r = pd.DataFrame({"ts_ms": ts, "running": 1, "vib_rms_g": 0.5, "vib_peak_g": 1.3, "usable": True})
    r.loc[:59, "running"] = 0                            # first bin: machine off for all of it
    r.loc[60:65, "usable"] = False
    r.loc[60:65, "vib_rms_g"] = 9.0                      # a flagged spike must not reach the mean
    b = bin_vibration(r, 10, 0.5)
    assert b.index.tolist() == [600_000]
    assert b.loc[600_000, "rms"] == pytest.approx(0.5) and b.loc[600_000, "n"] == 54


def test_cross_correlation_finds_a_known_lead() -> None:
    rng = np.random.default_rng(0)
    a = pd.Series(rng.standard_normal(300))
    b = a.shift(4) + 0.1 * rng.standard_normal(300)      # b follows a by 4 steps
    k, r = peak_lag(cross_correlation(a, b, 8))
    assert k == 4 and r > 0.9


def test_hourly_energy_mean_is_logarithmic() -> None:
    t = np.arange(60) * 60_000
    s = pd.Series([80.0] * 30 + [100.0] * 30, index=t)
    assert hourly_energy_mean(s).iloc[0] == pytest.approx(10 * np.log10((1e8 + 1e10) / 2))   # 97.03, not 90


# ---------------------------------------------------------------- the real M5 series

WEEK_DB = config.CONFIG_PATH.parent / config.get_ml_config()["anomaly"]["db_path"]


@pytest.mark.skipif(not Path(WEEK_DB).exists(), reason="7-day database not present")
def test_m5_detection_day_is_stable_across_reruns() -> None:
    from db.repo import Repo
    from ml.anomaly import load_readings
    c = config.get_ml_config()["anomaly"]
    exclude = set(config.get_exposure_methods()["personal"]["exclude_record_flags"])
    with Repo(WEEK_DB, read_only=True) as repo:
        start, end = repo.get_data_span()
        readings = load_readings(repo, "M5", start, end, exclude)
    bins = add_rolling(bin_vibration(readings, int(c["bin_min"]), float(c["min_running_fraction"])),
                       int(c["rolling_bins"]))
    ref = bins.index.to_numpy() < int(pd.Timestamp(start).value // 1_000_000) + int(c["reference_days"]) * DAY_MS
    a, b = detect(bins, ref, c, seed=42), detect(bins, ref, c, seed=42)
    for name in ("ewma", "isolation_forest"):
        assert a[name]["alarm_ms"] is not None
        assert a[name]["alarm_ms"] == b[name]["alarm_ms"]
