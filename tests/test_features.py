"""Tests for ml/features.py, starting with the leakage checks."""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import config
from db.repo import Repo, from_epoch_ms, to_epoch_ms
from ml.features import build_features, feature_columns, feature_source_offsets, target_columns

T0 = to_epoch_ms("2026-10-28T00:00:00Z")
RX = "2026-10-28T09:00:00.000Z"


def _env(ts_ms: int, noise: float | None, pm: float | None = 100.0, pm_flag: str = "ok") -> dict:
    return {"schema_version": "1.1", "device_id": "ZN-Z1", "ts": from_epoch_ms(ts_ms), "zone_id": "Z1",
            "noise_dba": noise, "noise_peak_dba": None if noise is None else noise + 3, "pm25_ugm3": pm,
            "pm10_ugm3": None if pm is None else 2 * pm, "temp_c": 30.0, "rh_pct": 60.0, "globe_temp_c": 32.0,
            "co2_ppm": 700,
            "sensor_flags": {"noise": "ok" if noise is not None else "error", "pm": pm_flag,
                             "temp_rh": "ok", "globe": "ok", "co2": "ok"}}


@pytest.fixture
def synthetic_db(tmp_path: Path) -> Path:
    """Z1 only, 120 minutes, 6 readings/min; minute k (bin ending T0+(k+1) min) is 80 + k/10 dBA."""
    db = tmp_path / "f.db"
    with Repo(db) as repo:
        repo.init_schema()
        for k in range(120):
            for j in range(6):
                ts = T0 + k * 60_000 + (j + 1) * 10_000
                repo.insert_zone_env("textile/zone/Z1/env", _env(ts, 80.0 + k / 10), RX, [])
        repo.commit()
    return db


def _row(df: pd.DataFrame, zone: str, t_ms: int) -> pd.Series:
    hit = df[(df["zone_id"] == zone) & (df["t_ms"] == t_ms)]
    assert len(hit) == 1, (zone, t_ms, len(hit))
    return hit.iloc[0]


# ---------------------------------------------------------------- leakage

def test_declared_feature_sources_never_lie_after_t() -> None:
    offsets = feature_source_offsets()
    assert offsets and all(v <= 0 for v in offsets.values()), {k: v for k, v in offsets.items() if v > 0}
    assert not set(target_columns()) & set(offsets)                 # targets are never features
    assert not any(c.startswith("seasonal_") for c in offsets)      # baselines are not features


def test_truncation_leakage_check_on_real_pipeline_data(pipeline_db: Path) -> None:
    """For sampled (zone, t): rebuild the features from data cut off at t. If any
    feature looked past t, its value would change. All must be identical."""
    feats = feature_columns()
    with Repo(pipeline_db) as repo:
        full = build_features(repo, drop_missing_targets=False)
        rng = np.random.default_rng(0)
        middle = full[(full["t_ms"] > full["t_ms"].min() + 3_600_000) & (full["t_ms"] < full["t_ms"].max() - 3_600_000)]
        samples = middle.iloc[rng.choice(len(middle), size=6, replace=False)]
        for _, sample in samples.iterrows():
            t = int(sample["t_ms"])
            cut = build_features(repo, until_ms=t, drop_missing_targets=False)
            truncated = _row(cut, sample["zone_id"], t)
            for f in feats:
                a, b = sample[f], truncated[f]
                assert (pd.isna(a) and pd.isna(b)) or a == pytest.approx(b, rel=1e-12, abs=1e-12), (f, t, a, b)


# ---------------------------------------------------------------- hand-computed values

def test_lag_features_match_hand_values(synthetic_db: Path) -> None:
    with Repo(synthetic_db) as repo:
        df = build_features(repo)
    t = T0 + 60 * 60_000                       # bin ending at minute 60 holds minute k = 59
    r = _row(df, "Z1", t)
    assert r["noise_dba"] == pytest.approx(80.0 + 5.9)
    for k in config.get_ml_config()["lags_min"]:
        assert r[f"noise_dba_lag{k}"] == pytest.approx(80.0 + (59 - k) / 10), k
    assert r["noise_dba_roc5"] == pytest.approx(0.5)
    assert r["target_noise_dba_h30"] == pytest.approx(80.0 + 8.9)
    assert r["target_noise_dba_h60"] == pytest.approx(80.0 + 11.9)


def test_noise_rolling_mean_is_energy_averaged(synthetic_db: Path) -> None:
    """rmean5 at minute 59 = 10 log10(mean(10^((80 + k/10)/10)) for k = 55..59)."""
    with Repo(synthetic_db) as repo:
        r = _row(build_features(repo), "Z1", T0 + 60 * 60_000)
    expected = 10 * math.log10(np.mean([10 ** ((80 + k / 10) / 10) for k in range(55, 60)]))
    assert r["noise_dba_rmean5"] == pytest.approx(expected, abs=1e-9)
    assert r["pm25_ugm3_rmean5"] == pytest.approx(100.0)              # linear quantity, plain mean


def test_minute_resample_of_noise_is_energy_not_arithmetic(tmp_path: Path) -> None:
    """One minute with three readings at 80 and three at 90 dBA -> 87.4 dBA, not 85."""
    db = tmp_path / "e.db"
    with Repo(db) as repo:
        repo.init_schema()
        for k in range(70):
            for j in range(6):
                level = (80.0 if j < 3 else 90.0) if k == 0 else 85.0
                repo.insert_zone_env("textile/zone/Z1/env", _env(T0 + k * 60_000 + (j + 1) * 10_000, level), RX, [])
        repo.commit()
        r = _row(build_features(repo), "Z1", T0 + 60_000)
    assert r["noise_dba"] == pytest.approx(87.40, abs=0.01)


def test_data_gap_reduces_coverage_instead_of_fabricating_lags(tmp_path: Path) -> None:
    db = tmp_path / "g.db"
    with Repo(db) as repo:
        repo.init_schema()
        for k in range(120):
            if 40 <= k < 45:
                continue                                              # 5-minute outage, no rows at all
            for j in range(6):
                repo.insert_zone_env("textile/zone/Z1/env", _env(T0 + k * 60_000 + (j + 1) * 10_000, 85.0), RX, [])
        repo.commit()
        df = build_features(repo)
    gap_row = _row(df, "Z1", T0 + 43 * 60_000)        # bin of minute 42, inside the gap
    assert pd.isna(gap_row["noise_dba"]) and pd.isna(gap_row["noise_dba_lag1"])
    after = _row(df, "Z1", T0 + 47 * 60_000)          # minute 46: lag3 and lag5 fall in the gap
    assert pd.isna(after["noise_dba_lag3"]) and pd.isna(after["noise_dba_lag5"])
    assert after["noise_dba_lag1"] == 85.0
    assert after["feature_coverage_pct"] < 100.0
    clean = _row(df, "Z1", T0 + 80 * 60_000)          # well after the gap, target still present
    assert clean["feature_coverage_pct"] > after["feature_coverage_pct"]


def test_row_count_for_a_known_range(synthetic_db: Path) -> None:
    """120 minutes of Z1 data: targets at t+30 exist for the first 90 bins. Zones with no
    data have no targets and are dropped. Expected rows: 90."""
    with Repo(synthetic_db) as repo:
        df = build_features(repo)
    assert len(df) == 90 and set(df["zone_id"]) == {"Z1"}
    assert df["target_noise_dba_h60"].notna().sum() == 60


def test_flagged_and_null_readings_are_excluded_not_zero_filled(tmp_path: Path) -> None:
    db = tmp_path / "x.db"
    with Repo(db) as repo:
        repo.init_schema()
        for k in range(70):
            for j in range(6):
                ts = T0 + k * 60_000 + (j + 1) * 10_000
                if k == 0 and j == 0:
                    repo.insert_zone_env("textile/zone/Z1/env", _env(ts, 135.0), RX, ["spike"])   # glitch
                elif k == 0 and j == 1:
                    repo.insert_zone_env("textile/zone/Z1/env", _env(ts, None, pm=None), RX, ["sensor_not_ok"])
                elif k == 1:
                    repo.insert_zone_env("textile/zone/Z1/env", _env(ts, 85.0, pm=900.0, pm_flag="needs_clean"),
                                         RX, ["sensor_not_ok"])
                else:
                    repo.insert_zone_env("textile/zone/Z1/env", _env(ts, 85.0), RX, [])
        repo.commit()
        df = build_features(repo)
    m0, m1 = _row(df, "Z1", T0 + 60_000), _row(df, "Z1", T0 + 120_000)
    assert m0["noise_dba"] == pytest.approx(85.0)       # the 135 dBA glitch is excluded, NULL skipped
    assert pd.isna(m1["pm25_ugm3"])                     # needs_clean PM values excluded, not zeroed
    assert m1["noise_dba"] == pytest.approx(85.0)       # other fields of those records still used


def test_features_are_built_from_stored_tables_only() -> None:
    """No simulator import, no ground-truth file, no exposure recomputation (imports checked)."""
    import re

    source = (Path(__file__).resolve().parents[1] / "ml" / "features.py").read_text(encoding="utf-8")
    imports = re.findall(r"^\s*(?:from|import)\s+([\w.]+)", source, re.MULTILINE)
    assert not [m for m in imports if m.startswith(("simulator", "exposure.personal", "exposure.engine",
                                                    "scripts"))], imports
    assert "truth.jsonl" not in source and "ground_truth" not in source
