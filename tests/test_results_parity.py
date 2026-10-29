"""Day 8: both result sets must come from the same model configuration.

The only permitted difference between the deterministic and stochastic results is the
simulator. docs/results_day8_comparison.json (committed) records the configuration each
result set was produced with; the raw metrics files are checked too when present.
"""

import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

import config
from ml.compare_datasets import (crossing_clock_share, load_results, parity_record, result_paths,
                                 schedule_boundaries_ms)

ROOT = config.CONFIG_PATH.parent
RECORD = ROOT / "docs" / "results_day8_comparison.json"
C = config.get_ml_config()["comparison"]


@pytest.mark.skipif(not RECORD.exists(), reason="comparison not generated yet")
def test_committed_record_shows_identical_model_configuration() -> None:
    rec = json.loads(RECORD.read_text(encoding="utf-8"))["parity"]
    assert rec["identical"] and rec["differences"] == []
    assert rec["deterministic"] == rec["stochastic"]
    for part in ("forecast", "classification", "ablation", "anomaly"):
        assert part in rec["deterministic"]
    d = rec["deterministic"]
    assert d["forecast"]["splits"] == rec["stochastic"]["forecast"]["splits"]          # same dates
    assert d["classification"]["random_seed"] == d["ablation"]["random_seed"] == config.get_ml_config()["random_seed"]


HAVE_RAW = all(result_paths(ROOT / C[k])[m].exists() for k in ("deterministic_dir", "stochastic_dir")
               for m in ("forecast", "classification", "ablation", "anomaly"))


@pytest.mark.skipif(not HAVE_RAW, reason="raw metrics of both datasets not present")
def test_raw_metrics_were_produced_by_the_same_configuration() -> None:
    det, sto = load_results(ROOT / C["deterministic_dir"]), load_results(ROOT / C["stochastic_dir"])
    rec = parity_record(det, sto)
    assert rec["identical"], rec["differences"]
    assert det["forecast"]["hyperparameters"] == sto["forecast"]["hyperparameters"]
    assert det["ablation"]["grid"] == sto["ablation"]["grid"] == config.get_ml_config()["ablation"]["grid"]


def test_parity_record_detects_a_changed_hyperparameter() -> None:
    base = {"forecast": {"hyperparameters": {"rf": 1}, "split_rule": "r", "splits": ["s"], "n_features": 60,
                         "exceedance_threshold_db": 90.0},
            "classification": {"grids": {}, "random_seed": 42, "max_false_positive_rate": 0.1, "threshold_db": 90,
                               "split_rule": "r", "splits": ["s"], "n_features": 60, "zone_groups": []},
            "ablation": {k: 1 for k in ("model", "grid", "random_seed", "bootstrap", "feature_sets",
                                        "nl_feature_sets", "ns_feature_sets", "split_rule", "splits", "history",
                                        "max_false_positive_rate")},
            "anomaly": {"settings": {}, "random_seed": 42}}
    other = json.loads(json.dumps(base))
    other["forecast"]["hyperparameters"]["rf"] = 2
    assert parity_record(base, base)["identical"]
    assert parity_record(base, other)["differences"] == ["forecast"]


def test_changeover_mirror_matches_the_simulation_setting() -> None:
    assert C["changeover_min"] == config.get_simulation_config()["machine_dynamics"]["changeover_min"]


def test_crossing_clock_share_by_hand() -> None:
    """Two crossings in Z2: one 3 min after a break ends (near), one mid-shift (far)."""
    start = datetime(2026, 10, 21, 0, 0, tzinfo=timezone.utc)
    s_ms = int(start.timestamp() * 1000)
    b = schedule_boundaries_ms(s_ms, s_ms + 86_400_000, C["changeover_min"])
    assert len(b) > 4
    t = np.arange(s_ms, s_ms + 86_400_000, 60_000)
    noise = np.full(len(t), 85.0)
    near = int((b[2] - s_ms) // 60_000) + 3                       # 3 min after a boundary
    gaps = np.diff(b)
    i = int(np.argmax(gaps))
    far = int((b[i] + gaps[i] // 2 - s_ms) // 60_000)              # middle of the longest gap
    for k in (near, far):
        noise[k:k + 10] = 95.0
    df = pd.DataFrame({"t_ms": t, "zone_id": "Z2", "noise_dba": noise})
    r = crossing_clock_share(df, ["Z2"], 90.0, 5, 10, C["changeover_min"])
    assert r["n_events"] == 2 and r["n_near_boundary"] == 1 and r["share_near_boundary"] == 0.5
