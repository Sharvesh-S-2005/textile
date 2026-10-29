"""Tests for ml/deep_forecast.py."""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from ml.deep_forecast import SeqRegressor, gather_windows, stack_zones, train  # noqa: E402
from ml.forecast import MIN_MS  # noqa: E402

T0 = 1_792_540_800_000


def two_zones(n: int = 10) -> pd.DataFrame:
    rows = [{"zone_id": z, "t_ms": T0 + i * MIN_MS, "a": float(k * 100 + i)} for k, z in enumerate(("Z1", "Z2"))
            for i in range(n)]
    return pd.DataFrame(rows)


def test_window_ends_at_t_and_never_reads_the_future() -> None:
    df = two_zones()
    arr, end = stack_zones(df, ["a"], seq_len=4)
    row = df.index[(df["zone_id"] == "Z1") & (df["t_ms"] == T0 + 6 * MIN_MS)][0]
    w = gather_windows(arr, np.array([end[row]]), 4)[0, :, 0]
    np.testing.assert_array_equal(w, [3.0, 4.0, 5.0, 6.0])         # t-3 .. t, nothing after t


def test_windows_are_padded_at_a_zone_start_not_filled_from_another_zone() -> None:
    df = two_zones()
    arr, end = stack_zones(df, ["a"], seq_len=4)
    first_z2 = df.index[(df["zone_id"] == "Z2") & (df["t_ms"] == T0)][0]
    w = gather_windows(arr, np.array([end[first_z2]]), 4)[0, :, 0]
    assert np.isnan(w[:3]).all() and w[3] == 100.0                  # never Z1's last values


def test_gaps_in_a_zone_count_as_minutes() -> None:
    df = two_zones().drop(index=[3])                                 # Z1 minute 3 missing
    arr, end = stack_zones(df, ["a"], seq_len=3)
    row = df.index[(df["zone_id"] == "Z1") & (df["t_ms"] == T0 + 4 * MIN_MS)][0]
    w = gather_windows(arr, np.array([end[row]]), 3)[0, :, 0]
    assert w[0] == 2.0 and np.isnan(w[1]) and w[2] == 4.0


def test_training_reduces_validation_error_on_a_learnable_sequence() -> None:
    rng = np.random.default_rng(0)
    n, seq = 3000, 8
    x = rng.normal(size=(n, 2)).astype(np.float32)
    y = (x[:, 0] + np.roll(x[:, 1], 3)).astype(np.float32)          # depends on a lagged input
    idx = np.arange(seq, n)
    c = {"seq_len_min": seq, "batch_size": 128, "hidden": 16, "layers": 1, "cell": "gru", "learning_rate": 0.01,
         "max_epochs": 8, "patience": 3}
    tr, va = idx[: 2000], idx[2000:]
    untrained = SeqRegressor(2, 16, 1, "gru")
    with torch.no_grad():
        base = float(np.mean(np.abs(untrained(torch.from_numpy(gather_windows(x, va, seq))).numpy() - y[va])))
    _, info = train(x, tr, y[tr], va, y[va], c, seed=0, deadline=float("inf"))
    assert min(info["val_mae_history"]) < 0.5 * base
    assert info["best_epoch"] <= info["epochs_run"]
