"""Forecasting feature table: one row per zone per 1-minute bin.

    python -m ml.features [--db PATH] [--out data/features.parquet]

Built only from stored tables through db/repo.py - no exposure recomputation, no
simulator code, no ground truth.

Row time `t` is the END of a 1-minute bin. A zone_env reading covers the 10 s
before its ts, so the bin ending at t holds readings with t-60 s < ts <= t: all
known by t.

No feature uses data from after t. Every feature column declares its source offset
(minutes relative to t, always <= 0) in FEATURE_SOURCE_OFFSET_MIN, and
tests/test_features.py checks it two ways: the declared offsets, and a truncation
test (features rebuilt from data cut off at t equal the full-data features at t).
Two leaks are avoided deliberately:
  * workers_present_count uses the assignment for the previous COMPLETE BLE window
    (t - window_s), because the window containing t may hold sightings after t;
  * it ignores `outside_session`, which depends on the (future) check-out time.

Excluded from features (never zero-filled): records flagged as detected anomalies
(spike, stuck_value), NULL values, and any value whose own sensor flag is not "ok"
(e.g. a drifting needs_clean PM value). Missing inputs leave NaN, and each row
carries `feature_coverage_pct`.

Noise is energy-averaged: the 1-minute resample uses exposure.noise.l_aeq, and the
rolling means of noise are computed on 10^(L/10) and converted back. Rolling std is
a spread measure and stays in dB.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import config
from db.repo import Repo, from_epoch_ms, to_epoch_ms
from exposure.heat import wbgt_estimate
from exposure.noise import l_aeq
from shifts import ShiftSchedule

log = logging.getLogger("ml.features")

UNKNOWN_ZONE = "UNKNOWN"
SHIFT_CODES = {"A": 0, "B": 1, "C": 2}
# Per-field sensor flag columns in zone_env.
FIELD_FLAG = {"noise_dba": "flag_noise", "pm25_ugm3": "flag_pm", "temp_c": "flag_temp_rh",
              "rh_pct": "flag_temp_rh", "co2_ppm": "flag_co2", "globe_temp_c": "flag_globe"}
ENV_FIELDS = ("temp_c", "rh_pct", "co2_ppm")


def _cfg() -> dict[str, Any]:
    return config.get_ml_config()


def target_columns() -> list[str]:
    """target_{var}_h{h} for every target and horizon."""
    c = _cfg()
    return [f"target_{v}_h{h}" for v in c["targets"] for h in c["horizons_min"]]


def feature_source_offsets() -> dict[str, int]:
    """Every model feature -> offset (minutes, <= 0) of the newest data it uses relative to t."""
    c = _cfg()
    out: dict[str, int] = {}
    for v in c["targets"]:
        out[v] = 0
        for k in c["lags_min"]:
            out[f"{v}_lag{k}"] = -k
        for w in c["rolling_min"]:
            out[f"{v}_rmean{w}"] = 0
            out[f"{v}_rstd{w}"] = 0
        out[f"{v}_roc{c['roc_min']}"] = 0
    for f in ENV_FIELDS:
        out[f] = 0
        out[f"{f}_rmean{c['env_rolling_min']}"] = 0
    for f in ("machines_running_count", "mean_current_a", "max_vib_rms_g", "workers_present_count",
              "minute_sin", "minute_cos", "shift_code", "is_break", "zone_code"):
        out[f] = 0
    return out


def feature_columns() -> list[str]:
    """Model feature columns, in a fixed order."""
    return list(feature_source_offsets())


# ---------------------------------------------------------------- per-zone series


def _bin_end(ts_ms: int, bin_ms: int) -> int:
    """End of the bin (m, m + bin] holding a reading at ts_ms."""
    return ((ts_ms - 1) // bin_ms + 1) * bin_ms


def zone_minute_series(rows: list[dict[str, Any]], bin_ms: int, exclude: set[str],
                       reading_interval_s: float) -> pd.DataFrame:
    """1-minute values for one zone from its zone_env rows (index: bin-end epoch ms).

    Noise is the energy average (exposure.noise.l_aeq); other fields are plain means.
    """
    per_bin: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if exclude & set(r["flags"]):
            continue
        b = _bin_end(r["ts_epoch_ms"], bin_ms)
        valid = {f: r[f] for f, flag in FIELD_FLAG.items() if r[f] is not None and r[flag] == "ok"}
        for f in ("noise_dba", "pm25_ugm3", "temp_c", "rh_pct", "co2_ppm"):
            if f in valid:
                per_bin[b][f].append(float(valid[f]))
        if {"temp_c", "rh_pct", "globe_temp_c"} <= set(valid):
            per_bin[b]["wbgt_c"].append(wbgt_estimate(valid["temp_c"], valid["rh_pct"], valid["globe_temp_c"]).wbgt_c)
    records = []
    for b, vals in per_bin.items():
        rec: dict[str, Any] = {"t_ms": b}
        noise = vals.get("noise_dba")
        rec["noise_dba"] = l_aeq(noise, [reading_interval_s] * len(noise)) if noise else np.nan
        for f in ("pm25_ugm3", "wbgt_c", "temp_c", "rh_pct", "co2_ppm"):
            rec[f] = float(np.mean(vals[f])) if vals.get(f) else np.nan
        records.append(rec)
    df = pd.DataFrame.from_records(records, columns=["t_ms", "noise_dba", "pm25_ugm3", "wbgt_c", "temp_c",
                                                     "rh_pct", "co2_ppm"])
    return df.set_index("t_ms").sort_index()


def machine_minute_series(rows_by_machine: dict[str, list[dict[str, Any]]], bin_ms: int,
                          exclude: set[str]) -> pd.DataFrame:
    """Per-bin machine features for one zone: running count, mean current, max vibration."""
    frames = []
    for machine_id, rows in rows_by_machine.items():
        recs: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for r in rows:
            if exclude & set(r["flags"]):
                continue
            b = _bin_end(r["ts_epoch_ms"], bin_ms)
            recs[b]["running"].append(float(r["running"]))
            if r["current_a"] is not None and r["flag_current"] == "ok":
                recs[b]["current"].append(r["current_a"])
            if r["vib_rms_g"] is not None and r["flag_vibration"] == "ok":
                recs[b]["vib"].append(r["vib_rms_g"])
        frames.append(pd.DataFrame.from_records(
            [{"t_ms": b, "machine": machine_id, "running": np.mean(v["running"]) if v.get("running") else np.nan,
              "current": np.mean(v["current"]) if v.get("current") else np.nan,
              "vib": np.max(v["vib"]) if v.get("vib") else np.nan} for b, v in recs.items()],
            columns=["t_ms", "machine", "running", "current", "vib"]))
    if not frames:
        return pd.DataFrame(columns=["machines_running_count", "mean_current_a", "max_vib_rms_g"])
    long = pd.concat(frames)
    g = long.groupby("t_ms")
    return pd.DataFrame({"machines_running_count": g["running"].sum(min_count=1),
                         "mean_current_a": g["current"].mean(), "max_vib_rms_g": g["vib"].max()})


def workers_present(assignments: list[dict[str, Any]], zone_id: str, grid_ms: np.ndarray, window_ms: int) -> np.ndarray:
    """Workers assigned to `zone_id` in the previous complete BLE window, at each grid time."""
    counts = np.zeros(len(grid_ms), dtype=float)
    for a in assignments:
        if a["zone_id"] != zone_id:
            continue
        # Present at t if the assignment covers t - window_ms: t in [from + w, to + w).
        lo = np.searchsorted(grid_ms, a["from_epoch_ms"] + window_ms, side="left")
        hi = np.searchsorted(grid_ms, a["to_epoch_ms"] + window_ms, side="left")
        counts[lo:hi] += 1
    return counts


# ---------------------------------------------------------------- table assembly


def _energy_rolling_mean(series: pd.Series, window: int) -> pd.Series:
    energy = 10.0 ** (series / 10.0)
    return 10.0 * np.log10(energy.rolling(window, min_periods=1).mean())


def _series_features(s: pd.Series, name: str, energy: bool) -> dict[str, pd.Series]:
    c = _cfg()
    out = {name: s}
    for k in c["lags_min"]:
        out[f"{name}_lag{k}"] = s.shift(k)
    for w in c["rolling_min"]:
        out[f"{name}_rmean{w}"] = _energy_rolling_mean(s, w) if energy else s.rolling(w, min_periods=1).mean()
        out[f"{name}_rstd{w}"] = s.rolling(w, min_periods=2).std()
    out[f"{name}_roc{c['roc_min']}"] = s - s.shift(c["roc_min"])
    return out


def build_features(repo: Repo, until_ms: int | None = None, drop_missing_targets: bool = True) -> pd.DataFrame:
    """Build the feature table from the database.

    Args:
        repo: open database.
        until_ms: if given, use only data with ts <= until_ms (the leakage test uses this).
        drop_missing_targets: drop rows with no target at all (rows missing only
            some targets are kept; the forecaster filters per target).
    """
    c = _cfg()
    bin_ms = int(c["bin_s"]) * 1000
    exclude = set(config.get_exposure_methods()["personal"]["exclude_record_flags"])
    reading_s = float(config.get_exposure_methods()["personal"]["env_reading_interval_s"])
    window_ms = int(config.get_localisation_config()["window_s"]) * 1000
    span = repo.get_data_span()
    if span is None:
        raise ValueError("no zone_env data in the database")
    lo = span[0]
    hi_ms = to_epoch_ms(span[1]) if until_ms is None else min(to_epoch_ms(span[1]), until_ms)
    hi_q = from_epoch_ms(_bin_end(hi_ms, bin_ms) + 1)
    machines_by_zone: dict[str, list[str]] = defaultdict(list)
    for m in config.get_machines():
        machines_by_zone[m["zone_id"]].append(m["machine_id"])
    assignments = [a for a in repo.get_all_zone_assignments() if a["zone_id"] != UNKNOWN_ZONE]
    if until_ms is not None:
        assignments = [dict(a, to_epoch_ms=min(a["to_epoch_ms"], until_ms)) for a in assignments
                       if a["from_epoch_ms"] < until_ms]

    def cut(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return rows if until_ms is None else [r for r in rows if r["ts_epoch_ms"] <= until_ms]

    zone_frames = {}
    for zi, zone in enumerate(config.get_zones()):
        zid = zone["zone_id"]
        env = zone_minute_series(cut(repo.get_zone_env(zid, lo, hi_q)), bin_ms, exclude, reading_s)
        mach = machine_minute_series({m: cut(repo.get_machine_status(m, lo, hi_q)) for m in machines_by_zone[zid]},
                                     bin_ms, exclude)
        zone_frames[zid] = (zi, env, mach)

    first = min(f[1].index.min() for f in zone_frames.values() if len(f[1]))
    last = max(f[1].index.max() for f in zone_frames.values() if len(f[1]))
    grid = np.arange(first, last + bin_ms, bin_ms, dtype=np.int64)
    schedule = ShiftSchedule(config.get_shifts())
    shift_code, is_break = {}, {}
    for t in grid:
        ts = pd.Timestamp(int(t), unit="ms", tz="UTC").to_pydatetime()
        inst = schedule.active_at(ts)
        shift_code[int(t)] = SHIFT_CODES[inst.shift_id] if inst else -1
        is_break[int(t)] = int(inst is not None and inst.break_at(ts) is not None)

    tables = []
    for zid, (zi, env, mach) in zone_frames.items():
        env = env.reindex(grid)
        mach = mach.reindex(grid)
        if not machines_by_zone[zid]:                      # no machines in this zone: true zeros
            mach = pd.DataFrame({"machines_running_count": 0.0, "mean_current_a": 0.0, "max_vib_rms_g": 0.0},
                                index=grid)
        cols: dict[str, Any] = {}
        for v in c["targets"]:
            cols.update(_series_features(env[v], v, energy=(v == "noise_dba")))
        for f in ENV_FIELDS:
            cols[f] = env[f]
            cols[f"{f}_rmean{c['env_rolling_min']}"] = env[f].rolling(c["env_rolling_min"], min_periods=1).mean()
        cols.update({k: mach[k] for k in ("machines_running_count", "mean_current_a", "max_vib_rms_g")})
        cols["workers_present_count"] = workers_present(assignments, zid, grid, window_ms)
        minute = (grid // 60000) % 1440
        cols["minute_sin"] = np.sin(2 * math.pi * minute / 1440)
        cols["minute_cos"] = np.cos(2 * math.pi * minute / 1440)
        cols["shift_code"] = [shift_code[int(t)] for t in grid]
        cols["is_break"] = [is_break[int(t)] for t in grid]
        cols["zone_code"] = zi
        # Targets (future values) and the seasonal baseline (same clock minute, previous day).
        day_bins = 86_400_000 // bin_ms
        for v in c["targets"]:
            for h in c["horizons_min"]:
                cols[f"target_{v}_h{h}"] = env[v].shift(-h)
                cols[f"seasonal_{v}_h{h}"] = env[v].shift(day_bins - h)
        df = pd.DataFrame(cols, index=grid)
        df.insert(0, "zone_id", zid)
        tables.append(df)

    table = pd.concat(tables)
    table.index.name = "t_ms"
    table = table.reset_index()
    table.insert(1, "ts", [from_epoch_ms(int(t)) for t in table["t_ms"]])
    feats = feature_columns()
    table["feature_coverage_pct"] = 100.0 * table[feats].notna().mean(axis=1)
    targets = target_columns()
    if drop_missing_targets:
        table = table[table[targets].notna().any(axis=1)].reset_index(drop=True)
    table["zone_id"] = table["zone_id"].astype("category")
    return table


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(prog="python -m ml.features", description=__doc__.split("\n")[0])
    p.add_argument("--db", type=Path, default=config.CONFIG_PATH.parent / config.get_paths()["db_path"])
    p.add_argument("--out", type=Path, default=config.CONFIG_PATH.parent / _cfg()["features_path"])
    args = p.parse_args(argv)
    with Repo(args.db) as repo:
        table = build_features(repo)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.out, index=False)
    log.info("wrote %d rows x %d columns to %s (mean feature coverage %.2f%%)", len(table), table.shape[1],
             args.out, table["feature_coverage_pct"].mean())
    return 0


if __name__ == "__main__":
    sys.exit(main())
