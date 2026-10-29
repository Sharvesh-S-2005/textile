"""Measure what the stochastic machine layer actually does (Day 8 pre-registration).

    python -m simulator.verify_stochastics [--days 28] [--seed 1] [--step-s 10]
    python -m simulator.verify_stochastics --dataset-window   # the exact 7-day dataset window, seed 42

Lives in simulator/ because it drives the simulator's own models; downstream code
(scripts/, ml/, ...) never imports the simulator.

Runs the simulator's own MachineModel objects (built by simulator.publisher.Simulation,
the same construction the MQTT run uses) without MQTT, and measures:

  * realised availability per machine = running time / deterministic scheduled time
    (late and skipped shift starts count against it), against the configured band;
  * whether unplanned stop onsets line up with the shift clock: Pearson r between a
    per-minute onset count and a "within 10 min of a schedule boundary" indicator,
    and the ratio (share of onsets near a boundary) / (share of running minutes near one);
  * how often the blowroom -> carding dependency fired;
  * the zone-noise effect: energy-averaged Z5 noise by number of Z5 machines running
    during scheduled time, and noise before / during / after each unplanned Z5 stop.

This is a measurement of the SIMULATOR, not of any model. It is the evidence the
pre-registration quotes for "realised availability lands in its band".
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

import config
from shifts import ShiftSchedule
from simulator.publisher import Simulation, align_window

BOUNDARY_WINDOW_MIN = 10
NOISE_ZONE = "Z5"


def energy_mean(levels: list[float]) -> float | None:
    """Energy (logarithmic) mean of dBA levels; None for an empty list."""
    if not levels:
        return None
    return 10.0 * math.log10(float(np.mean(10.0 ** (np.asarray(levels) / 10.0))))


def schedule_boundaries(start: datetime, end: datetime) -> list[datetime]:
    """Every moment the deterministic calendar switches machines on or off."""
    sched = ShiftSchedule(config.get_shifts())
    changeover = timedelta(minutes=config.get_simulation_config()["machine_dynamics"]["changeover_min"])
    out = []
    for inst in sched.instances_between(start, end):
        out += [inst.start + changeover, inst.end] + [b.start for b in inst.breaks] + [b.end for b in inst.breaks]
    return sorted(b for b in out if start <= b < end)


def run_machines(sim: Simulation, step_s: int, noise_zone: str | None = NOISE_ZONE) -> dict[str, Any]:
    """Step every machine (and optionally one zone's noise) through the window.

    Returns per-step arrays: ts, scheduled[m], running[m], and noise (dBA or NaN).
    """
    n = int((sim.end - sim.start).total_seconds() // step_s)
    ts_list = [sim.start + timedelta(seconds=i * step_s) for i in range(n)]
    scheduled = {m.machine_id: np.zeros(n, dtype=bool) for m in sim.machines}
    running = {m.machine_id: np.zeros(n, dtype=bool) for m in sim.machines}
    noise = np.full(n, np.nan)
    for i, ts in enumerate(ts_list):
        for m in sim.machines:
            scheduled[m.machine_id][i] = m.scheduled_on(ts)
            m.reading(ts)
            running[m.machine_id][i] = m.running_now
        if noise_zone is not None:
            factors = [m.noise_energy_factor(ts) for m in sim.machines if m.zone_id == noise_zone and m.running_now]
            value = sim.profiles[noise_zone].reading(ts, factors)["noise_dba"]
            if value is not None:
                noise[i] = value
    return {"ts": ts_list, "scheduled": scheduled, "running": running, "noise": noise, "step_s": step_s}


def availability(trace: dict[str, Any], sim: Simulation) -> dict[str, dict[str, Any]]:
    """Realised availability per machine against its configured band."""
    cfg = sim.sim["machine_stochastics"]
    out = {}
    for m in sim.machines:
        s, r = trace["scheduled"][m.machine_id], trace["running"][m.machine_id]
        a = float((s & r).sum() / s.sum()) if s.sum() else None
        mtype = cfg["zone_machine_type"][m.zone_id]
        lo, hi = cfg["types"][mtype]["target_availability"]
        out[m.machine_id] = {"zone_id": m.zone_id, "type": mtype, "availability": a, "band": [lo, hi],
                             "in_band": a is not None and lo <= a <= hi,
                             "scheduled_h": float(s.sum() * trace["step_s"] / 3600.0)}
    return out


def onset_alignment(trace: dict[str, Any], sim: Simulation) -> dict[str, Any]:
    """Are unplanned stop onsets aligned with the shift clock?"""
    start, end = sim.start, sim.end
    n_min = int((end - start).total_seconds() // 60)
    onsets = np.zeros(n_min)
    for st in (sim.plant.machines.values() if sim.plant else []):
        for s in st.stops:
            if s.kind == "unplanned" and start <= s.start < end:
                onsets[int((s.start - start).total_seconds() // 60)] += 1
    near = np.zeros(n_min, dtype=bool)
    for b in schedule_boundaries(start, end):
        k = int((b - start).total_seconds() // 60)
        near[max(0, k - BOUNDARY_WINDOW_MIN):min(n_min, k + BOUNDARY_WINDOW_MIN)] = True
    per_min = trace["step_s"] / 60.0
    any_running = np.zeros(n_min, dtype=bool)
    for r in trace["running"].values():
        idx = (np.flatnonzero(r) * per_min).astype(int)
        any_running[idx[idx < n_min]] = True
    mask = any_running
    r = float(np.corrcoef(onsets[mask], near[mask].astype(float))[0, 1]) if onsets[mask].std() > 0 else float("nan")
    share_onsets = float(onsets[mask & near].sum() / onsets[mask].sum()) if onsets[mask].sum() else float("nan")
    share_minutes = float(near[mask].mean())
    return {"n_onsets": int(onsets.sum()), "pearson_r": r, "share_of_onsets_near_boundary": share_onsets,
            "share_of_running_minutes_near_boundary": share_minutes,
            "ratio": share_onsets / share_minutes if share_minutes else float("nan"),
            "boundary_window_min": BOUNDARY_WINDOW_MIN}


def noise_effect(trace: dict[str, Any], sim: Simulation, zone: str = NOISE_ZONE) -> dict[str, Any]:
    """Zone noise by number of the zone's machines running (scheduled time only), and per-stop deltas."""
    ids = [m.machine_id for m in sim.machines if m.zone_id == zone]
    sched = np.all([trace["scheduled"][i] for i in ids], axis=0)
    n_run = np.sum([trace["running"][i] for i in ids], axis=0)
    noise = trace["noise"]
    by_count = {}
    for k in range(len(ids) + 1):
        sel = sched & (n_run == k) & ~np.isnan(noise)
        by_count[str(k)] = {"n_readings": int(sel.sum()), "l_aeq_dba": energy_mean(list(noise[sel]))}
    ts = trace["ts"]
    t0 = ts[0]
    step = trace["step_s"]
    deltas_drop, deltas_recover = [], []
    for i in ids:
        st = sim.plant.machines[i] if sim.plant else None
        for s in (st.stops if st else []):
            if s.kind != "unplanned" or (s.end - s.start) < timedelta(minutes=3):
                continue
            a = int((s.start - t0).total_seconds() // step)
            b = int((s.end - t0).total_seconds() // step)
            w = int(300 // step)
            if a - w < 0 or b + w + int(60 // step) >= len(ts) or not sched[a - w:b + w].all():
                continue
            before = energy_mean([x for x in noise[a - w:a] if not np.isnan(x)])
            during = energy_mean([x for x in noise[a + int(60 // step):b] if not np.isnan(x)])
            after = energy_mean([x for x in noise[b + int(60 // step):b + int(60 // step) + w] if not np.isnan(x)])
            if None not in (before, during, after):
                deltas_drop.append(before - during)
                deltas_recover.append(after - during)
    return {"zone": zone, "by_machines_running": by_count, "n_stops_measured": len(deltas_drop),
            "median_drop_db": float(np.median(deltas_drop)) if deltas_drop else None,
            "median_recovery_db": float(np.median(deltas_recover)) if deltas_recover else None,
            "share_of_stops_lowering_noise": float(np.mean(np.asarray(deltas_drop) > 0)) if deltas_drop else None}


def verify(start: datetime, duration: timedelta, seed: int, step_s: int,
           whole_sessions_only: bool = False) -> dict[str, Any]:
    """Build a stochastic Simulation for the window and measure it."""
    sim = Simulation(start, duration, seed, [], whole_sessions_only=whole_sessions_only, machine_stochastics=True)
    trace = run_machines(sim, step_s)
    return {"start": start.isoformat(), "end": (start + duration).isoformat(), "seed": seed, "step_s": step_s,
            "availability": availability(trace, sim), "onset_alignment": onset_alignment(trace, sim),
            "dependency_fired": sim.plant.dependency_fired,
            "stops": {m: {"unplanned": sum(s.kind == "unplanned" for s in st.stops),
                          "forced": sum(s.kind == "forced" for s in st.stops)}
                      for m, st in sim.plant.machines.items()},
            "noise_effect": noise_effect(trace, sim)}


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    p = argparse.ArgumentParser(prog="python -m simulator.verify_stochastics", description=__doc__.split("\n")[0])
    p.add_argument("--days", type=float, default=28.0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--step-s", type=int, default=int(config.get_simulation_config()["machine_interval_s"]))
    p.add_argument("--start", default="2026-11-01T00:00:00+00:00")
    p.add_argument("--dataset-window", action="store_true",
                   help="the 7-day dataset window and seed (start 2026-10-20 06:00 IST, 168 h, aligned, seed 42)")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)
    if args.dataset_window:
        req = datetime(2026, 10, 20, 0, 30, tzinfo=timezone.utc)
        start, end = align_window(req, timedelta(hours=168))
        result = verify(start, end - start, int(config.get_simulation_config()["random_seed"]), args.step_s,
                        whole_sessions_only=True)
    else:
        start = datetime.fromisoformat(args.start)
        result = verify(start, timedelta(days=args.days), args.seed, args.step_s)
    text = json.dumps(result, indent=2, default=str)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text)
    bad = [m for m, a in result["availability"].items() if not a["in_band"]]
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
