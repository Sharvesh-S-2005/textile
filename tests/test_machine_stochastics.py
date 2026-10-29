"""Tests for simulator/machine_stochastics.py (Day 8): measured, not assumed."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

import config
from simulator.verify_stochastics import (availability, noise_effect, onset_alignment, run_machines,
                                                verify)
from simulator.machine_stochastics import Plant, StopParams
from simulator.publisher import Simulation

START = datetime(2026, 11, 1, tzinfo=timezone.utc)
CFG = config.get_simulation_config()["machine_stochastics"]


@pytest.fixture(scope="module")
def long_run() -> tuple[Simulation, dict]:
    """14 simulated days of every machine at a 60-s step (the process is step-independent)."""
    sim = Simulation(START, timedelta(days=14), 3, [], machine_stochastics=True)
    return sim, run_machines(sim, step_s=60)


def test_realised_availability_falls_inside_each_target_band(long_run: tuple[Simulation, dict]) -> None:
    sim, trace = long_run
    for mid, a in availability(trace, sim).items():
        lo, hi = a["band"]
        assert lo <= a["availability"] <= hi, (mid, a)


def test_stop_onsets_are_not_aligned_to_the_shift_clock(long_run: tuple[Simulation, dict]) -> None:
    sim, trace = long_run
    a = onset_alignment(trace, sim)
    assert a["n_onsets"] > 500
    assert abs(a["pearson_r"]) < 0.05
    assert 0.7 < a["ratio"] < 1.3            # onsets near boundaries only as often as running time is


def test_deterministic_simulator_stops_only_at_the_clock() -> None:
    """Contrast: without stochastics every running -> stopped transition is a schedule boundary."""
    sim = Simulation(START, timedelta(days=2), 3, [], machine_stochastics=False)
    trace = run_machines(sim, step_s=60, noise_zone=None)
    from simulator.verify_stochastics import schedule_boundaries
    bounds = schedule_boundaries(sim.start, sim.end)
    for r in trace["running"].values():
        stops = np.flatnonzero(r[:-1] & ~r[1:]) + 1
        assert len(stops) > 0
        for k in stops:
            t = trace["ts"][k]
            assert min(abs((t - b).total_seconds()) for b in bounds) <= 180    # within the start stagger


def test_an_unplanned_stop_materially_lowers_zone_noise(long_run: tuple[Simulation, dict]) -> None:
    sim, trace = long_run
    e = noise_effect(trace, sim, "Z5")
    two, one = e["by_machines_running"]["2"]["l_aeq_dba"], e["by_machines_running"]["1"]["l_aeq_dba"]
    assert two - one > 2.0                                      # one loom down: > 2 dB quieter
    assert e["n_stops_measured"] > 50
    assert e["median_drop_db"] > 2.0 and e["median_recovery_db"] > 2.0   # falls on stop, recovers on restart


def test_the_correlated_dependency_fires() -> None:
    sim = Simulation(START, timedelta(days=3), 3, [], machine_stochastics=True)
    run_machines(sim, step_s=60, noise_zone=None)
    dep = CFG["dependencies"][0]
    up, down = sim.plant.machines[dep["upstream"]], sim.plant.machines[dep["downstream"]]
    forced = [s for s in down.stops if s.kind == "forced"]
    assert sim.plant.dependency_fired > 0 and len(forced) == sim.plant.dependency_fired
    long_up = [s for s in up.stops if s.kind == "unplanned" and s.end - s.start > timedelta(minutes=dep["buffer_min"])]
    # every forced stop starts buffer_min after a long upstream stop and ends restart_lag_min after it
    for f in forced:
        assert any(f.start == u.start + timedelta(minutes=dep["buffer_min"])
                   and f.end == u.end + timedelta(minutes=dep["restart_lag_min"]) for u in long_up)


def test_downstream_is_down_while_starved() -> None:
    plant = Plant(config.get_machines(), CFG, np.random.default_rng(0))
    dep = CFG["dependencies"][0]
    t0 = START + timedelta(hours=3)
    plant.on_stop(dep["upstream"], t0, t0 + timedelta(minutes=20))
    down = plant.for_machine(dep["downstream"])
    assert not down.in_stop(t0 + timedelta(minutes=dep["buffer_min"] - 1))    # buffer still feeding
    assert down.in_stop(t0 + timedelta(minutes=dep["buffer_min"] + 1))
    assert down.in_stop(t0 + timedelta(minutes=21))                          # restart lag
    assert not down.in_stop(t0 + timedelta(minutes=20 + dep["restart_lag_min"] + 1))
    plant.on_stop(dep["upstream"], t0 + timedelta(hours=1), t0 + timedelta(hours=1, minutes=2))
    assert plant.dependency_fired == 1                                       # a short choke is absorbed


def test_seeded_run_is_reproducible() -> None:
    a = verify(START, timedelta(days=1), 11, 60)
    b = verify(START, timedelta(days=1), 11, 60)
    assert a["availability"] == b["availability"] and a["stops"] == b["stops"]
    c = verify(START, timedelta(days=1), 12, 60)
    assert c["stops"] != a["stops"]


def test_load_stays_within_bounds_and_moves(long_run: tuple[Simulation, dict]) -> None:
    sim, _ = long_run
    lo, hi = CFG["load"]["bounds"]
    loads = [st.load for st in sim.plant.machines.values()]
    assert all(lo <= x <= hi for x in loads)
    assert len({round(x, 4) for x in loads}) == len(loads)


def test_expected_unplanned_availability_formula() -> None:
    """gap 25, median 3, sigma 0 -> 25 / 28."""
    assert StopParams(25.0, 3.0, 0.0, (0.0, 1.0)).expected_unplanned_availability == pytest.approx(25 / 28)


def test_stochastics_off_leaves_machines_deterministic() -> None:
    sim = Simulation(START, timedelta(hours=12), 3, [], machine_stochastics=False)
    assert sim.plant is None
    trace = run_machines(sim, step_s=60, noise_zone=None)
    for m in sim.machines:
        assert (trace["running"][m.machine_id] == trace["scheduled"][m.machine_id]).mean() > 0.99
