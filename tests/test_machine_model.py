"""Tests for simulator/schedule.py and simulator/machine_model.py."""

import copy
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pytest

import config
from simulator.machine_model import MachineModel
from shifts import ShiftSchedule

UTC = timezone.utc
SHIFT_A_START = datetime(2026, 10, 28, 0, 30, tzinfo=UTC)   # 06:00 IST
TEA_1 = datetime(2026, 10, 28, 2, 30, tzinfo=UTC)           # 08:00 IST, 10 min
DT = timedelta(seconds=config.get_simulation_config()["machine_interval_s"])
DYN = config.get_simulation_config()["machine_dynamics"]


@pytest.fixture
def schedule() -> ShiftSchedule:
    return ShiftSchedule(config.get_shifts())


def _machine(machine_id: str, schedule: ShiftSchedule, seed: int = 42,
             start: datetime = SHIFT_A_START, fault_rate_per_h: float = 0.0) -> MachineModel:
    """Machine with sensor faults off by default, so measurement tests see every value."""
    cfg = next(m for m in config.get_machines() if m["machine_id"] == machine_id)
    dynamics = copy.deepcopy(DYN)
    dynamics["sensor_faults"]["rate_per_sensor_per_h"] = fault_rate_per_h
    return MachineModel(cfg, dynamics, config.get_validation_config()["fields"], schedule,
                        np.random.default_rng(seed), start)


def _run(m: MachineModel, start: datetime, n: int) -> list[tuple[datetime, dict]]:
    return [(start + i * DT, m.reading(start + i * DT)) for i in range(n)]


# ---------------------------------------------------------------- schedule

def test_shift_a_times_in_utc(schedule: ShiftSchedule) -> None:
    a = schedule.instance("A", date(2026, 10, 28))
    assert a.start == SHIFT_A_START
    assert a.end == datetime(2026, 10, 28, 8, 30, tzinfo=UTC)
    assert a.breaks[0].start == TEA_1
    assert a.breaks[0].end == TEA_1 + timedelta(minutes=10)


def test_shift_c_crosses_midnight(schedule: ShiftSchedule) -> None:
    c = schedule.instance("C", date(2026, 10, 28))
    assert c.start == datetime(2026, 10, 28, 16, 30, tzinfo=UTC)   # 22:00 IST
    assert c.end == datetime(2026, 10, 29, 0, 30, tzinfo=UTC)      # 06:00 IST next day
    lunch = [b for b in c.breaks if b.kind == "lunch"][0]
    assert lunch.start == datetime(2026, 10, 28, 20, 30, tzinfo=UTC)  # 02:00 IST
    assert all(c.start <= b.start < c.end for b in c.breaks)


def test_every_instant_is_in_exactly_one_shift(schedule: ShiftSchedule) -> None:
    for minute in range(0, 48 * 60, 7):
        ts = SHIFT_A_START + timedelta(minutes=minute)
        covering = [i for i in schedule.instances_between(ts - timedelta(days=1), ts + timedelta(days=1))
                    if i.contains(ts)]
        assert len(covering) == 1, ts


# ---------------------------------------------------------------- machine

def test_running_is_derived_from_current_threshold(schedule: ShiftSchedule) -> None:
    m = _machine("M3", schedule)
    rated = 15.0
    for _, r in _run(m, SHIFT_A_START, 3 * 360):
        assert r["running"] == (r["current_a"] > DYN["running_threshold_fraction"] * rated)


def test_stopped_during_break_and_changeover(schedule: ShiftSchedule) -> None:
    for mid in ("M1", "M5", "M8"):
        m = _machine(mid, schedule)
        lag = DYN["stagger_max_s"]
        tea_rows = _run(m, TEA_1 + timedelta(seconds=lag), 40)       # inside the tea break
        changeover = _run(_machine(mid, schedule), SHIFT_A_START + timedelta(seconds=lag), 40)
        for _, r in tea_rows + changeover:
            assert r["running"] is False
            assert r["current_a"] < 0.5
            assert r["vib_rms_g"] < 0.05


def test_running_current_near_rated(schedule: ShiftSchedule) -> None:
    m = _machine("M1", schedule)  # rated 32 A
    rows = _run(m, SHIFT_A_START + timedelta(hours=1), 360)
    currents = [r["current_a"] for _, r in rows[1:] if r["running"]]  # skip inrush reading
    assert len(currents) == 359
    assert all(32.0 * 0.85 <= c <= 32.0 * 1.15 for c in currents)


def test_inrush_spike_at_start_up(schedule: ShiftSchedule) -> None:
    m = _machine("M2", schedule)  # rated 24 A
    rows = _run(m, SHIFT_A_START, 120)  # covers the 10-minute changeover and start-up
    first_on = next(cur for (_, prev), (_, cur) in zip(rows, rows[1:])
                    if cur["running"] and not prev["running"])
    assert first_on["current_a"] > 2.0 * 24.0


def test_no_inrush_when_already_running_at_simulation_start(schedule: ShiftSchedule) -> None:
    start = SHIFT_A_START + timedelta(hours=1)
    m = _machine("M2", schedule, start=start)
    r = m.reading(start)
    assert r["running"] and r["current_a"] < 24.0 * 1.15


def _mean_running_vib(m: MachineModel, start: datetime) -> float:
    rows = _run(m, start, 360)
    return float(np.mean([r["vib_rms_g"] for _, r in rows if r["running"]]))


def test_degrading_machine_vibration_rises_over_30_days(schedule: ShiftSchedule) -> None:
    m = _machine("M5", schedule)
    day0 = _mean_running_vib(m, SHIFT_A_START + timedelta(hours=1))
    day30 = _mean_running_vib(m, SHIFT_A_START + timedelta(days=30, hours=1))
    assert 2.2 <= day30 / day0 <= 2.8
    assert m.noise_energy_factor(SHIFT_A_START + timedelta(days=30)) == pytest.approx(2.5)


def test_non_degrading_machine_vibration_is_stable(schedule: ShiftSchedule) -> None:
    m = _machine("M6", schedule)
    day0 = _mean_running_vib(m, SHIFT_A_START + timedelta(hours=1))
    day30 = _mean_running_vib(m, SHIFT_A_START + timedelta(days=30, hours=1))
    assert 0.95 <= day30 / day0 <= 1.05
    assert m.noise_energy_factor(SHIFT_A_START + timedelta(days=30)) == 1.0


def test_machines_do_not_switch_at_the_same_second(schedule: ShiftSchedule) -> None:
    # Machines are still finishing shift C at 06:00 (their lag), so look for the
    # first off -> on transition after the changeover.
    first_on = []
    for mid in ("M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8"):
        rows = _run(_machine(mid, schedule, seed=int(mid[1:])), SHIFT_A_START, 120)
        first_on.append(next(ts for (_, prev), (ts, cur) in zip(rows, rows[1:])
                             if cur["running"] and not prev["running"]))
    assert len(set(first_on)) > 1
    assert all(ts >= SHIFT_A_START + timedelta(minutes=DYN["changeover_min"]) for ts in first_on)


def test_same_seed_reproduces_machine_sequence(schedule: ShiftSchedule) -> None:
    a = [r for _, r in _run(_machine("M4", schedule, seed=3), SHIFT_A_START, 200)]
    b = [r for _, r in _run(_machine("M4", schedule, seed=3), SHIFT_A_START, 200)]
    assert a == b


# ---------------------------------------------------------------- sensor faults (v1.1d)

def test_machine_sensor_faults_emit_null_with_flag(schedule: ShiftSchedule) -> None:
    m = _machine("M3", schedule, fault_rate_per_h=30.0)
    rows = [r for _, r in _run(m, SHIFT_A_START + timedelta(hours=1), 600)]
    seen = {(ch, r["sensor_flags"][ch]) for r in rows for ch in ("current", "vibration")}
    assert {("current", "stale"), ("current", "error"),
            ("vibration", "stale"), ("vibration", "error")} <= seen
    for r in rows:
        if r["sensor_flags"]["current"] == "ok":
            assert r["current_a"] is not None
        else:
            assert r["current_a"] is None
            assert isinstance(r["running"], bool)      # last decision is carried
        vib_ok = r["sensor_flags"]["vibration"] == "ok"
        assert (r["vib_rms_g"] is not None) == vib_ok
        assert (r["vib_peak_g"] is not None) == vib_ok


def test_current_fault_carries_last_running_decision(schedule: ShiftSchedule) -> None:
    m = _machine("M3", schedule, fault_rate_per_h=30.0)
    rows = [r for _, r in _run(m, SHIFT_A_START + timedelta(hours=1), 600)]
    for prev, cur in zip(rows, rows[1:]):
        if cur["current_a"] is None:
            assert cur["running"] == prev["running"]


def test_faults_do_not_change_physical_running_state(schedule: ShiftSchedule) -> None:
    clean = _machine("M3", schedule, seed=4)
    faulty = _machine("M3", schedule, seed=4, fault_rate_per_h=30.0)
    for i in range(300):
        ts = SHIFT_A_START + timedelta(hours=1) + i * DT
        clean.reading(ts)
        faulty.reading(ts)
        assert clean.running_now == faulty.running_now


def test_default_config_produces_occasional_faults(schedule: ShiftSchedule) -> None:
    m = MachineModel.from_config("M1", schedule, np.random.default_rng(0), SHIFT_A_START)
    rows = [r for _, r in _run(m, SHIFT_A_START, 6 * 60 * 24 * 7)]   # 7 days, ~7 faults expected
    faulted = sum(1 for r in rows if set(r["sensor_flags"].values()) != {"ok"})
    assert 0 < faulted < 0.1 * len(rows)
