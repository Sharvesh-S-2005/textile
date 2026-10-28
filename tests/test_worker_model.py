"""Tests for simulator/worker_model.py."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

import config
from simulator.schedule import ShiftSchedule
from simulator.worker_model import WorkerModel

UTC = timezone.utc
START = datetime(2026, 10, 28, 0, 30, tzinfo=UTC)   # 06:00 IST
END = START + timedelta(days=2)
DYN = config.get_simulation_config()["worker_dynamics"]
ZONE_IDS = {z["zone_id"] for z in config.get_zones()}


@pytest.fixture(scope="module")
def schedule() -> ShiftSchedule:
    return ShiftSchedule(config.get_shifts())


@pytest.fixture(scope="module")
def model(schedule: ShiftSchedule) -> WorkerModel:
    return WorkerModel.from_config(schedule, np.random.default_rng(42), START, END)


def test_timelines_are_contiguous_and_non_overlapping(model: WorkerModel) -> None:
    for wid, segs in model.timelines.items():
        for a, b in zip(segs, segs[1:]):
            assert a.end <= b.start, wid
        for s in segs:
            assert s.start < s.end and s.zone_id in ZONE_IDS


def test_every_on_site_worker_is_in_exactly_one_zone(model: WorkerModel) -> None:
    for minute in range(0, 48 * 60, 3):
        ts = START + timedelta(minutes=minute)
        for wid, segs in model.timelines.items():
            covering = [s for s in segs if s.start <= ts < s.end]
            assert len(covering) <= 1
            zone = model.zone_at(wid, ts)
            assert zone == (covering[0].zone_id if covering else None)


def test_workers_are_in_break_zone_during_breaks(model: WorkerModel, schedule: ShiftSchedule) -> None:
    high = set(DYN["high_exposure_workers"])
    for w in config.get_workers():
        for shift in schedule.instances_between(START, END, w["shift"]):
            for brk in shift.breaks:
                if w["worker_id"] in high and brk.kind == "tea":
                    continue  # the high-exposure case skips tea breaks by design
                for ts in (brk.start, brk.start + (brk.end - brk.start) / 2,
                           brk.end - timedelta(seconds=1)):
                    if START <= ts < END:
                        assert model.zone_at(w["worker_id"], ts) == DYN["break_zone"]


def test_check_in_precedes_check_out_and_shift_bounds(model: WorkerModel) -> None:
    by_worker: dict[str, list] = {}
    for e in model.rfid_events:
        by_worker.setdefault(e.worker_id, []).append(e)
    assert set(by_worker) == {w["worker_id"] for w in config.get_workers()}
    high = set(DYN["high_exposure_workers"])
    for wid, events in by_worker.items():
        kinds = [e.event for e in events]
        assert kinds == ["check_in", "check_out"] * (len(kinds) // 2), wid
        max_ot = DYN["high_exposure"]["overtime_min"]["max"] if wid in high \
            else DYN["overtime"]["duration_min"]["max"]
        slack = DYN["check_in_before_min"]["max"] + DYN["check_out_after_min"]["max"]
        for cin, cout in zip(events[::2], events[1::2]):
            on_site = cout.ts - cin.ts
            assert timedelta(hours=8) < on_site <= timedelta(hours=8, minutes=max_ot + slack)


def test_worker_is_on_site_only_between_check_in_and_check_out(model: WorkerModel) -> None:
    for e in model.rfid_events:
        if e.event == "check_in":
            assert model.zone_at(e.worker_id, e.ts - timedelta(seconds=1)) is None
            assert model.zone_at(e.worker_id, e.ts) == DYN["break_zone"]
        else:
            assert model.zone_at(e.worker_id, e.ts) is None


def test_workers_mostly_in_home_zone_with_some_trips(model: WorkerModel) -> None:
    for w in config.get_workers():
        segs = model.timelines[w["worker_id"]]
        time_in = {}
        for s in segs:
            time_in[s.zone_id] = time_in.get(s.zone_id, timedelta()) + (s.end - s.start)
        total = sum(time_in.values(), timedelta())
        assert time_in[w["home_zone"]] / total > 0.6, w["worker_id"]
        allowed = {w["home_zone"], DYN["break_zone"], *config.get_adjacency()[w["home_zone"]]}
        assert set(time_in) <= allowed, w["worker_id"]


def test_high_exposure_worker_has_most_time_in_z5(model: WorkerModel) -> None:
    def hours_in_z5(wid: str) -> float:
        return sum((s.end - s.start).total_seconds() for s in model.timelines[wid]
                   if s.zone_id == "Z5") / 3600

    z5_workers = [w["worker_id"] for w in config.get_workers() if w["home_zone"] == "Z5"]
    hours = {wid: hours_in_z5(wid) for wid in z5_workers}
    for wid in DYN["high_exposure_workers"]:
        assert hours[wid] == max(hours.values())
        assert hours[wid] / 2 > 9.0   # per day: > 9 h in the loom shed


def test_ppe_event_per_worker_shift_with_allowed_values(model: WorkerModel) -> None:
    allowed = set(config.get_validation_config()["allowed_ppe_values"])
    check_ins = [e for e in model.rfid_events if e.event == "check_in"]
    assert len(model.ppe_events) == len(check_ins)
    for e in model.ppe_events:
        assert e.hearing_ppe in allowed and e.respiratory_ppe in allowed
        assert e.recorded_by in DYN["supervisors"].values()


def test_ppe_compliance_drives_ppe_values(schedule: ShiftSchedule) -> None:
    long_end = START + timedelta(days=60)
    m = WorkerModel.from_config(schedule, np.random.default_rng(1), START, long_end)
    def worn_rate(wid: str) -> float:
        ev = [e for e in m.ppe_events if e.worker_id == wid]
        return np.mean([e.hearing_ppe == "worn" for e in ev])
    assert worn_rate("W108") > worn_rate("W105") + 0.4   # 0.95 vs 0.30 compliance


def test_seeded_run_is_reproducible(schedule: ShiftSchedule) -> None:
    a = WorkerModel.from_config(schedule, np.random.default_rng(9), START, END)
    b = WorkerModel.from_config(schedule, np.random.default_rng(9), START, END)
    c = WorkerModel.from_config(schedule, np.random.default_rng(10), START, END)
    assert a.timelines == b.timelines
    assert a.rfid_events == b.rfid_events and a.ppe_events == b.ppe_events
    assert a.timelines != c.timelines


def test_same_second_tap_pairs_check_in_within_one_second(model: WorkerModel) -> None:
    check_ins = {}
    for e in model.rfid_events:
        if e.event == "check_in":
            check_ins.setdefault(e.worker_id, []).append(e.ts)
    for first, second in DYN["same_second_tap_pairs"]:
        assert check_ins[first] and len(check_ins[first]) == len(check_ins[second])
        for a, b in zip(check_ins[first], check_ins[second]):
            assert a.replace(microsecond=0) == b.replace(microsecond=0)
            assert a != b
