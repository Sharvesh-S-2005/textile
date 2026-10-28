"""Tests for ingest/sessions.py."""

from pathlib import Path

import pytest

import config
from db.repo import Repo
from ingest.sessions import SessionFlag, build_sessions, clip_assignments, run_sessions
from shifts import ShiftSchedule

SCHEDULE = ShiftSchedule(config.get_shifts())
SHIFT = {w["worker_id"]: w["shift"] for w in config.get_workers()}
MAX_H = config.get_sessions_config()["max_session_h"]


def ev(ts: str, event: str, worker: str = "W101", n: int = 0) -> dict:
    return {"ts": ts, "event": event, "worker_id": worker, "event_id": f"e-{worker}-{ts}-{n}",
            "card_id": "RFID-x"}


def build(events: list[dict]) -> list[dict]:
    return build_sessions(events, SCHEDULE, SHIFT, MAX_H)


# W101 is on shift A: 06:00-14:00 IST = 00:30-08:30 UTC.

def test_normal_pair_makes_one_clean_session() -> None:
    (s,) = build([ev("2026-10-28T00:20:00Z", "check_in"), ev("2026-10-28T08:35:00Z", "check_out")])
    assert (s["session_start"], s["session_end"]) == ("2026-10-28T00:20:00Z", "2026-10-28T08:35:00Z")
    assert s["duration_s"] == (8 * 3600 + 15 * 60) and s["shift"] == "A" and s["flags"] == []


def test_events_out_of_order_are_paired_by_ts() -> None:
    (s,) = build([ev("2026-10-28T08:35:00Z", "check_out"), ev("2026-10-28T00:20:00Z", "check_in")])
    assert s["flags"] == []


def test_unmatched_check_in_is_closed_at_shift_end_and_flagged() -> None:
    (s,) = build([ev("2026-10-28T00:20:00Z", "check_in")])
    assert s["session_end"] == "2026-10-28T08:30:00Z"     # end of that shift A instance
    assert s["flags"] == [SessionFlag.UNMATCHED_CHECK_IN]


def test_orphan_check_out_is_recorded_and_flagged() -> None:
    (s,) = build([ev("2026-10-28T08:35:00Z", "check_out")])
    assert s["session_start"] is None and s["duration_s"] is None
    assert s["session_end"] == "2026-10-28T08:35:00Z"
    assert s["flags"] == [SessionFlag.ORPHAN_CHECK_OUT]


def test_two_consecutive_check_ins_are_flagged_duplicate() -> None:
    (s,) = build([ev("2026-10-28T00:20:00Z", "check_in"), ev("2026-10-28T00:20:03Z", "check_in", n=1),
                  ev("2026-10-28T08:35:00Z", "check_out")])
    assert s["session_start"] == "2026-10-28T00:20:00Z"   # first tap kept
    assert s["flags"] == [SessionFlag.DUPLICATE_CHECK_IN]


def test_implausibly_long_session_is_flagged() -> None:
    (s,) = build([ev("2026-10-28T00:20:00Z", "check_in"), ev("2026-10-28T16:30:00Z", "check_out")])
    assert s["flags"] == [SessionFlag.IMPLAUSIBLE_DURATION]


def test_unknown_card_events_are_not_paired() -> None:
    events = [ev("2026-10-28T00:20:00Z", "check_in"), {**ev("2026-10-28T00:21:00Z", "check_in"), "worker_id": None}]
    (s,) = build(events)
    assert s["worker_id"] == "W101"


def test_night_shift_session_crossing_midnight() -> None:
    # W116 is on shift C: 22:00-06:00 IST = 16:30-00:30 UTC.
    (s,) = build([ev("2026-10-28T16:20:00Z", "check_in", "W116"), ev("2026-10-29T00:40:00Z", "check_out", "W116")])
    assert s["shift"] == "C" and s["flags"] == []
    (u,) = build([ev("2026-10-28T16:20:00Z", "check_in", "W116")])
    assert u["session_end"] == "2026-10-29T00:30:00Z"


# ---------------------------------------------------------------- clipping

def _a(from_ts: str, to_ts: str, zone: str = "Z1", worker: str = "W101") -> dict:
    return {"worker_id": worker, "tag_id": "BLE-0001", "zone_id": zone, "from_ts": from_ts, "to_ts": to_ts,
            "confidence": 0.9, "params": {"x": 1}}


SESSION = {"worker_id": "W101", "session_start": "2026-10-28T00:20:00Z", "session_end": "2026-10-28T08:35:00Z"}


def test_clipping_marks_out_of_session_assignments() -> None:
    rows = clip_assignments([_a("2026-10-28T00:00:00Z", "2026-10-28T00:10:00Z"),     # before
                             _a("2026-10-28T01:00:00Z", "2026-10-28T02:00:00Z"),     # inside
                             _a("2026-10-28T09:00:00Z", "2026-10-28T09:30:00Z")],    # after
                            [SESSION])
    assert [r["outside_session"] for r in rows] == [True, False, True]


def test_clipping_splits_an_assignment_that_crosses_a_boundary() -> None:
    rows = clip_assignments([_a("2026-10-28T00:10:00Z", "2026-10-28T00:40:00Z")], [SESSION])
    assert [(r["from_ts"], r["to_ts"], r["outside_session"]) for r in rows] == [
        ("2026-10-28T00:10:00Z", "2026-10-28T00:20:00Z", True),
        ("2026-10-28T00:20:00Z", "2026-10-28T00:40:00Z", False)]


def test_orphan_check_out_covers_nothing() -> None:
    orphan = {"worker_id": "W101", "session_start": None, "session_end": "2026-10-28T08:35:00Z"}
    (r,) = clip_assignments([_a("2026-10-28T01:00:00Z", "2026-10-28T02:00:00Z")], [orphan])
    assert r["outside_session"] is True


def test_sessions_of_other_workers_do_not_count() -> None:
    (r,) = clip_assignments([_a("2026-10-28T01:00:00Z", "2026-10-28T02:00:00Z", worker="W102")], [SESSION])
    assert r["outside_session"] is True


def test_run_sessions_writes_sessions_and_clips_stored_assignments(tmp_path: Path) -> None:
    repo = Repo(tmp_path / "s.db")
    repo.init_schema()
    for i, (ts, event) in enumerate([("2026-10-28T00:20:00Z", "check_in"), ("2026-10-28T08:35:00Z", "check_out")]):
        repo.insert_gate_event("textile/gate/rfid", {
            "schema_version": "1.1", "device_id": "GW-01", "ts": ts,
            "event_id": f"{i}1111111-1111-4111-8111-111111111111", "card_id": "RFID-10001",
            "event": event, "gate_id": "GATE-A"}, "2026-10-28T09:00:00.000Z", [], "W101")
    repo.replace_zone_assignments("ble_ewma_hysteresis", [
        _a("2026-10-28T00:00:00Z", "2026-10-28T00:30:00Z"), _a("2026-10-28T00:30:00Z", "2026-10-28T09:00:00Z", "Z2")])
    repo.commit()
    summary = run_sessions(repo)
    assert summary["sessions"] == 1 and summary["flags"] == {}
    (s,) = repo.get_worker_sessions("W101", "2026-10-28")
    assert s["duration_s"] == 8 * 3600 + 15 * 60
    rows = repo.get_all_zone_assignments()
    assert [(r["from_ts"], r["outside_session"]) for r in rows] == [
        ("2026-10-28T00:00:00Z", 1), ("2026-10-28T00:20:00Z", 0), ("2026-10-28T00:30:00Z", 0),
        ("2026-10-28T08:35:00Z", 1)]
    run_sessions(repo)                                   # re-running is stable
    assert len(repo.get_all_zone_assignments()) == len(rows)
    repo.close()
