"""Tests for ingest/localisation.py."""

import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from db.repo import Repo
from ingest.localisation import UNKNOWN, Params, TagTracker, count_changes, resolve, run_localisation

P = Params.from_config()
W = P.window_s
T0 = datetime(2026, 10, 28, 0, 0, tzinfo=timezone.utc)      # on the window grid
TAGS = {"BLE-0001": "W101"}


def at(i: int) -> datetime:
    """A time inside window i."""
    return T0 + timedelta(seconds=i * W + 5)


def run(windows: list[dict[str, float]], params: Params = P) -> list[dict]:
    """Resolve a sequence of per-window {zone: rssi} sightings for one tag."""
    sightings = [(at(i), zone, "BLE-0001", rssi) for i, heard in enumerate(windows) for zone, rssi in heard.items()]
    return resolve(sightings, TAGS, T0, T0 + timedelta(seconds=len(windows) * W), params)


def per_window(windows: list[dict[str, float]], params: Params = P) -> list[str]:
    """Zone decision for each window."""
    t = TagTracker(params)
    return [t.step(h)[0] for h in windows]


# ---------------------------------------------------------------- required cases

def test_clean_single_zone_signal_resolves_to_that_zone() -> None:
    zones = per_window([{"Z4": -62}] * 20)
    # Debounce applies from UNKNOWN too: the first (debounce - 1) windows stay UNKNOWN.
    assert zones[: P.debounce_windows - 1] == [UNKNOWN] * (P.debounce_windows - 1)
    assert zones[P.debounce_windows - 1:] == ["Z4"] * (20 - P.debounce_windows + 1)


def test_two_zones_with_clear_margin_picks_the_stronger() -> None:
    zones = per_window([{"Z4": -62, "Z5": -85}] * 20)
    assert set(zones[P.debounce_windows:]) == {"Z4"}


def test_two_zones_within_margin_do_not_flap() -> None:
    # Lock onto Z4, then the two nodes trade places by 4 dB every window (< 6 dB margin).
    lock = [{"Z4": -65}] * 5
    trading = [{"Z4": -65, "Z5": -61} if i % 2 else {"Z4": -61, "Z5": -65} for i in range(200)]
    zones = per_window(lock + trading)
    after_lock = zones[len(lock):]
    assert set(after_lock) == {"Z4"}
    changes = sum(1 for a, b in zip(after_lock, after_lock[1:]) if a != b)
    assert changes == 0


def test_naive_strongest_wins_would_have_flapped() -> None:
    """Control for the previous test: without smoothing/hysteresis/debounce it flaps."""
    naive = Params.from_config({"ewma_alpha": 1.0, "rssi_margin_db": 0.0, "debounce_windows": 1})
    trading = [{"Z4": -65, "Z5": -61} if i % 2 else {"Z4": -61, "Z5": -65} for i in range(200)]
    zones = per_window([{"Z4": -65}] * 5 + trading, naive)
    assert sum(1 for a, b in zip(zones, zones[1:]) if a != b) > 100


def test_disappearing_tag_becomes_unknown_after_exactly_absence_windows() -> None:
    seen = [{"Z4": -62}] * 10
    zones = per_window(seen + [{}] * 10)
    n = P.absence_windows
    assert zones[9] == "Z4"
    assert zones[10: 10 + n - 1] == ["Z4"] * (n - 1)       # carried for N-1 unseen windows
    assert zones[10 + n - 1] == UNKNOWN                    # the Nth unseen window
    assert set(zones[10 + n - 1:]) == {UNKNOWN}            # never carried forward again


def test_intervals_merge_correctly() -> None:
    windows = [{"Z4": -62}] * 10 + [{"Z5": -60}] * 10 + [{}] * 10
    rows = run(windows)
    assert [r["zone_id"] for r in rows] == [UNKNOWN, "Z4", "Z5", UNKNOWN]
    # Contiguous, on the grid, covering the whole span exactly.
    assert rows[0]["from_ts"] == "2026-10-28T00:00:00Z"
    assert rows[-1]["to_ts"] == (T0 + timedelta(seconds=30 * W)).strftime("%Y-%m-%dT%H:%M:%SZ")
    for a, b in zip(rows, rows[1:]):
        assert a["to_ts"] == b["from_ts"]
    z5 = rows[2]
    # Z5 wins from window 10 and takes over after `debounce` windows.
    assert z5["from_ts"] == (T0 + timedelta(seconds=(10 + P.debounce_windows - 1) * W)).strftime("%Y-%m-%dT%H:%M:%SZ")


def test_worker_never_has_overlapping_intervals() -> None:
    rnd = random.Random(7)
    tags = {f"BLE-{i:04d}": f"W{100 + i}" for i in range(1, 6)}
    sightings = []
    for i in range(400):
        for tag in tags:
            if rnd.random() < 0.85:
                for zone in rnd.sample(["Z1", "Z2", "Z3", "Z4"], k=rnd.randint(1, 3)):
                    sightings.append((at(i), zone, tag, rnd.uniform(-99, -50)))
    rows = resolve(sightings, tags, T0, T0 + timedelta(seconds=400 * W), P)
    by_worker: dict[str, list] = {}
    for r in rows:
        by_worker.setdefault(r["worker_id"], []).append(r)
    assert set(by_worker) == set(tags.values())
    for intervals in by_worker.values():
        for a, b in zip(intervals, intervals[1:]):
            assert a["to_ts"] == b["from_ts"]               # no overlap, no gap
            assert a["zone_id"] != b["zone_id"]             # fully merged
        assert all(r["from_ts"] < r["to_ts"] for r in intervals)


# ---------------------------------------------------------------- behaviour details

def test_a_single_stray_window_does_not_move_the_worker() -> None:
    windows = [{"Z4": -62}] * 10 + [{"Z5": -55}] + [{"Z4": -62}] * 10
    assert "Z5" not in per_window(windows)


def test_clear_move_follows_after_debounce() -> None:
    windows = [{"Z4": -62}] * 10 + [{"Z5": -60}] * 10
    zones = per_window(windows)
    first_z5 = zones.index("Z5")
    assert first_z5 == 10 + P.debounce_windows - 1


def test_weak_sightings_are_ignored_entirely() -> None:
    weak = P.min_rssi_dbm - 1
    zones = per_window([{"Z4": -62, "Z5": weak}] * 5 + [{"Z5": weak}] * 10)
    assert "Z5" not in zones
    assert zones[-1] == UNKNOWN                            # weak-only counts as absent


def test_tag_seen_by_three_nodes_picks_the_clear_leader() -> None:
    zones = per_window([{"Z3": -80, "Z4": -60, "Z5": -78}] * 10)
    assert set(zones[P.debounce_windows - 1:]) == {"Z4"}


def test_confidence_reflects_margin_and_is_bounded() -> None:
    t = TagTracker(P)
    for _ in range(5):
        _, clear = t.step({"Z4": -55, "Z5": -90})
    t2 = TagTracker(P)
    for _ in range(5):
        _, close = t2.step({"Z4": -62, "Z5": -66})
    assert clear == 1.0
    assert 0.0 < close < 0.5
    t3 = TagTracker(P)
    assert t3.step({})[1] == 0.0                           # UNKNOWN has zero confidence


def test_count_changes() -> None:
    rows = run([{"Z4": -62}] * 10 + [{"Z5": -60}] * 10)
    assert count_changes(rows) == {"W101": 2}             # UNKNOWN -> Z4 -> Z5


def test_parameters_are_recorded_on_every_row() -> None:
    rows = run([{"Z4": -62}] * 5)
    assert all(r["params"] == P.as_dict() for r in rows)


# ---------------------------------------------------------------- database pass

def test_run_localisation_writes_labelled_intervals_and_replaces_on_rerun(tmp_path: Path) -> None:
    repo = Repo(tmp_path / "l.db")
    repo.init_schema()
    for i in range(20):
        ts = (T0 + timedelta(seconds=i * W)).strftime("%Y-%m-%dT%H:%M:%SZ")
        for zone, tags in (("Z4", [{"tag_id": "BLE-0001", "rssi_dbm": -60, "seen_count": 20}]), ("Z5", [])):
            repo.insert_ble_scan(f"textile/zone/{zone}/ble",
                                 {"schema_version": "1.1", "device_id": f"ZN-{zone}", "ts": ts, "zone_id": zone,
                                  "scan_window_s": 30, "tags": tags}, "2026-10-28T06:00:00.000Z", [])
    repo.commit()
    summary = run_localisation(repo)
    first = repo.get_all_zone_assignments()
    assert summary["rows"] == len(first) > 0
    w101 = [r for r in first if r["worker_id"] == "W101"]
    assert [r["zone_id"] for r in w101] == [UNKNOWN, "Z4"]
    assert all(r["label"] == "estimated_personal" and r["method"] == "ble_ewma_hysteresis" for r in first)
    assert all(r["params"]["rssi_margin_db"] == P.rssi_margin_db for r in first)
    # Workers never seen are UNKNOWN for the whole span.
    w120 = [r for r in first if r["worker_id"] == "W120"]
    assert [r["zone_id"] for r in w120] == [UNKNOWN]
    run_localisation(repo)                                  # re-run replaces, never duplicates
    assert len(repo.get_all_zone_assignments()) == len(first)
    repo.close()
