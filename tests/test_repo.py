"""Tests for db/schema.sql, db/repo.py and db/init_db.py."""

import contextlib
import io
import json
import re
from pathlib import Path

import pytest

from db import init_db as init_db_module
from db.repo import Repo
from simulator import publisher

RX = "2026-10-28T06:00:00Z"   # a received_ts for fixtures


@pytest.fixture
def repo(tmp_path: Path) -> Repo:
    r = Repo(tmp_path / "test.db")
    r.init_schema()
    yield r
    r.close()


def _env(ts: str, zone: str = "Z1", noise: float | None = 91.0, noise_flag: str = "ok") -> dict:
    return {
        "schema_version": "1.1", "device_id": f"ZN-{zone}", "ts": ts, "zone_id": zone,
        "noise_dba": noise, "noise_peak_dba": None if noise is None else noise + 4,
        "pm25_ugm3": 250.0, "pm10_ugm3": 500.0, "temp_c": 33.0, "rh_pct": 65.0,
        "globe_temp_c": 35.5, "co2_ppm": 800,
        "sensor_flags": {"noise": noise_flag, "pm": "ok", "temp_rh": "ok", "globe": "ok", "co2": "ok"},
    }


def _machine(ts: str, current: float | None = 15.0, current_flag: str = "ok") -> dict:
    return {
        "schema_version": "1.1", "device_id": "MN-M3", "ts": ts, "machine_id": "M3", "zone_id": "Z4",
        "current_a": current, "running": True, "vib_rms_g": 0.42, "vib_peak_g": 1.07,
        "sensor_flags": {"current": current_flag, "vibration": "ok"},
    }


def _ble(ts: str, tags: list[dict], zone: str = "Z4") -> dict:
    return {"schema_version": "1.1", "device_id": f"ZN-{zone}", "ts": ts, "zone_id": zone,
            "scan_window_s": 30, "tags": tags}


def _gate(ts: str, event_id: str, event: str = "check_in") -> dict:
    return {"schema_version": "1.1", "device_id": "GW-01", "ts": ts, "event_id": event_id,
            "card_id": "RFID-10001", "event": event, "gate_id": "GATE-A"}


def _ppe(ts: str, event_id: str) -> dict:
    return {"schema_version": "1.1", "device_id": "SUP-APP", "ts": ts, "event_id": event_id,
            "worker_id": "W101", "hearing_ppe": "worn", "respiratory_ppe": "partial",
            "recorded_by": "SUP-01"}


# ---------------------------------------------------------------- schema

def test_schema_creates_cleanly_with_wal_and_foreign_keys(repo: Repo) -> None:
    assert repo.journal_mode() == "wal"
    assert repo.foreign_keys_enabled()
    assert all(n == 0 for n in repo.table_counts().values())


def test_init_db_is_safe_to_rerun_and_reset_clears(tmp_path: Path) -> None:
    db = tmp_path / "x.db"
    init_db_module.init_db(db)
    with Repo(db) as r:
        r.insert_zone_env("textile/zone/Z1/env", _env("2026-10-28T00:30:00Z"), RX, [])
    init_db_module.init_db(db)                       # re-run keeps data
    with Repo(db) as r:
        assert r.table_counts()["zone_env"] == 1
    init_db_module.init_db(db, reset=True)           # explicit reset clears it
    with Repo(db) as r:
        assert r.table_counts()["zone_env"] == 0


# ---------------------------------------------------------------- round trips

def test_zone_env_round_trip_keeps_nulls(repo: Repo) -> None:
    topic = "textile/zone/Z1/env"
    assert repo.insert_zone_env(topic, _env("2026-10-28T00:30:00Z"), RX, [])
    assert repo.insert_zone_env(topic, _env("2026-10-28T00:30:10Z", noise=None, noise_flag="error"),
                                RX, ["sensor_not_ok"])
    rows = repo.get_zone_env("Z1", "2026-10-28T00:00:00Z", "2026-10-28T01:00:00Z")
    assert [r["noise_dba"] for r in rows] == [91.0, None]
    assert rows[1]["noise_peak_dba"] is None            # NULL, never a substituted 0
    assert rows[1]["flag_noise"] == "error"
    assert rows[1]["quality"] == "flagged" and rows[1]["flags"] == ["sensor_not_ok"]
    assert rows[0]["quality"] == "ok" and rows[0]["flags"] == []
    assert rows[0]["co2_ppm"] == 800 and rows[0]["received_ts"] == RX


def test_machine_round_trip_keeps_nulls(repo: Repo) -> None:
    topic = "textile/machine/M3/status"
    repo.insert_machine_status(topic, _machine("2026-10-28T00:30:00Z", current=None, current_flag="stale"),
                               RX, ["sensor_not_ok"])
    (row,) = repo.get_machine_status("M3", "2026-10-28T00:00:00Z", "2026-10-28T01:00:00Z")
    assert row["current_a"] is None and row["running"] == 1 and row["flag_current"] == "stale"


def test_ble_scan_round_trip_including_empty_scan(repo: Repo) -> None:
    tags = [{"tag_id": "BLE-0007", "rssi_dbm": -58, "seen_count": 11},
            {"tag_id": "BLE-0012", "rssi_dbm": -79, "seen_count": 4}]
    assert repo.insert_ble_scan("textile/zone/Z4/ble", _ble("2026-10-28T00:30:00Z", tags), RX, [])
    assert repo.insert_ble_scan("textile/zone/Z5/ble", _ble("2026-10-28T00:30:00Z", [], "Z5"), RX, [])
    scans = repo.get_ble_scans("2026-10-28T00:00:00Z", "2026-10-28T01:00:00Z")
    assert [(s["zone_id"], s["n_tags"]) for s in scans] == [("Z4", 2), ("Z5", 0)]
    sightings = repo.get_ble_sightings("2026-10-28T00:00:00Z", "2026-10-28T01:00:00Z")
    assert [(s["tag_id"], s["rssi_dbm"], s["seen_count"]) for s in sightings] == [
        ("BLE-0007", -58, 11), ("BLE-0012", -79, 4)]
    assert all(s["scan_id"] == scans[0]["id"] for s in sightings)


def test_event_round_trips(repo: Repo) -> None:
    eid = "3f2b8c1e-9a4d-4e7b-8c21-5d6f0a9e1b73"
    assert repo.insert_gate_event("textile/gate/rfid", _gate("2026-10-28T00:20:00Z", eid), RX, [], "W101")
    assert repo.insert_ppe_entry("textile/ppe/entry", _ppe("2026-10-28T00:45:00Z", "a7c41e02-6b3f-4d9a-9e58-0f2d7c1b4a96"), RX, [])
    (g,) = repo.get_gate_events()
    assert (g["event_id"], g["worker_id"], g["event"]) == (eid, "W101", "check_in")
    assert repo.table_counts()["ppe_entries"] == 1


def test_device_status_lwt_has_null_ts_and_is_not_duplicated(repo: Repo) -> None:
    topic = "textile/status/ZN-Z4"
    birth = {"schema_version": "1.1", "device_id": "ZN-Z4", "ts": "2026-10-28T00:00:00Z", "online": True,
             "firmware": "sim-0.1.0", "uptime_s": 0, "buffered_records": 0}
    lwt = {"schema_version": "1.1", "device_id": "ZN-Z4", "online": False}
    assert repo.insert_device_status(topic, birth, RX, [])
    assert repo.insert_device_status(topic, lwt, RX, [])
    assert not repo.insert_device_status(topic, lwt, "2026-10-28T06:05:00Z", [])   # retained redelivery
    assert not repo.insert_device_status(topic, birth, RX, [])                     # duplicate birth
    birth2 = {**birth, "ts": "2026-10-28T01:00:00Z"}
    assert repo.insert_device_status(topic, birth2, RX, [])
    assert repo.insert_device_status(topic, lwt, RX, [])                           # a new disconnect
    assert repo.table_counts()["device_status"] == 4


# ---------------------------------------------------------------- dedup and order

def test_duplicate_inserts_are_ignored(repo: Repo) -> None:
    env, eid = _env("2026-10-28T00:30:00Z"), "3f2b8c1e-9a4d-4e7b-8c21-5d6f0a9e1b73"
    assert repo.insert_zone_env("textile/zone/Z1/env", env, RX, [])
    assert not repo.insert_zone_env("textile/zone/Z1/env", env, "2026-10-28T06:01:00Z", [])
    assert repo.insert_gate_event("textile/gate/rfid", _gate("2026-10-28T00:20:00Z", eid), RX, [], "W101")
    # Same event_id at a different ts is still the same event.
    assert not repo.insert_gate_event("textile/gate/rfid", _gate("2026-10-28T00:20:05Z", eid), RX, [], "W101")
    assert repo.insert_rejected("textile/zone/Z1/env", b"{bad", "malformed_json", "x", RX)
    assert not repo.insert_rejected("textile/zone/Z1/env", b"{bad", "malformed_json", "x", RX)
    counts = repo.table_counts()
    assert (counts["zone_env"], counts["gate_events"], counts["rejected_records"]) == (1, 1, 1)


def test_two_gate_events_in_the_same_second_are_both_kept(repo: Repo) -> None:
    ts = "2026-10-28T00:20:00Z"
    assert repo.insert_gate_event("textile/gate/rfid", _gate(ts, "11111111-1111-4111-8111-111111111111"), RX, [], "W101")
    assert repo.insert_gate_event("textile/gate/rfid", _gate(ts, "22222222-2222-4222-8222-222222222222"), RX, [], "W102")
    assert repo.table_counts()["gate_events"] == 2


def test_out_of_order_inserts_read_back_in_ts_order(repo: Repo) -> None:
    for ts in ("2026-10-28T00:30:30Z", "2026-10-28T00:30:00Z", "2026-10-28T00:30:20Z", "2026-10-28T00:30:10Z"):
        repo.insert_zone_env("textile/zone/Z1/env", _env(ts), RX, [])
        repo.insert_machine_status("textile/machine/M3/status", _machine(ts), RX, [])
    expected = ["2026-10-28T00:30:00Z", "2026-10-28T00:30:10Z", "2026-10-28T00:30:20Z", "2026-10-28T00:30:30Z"]
    assert [r["ts"] for r in repo.get_zone_env("Z1", "2026-10-28T00:00:00Z", "2026-10-28T01:00:00Z")] == expected
    assert [r["ts"] for r in repo.get_machine_status("M3", "2026-10-28T00:00:00Z", "2026-10-28T01:00:00Z")] == expected


def test_time_range_is_half_open(repo: Repo) -> None:
    for ts in ("2026-10-28T00:30:00Z", "2026-10-28T00:31:00Z"):
        repo.insert_zone_env("textile/zone/Z1/env", _env(ts), RX, [])
    rows = repo.get_zone_env("Z1", "2026-10-28T00:30:00Z", "2026-10-28T00:31:00Z")
    assert [r["ts"] for r in rows] == ["2026-10-28T00:30:00Z"]


# ---------------------------------------------------------------- stats

def test_ingest_stats_on_known_fixture(repo: Repo) -> None:
    repo.insert_zone_env("textile/zone/Z1/env", _env("2026-10-28T00:30:00Z"), RX, [])
    repo.insert_zone_env("textile/zone/Z1/env", _env("2026-10-28T00:30:10Z", noise=None, noise_flag="stale"),
                         RX, ["sensor_not_ok"])
    repo.insert_zone_env("textile/zone/Z1/env", _env("2026-10-28T00:30:20Z"), RX, ["spike", "stuck_value"])
    repo.insert_machine_status("textile/machine/M3/status", _machine("2026-10-28T00:30:00Z"), RX, [])
    repo.insert_ble_scan("textile/zone/Z4/ble", _ble("2026-10-28T00:30:00Z", []), RX, [])
    repo.insert_rejected("textile/zone/Z1/env", b"{bad", "malformed_json", "x", RX)
    repo.insert_rejected("textile/zone/Z1/env", b"{}", "missing_field", "x", RX)
    repo.insert_rejected("textile/zone/Z2/env", b"{} ", "missing_field", "x", RX)
    repo.insert_rejected("textile/zone/Z2/env", b"old", "missing_field", "x", "2026-10-27T00:00:00Z")  # outside range
    stats = repo.get_ingest_stats("2026-10-28T00:00:00Z", "2026-10-29T00:00:00Z")
    assert stats["accepted"] == {"env": 3, "ble": 1, "machine": 1, "rfid": 0, "ppe": 0, "status": 0}
    assert stats["flagged"]["env"] == 2 and stats["flagged_total"] == 2
    assert stats["flag_counts"] == {"sensor_not_ok": 1, "spike": 1, "stuck_value": 1}
    assert stats["rejected"] == {"malformed_json": 1, "missing_field": 2}
    assert stats["rejected_total"] == 3
    assert stats["last_seen"]["ZN-Z1"] == {"last_received_ts": RX, "last_ts": "2026-10-28T00:30:20Z"}
    assert set(stats["last_seen"]) == {"ZN-Z1", "ZN-Z4", "MN-M3"}


# ---------------------------------------------------------------- idempotency on a real batch

def _captured_batch() -> list[tuple[str, dict]]:
    """A real simulator batch: 30 simulated minutes including an outage replay."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert publisher.main(["--dry-run", "--speed", "1e9", "--duration-h", "0.5",
                               "--start", "2026-10-28T05:30:00", "--outage-start-min", "10",
                               "--outage-duration-min", "3"]) == 0
    out = []
    for line in buf.getvalue().splitlines():
        topic, _, body = line.partition(" ")
        out.append((topic, json.loads(body)))
    return out


def _store(repo: Repo, batch: list[tuple[str, dict]]) -> None:
    for topic, p in batch:
        parts = topic.split("/")
        if parts[1] == "zone" and parts[3] == "env":
            repo.insert_zone_env(topic, p, RX, [])
        elif parts[1] == "zone" and parts[3] == "ble":
            repo.insert_ble_scan(topic, p, RX, [])
        elif parts[1] == "machine":
            repo.insert_machine_status(topic, p, RX, [])
        elif parts[1] == "gate":
            repo.insert_gate_event(topic, p, RX, [], None)
        elif parts[1] == "ppe":
            repo.insert_ppe_entry(topic, p, RX, [])
        else:
            repo.insert_device_status(topic, p, RX, [])
    repo.commit()


def test_reingesting_the_same_batch_changes_no_counts(repo: Repo) -> None:
    batch = _captured_batch()
    _store(repo, batch)
    first = repo.table_counts()
    assert first["zone_env"] > 0 and first["ble_sightings"] > 0 and first["gate_events"] > 0
    _store(repo, batch)
    assert repo.table_counts() == first


# ---------------------------------------------------------------- rule: no SQL outside db/repo.py

def test_no_sql_outside_db_repo() -> None:
    root = Path(__file__).resolve().parents[1]
    verbs = "|".join(["SEL" + "ECT", "INS" + "ERT", "UPD" + "ATE", "DEL" + "ETE", "CRE" + "ATE TABLE"])
    pattern = re.compile(rf"\b({verbs})\b[^\n]*\b(FROM|INTO|SET|WHERE|IF)\b")
    offenders = []
    for path in root.rglob("*.py"):
        rel = path.relative_to(root).as_posix()
        if rel.startswith((".venv/", "venv/")) or rel == "db/repo.py":
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if pattern.search(line):
                offenders.append(f"{rel}:{lineno}: {line.strip()}")
    assert not offenders, offenders


def test_savepoint_rolls_back_only_the_failing_message(repo: Repo) -> None:
    topic = "textile/zone/Z1/env"
    with repo.savepoint():
        repo.insert_zone_env(topic, _env("2026-10-28T00:30:00Z"), RX, [])
    with pytest.raises(RuntimeError):
        with repo.savepoint():
            repo.insert_zone_env(topic, _env("2026-10-28T00:30:10Z"), RX, [])
            raise RuntimeError("boom")
    with repo.savepoint():
        repo.insert_zone_env(topic, _env("2026-10-28T00:30:20Z"), RX, [])
    repo.rollback()   # nothing was committed yet: savepoints must not commit per message
    assert repo.table_counts()["zone_env"] == 0
    for ts in ("2026-10-28T00:30:00Z", "2026-10-28T00:30:10Z"):
        with repo.savepoint():
            repo.insert_zone_env(topic, _env(ts), RX, [])
    with pytest.raises(RuntimeError):
        with repo.savepoint():
            repo.insert_zone_env(topic, _env("2026-10-28T00:30:20Z"), RX, [])
            raise RuntimeError("boom")
    repo.commit()
    assert [r["ts"] for r in repo.get_zone_env("Z1", "2026-10-28T00:00:00Z", "2026-10-29T00:00:00Z")] == [
        "2026-10-28T00:30:00Z", "2026-10-28T00:30:10Z"]
