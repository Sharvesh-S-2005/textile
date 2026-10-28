"""Tests for ingest/subscriber.py using a fake MQTT client (no broker needed)."""

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from db.repo import Repo
from ingest.subscriber import Ingestor, Message, Subscriber, TimeReference

TS = "2026-10-28T04:48:10Z"
T0 = datetime(2026, 10, 28, 4, 48, 30, tzinfo=timezone.utc)


@dataclass
class FakeMsg:
    topic: str
    payload: bytes
    retain: bool = False
    mid: int = 0
    qos: int = 1


class FakeClient:
    """Just enough of paho's Client API for Subscriber."""

    def __init__(self) -> None:
        self.on_connect = self.on_disconnect = self.on_message = None
        self.subscriptions: list[tuple[str, int]] = []
        self.acked: list[int] = []
        self.connected = self.loop_running = False
        self._mid = 0

    def reconnect_delay_set(self, min_delay: int, max_delay: int) -> None:
        self.backoff = (min_delay, max_delay)

    def connect_async(self, host: str, port: int, keepalive: int) -> None:
        self.target = (host, port)

    def loop_start(self) -> None:
        self.loop_running = True
        self.connected = True
        self.on_connect(self, None, None, 0, None)

    def loop_stop(self) -> None:
        self.loop_running = False

    def disconnect(self) -> None:
        self.connected = False

    def subscribe(self, topic: str, qos: int) -> None:
        self.subscriptions.append((topic, qos))

    def ack(self, mid: int, qos: int) -> None:
        self.acked.append(mid)

    def deliver(self, topic: str, payload: Any, retain: bool = False) -> int:
        """Simulate the network thread delivering a message."""
        self._mid += 1
        raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        self.on_message(self, None, FakeMsg(topic, raw, retain, self._mid, 1))
        return self._mid


def env(ts: str = TS, zone: str = "Z4", **changes: Any) -> dict:
    p = {"schema_version": "1.1", "device_id": f"ZN-{zone}", "ts": ts, "zone_id": zone,
         "noise_dba": 96.4, "noise_peak_dba": 103.1, "pm25_ugm3": 118.2, "pm10_ugm3": 240.7,
         "temp_c": 33.8, "rh_pct": 72.5, "globe_temp_c": 36.2, "co2_ppm": 980,
         "sensor_flags": {"noise": "ok", "pm": "ok", "temp_rh": "ok", "globe": "ok", "co2": "ok"}}
    p.update(changes)
    return p


GOOD = [
    ("textile/zone/Z4/env", env()),
    ("textile/zone/Z4/ble", {"schema_version": "1.1", "device_id": "ZN-Z4", "ts": TS, "zone_id": "Z4",
                             "scan_window_s": 30, "tags": [{"tag_id": "BLE-0007", "rssi_dbm": -58, "seen_count": 11}]}),
    ("textile/machine/M3/status", {"schema_version": "1.1", "device_id": "MN-M3", "ts": TS, "machine_id": "M3",
                                   "zone_id": "Z4", "current_a": None, "running": True, "vib_rms_g": 0.42,
                                   "vib_peak_g": 1.07, "sensor_flags": {"current": "stale", "vibration": "ok"}}),
    ("textile/gate/rfid", {"schema_version": "1.1", "device_id": "GW-01", "ts": TS,
                           "event_id": "3f2b8c1e-9a4d-4e7b-8c21-5d6f0a9e1b73", "card_id": "RFID-10005",
                           "event": "check_in", "gate_id": "GATE-A"}),
    ("textile/ppe/entry", {"schema_version": "1.1", "device_id": "SUP-APP", "ts": TS,
                           "event_id": "a7c41e02-6b3f-4d9a-9e58-0f2d7c1b4a96", "worker_id": "W105",
                           "hearing_ppe": "not_worn", "respiratory_ppe": "partial", "recorded_by": "SUP-01"}),
    ("textile/status/ZN-Z4", {"schema_version": "1.1", "device_id": "ZN-Z4", "ts": TS, "online": True,
                              "firmware": "sim-0.1.0", "uptime_s": 10, "buffered_records": 0}),
    ("textile/status/MN-M3", {"schema_version": "1.1", "device_id": "MN-M3", "online": False}),   # LWT form
]
BAD = [
    ("textile/zone/Z4/env", b"{not json", "malformed_json"),
    ("textile/zone/Z4/env", env(zone_id="Z5"), "topic_mismatch"),
    ("textile/zone/Z4/env", env(noise_dba=200.0), "out_of_range"),
    ("textile/zone/Z4/env", env(schema_version="1.0"), "schema_version"),
    ("textile/zone/Z4/env", env(pm25_ugm3=None), "null_with_ok_flag"),
    ("textile/elsewhere", env(), "unknown_topic"),
]


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


@pytest.fixture
def setup(tmp_path: Path):
    repo = Repo(tmp_path / "ingest.db")
    repo.init_schema()
    clock = FakeClock()
    ingestor = Ingestor(repo, TimeReference("stream"), batch_size=500, batch_interval_s=1.0, clock=clock)
    client = FakeClient()
    sub = Subscriber(ingestor, client, qos=1, stats_interval_s=3600, poll_s=0.0, clock=clock)
    sub.start("localhost", 1883, 60, 1, 30)
    yield repo, ingestor, client, sub, clock
    repo.close()


def test_subscribes_to_everything_at_qos1(setup) -> None:
    _, _, client, _, _ = setup
    assert client.subscriptions == [("textile/#", 1)]
    assert client.backoff == (1, 30)


def test_every_topic_type_and_bad_messages_land_correctly(setup) -> None:
    repo, ingestor, client, sub, _ = setup
    for topic, payload in GOOD:
        client.deliver(topic, payload)
    for topic, payload, _ in BAD:
        client.deliver(topic, payload)
    sub.shutdown()                                      # flushes the open batch

    counts = repo.table_counts()
    assert (counts["zone_env"], counts["ble_scans"], counts["ble_sightings"], counts["machine_status"],
            counts["gate_events"], counts["ppe_entries"], counts["device_status"]) == (1, 1, 1, 1, 1, 1, 2)
    rejected = repo.get_rejected()
    assert sorted(r["reason"] for r in rejected) == sorted(reason for _, _, reason in BAD)
    malformed = next(r for r in rejected if r["reason"] == "malformed_json")
    assert malformed["raw_payload"] == b"{not json"      # exact bytes kept
    (machine,) = repo.get_machine_status("M3", "2026-10-28T00:00:00Z", "2026-10-29T00:00:00Z")
    assert machine["current_a"] is None and machine["quality"] == "flagged"
    assert machine["flags"] == ["sensor_not_ok"]
    (gate,) = repo.get_gate_events()
    assert gate["worker_id"] == "W105"                  # card_id resolved via config
    assert ingestor.counts["received"] == len(GOOD) + len(BAD)
    assert ingestor.counts["stored"] + ingestor.counts["rejected"] == len(GOOD) + len(BAD)


def test_exception_in_one_message_does_not_stop_the_next(setup, monkeypatch) -> None:
    repo, ingestor, client, sub, _ = setup
    original = repo.insert_zone_env
    calls = {"n": 0}

    def flaky(*args: Any, **kwargs: Any) -> bool:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("disk on fire")
        return original(*args, **kwargs)

    monkeypatch.setattr(repo, "insert_zone_env", flaky)
    client.deliver("textile/zone/Z4/env", env("2026-10-28T04:48:00Z"))
    client.deliver("textile/zone/Z4/env", env("2026-10-28T04:48:10Z"))
    sub.poll_once(timeout_s=0)
    sub.shutdown()
    assert ingestor.counts["error"] == 1 and ingestor.counts["stored"] == 1
    rows = repo.get_zone_env("Z4", "2026-10-28T00:00:00Z", "2026-10-29T00:00:00Z")
    assert [r["ts"] for r in rows] == ["2026-10-28T04:48:10Z"]


def test_writes_are_batched_and_acked_only_after_commit(setup) -> None:
    repo, ingestor, client, sub, clock = setup
    for i in range(5):
        client.deliver("textile/zone/Z4/env", env(f"2026-10-28T04:48:{i:02d}Z"))
    sub.poll_once(timeout_s=0)
    assert client.acked == []                           # batch not yet due: nothing acked
    assert ingestor.counts["commits"] == 0
    clock.t += 1.5                                      # batch interval elapses
    sub.poll_once(timeout_s=0)
    assert client.acked == [1, 2, 3, 4, 5]
    assert ingestor.counts["commits"] == 1
    other = Repo(repo.db_path)                          # a second connection sees the commit
    assert other.table_counts()["zone_env"] == 5
    other.close()


def test_batch_size_triggers_commit(tmp_path: Path) -> None:
    repo = Repo(tmp_path / "b.db")
    repo.init_schema()
    ingestor = Ingestor(repo, TimeReference("stream"), batch_size=3, batch_interval_s=3600, clock=FakeClock())
    client = FakeClient()
    sub = Subscriber(ingestor, client, qos=1, stats_interval_s=3600, poll_s=0.0, clock=FakeClock())
    sub.start("localhost", 1883, 60, 1, 30)
    for i in range(7):
        client.deliver("textile/zone/Z4/env", env(f"2026-10-28T04:48:{i:02d}Z"))
    sub.poll_once(timeout_s=0)
    assert client.acked == [1, 2, 3, 4, 5, 6]           # two full batches
    sub.shutdown()
    assert client.acked == [1, 2, 3, 4, 5, 6, 7]        # shutdown flushes the remainder
    repo.close()


def test_shutdown_flushes_queued_and_pending_messages(setup) -> None:
    repo, ingestor, client, sub, _ = setup
    for i in range(10):
        client.deliver("textile/zone/Z4/env", env(f"2026-10-28T04:48:{i:02d}Z"))
    sub.shutdown()                                      # never polled: all still queued
    assert repo.table_counts()["zone_env"] == 10
    assert len(client.acked) == 10 and not client.connected and not client.loop_running


def test_messages_after_shutdown_starts_are_not_acked(setup) -> None:
    repo, _, client, sub, _ = setup
    sub.shutdown()
    client.deliver("textile/zone/Z4/env", env())
    assert repo.table_counts()["zone_env"] == 0 and client.acked == []


def test_duplicate_delivery_is_stored_once(setup) -> None:
    repo, ingestor, client, sub, _ = setup
    client.deliver("textile/zone/Z4/env", env())
    client.deliver("textile/zone/Z4/env", env())        # QoS 1 redelivery
    client.deliver("textile/zone/Z4/env", b"{bad")
    client.deliver("textile/zone/Z4/env", b"{bad")
    sub.shutdown()
    assert repo.table_counts()["zone_env"] == 1 and repo.table_counts()["rejected_records"] == 1
    assert ingestor.counts["duplicate"] == 1 and ingestor.counts["rejected_duplicate"] == 1


def test_stuck_value_flag_uses_ring_buffer(setup) -> None:
    repo, ingestor, client, sub, _ = setup
    start = datetime(2026, 10, 28, 4, 0, tzinfo=timezone.utc)
    for i in range(32):
        ts = (start + timedelta(seconds=10 * i)).strftime("%Y-%m-%dT%H:%M:%SZ")
        client.deliver("textile/zone/Z4/env", env(ts))   # every value identical
    sub.shutdown()
    rows = repo.get_zone_env("Z4", "2026-10-28T00:00:00Z", "2026-10-29T00:00:00Z")
    flagged = [r["ts"] for r in rows if "stuck_value" in r["flags"]]
    # Readings 31 and 32 each follow 30 identical predecessors.
    assert len(flagged) == 2 and ingestor.flag_counts["stuck_value"] == 2


def test_spike_flag_across_consecutive_readings(setup) -> None:
    repo, _, client, sub, _ = setup
    client.deliver("textile/zone/Z4/env", env("2026-10-28T04:48:00Z", noise_dba=60.0))
    client.deliver("textile/zone/Z4/env", env("2026-10-28T04:48:10Z", noise_dba=96.0))
    sub.shutdown()
    rows = repo.get_zone_env("Z4", "2026-10-28T00:00:00Z", "2026-10-29T00:00:00Z")
    assert [r["flags"] for r in rows] == [[], ["spike"]]


def test_stream_watermark_rejects_future_and_flags_late(setup) -> None:
    repo, _, client, sub, _ = setup
    client.deliver("textile/zone/Z4/env", env("2026-10-28T04:48:00Z"))
    client.deliver("textile/zone/Z4/env", env("2026-10-28T05:00:00Z"))   # 12 min ahead of the stream
    client.deliver("textile/zone/Z4/env", env("2026-10-28T01:00:00Z"))   # 3 h behind: late upload
    sub.shutdown()
    assert [r["reason"] for r in repo.get_rejected()] == ["future_ts"]
    rows = repo.get_zone_env("Z4", "2026-10-28T00:00:00Z", "2026-10-29T00:00:00Z")
    assert {r["ts"]: r["flags"] for r in rows} == {"2026-10-28T01:00:00Z": ["late_upload"],
                                                   "2026-10-28T04:48:00Z": []}


def test_retained_status_does_not_set_the_watermark(setup) -> None:
    repo, _, client, sub, _ = setup
    old_birth = {"schema_version": "1.1", "device_id": "ZN-Z1", "ts": "2026-10-28T00:00:00Z", "online": True,
                 "firmware": "sim-0.1.0", "uptime_s": 0, "buffered_records": 0}
    client.deliver("textile/status/ZN-Z1", old_birth, retain=True)
    client.deliver("textile/zone/Z4/env", env("2026-10-28T03:00:00Z"))   # live data, 3 h after the birth
    sub.shutdown()
    assert repo.table_counts()["zone_env"] == 1 and repo.get_rejected() == []


def test_wall_time_reference_uses_receipt_time(tmp_path: Path) -> None:
    repo = Repo(tmp_path / "w.db")
    repo.init_schema()
    ingestor = Ingestor(repo, TimeReference("wall"), 500, 1.0)
    future = (datetime.now(timezone.utc) + timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    assert ingestor.handle(Message("textile/zone/Z4/env", json.dumps(env(future)).encode(),
                                   datetime.now(timezone.utc))) == "rejected"
    assert ingestor.handle(Message("textile/zone/Z4/env", json.dumps(env(now)).encode(),
                                   datetime.now(timezone.utc))) == "stored"
    ingestor.flush()
    repo.close()


def test_received_ts_is_fixed_width_utc(setup) -> None:
    repo, _, client, sub, _ = setup
    client.deliver("textile/zone/Z4/env", env())
    sub.shutdown()
    (row,) = repo.get_zone_env("Z4", "2026-10-28T00:00:00Z", "2026-10-29T00:00:00Z")
    assert len(row["received_ts"]) == len("2026-10-28T04:48:10.123Z") and row["received_ts"].endswith("Z")
    assert row["ts"] == TS                               # device ts stored exactly as published


def test_idle_exit_shuts_down_cleanly(setup) -> None:
    repo, _, client, sub, clock = setup

    def poll_and_tick(timeout_s: float | None = None) -> int:
        n = original_poll(timeout_s=0)
        clock.t += 1.0
        return n

    original_poll = sub.poll_once
    sub.poll_once = poll_and_tick                       # each poll advances 1 s of fake time
    client.deliver("textile/zone/Z4/env", env())
    sub.run(idle_exit_s=5)                              # returns once 5 s pass with no messages
    assert repo.table_counts()["zone_env"] == 1 and client.acked == [1]
    assert not client.connected


def test_commits_when_broker_inflight_window_is_full(tmp_path: Path) -> None:
    """Acks wait for commit, so a batch must commit before the broker's window fills."""
    repo = Repo(tmp_path / "w.db")
    repo.init_schema()
    ingestor = Ingestor(repo, TimeReference("stream"), batch_size=500, batch_interval_s=3600,
                        max_unacked=4, clock=FakeClock())
    client = FakeClient()
    sub = Subscriber(ingestor, client, qos=1, stats_interval_s=3600, poll_s=0.0, clock=FakeClock())
    sub.start("localhost", 1883, 60, 1, 30)
    for i in range(10):
        client.deliver("textile/zone/Z4/env", env(f"2026-10-28T04:48:{i:02d}Z"))
    sub.poll_once(timeout_s=0)
    assert client.acked == list(range(1, 9))            # two windows of 4, acked after commit
    sub.shutdown()
    repo.close()
