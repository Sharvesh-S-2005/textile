"""Day 8A: nothing the long run keeps may grow with its length.

The 7-day stochastic run was stopped by machine-wide memory pressure. The probe found no
growing collection (docs/review_notes.md, Day 8A), and these tests keep it that way: every
collection the publisher and the subscriber keep while running has a cap from
config.yaml, and a long in-process dry run must end within a configured margin of the
memory it used 10% of the way through.
"""

import ctypes
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import config
from db.repo import Repo
from ingest.subscriber import Ingestor, Message, Subscriber, TimeReference
from simulator.machine_stochastics import Plant
from simulator.publisher import Outage, Simulation

BOUNDS = config.get_simulation_config()["memory_bounds"]
INGEST = config.get_ingest_config()
START = datetime(2026, 10, 20, 0, 30, tzinfo=timezone.utc)


def process_memory_mb() -> tuple[float, float]:
    """(resident set, committed private memory) of this process in MB."""
    if sys.platform == "win32":
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (n, ctypes.c_size_t) for n in (
                    "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage",
                    "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage",
                    "PeakPagefileUsage", "PrivateUsage")]

        k = ctypes.windll.kernel32
        k.GetCurrentProcess.restype = wintypes.HANDLE
        k.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        c = Counters()
        c.cb = ctypes.sizeof(c)
        if not k.K32GetProcessMemoryInfo(k.GetCurrentProcess(), ctypes.byref(c), c.cb):
            raise OSError("GetProcessMemoryInfo failed")
        return c.WorkingSetSize / 2**20, c.PrivateUsage / 2**20
    fields = dict(line.split(":", 1) for line in Path("/proc/self/status").read_text().splitlines() if ":" in line)
    return int(fields["VmRSS"].split()[0]) / 1024, int(fields["VmData"].split()[0]) / 1024


class SamplingSink:
    """Discards messages (keeping only `keep_device`'s, if given); samples process memory
    when the stream passes `mark`."""

    def __init__(self, mark: datetime, keep_device: str | None = None) -> None:
        self.mark = mark
        self.keep_device = keep_device
        self.at_mark: tuple[float, float] | None = None
        self.n = 0
        self.replayed: list[dict[str, Any]] = []

    def publish(self, device_id: str, topic: str, payload: dict[str, Any], retain: bool) -> None:
        self.n += 1
        if self.at_mark is None and "ts" in payload and payload["ts"] >= self.mark.strftime("%Y-%m-%dT%H:%M:%SZ"):
            self.at_mark = process_memory_mb()
        if device_id == self.keep_device and "/status/" not in topic:
            self.replayed.append(payload)

    def close(self) -> None:
        pass


# ---------------------------------------------------------------- the regression test


def test_long_dry_run_memory_stays_flat(tmp_path: Path) -> None:
    """Private memory (and resident set) at the end of a long run within margin of 10% through."""
    window = timedelta(hours=BOUNDS["regression_window_h"])
    sim = Simulation(START, window, 42, [], machine_stochastics=True)
    sink = SamplingSink(sim.start + (sim.end - sim.start) / 10)
    with (tmp_path / "truth.jsonl").open("w", encoding="utf-8") as truth:
        sim.run(sink, speed=1e12, ground_truth=truth)
    end = process_memory_mb()
    assert sink.at_mark is not None and sink.n > 100_000
    margin = BOUNDS["regression_margin_mb"]
    print(f"memory at 10%: rss {sink.at_mark[0]:.1f} MB, private {sink.at_mark[1]:.1f} MB; "
          f"at end: rss {end[0]:.1f} MB, private {end[1]:.1f} MB ({sink.n} messages)")
    assert end[1] <= sink.at_mark[1] + margin, f"private memory grew {sink.at_mark[1]:.1f} -> {end[1]:.1f} MB"
    assert end[0] <= sink.at_mark[0] + margin, f"resident set grew {sink.at_mark[0]:.1f} -> {end[0]:.1f} MB"


# ---------------------------------------------------------------- publisher caps


def test_outage_buffer_never_exceeds_its_cap(caplog: pytest.LogCaptureFixture) -> None:
    """A 30-minute Z3 outage with a 5-record buffer: 5 replayed, the rest counted as overwritten."""
    sim = Simulation(START, timedelta(hours=2), 42, [Outage("ZN-Z3", START + timedelta(minutes=30),
                                                          START + timedelta(minutes=60))])
    cap = 5
    sim.sim = {**sim.sim, "memory_bounds": {**BOUNDS, "outage_buffer_max_records": cap}}
    sink = SamplingSink(sim.end, keep_device="ZN-Z3")
    sim.run(sink, speed=1e12)
    replay_ts = [p["ts"] for p in sink.replayed if START + timedelta(minutes=30) <= datetime.fromisoformat(
        p["ts"].replace("Z", "+00:00")) < START + timedelta(minutes=60)]
    assert len(replay_ts) == cap                                    # only the newest `cap` survive
    assert "buffer full" in caplog.text


def test_stop_log_and_shift_plan_are_bounded() -> None:
    """Two simulated weeks of one loom with a stop log capped at 10 records."""
    cfg = config.get_simulation_config()["machine_stochastics"]
    plant = Plant(config.get_machines(), cfg, np.random.default_rng(1), stop_log_max=10)
    sim = Simulation(START, timedelta(days=14), 42, [], machine_stochastics=True)
    loom = next(m for m in sim.machines if m.machine_id == "M5")
    loom.attach_stochastics(plant.for_machine("M5"))
    state = plant.for_machine("M5")
    lengths, plans = [], []
    for i in range(0, 14 * 24 * 60, 1):
        loom.reading(START + timedelta(minutes=i))
        lengths.append(len(state.stops))
        plans.append(len(state._shift_plan))
    assert max(lengths) == 10 and state.stops.maxlen == 10     # filled, never exceeded
    assert max(plans) <= 6                                      # about one day of shift plans


def test_ground_truth_is_flushed_while_running(tmp_path: Path) -> None:
    """The JSONL reaches the disk during the run, not only when the file is closed."""
    sizes: list[int] = []
    path = tmp_path / "truth.jsonl"

    class Watching(SamplingSink):
        def publish(self, device_id: str, topic: str, payload: dict[str, Any], retain: bool) -> None:
            if topic.endswith("/ble") and payload.get("zone_id") == "Z1":
                sizes.append(path.stat().st_size)

    sim = Simulation(START, timedelta(hours=3), 42, [])
    with path.open("w", encoding="utf-8") as truth:
        sim.run(Watching(sim.end), speed=1e12, ground_truth=truth)
    assert sizes[-1] > 0 and len(set(sizes)) > 3


# ---------------------------------------------------------------- subscriber caps


def env(zone: str, ts: str) -> dict[str, Any]:
    return {"schema_version": "1.1", "device_id": f"ZN-{zone}", "ts": ts, "zone_id": zone,
            "noise_dba": 96.4, "noise_peak_dba": 103.1, "pm25_ugm3": 118.2, "pm10_ugm3": 240.7,
            "temp_c": 33.8, "rh_pct": 72.5, "globe_temp_c": 36.2, "co2_ppm": 980,
            "sensor_flags": {"noise": "ok", "pm": "ok", "temp_rh": "ok", "globe": "ok", "co2": "ok"}}


def test_history_is_bounded_in_devices_and_length(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    repo = Repo(tmp_path / "h.db")
    repo.init_schema()
    ingestor = Ingestor(repo, TimeReference("stream"), 500, 1.0)
    ingestor._max_history_devices = 2
    t0 = datetime(2026, 10, 28, 4, 0, tzinfo=timezone.utc)
    for i in range(30):
        ts = (t0 + timedelta(seconds=10 * i)).strftime("%Y-%m-%dT%H:%M:%SZ")
        for zone in ("Z1", "Z2", "Z3"):
            assert ingestor.handle(Message(f"textile/zone/{zone}/env", json.dumps(env(zone, ts)).encode(),
                                           datetime.now(timezone.utc))) == "stored"
    assert set(ingestor._history) == {"ZN-Z1", "ZN-Z2"}                       # third device not tracked
    n = config.get_validation_config()["stuck_value_count"]
    assert all(len(d) <= n for fields in ingestor._history.values() for d in fields.values())
    assert ingestor.counts["history_untracked"] == 30 and "history cap" in caplog.text
    ingestor.flush()
    repo.close()


def test_subscriber_queue_has_a_cap() -> None:
    class Client:
        on_connect = on_disconnect = on_message = None

    sub = Subscriber(Ingestor.__new__(Ingestor), Client(), 1, 10.0, 0.1)
    assert sub._queue.maxsize == INGEST["max_queued_messages"] > 0


def test_wal_is_checkpointed_and_truncated(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("INFO", logger="db.repo")
    repo = Repo(tmp_path / "w.db", wal_checkpoint_every=3)
    repo.init_schema()
    ingestor = Ingestor(repo, TimeReference("stream"), 500, 1.0)
    t0 = datetime(2026, 10, 28, 4, 0, tzinfo=timezone.utc)
    sizes = []
    for i in range(6):
        ts = (t0 + timedelta(seconds=10 * i)).strftime("%Y-%m-%dT%H:%M:%SZ")
        ingestor.handle(Message("textile/zone/Z4/env", json.dumps(env("Z4", ts)).encode(), datetime.now(timezone.utc)))
        repo.commit()
        sizes.append(repo.wal_size_bytes())
    assert sizes[1] > 0 and sizes[2] == 0 and sizes[5] == 0              # truncated at commits 3 and 6
    assert caplog.text.count("WAL checkpoint after commit") == 2
    repo.close()
    with pytest.raises(ValueError):
        Repo(tmp_path / "x.db", wal_checkpoint_every=0)
