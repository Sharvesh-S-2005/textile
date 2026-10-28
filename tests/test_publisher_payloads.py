"""Contract tests for simulator/publisher.py.

Runs the publisher in --dry-run mode and checks every message against
docs/data_contract.md. A payload that fails here is a bug in the simulator.
"""

import contextlib
import io
import json
import re
import time
import uuid
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import config
from simulator import publisher

START_LOCAL = "2026-10-28T05:30:00"          # covers C check-outs, A check-ins and PPE rounds
START_UTC = datetime(2026, 10, 28, 0, 0, tzinfo=timezone.utc)
OUTAGE_START = START_UTC + timedelta(minutes=40)
OUTAGE_END = OUTAGE_START + timedelta(minutes=4)
TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z$")

VALIDATION = config.get_validation_config()
RANGES = VALIDATION["fields"]
FLAGS = set(VALIDATION["allowed_sensor_flags"])
ZONE_IDS = {z["zone_id"] for z in config.get_zones()}
MACHINE_IDS = {m["machine_id"] for m in config.get_machines()}
CARD_IDS = {w["card_id"] for w in config.get_workers()}
TAG_IDS = {w["tag_id"] for w in config.get_workers()}
WORKER_IDS = {w["worker_id"] for w in config.get_workers()}

ENV_FIELDS = {"noise_dba": float, "noise_peak_dba": float, "pm25_ugm3": float, "pm10_ugm3": float,
              "temp_c": float, "rh_pct": float, "globe_temp_c": float, "co2_ppm": int}
ENV_SENSORS = {"noise": ["noise_dba", "noise_peak_dba"], "pm": ["pm25_ugm3", "pm10_ugm3"],
               "temp_rh": ["temp_c", "rh_pct"], "globe": ["globe_temp_c"], "co2": ["co2_ppm"]}
MACHINE_FIELDS = {"current_a": float, "vib_rms_g": float, "vib_peak_g": float}
MACHINE_CHANNELS = {"current": ["current_a"], "vibration": ["vib_rms_g", "vib_peak_g"]}
EVENT_KINDS = ("rfid", "ppe")


def _run(argv: list[str]) -> list[tuple[str, dict]]:
    """Run the publisher in-process and parse its `topic payload` stdout lines."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert publisher.main(argv) == 0
    messages = []
    for line in buf.getvalue().splitlines():
        topic, _, body = line.partition(" ")
        messages.append((topic, json.loads(body)))
    return messages


@pytest.fixture(scope="module")
def messages() -> list[tuple[str, dict]]:
    return _run(["--dry-run", "--no-align-shifts", "--speed", "1e9", "--duration-h", "1.5",
                 "--start", START_LOCAL, "--seed", "42",
                 "--outage-zone", "Z3", "--outage-start-min", "40", "--outage-duration-min", "4"])


def _parse_ts(text: str) -> datetime:
    assert TS_RE.match(text), text
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def _kind(topic: str) -> str:
    parts = topic.split("/")
    if parts[1] == "zone":
        return parts[3]          # env / ble
    if parts[1] == "machine":
        return "machine"
    if parts[1] == "gate":
        return "rfid"
    if parts[1] == "ppe":
        return "ppe"
    return "status"


def _in_range(field: str, value: object, expected_type: type) -> None:
    assert not isinstance(value, bool), (field, value)
    assert isinstance(value, (int, float) if expected_type is float else int), (field, value)
    assert RANGES[field]["min"] <= value <= RANGES[field]["max"], (field, value)


# ---------------------------------------------------------------- tests


def test_all_six_topic_types_are_published(messages: list[tuple[str, dict]]) -> None:
    kinds = {_kind(t) for t, _ in messages}
    assert kinds == {"env", "ble", "machine", "rfid", "ppe", "status"}


def test_common_fields_and_ts(messages: list[tuple[str, dict]]) -> None:
    end = START_UTC + timedelta(hours=1.5)
    for topic, p in messages:
        assert p["schema_version"] == "1.1", topic
        assert isinstance(p["device_id"], str) and p["device_id"], topic
        ts = _parse_ts(p["ts"])
        assert START_UTC <= ts <= end, (topic, p["ts"])   # simulated time, not wall clock


def test_topic_prefix_and_device_ids(messages: list[tuple[str, dict]]) -> None:
    for topic, p in messages:
        assert topic.startswith("textile/")
        kind = _kind(topic)
        if kind in ("env", "ble"):
            assert p["device_id"] == f"ZN-{p['zone_id']}"
        elif kind == "machine":
            assert p["device_id"] == f"MN-{p['machine_id']}"
        elif kind == "status":
            assert topic == f"textile/status/{p['device_id']}"


def test_env_payloads(messages: list[tuple[str, dict]]) -> None:
    envs = [(t, p) for t, p in messages if _kind(t) == "env"]
    assert envs
    for topic, p in envs:
        assert p["zone_id"] in ZONE_IDS and topic == f"textile/zone/{p['zone_id']}/env"
        assert set(p["sensor_flags"]) == set(ENV_SENSORS)
        for sensor, fields in ENV_SENSORS.items():
            flag = p["sensor_flags"][sensor]
            assert flag in FLAGS
            for field in fields:
                assert field in p, (topic, field)          # present even when null
                if p[field] is None:
                    assert flag != "ok", (topic, field)    # null only with a non-ok flag
                else:
                    _in_range(field, p[field], ENV_FIELDS[field])


def test_ble_payloads(messages: list[tuple[str, dict]]) -> None:
    bles = [(t, p) for t, p in messages if _kind(t) == "ble"]
    assert bles
    for topic, p in bles:
        assert p["zone_id"] in ZONE_IDS and topic == f"textile/zone/{p['zone_id']}/ble"
        _in_range("scan_window_s", p["scan_window_s"], int)
        assert isinstance(p["tags"], list)
        for tag in p["tags"]:
            assert set(tag) == {"tag_id", "rssi_dbm", "seen_count"}
            assert tag["tag_id"] in TAG_IDS
            _in_range("rssi_dbm", tag["rssi_dbm"], int)
            assert isinstance(tag["seen_count"], int) and tag["seen_count"] >= 1
        assert len({t["tag_id"] for t in p["tags"]}) == len(p["tags"])


def test_ble_has_leakage_into_neighbouring_zones(messages: list[tuple[str, dict]]) -> None:
    seen_by: dict[tuple[str, str], set[str]] = defaultdict(set)
    for topic, p in messages:
        if _kind(topic) == "ble":
            for tag in p["tags"]:
                seen_by[(p["ts"], tag["tag_id"])].add(p["zone_id"])
    multi = sum(1 for zones in seen_by.values() if len(zones) > 1)
    assert multi / len(seen_by) > 0.2


def test_machine_payloads(messages: list[tuple[str, dict]]) -> None:
    machine_zone = {m["machine_id"]: m["zone_id"] for m in config.get_machines()}
    machines = [(t, p) for t, p in messages if _kind(t) == "machine"]
    assert machines
    for topic, p in machines:
        assert p["machine_id"] in MACHINE_IDS
        assert topic == f"textile/machine/{p['machine_id']}/status"
        assert p["zone_id"] == machine_zone[p["machine_id"]]
        assert isinstance(p["running"], bool)
        assert set(p["sensor_flags"]) == set(MACHINE_CHANNELS)
        assert set(p["sensor_flags"].values()) <= FLAGS
        for channel, fields in MACHINE_CHANNELS.items():
            for field in fields:
                assert field in p, (topic, field)          # present even when null
                if p[field] is None:
                    assert p["sensor_flags"][channel] != "ok", (topic, field)
                else:
                    _in_range(field, p[field], MACHINE_FIELDS[field])


def test_rfid_payloads(messages: list[tuple[str, dict]]) -> None:
    gate = config.get_devices()["gate"]
    rfid = [p for t, p in messages if _kind(t) == "rfid"]
    events = {p["event"] for p in rfid}
    assert events == {"check_in", "check_out"}
    last_event: dict[str, str] = {}
    for p in rfid:
        assert p["card_id"] in CARD_IDS
        assert p["device_id"] == gate["device_id"] and p["gate_id"] == gate["gate_id"]
        assert p["event"] in VALIDATION["allowed_rfid_events"]
        assert last_event.get(p["card_id"]) != p["event"]   # never two check-ins in a row
        last_event[p["card_id"]] = p["event"]


def test_ppe_payloads(messages: list[tuple[str, dict]]) -> None:
    ppe = [p for t, p in messages if _kind(t) == "ppe"]
    assert ppe
    allowed = set(VALIDATION["allowed_ppe_values"])
    for p in ppe:
        assert p["worker_id"] in WORKER_IDS
        assert p["hearing_ppe"] in allowed and p["respiratory_ppe"] in allowed
        assert p["device_id"] == config.get_devices()["ppe_app"]["device_id"]
        assert isinstance(p["recorded_by"], str) and p["recorded_by"]


def test_status_payloads(messages: list[tuple[str, dict]]) -> None:
    statuses = [p for t, p in messages if _kind(t) == "status"]
    devices = {p["device_id"] for p in statuses}
    assert len(devices) == 7 + 8 + 2
    for p in statuses:
        assert isinstance(p["online"], bool)
        assert p["firmware"].startswith("sim-")
        assert isinstance(p["uptime_s"], int) and p["uptime_s"] >= 0
        assert isinstance(p["buffered_records"], int) and p["buffered_records"] >= 0
    first = {}
    last = {}
    for p in statuses:
        first.setdefault(p["device_id"], p)
        last[p["device_id"]] = p
    assert all(p["online"] for p in first.values())       # birth message
    assert not any(p["online"] for p in last.values())    # clean shutdown


def test_dedup_keys_are_unique(messages: list[tuple[str, dict]]) -> None:
    # Contract v1.1 section 11: telemetry on (device_id, ts, topic), events on event_id.
    telemetry = [(p["device_id"], p["ts"], t) for t, p in messages if _kind(t) not in EVENT_KINDS]
    assert len(telemetry) == len(set(telemetry))
    event_ids = [p["event_id"] for t, p in messages if _kind(t) in EVENT_KINDS]
    assert event_ids and len(event_ids) == len(set(event_ids))


def test_event_ids_are_uuid4(messages: list[tuple[str, dict]]) -> None:
    for t, p in messages:
        if _kind(t) in EVENT_KINDS:
            parsed = uuid.UUID(p["event_id"])
            assert parsed.version == 4 and str(parsed) == p["event_id"]


def test_outage_buffers_then_replays_with_original_ts(messages: list[tuple[str, dict]]) -> None:
    z3 = [(i, t, p) for i, (t, p) in enumerate(messages) if p["device_id"] == "ZN-Z3"]
    live_in_outage = [p for _, t, p in z3 if _kind(t) != "status"
                      and OUTAGE_START <= _parse_ts(p["ts"]) < OUTAGE_END]
    # 4 min outage: 24 env (10 s) + 8 ble (30 s) readings, all delivered late.
    assert len(live_in_outage) == 24 + 8

    backlog = [(i, p) for i, t, p in z3 if _kind(t) == "status" and p["buffered_records"] > 0]
    assert len(backlog) == 1
    status_idx, status = backlog[0]
    assert status["online"] and status["buffered_records"] == 24 + 8
    assert _parse_ts(status["ts"]) >= OUTAGE_END

    replayed = [p for i, t, p in z3 if i > status_idx and _kind(t) != "status"
                and _parse_ts(p["ts"]) < OUTAGE_END]
    assert len(replayed) == 24 + 8
    assert all(_parse_ts(p["ts"]) < _parse_ts(status["ts"]) for p in replayed)

    # Nothing from Z3 was published while it was offline.
    before = [p for i, t, p in z3 if i < status_idx]
    assert not any(OUTAGE_START <= _parse_ts(p["ts"]) < OUTAGE_END for p in before)

    # Backlog cleared afterwards.
    after = [p for i, t, p in z3 if i > status_idx and _kind(t) == "status"]
    assert after[0]["online"] and after[0]["buffered_records"] == 0


def test_seeded_runs_are_identical() -> None:
    argv = ["--dry-run", "--no-align-shifts", "--speed", "1e9", "--duration-h", "0.1", "--start", START_LOCAL, "--seed", "5"]
    assert _run(argv) == _run(argv)


def test_speed_paces_wall_clock() -> None:
    # 0.02 h = 72 simulated seconds; at 100x that is ~0.72 s of wall time.
    t0 = time.monotonic()
    _run(["--dry-run", "--no-align-shifts", "--speed", "100", "--duration-h", "0.02", "--start", START_LOCAL])
    assert 0.5 < time.monotonic() - t0 < 3.0


def test_normal_run_has_same_second_gate_pair(messages: list[tuple[str, dict]]) -> None:
    # v1.1b: real gates see two taps in one second; the run must exercise that case.
    rfid = [p for t, p in messages if _kind(t) == "rfid"]
    by_second: dict[str, list[dict]] = defaultdict(list)
    for p in rfid:
        by_second[p["ts"]].append(p)
    pairs = [ps for ps in by_second.values() if len(ps) > 1]
    assert pairs
    for ps in pairs:
        assert len({p["event_id"] for p in ps}) == len(ps)
        assert len({p["card_id"] for p in ps}) == len(ps)


class _InterruptingSink:
    """Raises KeyboardInterrupt once, on the Nth publish, like Ctrl-C mid-run."""

    def __init__(self, interrupt_at: int) -> None:
        self.messages: list[tuple[str, dict]] = []
        self._interrupt_at = interrupt_at
        self.closed = False

    def publish(self, device_id: str, topic: str, payload: dict, retain: bool) -> None:
        if len(self.messages) == self._interrupt_at:
            self._interrupt_at = -1
            raise KeyboardInterrupt
        self.messages.append((topic, payload))

    def close(self) -> None:
        self.closed = True


def test_ctrl_c_publishes_offline_for_every_device() -> None:
    sim = publisher.Simulation(START_UTC, timedelta(hours=1), seed=1, outages=[])
    sink = _InterruptingSink(interrupt_at=500)
    sim.run(sink, speed=1e9)
    tail = sink.messages[-len(sim.device_ids):]
    assert sorted(p["device_id"] for _, p in tail) == sorted(sim.device_ids)
    assert all(t.startswith("textile/status/") and p["online"] is False for t, p in tail)
    assert _parse_ts(tail[0][1]["ts"]) < START_UTC + timedelta(hours=1)   # stopped early
    assert sink.closed


def test_long_machine_outage_replays_records_older_than_late_threshold() -> None:
    """--extra-outage on a machine node: replayed records keep their original ts."""
    msgs = _run(["--dry-run", "--no-align-shifts", "--speed", "1e9", "--duration-h", "3", "--start", START_LOCAL, "--no-outage",
                 "--extra-outage", "MN-M8:10:150"])
    m8 = [(i, t, p) for i, (t, p) in enumerate(msgs) if p["device_id"] == "MN-M8"]
    backlog = [(i, p) for i, t, p in m8 if _kind(t) == "status" and p["buffered_records"] > 0]
    assert len(backlog) == 1
    idx, status = backlog[0]
    assert status["buffered_records"] == 150 * 6          # 150 min of 10 s readings
    replayed = [p for i, t, p in m8 if i > idx and _kind(t) == "machine"
                and _parse_ts(p["ts"]) < _parse_ts(status["ts"])]
    assert len(replayed) == 150 * 6
    late_s = VALIDATION["late_upload_threshold_s"]
    oldest = min(_parse_ts(p["ts"]) for p in replayed)
    assert (_parse_ts(status["ts"]) - oldest).total_seconds() > late_s


def test_bad_extra_outage_spec_is_an_error() -> None:
    with pytest.raises(SystemExit):
        publisher.parse_args(["--extra-outage", "MN-M99:10:5"])
    with pytest.raises(SystemExit):
        publisher.parse_args(["--extra-outage", "MN-M8:ten:5"])


# ---------------------------------------------------------------- Day 3 Task 1b: --align-shifts

def _gate_events(msgs: list[tuple[str, dict]]) -> list[dict]:
    workers = {w["card_id"]: w["worker_id"] for w in config.get_workers()}
    return [{"ts": p["ts"], "event": p["event"], "event_id": p["event_id"], "card_id": p["card_id"],
             "worker_id": workers[p["card_id"]]} for t, p in msgs if _kind(t) == "rfid"]


def test_align_window_snaps_to_shift_boundaries_plus_margins() -> None:
    margins = config.get_simulation_config()["align_shifts"]
    ist = timezone(timedelta(hours=5, minutes=30))
    start, end = publisher.align_window(datetime(2026, 10, 28, 6, 0, tzinfo=ist), timedelta(hours=8))
    assert start == datetime(2026, 10, 28, 6, 0, tzinfo=ist) - timedelta(minutes=margins["margin_before_min"])
    assert end == datetime(2026, 10, 28, 14, 0, tzinfo=ist) + timedelta(minutes=margins["margin_after_min"])
    # A mid-shift start snaps BACK to the shift that is running (C, 22:00 the day before).
    start, _ = publisher.align_window(datetime(2026, 10, 28, 5, 30, tzinfo=ist), timedelta(hours=1))
    assert start == datetime(2026, 10, 27, 22, 0, tzinfo=ist) - timedelta(minutes=margins["margin_before_min"])


def test_aligned_run_produces_only_whole_unflagged_sessions() -> None:
    from ingest.sessions import build_sessions
    from shifts import ShiftSchedule

    msgs = _run(["--dry-run", "--speed", "1e9", "--duration-h", "24", "--start", "2026-10-28T06:00:00",
                 "--no-outage"])
    shift = {w["worker_id"]: w["shift"] for w in config.get_workers()}
    sessions = build_sessions(_gate_events(msgs), ShiftSchedule(config.get_shifts()), shift,
                              config.get_sessions_config()["max_session_h"])
    assert len(sessions) >= 20                      # every crew completes at least one shift
    assert all(s["flags"] == [] for s in sessions), [s for s in sessions if s["flags"]]


def test_raw_window_keeps_truncated_sessions_for_the_flagging_path() -> None:
    from ingest.sessions import build_sessions
    from shifts import ShiftSchedule

    msgs = _run(["--dry-run", "--no-align-shifts", "--speed", "1e9", "--duration-h", "2", "--start",
                 START_LOCAL, "--no-outage"])
    shift = {w["worker_id"]: w["shift"] for w in config.get_workers()}
    sessions = build_sessions(_gate_events(msgs), ShiftSchedule(config.get_shifts()), shift,
                              config.get_sessions_config()["max_session_h"])
    flags = {f for s in sessions for f in s["flags"]}
    assert {"orphan_check_out", "unmatched_check_in"} <= flags


def test_aligned_ground_truth_has_no_one_on_site_at_the_window_edges(tmp_path: Path) -> None:
    out = tmp_path / "t.jsonl"
    _run(["--dry-run", "--speed", "1e9", "--duration-h", "8", "--start", "2026-10-28T06:00:00",
          "--no-outage", "--ground-truth-out", str(out)])
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    first_ts, last_ts = rows[0]["ts"], rows[-1]["ts"]
    assert all(r["zone_id"] is None for r in rows if r["ts"] in (first_ts, last_ts))
