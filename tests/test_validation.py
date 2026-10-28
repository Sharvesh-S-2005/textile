"""Tests for ingest/validation.py - one explicit test per rule in contract section 10."""

import copy
from datetime import datetime, timedelta, timezone

import pytest

import config
from ingest.validation import Flag, Reason, parse_topic, validate

REF = datetime(2026, 10, 28, 4, 48, 30, tzinfo=timezone.utc)
TS = "2026-10-28T04:48:10Z"
N_STUCK = config.get_validation_config()["stuck_value_count"]


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def env() -> dict:
    return {
        "schema_version": "1.1", "device_id": "ZN-Z4", "ts": TS, "zone_id": "Z4",
        "noise_dba": 96.4, "noise_peak_dba": 103.1, "pm25_ugm3": 118.2, "pm10_ugm3": 240.7,
        "temp_c": 33.8, "rh_pct": 72.5, "globe_temp_c": 36.2, "co2_ppm": 980,
        "sensor_flags": {"noise": "ok", "pm": "ok", "temp_rh": "ok", "globe": "ok", "co2": "ok"},
    }


def ble() -> dict:
    return {"schema_version": "1.1", "device_id": "ZN-Z4", "ts": TS, "zone_id": "Z4", "scan_window_s": 30,
            "tags": [{"tag_id": "BLE-0007", "rssi_dbm": -58, "seen_count": 11},
                     {"tag_id": "BLE-0012", "rssi_dbm": -79, "seen_count": 4}]}


def machine() -> dict:
    return {"schema_version": "1.1", "device_id": "MN-M3", "ts": TS, "machine_id": "M3", "zone_id": "Z4",
            "current_a": 14.8, "running": True, "vib_rms_g": 0.42, "vib_peak_g": 1.07,
            "sensor_flags": {"current": "ok", "vibration": "ok"}}


def rfid() -> dict:
    return {"schema_version": "1.1", "device_id": "GW-01", "ts": TS,
            "event_id": "3f2b8c1e-9a4d-4e7b-8c21-5d6f0a9e1b73", "card_id": "RFID-10045",
            "event": "check_in", "gate_id": "GATE-A"}


def ppe() -> dict:
    return {"schema_version": "1.1", "device_id": "SUP-APP", "ts": TS,
            "event_id": "a7c41e02-6b3f-4d9a-9e58-0f2d7c1b4a96", "worker_id": "W101",
            "hearing_ppe": "worn", "respiratory_ppe": "partial", "recorded_by": "SUP-02"}


def status() -> dict:
    return {"schema_version": "1.1", "device_id": "ZN-Z4", "ts": TS, "online": True,
            "firmware": "sim-0.1.0", "uptime_s": 7200, "buffered_records": 0}


TOPIC = {
    "env": "textile/zone/Z4/env", "ble": "textile/zone/Z4/ble", "machine": "textile/machine/M3/status",
    "rfid": "textile/gate/rfid", "ppe": "textile/ppe/entry", "status": "textile/status/ZN-Z4",
}
BUILDERS = {"env": env, "ble": ble, "machine": machine, "rfid": rfid, "ppe": ppe, "status": status}


def rejected(topic: str, payload: object, reason: Reason, ref: datetime = REF) -> None:
    v = validate(topic, payload, ref)
    assert not v.accepted, v
    assert v.reason == reason, (v.reason, v.detail)
    assert v.detail


# ---------------------------------------------------------------- clean records

@pytest.mark.parametrize("kind", list(BUILDERS))
def test_clean_record_of_every_kind_is_accepted(kind: str) -> None:
    v = validate(TOPIC[kind], BUILDERS[kind](), REF)
    assert v.accepted and v.flags == [] and v.reason is None and v.kind == kind


def test_empty_ble_scan_is_accepted() -> None:
    p = ble()
    p["tags"] = []
    assert validate(TOPIC["ble"], p, REF).accepted


def test_topic_parsing() -> None:
    assert parse_topic("textile/zone/Z4/env").zone_id == "Z4"
    assert parse_topic("textile/machine/M3/status").machine_id == "M3"
    assert parse_topic("textile/status/GW-01").device_id == "GW-01"
    assert parse_topic("textile/zone/Z9/env") is None
    assert parse_topic("other/zone/Z1/env") is None


# ---------------------------------------------------------------- LWT form (v1.1a)

def test_lwt_form_without_ts_is_accepted() -> None:
    lwt = {"schema_version": "1.1", "device_id": "ZN-Z4", "online": False}
    original = copy.deepcopy(lwt)
    v = validate(TOPIC["status"], lwt, REF)
    assert v.accepted and v.flags == []
    assert lwt == original and "ts" not in lwt     # no ts invented


def test_offline_status_with_ts_is_also_accepted() -> None:
    p = status()
    p["online"] = False
    assert validate(TOPIC["status"], p, REF).accepted


def test_online_status_without_ts_is_rejected() -> None:
    p = status()
    del p["ts"]
    rejected(TOPIC["status"], p, Reason.MISSING_FIELD)


def test_offline_status_with_null_ts_is_rejected() -> None:
    rejected(TOPIC["status"], {"schema_version": "1.1", "device_id": "ZN-Z4", "online": False, "ts": None},
             Reason.BAD_TS)


# ---------------------------------------------------------------- each rejection reason

def test_reason_unknown_topic() -> None:
    rejected("textile/zone/Z9/env", env(), Reason.UNKNOWN_TOPIC)


def test_reason_not_an_object() -> None:
    rejected(TOPIC["env"], [env()], Reason.NOT_AN_OBJECT)


def test_reason_schema_version() -> None:
    p = env()
    p["schema_version"] = "1.0"
    rejected(TOPIC["env"], p, Reason.SCHEMA_VERSION)


@pytest.mark.parametrize("kind,field", [("env", "pm25_ugm3"), ("env", "zone_id"), ("ble", "tags"),
                                        ("machine", "running"), ("rfid", "card_id"), ("ppe", "worker_id"),
                                        ("env", "ts"), ("env", "device_id")])
def test_reason_missing_field(kind: str, field: str) -> None:
    p = BUILDERS[kind]()
    del p[field]
    rejected(TOPIC[kind], p, Reason.MISSING_FIELD)


def test_reason_missing_sensor_flag_channel() -> None:
    p = env()
    del p["sensor_flags"]["globe"]
    rejected(TOPIC["env"], p, Reason.MISSING_FIELD)


@pytest.mark.parametrize("kind,field,value", [("env", "noise_dba", "96.4"), ("env", "co2_ppm", 980.5),
                                              ("machine", "running", "yes"), ("env", "temp_c", True),
                                              ("ble", "tags", {}), ("status", "online", 1)])
def test_reason_wrong_type(kind: str, field: str, value: object) -> None:
    p = BUILDERS[kind]()
    p[field] = value
    rejected(TOPIC[kind], p, Reason.WRONG_TYPE)


@pytest.mark.parametrize("ts", ["2026-10-28 04:48:10", "2026-10-28T04:48:10+05:30", "28/10/2026",
                                "2026-13-28T04:48:10Z", 1730090890])
def test_reason_bad_ts(ts: object) -> None:
    p = env()
    p["ts"] = ts
    rejected(TOPIC["env"], p, Reason.BAD_TS)


def test_reason_topic_mismatch_zone_id() -> None:
    p = env()
    p["zone_id"] = "Z5"
    rejected(TOPIC["env"], p, Reason.TOPIC_MISMATCH)


def test_reason_topic_mismatch_machine_id() -> None:
    p = machine()
    p["machine_id"] = "M4"
    rejected(TOPIC["machine"], p, Reason.TOPIC_MISMATCH)


def test_reason_topic_mismatch_machine_zone() -> None:
    p = machine()
    p["zone_id"] = "Z1"          # M3 is configured in Z4
    rejected(TOPIC["machine"], p, Reason.TOPIC_MISMATCH)


def test_reason_topic_mismatch_device_id() -> None:
    p = env()
    p["device_id"] = "ZN-Z5"
    rejected(TOPIC["env"], p, Reason.TOPIC_MISMATCH)
    s = status()
    s["device_id"] = "ZN-Z5"
    rejected(TOPIC["status"], s, Reason.TOPIC_MISMATCH)


@pytest.mark.parametrize("kind,path,value", [("env", ("noise_dba",), 140.1), ("env", ("co2_ppm",), 299),
                                             ("env", ("rh_pct",), -0.1), ("machine", ("vib_peak_g",), 16.5),
                                             ("ble", ("tags", 0, "rssi_dbm"), -101),
                                             ("ble", ("tags", 0, "seen_count"), 0),
                                             ("ble", ("scan_window_s",), 121)])
def test_reason_out_of_range(kind: str, path: tuple, value: object) -> None:
    p = BUILDERS[kind]()
    target = p
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    rejected(TOPIC[kind], p, Reason.OUT_OF_RANGE)


@pytest.mark.parametrize("kind,mutate", [
    ("env", lambda p: p["sensor_flags"].update(noise="broken")),
    ("rfid", lambda p: p.update(event="check_through")),
    ("ppe", lambda p: p.update(hearing_ppe="sometimes")),
])
def test_reason_invalid_enum(kind: str, mutate) -> None:
    p = BUILDERS[kind]()
    mutate(p)
    rejected(TOPIC[kind], p, Reason.INVALID_ENUM)


def test_reason_null_with_ok_flag() -> None:
    p = env()
    p["pm25_ugm3"] = None                      # flag still "ok": a contradiction
    rejected(TOPIC["env"], p, Reason.NULL_WITH_OK_FLAG)
    m = machine()
    m["vib_rms_g"] = None
    rejected(TOPIC["machine"], m, Reason.NULL_WITH_OK_FLAG)


@pytest.mark.parametrize("event_id", ["not-a-uuid", "3f2b8c1e-9a4d-1e7b-8c21-5d6f0a9e1b73",  # v1
                                      "3F2B8C1E-9A4D-4E7B-8C21-5D6F0A9E1B73", 12345, None])
def test_reason_invalid_event_id(event_id: object) -> None:
    for kind in ("rfid", "ppe"):
        p = BUILDERS[kind]()
        p["event_id"] = event_id
        rejected(TOPIC[kind], p, Reason.INVALID_EVENT_ID)


def test_missing_event_id_is_rejected() -> None:
    p = rfid()
    del p["event_id"]
    rejected(TOPIC["rfid"], p, Reason.MISSING_FIELD)


def test_reason_future_ts_11_minutes_rejected() -> None:
    p = env()
    p["ts"] = _iso(REF + timedelta(minutes=11))
    rejected(TOPIC["env"], p, Reason.FUTURE_TS)


def test_future_ts_9_minutes_accepted() -> None:
    p = env()
    p["ts"] = _iso(REF + timedelta(minutes=9))
    assert validate(TOPIC["env"], p, REF).accepted


def test_future_ts_exactly_at_tolerance_accepted() -> None:
    p = env()
    p["ts"] = _iso(REF + timedelta(minutes=10))
    assert validate(TOPIC["env"], p, REF).accepted


# ---------------------------------------------------------------- boundaries are inclusive

@pytest.mark.parametrize("field,value", [("noise_dba", 30.0), ("noise_dba", 140.0), ("co2_ppm", 300),
                                         ("co2_ppm", 10000), ("rh_pct", 0.0), ("rh_pct", 100.0),
                                         ("temp_c", -10.0), ("pm10_ugm3", 3000.0)])
def test_value_exactly_on_range_boundary_is_accepted(field: str, value: float) -> None:
    p = env()
    p[field] = value
    assert validate(TOPIC["env"], p, REF).accepted


def test_ble_boundaries_are_accepted() -> None:
    p = ble()
    p["tags"] = [{"tag_id": "BLE-0001", "rssi_dbm": -100, "seen_count": 1},
                 {"tag_id": "BLE-0002", "rssi_dbm": -20, "seen_count": 1000}]
    p["scan_window_s"] = 5
    assert validate(TOPIC["ble"], p, REF).accepted


# ---------------------------------------------------------------- each flag

def test_flag_sensor_not_ok_with_null_value() -> None:
    p = env()
    p["sensor_flags"]["noise"] = "error"
    p["noise_dba"] = p["noise_peak_dba"] = None
    v = validate(TOPIC["env"], p, REF)
    assert v.accepted and v.flags == [Flag.SENSOR_NOT_OK]


def test_flag_sensor_not_ok_with_value_present() -> None:
    p = env()
    p["sensor_flags"]["pm"] = "needs_clean"       # a lint warning still has a reading
    v = validate(TOPIC["env"], p, REF)
    assert v.accepted and v.flags == [Flag.SENSOR_NOT_OK]


def test_flag_sensor_not_ok_on_machine() -> None:
    p = machine()
    p["sensor_flags"]["current"] = "stale"
    p["current_a"] = None
    v = validate(TOPIC["machine"], p, REF)
    assert v.accepted and v.flags == [Flag.SENSOR_NOT_OK]


def test_flag_stuck_value_after_exactly_n_identical_readings() -> None:
    p = env()
    assert Flag.STUCK_VALUE in validate(TOPIC["env"], p, REF, {"co2_ppm": [980] * N_STUCK}).flags
    # N-1 identical previous readings is not yet stuck.
    assert validate(TOPIC["env"], p, REF, {"co2_ppm": [980] * (N_STUCK - 1)}).flags == []
    # One different reading inside the window breaks it.
    assert validate(TOPIC["env"], p, REF, {"co2_ppm": [981] + [980] * (N_STUCK - 1)}).flags == []


def test_flag_spike() -> None:
    p = env()
    threshold = config.get_validation_config()["fields"]["noise_dba"]["spike_threshold"]
    assert validate(TOPIC["env"], p, REF, {"noise_dba": [96.4 - threshold - 0.1]}).flags == [Flag.SPIKE]
    assert validate(TOPIC["env"], p, REF, {"noise_dba": [96.4 - threshold]}).flags == []


def test_flag_late_upload() -> None:
    late_s = config.get_validation_config()["late_upload_threshold_s"]
    p = env()
    p["ts"] = _iso(REF - timedelta(seconds=late_s + 60))
    v = validate(TOPIC["env"], p, REF)
    assert v.accepted and v.flags == [Flag.LATE_UPLOAD]
    p["ts"] = _iso(REF - timedelta(seconds=late_s - 60))
    assert validate(TOPIC["env"], p, REF).flags == []


def test_late_upload_applies_to_events_too() -> None:
    p = rfid()
    p["ts"] = _iso(REF - timedelta(hours=3))
    assert validate(TOPIC["rfid"], p, REF).flags == [Flag.LATE_UPLOAD]


def test_multiple_flags_are_all_reported() -> None:
    p = env()
    p["sensor_flags"]["co2"] = "uncalibrated"
    p["ts"] = _iso(REF - timedelta(hours=3))
    v = validate(TOPIC["env"], p, REF, {"co2_ppm": [980] * N_STUCK})
    assert v.flags == sorted([Flag.SENSOR_NOT_OK, Flag.LATE_UPLOAD, Flag.STUCK_VALUE])


def test_null_values_are_ignored_by_history_checks() -> None:
    p = env()
    p["sensor_flags"]["co2"] = "stale"
    p["co2_ppm"] = None
    assert validate(TOPIC["env"], p, REF, {"co2_ppm": [5000.0]}).flags == [Flag.SENSOR_NOT_OK]


def test_reasons_and_flags_are_closed_sets() -> None:
    assert {r.value for r in Reason} == {
        "malformed_json", "unknown_topic", "not_an_object", "schema_version", "missing_field",
        "wrong_type", "bad_ts", "topic_mismatch", "out_of_range", "invalid_enum",
        "null_with_ok_flag", "invalid_event_id", "future_ts"}
    assert {f.value for f in Flag} == {"sensor_not_ok", "stuck_value", "spike", "late_upload"}


def test_negative_status_counter_is_out_of_range() -> None:
    p = status()
    p["buffered_records"] = -1
    rejected(TOPIC["status"], p, Reason.OUT_OF_RANGE)


def test_every_simulator_message_passes_validation() -> None:
    """The simulator's own output, checked against a stream watermark, is all accepted."""
    import contextlib
    import io
    import json

    from simulator import publisher

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert publisher.main(["--dry-run", "--speed", "1e9", "--duration-h", "1", "--start",
                               "2026-10-28T05:30:00", "--outage-start-min", "10",
                               "--outage-duration-min", "5"]) == 0
    watermark = None
    for line in buf.getvalue().splitlines():
        topic, _, body = line.partition(" ")
        payload = json.loads(body)
        ts = datetime.fromisoformat(payload["ts"].replace("Z", "+00:00"))
        watermark = ts if watermark is None else max(watermark, ts)
        v = validate(topic, payload, watermark)
        assert v.accepted, (topic, v.reason, v.detail)
        assert Flag.LATE_UPLOAD not in v.flags
