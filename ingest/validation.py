"""Record validation: data contract v1.1, section 10.

Pure functions - no MQTT, no database, no clock. The caller supplies:
  * `reference_ts`: "now" for the future-ts and late-upload checks (the subscriber
    passes wall-clock time, or the stream watermark when replaying at speed);
  * `history`: recent accepted values per numeric field for this device, from
    the caller's in-memory ring buffers, for stuck-value and spike detection.

Rejection reasons and flag names are closed enumerations defined here and nowhere
else, so the dashboard can group by them.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import StrEnum
from typing import Any

import config


class Reason(StrEnum):
    """Why a record was rejected (not stored in the main tables)."""

    MALFORMED_JSON = "malformed_json"          # payload is not valid UTF-8 JSON
    UNKNOWN_TOPIC = "unknown_topic"            # topic matches no contract topic
    NOT_AN_OBJECT = "not_an_object"            # top-level JSON is not an object
    SCHEMA_VERSION = "schema_version"          # schema_version is not the current one
    MISSING_FIELD = "missing_field"            # a required field is absent
    WRONG_TYPE = "wrong_type"                  # a field has the wrong JSON type
    BAD_TS = "bad_ts"                          # ts is not UTC ISO-8601 with Z
    TOPIC_MISMATCH = "topic_mismatch"          # zone_id / machine_id / device_id disagrees with topic
    OUT_OF_RANGE = "out_of_range"              # a numeric value outside the contract range
    INVALID_ENUM = "invalid_enum"              # a value outside its allowed set
    NULL_WITH_OK_FLAG = "null_with_ok_flag"    # null reading while its sensor flag says ok
    INVALID_EVENT_ID = "invalid_event_id"      # event_id is not a UUID v4 string
    FUTURE_TS = "future_ts"                    # ts too far ahead of the reference time


class Flag(StrEnum):
    """Why an accepted record was stored as 'flagged'."""

    SENSOR_NOT_OK = "sensor_not_ok"    # a sensor_flags value is not "ok"
    STUCK_VALUE = "stuck_value"        # value identical to the previous N readings
    SPIKE = "spike"                    # jump from the previous reading above the threshold
    LATE_UPLOAD = "late_upload"        # ts far older than the reference time (buffered replay)


@dataclass(frozen=True)
class TopicInfo:
    """What a topic says about its message."""

    kind: str                          # env | ble | machine | rfid | ppe | status
    zone_id: str | None = None
    machine_id: str | None = None
    device_id: str | None = None


@dataclass
class Verdict:
    """The outcome of validating one record."""

    accepted: bool
    flags: list[str] = field(default_factory=list)
    reason: str | None = None
    detail: str = ""
    kind: str | None = None


class Rejection(Exception):
    """Internal: raised by the check helpers, turned into a rejected Verdict."""

    def __init__(self, reason: Reason, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


TS_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{1,6})?Z$")

ENV_SENSORS: dict[str, tuple[str, ...]] = {
    "noise": ("noise_dba", "noise_peak_dba"),
    "pm": ("pm25_ugm3", "pm10_ugm3"),
    "temp_rh": ("temp_c", "rh_pct"),
    "globe": ("globe_temp_c",),
    "co2": ("co2_ppm",),
}
MACHINE_CHANNELS: dict[str, tuple[str, ...]] = {
    "current": ("current_a",),
    "vibration": ("vib_rms_g", "vib_peak_g"),
}
INT_FIELDS: frozenset[str] = frozenset({"co2_ppm", "scan_window_s", "rssi_dbm", "seen_count", "uptime_s",
                                        "buffered_records"})
EVENT_KINDS: frozenset[str] = frozenset({"rfid", "ppe"})


# ---------------------------------------------------------------- topic


def parse_topic(topic: str) -> TopicInfo | None:
    """Classify a topic per contract section 2, or None if it is not a contract topic."""
    prefix = config.get_mqtt_config()["topic_prefix"]
    zone_ids = {z["zone_id"] for z in config.get_zones()}
    machine_ids = {m["machine_id"] for m in config.get_machines()}
    parts = topic.split("/")
    if not parts or parts[0] != prefix:
        return None
    rest = parts[1:]
    if len(rest) == 3 and rest[0] == "zone" and rest[1] in zone_ids and rest[2] in ("env", "ble"):
        return TopicInfo(kind=rest[2], zone_id=rest[1])
    if len(rest) == 3 and rest[0] == "machine" and rest[1] in machine_ids and rest[2] == "status":
        return TopicInfo(kind="machine", machine_id=rest[1])
    if rest == ["gate", "rfid"]:
        return TopicInfo(kind="rfid")
    if rest == ["ppe", "entry"]:
        return TopicInfo(kind="ppe")
    if len(rest) == 2 and rest[0] == "status" and rest[1]:
        return TopicInfo(kind="status", device_id=rest[1])
    return None


# ---------------------------------------------------------------- helpers


def parse_ts(text: Any) -> datetime:
    """Parse a contract timestamp (UTC ISO-8601 with Z). Raises Rejection(BAD_TS)."""
    if not isinstance(text, str) or not TS_PATTERN.match(text):
        raise Rejection(Reason.BAD_TS, f"ts {text!r} is not UTC ISO-8601 with Z")
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise Rejection(Reason.BAD_TS, f"ts {text!r}: {exc}") from exc


def _require(payload: Mapping[str, Any], names: Sequence[str]) -> None:
    """Every name must be present (a JSON null counts as present)."""
    missing = [n for n in names if n not in payload]
    if missing:
        raise Rejection(Reason.MISSING_FIELD, f"missing {missing}")


def _string(payload: Mapping[str, Any], name: str) -> str:
    """A required non-empty string field."""
    value = payload[name]
    if not isinstance(value, str) or not value:
        raise Rejection(Reason.WRONG_TYPE, f"{name} must be a non-empty string, got {value!r}")
    return value


def _number(name: str, value: Any) -> float:
    """A numeric field in its contract range (inclusive). Bools are not numbers."""
    is_int = isinstance(value, int) and not isinstance(value, bool)
    is_float = isinstance(value, float)
    if name in INT_FIELDS and not is_int:
        raise Rejection(Reason.WRONG_TYPE, f"{name} must be an integer, got {value!r}")
    if not (is_int or is_float):
        raise Rejection(Reason.WRONG_TYPE, f"{name} must be a number, got {value!r}")
    spec = config.get_validation_config()["fields"][name]
    if not spec["min"] <= value <= spec["max"]:
        raise Rejection(Reason.OUT_OF_RANGE, f"{name}={value} outside [{spec['min']}, {spec['max']}]")
    return float(value)


def _enum(name: str, value: Any, allowed: Sequence[str]) -> str:
    """A string field restricted to `allowed`."""
    if value not in allowed:
        raise Rejection(Reason.INVALID_ENUM, f"{name}={value!r} not in {list(allowed)}")
    return str(value)


def _sensor_values(payload: Mapping[str, Any], channels: Mapping[str, tuple[str, ...]]) -> list[str]:
    """Check sensor_flags and their fields. Returns flags raised (SENSOR_NOT_OK or none)."""
    _require(payload, ["sensor_flags", *[f for fields in channels.values() for f in fields]])
    sf = payload["sensor_flags"]
    if not isinstance(sf, dict):
        raise Rejection(Reason.WRONG_TYPE, "sensor_flags must be an object")
    missing = [c for c in channels if c not in sf]
    if missing:
        raise Rejection(Reason.MISSING_FIELD, f"sensor_flags missing {missing}")
    allowed = config.get_validation_config()["allowed_sensor_flags"]
    not_ok = False
    for channel, fields in channels.items():
        flag = _enum(f"sensor_flags.{channel}", sf[channel], allowed)
        for name in fields:
            if payload[name] is None:
                if flag == "ok":
                    raise Rejection(Reason.NULL_WITH_OK_FLAG, f"{name} is null but {channel} flag is ok")
            else:
                _number(name, payload[name])
        not_ok = not_ok or flag != "ok"
    return [Flag.SENSOR_NOT_OK] if not_ok else []


def _check_event_id(payload: Mapping[str, Any]) -> None:
    """event_id must be a canonical lower-case UUID v4 string."""
    _require(payload, ["event_id"])
    value = payload["event_id"]
    try:
        parsed = uuid.UUID(value) if isinstance(value, str) else None
    except ValueError:
        parsed = None
    if parsed is None or parsed.version != 4 or str(parsed) != value:
        raise Rejection(Reason.INVALID_EVENT_ID, f"event_id {value!r} is not a UUID v4 string")


# ---------------------------------------------------------------- per-kind checks


def _check_env(payload: Mapping[str, Any], info: TopicInfo) -> list[str]:
    _require(payload, ["zone_id"])
    if payload["zone_id"] != info.zone_id:
        raise Rejection(Reason.TOPIC_MISMATCH, f"zone_id {payload['zone_id']!r} != topic {info.zone_id}")
    return _sensor_values(payload, ENV_SENSORS)


def _check_ble(payload: Mapping[str, Any], info: TopicInfo) -> list[str]:
    _require(payload, ["zone_id", "scan_window_s", "tags"])
    if payload["zone_id"] != info.zone_id:
        raise Rejection(Reason.TOPIC_MISMATCH, f"zone_id {payload['zone_id']!r} != topic {info.zone_id}")
    _number("scan_window_s", payload["scan_window_s"])
    tags = payload["tags"]
    if not isinstance(tags, list):
        raise Rejection(Reason.WRONG_TYPE, "tags must be an array")
    seen = set()
    for i, tag in enumerate(tags):
        if not isinstance(tag, dict):
            raise Rejection(Reason.WRONG_TYPE, f"tags[{i}] must be an object")
        _require(tag, ["tag_id", "rssi_dbm", "seen_count"])
        tag_id = _string(tag, "tag_id")
        _number("rssi_dbm", tag["rssi_dbm"])
        _number("seen_count", tag["seen_count"])
        if tag_id in seen:
            raise Rejection(Reason.INVALID_ENUM, f"tag {tag_id} listed twice in one scan")
        seen.add(tag_id)
    return []


def _check_machine(payload: Mapping[str, Any], info: TopicInfo) -> list[str]:
    _require(payload, ["machine_id", "zone_id", "running"])
    if payload["machine_id"] != info.machine_id:
        raise Rejection(Reason.TOPIC_MISMATCH,
                        f"machine_id {payload['machine_id']!r} != topic {info.machine_id}")
    expected_zone = {m["machine_id"]: m["zone_id"] for m in config.get_machines()}[info.machine_id]
    if payload["zone_id"] != expected_zone:
        raise Rejection(Reason.TOPIC_MISMATCH,
                        f"zone_id {payload['zone_id']!r} != configured zone {expected_zone} of {info.machine_id}")
    if not isinstance(payload["running"], bool):
        raise Rejection(Reason.WRONG_TYPE, f"running must be a boolean, got {payload['running']!r}")
    return _sensor_values(payload, MACHINE_CHANNELS)


def _check_rfid(payload: Mapping[str, Any], info: TopicInfo) -> list[str]:
    _check_event_id(payload)
    _require(payload, ["card_id", "event", "gate_id"])
    _string(payload, "card_id")
    _string(payload, "gate_id")
    _enum("event", payload["event"], config.get_validation_config()["allowed_rfid_events"])
    return []


def _check_ppe(payload: Mapping[str, Any], info: TopicInfo) -> list[str]:
    _check_event_id(payload)
    _require(payload, ["worker_id", "hearing_ppe", "respiratory_ppe", "recorded_by"])
    _string(payload, "worker_id")
    _string(payload, "recorded_by")
    allowed = config.get_validation_config()["allowed_ppe_values"]
    _enum("hearing_ppe", payload["hearing_ppe"], allowed)
    _enum("respiratory_ppe", payload["respiratory_ppe"], allowed)
    return []


def _check_status(payload: Mapping[str, Any], info: TopicInfo) -> list[str]:
    _require(payload, ["online"])
    if payload["device_id"] != info.device_id:
        raise Rejection(Reason.TOPIC_MISMATCH, f"device_id {payload['device_id']!r} != topic {info.device_id}")
    if not isinstance(payload["online"], bool):
        raise Rejection(Reason.WRONG_TYPE, f"online must be a boolean, got {payload['online']!r}")
    optional = ("uptime_s", "buffered_records")
    if payload["online"]:
        _require(payload, ["firmware", *optional])
    if payload.get("firmware") is not None:
        _string(payload, "firmware")
    for name in optional:
        if payload.get(name) is not None:
            _number(name, payload[name])
    return []


_KIND_CHECKS = {
    "env": _check_env, "ble": _check_ble, "machine": _check_machine,
    "rfid": _check_rfid, "ppe": _check_ppe, "status": _check_status,
}


def _expected_device_id(info: TopicInfo) -> str | None:
    """Device ID the contract assigns to a zone or machine topic (section 2)."""
    if info.zone_id is not None:
        return next(z["device_id"] for z in config.get_zones() if z["zone_id"] == info.zone_id)
    if info.machine_id is not None:
        return next(m["device_id"] for m in config.get_machines() if m["machine_id"] == info.machine_id)
    return None


# ---------------------------------------------------------------- history-based flags


def history_values(kind: str, payload: Mapping[str, Any]) -> dict[str, float]:
    """Non-null numeric readings that feed stuck/spike history (env and machine only)."""
    channels = ENV_SENSORS if kind == "env" else MACHINE_CHANNELS if kind == "machine" else {}
    return {name: float(payload[name]) for fields in channels.values() for name in fields
            if payload.get(name) is not None}


def _history_flags(kind: str, payload: Mapping[str, Any],
                   history: Mapping[str, Sequence[float]]) -> list[str]:
    """STUCK_VALUE if a value equals each of the previous N; SPIKE if it jumps too far."""
    vcfg = config.get_validation_config()
    n = vcfg["stuck_value_count"]
    flags = set()
    for name, value in history_values(kind, payload).items():
        past = list(history.get(name, ()))
        if len(past) >= n and all(v == value for v in past[-n:]):
            flags.add(Flag.STUCK_VALUE)
        if past and abs(value - past[-1]) > vcfg["fields"][name]["spike_threshold"]:
            flags.add(Flag.SPIKE)
    return sorted(flags)


# ---------------------------------------------------------------- entry point


def validate(
    topic: str,
    payload: Any,
    reference_ts: datetime,
    history: Mapping[str, Sequence[float]] | None = None,
) -> Verdict:
    """Validate one parsed message against the data contract.

    Args:
        topic: MQTT topic the message arrived on.
        payload: the parsed JSON value.
        reference_ts: "now" for the future-ts and late-upload checks (tz-aware UTC).
        history: previous accepted values per field for this device, oldest first.

    Returns:
        A Verdict. Rejected verdicts carry a Reason value; accepted ones carry flags.
    """
    info = parse_topic(topic)
    if info is None:
        return Verdict(False, reason=Reason.UNKNOWN_TOPIC, detail=f"unrecognised topic {topic!r}")
    try:
        if not isinstance(payload, dict):
            raise Rejection(Reason.NOT_AN_OBJECT, f"top-level JSON is {type(payload).__name__}, not an object")
        _require(payload, ["schema_version", "device_id"])
        expected_version = config.get_simulation_config()["schema_version"]
        if payload["schema_version"] != expected_version:
            raise Rejection(Reason.SCHEMA_VERSION,
                            f"schema_version {payload['schema_version']!r} != {expected_version!r}")
        device_id = _string(payload, "device_id")
        expected_device = _expected_device_id(info)
        if expected_device is not None and device_id != expected_device:
            raise Rejection(Reason.TOPIC_MISMATCH, f"device_id {device_id!r} != {expected_device} for topic")

        # v1.1a: ts is optional on online:false status, and is then never invented.
        lwt_form = info.kind == "status" and payload.get("online") is False and "ts" not in payload
        ts = None
        if not lwt_form:
            _require(payload, ["ts"])
            ts = parse_ts(payload["ts"])

        flags = list(_KIND_CHECKS[info.kind](payload, info))

        vcfg = config.get_validation_config()
        if ts is not None:
            if ts - reference_ts > timedelta(seconds=vcfg["future_ts_tolerance_s"]):
                raise Rejection(Reason.FUTURE_TS,
                                f"ts {payload['ts']} is more than {vcfg['future_ts_tolerance_s']} s "
                                f"after reference {reference_ts.astimezone(timezone.utc).isoformat()}")
            if reference_ts - ts > timedelta(seconds=vcfg["late_upload_threshold_s"]):
                flags.append(Flag.LATE_UPLOAD)
        flags.extend(_history_flags(info.kind, payload, history or {}))
    except Rejection as rej:
        return Verdict(False, reason=rej.reason, detail=rej.detail, kind=info.kind)
    return Verdict(True, flags=sorted(set(flags)), kind=info.kind)
