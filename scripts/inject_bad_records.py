"""Publish one deliberately invalid record per validation rejection reason.

    python scripts/inject_bad_records.py [--broker HOST] [--port N]

Used during an end-to-end run to prove, over the real MQTT path, that every
rejection reason in ingest/validation.py fires. Every record is crafted to be
REJECTED, so nothing injected ever reaches the main tables; each carries an
`injected_by` marker so its row in rejected_records is identifiable.

This is deliberately not part of simulator/: the simulator only ever produces
contract-valid data (plus realistic sensor failures), and stays that way.

The future-ts record is built relative to the live stream: the script first reads
the newest `ts` on textile/zone/+/env, then dates the record a full day ahead of
it. (A margin of just over the 10-minute tolerance is NOT safe: at --speed 100 the
stream advances ~100 simulated seconds per wall second while the script runs, so
the record could arrive inside the tolerance, be accepted as a fake reading, and
displace the real reading with that ts as a "duplicate". The exact boundary is
covered by the unit tests instead.)
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import paho.mqtt.client as mqtt  # noqa: E402

import config  # noqa: E402
from ingest.validation import Reason  # noqa: E402

MARKER = "scripts/inject_bad_records.py"
LISTEN_TIMEOUT_S = 15.0


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_records(reference: datetime) -> list[tuple[Reason, str, bytes]]:
    """(expected reason, topic, raw payload) for every rejection reason."""
    prefix = config.get_mqtt_config()["topic_prefix"]
    schema = config.get_simulation_config()["schema_version"]
    tolerance = config.get_validation_config()["future_ts_tolerance_s"]
    ts = _iso(reference)
    env_topic = f"{prefix}/zone/Z1/env"
    env = {
        "schema_version": schema, "device_id": "ZN-Z1", "ts": ts, "zone_id": "Z1",
        "noise_dba": 91.0, "noise_peak_dba": 95.0, "pm25_ugm3": 280.0, "pm10_ugm3": 560.0,
        "temp_c": 33.0, "rh_pct": 65.0, "globe_temp_c": 35.5, "co2_ppm": 760,
        "sensor_flags": {"noise": "ok", "pm": "ok", "temp_rh": "ok", "globe": "ok", "co2": "ok"},
        "injected_by": MARKER,
    }

    def variant(**changes: Any) -> dict[str, Any]:
        p = copy.deepcopy(env)
        p.update(changes)
        return p

    missing = variant()
    del missing["pm25_ugm3"]
    bad_flag = variant()
    bad_flag["sensor_flags"]["noise"] = "broken"
    rfid = {"schema_version": schema, "device_id": "GW-01", "ts": ts, "event_id": "1234",
            "card_id": "RFID-10001", "event": "check_in", "gate_id": "GATE-A", "injected_by": MARKER}

    def enc(p: Any) -> bytes:
        return json.dumps(p).encode()

    future = reference + timedelta(days=1) + timedelta(seconds=tolerance)   # see module docstring
    return [
        (Reason.MALFORMED_JSON, env_topic, f'{{"schema_version": "{schema}", "injected_by": "{MARKER}"'.encode()),
        (Reason.UNKNOWN_TOPIC, f"{prefix}/zone/Z9/env", enc(variant(zone_id="Z9"))),
        (Reason.NOT_AN_OBJECT, env_topic, enc([MARKER])),
        (Reason.SCHEMA_VERSION, env_topic, enc(variant(schema_version="0.9"))),
        (Reason.MISSING_FIELD, env_topic, enc(missing)),
        (Reason.WRONG_TYPE, env_topic, enc(variant(noise_dba="loud"))),
        (Reason.BAD_TS, env_topic, enc(variant(ts="yesterday"))),
        (Reason.TOPIC_MISMATCH, env_topic, enc(variant(zone_id="Z2"))),
        (Reason.OUT_OF_RANGE, env_topic, enc(variant(noise_dba=200.0))),
        (Reason.INVALID_ENUM, env_topic, enc(bad_flag)),
        (Reason.NULL_WITH_OK_FLAG, env_topic, enc(variant(co2_ppm=None))),
        (Reason.INVALID_EVENT_ID, f"{prefix}/gate/rfid", enc(rfid)),
        (Reason.FUTURE_TS, env_topic, enc(variant(ts=_iso(future)))),
    ]


def latest_stream_ts(host: str, port: int) -> datetime:
    """Newest ts seen on any zone env topic within a short listening window."""
    seen: list[datetime] = []
    got = threading.Event()

    def on_message(client: Any, userdata: Any, msg: Any) -> None:
        try:
            seen.append(datetime.fromisoformat(json.loads(msg.payload)["ts"].replace("Z", "+00:00")))
        except (ValueError, KeyError, TypeError):
            return
        if len(seen) >= 7:          # one reading per zone is enough
            got.set()

    client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
                         client_id="ohs-fault-injector-listen", protocol=mqtt.MQTTv311)
    client.on_message = on_message
    client.connect(host, port, 60)
    client.subscribe(f"{config.get_mqtt_config()['topic_prefix']}/zone/+/env", qos=0)
    client.loop_start()
    got.wait(LISTEN_TIMEOUT_S)
    client.disconnect()
    client.loop_stop()
    if not seen:
        raise RuntimeError("no zone env messages seen; is the simulator publishing?")
    return max(seen)


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    mqtt_cfg = config.get_mqtt_config()
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--broker", default=mqtt_cfg["host"])
    p.add_argument("--port", type=int, default=int(mqtt_cfg["port"]))
    args = p.parse_args(argv)

    reference = latest_stream_ts(args.broker, args.port)
    records = build_records(reference)
    client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
                         client_id="ohs-fault-injector", protocol=mqtt.MQTTv311)
    client.connect(args.broker, args.port, 60)
    client.loop_start()
    for reason, topic, raw in records:
        info = client.publish(topic, raw, qos=int(mqtt_cfg["qos"]))
        info.wait_for_publish(timeout=10)
        print(f"injected {reason.value:<18} on {topic}")
        time.sleep(0.05)
    client.disconnect()
    client.loop_stop()
    print(f"{len(records)} invalid records injected relative to stream ts {_iso(reference)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
