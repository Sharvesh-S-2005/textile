"""Simulator publisher: the ONLY simulator file that talks to MQTT.

Ties the zone, machine and worker models together and publishes messages in exactly
the format of docs/data_contract.md. When the ESP32 firmware arrives it replaces
this program and nothing downstream changes.

Every payload's `ts` is *simulated* time in UTC with a `Z` suffix.

Usage:
    python -m simulator.publisher --dry-run --duration-h 0.1
    python -m simulator.publisher --speed 100 --duration-h 8
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol
from zoneinfo import ZoneInfo

import numpy as np
import paho.mqtt.client as mqtt

import config
from simulator.machine_model import MachineModel
from simulator.profiles import ZoneProfile
from simulator.schedule import ShiftSchedule
from simulator.worker_model import WorkerModel

log = logging.getLogger("simulator.publisher")

SUMMARY_EVERY = timedelta(minutes=10)
CONNECT_TIMEOUT_S = 10.0


def iso_z(ts: datetime) -> str:
    """Format a UTC datetime as ISO-8601 with a `Z` suffix, whole seconds."""
    return ts.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------- sinks


class Sink(Protocol):
    """Destination for messages: real MQTT, or stdout in dry-run mode."""

    def publish(self, device_id: str, topic: str, payload: dict[str, Any], retain: bool) -> None:
        """Publish one payload on behalf of `device_id`."""

    def close(self) -> None:
        """Release resources."""


class DryRunSink:
    """Prints `topic payload` lines to stdout, in the same layout as `mosquitto_sub -v`."""

    def __init__(self, stream: Any = None) -> None:
        """Write to `stream`, or to sys.stdout when not given."""
        self._stream = stream

    def publish(self, device_id: str, topic: str, payload: dict[str, Any], retain: bool) -> None:
        """Print one message."""
        stream = self._stream or sys.stdout
        stream.write(f"{topic} {json.dumps(payload, separators=(',', ':'))}\n")

    def close(self) -> None:
        """Flush stdout."""
        (self._stream or sys.stdout).flush()


class MqttSink:
    """One MQTT client per device, each with client ID = device ID and its own LWT."""

    def __init__(
        self, host: str, port: int, keepalive_s: int, qos: int,
        device_ids: list[str], status_topic: dict[str, str], schema_version: str,
    ) -> None:
        """Connect a client for every device. Raises ConnectionError if the broker is unreachable."""
        self._qos = qos
        self._clients: dict[str, mqtt.Client] = {}
        for device_id in device_ids:
            client = mqtt.Client(
                callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
                client_id=device_id,
                protocol=mqtt.MQTTv311,
            )
            # An LWT is registered at connect time, so it cannot carry the time of death.
            will = {"schema_version": schema_version, "device_id": device_id, "online": False}
            client.will_set(status_topic[device_id], json.dumps(will), qos=qos, retain=True)
            try:
                client.connect(host, port, keepalive_s)
            except OSError as exc:
                self.close()
                raise ConnectionError(f"cannot reach MQTT broker at {host}:{port}: {exc}") from exc
            client.loop_start()
            self._clients[device_id] = client
        deadline = time.monotonic() + CONNECT_TIMEOUT_S
        while not all(c.is_connected() for c in self._clients.values()):
            if time.monotonic() > deadline:
                self.close()
                raise ConnectionError(f"MQTT broker at {host}:{port} did not accept all clients")
            time.sleep(0.05)
        log.info("connected %d device clients to %s:%d", len(self._clients), host, port)

    def publish(self, device_id: str, topic: str, payload: dict[str, Any], retain: bool) -> None:
        """Publish via the device's own client. Logs an error if the publish is refused."""
        info = self._clients[device_id].publish(topic, json.dumps(payload), qos=self._qos, retain=retain)
        if info.rc != mqtt.MQTT_ERR_SUCCESS:
            log.error("publish failed on %s (rc=%s)", topic, info.rc)

    def close(self) -> None:
        """Flush outstanding messages and disconnect every client cleanly."""
        for client in self._clients.values():
            client.disconnect()
            client.loop_stop()
        self._clients = {}


# ---------------------------------------------------------------- simulation


@dataclass(frozen=True)
class Outage:
    """One zone node offline during [start, end)."""

    zone_id: str
    start: datetime
    end: datetime


class Simulation:
    """Owns every model and turns each simulated tick into contract payloads."""

    def __init__(self, start: datetime, duration: timedelta, seed: int, outage: Outage | None) -> None:
        """Build all models from config.yaml for the window [start, start + duration)."""
        self.start = start
        self.end = start + duration
        self.outage = outage
        self.sim = config.get_simulation_config()
        mqtt_cfg = config.get_mqtt_config()
        self._prefix = mqtt_cfg["topic_prefix"]
        self._schema = self.sim["schema_version"]
        self._ble = self.sim["ble_model"]
        self._adjacency = config.get_adjacency()
        devices = config.get_devices()
        self.gate = devices["gate"]
        self.ppe_device_id: str = devices["ppe_app"]["device_id"]

        root = np.random.default_rng(seed)
        schedule = ShiftSchedule(config.get_shifts())
        self.zones = config.get_zones()
        self.profiles = {
            z["zone_id"]: ZoneProfile.from_config(z["zone_id"], rng, start)
            for z, rng in zip(self.zones, root.spawn(len(self.zones)))
        }
        machines_cfg = config.get_machines()
        self.machines = [
            MachineModel.from_config(m["machine_id"], schedule, rng, start)
            for m, rng in zip(machines_cfg, root.spawn(len(machines_cfg)))
        ]
        self.workers = WorkerModel.from_config(schedule, root.spawn(1)[0], start, self.end)
        self._ble_rng = root.spawn(1)[0]
        self._last_event_ts: dict[str, datetime] = {}

        self.zone_device = {z["zone_id"]: z["device_id"] for z in self.zones}
        self.device_ids = ([z["device_id"] for z in self.zones]
                           + [m.device_id for m in self.machines]
                           + [self.gate["device_id"], self.ppe_device_id])
        self.status_topic = {d: f"{self._prefix}/status/{d}" for d in self.device_ids}

    # ----------------------------------------------------------- payloads

    def _header(self, device_id: str, ts: datetime) -> dict[str, Any]:
        """The three fields every payload carries."""
        return {"schema_version": self._schema, "device_id": device_id, "ts": iso_z(ts)}

    def status_payload(self, device_id: str, ts: datetime, online: bool, buffered: int = 0) -> dict[str, Any]:
        """Device status (section 9)."""
        return {
            **self._header(device_id, ts),
            "online": online,
            "firmware": self.sim["firmware_tag"],
            "uptime_s": int((ts - self.start).total_seconds()),
            "buffered_records": buffered,
        }

    def env_message(self, zone_id: str, ts: datetime, running_factors: list[float]) -> tuple[str, dict]:
        """Zone environmental telemetry (section 4)."""
        payload = {**self._header(self.zone_device[zone_id], ts), "zone_id": zone_id}
        payload.update(self.profiles[zone_id].reading(ts, running_factors))
        return f"{self._prefix}/zone/{zone_id}/env", payload

    def machine_message(self, machine: MachineModel, ts: datetime, reading: dict) -> tuple[str, dict]:
        """Machine status (section 6)."""
        payload = {**self._header(machine.device_id, ts), "machine_id": machine.machine_id, **reading}
        return f"{self._prefix}/machine/{machine.machine_id}/status", payload

    def _sighting(self, kind: str) -> dict[str, int] | None:
        """RSSI and seen_count for one tag at one node, or None if not heard this window."""
        spec = self._ble[kind]
        rng = self._ble_rng
        if rng.random() >= spec["detect_probability"]:
            return None
        ranges = config.get_validation_config()["fields"]
        rssi = spec["rssi_dbm"]["mean"] + spec["rssi_dbm"]["sd"] * rng.standard_normal()
        rssi = int(round(min(max(rssi, ranges["rssi_dbm"]["min"]), ranges["rssi_dbm"]["max"])))
        adverts = self.sim["ble_interval_s"] / self._ble["advert_interval_s"]
        frac = rng.uniform(spec["seen_fraction"]["min"], spec["seen_fraction"]["max"])
        return {"rssi_dbm": rssi, "seen_count": max(1, int(round(adverts * frac)))}

    def ble_messages(self, ts: datetime) -> list[tuple[str, dict]]:
        """One BLE scan per zone node (section 5), with leakage into adjacent zones."""
        tags: dict[str, list[dict]] = {z["zone_id"]: [] for z in self.zones}
        for worker_id, true_zone in sorted(self.workers.zones_at(ts).items()):
            tag_id = self.workers.tag_id(worker_id)
            heard_by = [(true_zone, "same_zone")] + [(n, "adjacent_zone") for n in self._adjacency[true_zone]]
            for zone_id, kind in heard_by:
                sighting = self._sighting(kind)
                if sighting is not None:
                    tags[zone_id].append({"tag_id": tag_id, **sighting})
        messages = []
        for zone_id, seen in tags.items():
            payload = {**self._header(self.zone_device[zone_id], ts), "zone_id": zone_id,
                       "scan_window_s": self.sim["ble_interval_s"],
                       "tags": sorted(seen, key=lambda t: t["tag_id"])}
            messages.append((f"{self._prefix}/zone/{zone_id}/ble", payload))
        return messages

    def _event_ts(self, device_id: str, ts: datetime) -> datetime:
        """Whole-second event time, at least 1 s after the device's previous event.

        A gate reader or supervisor app handles one record at a time, and this keeps
        the backend's (device_id, ts, topic) dedup key unique.
        """
        ts = ts.replace(microsecond=0)
        last = self._last_event_ts.get(device_id)
        if last is not None and ts <= last:
            ts = last + timedelta(seconds=1)
        self._last_event_ts[device_id] = ts
        return ts

    def rfid_message(self, event: Any) -> tuple[str, dict]:
        """Gate check-in / check-out (section 7)."""
        ts = self._event_ts(self.gate["device_id"], event.ts)
        payload = {**self._header(self.gate["device_id"], ts), "card_id": event.card_id,
                   "event": event.event, "gate_id": self.gate["gate_id"]}
        return f"{self._prefix}/gate/rfid", payload

    def ppe_message(self, event: Any) -> tuple[str, dict]:
        """Supervisor PPE entry (section 8)."""
        ts = self._event_ts(self.ppe_device_id, event.ts)
        payload = {**self._header(self.ppe_device_id, ts), "worker_id": event.worker_id,
                   "hearing_ppe": event.hearing_ppe, "respiratory_ppe": event.respiratory_ppe,
                   "recorded_by": event.recorded_by}
        return f"{self._prefix}/ppe/entry", payload

    # ----------------------------------------------------------- run loop

    def run(self, sink: Sink, speed: float) -> None:
        """Publish the whole window, pacing wall-clock time at `speed` x simulated time.

        Ctrl-C stops the run early; every device still publishes `online: false`.
        """
        sim = self.sim
        tick_s = math.gcd(sim["env_interval_s"], sim["ble_interval_s"], sim["machine_interval_s"])
        n_ticks = int((self.end - self.start).total_seconds() // tick_s)
        counts: Counter[str] = Counter()
        buffer: list[tuple[str, dict]] = []
        rfid = [e for e in self.workers.rfid_events if self.start <= e.ts < self.end]
        ppe = [e for e in self.workers.ppe_events if self.start <= e.ts < self.end]
        wall_start = time.monotonic()
        next_summary = self.start + SUMMARY_EVERY
        ts = self.start
        outage_zone = self.outage.zone_id if self.outage else None
        outage_device = self.zone_device[outage_zone] if outage_zone else None
        clear_backlog_status = False

        def send(device_id: str, topic: str, payload: dict, kind: str, retain: bool = False) -> None:
            in_outage = (self.outage is not None and device_id == outage_device
                         and self.outage.start <= ts < self.outage.end)
            if in_outage:
                buffer.append((topic, payload))
                counts["buffered"] += 1
                return
            sink.publish(device_id, topic, payload, retain)
            counts[kind] += 1

        try:
            for device_id in self.device_ids:
                send(device_id, self.status_topic[device_id],
                     self.status_payload(device_id, ts, True), "status", retain=True)

            for i in range(n_ticks):
                ts = self.start + timedelta(seconds=i * tick_s)
                elapsed_s = i * tick_s
                target_wall = wall_start + elapsed_s / speed
                delay = target_wall - time.monotonic()
                if delay > 0:
                    time.sleep(delay)

                if clear_backlog_status:
                    # Replay finished during the previous tick; a new ts keeps the
                    # (device_id, ts, topic) dedup key unique.
                    send(outage_device, self.status_topic[outage_device],
                         self.status_payload(outage_device, ts, True, 0), "status", retain=True)
                    clear_backlog_status = False
                if self.outage and buffer and ts >= self.outage.end:
                    self._replay(sink, outage_device, ts, buffer, counts)
                    clear_backlog_status = True

                while rfid and rfid[0].ts <= ts:
                    send(self.gate["device_id"], *self.rfid_message(rfid.pop(0)), "rfid")
                while ppe and ppe[0].ts <= ts:
                    send(self.ppe_device_id, *self.ppe_message(ppe.pop(0)), "ppe")

                if elapsed_s % sim["machine_interval_s"] == 0 or elapsed_s % sim["env_interval_s"] == 0:
                    readings = {m.machine_id: m.reading(ts) for m in self.machines}
                if elapsed_s % sim["machine_interval_s"] == 0:
                    for m in self.machines:
                        send(m.device_id, *self.machine_message(m, ts, readings[m.machine_id]), "machine")
                if elapsed_s % sim["env_interval_s"] == 0:
                    for z in self.zones:
                        factors = [m.noise_energy_factor(ts) for m in self.machines
                                   if m.zone_id == z["zone_id"] and readings[m.machine_id]["running"]]
                        send(z["device_id"], *self.env_message(z["zone_id"], ts, factors), "env")
                if elapsed_s % sim["ble_interval_s"] == 0:
                    for topic, payload in self.ble_messages(ts):
                        send(payload["device_id"], topic, payload, "ble")

                if ts >= next_summary:
                    self._log_summary(ts, counts, wall_start)
                    next_summary += SUMMARY_EVERY
            ts = self.end
        except KeyboardInterrupt:
            log.warning("interrupted at simulated %s; shutting down", iso_z(ts))
        finally:
            if buffer:
                log.warning("%d records still buffered on %s at end of run", len(buffer), outage_device)
            for device_id in self.device_ids:
                sink.publish(device_id, self.status_topic[device_id],
                             self.status_payload(device_id, ts, False, len(buffer) if device_id == outage_device else 0),
                             retain=True)
                counts["status"] += 1
            self._log_summary(ts, counts, wall_start)
            sink.close()

    def _replay(self, sink: Sink, device_id: str, ts: datetime,
                buffer: list[tuple[str, dict]], counts: Counter[str]) -> None:
        """Reconnect after an outage: announce the backlog, then replay it with original ts."""
        topic = self.status_topic[device_id]
        log.info("%s reconnected at %s; replaying %d buffered records", device_id, iso_z(ts), len(buffer))
        sink.publish(device_id, topic, self.status_payload(device_id, ts, True, len(buffer)), retain=True)
        for buffered_topic, payload in buffer:
            sink.publish(device_id, buffered_topic, payload, retain=False)
            counts["replayed"] += 1
        counts["status"] += 1
        buffer.clear()

    def _log_summary(self, ts: datetime, counts: Counter[str], wall_start: float) -> None:
        """One-line progress summary."""
        kinds = ("env", "ble", "machine", "rfid", "ppe", "status", "buffered", "replayed")
        parts = " ".join(f"{k}={counts[k]}" for k in kinds)
        log.info("sim=%s wall=%.1fs %s", iso_z(ts), time.monotonic() - wall_start, parts)


# ---------------------------------------------------------------- CLI


def _default_start() -> datetime:
    """Today at 06:00 local mill time (start of shift A), as UTC."""
    tz = ZoneInfo(config.get_shifts()["timezone"])
    first = config.get_shifts()["definitions"][0]["start"]
    hours, minutes = (int(x) for x in first.split(":"))
    today = datetime.now(tz).date()
    return datetime(today.year, today.month, today.day, hours, minutes, tzinfo=tz).astimezone(timezone.utc)


def _parse_start(text: str) -> datetime:
    """Parse an ISO-8601 start time; naive values are taken as local mill time."""
    ts = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=ZoneInfo(config.get_shifts()["timezone"]))
    return ts.astimezone(timezone.utc)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command-line options. Defaults come from config.yaml."""
    sim = config.get_simulation_config()
    mqtt_cfg = config.get_mqtt_config()
    outage = sim["outage"]
    p = argparse.ArgumentParser(prog="python -m simulator.publisher", description=__doc__.split("\n")[0])
    p.add_argument("--speed", type=float, default=float(sim["speed_factor"]),
                   help="simulated-time multiplier (100 = 8 h shift in ~5 min)")
    p.add_argument("--start", type=_parse_start, default=None,
                   help="simulated start, ISO-8601 (default: today 06:00 Asia/Kolkata)")
    p.add_argument("--duration-h", type=float, default=8.0, help="simulated hours to run")
    p.add_argument("--seed", type=int, default=int(sim["random_seed"]))
    p.add_argument("--dry-run", action="store_true", help="print payloads instead of publishing")
    p.add_argument("--broker", default=mqtt_cfg["host"])
    p.add_argument("--port", type=int, default=int(mqtt_cfg["port"]))
    p.add_argument("--outage-zone", default=outage["zone_id"], help="zone node that goes offline")
    p.add_argument("--outage-start-min", type=float, default=float(outage["start_offset_min"]),
                   help="simulated minutes after start when the outage begins")
    p.add_argument("--outage-duration-min", type=float, default=float(outage["duration_min"]))
    p.add_argument("--no-outage", action="store_true", help="disable the outage demo")
    args = p.parse_args(argv)
    if args.speed <= 0:
        p.error("--speed must be positive")
    if args.duration_h <= 0:
        p.error("--duration-h must be positive")
    if args.outage_zone not in {z["zone_id"] for z in config.get_zones()}:
        p.error(f"--outage-zone {args.outage_zone} is not a configured zone")
    return args


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    start = args.start or _default_start()
    outage = None
    if config.get_simulation_config()["outage"]["enabled"] and not args.no_outage:
        o_start = start + timedelta(minutes=args.outage_start_min)
        outage = Outage(args.outage_zone, o_start, o_start + timedelta(minutes=args.outage_duration_min))
        log.info("outage demo: %s offline %s -> %s", args.outage_zone, iso_z(outage.start), iso_z(outage.end))

    simulation = Simulation(start, timedelta(hours=args.duration_h), args.seed, outage)
    if args.dry_run:
        sink: Sink = DryRunSink()
    else:
        mqtt_cfg = config.get_mqtt_config()
        try:
            sink = MqttSink(args.broker, args.port, int(mqtt_cfg["keepalive_s"]), int(mqtt_cfg["qos"]),
                            simulation.device_ids, simulation.status_topic,
                            config.get_simulation_config()["schema_version"])
        except ConnectionError as exc:
            log.error("%s. Is Mosquitto running? Use --dry-run to run without a broker.", exc)
            return 2
    log.info("simulating %s -> %s at speed %gx (seed %d)",
             iso_z(simulation.start), iso_z(simulation.end), args.speed, args.seed)
    simulation.run(sink, args.speed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
