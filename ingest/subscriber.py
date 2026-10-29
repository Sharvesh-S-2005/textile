"""MQTT ingest: subscribe to textile/#, validate every message, store via db.repo.

    python -m ingest.subscriber
    python -m ingest.subscriber --time-reference stream --stats-interval 5

Threading: paho's network thread only puts received messages on a queue. The main
thread does all validation and database work, so SQLite is used from one thread.

Delivery: the session is persistent (clean_session off) and acknowledgements are
manual. A QoS 1 message is acknowledged only after the batch holding it has been
committed, so a crash before commit causes redelivery (absorbed by deduplication),
never loss.
"""

from __future__ import annotations

import argparse
import json
import logging
import queue
import sys
import time
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import paho.mqtt.client as mqtt

import config
from db.repo import Repo
from ingest.validation import Reason, Verdict, history_values, parse_ts, validate

log = logging.getLogger("ingest.subscriber")

SUBSCRIBE_TOPIC_SUFFIX = "#"
WATERMARK_KINDS = frozenset({"env", "ble", "machine", "rfid", "ppe"})


def iso_ms(dt: datetime) -> str:
    """UTC ISO-8601 with milliseconds and Z (fixed width, so it sorts as text)."""
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


@dataclass(frozen=True)
class Message:
    """One received MQTT message, as handed from the network thread to the main thread."""

    topic: str
    payload: bytes
    received_at: datetime
    retained: bool = False
    mid: int = 0
    qos: int = 0


class TimeReference:
    """Supplies "now" for the future-ts and late-upload checks.

    wall:   the wall clock at receipt (real hardware).
    stream: the newest `ts` among accepted (new or duplicate), non-retained
            telemetry and event messages (replays and accelerated simulation).
            Retained messages and status messages never advance it, so a stale
            retained birth cannot make live data look like it is from the future.
            Limitation: until the first telemetry/event record is accepted there is
            no watermark, so that first record is judged against its own ts. Stream
            mode is for replays and simulation; real hardware uses `wall`.
    """

    def __init__(self, mode: str) -> None:
        """mode is 'wall' or 'stream'."""
        if mode not in ("wall", "stream"):
            raise ValueError(f"time reference must be 'wall' or 'stream', not {mode!r}")
        self.mode = mode
        self.watermark: datetime | None = None

    def reference(self, payload: Any, received_at: datetime) -> datetime:
        """The reference time for validating `payload`."""
        if self.mode == "wall":
            return received_at
        if self.watermark is not None:
            return self.watermark
        # No watermark yet: judge the first record against its own ts.
        if isinstance(payload, dict) and "ts" in payload:
            try:
                return parse_ts(payload["ts"])
            except Exception:
                pass
        return received_at

    def advance(self, kind: str, payload: dict[str, Any], retained: bool) -> None:
        """Move the watermark forward after an accepted record."""
        if self.mode != "stream" or retained or kind not in WATERMARK_KINDS:
            return
        ts = parse_ts(payload["ts"])
        if self.watermark is None or ts > self.watermark:
            self.watermark = ts


class Ingestor:
    """Validates and stores messages in batches. No MQTT knowledge."""

    def __init__(
        self,
        repo: Repo,
        time_reference: TimeReference,
        batch_size: int,
        batch_interval_s: float,
        max_unacked: int | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Args: an open Repo, the time reference, and the batch commit limits.

        max_unacked: commit once this many QoS > 0 messages await acknowledgement,
        so the broker's in-flight window never stalls delivery (None: no limit).
        """
        self.repo = repo
        self.time_reference = time_reference
        self._batch_size = batch_size
        self._batch_interval_s = batch_interval_s
        self._max_unacked = max_unacked
        self._unacked = 0
        self._clock = clock
        self._last_commit = clock()
        self._pending: list[Message] = []
        n = config.get_validation_config()["stuck_value_count"]
        # Bounded both ways: at most max_history_devices device IDs, each field a deque of n.
        self._history_len = n
        self._max_history_devices = int(config.get_ingest_config()["max_history_devices"])
        self._history: dict[str, dict[str, deque[float]]] = {}
        self._card_to_worker = {w["card_id"]: w["worker_id"] for w in config.get_workers()}
        self._excerpt = config.get_ingest_config()["payload_excerpt_chars"]
        self.counts: Counter[str] = Counter()      # received, stored, duplicate, rejected, ...
        self.accepted_by_kind: Counter[str] = Counter()
        self.rejected_by_reason: Counter[str] = Counter()
        self.flag_counts: Counter[str] = Counter()

    # ------------------------------------------------------------ per message

    def handle(self, msg: Message) -> str:
        """Process one message. Never raises; returns the outcome name.

        Outcomes: stored, duplicate, rejected, rejected_duplicate, error.
        """
        self.counts["received"] += 1
        self._pending.append(msg)
        if msg.qos > 0:
            self._unacked += 1
        try:
            with self.repo.savepoint():
                outcome = self._process(msg)
        except Exception:
            excerpt = msg.payload[: self._excerpt].decode("utf-8", errors="replace")
            log.exception("error handling message on %s: %s", msg.topic, excerpt)
            outcome = "error"
        self.counts[outcome] += 1
        return outcome

    def _process(self, msg: Message) -> str:
        """Decode, validate and write one message (inside a savepoint)."""
        received_ts = iso_ms(msg.received_at)
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            return self._reject(msg, received_ts, Reason.MALFORMED_JSON, str(exc), None)

        device_id = payload.get("device_id") if isinstance(payload, dict) else None
        device_id = device_id if isinstance(device_id, str) else None
        history = self._history.get(device_id, {}) if device_id else {}
        reference = self.time_reference.reference(payload, msg.received_at)
        verdict = validate(msg.topic, payload, reference, history)
        if not verdict.accepted:
            return self._reject(msg, received_ts, Reason(verdict.reason), verdict.detail, device_id)

        # A duplicate is still a valid record of the stream, so it advances the
        # watermark. (Advancing only on new rows let a replay of already-stored data
        # run with no watermark, and a far-future record was accepted - found by the
        # Day 2 idempotency check.)
        self.time_reference.advance(verdict.kind, payload, msg.retained)
        if not self._store(msg.topic, payload, received_ts, verdict):
            return "duplicate"
        values = history_values(verdict.kind, payload)
        stored = self._device_history(device_id) if device_id and values else None
        if stored is not None:
            for name, value in values.items():
                stored.setdefault(name, deque(maxlen=self._history_len)).append(value)
        self.accepted_by_kind[verdict.kind] += 1
        if verdict.flags:
            self.counts["flagged"] += 1
            self.flag_counts.update(verdict.flags)
        return "stored"

    def _device_history(self, device_id: str) -> dict[str, deque[float]] | None:
        """The history slot for an accepted device, created on first use; None (logged) once
        max_history_devices IDs are tracked, so a stream of new IDs cannot grow memory."""
        if device_id in self._history:
            return self._history[device_id]
        if len(self._history) >= self._max_history_devices:
            self.counts["history_untracked"] += 1
            log.error("history cap of %d devices reached; no stuck/spike history for %s",
                      self._max_history_devices, device_id)
            return None
        self._history[device_id] = {}
        return self._history[device_id]

    def _reject(self, msg: Message, received_ts: str, reason: Reason, detail: str,
                device_id: str | None) -> str:
        """Record a rejection. Logged at INFO so validation firing is visible."""
        log.info("rejected %s on %s: %s", reason.value, msg.topic, detail)
        if not self.repo.insert_rejected(msg.topic, msg.payload, reason.value, detail, received_ts, device_id):
            return "rejected_duplicate"
        self.rejected_by_reason[reason.value] += 1
        return "rejected"

    def _store(self, topic: str, payload: dict[str, Any], received_ts: str, verdict: Verdict) -> bool:
        """Write an accepted record to its table. True if the row was new."""
        flags = [str(f) for f in verdict.flags]
        if verdict.kind == "env":
            return self.repo.insert_zone_env(topic, payload, received_ts, flags)
        if verdict.kind == "ble":
            return self.repo.insert_ble_scan(topic, payload, received_ts, flags)
        if verdict.kind == "machine":
            return self.repo.insert_machine_status(topic, payload, received_ts, flags)
        if verdict.kind == "rfid":
            worker_id = self._card_to_worker.get(payload["card_id"])
            return self.repo.insert_gate_event(topic, payload, received_ts, flags, worker_id)
        if verdict.kind == "ppe":
            return self.repo.insert_ppe_entry(topic, payload, received_ts, flags)
        return self.repo.insert_device_status(topic, payload, received_ts, flags)

    # ------------------------------------------------------------ batching

    def due(self) -> bool:
        """True when the pending batch should be committed."""
        if not self._pending:
            return False
        return (len(self._pending) >= self._batch_size
                or (self._max_unacked is not None and self._unacked >= self._max_unacked)
                or self._clock() - self._last_commit >= self._batch_interval_s)

    def flush(self) -> list[Message]:
        """Commit the pending batch. Returns its messages, now safe to acknowledge."""
        self.repo.commit()
        self._last_commit = self._clock()
        self._unacked = 0
        done, self._pending = self._pending, []
        if done:
            self.counts["commits"] += 1
        return done

    def summary(self) -> str:
        """One-line counts for logging."""
        c = self.counts
        kinds = " ".join(f"{k}={self.accepted_by_kind[k]}" for k in ("env", "ble", "machine", "rfid", "ppe", "status"))
        reasons = ",".join(f"{r}={n}" for r, n in sorted(self.rejected_by_reason.items())) or "none"
        wm = self.time_reference.watermark
        return (f"received={c['received']} stored={c['stored']} [{kinds}] flagged={c['flagged']} "
                f"duplicate={c['duplicate']} rejected={c['rejected']} ({reasons}) "
                f"rejected_dup={c['rejected_duplicate']} errors={c['error']}"
                + (f" watermark={wm.strftime('%Y-%m-%dT%H:%M:%SZ')}" if wm else ""))


class Subscriber:
    """Connects an MQTT client to an Ingestor: queueing, batching, acks, reconnect, stats."""

    def __init__(self, ingestor: Ingestor, client: Any, qos: int, stats_interval_s: float,
                 poll_s: float, clock: Callable[[], float] = time.monotonic) -> None:
        """Args: the ingestor, a paho-compatible client (manual_ack), QoS, timings."""
        self.ingestor = ingestor
        self.client = client
        self._qos = qos
        self._stats_interval_s = stats_interval_s
        self._poll_s = poll_s
        self._clock = clock
        # Bounded: a full queue blocks the network thread, so the broker holds the
        # backlog (back-pressure). Nothing is dropped, unlike a deque(maxlen=...).
        self._queue: queue.Queue[Message] = queue.Queue(maxsize=int(config.get_ingest_config()["max_queued_messages"]))
        self._closing = False
        self._stop = False
        self._last_stats = clock()
        self._received_at_last_stats = 0
        self.topic = f"{config.get_mqtt_config()['topic_prefix']}/{SUBSCRIBE_TOPIC_SUFFIX}"
        client.on_connect = self._on_connect
        client.on_disconnect = self._on_disconnect
        client.on_message = self._on_message

    # ------------------------------------------------------------ paho callbacks (network thread)

    def _on_connect(self, client: Any, userdata: Any, flags: Any, reason_code: Any, properties: Any = None) -> None:
        if getattr(reason_code, "is_failure", False):
            log.error("broker refused connection: %s", reason_code)
            return
        log.info("connected; subscribing to %s at QoS %d", self.topic, self._qos)
        client.subscribe(self.topic, qos=self._qos)

    def _on_disconnect(self, client: Any, userdata: Any, flags: Any, reason_code: Any, properties: Any = None) -> None:
        if not self._closing:
            log.warning("disconnected from broker (%s); paho will reconnect with backoff", reason_code)

    def _on_message(self, client: Any, userdata: Any, msg: Any) -> None:
        if self._closing:
            return    # not acknowledged, so the broker redelivers it next session
        self._queue.put(Message(
            topic=msg.topic, payload=bytes(msg.payload), received_at=datetime.now(timezone.utc),
            retained=bool(msg.retain), mid=int(msg.mid), qos=int(msg.qos)))

    # ------------------------------------------------------------ main thread

    def _ack(self, messages: list[Message]) -> None:
        """Acknowledge QoS > 0 messages whose batch is committed."""
        for m in messages:
            if m.qos > 0:
                self.client.ack(m.mid, m.qos)

    def poll_once(self, timeout_s: float | None = None) -> int:
        """Process every queued message, commit if due, log stats if due. Returns messages handled."""
        handled = 0
        try:
            msg = self._queue.get(timeout=self._poll_s if timeout_s is None else timeout_s)
        except queue.Empty:
            msg = None
        while msg is not None:
            self.ingestor.handle(msg)
            handled += 1
            if self.ingestor.due():
                self._ack(self.ingestor.flush())
            try:
                msg = self._queue.get_nowait()
            except queue.Empty:
                msg = None
        if self.ingestor.due():
            self._ack(self.ingestor.flush())
        if self._clock() - self._last_stats >= self._stats_interval_s:
            self.log_stats()
        return handled

    def log_stats(self) -> None:
        """Log a summary line with the message rate since the last one."""
        now = self._clock()
        elapsed = max(now - self._last_stats, 1e-9)
        received = self.ingestor.counts["received"]
        rate = (received - self._received_at_last_stats) / elapsed
        log.info("%s rate=%.1f msg/s queue=%d", self.ingestor.summary(), rate, self._queue.qsize())
        self._last_stats, self._received_at_last_stats = now, received

    def start(self, host: str, port: int, keepalive_s: int, reconnect_min_s: int, reconnect_max_s: int) -> None:
        """Connect asynchronously; paho retries the first connection and every drop."""
        self.client.reconnect_delay_set(min_delay=reconnect_min_s, max_delay=reconnect_max_s)
        self.client.connect_async(host, port, keepalive_s)
        self.client.loop_start()
        log.info("connecting to %s:%d", host, port)

    def run(self, idle_exit_s: float | None = None) -> None:
        """Process messages until stop(), Ctrl-C, or (if set) idle_exit_s with no messages
        after at least one has arrived. Always shuts down cleanly."""
        last_message = None
        try:
            while not self._stop:
                if self.poll_once() > 0:
                    last_message = self._clock()
                elif (idle_exit_s is not None and last_message is not None
                      and self._clock() - last_message >= idle_exit_s):
                    log.info("no messages for %.0f s; exiting", idle_exit_s)
                    break
        except KeyboardInterrupt:
            log.info("interrupted; shutting down")
        finally:
            self.shutdown()

    def stop(self) -> None:
        """Ask run() to return after the current poll."""
        self._stop = True

    def shutdown(self) -> None:
        """Stop taking messages, store and acknowledge everything queued, then disconnect."""
        self._closing = True
        while True:
            try:
                self.ingestor.handle(self._queue.get_nowait())
            except queue.Empty:
                break
        self._ack(self.ingestor.flush())
        self.client.disconnect()
        self.client.loop_stop()
        log.info("final: %s", self.ingestor.summary())


# ---------------------------------------------------------------- CLI


def default_db_path() -> Path:
    """`paths.db_path` resolved against the project root."""
    return config.CONFIG_PATH.parent / config.get_paths()["db_path"]


def make_client() -> mqtt.Client:
    """A paho client with a stable ID, persistent session and manual acknowledgements."""
    mqtt_cfg, ingest_cfg = config.get_mqtt_config(), config.get_ingest_config()
    return mqtt.Client(
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        client_id=f"{mqtt_cfg['client_id_prefix']}-{ingest_cfg['client_id_suffix']}",
        clean_session=bool(ingest_cfg["clean_session"]),
        protocol=mqtt.MQTTv311,
        manual_ack=True,
    )


def discard_session(host: str, port: int, keepalive_s: int, client_factory: Callable[..., Any] = mqtt.Client) -> None:
    """Drop the broker's stored session (and any queued messages) for the ingest client ID.

    The persistent session (`clean_session: false`) is what lets ingest survive its own
    restarts. Before a NEW dataset is ingested, though, messages queued from an earlier,
    interrupted run must not arrive first: in `stream` mode they would move the
    watermark (Day 8A found 4 such messages flagging every later record late_upload).
    Connecting once with clean_session=True under the same ID discards that session.
    """
    mqtt_cfg, ingest_cfg = config.get_mqtt_config(), config.get_ingest_config()
    client = client_factory(
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        client_id=f"{mqtt_cfg['client_id_prefix']}-{ingest_cfg['client_id_suffix']}",
        clean_session=True,
        protocol=mqtt.MQTTv311,
    )
    client.connect(host, port, keepalive_s)
    client.loop_start()
    deadline = time.monotonic() + keepalive_s
    while not client.is_connected():
        if time.monotonic() > deadline:
            client.loop_stop()
            raise ConnectionError(f"could not connect to {host}:{port} to discard the stored session")
        time.sleep(0.05)
    client.disconnect()
    client.loop_stop()
    log.warning("discarded the broker's stored session for this client: queued messages from earlier runs dropped")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command-line options. Defaults come from config.yaml."""
    mqtt_cfg, ingest_cfg = config.get_mqtt_config(), config.get_ingest_config()
    p = argparse.ArgumentParser(prog="python -m ingest.subscriber", description=__doc__.split("\n")[0])
    p.add_argument("--broker", default=mqtt_cfg["host"])
    p.add_argument("--port", type=int, default=int(mqtt_cfg["port"]))
    p.add_argument("--db", type=Path, default=default_db_path())
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("--stats-interval", type=float, default=float(ingest_cfg["stats_interval_s"]),
                   help="seconds between summary log lines")
    p.add_argument("--time-reference", choices=["wall", "stream"], default=ingest_cfg["time_reference"],
                   help="'now' for future/late checks: wall clock, or newest accepted ts (replays)")
    p.add_argument("--discard-session", action="store_true",
                   help="drop messages queued for this client by an earlier run before subscribing "
                        "(use when starting a NEW dataset, never to resume one)")
    p.add_argument("--idle-exit-s", type=float, default=None,
                   help="exit cleanly after this many seconds without messages (scripted runs)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), stream=sys.stderr,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    mqtt_cfg, ingest_cfg = config.get_mqtt_config(), config.get_ingest_config()
    if args.discard_session:
        discard_session(args.broker, args.port, int(mqtt_cfg["keepalive_s"]))
    repo = Repo(args.db, wal_checkpoint_every=int(ingest_cfg["wal_checkpoint_every_commits"]))
    repo.init_schema()     # idempotent: creates missing tables, keeps data
    ingestor = Ingestor(repo, TimeReference(args.time_reference),
                        int(ingest_cfg["batch_size"]), float(ingest_cfg["batch_interval_s"]),
                        int(ingest_cfg["max_unacked"]))
    subscriber = Subscriber(ingestor, make_client(), int(mqtt_cfg["qos"]), args.stats_interval,
                            float(ingest_cfg["queue_poll_s"]))
    log.info("ingesting into %s (time reference: %s)", args.db, args.time_reference)
    subscriber.start(args.broker, args.port, int(mqtt_cfg["keepalive_s"]),
                     int(ingest_cfg["reconnect_min_s"]), int(ingest_cfg["reconnect_max_s"]))
    try:
        subscriber.run(args.idle_exit_s)
    finally:
        repo.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
