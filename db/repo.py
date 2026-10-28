"""All database access. No other module in the project may contain SQL.

SQLite in this phase, opened in WAL mode so the dashboard can read while the ingest
process writes. Write methods do not commit: the caller batches rows and calls
`commit()`. Every insert is `INSERT OR IGNORE` and returns True only if the row was
new, so replaying the same records (outage replay, QoS 1 redelivery, re-ingest)
never double-counts.

Time ranges are half-open [start_ts, end_ts) on the stored ISO-8601 strings, and
every time-range read is ordered by `ts` - never by arrival order.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable

SCHEMA_PATH: Path = Path(__file__).resolve().parent / "schema.sql"
BUSY_TIMEOUT_MS = 5000

# Telemetry/event/status tables and the topic type each one stores.
ACCEPTED_TABLES: dict[str, str] = {
    "zone_env": "env",
    "ble_scans": "ble",
    "machine_status": "machine",
    "gate_events": "rfid",
    "ppe_entries": "ppe",
    "device_status": "status",
}
ALL_TABLES: tuple[str, ...] = (
    "zone_env", "ble_scans", "ble_sightings", "machine_status", "gate_events",
    "ppe_entries", "device_status", "rejected_records", "zone_assignments",
    "worker_sessions",
)


def _quality(flags: list[str]) -> str:
    """'flagged' if any validation flag was raised, else 'ok'."""
    return "flagged" if flags else "ok"


def _common(topic: str, payload: dict[str, Any], received_ts: str, flags: list[str]) -> dict[str, Any]:
    """Columns every telemetry, event and status table carries."""
    return {
        "topic": topic,
        "device_id": payload["device_id"],
        "ts": payload.get("ts"),
        "received_ts": received_ts,
        "quality": _quality(flags),
        "flags": json.dumps(sorted(flags)),
        "schema_version": payload["schema_version"],
    }


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    """Convert a row to a plain dict, decoding JSON columns."""
    out = dict(row)
    for key in ("flags", "params"):
        if key in out and isinstance(out[key], str):
            out[key] = json.loads(out[key])
    return out


class Repo:
    """A connection to the project database and every query the project runs."""

    def __init__(self, db_path: Path | str) -> None:
        """Open (creating if needed) the database file and apply connection settings."""
        self.db_path = Path(db_path)
        if str(db_path) != ":memory:":
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), timeout=BUSY_TIMEOUT_MS / 1000)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")

    # ------------------------------------------------------------ lifecycle

    def init_schema(self) -> None:
        """Create every table and index from schema.sql. Safe to re-run."""
        self._conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        self._conn.commit()

    def journal_mode(self) -> str:
        """The connection's journal mode (expected 'wal')."""
        return str(self._conn.execute("PRAGMA journal_mode").fetchone()[0])

    def foreign_keys_enabled(self) -> bool:
        """True if foreign-key enforcement is on for this connection."""
        return bool(self._conn.execute("PRAGMA foreign_keys").fetchone()[0])

    def commit(self) -> None:
        """Commit the current batch."""
        self._conn.commit()

    def rollback(self) -> None:
        """Discard the current uncommitted batch."""
        self._conn.rollback()

    def close(self) -> None:
        """Commit anything pending and close the connection."""
        self._conn.commit()
        self._conn.close()

    def __enter__(self) -> Repo:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _insert(self, table: str, row: dict[str, Any]) -> bool:
        """INSERT OR IGNORE one row; True if it was new."""
        cols = ", ".join(row)
        marks = ", ".join(f":{c}" for c in row)
        cur = self._conn.execute(f"INSERT OR IGNORE INTO {table} ({cols}) VALUES ({marks})", row)
        return cur.rowcount == 1

    # ------------------------------------------------------------ writes: telemetry

    def insert_zone_env(self, topic: str, payload: dict[str, Any], received_ts: str,
                        flags: list[str]) -> bool:
        """Store one zone environmental reading. Null values stay NULL."""
        sf = payload["sensor_flags"]
        row = {
            **_common(topic, payload, received_ts, flags),
            "zone_id": payload["zone_id"],
            "noise_dba": payload["noise_dba"],
            "noise_peak_dba": payload["noise_peak_dba"],
            "pm25_ugm3": payload["pm25_ugm3"],
            "pm10_ugm3": payload["pm10_ugm3"],
            "temp_c": payload["temp_c"],
            "rh_pct": payload["rh_pct"],
            "globe_temp_c": payload["globe_temp_c"],
            "co2_ppm": payload["co2_ppm"],
            "flag_noise": sf["noise"],
            "flag_pm": sf["pm"],
            "flag_temp_rh": sf["temp_rh"],
            "flag_globe": sf["globe"],
            "flag_co2": sf["co2"],
        }
        return self._insert("zone_env", row)

    def insert_ble_scan(self, topic: str, payload: dict[str, Any], received_ts: str,
                        flags: list[str]) -> bool:
        """Store one BLE scan and one ble_sightings row per tag. True if the scan was new."""
        common = _common(topic, payload, received_ts, flags)
        scan = {**common, "zone_id": payload["zone_id"], "scan_window_s": payload["scan_window_s"],
                "n_tags": len(payload["tags"])}
        if not self._insert("ble_scans", scan):
            return False
        scan_id = self._conn.execute(
            "SELECT id FROM ble_scans WHERE device_id = ? AND ts = ? AND topic = ?",
            (common["device_id"], common["ts"], topic),
        ).fetchone()[0]
        for tag in payload["tags"]:
            self._insert("ble_sightings", {
                **common, "scan_id": scan_id, "zone_id": payload["zone_id"],
                "tag_id": tag["tag_id"], "rssi_dbm": tag["rssi_dbm"], "seen_count": tag["seen_count"],
            })
        return True

    def insert_machine_status(self, topic: str, payload: dict[str, Any], received_ts: str,
                              flags: list[str]) -> bool:
        """Store one machine reading. Null values stay NULL."""
        sf = payload["sensor_flags"]
        row = {
            **_common(topic, payload, received_ts, flags),
            "machine_id": payload["machine_id"],
            "zone_id": payload["zone_id"],
            "current_a": payload["current_a"],
            "running": int(payload["running"]),
            "vib_rms_g": payload["vib_rms_g"],
            "vib_peak_g": payload["vib_peak_g"],
            "flag_current": sf["current"],
            "flag_vibration": sf["vibration"],
        }
        return self._insert("machine_status", row)

    # ------------------------------------------------------------ writes: events

    def insert_gate_event(self, topic: str, payload: dict[str, Any], received_ts: str,
                          flags: list[str], worker_id: str | None) -> bool:
        """Store one RFID gate event, deduplicated on event_id."""
        row = {
            **_common(topic, payload, received_ts, flags),
            "event_id": payload["event_id"],
            "card_id": payload["card_id"],
            "worker_id": worker_id,
            "event": payload["event"],
            "gate_id": payload["gate_id"],
        }
        return self._insert("gate_events", row)

    def insert_ppe_entry(self, topic: str, payload: dict[str, Any], received_ts: str,
                         flags: list[str]) -> bool:
        """Store one PPE entry, deduplicated on event_id."""
        row = {
            **_common(topic, payload, received_ts, flags),
            "event_id": payload["event_id"],
            "worker_id": payload["worker_id"],
            "hearing_ppe": payload["hearing_ppe"],
            "respiratory_ppe": payload["respiratory_ppe"],
            "recorded_by": payload["recorded_by"],
        }
        return self._insert("ppe_entries", row)

    def insert_device_status(self, topic: str, payload: dict[str, Any], received_ts: str,
                             flags: list[str]) -> bool:
        """Store a birth or LWT message.

        The LWT form has no `ts` (contract v1.1a), and SQL UNIQUE treats NULLs as
        distinct, so a retained LWT redelivered on resubscribe is skipped here when
        the device's latest stored status is already that same offline form.
        """
        row = {
            **_common(topic, payload, received_ts, flags),
            "online": int(payload["online"]),
            "firmware": payload.get("firmware"),
            "uptime_s": payload.get("uptime_s"),
            "buffered_records": payload.get("buffered_records"),
        }
        if row["ts"] is None:
            latest = self._conn.execute(
                "SELECT online, ts FROM device_status WHERE device_id = ? AND topic = ? "
                "ORDER BY id DESC LIMIT 1",
                (row["device_id"], topic),
            ).fetchone()
            if latest is not None and latest["online"] == row["online"] and latest["ts"] is None:
                return False
        return self._insert("device_status", row)

    def insert_rejected(self, topic: str, raw_payload: bytes, reason: str, detail: str,
                        received_ts: str, device_id: str | None = None) -> bool:
        """Store a record that failed validation, with the exact bytes received."""
        row = {
            "received_ts": received_ts,
            "topic": topic,
            "device_id": device_id,
            "reason": reason,
            "detail": detail,
            "raw_payload": raw_payload,
            "payload_sha256": hashlib.sha256(raw_payload).hexdigest(),
        }
        return self._insert("rejected_records", row)

    # ------------------------------------------------------------ writes: derived

    def replace_zone_assignments(self, method: str, rows: Iterable[dict[str, Any]]) -> int:
        """Replace every assignment produced by `method` with `rows`. Returns rows written.

        Each row needs worker_id, tag_id, zone_id, from_ts, to_ts, confidence,
        params (dict); label is always 'estimated_personal'.
        """
        self._conn.execute("DELETE FROM zone_assignments WHERE method = ?", (method,))
        n = 0
        for r in rows:
            self._conn.execute(
                "INSERT INTO zone_assignments (worker_id, tag_id, zone_id, from_ts, to_ts, "
                "confidence, method, label, outside_session, params) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 'estimated_personal', ?, ?)",
                (r["worker_id"], r["tag_id"], r["zone_id"], r["from_ts"], r["to_ts"],
                 r["confidence"], method, int(r.get("outside_session", False)),
                 json.dumps(r["params"], sort_keys=True)),
            )
            n += 1
        return n

    def replace_worker_sessions(self, rows: Iterable[dict[str, Any]]) -> int:
        """Replace all worker sessions with `rows`. Returns rows written."""
        self._conn.execute("DELETE FROM worker_sessions")
        n = 0
        for r in rows:
            self._conn.execute(
                "INSERT INTO worker_sessions (worker_id, session_start, session_end, duration_s, "
                "shift, flags, check_in_event_id, check_out_event_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (r["worker_id"], r["session_start"], r["session_end"], r["duration_s"], r["shift"],
                 json.dumps(sorted(r["flags"])), r.get("check_in_event_id"), r.get("check_out_event_id")),
            )
            n += 1
        return n

    # ------------------------------------------------------------ reads

    def _select(self, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        """Run a query and return dict rows."""
        return [_row_to_dict(r) for r in self._conn.execute(sql, params).fetchall()]

    def get_zone_env(self, zone_id: str, start_ts: str, end_ts: str) -> list[dict[str, Any]]:
        """Zone readings with start_ts <= ts < end_ts, ordered by ts."""
        return self._select(
            "SELECT * FROM zone_env WHERE zone_id = ? AND ts >= ? AND ts < ? ORDER BY ts",
            (zone_id, start_ts, end_ts))

    def get_ble_scans(self, start_ts: str, end_ts: str) -> list[dict[str, Any]]:
        """BLE scan headers (including empty scans) in the range, ordered by ts then zone."""
        return self._select(
            "SELECT * FROM ble_scans WHERE ts >= ? AND ts < ? ORDER BY ts, zone_id",
            (start_ts, end_ts))

    def get_ble_sightings(self, start_ts: str, end_ts: str) -> list[dict[str, Any]]:
        """Raw tag sightings in the range, ordered by ts, then zone, then tag."""
        return self._select(
            "SELECT * FROM ble_sightings WHERE ts >= ? AND ts < ? ORDER BY ts, zone_id, tag_id",
            (start_ts, end_ts))

    def get_machine_status(self, machine_id: str, start_ts: str, end_ts: str) -> list[dict[str, Any]]:
        """Machine readings in the range, ordered by ts."""
        return self._select(
            "SELECT * FROM machine_status WHERE machine_id = ? AND ts >= ? AND ts < ? ORDER BY ts",
            (machine_id, start_ts, end_ts))

    def get_gate_events(self, start_ts: str | None = None, end_ts: str | None = None) -> list[dict[str, Any]]:
        """Gate events (optionally in a range), ordered by ts then event_id."""
        if start_ts is None or end_ts is None:
            return self._select("SELECT * FROM gate_events ORDER BY ts, event_id")
        return self._select(
            "SELECT * FROM gate_events WHERE ts >= ? AND ts < ? ORDER BY ts, event_id",
            (start_ts, end_ts))

    def get_zone_assignments(self, worker_id: str, start_ts: str, end_ts: str) -> list[dict[str, Any]]:
        """Assignment intervals for a worker that overlap [start_ts, end_ts), ordered by from_ts."""
        return self._select(
            "SELECT * FROM zone_assignments WHERE worker_id = ? AND from_ts < ? AND to_ts > ? "
            "ORDER BY from_ts",
            (worker_id, end_ts, start_ts))

    def get_all_zone_assignments(self, method: str | None = None) -> list[dict[str, Any]]:
        """Every assignment (optionally for one method), ordered by worker then from_ts."""
        if method is None:
            return self._select("SELECT * FROM zone_assignments ORDER BY worker_id, from_ts")
        return self._select(
            "SELECT * FROM zone_assignments WHERE method = ? ORDER BY worker_id, from_ts", (method,))

    def set_outside_session(self, assignment_ids: Iterable[int], outside: bool) -> None:
        """Mark assignments as inside or outside any worker session."""
        self._conn.executemany(
            "UPDATE zone_assignments SET outside_session = ? WHERE id = ?",
            [(int(outside), i) for i in assignment_ids])

    def get_worker_sessions(self, worker_id: str, date: str | None = None) -> list[dict[str, Any]]:
        """Sessions for a worker, optionally those starting (or, if orphaned, ending) on
        UTC calendar date `date` ('YYYY-MM-DD'). Ordered by start."""
        if date is None:
            return self._select(
                "SELECT * FROM worker_sessions WHERE worker_id = ? "
                "ORDER BY COALESCE(session_start, session_end)", (worker_id,))
        return self._select(
            "SELECT * FROM worker_sessions WHERE worker_id = ? "
            "AND substr(COALESCE(session_start, session_end), 1, 10) = ? "
            "ORDER BY COALESCE(session_start, session_end)",
            (worker_id, date))

    def get_all_worker_sessions(self) -> list[dict[str, Any]]:
        """Every session, ordered by worker then start."""
        return self._select(
            "SELECT * FROM worker_sessions ORDER BY worker_id, COALESCE(session_start, session_end)")

    def get_rejected(self) -> list[dict[str, Any]]:
        """Every rejected record, ordered by arrival."""
        return self._select("SELECT * FROM rejected_records ORDER BY received_ts, id")

    def table_counts(self) -> dict[str, int]:
        """Row count of every table."""
        return {t: int(self._conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]) for t in ALL_TABLES}

    def get_ingest_stats(self, start_ts: str, end_ts: str) -> dict[str, Any]:
        """Ingest health over records *received* in [start_ts, end_ts).

        Returns accepted and flagged counts per topic type, rejected counts by reason,
        flag-name counts, and per-device last-seen (latest received_ts and ts).
        """
        accepted: dict[str, int] = {}
        flagged: dict[str, int] = {}
        flag_counts: dict[str, int] = {}
        last_seen: dict[str, dict[str, Any]] = {}
        for table, kind in ACCEPTED_TABLES.items():
            row = self._conn.execute(
                f"SELECT COUNT(*), SUM(quality = 'flagged') FROM {table} "
                "WHERE received_ts >= ? AND received_ts < ?", (start_ts, end_ts)).fetchone()
            accepted[kind] = int(row[0])
            flagged[kind] = int(row[1] or 0)
            for (flags_json, n) in self._conn.execute(
                    f"SELECT flags, COUNT(*) FROM {table} WHERE quality = 'flagged' "
                    "AND received_ts >= ? AND received_ts < ? GROUP BY flags", (start_ts, end_ts)):
                for name in json.loads(flags_json):
                    flag_counts[name] = flag_counts.get(name, 0) + int(n)
            for device_id, last_rx, last_ts in self._conn.execute(
                    f"SELECT device_id, MAX(received_ts), MAX(ts) FROM {table} "
                    "WHERE received_ts >= ? AND received_ts < ? GROUP BY device_id", (start_ts, end_ts)):
                seen = last_seen.setdefault(device_id, {"last_received_ts": last_rx, "last_ts": last_ts})
                seen["last_received_ts"] = max(seen["last_received_ts"], last_rx)
                if last_ts is not None and (seen["last_ts"] is None or last_ts > seen["last_ts"]):
                    seen["last_ts"] = last_ts
        rejected = {
            reason: int(n) for reason, n in self._conn.execute(
                "SELECT reason, COUNT(*) FROM rejected_records "
                "WHERE received_ts >= ? AND received_ts < ? GROUP BY reason ORDER BY reason",
                (start_ts, end_ts))
        }
        return {
            "accepted": accepted,
            "flagged": flagged,
            "flagged_total": sum(flagged.values()),
            "flag_counts": dict(sorted(flag_counts.items())),
            "rejected": rejected,
            "rejected_total": sum(rejected.values()),
            "last_seen": dict(sorted(last_seen.items())),
        }
