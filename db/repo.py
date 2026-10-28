"""All database access. No other module in the project may contain SQL.

SQLite in this phase, opened in WAL mode so the dashboard can read while the ingest
process writes. Write methods do not commit: the caller batches rows and calls
`commit()`. Every insert is `INSERT OR IGNORE` and returns True only if the row was
new, so replaying the same records (outage replay, QoS 1 redelivery, re-ingest)
never double-counts.

Time ranges are half-open [start_ts, end_ts). Callers pass ISO-8601 strings; every
range filter and every ORDER BY uses the numeric `ts_epoch_ms` (or the derived
tables' *_ms columns), never the `ts` text - as text, "...:10.500Z" sorts before
"...:10Z". The `ts` text is kept exactly as the device published it.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

SCHEMA_PATH: Path = Path(__file__).resolve().parent / "schema.sql"
BUSY_TIMEOUT_MS = 5000
# PRAGMA user_version. 1 = Day 2 (no epoch columns); 2 = ts_epoch_ms;
# 3 = materialised exposure tables (new tables only, created by schema.sql);
# 4 = excluded-anomaly columns on the exposure tables (added by _ensure_columns).
SCHEMA_VERSION = 4
# Columns added after a table first shipped: created by schema.sql on new databases,
# added with ALTER TABLE on older ones. (table -> {column: definition})
ADDED_COLUMNS: dict[str, dict[str, str]] = {
    "personal_exposure_shift": {
        "excluded_anomalous_readings": "INTEGER NOT NULL DEFAULT 0",
        "excluded_absolute_max_candidates": "TEXT NOT NULL DEFAULT '[]'",
    },
    "zone_exposure_summary": {
        "excluded_anomalous_readings": "INTEGER NOT NULL DEFAULT 0",
        "excluded_absolute_max_candidates": "INTEGER NOT NULL DEFAULT 0",
    },
}
# Tables whose device `ts` gets a `ts_epoch_ms` companion.
TS_TABLES: tuple[str, ...] = (
    "zone_env", "ble_scans", "ble_sightings", "machine_status", "gate_events",
    "ppe_entries", "device_status",
)
# Derived tables: text column -> epoch companion.
DERIVED_EPOCH_COLUMNS: dict[str, dict[str, str]] = {
    "zone_assignments": {"from_ts": "from_epoch_ms", "to_ts": "to_epoch_ms"},
    "worker_sessions": {"session_start": "session_start_ms", "session_end": "session_end_ms"},
}

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
    "worker_sessions", "personal_exposure_shift", "personal_exposure_cumulative",
    "zone_exposure_summary", "worker_risk_index",
)

# Materialised exposure tables: the date column a recompute replaces by, the JSON
# columns, boolean columns, and the read order.
EXPOSURE_TABLES: dict[str, dict[str, Any]] = {
    "personal_exposure_shift": {
        "date_col": "shift_date", "order": "worker_id, session_start_ms",
        "json": ("session_flags", "absolute_max_events", "excluded_absolute_max_candidates", "coverage_by_metric",
                 "zones_visited", "flags", "measured_personal", "criteria_sources", "params"),
        "bool": ("wbgt_approximate",),
    },
    "personal_exposure_cumulative": {
        "date_col": "as_of_date", "order": "worker_id, window_days", "json": ("params",), "bool": ("partial",),
    },
    "worker_risk_index": {
        "date_col": "date", "order": "worker_id",
        "json": ("components", "contributions", "drivers", "ppe_status", "weights", "params"),
        "bool": ("insufficient_data",),
    },
    "zone_exposure_summary": {
        "date_col": "shift_date", "order": "zone_id, shift_start_ms",
        "json": ("coverage_by_metric", "flags", "params"), "bool": ("wbgt_approximate",),
    },
}


def to_epoch_ms(ts: str) -> int:
    """UTC ISO-8601 with Z (whole or fractional seconds) -> integer ms since the epoch."""
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError(f"timestamp without timezone: {ts!r}")
    return int(round(dt.timestamp() * 1000))


def from_epoch_ms(ms: int) -> str:
    """Integer ms since the epoch -> canonical UTC ISO-8601 with Z (ms only if non-zero)."""
    dt = datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
    text = dt.strftime("%Y-%m-%dT%H:%M:%S")
    return f"{text}.{ms % 1000:03d}Z" if ms % 1000 else f"{text}Z"


def _epoch_or_none(ts: str | None) -> int | None:
    return None if ts is None else to_epoch_ms(ts)


def _quality(flags: list[str]) -> str:
    """'flagged' if any validation flag was raised, else 'ok'."""
    return "flagged" if flags else "ok"


def _common(topic: str, payload: dict[str, Any], received_ts: str, flags: list[str]) -> dict[str, Any]:
    """Columns every telemetry, event and status table carries."""
    return {
        "topic": topic,
        "device_id": payload["device_id"],
        "ts": payload.get("ts"),
        "ts_epoch_ms": _epoch_or_none(payload.get("ts")),     # parsed once, here, at ingest
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
        """Create every table and index from schema.sql, migrating an older database
        first if needed. Safe to re-run."""
        if self._table_exists("zone_env") and not self._has_column("zone_env", "ts_epoch_ms"):
            self.migrate_to_v2()
        self._conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        self._ensure_columns()
        self._conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        self._conn.commit()

    def _ensure_columns(self) -> None:
        """Add any ADDED_COLUMNS missing from an older database (idempotent)."""
        for table, columns in ADDED_COLUMNS.items():
            for column, definition in columns.items():
                if self._table_exists(table) and not self._has_column(table, column):
                    self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def schema_version(self) -> int:
        """PRAGMA user_version of this database."""
        return int(self._conn.execute("PRAGMA user_version").fetchone()[0])

    def _table_exists(self, table: str) -> bool:
        return self._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)).fetchone() is not None

    def _has_column(self, table: str, column: str) -> bool:
        return any(r["name"] == column for r in self._conn.execute(f"PRAGMA table_info({table})"))

    def migrate_to_v2(self) -> dict[str, int]:
        """Add and backfill the epoch-ms columns on a Day-2 (v1) database.

        Each `ts` is parsed once in Python and written to its companion column; the
        `ts` text is never modified. Old text-ordered indices are dropped (schema.sql
        then creates the epoch ones). Returns rows backfilled per table. Raises if
        any row with a `ts` is left without an epoch value.
        """
        backfilled: dict[str, int] = {}
        columns = {t: {"ts": "ts_epoch_ms"} for t in TS_TABLES} | DERIVED_EPOCH_COLUMNS
        for table, pairs in columns.items():
            if not self._table_exists(table):
                continue
            for text_col, epoch_col in pairs.items():
                if not self._has_column(table, epoch_col):
                    self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {epoch_col} INTEGER")
                rows = self._conn.execute(
                    f"SELECT id, {text_col} FROM {table} WHERE {text_col} IS NOT NULL AND {epoch_col} IS NULL"
                ).fetchall()
                self._conn.executemany(f"UPDATE {table} SET {epoch_col} = ? WHERE id = ?",
                                       [(to_epoch_ms(r[1]), r[0]) for r in rows])
                left = self._conn.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE {text_col} IS NOT NULL AND {epoch_col} IS NULL"
                ).fetchone()[0]
                if left:
                    raise RuntimeError(f"migration left {left} rows of {table}.{epoch_col} empty")
                backfilled[f"{table}.{epoch_col}"] = len(rows)
        for old_index in ("ix_zone_env_zone_ts", "ix_ble_scans_zone_ts", "ix_ble_sightings_tag_ts",
                          "ix_ble_sightings_ts", "ix_machine_status_machine_ts", "ix_gate_events_worker_ts",
                          "ix_ppe_entries_worker_ts", "ix_zone_assignments_worker_from",
                          "ix_worker_sessions_worker_start"):
            self._conn.execute(f"DROP INDEX IF EXISTS {old_index}")
        self._conn.commit()
        return backfilled

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

    @contextlib.contextmanager
    def savepoint(self) -> Iterator[None]:
        """Make the enclosed writes all-or-nothing within the current batch.

        If the block raises, its writes are rolled back and the exception
        re-raised; earlier writes in the batch are kept.
        """
        # Outside a transaction a SAVEPOINT *is* the transaction and RELEASE would
        # commit; open the batch transaction first so commits stay batched.
        if not self._conn.in_transaction:
            self._conn.execute("BEGIN")
        self._conn.execute("SAVEPOINT msg")
        try:
            yield
        except BaseException:
            self._conn.execute("ROLLBACK TO msg")
            self._conn.execute("RELEASE msg")
            raise
        self._conn.execute("RELEASE msg")

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
        scan_id = self._conn.execute(       # the UNIQUE key, so exact text match is correct
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
                "from_epoch_ms, to_epoch_ms, confidence, method, label, outside_session, params) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'estimated_personal', ?, ?)",
                (r["worker_id"], r["tag_id"], r["zone_id"], r["from_ts"], r["to_ts"],
                 to_epoch_ms(r["from_ts"]), to_epoch_ms(r["to_ts"]),
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
                "INSERT INTO worker_sessions (worker_id, session_start, session_end, session_start_ms, "
                "session_end_ms, duration_s, shift, flags, check_in_event_id, check_out_event_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (r["worker_id"], r["session_start"], r["session_end"], _epoch_or_none(r["session_start"]),
                 to_epoch_ms(r["session_end"]), r["duration_s"], r["shift"],
                 json.dumps(sorted(r["flags"])), r.get("check_in_event_id"), r.get("check_out_event_id")),
            )
            n += 1
        return n

    # ------------------------------------------------------------ reads

    def _select(self, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        """Run a query and return dict rows."""
        return [_row_to_dict(r) for r in self._conn.execute(sql, params).fetchall()]

    def get_zone_env(self, zone_id: str, start_ts: str, end_ts: str) -> list[dict[str, Any]]:
        """Zone readings with start_ts <= ts < end_ts, in true time order."""
        return self._select(
            "SELECT * FROM zone_env WHERE zone_id = ? AND ts_epoch_ms >= ? AND ts_epoch_ms < ? "
            "ORDER BY ts_epoch_ms, id",
            (zone_id, to_epoch_ms(start_ts), to_epoch_ms(end_ts)))

    def get_ble_scans(self, start_ts: str, end_ts: str) -> list[dict[str, Any]]:
        """BLE scan headers (including empty scans) in the range, ordered by ts then zone."""
        return self._select(
            "SELECT * FROM ble_scans WHERE ts_epoch_ms >= ? AND ts_epoch_ms < ? ORDER BY ts_epoch_ms, zone_id",
            (to_epoch_ms(start_ts), to_epoch_ms(end_ts)))

    def get_ble_sightings(self, start_ts: str, end_ts: str) -> list[dict[str, Any]]:
        """Raw tag sightings in the range, ordered by ts, then zone, then tag."""
        return self._select(
            "SELECT * FROM ble_sightings WHERE ts_epoch_ms >= ? AND ts_epoch_ms < ? "
            "ORDER BY ts_epoch_ms, zone_id, tag_id",
            (to_epoch_ms(start_ts), to_epoch_ms(end_ts)))

    def get_ble_time_range(self) -> tuple[str, str] | None:
        """(earliest ts, latest ts) over all BLE scans, or None if there are none."""
        row = self._conn.execute("SELECT MIN(ts_epoch_ms), MAX(ts_epoch_ms) FROM ble_scans").fetchone()
        return None if row[0] is None else (from_epoch_ms(row[0]), from_epoch_ms(row[1]))

    def get_machine_status(self, machine_id: str, start_ts: str, end_ts: str) -> list[dict[str, Any]]:
        """Machine readings in the range, ordered by ts."""
        return self._select(
            "SELECT * FROM machine_status WHERE machine_id = ? AND ts_epoch_ms >= ? AND ts_epoch_ms < ? "
            "ORDER BY ts_epoch_ms, id",
            (machine_id, to_epoch_ms(start_ts), to_epoch_ms(end_ts)))

    def get_gate_events(self, start_ts: str | None = None, end_ts: str | None = None) -> list[dict[str, Any]]:
        """Gate events (optionally in a range), ordered by ts then event_id."""
        if start_ts is None or end_ts is None:
            return self._select("SELECT * FROM gate_events ORDER BY ts_epoch_ms, event_id")
        return self._select(
            "SELECT * FROM gate_events WHERE ts_epoch_ms >= ? AND ts_epoch_ms < ? ORDER BY ts_epoch_ms, event_id",
            (to_epoch_ms(start_ts), to_epoch_ms(end_ts)))

    def get_zone_assignments(self, worker_id: str, start_ts: str, end_ts: str) -> list[dict[str, Any]]:
        """Assignment intervals for a worker that overlap [start_ts, end_ts), ordered by from_ts."""
        return self._select(
            "SELECT * FROM zone_assignments WHERE worker_id = ? AND from_epoch_ms < ? AND to_epoch_ms > ? "
            "ORDER BY from_epoch_ms",
            (worker_id, to_epoch_ms(end_ts), to_epoch_ms(start_ts)))

    def get_all_zone_assignments(self, method: str | None = None) -> list[dict[str, Any]]:
        """Every assignment (optionally for one method), ordered by worker then from_ts."""
        if method is None:
            return self._select("SELECT * FROM zone_assignments ORDER BY worker_id, from_epoch_ms")
        return self._select(
            "SELECT * FROM zone_assignments WHERE method = ? ORDER BY worker_id, from_epoch_ms", (method,))

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
                "ORDER BY COALESCE(session_start_ms, session_end_ms)", (worker_id,))
        return self._select(
            "SELECT * FROM worker_sessions WHERE worker_id = ? "
            "AND COALESCE(session_start_ms, session_end_ms) >= ? "
            "AND COALESCE(session_start_ms, session_end_ms) < ? "
            "ORDER BY COALESCE(session_start_ms, session_end_ms)",
            (worker_id, to_epoch_ms(f"{date}T00:00:00Z"), to_epoch_ms(f"{date}T00:00:00Z") + 86_400_000))

    def get_all_worker_sessions(self) -> list[dict[str, Any]]:
        """Every session, ordered by worker then start."""
        return self._select(
            "SELECT * FROM worker_sessions ORDER BY worker_id, COALESCE(session_start_ms, session_end_ms)")

    # ------------------------------------------------------------ exposure tables

    def replace_exposure_rows(self, table: str, date: str, rows: Iterable[dict[str, Any]],
                              worker_id: str | None = None) -> int:
        """Replace one date's rows in a materialised exposure table (idempotent).

        Deletes the rows for `date` (optionally only `worker_id`'s) and inserts
        `rows`, every one of which must carry that date. Rows for other dates are
        never touched. Returns rows written.
        """
        spec = EXPOSURE_TABLES[table]
        date_col = spec["date_col"]
        if worker_id is None:
            self._conn.execute(f"DELETE FROM {table} WHERE {date_col} = ?", (date,))
        else:
            self._conn.execute(f"DELETE FROM {table} WHERE {date_col} = ? AND worker_id = ?", (date, worker_id))
        n = 0
        for r in rows:
            if r[date_col] != date:
                raise ValueError(f"{table} row has {date_col}={r[date_col]!r}, expected {date!r}")
            if worker_id is not None and r.get("worker_id") != worker_id:
                raise ValueError(f"{table} row is for {r.get('worker_id')!r}, expected {worker_id!r}")
            row = {k: (json.dumps(v, sort_keys=True) if k in spec["json"] and v is not None else
                       int(v) if k in spec["bool"] else v) for k, v in r.items()}
            cols = ", ".join(row)
            self._conn.execute(f"INSERT INTO {table} ({cols}) VALUES ({', '.join('?' for _ in row)})",
                               tuple(row.values()))
            n += 1
        return n

    def get_exposure_rows(self, table: str, start_date: str | None = None, end_date: str | None = None,
                          worker_id: str | None = None) -> list[dict[str, Any]]:
        """Rows of a materialised exposure table with start_date <= date <= end_date
        (inclusive, 'YYYY-MM-DD'), optionally for one worker; JSON decoded."""
        spec = EXPOSURE_TABLES[table]
        clauses, params = [], []
        if start_date is not None:
            clauses.append(f"{spec['date_col']} >= ?")
            params.append(start_date)
        if end_date is not None:
            clauses.append(f"{spec['date_col']} <= ?")
            params.append(end_date)
        if worker_id is not None:
            clauses.append("worker_id = ?")
            params.append(worker_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._conn.execute(
            f"SELECT * FROM {table} {where} ORDER BY {spec['date_col']}, {spec['order']}", params).fetchall()
        out = []
        for row in rows:
            d = dict(row)
            for k in spec["json"]:
                if d.get(k) is not None:
                    d[k] = json.loads(d[k])
            for k in spec["bool"]:
                d[k] = bool(d[k])
            out.append(d)
        return out

    def get_exposure_dates(self, table: str) -> list[str]:
        """Distinct dates present in a materialised exposure table, ascending."""
        date_col = EXPOSURE_TABLES[table]["date_col"]
        return [r[0] for r in self._conn.execute(f"SELECT DISTINCT {date_col} FROM {table} ORDER BY 1")]

    def get_data_span(self) -> tuple[str, str] | None:
        """(first ts, last ts) of zone_env telemetry - the period the system was collecting."""
        row = self._conn.execute("SELECT MIN(ts_epoch_ms), MAX(ts_epoch_ms) FROM zone_env").fetchone()
        return None if row[0] is None else (from_epoch_ms(row[0]), from_epoch_ms(row[1]))

    def get_ppe_entries(self, start_ts: str, end_ts: str) -> list[dict[str, Any]]:
        """PPE entries with start_ts <= ts < end_ts, in time order (display only, never a dose input)."""
        return self._select(
            "SELECT * FROM ppe_entries WHERE ts_epoch_ms >= ? AND ts_epoch_ms < ? ORDER BY ts_epoch_ms, event_id",
            (to_epoch_ms(start_ts), to_epoch_ms(end_ts)))

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
            for device_id, last_rx, last_ms in self._conn.execute(
                    f"SELECT device_id, MAX(received_ts), MAX(ts_epoch_ms) FROM {table} "
                    "WHERE received_ts >= ? AND received_ts < ? GROUP BY device_id", (start_ts, end_ts)):
                seen = last_seen.setdefault(device_id, {"last_received_ts": last_rx, "last_ts_ms": None})
                seen["last_received_ts"] = max(seen["last_received_ts"], last_rx)
                if last_ms is not None and (seen["last_ts_ms"] is None or last_ms > seen["last_ts_ms"]):
                    seen["last_ts_ms"] = last_ms
        for seen in last_seen.values():
            ms = seen.pop("last_ts_ms")
            seen["last_ts"] = None if ms is None else from_epoch_ms(ms)
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
