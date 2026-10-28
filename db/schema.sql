-- SQLite schema for the textile OHS backend. Safe to re-run (IF NOT EXISTS).
--
-- Timestamps are UTC ISO-8601 strings with a Z suffix. `ts` is the device's own
-- timestamp, stored exactly as published; `received_ts` is when the backend got
-- the message. Rows can arrive out of `ts` order (outage replay).
--
-- `ts_epoch_ms` is `ts` parsed once at ingest (UTC milliseconds since the epoch).
-- Every ordered or ranged read uses it, never the `ts` text: as text,
-- "...:10.500Z" sorts BEFORE "...:10Z" because '.' (0x2E) < 'Z' (0x5A).
--
-- Every telemetry table carries: id, topic, device_id, ts, received_ts, quality
-- ('ok' | 'flagged'), flags (JSON array of flag names, '[]' when clean),
-- schema_version, ts_epoch_ms. A value the device reported as null is stored as NULL; the
-- reason is in the per-sensor flag column and in `flags`.

-- ---------------------------------------------------------------- telemetry

CREATE TABLE IF NOT EXISTS zone_env (
    id              INTEGER PRIMARY KEY,
    topic           TEXT NOT NULL,
    device_id       TEXT NOT NULL,
    ts              TEXT NOT NULL,
    ts_epoch_ms     INTEGER NOT NULL,
    received_ts     TEXT NOT NULL,
    quality         TEXT NOT NULL CHECK (quality IN ('ok', 'flagged')),
    flags           TEXT NOT NULL DEFAULT '[]',
    schema_version  TEXT NOT NULL,
    zone_id         TEXT NOT NULL,
    noise_dba       REAL,
    noise_peak_dba  REAL,
    pm25_ugm3       REAL,   -- raw, uncorrected (humidity correction is exposure/dust.py)
    pm10_ugm3       REAL,   -- raw, uncorrected
    temp_c          REAL,
    rh_pct          REAL,
    globe_temp_c    REAL,
    co2_ppm         INTEGER,
    flag_noise      TEXT NOT NULL,
    flag_pm         TEXT NOT NULL,
    flag_temp_rh    TEXT NOT NULL,
    flag_globe      TEXT NOT NULL,
    flag_co2        TEXT NOT NULL,
    UNIQUE (device_id, ts, topic)
);
CREATE INDEX IF NOT EXISTS ix_zone_env_zone_epoch ON zone_env (zone_id, ts_epoch_ms);
CREATE INDEX IF NOT EXISTS ix_zone_env_received ON zone_env (received_ts);

-- One row per BLE scan message, including scans that saw no tags (meaningful:
-- the node was alive and nobody was in range).
CREATE TABLE IF NOT EXISTS ble_scans (
    id              INTEGER PRIMARY KEY,
    topic           TEXT NOT NULL,
    device_id       TEXT NOT NULL,
    ts              TEXT NOT NULL,
    ts_epoch_ms     INTEGER NOT NULL,
    received_ts     TEXT NOT NULL,
    quality         TEXT NOT NULL CHECK (quality IN ('ok', 'flagged')),
    flags           TEXT NOT NULL DEFAULT '[]',
    schema_version  TEXT NOT NULL,
    zone_id         TEXT NOT NULL,
    scan_window_s   INTEGER NOT NULL,
    n_tags          INTEGER NOT NULL,
    UNIQUE (device_id, ts, topic)
);
CREATE INDEX IF NOT EXISTS ix_ble_scans_zone_epoch ON ble_scans (zone_id, ts_epoch_ms);
CREATE INDEX IF NOT EXISTS ix_ble_scans_epoch ON ble_scans (ts_epoch_ms);
CREATE INDEX IF NOT EXISTS ix_ble_scans_received ON ble_scans (received_ts);

-- One row per tag seen per zone per scan window (raw, unresolved).
CREATE TABLE IF NOT EXISTS ble_sightings (
    id              INTEGER PRIMARY KEY,
    scan_id         INTEGER NOT NULL REFERENCES ble_scans (id),
    topic           TEXT NOT NULL,
    device_id       TEXT NOT NULL,
    ts              TEXT NOT NULL,
    ts_epoch_ms     INTEGER NOT NULL,
    received_ts     TEXT NOT NULL,
    quality         TEXT NOT NULL CHECK (quality IN ('ok', 'flagged')),
    flags           TEXT NOT NULL DEFAULT '[]',
    schema_version  TEXT NOT NULL,
    zone_id         TEXT NOT NULL,
    tag_id          TEXT NOT NULL,
    rssi_dbm        INTEGER NOT NULL,
    seen_count      INTEGER NOT NULL,
    UNIQUE (device_id, ts, topic, tag_id)
);
CREATE INDEX IF NOT EXISTS ix_ble_sightings_tag_epoch ON ble_sightings (tag_id, ts_epoch_ms);
CREATE INDEX IF NOT EXISTS ix_ble_sightings_epoch ON ble_sightings (ts_epoch_ms);

CREATE TABLE IF NOT EXISTS machine_status (
    id              INTEGER PRIMARY KEY,
    topic           TEXT NOT NULL,
    device_id       TEXT NOT NULL,
    ts              TEXT NOT NULL,
    ts_epoch_ms     INTEGER NOT NULL,
    received_ts     TEXT NOT NULL,
    quality         TEXT NOT NULL CHECK (quality IN ('ok', 'flagged')),
    flags           TEXT NOT NULL DEFAULT '[]',
    schema_version  TEXT NOT NULL,
    machine_id      TEXT NOT NULL,
    zone_id         TEXT NOT NULL,
    current_a       REAL,
    running         INTEGER NOT NULL CHECK (running IN (0, 1)),
    vib_rms_g       REAL,
    vib_peak_g      REAL,
    flag_current    TEXT NOT NULL,
    flag_vibration  TEXT NOT NULL,
    UNIQUE (device_id, ts, topic)
);
CREATE INDEX IF NOT EXISTS ix_machine_status_machine_epoch ON machine_status (machine_id, ts_epoch_ms);
CREATE INDEX IF NOT EXISTS ix_machine_status_received ON machine_status (received_ts);

-- ---------------------------------------------------------------- events

CREATE TABLE IF NOT EXISTS gate_events (
    id              INTEGER PRIMARY KEY,
    event_id        TEXT NOT NULL UNIQUE,
    topic           TEXT NOT NULL,
    device_id       TEXT NOT NULL,
    ts              TEXT NOT NULL,
    ts_epoch_ms     INTEGER NOT NULL,
    received_ts     TEXT NOT NULL,
    quality         TEXT NOT NULL CHECK (quality IN ('ok', 'flagged')),
    flags           TEXT NOT NULL DEFAULT '[]',
    schema_version  TEXT NOT NULL,
    card_id         TEXT NOT NULL,
    worker_id       TEXT,           -- resolved from card_id via config; NULL if unknown card
    event           TEXT NOT NULL CHECK (event IN ('check_in', 'check_out')),
    gate_id         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_gate_events_worker_epoch ON gate_events (worker_id, ts_epoch_ms);
CREATE INDEX IF NOT EXISTS ix_gate_events_epoch ON gate_events (ts_epoch_ms);
CREATE INDEX IF NOT EXISTS ix_gate_events_received ON gate_events (received_ts);

-- Supervisor-observed PPE. Never used to reduce a calculated dose.
CREATE TABLE IF NOT EXISTS ppe_entries (
    id              INTEGER PRIMARY KEY,
    event_id        TEXT NOT NULL UNIQUE,
    topic           TEXT NOT NULL,
    device_id       TEXT NOT NULL,
    ts              TEXT NOT NULL,
    ts_epoch_ms     INTEGER NOT NULL,
    received_ts     TEXT NOT NULL,
    quality         TEXT NOT NULL CHECK (quality IN ('ok', 'flagged')),
    flags           TEXT NOT NULL DEFAULT '[]',
    schema_version  TEXT NOT NULL,
    worker_id       TEXT NOT NULL,
    hearing_ppe     TEXT NOT NULL,
    respiratory_ppe TEXT NOT NULL,
    recorded_by     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_ppe_entries_worker_epoch ON ppe_entries (worker_id, ts_epoch_ms);
CREATE INDEX IF NOT EXISTS ix_ppe_entries_received ON ppe_entries (received_ts);

-- Birth (online: true, with ts) and LWT (online: false, ts NULL per contract v1.1a).
CREATE TABLE IF NOT EXISTS device_status (
    id               INTEGER PRIMARY KEY,
    topic            TEXT NOT NULL,
    device_id        TEXT NOT NULL,
    ts               TEXT,           -- NULL for the LWT form; never invented
    ts_epoch_ms      INTEGER,        -- NULL exactly when ts is NULL
    received_ts      TEXT NOT NULL,
    quality          TEXT NOT NULL CHECK (quality IN ('ok', 'flagged')),
    flags            TEXT NOT NULL DEFAULT '[]',
    schema_version   TEXT NOT NULL,
    online           INTEGER NOT NULL CHECK (online IN (0, 1)),
    firmware         TEXT,
    uptime_s         INTEGER,
    buffered_records INTEGER,
    UNIQUE (device_id, ts, topic)    -- NULL ts rows are deduplicated in db/repo.py
);
CREATE INDEX IF NOT EXISTS ix_device_status_device ON device_status (device_id, received_ts);
CREATE INDEX IF NOT EXISTS ix_device_status_epoch ON device_status (ts_epoch_ms);

-- ---------------------------------------------------------------- rejections

CREATE TABLE IF NOT EXISTS rejected_records (
    id              INTEGER PRIMARY KEY,
    received_ts     TEXT NOT NULL,
    topic           TEXT NOT NULL,
    device_id       TEXT,           -- if it could be read from the payload
    reason          TEXT NOT NULL,  -- closed set, defined in ingest/validation.py
    detail          TEXT NOT NULL,
    raw_payload     BLOB NOT NULL,  -- exact bytes received
    payload_sha256  TEXT NOT NULL,
    UNIQUE (topic, payload_sha256)  -- identical redelivery is not counted twice
);
CREATE INDEX IF NOT EXISTS ix_rejected_reason ON rejected_records (reason);
CREATE INDEX IF NOT EXISTS ix_rejected_received ON rejected_records (received_ts);

-- ---------------------------------------------------------------- derived

-- Resolved worker-in-zone intervals [from_ts, to_ts). Written by the
-- localisation pass. Every row is an inference: label = 'estimated_personal'.
CREATE TABLE IF NOT EXISTS zone_assignments (
    id              INTEGER PRIMARY KEY,
    worker_id       TEXT NOT NULL,
    tag_id          TEXT NOT NULL,
    zone_id         TEXT NOT NULL,  -- a configured zone, or 'UNKNOWN'
    from_ts         TEXT NOT NULL,
    to_ts           TEXT NOT NULL,
    from_epoch_ms   INTEGER NOT NULL,
    to_epoch_ms     INTEGER NOT NULL,
    confidence      REAL NOT NULL CHECK (confidence BETWEEN 0.0 AND 1.0),
    method          TEXT NOT NULL,
    label           TEXT NOT NULL CHECK (label = 'estimated_personal'),
    outside_session INTEGER NOT NULL DEFAULT 0 CHECK (outside_session IN (0, 1)),
    params          TEXT NOT NULL,  -- JSON of the algorithm parameters used
    CHECK (from_ts < to_ts)
);
CREATE INDEX IF NOT EXISTS ix_zone_assignments_worker_from ON zone_assignments (worker_id, from_epoch_ms);

-- Shift sessions resolved from gate events [session_start, session_end).
CREATE TABLE IF NOT EXISTS worker_sessions (
    id                 INTEGER PRIMARY KEY,
    worker_id          TEXT NOT NULL,
    session_start      TEXT,         -- inferred shift start for an orphan check-out (flagged)
    session_end        TEXT NOT NULL,  -- inferred shift end for an unmatched check-in (flagged)
    session_start_ms   INTEGER,
    session_end_ms     INTEGER NOT NULL,
    duration_s         INTEGER,
    shift              TEXT NOT NULL,
    flags              TEXT NOT NULL DEFAULT '[]',
    check_in_event_id  TEXT,
    check_out_event_id TEXT
);
CREATE INDEX IF NOT EXISTS ix_worker_sessions_worker_start ON worker_sessions (worker_id, session_start_ms);

-- ---------------------------------------------------------------- exposure (materialised)
-- Written by exposure/engine.py. Every row stores the engine version and the
-- parameters used, so a value computed under different assumptions stays
-- identifiable. Recomputing a date replaces that date's rows only. No computed_at
-- column: rows are a pure function of their inputs, so a recompute is identical.
-- `shift_date` is the UTC date of the session/shift start; for this mill's shift
-- times (06:00/14:00/22:00 IST = 00:30/08:30/16:30 UTC) it equals the local date.

-- One row per worker per session (exposure/personal.py result fields).
CREATE TABLE IF NOT EXISTS personal_exposure_shift (
    id                      INTEGER PRIMARY KEY,
    worker_id               TEXT NOT NULL,
    shift_date              TEXT NOT NULL,
    shift                   TEXT NOT NULL,
    session_start           TEXT NOT NULL,
    session_end             TEXT NOT NULL,
    session_start_ms        INTEGER NOT NULL,
    session_end_ms          INTEGER NOT NULL,
    session_duration_s      REAL NOT NULL,
    session_flags           TEXT NOT NULL,     -- JSON: flags of the gate session itself
    l_aeq_db                REAL,              -- NULL = not computable, never 0
    l_ex_8h_db              REAL,
    dose_niosh_pct          REAL,
    dose_india_pct          REAL,
    absolute_max_events     TEXT NOT NULL,     -- JSON list
    excluded_anomalous_readings INTEGER NOT NULL DEFAULT 0,   -- spike/stuck records left out
    excluded_absolute_max_candidates TEXT NOT NULL DEFAULT '[]', -- excluded readings >= 115 dBA (JSON)
    pm25_twa_raw            REAL,              -- indicative (optical PM)
    pm25_twa_corrected      REAL,
    pm10_twa_raw            REAL,
    pm10_twa_corrected      REAL,
    wbgt_max_c              REAL,              -- approximate (see wbgt_t_nw_method)
    wbgt_time_above_limit_s REAL,
    co2_twa_ppm             REAL,
    data_coverage_pct       REAL,
    coverage_by_metric      TEXT NOT NULL,     -- JSON
    accounted_time_s        REAL NOT NULL,
    exposure_duration_s     REAL NOT NULL,
    unaccounted_time_s      REAL NOT NULL,
    unaccounted_pct         REAL NOT NULL,
    zones_visited           TEXT NOT NULL,     -- JSON zone -> seconds
    flags                   TEXT NOT NULL,     -- JSON result flags
    label                   TEXT NOT NULL CHECK (label = 'estimated_personal'),
    measured_personal       TEXT,              -- JSON; wearable path, never merged into the above
    dust_label              TEXT NOT NULL,
    wbgt_t_nw_method        TEXT NOT NULL,
    wbgt_approximate        INTEGER NOT NULL CHECK (wbgt_approximate IN (0, 1)),
    wbgt_metabolic_category TEXT NOT NULL,
    criteria_sources        TEXT NOT NULL,     -- JSON
    engine_version          TEXT NOT NULL,
    params                  TEXT NOT NULL,     -- JSON
    UNIQUE (worker_id, session_start_ms)
);
CREATE INDEX IF NOT EXISTS ix_pes_date ON personal_exposure_shift (shift_date);
CREATE INDEX IF NOT EXISTS ix_pes_worker ON personal_exposure_shift (worker_id, session_start_ms);

-- One row per worker per as-of date per rolling window (7 and 30 days).
CREATE TABLE IF NOT EXISTS personal_exposure_cumulative (
    id                          INTEGER PRIMARY KEY,
    worker_id                   TEXT NOT NULL,
    as_of_date                  TEXT NOT NULL,
    window_days                 INTEGER NOT NULL,
    window_start_date           TEXT NOT NULL,
    window_coverage_pct         REAL NOT NULL,   -- window days inside the data period
    partial                     INTEGER NOT NULL CHECK (partial IN (0, 1)),
    days_worked                 INTEGER NOT NULL,
    days_without_data           INTEGER NOT NULL,
    cumulative_dose_niosh_pct   REAL,
    cumulative_dose_india_pct   REAL,
    mean_daily_l_ex_8h_db       REAL,            -- energy mean, never arithmetic
    max_daily_l_ex_8h_db        REAL,
    pm25_twa_hours_raw          REAL,            -- ug.h/m3, indicative
    pm25_twa_hours_corrected    REAL,
    pm10_twa_hours_raw          REAL,
    pm10_twa_hours_corrected    REAL,
    wbgt_hours_above_limit      REAL,
    days_dose_niosh_over_100    INTEGER NOT NULL,
    days_dose_india_over_100    INTEGER NOT NULL,
    mean_data_coverage_pct      REAL,
    mean_unaccounted_pct        REAL,
    label                       TEXT NOT NULL CHECK (label = 'estimated_personal'),
    engine_version              TEXT NOT NULL,
    params                      TEXT NOT NULL,
    UNIQUE (worker_id, as_of_date, window_days)
);
CREATE INDEX IF NOT EXISTS ix_pec_date ON personal_exposure_cumulative (as_of_date);

-- One row per zone per shift instance (heat map; avoids recomputing from raw).
CREATE TABLE IF NOT EXISTS zone_exposure_summary (
    id                      INTEGER PRIMARY KEY,
    zone_id                 TEXT NOT NULL,
    shift_date              TEXT NOT NULL,
    shift                   TEXT NOT NULL,
    shift_start             TEXT NOT NULL,
    shift_end               TEXT NOT NULL,
    shift_start_ms          INTEGER NOT NULL,
    shift_end_ms            INTEGER NOT NULL,
    l_aeq_db                REAL,
    l_ex_8h_db              REAL,
    dose_niosh_pct          REAL,      -- for someone present the whole shift
    dose_india_pct          REAL,
    absolute_max_events     INTEGER NOT NULL,
    excluded_anomalous_readings INTEGER NOT NULL DEFAULT 0,
    excluded_absolute_max_candidates INTEGER NOT NULL DEFAULT 0,
    pm25_twa_raw            REAL,
    pm25_twa_corrected      REAL,
    pm10_twa_raw            REAL,
    pm10_twa_corrected      REAL,
    wbgt_max_c              REAL,
    wbgt_time_above_limit_s REAL,
    co2_twa_ppm             REAL,
    data_coverage_pct       REAL,
    coverage_by_metric      TEXT NOT NULL,
    flags                   TEXT NOT NULL,
    label                   TEXT NOT NULL CHECK (label = 'measured_environmental'),
    dust_label              TEXT NOT NULL,
    wbgt_t_nw_method        TEXT NOT NULL,
    wbgt_approximate        INTEGER NOT NULL CHECK (wbgt_approximate IN (0, 1)),
    engine_version          TEXT NOT NULL,
    params                  TEXT NOT NULL,
    UNIQUE (zone_id, shift_start_ms)
);
CREATE INDEX IF NOT EXISTS ix_zes_date ON zone_exposure_summary (shift_date);

-- One row per worker per day: deterministic risk index (exposure/risk_index.py).
-- label = 'deterministic_index' (nothing is predicted). The weights are a design
-- choice of this project, not a validated clinical model; they are stored per row.
CREATE TABLE IF NOT EXISTS worker_risk_index (
    id                  INTEGER PRIMARY KEY,
    worker_id           TEXT NOT NULL,
    date                TEXT NOT NULL,
    overall_index       REAL NOT NULL,
    band                TEXT NOT NULL CHECK (band IN ('Low', 'Medium', 'High', 'insufficient data')),
    insufficient_data   INTEGER NOT NULL CHECK (insufficient_data IN (0, 1)),
    components          TEXT NOT NULL,   -- JSON: hearing / heat / respiratory, each 0-100
    contributions       TEXT NOT NULL,   -- JSON: absolute and % share of the overall index
    drivers             TEXT NOT NULL,   -- JSON: top two factors in plain language
    confidence_pct      REAL NOT NULL,
    ppe_status          TEXT NOT NULL,   -- JSON: displayed alongside, never reduces the index
    label               TEXT NOT NULL CHECK (label = 'deterministic_index'),
    weights             TEXT NOT NULL,
    version             TEXT NOT NULL,
    params              TEXT NOT NULL,
    UNIQUE (worker_id, date)
);
CREATE INDEX IF NOT EXISTS ix_wri_date ON worker_risk_index (date);
