# Data Contract (AUTHORITATIVE)

Version 1.1 — 2026-10-28 (see the Changelog at the end).

This document defines the MQTT topics and JSON payloads exchanged between field
devices and the backend. It is the interface between hardware and software.

**Today** `simulator/publisher.py` produces these messages.
**Later** ESP32 firmware produces the identical messages, and no backend code changes.

Any change to this file requires bumping `schema_version` and updating
`ingest/validation.py` and `tests/` in the same commit.

---

## 1. Transport

- Protocol: MQTT 3.1.1 over TCP
- Broker: Eclipse Mosquitto, `localhost:1883` in development
- QoS: 1 (at least once) for all telemetry
- Retain: `false` for telemetry, `true` for status/birth messages
- Payload: UTF-8 JSON, one object per message, no arrays at the top level
- Client ID: the device ID (for example `ZN-Z1`, `MN-M3`, `GW-01`)

### Last Will and Testament
Every device registers a retained LWT on `textile/status/{device_id}` with
`{"schema_version": "1.1", "device_id": "...", "online": false}`, which has **no
`ts`** (see section 3). On connect it publishes a retained
`{"schema_version": "1.1", "device_id": "...", "ts": "...", "online": true, ...}`.

---

## 2. Topic map

| Topic | Publisher | Rate | Purpose |
|---|---|---|---|
| `textile/zone/{zone_id}/env` | Zone node | 1 msg / 10 s | Environmental telemetry |
| `textile/zone/{zone_id}/ble` | Zone node | 1 msg / 30 s | BLE tags seen in this zone |
| `textile/machine/{machine_id}/status` | Machine node | 1 msg / 10 s | Machine current + vibration |
| `textile/gate/rfid` | Gate reader | On event | Worker shift check-in / check-out |
| `textile/ppe/entry` | Supervisor app | On event | PPE usage record |
| `textile/status/{device_id}` | Any device | On connect / LWT | Device online state |

`zone_id` is one of `Z1`..`Z7`. `machine_id` is `M1`..`M8`. Device IDs are
`ZN-{zone_id}` for zone nodes and `MN-{machine_id}` for machine nodes.

---

## 3. Common fields

Every payload carries these three fields, with one exception for `ts` (below).

| Field | Type | Description |
|---|---|---|
| `schema_version` | string | `"1.1"` |
| `device_id` | string | Publishing device |
| `ts` | string | UTC ISO-8601 with `Z`, e.g. `"2026-10-28T04:48:10Z"` |

**Exception (v1.1):** on a `textile/status/{device_id}` message with
`"online": false`, `ts` is **optional and expected to be absent**. The LWT is
registered at connect time and cannot know when the device will disconnect. The
backend stores the message with `ts` as NULL plus its own `received_ts`, and must
**not** invent a `ts`. A clean shutdown may still include `ts`. Every other message,
including `online: true` status, requires `ts`.

The backend stores `ts` exactly as published and adds its own `received_ts`.
Clock drift is expected; the RTC on each node is the authority for `ts`.

---

## 4. Zone environmental telemetry

**Topic:** `textile/zone/{zone_id}/env`

```json
{
  "schema_version": "1.1",
  "device_id": "ZN-Z4",
  "ts": "2026-10-28T04:48:10Z",
  "zone_id": "Z4",
  "noise_dba": 96.4,
  "noise_peak_dba": 103.1,
  "pm25_ugm3": 118.2,
  "pm10_ugm3": 240.7,
  "temp_c": 33.8,
  "rh_pct": 72.5,
  "globe_temp_c": 36.2,
  "co2_ppm": 980,
  "sensor_flags": {
    "noise": "ok",
    "pm": "ok",
    "temp_rh": "ok",
    "globe": "ok",
    "co2": "ok"
  }
}
```

| Field | Type | Unit | Valid range | Notes |
|---|---|---|---|---|
| `zone_id` | string | — | Z1..Z7 | Must match the topic |
| `noise_dba` | float | dBA | 30.0–140.0 | L_Aeq over the 10 s interval, A-weighted |
| `noise_peak_dba` | float | dBA | 30.0–150.0 | Max level in the interval |
| `pm25_ugm3` | float | ug/m3 | 0.0–2000.0 | Raw sensor value, uncorrected |
| `pm10_ugm3` | float | ug/m3 | 0.0–3000.0 | Raw sensor value, uncorrected |
| `temp_c` | float | degC | -10.0–70.0 | |
| `rh_pct` | float | % | 0.0–100.0 | |
| `globe_temp_c` | float | degC | -10.0–90.0 | Black globe, for WBGT |
| `co2_ppm` | int | ppm | 300–10000 | Ventilation indicator |
| `sensor_flags` | object | — | — | Per-sensor health, see below |

**Important:** `pm25_ugm3` and `pm10_ugm3` are *raw, uncorrected* sensor outputs.
Humidity correction happens in `exposure/dust.py`, never on the device. Both the
raw and corrected values must be retrievable.

**`sensor_flags` values:** `"ok"`, `"stale"` (no fresh reading), `"error"` (read
failed), `"uncalibrated"`, `"needs_clean"` (PM sensor lint warning).
A sensor whose flag is not `"ok"` must still publish its last field, using `null`
for the value when no reading exists.

---

## 5. BLE worker presence

**Topic:** `textile/zone/{zone_id}/ble`

A zone node reports every worker tag it can currently see, with signal strength.
Zone assignment (strongest-RSSI-wins, with smoothing) is a **backend** decision,
not a device decision, because a tag is often seen by two nodes at once.

```json
{
  "schema_version": "1.1",
  "device_id": "ZN-Z4",
  "ts": "2026-10-28T04:48:00Z",
  "zone_id": "Z4",
  "scan_window_s": 30,
  "tags": [
    { "tag_id": "BLE-0007", "rssi_dbm": -58, "seen_count": 11 },
    { "tag_id": "BLE-0012", "rssi_dbm": -79, "seen_count": 4  }
  ]
}
```

| Field | Type | Valid range | Notes |
|---|---|---|---|
| `scan_window_s` | int | 5–120 | Length of the scan window |
| `tags[].tag_id` | string | — | Maps to a worker via `config.yaml` |
| `tags[].rssi_dbm` | int | -100 to -20 | More negative is farther away |
| `tags[].seen_count` | int | >= 1 | Advertisements received in the window |

An empty `tags` array is valid and meaningful (nobody in the zone).

---

## 6. Machine status

**Topic:** `textile/machine/{machine_id}/status`

```json
{
  "schema_version": "1.1",
  "device_id": "MN-M3",
  "ts": "2026-10-28T04:48:10Z",
  "machine_id": "M3",
  "zone_id": "Z4",
  "current_a": 14.8,
  "running": true,
  "vib_rms_g": 0.42,
  "vib_peak_g": 1.07,
  "sensor_flags": { "current": "ok", "vibration": "ok" }
}
```

| Field | Type | Unit | Valid range | Notes |
|---|---|---|---|---|
| `machine_id` | string | — | M1..M8 | Must match the topic |
| `zone_id` | string | — | Z1..Z7 | Which zone the machine sits in |
| `current_a` | float | A | 0.0–200.0 | From the clip-on clamp |
| `running` | bool | — | — | Device-side threshold on `current_a` |
| `vib_rms_g` | float | g | 0.0–16.0 | RMS over the interval |
| `vib_peak_g` | float | g | 0.0–16.0 | Peak in the interval |
| `sensor_flags` | object | — | — | Health of the `current` and `vibration` channels |

`sensor_flags` uses the same values as section 4. When the `current` channel is
not `"ok"`, `current_a` is `null` and `running` carries the node's last decision.
When the `vibration` channel is not `"ok"`, `vib_rms_g` and `vib_peak_g` are `null`.

---

## 7. RFID gate events

**Topic:** `textile/gate/rfid`

```json
{
  "schema_version": "1.1",
  "device_id": "GW-01",
  "ts": "2026-10-28T02:28:00Z",
  "event_id": "3f2b8c1e-9a4d-4e7b-8c21-5d6f0a9e1b73",
  "card_id": "RFID-10045",
  "event": "check_in",
  "gate_id": "GATE-A"
}
```

`event_id` (required, v1.1) is a UUID v4 string the reader generates for each tap.
It is this topic's deduplication key (section 11). Two workers can tap the same gate
within the same second, so `(device_id, ts, topic)` is not unique for events.

`event` is `"check_in"` or `"check_out"`. `card_id` maps to a worker via
`config.yaml`. A check-in with no matching check-out is closed at the end of the
shift by the backend, and flagged.

---

## 8. PPE entry (supervisor-recorded)

**Topic:** `textile/ppe/entry`

```json
{
  "schema_version": "1.1",
  "device_id": "SUP-APP",
  "ts": "2026-10-28T03:00:00Z",
  "event_id": "a7c41e02-6b3f-4d9a-9e58-0f2d7c1b4a96",
  "worker_id": "W101",
  "hearing_ppe": "worn",
  "respiratory_ppe": "partial",
  "recorded_by": "SUP-02"
}
```

`event_id` (required, v1.1) is a UUID v4 string, one per entry, and is this topic's
deduplication key.

`hearing_ppe` and `respiratory_ppe` are each `"worn"`, `"partial"`, `"not_worn"`
or `"unknown"`.

**This is self-reported or supervisor-observed data, not sensor data.** It is used
as an ML feature and displayed to supervisors. It must never be used to
automatically reduce a calculated exposure dose.

---

## 9. Device status

**Topic:** `textile/status/{device_id}` (retained)

```json
{
  "schema_version": "1.1",
  "device_id": "ZN-Z4",
  "ts": "2026-10-28T02:00:00Z",
  "online": true,
  "firmware": "sim-0.1.0",
  "uptime_s": 7200,
  "buffered_records": 0
}
```

`buffered_records` is the count of records held on microSD awaiting upload after a
network outage. The simulator sets `firmware` to a value starting with `sim-` so
that simulated runs are identifiable in the database later.

---

## 10. Validation rules for `ingest/validation.py`

A record is **rejected** (logged, counted, not stored in the main table) if:
- `schema_version` is not `"1.1"`
- a required field is missing, or `ts` does not parse as ISO-8601 UTC (except the
  offline-status form in section 3, where `ts` may be absent)
- `zone_id` or `machine_id` does not match the topic
- a numeric field falls outside the valid range in the tables above (boundaries
  are inclusive)
- a value is `null` while its sensor flag is `"ok"` (a contradiction)
- `event_id` is missing or is not a UUID v4 string on an event topic
- `ts` is more than 10 minutes in the future

A record is **stored but flagged** if:
- any `sensor_flags` value is not `"ok"`
- a value is identical to the previous 30 readings from the same sensor
  (stuck-value detection)
- a value jumps more than the per-field spike threshold in `config.yaml`
  between consecutive readings
- `ts` is more than 2 hours older than `received_ts` (late buffered upload)

### Validation constants

The values live in `config.yaml` (`validation:`); this table documents them. The
valid ranges are in sections 4–6.

| Field | Spike threshold (consecutive readings) | How it was chosen |
|---|---|---|
| `noise_dba` | 25 dB | More than one machine starting or stopping within 10 s in the noisiest zone; a larger jump suggests a microphone fault |
| `noise_peak_dba` | 35 dB | Peaks are naturally spikier than L_Aeq (impacts, shuttle knocks), so wider than `noise_dba` |
| `pm25_ugm3` | 500 ug/m3 | Roughly the largest 10 s rise a carding-room blowdown produces |
| `pm10_ugm3` | 800 ug/m3 | PM10 is about 2x PM2.5 in cotton dust, with some extra margin |
| `temp_c` | 5 degC | Air temperature in a mill hall cannot move this fast; a larger jump is a sensor fault |
| `rh_pct` | 15 % | As for temperature, with room for a humidifier switching on |
| `globe_temp_c` | 8 degC | The globe responds to radiant load faster than air, so wider than `temp_c` |
| `co2_ppm` | 1500 ppm | Well above any change occupancy can cause in 10 s |
| `current_a` | 150 A | Catches only implausible jumps; normal start-up inrush must not be flagged |
| `vib_rms_g` | 5 g | Far above any wear-driven change; catches a loose or dropped sensor |
| `vib_peak_g` | 10 g | Peaks are wider than RMS |
| `seen_count` | cap of 1000 (range maximum) | Tags advertise about once per second (~30 per 30 s window); 1000 leaves room for faster tags and longer windows while rejecting counter overflow |
| `uptime_s`, `buffered_records` | range 0–4294967295, no spike check | Non-negative counters held as unsigned 32-bit on the node |

Other constants: stuck-value detection compares against the previous **30**
readings (`stuck_value_count`); the future-`ts` tolerance is **600 s**
(`future_ts_tolerance_s`); a late upload is one more than **7200 s** older than
`received_ts` (`late_upload_threshold_s`).

Every rejection and flag is counted and exposed on the dashboard's sensor-health
panel. Rejected records go to a `rejected_records` table with the reason, so the
review demo can show that validation actually fires.

---

## 11. Offline behaviour

When a device cannot reach the broker it stores records on microSD and replays them
on reconnect with their **original** `ts`. The backend therefore must never assume
messages arrive in chronological order, and must deduplicate:

- **event topics** (`textile/gate/rfid`, `textile/ppe/entry`) on `event_id`;
- **telemetry topics** (env, BLE, machine status) on `(device_id, ts, topic)`.

---

## Changelog

| Version | Change | Reason |
|---|---|---|
| 1.1a | `ts` is optional, and expected to be absent, on `online: false` status | An LWT is registered at connect and cannot know the disconnect time; the backend stamps `received_ts` and must not invent a `ts` |
| 1.1b | Required `event_id` (UUID v4) on RFID and PPE events; events deduplicate on `event_id` | Two workers can tap the same gate in the same second, so `(device_id, ts, topic)` is not a unique key for events |
| 1.1c | Spike thresholds, the `seen_count` cap and the other validation constants documented in section 10 | They were implementer's choices; they are now part of the contract (values still held in `config.yaml`) |
| 1.1d | Machine nodes report `stale`/`error` on the `current` and `vibration` channels, with `null` values | Machine sensors fail too, and the sensor-health view must cover machine nodes |
| 1.1e | India noise criterion confirmed: 90 dBA for 8 h, 5 dB exchange rate, 90 dBA threshold, 115 dBA absolute maximum (Schedule XXVIII, Rule 95, Tamil Nadu Factories Rules) | Resolves the open verification note from v1.0; values are in `config.yaml` |
| 1.1 | `schema_version` is `"1.1"`; a `null` value with an `"ok"` flag is an explicit rejection | Version bump for the amendments above |