# Review Notes

Running log of what is built and what is pending. All data in this phase is
**SIMULATED**.

---

## Day 1 — Project skeleton, config, data simulator (2026-10-28)

### What was built

| Module | What it does |
|---|---|
| Skeleton | Package layout from `CLAUDE.md`, `.gitignore`, `requirements.txt` (plus `tzdata`, needed by `zoneinfo` on Windows), `README.md`. `ingest/`, `db/`, `exposure/`, `ml/`, `api/`, `dashboard/` are empty packages - **not built yet**. |
| `config.yaml` | Single source of truth: MQTT, simulation intervals and seed, 7 zones with hazard profiles, 8 machines (M5 degrading), 20 workers, 3 shifts with breaks, NIOSH and India noise criteria, WBGT moderate-work limit, validation ranges and spike thresholds, simulator dynamics, zone adjacency. |
| `config.py` | Loads `config.yaml` once and checks every required key. Missing keys raise `ConfigError`, never a silent default. Typed accessors (`get_zones()`, `get_workers()`, ...). |
| `simulator/schedule.py` (moved to `shifts.py` on Day 2, so the backend can use it without importing simulator code) | Shift and break calendar in UTC from the local (Asia/Kolkata) times in config, including shift C crossing midnight. Shared by the machine and worker models. |
| `simulator/profiles.py` | Zone hazard model: noise, PM2.5/PM10, temperature, RH, globe temperature, CO2 for one zone at one instant. |
| `simulator/machine_model.py` | Clamp-meter current and vibration per machine. Exposes `noise_energy_factor()` for the zone noise coupling. |
| `simulator/worker_model.py` | Each worker's true zone over time, RFID check-in/out events and PPE entries. |
| `simulator/publisher.py` | The only file that talks to MQTT. Runs the three models and publishes contract-format messages. |

Tests: `tests/test_config.py`, `test_profiles.py`, `test_machine_model.py`,
`test_worker_model.py` and `test_publisher_payloads.py`. The publisher test runs
`--dry-run` and validates every payload against `docs/data_contract.md`.

### Commands

```bash
# Broker (Windows service is installed and started by the Mosquitto installer)
winget install -e --id EclipseFoundation.Mosquitto

# Without a broker: print payloads
python -m simulator.publisher --dry-run --duration-h 0.1

# Full 8-hour shift in ~5 minutes (measured: 288 s wall)
python -m simulator.publisher --speed 100 --duration-h 8

# Watch it (second terminal)
mosquitto_sub -h localhost -t "textile/#" -v

# Outage demo options (default: Z3 offline 90 sim-min after start for 5 min)
python -m simulator.publisher --speed 100 --outage-zone Z5 --outage-start-min 30 --outage-duration-min 10
python -m simulator.publisher --speed 100 --no-outage

pytest
```

### End-to-end verification (real broker)

`python -m simulator.publisher --speed 100 --duration-h 8` (start 06:00 IST, seed 42)
captured with `mosquitto_sub -t "textile/#" -v`:

| Topic type | Messages received |
|---|---|
| `textile/zone/{zone_id}/env` | 20,160 (20,130 live + 30 replayed after outage) |
| `textile/zone/{zone_id}/ble` | 6,720 (6,710 live + 10 replayed) |
| `textile/machine/{machine_id}/status` | 23,040 |
| `textile/gate/rfid` | 12 (5 shift-C check-outs, 7 shift-B check-ins) |
| `textile/ppe/entry` | 8 (shift-A supervisor round) |
| `textile/status/{device_id}` | 36 (17 births, outage backlog + clear, 17 shutdowns) |

Wall-clock time: 288 s for 8 simulated hours.

### Sample payloads (captured from the broker)

```text
textile/zone/Z1/env {"schema_version":"1.0","device_id":"ZN-Z1","ts":"2026-10-28T00:30:00Z","zone_id":"Z1","noise_dba":92.2,"noise_peak_dba":96.1,"pm25_ugm3":351.7,"pm10_ugm3":603.2,"temp_c":32.4,"rh_pct":69.2,"globe_temp_c":35.1,"co2_ppm":750,"sensor_flags":{"noise":"ok","pm":"ok","temp_rh":"ok","globe":"ok","co2":"ok"}}

textile/zone/Z1/ble {"schema_version":"1.0","device_id":"ZN-Z1","ts":"2026-10-28T00:30:00Z","zone_id":"Z1","scan_window_s":30,"tags":[{"tag_id":"BLE-0001","rssi_dbm":-63,"seen_count":18},{"tag_id":"BLE-0002","rssi_dbm":-90,"seen_count":6}]}

textile/machine/M1/status {"schema_version":"1.0","device_id":"MN-M1","ts":"2026-10-28T00:30:00Z","machine_id":"M1","zone_id":"Z1","current_a":33.47,"running":true,"vib_rms_g":0.441,"vib_peak_g":1.104,"sensor_flags":{"current":"ok","vibration":"ok"}}

textile/gate/rfid {"schema_version":"1.0","device_id":"GW-01","ts":"2026-10-28T00:33:01Z","card_id":"RFID-10019","event":"check_out","gate_id":"GATE-A"}

textile/ppe/entry {"schema_version":"1.0","device_id":"SUP-APP","ts":"2026-10-28T00:44:41Z","worker_id":"W102","hearing_ppe":"worn","respiratory_ppe":"worn","recorded_by":"SUP-01"}

textile/status/ZN-Z3 {"schema_version":"1.0","device_id":"ZN-Z3","ts":"2026-10-28T02:05:00Z","online":true,"firmware":"sim-0.1.0","uptime_s":5700,"buffered_records":40}
```

In the BLE sample, `BLE-0002` (Worker 102, home zone Z2) is heard weakly (-90 dBm)
by the Z1 node. That is the leakage the backend must resolve.

### Realism features and why each matters

1. **Machine coupling.** Zone noise is the energy sum of a background level
   (~60 dBA) and each running machine's share. When machines stop, noise falls to
   background. Observed in Z5: ~99 dBA Leq while running, **~60 dBA during the
   10:00–10:30 IST lunch break**, with dips at both tea breaks. PM decays toward a
   low floor when machines stop. *Why:* a reviewer who sees 95 dBA during a
   shutdown will not trust anything else.
2. **Autocorrelation.** Every quantity uses an Ornstein-Uhlenbeck / AR(1) update
   with time constants in config, scaled so the stationary sd matches the zone
   profile. Measured lag-1 correlations in tests: noise > 0.4, PM > 0.9,
   temperature and CO2 > 0.95. *Why:* independent draws look fake on a chart and
   would make forecasting trivially impossible, or trivially easy, rather than
   realistic.
3. **Degradation.** M5 vibration grows 2.5x over 30 simulated days and feeds the
   Z5 noise level through `noise_energy_factor()`. Mean running `vib_rms_g`:
   M5 0.497 → 0.677 → 0.920 → 1.255 g at days 0/10/20/30; healthy M6 stays at
   0.497–0.502 g. *Why:* it gives the forecasting and maintenance features a real
   trend to find.
4. **BLE leakage.** The worker model reports true zones. The publisher turns them
   into sightings: strong RSSI in the true zone (−65 ± 7 dBm, 97 % detected) and
   weak sightings in adjacent zones (−83 ± 6 dBm, 50 % detected). In tests, more
   than 20 % of tags are heard by more than one node per scan. *Why:* real BLE
   localisation must cope with this, and clean data would hide the problem.
5. **Sensor faults.** Random `stale`/`error` episodes (2–10 min) publish `null`
   values with the matching flag (two Z1 episodes in the 8 h run). The optical PM
   sensor's gain drifts upward 1.5 %/day, raises `needs_clean` at 1.25x, and resets
   8 h later after cleaning. *Why:* it exercises the validation and sensor-health
   paths, and shows why low-cost PM readings are indicative only.
6. **Outage replay.** A configurable zone node goes silent, buffers its records,
   reconnects with `buffered_records: N` in a retained status, replays with
   original `ts`, and then clears the backlog status. Observed: ZN-Z3 offline
   02:00–02:05Z, 40 records replayed. *Why:* it proves the offline-first design;
   the backend must accept out-of-order data and deduplicate on
   `(device_id, ts, topic)`.
7. **Other details:** diurnal temperature/RH curves, CO2 that falls during breaks,
   a globe temperature offset scaled by machine load, inrush current at start-up,
   per-machine stagger, overtime, and a deliberate high-exposure worker (W105: long
   overtime in Z5, skips tea breaks, low PPE compliance).

### Decisions and open points for the reviewer

- **LWT payload has no `ts`.** Contract §1 defines the LWT as
  `{"device_id", "online": false}`, while §3 says every payload carries `ts`. A
  will message is registered at connect time and cannot know when the device
  died, so the LWT is `{schema_version, device_id, online:false}`. Every message
  the simulator itself sends (including the clean-shutdown status) has `ts`.
  `ingest/validation.py` must accept the LWT form. *Resolved in contract v1.1a:
  `ts` is optional and expected absent on `online: false` status.*
- **Client IDs equal device IDs**, per the contract (one MQTT client per device,
  17 in total). `mqtt.client_id_prefix` is kept for backend clients.
- **Default `--start` is 06:00 IST** as the brief specifies. Shift-A workers
  check in 3–15 min before 06:00, so a run from the default start has no shift-A
  check-ins (they happened before the window). Use `--start 2026-10-28T05:30:00`
  to include them. Consider changing the default.
- **Event timestamps are serialised per device:** gate and supervisor-app events
  are at least 1 s apart, so the dedup key stays unique. *Superseded by v1.1b:
  events carry a UUID v4 `event_id` and deduplicate on it; the spacing was
  removed, and same-second gate pairs now occur in every run.*
- `validation.fields.seen_count.max: 1000` and the spike thresholds are
  plausibility values chosen here; the contract gives none. *Resolved in v1.1c:
  documented, with reasons, in contract section 10.*
- **India noise criterion — resolved (v1.1e).** The 5 dB exchange rate is
  confirmed: Schedule XXVIII under Rule 95 of the Tamil Nadu Factories Rules
  (8 h at 90 dBA, 4 h at 95, 2 h at 100, 1 h at 105, 30 min at 110, 15 min at 115).
  Levels below 90 dBA contribute nothing to the India dose (`threshold_db: 90`,
  compared with 80 dBA for NIOSH), and 115 dBA is an absolute maximum, a hard
  alarm rather than a dose contribution. Rules under the OSH Code 2020 may
  supersede this schedule and should be re-checked before final submission.
- Machine `sensor_flags` are always `ok`; machine-node fault simulation is not
  implemented. *Resolved in v1.1d: `stale`/`error` episodes on the current and
  vibration channels (`simulator/faults.py`).*

### Pending for Day 2

- `db/schema.sql` and `db/repo.py` (all SQL in one place; SQLite)
- `ingest/subscriber.py` and `ingest/validation.py` (contract §10: reject/flag
  rules, `rejected_records` table, dedup on `(device_id, ts, topic)`, LWT form)
- Then the exposure engine (`exposure/`), ML, API, dashboard, ESP32 firmware stub,
  and `scripts/run_all.sh`.

---

## Day 2 — Contract v1.1, database, ingest path, BLE zone resolution (2026-10-28)

### What was built

| Module | What it does |
|---|---|
| `docs/data_contract.md` v1.1 | Five agreed amendments (changelog at the end): `ts` optional on offline status (LWT); UUID v4 `event_id` on RFID/PPE with dedup on it; validation constants documented in §10; machine-node sensor faults; India noise criterion confirmed. `schema_version` is `"1.1"` everywhere. |
| `config.yaml` | India: 90 dBA / 8 h / 5 dB exchange rate, `threshold_db: 90`, `absolute_max_db: 115`, source Schedule XXVIII, Rule 95, TN Factories Rules (OSH Code 2020 to be re-checked). NIOSH `threshold_db: 80`. New `localisation`, `sessions`, `ingest` blocks. |
| `db/schema.sql`, `db/repo.py`, `db/init_db.py` | SQLite in WAL mode with foreign keys and `busy_timeout`. Typed columns, NULL kept for null readings. Telemetry is unique on `(device_id, ts, topic)`, events on `event_id`, rejections on `(topic, sha256)`. `INSERT OR IGNORE` everywhere; reads ordered by `ts`. **All SQL is in `db/repo.py`** (enforced by a test). |
| `ingest/validation.py` | Pure checks for contract §10. Closed enums: 13 rejection reasons and 4 flags. |
| `ingest/subscriber.py` | MQTT → validate → batch → commit → ack. Persistent session with manual acks after commit, a savepoint per message, ring buffers for stuck/spike, reconnect with backoff, clean shutdown. |
| `ingest/localisation.py` | BLE sightings → merged zone intervals (`ble_ewma_hysteresis`), run as a separate pass. |
| `ingest/sessions.py` | Gate events → sessions with explicit flags; assignments clipped to sessions. |
| `shifts.py` | The shift calendar, moved from `simulator/schedule.py` so the backend can use it without importing simulator code. |
| `simulator/` | Machine-node faults (`faults.py`), silent stuck/glitch failures on zone sensors, same-second gate pairs, outages on any device (`--extra-outage`), `--ground-truth-out`. |
| `scripts/` | `score_localisation.py`, `sweep_localisation.py`, `inject_bad_records.py`, `replay_capture.py`. |

### Design decisions taken with the reviewer in mind

- **Time reference for future/late checks** (agreed): at `--speed 100` simulated
  `ts` is hours away from the wall clock, so `--time-reference stream` uses the
  newest accepted telemetry/event `ts` as "now". Real hardware uses `wall` (the
  default). Retained messages never advance the watermark. *Limitation:* in
  stream mode the first record is judged against its own `ts`.
- **Every rejection reason and flag fires in a real run, without polluting the
  data** (agreed). Flags come from realistic failures in the simulator: sensor
  faults, *stuck* sensors and single-reading *glitches* (both with flag `ok`), and
  a 2.5 h outage on machine node M8. Rejections come from
  `scripts/inject_bad_records.py`, whose records are all rejected and all carry an
  `injected_by` marker. Nothing injected reaches the main tables.
- **Zero-loss delivery:** a QoS 1 message is acknowledged only after the batch
  holding it is committed, so a crash means redelivery, which dedup absorbs.
  *Found in a real-broker smoke test:* Mosquitto stops delivering after
  `max_inflight_messages` (20) unacked messages, which capped ingest at ~19 msg/s.
  The backlog then exceeded `max_queued_messages` and the broker **dropped**
  messages. Fix: also commit once 20 QoS 1 messages are waiting (`max_unacked`).
- **Sessions:** both inferred cases are symmetric and flagged. An unmatched
  check-in ends at its shift end; an orphan check-out starts at its shift start.
  (The first rule here, "an orphan covers nothing", erased the real end of the
  night shift when data starts mid-shift: 381 on-site windows became UNKNOWN.
  Under-reporting exposure is the unsafe direction, so it was changed.)

### Commands

```bash
python -m db.init_db --reset
python -m ingest.subscriber --time-reference stream --idle-exit-s 20
python -m simulator.publisher --speed 100 --duration-h 8 --start 2026-10-28T05:30:00 \
    --ground-truth-out data/truth.jsonl --extra-outage MN-M8:60:150
python scripts/inject_bad_records.py          # mid-run
python -m ingest.localisation
python -m ingest.sessions
python scripts/score_localisation.py --truth data/truth.jsonl
python scripts/sweep_localisation.py --truth data/truth.jsonl
mosquitto_sub -t "textile/#" -v -q 1 > data/capture.txt     # during the run
python scripts/replay_capture.py --capture data/capture.txt # idempotency
```

Before a measured run, clear retained `textile/status/*` messages and purge the
ingest client's persistent session, so earlier runs cannot leak in.

### Measured results (run 3: 8 h shift at --speed 100, seed 42, start 05:30 IST)

**Run history, reported honestly.** Run 1 exposed an injector bug: its
"11 minutes ahead" record arrived inside the tolerance at 100× speed, was
*accepted*, and displaced one real reading as a duplicate. Fix: date it a day
ahead. Run 2 was clean, but its idempotency replay exposed a watermark bug:
duplicates did not advance the watermark, so a replay judged the far-future
record against itself and stored it. Fix: duplicates advance the watermark. Run 3
is the final run below. Runs 1 and 2 are kept in `data/run1_invalid/` and
`data/run2_superseded/` (not committed; `data/` is git-ignored).

**Data loss: 0 — published = stored + rejected, exactly.**

| | Messages |
|---|---|
| Published by simulator (env 20,130 + BLE 6,710 + machine 22,140 + RFID 13 + PPE 8 + status 38 + replayed after outages 940) | 49,979 |
| Injected invalid records | 13 |
| **Received by ingest** | **49,992** |
| Stored (new rows) | 49,979 |
| Rejected | 13 |
| Duplicates / processing errors | 0 / 0 |

Rows per table: `zone_env` 20,160, `ble_scans` 6,720 (13,927 sightings),
`machine_status` 23,040, `gate_events` 13, `ppe_entries` 8, `device_status` 38,
`rejected_records` 13.

**Ingest throughput:** 49,992 messages over 287.9 s of wall time =
**173.6 msg/s sustained**. 30-s intervals ranged 162–201 msg/s (the peak is the
M8 replay burst). The ingest queue was empty at every stats line, i.e. ingest
kept up with the publisher throughout.

**Validation results.**

| Rejection reason | Count | Flag | Count |
|---|---|---|---|
| malformed_json | 1 | sensor_not_ok | 224 |
| unknown_topic | 1 | stuck_value | 171 |
| not_an_object | 1 | late_upload | 180 |
| schema_version | 1 | spike | 20 |
| missing_field | 1 | **flagged records** | **595** (env 313, machine 282) |
| wrong_type | 1 | | |
| bad_ts | 1 | | |
| topic_mismatch | 1 | | |
| out_of_range | 1 | | |
| invalid_enum | 1 | | |
| null_with_ok_flag | 1 | | |
| invalid_event_id | 1 | | |
| future_ts | 1 | | |

All 13 rejected rows carry the injector marker. The deliberately faulty sensors
appear: machine-node `stale`/`error` (contract v1.1d) and zone faults, both under
`sensor_not_ok`. `late_upload` is exactly M8's first 180 buffered readings
(01:00:00–01:29:50Z, more than 2 h old when replayed at 03:30Z). The **same-second
gate pair** appears as W101 and W102 at `2026-10-28T00:26:48Z`, with two distinct
`event_id`s, both stored.

**Idempotency:** replaying all 49,992 captured messages into the same database gave
49,979 duplicates and 13 rejected-duplicates. **Every table's row count is
unchanged.**

**Outage handling (Z3, 01:30–01:35Z):** all 30 env readings are present with their
**original `ts`** (01:30:00–01:34:50Z). Across the stream they are exactly the 30
rows (and the only ones, 01:25–01:40Z) that arrived after rows with a later `ts`.
For example, Z1's live 01:35:00Z reading was received before Z3's replayed
01:32:20Z reading. The node's retained status announced
`buffered_records: 40` (30 env + 10 BLE) on reconnect at 01:35:00Z, and 0 at
01:35:10Z.

**Localisation accuracy** (`scripts/score_localisation.py`, ground truth from the
simulator; it never touches MQTT or the database):

| Metric | Raw assignments | Clipped to sessions |
|---|---|---|
| **On-site worker-minutes in the correct zone** | **96.4%** (7,690 windows) | **96.4%** |
| Accuracy incl. off-site (off-site correct only if UNKNOWN) | 98.5% | 98.6% |
| Correctly UNKNOWN (really off site) | 11,495 | 11,510 |
| Incorrectly UNKNOWN (on site) | 14 | 14 |
| Off-site windows given a zone | 15 | **0** |
| Zone-change latency (n=173) | mean 45.8 s, median 30 s, p90 90 s | same |
| Exit latency to UNKNOWN (n=5) | 90 s | 0 s (session end) |
| True changes never followed | 3 | 3 |

Per-zone recall: Z1 95.7%, Z2 95.1%, Z3 94.4%, Z4 96.2%, Z5 98.3%, **Z6 92.5%
(lowest)**, Z7 97.4%. Parameters: `ewma_alpha 0.5`, `rssi_margin_db 6`,
`debounce_windows 2`, `absence_windows 4`, `min_rssi_dbm -95`, `window_s 30`.

Confusion matrix (raw; rows = true, columns = predicted):

```text
           UNKNOWN       Z1       Z2       Z3       Z4       Z5       Z6       Z7
 OFF_SITE    11495        0        0        0        0        0        0       15
       Z1        1      781       29        0        0        0        0        5
       Z2        1       25      821       12        0        0        0        4
       Z3        0        1       15      804       28        0        0        4
       Z4        0        0        0       35     1514       14        0       10
       Z5        2        0        0        0       12     1636        8        7
       Z6        2        0        0        0        0       11      245        7
       Z7        8        4        4        3        6        5       13     1613
```

Every error is between **adjacent** zones (the layout is a chain, Z1–Z7). About
95% of the wrong windows are **transition latency**: 173 changes × 45.8 s ≈ 264 of
the 277 wrong windows. Steady-state misclassification is rare. Z6 has the lowest
recall because its workers make short trips: the 30–60 s debounce latency is a
larger share of a short visit. The 15 off-site windows given a zone are workers at
the gate, assigned to Z7 before check-in or after check-out; **session clipping
removes all of them**. That is the purpose of the clip, and the number to quote.

**Parameter sensitivity** (`scripts/sweep_localisation.py`, one parameter at a time):

| Parameter | Value | Accuracy (on site) | Wrong UNKNOWN | Off-site → zone | Latency median / p90 | Missed | Zone changes* |
|---|---|---|---|---|---|---|---|
| baseline | | 96.4% | 14 | 15 | 30 / 90 s | 3 | 182 |
| ewma_alpha | 0.3 | 94.4% | 14 | 15 | 90 / 120 s | 5 | 181 |
| ewma_alpha | 0.8 | 97.2% | 16 | 15 | 30 / 30 s | 2 | 192 |
| ewma_alpha | 1.0 | 97.1% | 16 | 15 | 30 / 30 s | 2 | 191 |
| rssi_margin_db | 0 | 96.9% | 14 | 15 | 30 / 60 s | 2 | 194 |
| rssi_margin_db | 3 | 96.7% | 14 | 15 | 30 / 60 s | 3 | 185 |
| rssi_margin_db | 9 | 95.9% | 14 | 15 | 60 / 90 s | 4 | 181 |
| debounce_windows | 1 | 98.6% | 0 | 15 | 0 / 30 s | 0 | **223** |
| debounce_windows | 3 | 94.0% | 27 | 15 | 60 / 120 s | 4 | 181 |
| absence_windows | 2 | 96.4% | 16 | 5 | 30 / 90 s | 3 | 184 |
| absence_windows | 6 | 96.4% | 14 | 25 | 30 / 90 s | 3 | 182 |

\*Interval boundaries across all workers. The true number of changes is about 181.

**Interpretation, and why the defaults were not changed.** Lighter filtering scores
higher here, e.g. `debounce_windows=1` reaches 98.6%. But it produces **223 zone
changes against ~181 real ones**: about 40 spurious flaps, the oscillation the
algorithm exists to suppress, and a flapping assignment splits a worker's exposure
across zones. The accuracy metric rewards responsiveness because the simulated
RSSI is clean (true zone ≈ −65 dBm, neighbours ≈ −83 dBm, ~18 dB apart). A real
mill full of metal machinery will be worse. Choosing the best row would also
select parameters on the data being scored. So the physically-motivated defaults
stay, and **re-tuning is deferred to hardware calibration walks**. Stated as a
limitation: on simulated data the localisation is good, its main cost is 30–90 s
latency at zone changes, and its real-world accuracy is untested.

### Integrity checks (all enforced by tests)

- Ground truth: written only by `--ground-truth-out` to a local file. No backend
  module mentions it, the publisher emits the same messages with or without the
  flag, and no table has a column for it.
- No module in `ingest/`, `db/`, `exposure/`, `ml/`, `api/`, `dashboard/` or
  `scripts/` imports `simulator`.
- No SQL outside `db/repo.py`.
- Every zone assignment is labelled `estimated_personal`, with
  `method = ble_ewma_hysteresis` and its parameters stored per row.

### Known limitations and open points

- Every shift-A session in this run is flagged `unmatched_check_in`, because the
  run ends at 13:30 IST, before 14:00. Every shift-C session is
  `orphan_check_out`, because it started before 05:30. Both are artefacts of the
  run window, closed by the inferred shift boundaries and visibly flagged.
- Stream-mode bootstrap: the first telemetry record seeds the watermark (see
  above). Wall mode, which real hardware uses, has no such caveat.
- `ts` is stored exactly as published and sorted as text. That is correct for the
  contract's whole-second format, but a device sending fractional seconds would
  sort incorrectly against whole-second strings. Validation accepts fractions;
  worth tightening when the firmware is written.
- Broker limits matter for zero loss: keep `ingest.max_unacked` ≤ Mosquitto's
  `max_inflight_messages`, and size `max_queued_messages` for the longest expected
  ingest downtime.
- `simulator/profiles.py` still has its Day 1 built-in fault logic, which
  duplicates `simulator/faults.SensorFaults`. It was left alone because it passes
  its tests and the brief says not to refactor passing modules unasked.

### Pending for Day 3 onward

- Exposure engine (`exposure/noise.py`, `dust.py`, `heat.py`, `cumulative.py`)
  using session-clipped `zone_assignments` (exclude `outside_session`), with NIOSH
  and India doses side by side. The India dose uses the 90 dBA threshold, and
  115 dBA is a hard alarm.
- ML forecasting (Task A), risk scoring (Task B), SHAP, API, dashboard with the
  `SIMULATED DATA` badge, ESP32 firmware stub, `scripts/run_all.sh`.
- Localisation tuning on real hardware (calibration walks).

---

## Day 3 — The exposure engine (2026-10-28)

### What was built

| Module | What it does |
|---|---|
| `db/` (Task 1a) | `ts_epoch_ms` on every table with a `ts`, parsed once at ingest; every ordered or ranged read uses it. As text, `…:10.500Z` sorted **before** `…:10Z`; reproduced first, then fixed. `Repo.migrate_to_v2` backfilled all three Day 2 databases with row counts unchanged. |
| `simulator/publisher.py` (1b) | `--align-shifts` (default on): window snapped to shift boundaries plus margins, and only crews whose whole session fits are simulated. Shifts overlap at every handover, so boundaries alone cannot give whole sessions. `--no-align-shifts` keeps the raw window, so the truncated-session flags stay tested. |
| `exposure/noise.py` | L_Aeq, L_EX,8h, allowed time, dose with per-criterion threshold, 115 dBA absolute maximum. |
| `exposure/dust.py`, `heat.py` | Linear TWA; κ-Köhler humidity correction (raw always kept); WBGT with Stull wet-bulb, flagged `approximate`. |
| `exposure/personal.py` | Per-worker, per-session exposure with unaccounted-time and coverage accounting. |
| `exposure/cumulative.py` | Rolling 7- and 30-day aggregates; partial windows reported as partial. |
| `exposure/engine.py` | CLI: three materialised tables, idempotent per date, summary table. |
| `docs/exposure_methods.md` | Formulas, verification table, Schedule XXVIII agreement, assumptions, results. |

### Headline verification

- **All 12 hand-verified noise cases pass**, including 4 h @ 91 dBA = **200%** NIOSH,
  the **87.4 dBA** energy-average case, and four Schedule XXVIII rows (all six rows
  checked). The wrong "100%" planning figure appears nowhere.
- **Internal consistency (single zone): residual ≤ 1×10⁻⁹ dB** in the test, through
  the real ingest, localisation and session pipelines, and **0.0 dB on the 7-day
  run data** (W105, Z5, 6.42 h).
- **7-day run:** 1,071,656 published = 1,071,656 stored (zero loss) at 156 msg/s.
  140 whole, unflagged sessions. Localisation 96.3% over the week.
- **Exposure (140 worker-days):** worker-days over 100% are **NIOSH 90.0%** and
  **India 40.0%**. Median dose is 445% NIOSH and 89% India. Mean coverage is 99.06%
  and mean unaccounted time 0.15%. Weaving (Z5) drives the highest doses (zone
  L_Aeq 99.1 dBA) and carding (Z2) the highest dust. Full tables are in
  `exposure_methods.md` §7.

### Decisions and findings for the reviewer

- **Conservative readings:** the threshold is inclusive, the absolute-max alarm
  fires at ≥ 115 dBA, and the 115 dBA check uses `noise_dba`, not the peak.
- **NIOSH 3 dB vs the exact 3.0103 dB:** dose computed directly and dose inferred
  from L_EX,8h differ by 0.24% (200.0% vs 199.5%). The direct Σ C/T value is
  reported.
- **UNKNOWN time is reported, never filled.** Figures cover accounted time and are
  lower bounds where it is missing. NULL readings lower coverage, not the dose.
- **κ = 0.05** (small on purpose: a larger κ lowers reported dust). The RH cap at
  95% is a project assumption.
- **Finding: glitch readings had entered the doses.** Engine v1.0 integrated 20
  spike-flagged microphone glitches (120–138 dBA). The maximum session dose was
  16,688% and there were 20 "≥115 dBA alarms". v1.1 excludes `spike` and
  `stuck_value` records (lost coverage) and lists excluded ≥115 dBA readings
  separately instead of dropping them. Results: maximum 3,515%, 0 genuine alarms,
  20 listed candidates. Validation had flagged every glitch correctly; the engine
  had ignored the flag.
- **Process mistake:** commit 6d3cc04 went in with a failing guard test, because
  `pytest | tail` masked the exit code. It was fixed in the next commit, and exit
  codes are now checked directly.

### Pending for Day 4

Forecasting (features, baselines, RF, XGBoost, exceedance lead time), SHAP on the
forecaster, the deterministic risk index, and the results pack.

---

## Day 4 — Forecasting, explainability, risk index (2026-10-28)

Scope decisions kept: **no FastAPI** (the dashboard reads SQLite through
`db/repo.py`); **no ML risk classifier** (it would only re-learn the dose
thresholds). SHAP is applied to the forecaster.

### What was built

| Module | What it does |
|---|---|
| `ml/features.py` | One row per zone per 1-minute bin (71,897 rows, 60 features, mean coverage 99.19%) from stored tables only. Noise resampled and rolling-averaged in the energy domain. Anomaly, NULL and non-ok sensor values are excluded, never zeroed. |
| `ml/forecast.py` | Persistence, seasonal, random forest, XGBoost; whole-day chronological split with a boundary purge; skill, MAE, RMSE, R²; steady vs schedule-transition subsets; exceedance lead time. |
| `ml/explain.py` | SHAP `TreeExplainer` on the saved noise t+30 model: beeswarm, bar, and two waterfalls. |
| `exposure/risk_index.py` | Deterministic, transparent index (`deterministic_index`) and the `worker_risk_index` table. |
| `figstyle.py` | One colour per hazard (Okabe-Ito), no colour-only encoding, slide-legible fonts, SIMULATED DATA badge. |
| `scripts/build_results_pack.py` | Idempotent pack: 9 figures, `results_forecast.md`, `results_exposure.md`, `results_summary.md`. Every number is read from its source, and the consistency residual is re-measured. |

### Leakage protection (tests, not comments)

- Every feature declares its source offset relative to *t* (all ≤ 0).
- **Truncation test:** features rebuilt from data cut off at *t* are identical to the
  full-data features at *t*, checked on real pipeline data.
- Two subtle leaks avoided: `workers_present_count` uses the previous *complete*
  BLE window, and it ignores `outside_session`, which depends on a future check-out
  time.

### Headline results (test day 2026-10-26; full tables in `results_forecast.md`)

| Target | Seasonal skill | RF | XGBoost | Note |
|---|---|---|---|---|
| Noise t+30 | +0.765 | **+0.805** | +0.745 | RF beats the seasonal baseline |
| Noise t+60 | +0.766 | +0.741 | +0.746 | **Negative finding:** no ML model beats the seasonal baseline |
| PM2.5 t+30 / t+60 | +0.254 / +0.298 | +0.344 / +0.401 | **+0.384 / +0.434** | clearest ML result |
| WBGT t+30 / t+60 | **−0.280 / −0.040** | +0.048 / +0.149 | +0.125 / +0.214 | seasonal is worse than persistence; heat is barely predictable |

- **Timetable vs forecasting.** Most of the noise skill is the fixed break and shift
  schedule. On steady rows (no scheduled change within the horizon), ML noise skill
  is +0.56–0.62, against +0.39–0.51 for the seasonal baseline: genuine skill beyond
  the timetable.
- **Exceedance (90 dBA).** 36/36 sustained crossings were warned, median lead
  17.0 min (t+30) and 18.5 min (t+60). False-alarm ratio 12.3% / 5.3%, against
  36–42% for the seasonal baseline. Persistence cannot warn ahead by definition.
  The crossings are mostly post-break restarts, so the lead time reflects the
  schedule too.
- **SHAP (noise t+30).** Recent noise history 37%, schedule 29%, machine state 21%
  (maximum vibration in the top six), zone 12%, ambient and occupancy under 1%.
- **Risk index.** 140 worker-days: High 119, Medium 7, Low 14; median 83.5.
  Contributions sum to 100%; monotonicity is tested over 7 inputs × 200 random
  baselines; low confidence suppresses the band, not the number; PPE is never
  applied.

### Findings and limitations to state at review

- **The index's hearing cap hides severity above 100% dose.** W102 (carding, 235%)
  ranks above W105 (weaving, 3,378%), because both hearing components cap at 100
  and carding is the dustiest zone. The dose is always shown beside the index; a
  log-scaled hearing component is a documented option, not applied.
- The respiratory component is **cohort-relative**: it ranks, it does not judge
  safety.
- The simulated schedule is rigid. Transition skill should be expected to fall on
  real data; the steady-state figures are the better guide. With a single test day,
  there are no confidence intervals.
- **Process:** e398a5c was committed with a failing test (chained with `;`) and was
  fixed in the next commit. Every commit since is gated on a separately captured
  pytest exit code.

### Pending for Day 5

The Streamlit dashboard: it reads the materialised tables and parquet files; shows
the SIMULATED DATA badge, the risk-index weights caveat and the WBGT approximation
flag in the UI; and uses the results-pack figures. Optional LSTM/GRU only after
everything else.

---

## Day 5 — Dashboard and demo readiness (2026-10-28)

### What was built

| Commit | Piece |
|---|---|
| c1b2030 | **risk-1.1**: the hearing component comes from the daily L_EX,8h (75 dBA → 0, 105 dBA → 100) instead of a dose capped at 100%. Regression test: a higher L_EX never scores below a lower one. |
| 516cc9a | Dashboard scaffolding (`dashboard/app.py`, `common.py`) and **Worker Drilldown**: L_EX,8h, NIOSH and India doses side by side, coverage, unaccounted time, the lower-bound sentence, zone timeline with energy-averaged noise (flagged readings excluded), index contributions with the design-choice caveat, 7/30-day cumulative table. |
| bef2cc6 | **Live Zones**: seven zone tiles, noise–machine coupling chart with machines-stopped shading and break bands. |
| 804a9cd | **System & Results**: alerts, device states, rejections and flags, ingest reconciliation, forecast skill table and results-pack figures. |
| 53a47f6 | **Offline demo**: `demo/textile_ohs_demo.db` (44 MB, committed), `docs/demo_script.md`, `docs/numbers_sheet.md` (generated by the results pack). |
| (final) | `firmware/zone_node/zone_node.ino` stub, `firmware/README.md`, `docs/review_qa.md`. |

The dashboard opens SQLite **read-only** (`mode=ro` URI) through `db/repo.py`, with no
SQL in `dashboard/` (enforced by a test). Every page shows the `SIMULATED DATA` badge
and has a plain empty state for an empty or missing database (AppTest render tests).
`dashboard/views/` is not named `pages/`, because Streamlit auto-discovers that folder
and bypasses `st.navigation`.

### risk-1.1 results (140 worker-days)

High 110, Medium 16, Low 14; median 72/100. The Day 4 finding is fixed for the hearing
component: W105 (weaving) now scores 84.2 against W102 (carding) 46.1. **W102 still
leads overall (80.3 against 73.1)** because carding is the dustiest zone. That is what
the weights say; they were not retuned to change the ordering.

### Demo database

`scripts/make_demo_db.py` → `Repo.export_demo_copy`. It keeps every derived table in
full (sessions, assignments, shift and cumulative exposure, risk index, device status)
plus raw env and machine telemetry for 2026-10-26 only, and drops raw BLE sightings. The
source database is never modified (tested). `streamlit run dashboard/app.py` uses it
by default, with no broker and no simulator.

### Honest gaps

- **The network-off rehearsal has not been done.** The development session could not
  turn the network off. The dashboard-only path (no broker, no simulator, no other
  Python process) was rehearsed. The presenter must run the rehearsal in
  `docs/demo_script.md` ("Rehearsal with the network disabled") before the review.
- The demo database has raw telemetry for one day only. Noise overlays and Live Zones
  on other dates have no raw data to plot (Worker Drilldown says so in an info message).
- The firmware is an uncompiled stub with fixed sensor values. The pin plan comes from
  the component selection and is unverified. PubSubClient's QoS 0 conflicts with the
  contract's QoS 1 (open item in `firmware/README.md`). There is no machine-node sketch.
- The "feasibility table" for Q9 of `docs/review_qa.md` is in the project report, not
  in this repository.
- The ML risk classifier is still deferred until questionnaire labels exist.

### Pending after the first review

Hardware bring-up and calibration (handheld sound-level meter, reference dust sampler,
BLE calibration walks), questionnaire labels, TimescaleDB migration.

---

## Day 6 — ML depth: classification, the ablation, anomaly detection (2026-10-29)

### What was built

| Commit | Piece |
|---|---|
| 1ffcd89 | **Exceedance classification** (`ml/classify_exceedance.py`): will zone noise at t+30/t+60 be ≥ 90 dBA? Label = the measured future level (not circular). Variants `above` and `crossing`; persistence classifier baseline; logistic regression, random forest, XGBoost grid-searched on the validation day; threshold = highest recall with FPR ≤ 10% (config). Prevalence before every metric; PR-AUC leads; steady vs schedule-transition breakdown; per-zone appendix. |
| d3c1010 | **Model 1/2/3 ablation** (`ml/ablation.py`, Objective 6), reframed onto forecasting and classification. Strictly nested feature sets as explicit column lists; one model class and seed; each set tuned separately; moving-block bootstrap on paired differences; per-zone breakdown. |
| e968ff5 | **Machine condition monitoring** (`ml/anomaly.py`): EWMA control chart and isolation forest per machine, each calibrated on its own first day; lead-lag M5 → Z5 noise. |
| ba7aeef | `ml/report_ml.py` → `docs/results_ml.md` (consolidated, with the model comparison table) and `docs/model_cards.md`, both generated from the saved metrics. Risk index **worst single component** beside the composite on the drilldown. Dashboard **Models** page (reads saved artefacts only). Results pack and numbers sheet extended. |

### Headline findings (all simulated, one test day)

- **Classification.** `above`: XGBoost PR-AUC 0.97 vs persistence 0.67 (prevalence 40%).
  `crossing`: PR-AUC 0.83–0.84 vs 0.16 for persistence. On break-free windows it is
  0.69 / 0.77, and in Z2, which hovers near 90 dBA, it barely beats chance.
- **Ablation.** Model 3 (machine + zone history) improves noise forecast skill by
  +0.017 / +0.028 (CIs exclude 0). Model 2 (worker data) adds at most +0.007. There is
  no practical gain for dust, heat or classification.
- **Anomaly.** M5 is flagged on day 2.08 (EWMA) and 2.12 (isolation forest). No healthy
  machine alarms. Healthy per-bin FPR is 0.02% for EWMA against 1.03% for the forest,
  so the control chart is the one to deploy.
- **Risk index.** The worst component's band is above the composite band on 24 of 140
  worker-days. That is the case the fix was for.

### Honest gaps

- **Task 4 (GRU), done as a negative result.** One GRU layer over the last 60 min of the
  Model 3 features, noise only, with no architecture search. Skill +0.52 / +0.68 against
  XGBoost +0.76 / +0.79 on identical rows; it trained in 4.7 min, inside the 45-min cap.
  This needed **`torch` (CPU build), added to `requirements.txt`** (Task 4 only; nothing
  else imports it, and its tests skip without it).
- The ground-truth isolation guard (`tests/test_scoring.py`) caught the word "truth"
  in `ml/anomaly.py`. The scoring label is `configured_degrading`, read from
  `config.yaml`, never from simulator output files.
- The Task 2 commit message says Model 2 adds "nothing distinguishable". Its +0.007 on
  noise at t+30 *is* distinguishable, though negligible. The documents were corrected;
  the commit message cannot be.
- The machine-to-noise results (SHAP share, Model 3 gain, M5–Z5 analysis) reflect a
  coupling the simulator builds in. The caveat is stated in each place.
- One test day: no between-day variance estimate. The risk classifier is still
  deferred pending questionnaire labels.

---

## Day 7 — Redundancy analysis, zone-group reporting, final state (2026-10-29)

### What was built

| Commit | Piece |
|---|---|
| 740560b | **Ablation redundancy analysis** (`ml/ablation.py --nl-only`). Model 1-NL (context: temperature/RH/CO2 and their means, time of day, shift, break, zone) against Model 3-NL (+ machines running, current, vibration, workers present), with no column derived from any hazard series (enforced by a test). A supplementary pair (1-NS / 3-NS) also drops the timetable. Merged into the saved Day 6 metrics; Day 6 numbers were not recomputed. Figure `ablation_redundancy.png`. |
| 8e0c2e4 | **Classification by zone group** (borderline Z2/Z3/Z6, always-loud Z1/Z4/Z5, excluded Z7; in `config.yaml`), leading with Z2, in `results_classification.md`, on the Models page and in the numbers sheet. **EWMA-over-isolation-forest** conclusion promoted to a paragraph in `results_ml.md` and on the Models page. |
| (final) | `docs/review_qa.md` entries 12 ("ablation gains are tiny") and 13 ("why worse in carding?"); `DAY_7.md` committed. |

### Findings that changed the headline

- **Objective 6 has a negative answer on this dataset.** Without target lags, machine
  and worker data still add only +0.014 noise skill (CIs exclude 0; about 0.1 dB of
  MAE). Without the timetable as well, they add nothing distinguishable. The Day 7
  brief expected a *large* no-lag gain; it did not appear. The redundancy is with
  the **timetable**, not mainly with target autocorrelation. A model with no noise
  input reaches skill +0.76 / +0.77, which supports a **sensor-failure resilience**
  claim carried mainly by the timetable.
- **Exceedance warnings are weak exactly where they matter.** In Z2, `crossing` PR-AUC
  is 0.22 against a base rate of 0.18, with false alarms on 44–53% of quiet minutes at
  the pooled threshold. In the borderline group it is 0.18 against 0.08. The pooled
  0.83 comes from trivial restarts in the loud zones (66% of positives) and the
  office's certain negatives (24% of rows).

### Deviations from the brief, stated

- `zone_code` is kept in both no-lag sets (the brief's list omits it). Without it, the
  machine counts would act as a zone label and inflate the gain.
- Z1 is placed in the always-loud group (the brief's table omits it): its `crossing`
  label is 82–85% positive and the model predicts "yes" there almost always.
- The supplementary no-timetable pair goes beyond the brief. It was added to explain
  the small no-lag gain with a measurement rather than an assertion.

### Offline rehearsal (Task 3a): NOT YET PERFORMED

This needs a person at the machine with the network physically off; the development
session cannot do it. A **static check** was done instead: no `http(s)://` URLs in
`dashboard/`, no map tiles or font CDNs, plots rendered by Streamlit's bundled
Plotly, and `gatherUsageStats = false` in `.streamlit/config.toml`. This lowers the
risk; it does not replace the rehearsal. **Record the outcome here after running the
checklist in `DAY_7.md` Task 3a** (Wi-Fi off → `streamlit run dashboard/app.py` → every
page and sidebar option → W105 drilldown, Live Zones chart, every Models-page figure).

### Open items

- `DAY_7.md` asks that review Q7 and Q9 point at `docs/feasibility_table.md`. That
  file does not exist: the feasibility table is in the project report. Q7 in this
  repository is about localisation settings, not feasibility. Q9 still points to the
  report. To be resolved by adding the table from the report.
- Zone-specific classification thresholds (chosen on validation) were not tried, to
  avoid any further tuning after the test day had been seen.
- A nowcast test (current zone noise from machine state alone) is the direct test of
  the machine-state fallback, and was not run.

## Day 8A — Why the 7-day stochastic run was killed (2026-10-29)

### Task 1: memory measured before changing anything

The 7-day generation (Day 8 Task 2) was stopped at about 2.4 simulated days, when
Claude Code reported LOW SYSTEM MEMORY and terminated its background processes. To
find out what grew, the same command line (stochastics on, seed 42, same start,
speed 90) was run for 25 wall minutes (about 37 simulated hours) into a scratch
database. The resident set of every process was sampled every 15 s.
Data: `docs/day8a/memory_probe_rss.csv`; figure: `docs/figures/day8a_memory_probe.png`.

| Process | Resident set, first → last | Committed private memory | Trend |
|---|---|---|---|
| Publisher | 46.3 → 46.4 MB | 511 MB, constant | +0.004 MB/min (flat) |
| Subscriber | 33.9 → 36.3 MB (warm-up in the first minute, then flat) | 19 → 21 MB | +0.015 MB/min (flat) |
| Mosquitto | 8.1 → 8.1 MB | 2.8 MB | 0 |
| Whole machine, free RAM | swung between 2.0 and 4.1 GB of 16 GB | — | driven by other applications |

**Finding: nothing in the project grows.** The kill came from elsewhere on the machine.
During the probe, the browser (about 2.7 GB), VS Code (about 2.0 GB) and system services
pushed free RAM down to 2.0 GB. Claude Code's background-process reaper then stops the
processes it started, and ours were those processes. Checked against the brief's
suspects:

1. **The WAL is not the cause.** It stayed at about 4.5 MB. SQLite's default
   auto-checkpoint (1000 pages) keeps it bounded, because nothing holds a long read
   transaction.
2. **The ground truth already streams to disk**, one `write` per line through an open
   file. No list is built.
3. **The publisher has no growing list.** Message counts are a `Counter`; the outage
   buffer is replayed and emptied. One slow growth is negligible: the stop log, about
   50 records per machine per day.
4. **The subscriber's stuck-value history** is a `deque(maxlen=stuck_value_count)` per
   (device, field). Its incoming queue is unbounded in type but stays at 0 in
   practice: commits come every 20 unacknowledged messages, so the broker never has
   more than 20 messages in flight.

**The publisher's 511 MB is not a leak.** It is OpenBLAS reserving per-thread buffers
when numpy is imported (16 logical CPUs). With `OPENBLAS_NUM_THREADS=1` the same
publisher commits 27 MB. The window length does not change it either: building a 30 h
and a 168 h simulation adds the same 0.4 MB. Only about 46 MB of it is resident.

**A second, unrelated defect found by the probe: stale broker session.** The ingest
client uses a persistent session (`clean_session: false`, so the broker queues data
while ingest is down). When the run was killed, 4 telemetry messages (ts
2026-10-22T11:01:50Z) stayed queued for it. The next subscriber received them first,
and in `stream` time-reference mode they moved the watermark 2.4 days ahead. **Every
later record was then flagged `late_upload` (222,680 of 222,701 in the probe)**, and
the 4 foreign rows were stored. The deterministic dataset has zero `late_upload`
flags, so restarting without a purge would have made the two datasets differ in
something other than the simulator. The restart therefore begins by discarding that
session (Task 3).

### Task 2: every collection bounded (hardening; the probe found no leak)

None of these fixed a leak, because there was none. They make sure a longer run cannot
grow, and `tests/test_memory_bounds.py` turns that into a regression test. The new caps
are in `config.yaml` (`simulation.memory_bounds`, `ingest`).

- **2a WAL checkpoint.** `Repo(..., wal_checkpoint_every=N)` runs
  `PRAGMA wal_checkpoint(TRUNCATE)` every N commits (N = 200,
  `ingest.wal_checkpoint_every_commits`) and logs the WAL size before and after. A
  checkpoint blocked by a reader is logged as a warning and retried at the next one.
- **2b Ground truth.** It was already written line by line through one open file. It
  is now also flushed every 1000 lines, so a killed run keeps its ground truth up to
  that point.
- **2c Bounds:**
  - The subscriber's queue is `queue.Queue(maxsize=10000)`. A full queue blocks the
    network thread, so the backlog stays at the broker and nothing is lost. A
    `deque(maxlen=...)` was deliberately *not* used for ingest data: it drops the
    oldest message silently, which would break the zero-loss guarantee and the
    fail-loudly rule.
  - Stuck/spike history is kept for at most 64 device IDs, created only for accepted
    records. Each field's history was already a `deque(maxlen=stuck_value_count)`. An
    untracked device is counted and logged.
  - The publisher's outage buffer is a `deque(maxlen=50000)`, a finite microSD
    buffer. An overwrite is counted and logged as an error.
  - Each machine's stop log is a `deque(maxlen=5000)`. `verify_stochastics` raises if
    a log saturates.
  - Shift-start plans older than one day are pruned.
- **2d Counters.** Summary lines were already built from `Counter`s. No list was being
  kept only for a summary.
- **No draw changed.** Before and after the changes, the stochastic 48 h dry run, its
  ground-truth file and the deterministic 6 h dry run are byte-identical (same
  SHA-256).
- **Regression test:** an in-process 24 h stochastic dry run (150k messages). Memory at
  the end must be within 20 MB of the value 10% of the way through. Measured: resident
  59.0 → 59.0 MB, private 523.6 → 523.6 MB. A first version of the test kept every Z3
  payload in its own sink and showed +9 MB, which was the test's own growth, not the
  publisher's.

### Task 3: reduce peak pressure

- **3a Data on D:.** Already true: the repository is `D:/textile-ohs`, and every path in
  `config.yaml` is relative to it. The database, WAL and ground truth are in
  `data/week_stochastic/` on D:. The `paths` block now says so.
- **3b Speed 60.** About 2.8 h for 168 simulated hours.
  `scripts/run_stochastic_generation.sh` is the exact restart command (reset, purge the
  stale session, subscriber, publisher). Speed only paces the wall clock, so the data
  are unchanged.
- **3c Other applications.** This needs the user. During the probe the browser held
  about 2.7 GB and VS Code about 2.0 GB of 16 GB. Closing them is the most effective
  single step, because the Task 1 probe showed the pressure comes from outside the
  project.
- The restart conditions are appended to `docs/preregistration_day8.md` as Addendum A
  (dated; the original text is untouched).
