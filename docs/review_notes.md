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
