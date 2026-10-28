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
| `simulator/schedule.py` | Shift and break calendar in UTC from the local (Asia/Kolkata) times in config, including shift C crossing midnight. Shared by the machine and worker models. |
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
  `ingest/validation.py` must accept the LWT form. **Contract wording should be
  clarified.**
- **Client IDs equal device IDs**, per the contract (one MQTT client per device,
  17 in total). `mqtt.client_id_prefix` is kept for backend clients.
- **Default `--start` is 06:00 IST** as the brief specifies. Shift-A workers
  check in 3–15 min before 06:00, so a run from the default start has no shift-A
  check-ins (they happened before the window). Use `--start 2026-10-28T05:30:00`
  to include them. Consider changing the default.
- **Event timestamps are serialised per device:** gate and supervisor-app events
  are at least 1 s apart, so the dedup key stays unique.
- `validation.fields.seen_count.max: 1000` and the spike thresholds are
  plausibility values chosen here; the contract gives none.
- The India noise exchange rate (5 dB) **must be verified** against the current
  Tamil Nadu Factories Rules before final submission.
- Machine `sensor_flags` are always `ok`; machine-node fault simulation is not
  implemented.

### Pending for Day 2

- `db/schema.sql` and `db/repo.py` (all SQL in one place; SQLite)
- `ingest/subscriber.py` and `ingest/validation.py` (contract §10: reject/flag
  rules, `rejected_records` table, dedup on `(device_id, ts, topic)`, LWT form)
- Then the exposure engine (`exposure/`), ML, API, dashboard, ESP32 firmware stub,
  and `scripts/run_all.sh`.
