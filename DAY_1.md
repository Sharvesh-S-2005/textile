# DAY 1 — Project Skeleton, Config, and Data Simulator

**Read `CLAUDE.md` and `docs/data_contract.md` before starting. The data contract is
authoritative; if anything here conflicts with it, the contract wins.**

**Goal of today:** a runnable simulator that publishes realistic textile-mill data
over MQTT in exactly the contract format, plus the project skeleton and config that
every later module depends on. At the end of today we should be able to run the
simulator and watch messages arrive with `mosquitto_sub`.

**Do NOT build today:** database, ingest, exposure engine, ML, API, dashboard.
Those are Day 2 onward. Build only what is listed below.

---

## Task 1 — Project skeleton

Create the directory structure from `CLAUDE.md`, with an `__init__.py` in every
Python package directory. Create empty-but-valid placeholder packages for
`ingest/`, `db/`, `exposure/`, `ml/`, `api/`, `dashboard/`, `tests/`, `firmware/`,
`scripts/`.

Create `.gitignore` covering `__pycache__/`, `*.pyc`, `.venv/`, `venv/`, `*.db`,
`*.sqlite3`, `.pytest_cache/`, `.DS_Store`, `data/`, `models/`, `.env`.

Create `requirements.txt` pinned to major versions:

```
paho-mqtt>=2.1
PyYAML>=6.0
numpy>=1.26
pandas>=2.2
scipy>=1.13
scikit-learn>=1.5
xgboost>=2.1
shap>=0.46
fastapi>=0.115
uvicorn>=0.30
streamlit>=1.38
plotly>=5.24
pytest>=8.3
```

Create `README.md` with: one-paragraph project description, how to install the
broker, how to create the venv, how to install requirements, how to run the
simulator. Keep it short and accurate.

Initialise a git repository and make the first commit.

---

## Task 2 — `config.yaml` (single source of truth)

Everything configurable lives here. No module may hard-code any of these values.

Include these sections:

**`mqtt`** — host (`localhost`), port (`1883`), keepalive, qos (`1`),
`topic_prefix` (`textile`), client_id_prefix.

**`simulation`** — `schema_version: "1.0"`, `env_interval_s: 10`,
`ble_interval_s: 30`, `machine_interval_s: 10`, `speed_factor` (default `1.0`),
`random_seed: 42`, `firmware_tag: "sim-0.1.0"`.

**`zones`** — a list of 7 zones. For each: `zone_id`, `name`, `device_id`, and a
`profile` block holding the baseline mean and standard deviation for `noise_dba`,
`pm25_ugm3`, `pm10_ugm3`, `temp_c`, `rh_pct`, `co2_ppm`, plus a
`globe_offset_c` (how many degrees the black globe reads above air temperature).
Use the zone table in `CLAUDE.md` for the ranges — convert each range to a sensible
mean and standard deviation so that roughly 95% of samples land inside the range.

**`machines`** — 8 machines, each with `machine_id`, `zone_id`, `device_id`,
`rated_current_a`, `baseline_vib_rms_g`, and a `degrading: true|false` flag. Make
exactly one machine (`M5`, in Z5) degrading.

**`workers`** — 20 workers. Each with `worker_id` (`W101`..`W120`),
`name` (a generic placeholder such as `Worker 101`, never a real person's name),
`tag_id` (`BLE-0001`..), `card_id` (`RFID-10001`..), `home_zone`, `shift`
(`A`, `B` or `C`), and `ppe_compliance` (a float 0.0–1.0 describing how reliably
this worker wears PPE).

**`shifts`** — A: 06:00–14:00, B: 14:00–22:00, C: 22:00–06:00, all Asia/Kolkata,
each with a 30-minute lunch break and two 10-minute tea breaks at defined times.

**`exposure_limits`** — two named criteria:
- `niosh`: `criterion_db: 85`, `exchange_rate_db: 3`, `duration_h: 8`
- `india`: `criterion_db: 90`, `exchange_rate_db: 5`, `duration_h: 8`,
  with a comment that the exchange rate must be verified against the current
  Tamil Nadu Factories Rules before final submission.
Also include `wbgt_limit_c` for moderate work, and `rh_correction_threshold_pct: 75`
for the PM humidity correction.

**`validation`** — per-field `min`, `max` and `spike_threshold`, matching section 10
of the data contract. Also `stuck_value_count: 30` and
`future_ts_tolerance_s: 600`.

**`paths`** — `db_path: data/textile_ohs.db`, `model_dir: models/`, `log_dir: logs/`.

Write `config.py` at the project root: a small loader that reads `config.yaml` once,
validates that required keys exist, and exposes typed accessor functions
(`get_zones()`, `get_workers()`, `get_machines()`, `get_mqtt_config()`,
`get_exposure_limits()`, ...). Raise a clear error on a missing key; never return a
silent default.

Write `tests/test_config.py`: config loads, there are exactly 7 zones / 8 machines /
20 workers, all `zone_id` values referenced by machines and workers exist, all
`tag_id` and `card_id` values are unique.

**Run the tests. Commit.**

---

## Task 3 — `simulator/profiles.py` (zone hazard model)

This module generates environmental values for one zone at one instant. Realism
here is what makes the demo convincing, so it must do more than draw random numbers.

Implement a `ZoneProfile` class built from a zone's config entry, with a method
that returns a reading for a given simulated timestamp and a given set of
currently-running machines in that zone.

The reading must combine:

1. **Baseline** — the zone's configured mean.
2. **Machine load** — noise and dust rise when machines in the zone are running.
   With all machines off, noise should fall to roughly background (55–65 dBA) and
   PM should decay toward a low floor. This is the single most important realism
   feature: a reviewer who sees noise stay at 95 dBA during a shutdown will not
   trust anything else.
3. **Time-of-day effects** — temperature and humidity follow a smooth daily curve
   (hotter in the afternoon, more humid at night); CO2 rises through an occupied
   shift and falls between shifts.
4. **Autocorrelation** — consecutive readings must be correlated, not independent
   draws. Use a simple AR(1) / Ornstein-Uhlenbeck style update:
   `value_next = value + theta * (target - value) + sigma * noise`.
   Independent random draws per reading look obviously fake on a chart.
5. **Transients** — occasional short noise peaks (a loom start, a blowdown) and
   dust spikes (cleaning, blowdown cycles), configurable in frequency.
6. **Globe temperature** — air temperature plus the configured offset, scaled by
   how many machines are running.
7. **Sensor health events** — with a low configured probability, a sensor enters a
   `stale`, `error` or `needs_clean` state for a few minutes and emits `null` for
   its value with the appropriate flag. The PM sensor should drift upward slowly
   over simulated days and then trigger `needs_clean`.

Clamp every output to the valid range in the data contract, but clamp *after*
generation so that occasional extreme values are preserved rather than smoothed away.

`tests/test_profiles.py`: readings stay within contract ranges; consecutive readings
are correlated (lag-1 correlation above a threshold); noise with all machines off is
materially lower than with all machines running; the same `random_seed` reproduces
the same sequence exactly.

**Run the tests. Commit.**

---

## Task 4 — `simulator/machine_model.py`

A `MachineModel` class per machine that produces `current_a`, `running`,
`vib_rms_g` and `vib_peak_g` for a given simulated timestamp.

Behaviour:
- Machines run during their zone's shift hours and stop during breaks and between
  shifts, with slight random stagger so they do not all switch at the same second.
- `current_a` is near zero when stopped, and near `rated_current_a` with
  realistic small fluctuation when running; include a brief inrush spike at start-up.
- `running` is derived from `current_a` crossing a threshold, exactly as the real
  device would decide it.
- `vib_rms_g` is near zero when stopped and near `baseline_vib_rms_g` with noise
  when running.
- For the machine flagged `degrading`, `vib_rms_g` grows gradually over simulated
  days (roughly a 2.5x increase over 30 simulated days), and that machine's zone
  noise contribution rises with it. Expose a method the zone profile can use so
  this coupling is real and demonstrable, not asserted.

`tests/test_machine_model.py`: stopped machines report `running: false` and near-zero
current; a running machine reports current within a sensible band of rated; the
degrading machine's vibration at day 30 is materially higher than at day 0; a
non-degrading machine's is not.

**Run the tests. Commit.**

---

## Task 5 — `simulator/worker_model.py`

A `WorkerModel` that tracks where each worker is over a simulated shift.

Behaviour:
- Each worker checks in at the gate a few minutes before shift start and checks out
  after shift end (emit RFID events for these).
- During the shift the worker spends most time in their `home_zone` but moves to
  adjacent zones occasionally, and goes to Z7 (Office/Store) during breaks.
- Movement is a simple state machine with configured dwell times, not uniform
  random teleporting between zones.
- For each simulated instant, the model can report which zone each worker is in.
- A small fraction of workers do overtime, staying past shift end. Make at least one
  worker consistently high-exposure (long hours in Z5) so the dashboard has an
  obvious high-risk case to show.

**BLE output:** the worker model does not assign zones in its output. It reports
the true zone, and the publisher converts that into per-zone BLE sightings with
realistic RSSI: strong RSSI in the true zone, and weaker sightings leaking into
one or two neighbouring zones. The backend's job later is to resolve this. Do not
make the BLE data unrealistically clean — the leakage is the point, because it is
what the real BLE localisation has to cope with.

**PPE events:** at shift start, each worker emits a PPE entry whose values are drawn
using their `ppe_compliance` value.

`tests/test_worker_model.py`: every worker is in exactly one zone at any instant;
workers are in Z7 during breaks; check-in precedes check-out; shift durations are
within expected bounds; the seeded run is reproducible.

**Run the tests. Commit.**

---

## Task 6 — `simulator/publisher.py` (the only MQTT-facing file)

A command-line program that ties the three models together and publishes to MQTT in
exactly the contract format.

Requirements:

- `--speed N` — simulated-time multiplier. `--speed 100` replays an 8-hour shift in
  about 5 minutes. Default `1.0`.
- `--start` — simulated start time, ISO-8601, default today at 06:00 Asia/Kolkata.
- `--duration-h` — how many simulated hours to run, default 8.
- `--seed` — override the config seed.
- `--dry-run` — print payloads to stdout instead of publishing, so the output can be
  inspected without a broker running.
- `--broker` / `--port` — override the config values.

Behaviour:

- Publish a retained birth message on `textile/status/{device_id}` for every device
  at start-up, and register an LWT.
- Publish env, BLE, machine, RFID and PPE messages on their contract topics at their
  contract intervals (scaled by `--speed`).
- Every payload must include `schema_version`, `device_id` and `ts`, with `ts` being
  **simulated** time in UTC with a `Z` suffix, not wall-clock time.
- Simulate a network outage: at a configurable simulated time, one zone node stops
  publishing for a few minutes, then replays its buffered records with their
  original `ts` and a non-zero `buffered_records` in its status message. This is
  what proves the offline-first design works, so it must be demonstrable from the
  command line.
- Log a one-line summary every simulated 10 minutes: messages published per topic
  type, current simulated time, elapsed wall-clock time.
- Handle Ctrl-C cleanly: publish `online: false` and disconnect.

**Validate your own output.** Write `tests/test_publisher_payloads.py` which runs
the publisher in `--dry-run` mode for a short simulated period, captures the
payloads, and asserts that every message validates against the data contract:
required fields present, `ts` parses as UTC ISO-8601, `zone_id`/`machine_id` match
the topic, all numeric fields within the contract ranges, `sensor_flags` values from
the allowed set, and `tags` entries well-formed. A payload that fails the contract is
a bug in the simulator, not in the test.

**Run the tests. Commit.**

---

## Task 7 — Verify end to end, then document

1. Install Mosquitto locally and start it.
2. Run `python -m simulator.publisher --speed 100 --duration-h 8`.
3. In another terminal run `mosquitto_sub -h localhost -t 'textile/#' -v` and
   confirm messages flow on every topic.
4. Capture a short sample of real output.

Then write `docs/review_notes.md` with:
- what was built today, module by module
- the exact commands to run the simulator and watch output
- one captured sample payload per topic type
- a short note on which realism features were implemented (machine coupling,
  autocorrelation, BLE leakage, sensor faults, outage replay) and why each matters
- what is pending for Day 2

Run `pytest` across the whole project one final time and show the full output.
Make a final commit for the day.

---

## Definition of done for Day 1

- [ ] `pytest` passes with no failures and no skipped tests
- [ ] `python -m simulator.publisher --dry-run` prints contract-valid payloads
- [ ] With Mosquitto running, `mosquitto_sub -t 'textile/#'` shows live messages on
      all six topic types
- [ ] `--speed 100` completes an 8-hour shift in roughly 5 minutes
- [ ] No module outside `simulator/` imports anything from `simulator/`
- [ ] No hard-coded zone names, limits or thresholds outside `config.yaml`
- [ ] Noise visibly drops when machines stop; the degrading machine's vibration
      visibly rises over simulated days
- [ ] `docs/review_notes.md` written
- [ ] All work committed to git

---

## Working instructions for this session

Work through the tasks **in order**. After each task: run that task's tests, show
the output, commit, then state briefly what is next. Do not jump ahead to the
database, ingest or dashboard.

If a task is ambiguous, choose the simplest option that satisfies
`docs/data_contract.md` and say in one line what you chose and why.

If you cannot complete a task, say so plainly rather than committing a stub that
looks finished.