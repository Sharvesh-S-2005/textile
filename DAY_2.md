# DAY 2 — Contract v1.1, Database, Ingest Path, and BLE Zone Resolution

**Read `CLAUDE.md`, `docs/data_contract.md` and `docs/review_notes.md` before
starting.**

**Goal of today:** everything the simulator publishes lands correctly in SQLite
through a real MQTT ingest path, with validation that actually rejects bad records;
and raw BLE sightings are resolved into defensible per-worker zone assignments whose
accuracy we can **measure against the simulator's ground truth**.

That last part is the most valuable thing built today. It turns "we track workers
with BLE" into "our zone localisation is 94.2% accurate, here is the confusion
matrix" — which is a results table for the paper, not a claim.

**Do NOT build today:** exposure calculations, ML, API, dashboard. Those are Day 4
onward. Resist the temptation to start plotting things.

---

## Task 1 — Data contract v1.1 (amendments)

Five amendments were agreed after Day 1. Apply all of them to
`docs/data_contract.md`, bump `schema_version` to `"1.1"` everywhere (config,
simulator, tests), and add a short **Changelog** section at the bottom of the
contract recording each change and its reason.

**1.1a — Optional `ts` on offline status.**
Section 3 currently says every payload carries `ts`. Amend it: `ts` is **optional
and expected to be absent** on a `textile/status/{device_id}` message with
`"online": false`, because an LWT is registered at connect and cannot know the
disconnect time. The backend stamps `received_ts` for these. All other message
types still require `ts`. Validation must accept the LWT form and must **not**
invent a `ts` for it.

**1.1b — `event_id` on event topics.**
Add a required `event_id` field (UUID v4 string) to `textile/gate/rfid` and
`textile/ppe/entry` payloads. Update section 11: deduplication is on `event_id` for
event topics and on `(device_id, ts, topic)` for telemetry topics.

Then **remove the 1-second minimum spacing** between gate and PPE events in the
simulator. That rule was a workaround for the weak dedup key, and real hardware
will not honour it — two workers genuinely can tap the same gate within one second.
Make the simulator emit at least one such same-second pair during a normal run, so
the dedup path is exercised rather than assumed.

**1.1c — Document the validation constants.**
Move the spike thresholds and the `seen_count` cap (1000) out of "implementer's
choice" and into section 10 of the contract as an explicit table, with one line
saying how each was chosen. Keep the values in `config.yaml`; the contract documents
them, `config.yaml` holds them.

**1.1d — Machine node sensor faults.**
Machine nodes currently always report `sensor_flags` of `ok`. Give
`machine_model.py` the same fault behaviour the zone profiles have: occasional
`stale` and `error` states on the current and vibration channels, emitting `null`
for the value with the matching flag. Without this, the sensor-health panel can
only ever show zone nodes, and the demo looks half-finished.

**1.1e — India noise criterion (resolved).**
The 5 dB exchange rate is **confirmed correct**. Source: Schedule XXVIII under
Rule 95 of the Tamil Nadu Factories Rules (8 h at 90 dBA, 4 h at 95, 2 h at 100,
1 h at 105, 30 min at 110, 15 min at 115 — a halving of permitted time per 5 dB).

Update `config.yaml` under `exposure_limits.india`:
- `criterion_db: 90`, `exchange_rate_db: 5`, `duration_h: 8`
- `threshold_db: 90` — **levels below 90 dBA contribute zero to the India dose.**
  This differs from NIOSH, where accumulation starts at 80 dBA. Add
  `threshold_db: 80` under `niosh` to make the contrast explicit.
- `absolute_max_db: 115` — no exposure above this is permitted at all. This is a
  hard alarm condition, not a dose contribution.
- `source: "Schedule XXVIII, Rule 95, Tamil Nadu Factories Rules"`
- A comment noting that rules under the OSH Code 2020 may supersede this and should
  be re-checked before final submission.

Replace the "needs verification" note in `docs/review_notes.md` with this resolution.
Do not implement the dose maths today — that is Day 4. Today only records the
constants.

**Also update the repository layout in `CLAUDE.md`** to include
`simulator/schedule.py`, and add `ingest/localisation.py` and
`scripts/score_localisation.py`, which today creates.

Run the full test suite after the schema version bump and fix whatever breaks.
**Commit.**

---

## Task 2 — `db/schema.sql` and `db/repo.py`

SQLite. Every statement that touches SQL lives in `db/repo.py`; no other module in
the project may contain a SQL string.

**Connection settings.** Open with `PRAGMA journal_mode=WAL` and
`PRAGMA foreign_keys=ON`. WAL is not optional: from Day 7 the dashboard reads the
database while the ingest process writes to it, and the default rollback journal
will lock them against each other. Set a `busy_timeout` of a few seconds.

**Tables.**

| Table | Holds |
|---|---|
| `zone_env` | One row per zone environmental reading |
| `ble_sightings` | One row per tag seen per zone per scan window (raw) |
| `machine_status` | One row per machine reading |
| `gate_events` | RFID check-in / check-out events |
| `ppe_entries` | Supervisor-recorded PPE events |
| `device_status` | Birth and LWT messages |
| `rejected_records` | Records that failed validation, with the reason and the raw payload |
| `zone_assignments` | Resolved worker-in-zone intervals (written in Task 5) |
| `worker_sessions` | Resolved shift sessions from gate events (written in Task 6) |

**Columns that every telemetry table carries:**
- `id` INTEGER PRIMARY KEY
- `device_id` TEXT NOT NULL
- `ts` TEXT NOT NULL — the device's own UTC ISO-8601 timestamp
- `received_ts` TEXT NOT NULL — when the backend got it
- `quality` TEXT NOT NULL — `'ok'` or `'flagged'`
- `flags` TEXT NOT NULL — JSON array of flag strings, `'[]'` when clean
- `schema_version` TEXT NOT NULL

Store each payload's own fields as real typed columns, not a JSON blob. A value that
the device reported as `null` is stored as SQL `NULL`, with the reason captured in
`flags`. Never substitute a zero or a default for a missing reading.

**Uniqueness and idempotency.**
- Telemetry tables: `UNIQUE (device_id, ts, topic)` — add a `topic` column.
- `gate_events` and `ppe_entries`: `UNIQUE (event_id)`.
- All inserts use `INSERT OR IGNORE` and report whether the row was new.

This matters because buffered records replay after an outage and may be delivered
twice. Re-running the same simulation run against the same database must not
double-count anything. Write a test that proves it: ingest a captured batch, count
rows, ingest the identical batch again, assert the counts are unchanged.

**Out-of-order arrival.** Records arrive non-chronologically after an outage replay.
No code may assume `received_ts` order matches `ts` order. Index on `ts`, and always
sort by `ts` when reading a time range.

**Indices.** `(zone_id, ts)` on `zone_env`, `(machine_id, ts)` on `machine_status`,
`(tag_id, ts)` on `ble_sightings`, `(worker_id, ts)` on the event tables,
`(worker_id, from_ts)` on `zone_assignments`.

**`db/repo.py` API.** Write-side: one `insert_*` per table, plus
`insert_rejected()`. Read-side, which later modules will use:
`get_zone_env(zone_id, start_ts, end_ts)`,
`get_ble_sightings(start_ts, end_ts)`,
`get_machine_status(machine_id, start_ts, end_ts)`,
`get_zone_assignments(worker_id, start_ts, end_ts)`,
`get_worker_sessions(worker_id, date)`,
`get_ingest_stats(start_ts, end_ts)` (counts by topic, flagged count, rejected
count by reason, per-device last-seen).

Provide `db/init_db.py` or a `--init` flag that creates a fresh database from
`schema.sql`, and make it safe to re-run.

`tests/test_repo.py`: schema creates cleanly; insert and read back each row type
including `NULL` values; duplicate insert is ignored; out-of-order inserts read back
in `ts` order; `get_ingest_stats` returns correct counts on a known fixture.

**Run the tests. Commit.**

---

## Task 3 — `ingest/validation.py`

Pure functions, no MQTT and no database. Input: a topic string and a parsed payload
dict. Output: a verdict object carrying `accepted` (bool), `flags` (list of strings),
and `reason` (string, only when rejected).

Implement exactly the rules in section 10 of the contract, including the v1.1
amendments. Rejection reasons and flag names must be a closed, enumerated set
defined in one place — the dashboard groups by them later, and free-text reasons
will not group.

Be careful with these:

- **The LWT form** (status, `online: false`, no `ts`) is valid. Do not reject it and
  do not fabricate a timestamp.
- **Stuck-value detection needs history.** Validation itself stays pure, so it takes
  the recent values for that sensor as an argument; the caller supplies them from a
  small in-memory ring buffer per `(device_id, field)`. Do not query the database
  per message for this.
- **A `null` value with a non-`ok` flag is valid.** A `null` value with an `ok` flag
  is a contradiction and must be rejected.
- **`zone_id` or `machine_id` disagreeing with the topic is a rejection**, not a
  flag. That is a device misconfiguration and silently accepting it would corrupt
  every downstream exposure calculation.

`tests/test_validation.py` is the most important test file of the day. Cover, with
one explicit test each: a clean record accepted; each rejection reason fires on a
crafted bad record; each flag fires; the LWT form accepted; a `null`-with-`ok`
contradiction rejected; a value exactly on a range boundary accepted (boundaries are
inclusive); a `ts` 11 minutes in the future rejected and one 9 minutes in the future
accepted.

**Run the tests. Commit.**

---

## Task 4 — `ingest/subscriber.py`

The long-running process that connects to MQTT, subscribes to `textile/#`, validates
every message and writes it through `db/repo.py`.

Requirements:

- CLI flags: `--broker`, `--port`, `--db`, `--log-level`, and `--stats-interval`
  (seconds between summary log lines).
- Subscribe at QoS 1. Parse the topic to determine the message type; an unrecognised
  topic is logged and counted, never crashes the process.
- Malformed JSON is a rejection with reason `malformed_json`, stored in
  `rejected_records` with the raw payload bytes.
- **An exception while handling one message must never kill the subscriber.** Catch
  per message, log with the topic and a payload excerpt, count it, continue.
- Maintain the ring buffers needed for stuck-value detection.
- Log a summary line every `--stats-interval`: accepted and rejected counts by topic
  type, flagged count, and messages per second.
- Handle Ctrl-C cleanly: flush, close the database, log a final summary.
- Reconnect automatically if the broker drops, with backoff, and log it.

**Batch the writes.** At `--speed 100` the simulator produces messages far faster
than one-insert-per-message will comfortably handle. Buffer rows and commit in
batches (a few hundred rows, or every second, whichever comes first), and flush the
buffer on shutdown. A dropped batch at exit is a data-loss bug.

`tests/test_subscriber.py`: using a fake MQTT client (do not require a live broker in
the test suite), feed a sequence of messages covering every topic type plus several
bad ones, and assert the correct rows land in a temporary database and the correct
rejections are recorded. Assert that an exception raised while handling one message
does not prevent the next message from being processed.

**Run the tests. Commit.**

---

## Task 5 — `ingest/localisation.py` (BLE zone resolution)

This converts raw, overlapping BLE sightings into one defensible zone per worker per
time window. The simulator deliberately leaks each tag into neighbouring zones, so
naive strongest-RSSI-wins will flap between zones. The algorithm must be explicit,
tunable from `config.yaml`, and documented — a reviewer will ask how you handle a
tag seen by three nodes at once, and "we take the strongest" is a weak answer.

**Algorithm.** For each scan window and each `tag_id`:

1. **Smooth** the RSSI per `(tag_id, zone_id)` with an exponentially weighted moving
   average (`ewma_alpha` in config). Raw RSSI is noisy; smoothing before comparing is
   what stops most flapping.
2. **Rank** the candidate zones by smoothed RSSI.
3. **Hysteresis:** if the best zone differs from the worker's current zone but beats
   it by less than `rssi_margin_db`, keep the current zone. Only a clear win moves a
   worker.
4. **Debounce:** a new zone must win for `debounce_windows` consecutive windows
   before the assignment actually changes.
5. **Absence:** if a tag is not seen anywhere for `absence_windows` consecutive
   windows, assign zone `UNKNOWN` — never carry the last known zone forward
   indefinitely. A worker who left the building must not keep accruing exposure.
   This is a correctness requirement, not a nicety.
6. **Weak-signal floor:** ignore sightings below `min_rssi_dbm` entirely.

Add a `localisation` block to `config.yaml` with `ewma_alpha`, `rssi_margin_db`,
`debounce_windows`, `absence_windows` and `min_rssi_dbm`. Choose defensible starting
values and record in a comment why each was chosen.

**Output:** rows in `zone_assignments` as intervals —
`worker_id`, `zone_id`, `from_ts`, `to_ts`, `confidence`, `method`. Merge consecutive
identical assignments into one interval rather than storing one row per window;
the exposure engine wants intervals. `confidence` is a simple, documented function of
the margin over the runner-up. `method` is `'ble_ewma_hysteresis'`, so a later
change of algorithm stays distinguishable in stored data.

Label every assignment as `estimated_personal` in line with the integrity rules in
`CLAUDE.md`. This is an inference, not a measurement, and the code should say so.

Resolution can run either inside the subscriber or as a separate pass over stored
sightings. **Choose the separate pass** — it is easier to test, easier to re-run with
different parameters, and re-running it after tuning should not require replaying the
whole simulation.

`tests/test_localisation.py`: a clean single-zone signal resolves to that zone;
a tag seen in two zones with a clear margin picks the stronger; a tag seen in two
zones within the margin does **not** flap (assert zero changes across the window);
a disappearing tag becomes `UNKNOWN` after exactly `absence_windows`; intervals merge
correctly; a worker is never assigned two overlapping intervals.

**Run the tests. Commit.**

---

## Task 6 — Worker sessions, and measuring localisation accuracy

**6a — `worker_sessions` from gate events.**
Pair each `check_in` with the next `check_out` for that worker. Produce rows with
`worker_id`, `session_start`, `session_end`, `duration_s`, `shift`, `flags`.

Handle the broken cases explicitly, because they will occur:
- a check-in with no check-out → close it at shift end and flag
  `unmatched_check_in`
- a check-out with no preceding check-in → record and flag `orphan_check_out`
- two consecutive check-ins → flag `duplicate_check_in`
- a session longer than a configured maximum → flag `implausible_duration`

Then **clip `zone_assignments` to sessions.** A worker who is not checked in cannot
be accruing exposure, whatever a stray BLE sighting from the car park suggests.
Assignments outside any session are retained but marked `outside_session` so the
exposure engine can exclude them and the dashboard can show that it happened.

**6b — Ground truth and the accuracy score.**

Add `--ground-truth-out PATH` to `simulator/publisher.py`, writing JSONL of the
**true** worker zone per worker per window.

**This file must never travel over MQTT and must never enter the main database
tables.** Real hardware has no ground truth, so anything downstream that touches it
would be cheating. It is written to disk by the simulator and read only by the
scoring script.

Write `scripts/score_localisation.py` which loads the ground-truth file and the
resolved `zone_assignments`, aligns them on a common time grid, and reports:

- overall accuracy: percentage of worker-minutes assigned to the correct zone
- a confusion matrix over zones (which zones get mistaken for which)
- a separate count of `UNKNOWN` assignments, split into correctly unknown
  (the worker really was off-site) and incorrectly unknown
- per-zone recall, so a zone that localises badly is visible
- mean assignment latency: how long after a real zone change the algorithm follows
- the parameter values used for the run, printed alongside the results

Run it once and record the numbers in `docs/review_notes.md`. If accuracy is poor,
**report the number honestly and tune the parameters in a second run, recording
both.** Showing a tuning table at the review is a much stronger result than a single
unexplained figure, and the latency number is exactly the kind of limitation an
examiner respects you for stating yourself.

`tests/test_sessions.py` and `tests/test_scoring.py`: each malformed gate-event case
produces the right flag; clipping excludes out-of-session assignments; the scorer
returns 100% on a synthetic perfect match and a known lower value on a synthetic
case with one deliberate error.

**Run the tests. Commit.**

---

## Task 7 — End-to-end verification and notes

Run the full pipeline and measure it. This produces several of the IoT evaluation
metrics the project needs.

1. Start Mosquitto. Initialise a fresh database.
2. Start `ingest/subscriber.py`.
3. Run `python -m simulator.publisher --speed 100 --duration-h 8 --start <date>T05:30:00 --ground-truth-out data/truth.jsonl`
4. When it finishes, run the localisation pass, then the session builder, then
   `scripts/score_localisation.py`.

Record in `docs/review_notes.md`:

- **Data loss rate:** messages published (from the publisher's log) versus rows
  stored plus rows rejected. These should reconcile exactly. **If they do not,
  find out why before moving on** — a silent gap here invalidates every exposure
  number computed later in the project.
- **Ingest throughput:** messages per second sustained at `--speed 100`.
- **Validation results:** counts by rejection reason and by flag. Confirm the
  deliberately-faulty sensors and the same-second gate pair actually appear.
- **Idempotency:** re-run the same ingest over the stored payloads and show the row
  counts are unchanged.
- **Localisation accuracy:** the full scorer output, plus the parameters used.
- **Outage handling:** show that the Z3 outage records are present with their
  original timestamps and are correctly out of order by `received_ts`.
- What is pending for Day 3.

Run `pytest` across the whole project and show the full output. **Final commit.**

---

## Definition of done for Day 2

- [ ] `pytest` passes, no failures, no skips
- [ ] Contract is at v1.1 with a changelog; all five amendments applied
- [ ] A full `--speed 100` run reconciles exactly: published = stored + rejected
- [ ] Re-ingesting the same data changes no row counts
- [ ] Every validation rejection reason and flag has fired at least once in a real run
- [ ] `zone_assignments` contains merged intervals with no overlaps per worker
- [ ] `scripts/score_localisation.py` prints an accuracy figure and a confusion matrix
- [ ] Ground truth never enters the database or MQTT
- [ ] No SQL outside `db/repo.py`; no constants outside `config.yaml`
- [ ] `docs/review_notes.md` updated with all measured numbers
- [ ] All work committed

---

## Working instructions for this session

Work the tasks **in order**. Task 1 first — the schema version bump will break tests,
and it is much cheaper to fix that before new code depends on the old version.

After each task: run its tests, show the output, commit, say what is next.

Do not start the exposure engine, ML or dashboard today, however tempting it is once
data is in the database.

If the localisation accuracy comes out low, that is information, not failure. Report
the real number, say which parameter you think is responsible, and tune once. Do not
quietly adjust the simulator to make the algorithm look better — that would make
every number in the paper meaningless.