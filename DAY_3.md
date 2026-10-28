# DAY 3 — The Exposure Engine

**Read `CLAUDE.md`, `docs/data_contract.md` and `docs/review_notes.md` first.**

**Goal of today:** turn stored sensor data and zone assignments into per-worker
exposure figures computed by published standard methods, with every number
hand-verifiable and every uncertainty stated rather than hidden.

This is the scientifically load-bearing module of the whole project. Everything the
ML layer and the dashboard do on later days rests on these numbers being right. It
is also the part an examiner is most likely to interrogate line by line, because the
formulas are published and checkable.

**Do NOT build today:** ML, API, dashboard. No plotting, no models.

**One correction carried in from the planning discussion.** An earlier chat message
stated that "4 hours at 91 dBA gives a 100% NIOSH dose." **That is wrong.** Under
NIOSH, allowed time is `8 / 2^((L − 85) / 3)` hours, so 91 dBA allows 2 hours and
4 hours at 91 dBA is a **200%** dose. If that figure reached any fixture or test,
fix it. Use the verified table in Task 2.

---

## Task 1 — Pre-flight fixes

Two Day-2 limitations must be closed before any exposure maths is written, because
both silently corrupt results rather than failing loudly.

**1a — Canonical timestamp ordering.**
`ts` is stored as TEXT and sorted lexicographically. A fractional-second timestamp
sorts *before* a whole-second one at the same instant, because `.` is 0x2E and `Z`
is 0x5A. Today's engine integrates over time-ordered readings, so a mis-sort
corrupts every dose it computes.

Add a `ts_epoch_ms` INTEGER column to every table that carries a `ts`. Populate it
at ingest by parsing `ts` once. Index on it. **Every read path and every ORDER BY
in `db/repo.py` switches to `ts_epoch_ms`.** Keep the original `ts` text column
exactly as received — it is the device's own record and must not be rewritten.

Write a migration that backfills `ts_epoch_ms` for the existing run data, so the
Day-2 runs stay usable. Add a test that ingests both `...:10Z` and `...:10.500Z`
and asserts they come back in true chronological order.

**1b — Shift-aligned run windows.**
Every session in the Day-2 run is flagged, because the window starts mid-night-shift
and ends mid-day-shift. The flags are correct, but a demo in which every session
carries a warning looks broken.

Add `--align-shifts` to `simulator/publisher.py` (default on): snap the start time
back to the nearest shift boundary and extend the end forward to the next one, so
whole sessions are captured. Keep the raw `--start` behaviour available with
`--no-align-shifts`, and keep at least one deliberately truncated session in the
test fixtures so the flagging path stays covered.

**Run the full suite. Commit.**

---

## Task 2 — `exposure/noise.py`

Pure functions. No database access, no config reads inside the functions — limits
are passed in as arguments so the functions stay testable in isolation.

### Functions to implement

**`l_aeq(levels_db, durations_s)` — energy-equivalent continuous level.**

```
L_Aeq,T = 10 * log10( (1/T) * Σ t_i * 10^(L_i / 10) )
```

Supports unequal durations. With equal durations it reduces to
`10*log10(mean(10**(L/10)))`. **Never average dB arithmetically anywhere in this
file.** Cite ISO 9612 in the docstring.

**`l_ex_8h(l_aeq_db, exposure_duration_h)` — daily noise exposure level.**

```
L_EX,8h = L_Aeq,Te + 10 * log10(T_e / 8)
```

**`allowed_time_h(level_db, criterion_db, exchange_rate_db, duration_h)`**

```
T_i = duration_h / 2 ** ((L_i − criterion_db) / exchange_rate_db)
```

**`noise_dose_pct(levels_db, durations_s, criterion)` — the dose.**

```
Dose% = 100 * Σ (C_i / T_i)
```

where `C_i` is time spent at level `L_i` and `T_i` is the allowed time at that
level. Levels **below the criterion's `threshold_db` contribute zero**. This is the
form given both by NIOSH and by Schedule XXVIII of the Tamil Nadu Factories Rules.

**`absolute_max_exceeded(levels_db, criterion)`** — returns the levels and their
timestamps where `absolute_max_db` (115 dBA for the India criterion) is exceeded.
This is a hard alarm condition, not a dose contribution, and must be reported
separately. No exposure above 115 dBA is permitted at all.

### The two criteria differ in three ways, not one

| | NIOSH | India (TN Schedule XXVIII) |
|---|---|---|
| Criterion level | 85 dBA | 90 dBA |
| Exchange rate | 3 dB | 5 dB |
| Accumulation threshold | 80 dBA | 90 dBA |
| Absolute maximum | — | 115 dBA |

The differing threshold is the one most implementations get wrong. Under the India
criterion, an hour at 85 dBA contributes **nothing** to the dose; under NIOSH it
contributes. Both figures are reported side by side, always labelled with their
source.

### `tests/test_noise.py` — hand-verified cases

Every one of these is computed by hand; assert to within 0.1 dB or 0.5% dose.

| # | Input | Criterion | Expected |
|---|---|---|---|
| 1 | 8 h @ 85 dBA | NIOSH | L_EX,8h = 85.0, dose = 100% |
| 2 | 2 h @ 91 dBA | NIOSH | dose = 100% |
| 3 | 4 h @ 91 dBA | NIOSH | dose = **200%** |
| 4 | 1 h @ 94 dBA | NIOSH | dose = 100% |
| 5 | 8 h @ 90 dBA | India | dose = 100% |
| 6 | 4 h @ 95 dBA | India | dose = 100% |
| 7 | 2 h @ 100 dBA | India | dose = 100% |
| 8 | 0.25 h @ 115 dBA | India | dose = 100% **and** absolute-max alarm fires |
| 9 | 8 h @ 85 dBA | India | dose = **0%** (below the 90 dBA threshold) |
| 10 | 4 h @ 80 + 4 h @ 90 dBA | — | L_Aeq = **87.4 dBA** (arithmetic mean would give 85 — this test is specifically here to catch that error) |
| 11 | 8 h @ 92 dBA | both | NIOSH ≈ 503%, India ≈ 132% — the criteria diverge sharply, which is why both are shown |
| 12 | Empty input | — | raises, does not return 0 |

Cases 5 to 8 reproduce four rows of the published Schedule XXVIII table, so passing
them demonstrates agreement with the regulation itself. Say so in the test
docstrings.

**Run the tests. Commit.**

---

## Task 3 — `exposure/dust.py` and `exposure/heat.py`

### `exposure/dust.py`

**`twa(concentrations, durations_s)`** — time-weighted average,
`Σ(C_i · t_i) / Σ t_i`. Unlike noise, this is a **linear** average; dust mass
concentration is not logarithmic. Make that explicit in the docstring, because the
noise module sitting next door invites the wrong instinct.

**`correct_humidity(pm_raw, rh_pct, params)`** — hygroscopic growth correction,
applied only above `rh_correction_threshold_pct` (75%). Implement it as a
**pluggable, cited** function, not an invented formula. Use the hygroscopic growth
form:

```
C_corrected = C_raw / (1 + kappa * (a_w / (1 − a_w)))      where a_w = RH / 100
```

with `kappa` in `config.yaml`, defaulted conservatively and commented with the
reasoning. Cite Crilley et al. (2018), *Atmospheric Measurement Techniques*, as the
source of the correction form.

Three requirements on this function:
1. The **raw value must always remain retrievable** alongside the corrected one.
   Never overwrite raw sensor data with a corrected value.
2. Output carries a `correction_applied` boolean and the `kappa` used, so a later
   change of parameters is traceable in stored data.
3. Cotton dust is fibrous and far less hygroscopic than the ambient aerosol this
   correction was derived for, so `kappa` is an assumption, not a measurement. State
   that in the docstring and in `docs/exposure_methods.md`.

**Do not implement any comparison against regulatory cotton dust limits.** Those
limits are defined for the lint-free respirable fraction collected by vertical
elutriator; an optical PM2.5/PM10 sensor does not measure that fraction. Every dust
output is labelled `indicative` and is used for trends, zone comparison and ML
features only. A function that compares PM readings to a legal limit must not exist
in this codebase.

### `exposure/heat.py`

**`wbgt_indoor(t_nw_c, t_g_c)`** — ISO 7243, no solar load:

```
WBGT = 0.7 * T_nw + 0.3 * T_g
```

**`estimate_natural_wet_bulb(temp_c, rh_pct)`** — the honest weak point of the
system, and it must be labelled as such.

Implement Stull's (2011) psychrometric wet-bulb approximation. Then state clearly,
in the docstring and in the returned metadata, that **psychrometric wet-bulb is not
natural wet-bulb**. T_nw is generally higher than T_pwb in still air, so using
T_pwb as a substitute **underestimates** WBGT — it errs toward reporting conditions
as safer than they are. Every WBGT value therefore carries
`t_nw_method: "stull_psychrometric_approximation"` and an `approximate: true` flag
that follows it all the way to the dashboard.

Returning a conservative-direction error that is clearly labelled is acceptable
engineering. Returning it silently is not.

**`wbgt_exceedance(wbgt_series, durations_s, limit_c)`** — time above the ISO 7243
limit for the configured metabolic-rate category, plus the maximum WBGT reached.
The metabolic-rate assumption (moderate work) is recorded with every result.

`tests/test_dust.py` and `tests/test_heat.py`: TWA on hand-computed cases including
unequal durations; correction is a no-op below the RH threshold and monotonic above
it; raw values survive correction; WBGT matches hand-computed values; Stull's formula
reproduces two published reference points; the approximation flag is always present.

**Run the tests. Commit.**

---

## Task 4 — `exposure/personal.py`

This is the integration step that makes the project what it claims to be: it joins
*where each worker was* to *what the sensors measured there*. It is also where
honesty about uncertainty is implemented.

For a given worker and a given shift session:

1. Fetch that worker's `zone_assignments`, clipped to the session, excluding
   intervals marked `outside_session`.
2. For each interval, fetch `zone_env` rows for that zone over that time range,
   ordered by `ts_epoch_ms`.
3. Accumulate across all intervals: energy-weighted L_Aeq and L_EX,8h; dose under
   both criteria; dust TWA on both raw and corrected PM; WBGT exceedance time and
   maximum; CO2 time-weighted mean.
4. Return a result object carrying the values **and** the accounting described below.

### Three rules that are not optional

**`UNKNOWN` zone time is never silently handled.** Do not treat it as zero exposure,
and do not carry the last known zone forward. Report it as `unaccounted_time_s` and
as a percentage of the session. A worker with 25% unaccounted time has an exposure
figure that should not be trusted, and the system must say so rather than quietly
reporting a confident-looking number.

**Missing sensor readings are never treated as zero.** A `NULL` noise value is
absent data, not silence. Exclude it from the integration and reduce the reported
`data_coverage_pct` accordingly. A dose computed from 40% coverage must be visibly
distinguishable from one computed from 99% coverage.

**Every result is labelled `estimated_personal`** per the integrity rules in
`CLAUDE.md`, because it is zone data weighted by inferred presence — not a personal
measurement. The optional wearable path, if it ever produces data, is
`measured_personal` and must remain a separate field rather than overwriting this one.

### Result fields

`worker_id`, `session_start`, `session_end`, `l_aeq_db`, `l_ex_8h_db`,
`dose_niosh_pct`, `dose_india_pct`, `absolute_max_events`, `pm25_twa_raw`,
`pm25_twa_corrected`, `pm10_twa_raw`, `pm10_twa_corrected`, `wbgt_max_c`,
`wbgt_time_above_limit_s`, `co2_twa_ppm`, `data_coverage_pct`,
`unaccounted_time_s`, `unaccounted_pct`, `zones_visited`, `label`, `flags`.

### The internal consistency test that matters most

For a worker who never left a single zone during a session, the personal L_EX,8h
**must** equal the zone's own L_Aeq normalised to that worker's exposure duration,
within floating-point tolerance. Find or construct such a worker in the run data and
assert this. It independently cross-checks the zone pipeline, the assignment
pipeline and the personal integration against each other — if any one of the three
is wrong, this test fails. It is the single most valuable test in the project.

`tests/test_personal.py`: the consistency test above; a synthetic worker split
50/50 between a 95 dBA zone and a 65 dBA zone gives an L_Aeq dominated by the louder
zone (hand-computed); unaccounted time is reported, not absorbed; `NULL` readings
reduce coverage rather than depressing the dose; a session with zero valid readings
returns a null result with a flag rather than a zero dose.

**Run the tests. Commit.**

---

## Task 5 — `exposure/cumulative.py` and materialised tables

### Long-horizon data

Cumulative exposure needs multiple days, and the Day-2 run is 8 hours. Before
building this, generate a longer dataset.

Ingest sustained 173 msg/s on Day 2, which caps how fast the simulator can run. A
30-day run at that rate takes hours. **Do a 7-day run for the review** at a speed
the ingest path sustains comfortably, and design `cumulative.py` so a 30-day window
over 7 days of data returns a **partial** result with an explicit
`window_coverage_pct`, rather than silently averaging over whatever exists. A
30-day run can be left going overnight before the review if you want the full
window; the code must not care either way.

### Functions

Rolling 7-day and 30-day aggregates per worker: cumulative noise dose (sum of daily
dose percentages), mean and maximum daily L_EX,8h, cumulative dust TWA-hours, total
hours above the WBGT limit, number of days with dose above 100%, and
`window_coverage_pct`.

Partial windows are reported as partial. Never scale a 7-day total up to a notional
30-day figure.

### Materialised tables

Add to `db/schema.sql`, written through `db/repo.py` as always:

- `personal_exposure_shift` — one row per worker per session, holding the Task 4
  result fields
- `personal_exposure_cumulative` — one row per worker per as-of date
- `zone_exposure_summary` — one row per zone per shift, so the dashboard's heat map
  does not recompute from raw data on every page load

Recomputation must be **idempotent**: recomputing a date replaces that date's rows
and leaves every other date untouched. Store the engine version and the parameters
used with each row, so a row computed under different assumptions is identifiable
later. This matters when a reviewer asks whether a number came before or after a
parameter change.

`tests/test_cumulative.py`: rolling windows over a synthetic multi-day fixture give
hand-computed totals; a partial window reports partial coverage and is not scaled up;
recomputing one date leaves adjacent dates byte-identical.

**Run the tests. Commit.**

---

## Task 6 — `exposure/engine.py`

The orchestration CLI that later days and the dashboard both call.

- `--date YYYY-MM-DD` or `--from` / `--to`
- `--worker` to restrict to one worker, otherwise all
- `--recompute` to overwrite existing rows for the range
- `--dry-run` to print results without writing

Behaviour: resolve sessions, compute per-session personal exposure, compute zone
summaries, compute cumulative aggregates, write all three tables, and print a
summary table to stdout — worker, dose under each criterion, coverage, unaccounted
percentage, flags.

Log loudly and skip, rather than crashing, when a worker has no sessions or a
session has no readings. One bad worker must not abort the batch.

**Run the tests. Commit.**

---

## Task 7 — Verification and `docs/exposure_methods.md`

Run the engine over the 7-day dataset, then write `docs/exposure_methods.md`
containing:

- every formula implemented, in the same notation as the code, each with its
  standard cited (ISO 9612, ISO 7243, NIOSH 98-126, Schedule XXVIII under Rule 95 of
  the Tamil Nadu Factories Rules, Crilley et al. 2018 for the PM correction)
- the full hand-verified test table from Task 2, with the computed values beside the
  expected ones
- the four rows of Schedule XXVIII that the tests reproduce, shown as agreement with
  the published regulation
- **a stated-assumptions section**: psychrometric wet-bulb standing in for natural
  wet-bulb and the direction of that error; `kappa` for the PM correction; the
  moderate-work metabolic category; zone-weighted presence as a proxy for personal
  exposure; optical PM not being the regulatory elutriated fraction
- the internal consistency result from Task 4
- results over the 7-day dataset: exposure distribution across workers, how many
  worker-days exceed 100% under each criterion, mean data coverage, mean unaccounted
  time, and which zones drive the highest doses

Then update `docs/review_notes.md` with what was built, the measured numbers, and
what is pending for Day 4.

Run `pytest` across the project and show the full output. **Final commit.**

---

## Definition of done for Day 3

- [ ] `pytest` passes, no failures, no skips
- [ ] `ts_epoch_ms` present, indexed, backfilled, and used by every ordered read
- [ ] All twelve hand-verified noise cases pass, including the 200% case and the
      87.4 dBA energy-averaging case
- [ ] Four rows of Schedule XXVIII reproduced by the India criterion
- [ ] Both criteria reported side by side with differing thresholds applied correctly
- [ ] No function anywhere compares optical PM against a regulatory dust limit
- [ ] WBGT carries its approximation flag end to end
- [ ] The single-zone internal consistency test passes
- [ ] Unaccounted time and data coverage are reported on every personal result
- [ ] 7-day dataset ingested; cumulative reports partial windows honestly
- [ ] Recomputation is idempotent
- [ ] `docs/exposure_methods.md` written, including stated assumptions
- [ ] All work committed

---

## Working instructions for this session

Tasks in order. Task 1 before any maths — a timestamp mis-sort corrupts results
silently, which is the worst failure mode available to us.

After each task: run its tests, show the output, commit, state what is next.

Where a standard is ambiguous, implement the conservative reading, state the choice
in the docstring and in `docs/exposure_methods.md`, and flag it in your response. Do
not average anything in dB arithmetically. Do not invent a formula and present it as
established practice — if a correction or approximation has no citable source, label
it as an assumption of this project.

If a hand-verified test case disagrees with your implementation, **the test is
probably right.** Recompute by hand before changing the expected value, and if you
do change one, show the hand calculation in your response.