# DAY 5 — Dashboard and Review Pack (TRIMMED FOR FIRST REVIEW)

**This file replaces the earlier DAY_5.md.** Scope is cut deliberately. This is a
first review, not a final submission, and a rehearsed three-page demo beats a
half-finished five-page one.

**Read `CLAUDE.md`, `docs/results_summary.md` and `docs/review_notes.md` first.**

**No API tier.** The dashboard reads SQLite through `db/repo.py` only. No SQL in
`dashboard/`.

---

## Priority bands — read before starting

Work strictly in this order. **If you reach two hours of work, stop wherever you are
and jump to Task 6.**

| Band | Tasks | Why |
|---|---|---|
| **MUST** | 1, 2, 3, 6 | Without these there is no demo |
| **SHOULD** | 4, 5 | Visible value, but the Day-4 PNGs already cover the content |
| **IF TIME** | 7 | Nice to have, cut without hesitation |

**Explicitly cut from the original plan:** the standalone Factory Heat Map page (the
PNG from the results pack covers it), the separate Models page (merged into Task 5),
and the separate Alerts page (merged into Task 5). Do not rebuild them.

---

## Task 1 (MUST) — Fix the risk index hearing component

Do this first; it changes stored data that the dashboard then reads.

The brief's "100% dose maps to 100" rule saturates. It cannot distinguish a 235%
dose from a 3,378% dose, which produces the ordering inversion you found between
W102 and W105. **This was an error in the specification, not in your
implementation.**

**Replace the hearing component's input: use `l_ex_8h_db` instead of dose %.**

- dB is already logarithmic in acoustic energy, so no ad-hoc log transform is needed
- it is the unit the standards themselves use, so the choice is citable rather than
  invented
- map linearly over a configured range, defaulting to `75 dBA → 0` and
  `105 dBA → 100`, clamped at both ends, with both bounds in `config.yaml`

Keep dose % under both criteria displayed beside the index everywhere it appears —
the index ranks, the dose is the regulated quantity.

Then re-run the index over the 7 days and record the new distribution.

**Do not try to flatten the High count.** 119 High out of 140 worker-days is likely
the correct finding: the Erode power-loom study found 88.5% of weaving-section
workers above 95 dBA, so a loud mill producing mostly-High ratings is a result, not
a calibration fault. Only the ordering within High was broken. If the new
distribution is still mostly High, report it and cite that study as corroboration.

Keep the monotonicity test, and add one asserting that a worker with a materially
higher `l_ex_8h_db` never scores below one with a lower value when other components
are equal. That is the regression test for the bug just fixed.

**Run tests. Commit.**

---

## Task 2 (MUST) — Scaffolding

`dashboard/app.py` plus `dashboard/pages/`. Keep `app.py` to routing and shared setup.

- **Header band on every page:** project title, `SIMULATED DATA` badge, the loaded
  date range, engine version. The badge is a header element, not a footnote.
- **`@st.cache_data` with a ~10 s TTL on every read.** Without it Streamlit
  re-queries on every widget interaction and the demo feels broken.
- **Open the database read-only** (`file:...?mode=ro` URI). WAL is already on, so
  live ingest can write while the dashboard reads; read-only makes it impossible for
  the dashboard to damage a run mid-demo.
- Shared date/shift selector in the sidebar via session state.
- **Graceful empty state on every page.** No data for the selected range must print
  a plain message, never raise. This is the most likely live failure.

**Visual conventions**, fixed across all pages: one colour per hazard (noise, dust,
heat, CO2); never colour alone to carry meaning; units on every axis and tile; the
same three band colours in the same order everywhere. Check legibility projected.

**Run it. Commit.**

---

## Task 3 (MUST) — Page: Worker Drilldown

The single most important page. It shows the entire value chain in one screen:
sensors to localisation to exposure to index. If only one page works, this is it.

**Top row — four tiles:** L_EX,8h with exposure duration; dose under NIOSH **and**
the India criterion side by side, each labelled with its source; WBGT max and time
above limit, carrying the approximation flag; PM2.5 TWA raw and corrected, labelled
`indicative`.

**Data quality band — directly under the tiles, not at the page bottom:**
`data_coverage_pct`, `unaccounted_pct`, and one plain sentence stating that where
unaccounted time exists the figures are **lower bounds**. This sentence is what makes
every other number on the page credible.

**Exposure timeline:** a horizontal band of which zone the worker occupied through
the shift, with noise overlaid. Draw `UNKNOWN` periods in a distinct neutral style —
blank reads as zero exposure, distinct reads as missing.

**Risk index panel:** overall index and band; a horizontal bar of the three component
contributions with percentages; the top two drivers in plain language; the confidence
state, showing "insufficient data" instead of a band where confidence is low. Beside
it, the dose figures from the top row, so the ranking and the regulated quantity are
visible together.

Permanently on screen, one line: *this index is a weighted design choice of this
project, not a validated clinical model.*

**Cumulative panel:** rolling 7-day and 30-day dose with `window_coverage_pct`,
partial windows marked partial.

**Default the page to W105** (or whichever worker now ranks highest), so the demo
opens on the interesting case rather than on W101.

**Commit.**

---

## Task 4 (SHOULD) — Page: Live Zones

Seven zone tiles: current noise dBA, PM2.5, WBGT, CO2, machines running, workers
present, sensor-health state, and the age of the most recent reading.

Below them, **the chart that wins arguments**: noise time series for the selected
zone and shift, with machine-running state shaded behind it and break periods
marked. A sceptical examiner tests the machine-noise coupling claim first, and this
chart answers it without a word.

**Commit.**

---

## Task 5 (SHOULD) — Page: System & Results

One page, three short sections. Mostly displaying artifacts that already exist.

**Alerts:** table of workers above 100% dose under either criterion, any 115 dBA
absolute-max event, WBGT exceedances, machine vibration anomalies. Columns: time,
worker or zone, type, value, threshold, criterion source.

**System health:** per device, last-seen and online state; rejection counts by
reason; flag counts by flag; `needs_clean` surfaced prominently. Then the ingest
reconciliation — published versus stored versus rejected. That these reconcile
exactly is a credibility result and belongs on screen.

**Results:** display the existing PNGs from `docs/figures/` as images — forecast
versus actual with the persistence baseline, the SHAP summary, the localisation
confusion matrix. Beneath them, the skill-score table with the break-free-window
figures called out separately, the exceedance lead-time numbers, and one short
paragraph stating that the ML risk classifier is deferred until questionnaire labels
exist, with the deterministic index as its baseline.

Do not run model inference in the dashboard. Read
`data/forecast_predictions.parquet` and the saved figures.

**Commit.**

---

## Task 6 (MUST) — Fallback, demo script, numbers sheet

**The highest-value work of the day. Do this even if Tasks 4 and 5 are unfinished.**

**6a — Fallback path.** Pre-load a database with the full 7-day run and commit a
demo-ready copy where `.gitignore` does not exclude it. Every page must work from
stored data with **no simulator and no broker running**.

Then rehearse the demo once **with the network disabled and nothing running but the
dashboard.** Your pipeline has four moving parts; if one stalls in front of the
panel, the static path is the only thing that saves the session. This is 15 minutes
and it is not optional.

**6b — `docs/demo_script.md`.** The exact command sequence with the expected visible
outcome and approximate duration of each step:

```
1. Start Mosquitto                       (service already running)
2. python -m ingest.subscriber           (watch the stats line)
3. python -m simulator.publisher --speed 100 --align-shifts
4. Dashboard: Worker Drilldown (W105)    (the full value chain)
5. Dashboard: Live Zones                 (noise tracking machine state)
6. pytest                                (364 tests, green)
7. docs/data_contract.md + firmware/     (drop-in readiness)
```

**6c — Numbers sheet.** One printed page from `docs/results_summary.md` to hold
during the review: 364 tests; data-loss reconciliation (49,979 + 13 = 49,992);
173.6 msg/s; localisation 96.4% at 30 s median latency; consistency residual 0.0 dB;
forecast skill scores including the break-free-window figures; 36/36 crossings warned
at 17–20 min median lead with 5–12% false alarms; risk-index distribution. When asked
"how accurate is it", read a number rather than hedging.

**Commit.**

---

## Task 7 (IF TIME) — Firmware stub and Q&A notes

**7a — `firmware/zone_node/zone_node.ino`.** An Arduino sketch that connects to
Wi-Fi and MQTT, declares the same topics, and builds the identical v1.1 JSON payload
with ArduinoJson. Sensor reads are clearly-marked stubs returning fixed values, each
commented with the actual sensor and library it will use. Register the LWT exactly as
the contract specifies, with no `ts` on the offline message. Include the RTC
timestamp call and the microSD buffering path as commented stubs.

`firmware/README.md`: pin assignments, required libraries, and the sentence that
matters — **the backend requires no changes when this replaces the simulator,
because both publish contract v1.1.** Do not compile it, and say it is untested on
hardware rather than implying otherwise.

**7b — `docs/review_qa.md`**, two or three sentences each:

1. Is this real data? — Simulated; here is the zone-profile basis and the drop-in path.
2. How do you know the exposure maths is right? — Twelve hand-verified cases, four
   rows of Schedule XXVIII reproduced, single-zone consistency at 0.0 dB.
3. Why two noise criteria? — NIOSH and the TN rules differ in criterion, exchange
   rate **and** accumulation threshold; both shown rather than choosing.
4. Didn't your model just learn the break schedule? — Partly, and we measured it:
   skill on break-free windows is +0.56–0.62 against +0.39–0.51 for the seasonal
   baseline.
5. Why does the seasonal baseline beat ML at 60 minutes? — The simulator runs an
   identical daily timetable; a real mill varies, so this understates the ML
   advantage rather than overstating it.
6. How accurate is the worker tracking? — 96.4% of on-site worker-minutes, 30 s
   median latency, errors almost entirely transition lag between neighbours.
7. Why not the better-scoring localisation settings? — 98.6% came with ~40 false
   zone changes; retuning waits for real hardware.
8. Is the risk score medically validated? — No. A transparent weighted index; the ML
   classifier is deferred until questionnaire labels exist.
9. Why drop the heart-rate and respiratory sensors? — The feasibility table; lead
   with it.
10. What happens when the network fails? — microSD buffering with original
    timestamps; demonstrate the Z3 outage replay.
11. Can the dust readings be compared to legal limits? — No, and no function in the
    codebase does so; regulatory cotton dust is the elutriated fraction.
12. What is left? — Hardware deployment, questionnaire labels, calibration against a
    handheld meter, TimescaleDB migration.

Update `docs/review_notes.md`. Run the full suite. **Final commit.**

---

## Definition of done

**Minimum viable (MUST band):**
- [ ] `pytest` passes; exit code checked directly, not chained with `;` or piped
- [ ] Risk index uses `l_ex_8h_db`; ordering regression test passes
- [ ] Worker Drilldown renders with coverage, unaccounted time, lower-bound sentence,
      index contributions and the design-choice caveat
- [ ] `SIMULATED DATA` badge on every page; database opened read-only
- [ ] Graceful empty state on every page
- [ ] Demo-ready database committed and **the static-only path rehearsed with the
      network disabled**
- [ ] `docs/demo_script.md` and the numbers sheet written

**Then, if reached:** Live Zones, System & Results, firmware stub, Q&A notes.

---

## Working instructions

Tasks in strict order. Render before polishing; polish only if the MUST band is done.

**Add nothing that is not listed here.** Every addition is a thing that can break in
front of the panel, and this is a first review — the system already demonstrates far
more than a first review requires.

If something is unfinished when you stop, say so plainly in `review_notes.md` and in
your summary. An honest gap is fine at a first review. A surprise during the demo is
not.