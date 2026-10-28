# Exposure Methods

How the exposure engine (`exposure/`) turns zone sensor data and zone assignments into
per-worker exposure figures. Every formula below is the one the code implements, in
the same notation, with its source. Every number in the verification table was
computed by hand and is asserted by `tests/test_noise.py`.

**Status of all figures: SIMULATED DATA.** Personal figures are `estimated_personal`
(zone data weighted by inferred presence, not a personal measurement); zone figures
are `measured_environmental`; dust figures are `indicative`; WBGT figures are
`approximate`.

---

## 1. Noise (`exposure/noise.py`)

Sources: **ISO 9612:2009** (L_Aeq,T, L_EX,8h); **NIOSH 98-126** (1998), *Criteria for
a Recommended Standard: Occupational Noise Exposure*; **Schedule XXVIII under Rule 95
of the Tamil Nadu Factories Rules**.

**Equivalent continuous level** (ISO 9612). Energy average, unequal durations allowed:

```
L_Aeq,T = 10 · log10( (1/T) · Σ t_i · 10^(L_i/10) ),    T = Σ t_i
```

Decibels are never averaged arithmetically anywhere in the engine; a token-level
test fails the build if `mean`, `average` or `/ len(` appears in the noise module.

**Daily exposure level** (ISO 9612), T_0 = 8 h by definition:

```
L_EX,8h = L_Aeq,Te + 10 · log10(T_e / T_0)
```

**Permitted time** at level L_i:

```
T_i = duration_h / 2^((L_i − criterion_db) / exchange_rate_db)
```

**Dose**, levels below the criterion's threshold contributing nothing:

```
Dose% = 100 · Σ (C_i / T_i)        over readings with L_i ≥ threshold_db
```

**Absolute maximum** (India): any reading at or above 115 dBA is a hard alarm,
reported as an event list separate from the dose.

### The two criteria differ in three ways

| | NIOSH | India (TN Schedule XXVIII) |
|---|---|---|
| Criterion level | 85 dBA | 90 dBA |
| Exchange rate | 3 dB | 5 dB |
| Accumulation threshold | 80 dBA | 90 dBA |
| Absolute maximum | — | 115 dBA |

Both doses are always computed and reported side by side, each labelled with its
source. An hour at 85 dBA adds 12.5% under NIOSH and **nothing** under India.

### Choices where the standards leave room (conservative readings)

- **Threshold inclusive.** A level exactly at the threshold counts toward the dose.
- **Absolute maximum at ≥ 115 dBA.** "No exposure in excess of 115 dBA" read
  literally would exclude 115 itself. The alarm fires at 115 because that is the
  schedule's own 15-minute ceiling row (test case 8).
- **The 115 dBA check uses `noise_dba`** (the 10-s L_Aeq), not `noise_peak_dba`: the
  schedule limits the sound level, not the instantaneous peak.
- **NIOSH 3 dB is a rounded energy step.** Exact energy doubling is
  10·log10(2) = 3.0103 dB, so the dose computed directly (Σ C/T) and the dose inferred
  from L_EX,8h, 100·2^((L_EX,8h − 85)/3), differ slightly. For 4 h @ 91 dBA: 200.0% vs
  199.5% (0.24%). The engine reports the direct Σ C/T value, which is the definition.

### Hand-verified test table (`tests/test_noise.py`)

Tolerance 0.1 dB or 0.5% dose. "Computed" is the implementation's output.

| # | Input | Criterion | Hand calculation | Expected | Computed |
|---|---|---|---|---|---|
| 1 | 8 h @ 85 dBA | NIOSH | 85 + 10 log10(8/8); 8 / (8/2^0) | L_EX,8h 85.0, 100% | 85.0, 100.0% |
| 2 | 2 h @ 91 dBA | NIOSH | T = 8/2^(6/3) = 2 h | 100% | 100.0% |
| 3 | 4 h @ 91 dBA | NIOSH | 4 / 2 | **200%** | 200.0% |
| 4 | 1 h @ 94 dBA | NIOSH | T = 8/2^(9/3) = 1 h | 100% | 100.0% |
| 5 | 8 h @ 90 dBA | India | T = 8/2^0 = 8 h | 100% | 100.0% |
| 6 | 4 h @ 95 dBA | India | T = 8/2^(5/5) = 4 h | 100% | 100.0% |
| 7 | 2 h @ 100 dBA | India | T = 8/2^(10/5) = 2 h | 100% | 100.0% |
| 8 | 0.25 h @ 115 dBA | India | T = 8/2^(25/5) = 0.25 h; ≥ 115 | 100% **and** alarm | 100.0%, 1 alarm event |
| 9 | 8 h @ 85 dBA | India | 85 < 90 threshold | **0%** | 0.0% |
| 10 | 4 h @ 80 + 4 h @ 90 | — | 10 log10((4·10^8 + 4·10^9)/8) | L_Aeq **87.4 dBA** (not 85) | 87.40 dBA |
| 11 | 8 h @ 92 dBA | both | NIOSH 8/1.587; India 8/6.063 | ≈ 504% / ≈ 132% | 504.0% / 132.0% |
| 12 | empty input | — | — | raises | raises `ValueError` |

The planning-stage claim "4 h at 91 dBA gives 100% NIOSH" is wrong; case 3 shows it
is 200%. It appears in no fixture or test.

### Agreement with the published regulation

Cases 5–8 reproduce four rows of Schedule XXVIII of the Tamil Nadu Factories Rules;
`test_allowed_time_table_rows` checks all six:

| Level (dBA) | Permitted (Schedule XXVIII) | `allowed_time_h(L, 90, 5, 8)` |
|---|---|---|
| 90 | 8 h | 8.0 h |
| 95 | 4 h | 4.0 h |
| 100 | 2 h | 2.0 h |
| 105 | 1 h | 1.0 h |
| 110 | 30 min | 0.5 h |
| 115 | 15 min | 0.25 h |

---

## 2. Dust (`exposure/dust.py`)

**Time-weighted average** — a **linear** average, because mass concentration is not
logarithmic:

```
TWA = Σ (C_i · t_i) / Σ t_i
```

**Humidity correction** (form from Crilley et al. 2018, *Atmospheric Measurement
Techniques* 11, 709–720; κ-Köhler hygroscopic growth), applied only above 75% RH:

```
C_corrected = C_raw / (1 + κ · a_w / (1 − a_w)),     a_w = RH / 100
```

The raw value is always kept beside the corrected one; each result records
`correction_applied` and the κ used.

**No comparison with a regulatory cotton-dust limit exists in this codebase**, and a
test enforces it. Those limits apply to the lint-free respirable fraction collected by
vertical elutriator and weighed; an optical PM2.5/PM10 sensor does not measure that
fraction. Dust figures are labelled `indicative`, for trends, zone comparison and
ML features only.

---

## 3. Heat (`exposure/heat.py`)

**Indoor WBGT**, no solar load (ISO 7243:2017):

```
WBGT = 0.7 · T_nw + 0.3 · T_g
```

**Natural wet-bulb is not measured.** It is replaced by Stull's (2011) psychrometric
wet-bulb (*J. Appl. Meteor. Climatol.* 50, 2267–2269):

```
T_w = T·atan[0.151977 (RH + 8.313659)^½] + atan(T + RH) − atan(RH − 1.676331)
      + 0.00391838 · RH^(3/2) · atan(0.023101 · RH) − 4.686035
```

Verification: it reproduces the paper's worked example (20 °C, 50% RH → 13.7 °C), and
across mill conditions (28–38 °C, 50–85% RH) it agrees with an independent
sea-level psychrometric solve within Stull's stated error (−1 to +0.65 °C).

**Exceedance:** time with WBGT strictly above the ISO 7243 reference value for the
configured metabolic class (moderate work, acclimatised: 28 °C), plus the maximum.
Every WBGT value carries `t_nw_method = "stull_psychrometric_approximation"` and
`approximate = true`, all the way into the stored tables.

---

## 4. Personal exposure (`exposure/personal.py`)

For each worker and shift session:

1. Take the worker's `zone_assignments` overlapping the session, excluding pieces
   marked `outside_session`, and clip them to the session.
2. For each zone interval, take that zone's `zone_env` rows. **Each reading covers the
   10 s before its `ts`** (the contract's L_Aeq over the reporting interval), and is
   weighted by its exact overlap with the interval.
3. Within each interval, the valid readings stand for the whole interval. This is
   the ISO 9612 task-based principle: a sample represents the task duration.
4. Combine intervals: energy-weighted L_Aeq over the accounted time, L_EX,8h with
   T_e = the accounted time that had data, and doses under both criteria. Also raw
   and corrected PM TWAs, WBGT time above the limit and maximum, and the CO2 TWA.

**Accounting reported with every result.**

- **`unaccounted_time_s` / `unaccounted_pct`:** session time whose zone is UNKNOWN or
  unassigned. It is never zero-filled and never carried forward. The figures cover
  accounted time only, so **they are lower bounds where unaccounted time exists**.
  The result is flagged `incomplete_presence` above 10%.
- **`data_coverage_pct`:** the share of accounted time covered by valid noise
  readings. A NULL reading lowers coverage, not the dose. An interval with no valid
  reading at all is not imputed; it shows up as lost coverage. The result is flagged
  `low_data_coverage` below 90%.
- **No valid readings** gives a null result flagged `no_valid_readings`, never a zero
  dose.
- **Detected sensor anomalies are excluded.** Records that validation flagged `spike`
  or `stuck_value` are left out of integration and counted in
  `excluded_anomalous_readings`, so they reduce coverage. An excluded reading at or
  above 115 dBA is listed in `excluded_absolute_max_candidates`, not dropped silently,
  because a real event and a glitch look the same in the data. Records flagged
  `sensor_not_ok` are not excluded: their bad fields are already null, and a
  `needs_clean` PM value reads high, which errs safe.

  *Why this rule exists, a finding of the 7-day run.* Engine v1.0 integrated every
  non-null reading. Twenty single-reading microphone glitches (120–138 dBA, every one
  correctly flagged `spike` at ingest) entered the doses, and one 10-s reading at
  135 dBA adds about 3,600% NIOSH dose. The maximum session dose fell from 16,688% to
  3,515% under v1.1, and the "≥115 dBA alarms" fell from 20 to 0, with all 20 now
  listed as excluded candidates. Rows store `engine_version`, so v1.0 and v1.1
  figures stay distinguishable.

Zone summaries (`zone_exposure_summary`) use the same integration, for a notional
person present in the zone for the whole shift.

### Internal consistency check

For a worker who stays in one zone, the personal L_EX,8h must equal that zone's own
L_Aeq, computed directly from `zone_env`, normalised to the same duration.
`test_single_zone_personal_l_ex_equals_zone_l_aeq_normalised` checks this on a real
simulated shift. That shift went through the real ingest path, the real BLE
localisation and the real session builder, and the check used a worker's longest
single-zone stretch (≥ 1 h). **Result: the two agree to within 1×10⁻⁹ dB.** The
three pipelines (zone data, assignments, personal integration) therefore agree with
one another; an error in any one of them would break the equality.

**On the 7-day run data:** the longest single-zone stretch is W105 in Z5,
2026-10-23 05:00:30Z → 11:26:00Z (6.42 h, 2,313 readings). Personal L_EX,8h =
98.440781 dBA; the zone's own L_Aeq normalised to 6.42 h = 98.440781 dBA.
**Residual 0.0 dB.**

---

## 5. Cumulative exposure (`exposure/cumulative.py`)

- A **day** is a shift date: the UTC date of the session start. For this mill
  (06:00/14:00/22:00 IST = 00:30/08:30/16:30 UTC) it equals the local shift date. A
  worker's sessions on one date are combined first.
- **Cumulative dose** is the sum of daily dose percentages, per criterion.
- **Daily L_EX,8h** of several sessions combines on an energy basis:
  10 log10 Σ 10^(L_EX,8h,i/10).
- **Mean daily L_EX,8h** is the energy mean, 10 log10(mean 10^(L/10)). This is the
  form of the ISO 9612 / EU 2003/10/EC weekly exposure level; it is never an
  arithmetic mean of dB.
- **Dust TWA-hours** are TWA × accounted hours (µg·h/m³), indicative.
- **Days without data** are counted separately and excluded, never added as zero.
- **`window_coverage_pct`** is the share of the window's days inside the period the
  system was collecting data. A 30-day window over 7 days of data reports 23.3% and
  `partial = true`. **Totals are never scaled up** to a notional full window.

Every materialised row stores `engine_version` and the full parameter set.
Recomputing a date replaces only that date's rows. There is deliberately no
`computed_at` column, so a recompute is byte-identical.

---

## 6. Stated assumptions

1. **Psychrometric wet-bulb stands in for natural wet-bulb.** T_nw is generally
   higher than T_pwb in still air, so the WBGT reported here **underestimates** heat
   stress: the error leans toward "safer than it is". It is flagged `approximate` on
   every value. A natural wet-bulb sensor would remove this assumption.
2. **κ = 0.05 for the PM humidity correction.** Crilley et al. fitted κ ≈ 0.4 for
   ambient UK aerosol. Cotton dust is fibrous cellulose and far less hygroscopic. A
   larger κ *lowers* the reported dust, so a small κ errs toward over-reporting. The
   RH cap at 95% (the formula diverges as RH → 100%) is also a project assumption.
   Neither value is measured; both are in `config.yaml`.
3. **Moderate-work metabolic category** (ISO 7243 class 2, acclimatised, 28 °C) for
   everyone. Heavier work has a lower reference value, so this assumption may
   understate exceedance for heavy tasks (e.g. blowroom feeding).
4. **Zone-weighted presence is a proxy for personal exposure.** Zone sensors measure
   the zone, not the worker's breathing zone or ear. Localisation errors (~3.6% of
   on-site minutes on simulated data, mostly 30–90 s latency at zone changes) shift
   time between neighbouring zones.
5. **Optical PM is not the regulatory elutriated fraction.** Dust figures are
   indicative; no legal-limit comparison is made.
6. **PPE is not applied.** Recorded PPE use never reduces a computed dose
   (CLAUDE.md), because real-world attenuation is unreliable.
7. **Within an assignment interval, valid readings represent missing ones.** This is
   stated, and visible through `data_coverage_pct`.

---

## 7. Results over the 7-day dataset (engine `exposure-1.1`)

**Dataset.** 2026-10-20 → 2026-10-26 (7 shift dates), shift-aligned run at
`--speed 90`. 1,071,656 messages were published and 1,071,656 stored: zero loss,
zero rejected, zero errors, with ingest at 156 msg/s and an empty queue throughout.
The run produced 140 worker-sessions (20 workers × 7 days), **all unflagged**:
`--align-shifts` gives whole sessions only. BLE localisation over the full week was
96.3% of on-site worker-minutes correct (median latency 30 s, n = 3,304 zone
changes).

**Data quality per result.** Mean data coverage was **99.06%** (minimum 96.47%) and
mean unaccounted time **0.15%** (maximum 0.32%). There were no null results. 135 of
the 140 sessions had at least one anomaly record excluded (3,284 records in total).
No session reached the `low_data_coverage` or `incomplete_presence` thresholds.

**Exposure distribution across the 140 worker-days.**

| | min | p25 | median | p75 | max |
|---|---|---|---|---|---|
| NIOSH dose % | 13.1 | 244.0 | 444.7 | 1,294.6 | 3,515.3 |
| India dose % | 0.2 | 25.7 | 88.9 | 207.7 | 425.1 |
| L_EX,8h (dBA) | 76.2 | 88.8 | 91.5 | 96.1 | 100.4 |

**Worker-days above 100% dose:** NIOSH **126 / 140 (90.0%)**; India **56 / 140
(40.0%)**. The gap between the two criteria is driven by the thresholds and the
exchange rate: India discounts everything below 90 dBA and halves only every 5 dB.
The two Z7 office workers (W108, W115) are the only ones below 100% NIOSH on every
day.

**Per worker** (7 days; energy-mean daily L_EX,8h):

| Worker | Home zone | Shift | Median NIOSH % | Median India % | Mean L_EX,8h | 7-day cumulative NIOSH % |
|---|---|---|---|---|---|---|
| W105 (high-exposure case) | Z5 | A | 3,378 | 417 | 100.2 | 23,272 |
| W106 | Z5 | A | 2,382 | 290 | 98.7 | |
| W118 | Z5 | C | 2,387 | 294 | 98.7 | |
| W112 | Z5 | B | 2,362 | 286 | 98.9 | |
| W104 / W107 / W111 / W117 | Z4 | A/A/B/C | 1,192–1,232 | 194–196 | 95.8–96.0 | |
| W101 / W109 / W116 | Z1 | A/B/C | 448–451 | 91–94 | 91.4–91.5 | 3,156 (W101) |
| W113 / W119 | Z6 | B/C | 290 / 337 | 21 / 28 | 89.9 / 90.4 | |
| W102 / W110 / W120 | Z2 | A/B/C | 237–250 | 27–30 | 88.8–89.1 | |
| W103 / W114 | Z3 | A/B | 167 / 184 | 15 / 17 | 88.0 / 87.8 | |
| W108 / W115 | Z7 | A/B | 28 / 23 | 0.9 / 0.9 | 79.4 / 78.2 | 195 (W108) |

**Zones driving the highest doses** (zone-shift summaries: a notional person present
the whole shift, mean of 21 shifts):

| Zone | L_Aeq (dBA, energy mean) | NIOSH % | India % | PM2.5 TWA raw / corrected (µg/m³, indicative) | WBGT max (°C, approx.) | h above 28 °C |
|---|---|---|---|---|---|---|
| Z5 Weaving | **99.1** | **2,593** | **319** | 85 / 83 | 35.8 | 166.9 |
| Z4 Ring spinning | 95.8 | 1,218 | 202 | 117 / 108 | 36.7 | 168.0 |
| Z1 Blowroom | 91.8 | 485 | 102 | 290 / 290 | 34.6 | 165.0 |
| Z2 Carding | 88.8 | 238 | 25 | **384 / 384** | 35.7 | 167.2 |
| Z6 Winding | 87.6 | 183 | 7.5 | 73 / 73 | 31.5 | 98.9 |
| Z3 Drawing | 85.2 | 105 | 3.3 | 141 / 141 | 34.0 | 149.8 |
| Z7 Office | 56.3 | 0 | 0 | 25 / 25 | 27.3 | 0 |

Weaving (Z5) dominates noise, and its L_Aeq is raised further by the deliberately
degrading loom M5. Carding (Z2) has the highest dust, and Ring spinning (Z4) the
highest WBGT. Only Z4 shows a PM humidity correction, because it is the only zone
whose RH regularly exceeds 75%. WBGT is above the moderate-work reference value for
nearly every production-zone hour. That is plausible for a Tamil Nadu mill at
28–38 °C and 50–85% RH, and it is an **underestimate** (assumption 1).

**Cumulative windows** (as of 2026-10-26). Every 7-day window has
`window_coverage_pct` 100%. Every 30-day window reports **23.33%, `partial = true`**,
with totals equal to the 7-day totals: never scaled up.

---

## 8. Deterministic risk index (`exposure/risk_index.py`)

> **The weights and ramps are a design choice of this project, not a validated
> clinical model.** The index orders exposure situations. It does not estimate the
> probability of any health outcome, and it diagnoses nothing. Its label is
> `deterministic_index`, not `predicted_risk`: nothing is predicted or trained.

| Component (0–100) | From | Normalisation |
|---|---|---|
| Hearing | L_EX,8h: today's value and the 7-day energy-mean daily value (ISO 9612) | ½·today + ½·7-day mean, each mapped linearly **75 dBA → 0, 105 dBA → 100**, clamped at both ends (risk-1.1) |
| Heat | WBGT maximum; hours above the ISO 7243 reference (28 °C, moderate work) | Mean of two ramps: WBGT 26 → 34 °C maps to 0 → 100; 0 → 4 h above maps to 0 → 100 |
| Respiratory | PM2.5 TWA, PM10 TWA, 7-day PM2.5 TWA-hours | **Cohort-relative**: mean percentile rank among that day's workers |

`overall = Σ wᵢ·cᵢ / Σ wᵢ`, with equal weights by default (`config.yaml`). Each
component's absolute and percentage contribution is stored (the percentages sum to
100%), as are the top two drivers in plain language, the weights and the version.

- **The respiratory component ranks workers against each other.** It is relative to
  the cohort, not to a health threshold, because no regulatory anchor exists for
  optical PM. A worker at rank 100 has the dustiest day *of this cohort*, which says
  nothing about whether that level is safe. Of the three components, this is the
  easiest to misread.
- **Confidence** = `data_coverage_pct × (1 − unaccounted_pct/100)`. Below 80%, the
  **band is suppressed** ("insufficient data") but the number is still shown.
- **PPE is displayed alongside and never reduces the index**, nor any dose.
- **Monotonic by construction, and tested.** Raising any single exposure input,
  with the others and the cohort held fixed, never lowers any component or the
  index (`test_index_is_monotonic_in_each_input_with_the_others_fixed`).
- **Hearing uses L_EX,8h, not dose % (risk-1.1).** decibels are already logarithmic
  in acoustic energy, and L_EX,8h is the standards' own unit, so the choice is
  citable rather than an invented transform. Dose % under both criteria is stored
  and shown beside the index everywhere: **the index ranks; the dose is the
  regulated quantity.**

**History: why risk-1.0 was replaced.** risk-1.0 mapped dose % to the hearing
component, with 100% dose → 100 and a cap. That rule was an error in the
specification: it saturated above the limit, so it could not tell a 235% dose from a
3,378% one. On 2026-10-26 it ranked W102 (carding, 235% NIOSH) above W105 (weaving,
3,378%) with both hearing components at 100. The regression test
`test_regression_higher_l_ex_never_scores_below_lower_l_ex` now asserts that, with
other components equal, a materially higher L_EX,8h never scores lower.

**Distribution after the fix (7 days, 140 worker-days, risk-1.1):** High **110**,
Medium **16**, Low **14** (risk-1.0: 119 / 7 / 14). The index runs from 11.0 to 82.2
(median 71.7). The rating stays mostly High, and that is a finding rather than a
calibration fault: this is a loud mill. It is consistent with the Erode power-loom
study cited in the project brief, which found 88.5% of weaving-section workers above
95 dBA *(full reference to be added from the project literature review)*.

**The hearing component now separates the two workers, and the overall order is a
multi-hazard result.** On 2026-10-26, W105's hearing component is **84.2** against
W102's **46.1** (L_EX,8h 100.4 vs 88.7 dBA). W102 still scores higher overall (80.3 vs
73.1), but now for a different reason: W102 works in carding, the dustiest zone
(respiratory rank 94.7 against 35.1), and both workers have heat at the top of its
scale. With equal weights, severe dust plus a 235% noise dose outranks extreme noise
with little dust. The weights were not re-tuned to change this, because that would
be fitting the index to a preferred answer. Anyone who weights hearing more heavily
can change `risk_index.weights` in `config.yaml` (the weight-direction test covers
this).
