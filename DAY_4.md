
# DAY 4 — Forecasting, Explainability, and the Risk Index

**Read `CLAUDE.md`, `docs/exposure_methods.md` and `docs/review_notes.md` first.**
Day 3 must be fully committed, including Task 7, before starting.

**Goal of today:** one legitimate ML result with a baseline it beats, an
explainability layer on that model, and a transparent risk index that needs no
training. Everything written to disk as metrics tables and figures the dashboard and
the slides can both consume.

**Two scope decisions carried in from the plan — do not reverse them:**

**FastAPI is cut.** The dashboard reads SQLite through `db/repo.py` directly. Do not
build an API tier.

**The ML risk classifier is cut.** Training a classifier on labels derived from dose
thresholds only re-learns the threshold rule and produces a meaningless accuracy
figure. Risk is instead a deterministic, transparent index (Task 4). SHAP is applied
to the **forecasting** model, where it explains something real.

---

## Task 1 — `ml/features.py`

Build the modelling table from the stored tables only, through `db/repo.py`. No
recomputation of exposure here, and no access to simulator code or ground truth.

**Target granularity:** one row per zone per 1-minute bin. Resample the 10-second
telemetry to 1-minute means (energy-averaged for noise — import from
`exposure/noise.py`, do not reimplement).

**Features per row:**
- Lags of the target: t−1, t−2, t−3, t−5, t−10, t−15, t−30 minutes
- Rolling mean and standard deviation over 5, 15 and 30 minutes
- Rate of change over the last 5 minutes
- `machines_running_count` and `mean_current_a` for that zone
- `max_vib_rms_g` in the zone (couples machine health to noise)
- Minute of day as sine and cosine pair, shift ID, `is_break` flag
- `temp_c`, `rh_pct`, `co2_ppm` and their 15-minute rolling means
- `workers_present_count` from `zone_assignments`
- `zone_id` as a categorical

**Targets:** `noise_dba`, `pm25_ugm3` and `wbgt_c` at t+30 and t+60 minutes.
Build them as separate target columns on the same feature table.

**Rules that protect the result:**
- **No feature may use data from after the prediction timestamp.** Write an explicit
  leakage check: for a sample of rows, assert every feature's source timestamp is
  ≤ t. Run it as a test, not as a comment.
- Rows whose target is missing are dropped. Rows whose *features* are partly missing
  keep an explicit `feature_coverage_pct` column and are retained.
- Flagged and NULL readings are excluded from feature computation, not zero-filled.
- Persist the table to `data/features.parquet` (or CSV) so later tasks and the
  dashboard do not rebuild it.

`tests/test_features.py`: the leakage check; lag features match hand-computed values
on a small fixture; a zone with a data gap produces reduced coverage rather than
fabricated lags; the feature table row count matches expectation for a known range.

**Run tests. Commit.**

---

## Task 2 — `ml/forecast.py`

### Split

**Chronological, never shuffled.** Train on the first 5 days, validate on day 6,
test on day 7. Print the exact date ranges and row counts for each split so they can
go straight into the report. If the 7-day run came out shorter, use a 60/20/20
chronological split and say so.

### Models

Four, in this order — each must beat the one before it or the result is reported as
a negative finding:

1. **Persistence baseline.** Predict the value at t+h to equal the value at t. This
   is the number to beat and it is often surprisingly strong. **Do not skip it.** A
   forecasting result with no baseline is worthless at review.
2. **Seasonal baseline.** Predict the value at the same clock-minute of the previous
   day.
3. **Random Forest.**
4. **XGBoost.**

LSTM/GRU is **optional and last**. Only attempt it if everything else on Day 4 and
Day 5 is done. A deep model that marginally beats XGBoost adds nothing to the review
and can eat hours.

### Metrics

Per target, per horizon (30 and 60 min), per model: MAE, RMSE, R², and
**skill score against persistence** = `1 − MAE_model / MAE_persistence`. The skill
score is the single most communicative number; put it first in the table.

Also compute the metric the project actually cares about:

**Exceedance lead time.** For each occasion in the test set where a zone's measured
noise crosses the 90 dBA India criterion, how many minutes earlier did the model
predict the crossing? Report median lead time, and the hit rate and false-alarm rate
at the operating threshold. This is the result that connects the model to the
project's purpose — a dose warning is only useful if it arrives before the exposure.

### Outputs

- `models/forecast_{target}_{horizon}.pkl` for the chosen model
- `docs/results_forecast.md`: the full metrics table, the split dates, the
  hyperparameters used, and a short honest paragraph on what the model learned and
  what it did not
- `data/forecast_predictions.parquet`: test-set predictions alongside actuals, for
  the dashboard to plot without rerunning the model

`tests/test_forecast.py`: the split is chronological and non-overlapping; persistence
is computed correctly on a fixture with a known answer; the skill score is 0 when a
model equals persistence and positive when it beats it; a trained model round-trips
through save and load.

**Report the numbers you get.** If XGBoost barely beats persistence at 60 minutes,
that is a finding about predictability, not a failure — say so and keep it.

**Run tests. Commit.**

---

## Task 3 — `ml/explain.py`

SHAP on the chosen forecasting model. Use `shap.TreeExplainer`, which is fast on
tree models — do not use the model-agnostic explainer, it will be far too slow.

Produce and save as PNG into `docs/figures/`:
1. A SHAP summary (beeswarm) plot over the test set for noise at t+30
2. A SHAP bar plot of mean absolute importance for the same
3. Two waterfall plots for individual predictions: one where the model correctly
   predicted a crossing, one where it missed

Then write a short interpretation into `docs/results_forecast.md` — three or four
sentences on which features actually drive the forecast. If machine state and recent
lags dominate, say so plainly; that is a sensible and defensible result and it
supports the machine-health-to-noise link the project claims.

Keep figures legible at slide size: readable axis labels, no default rainbow
colormaps, and check they survive being projected.

**Run tests. Commit.**

---

## Task 4 — `exposure/risk_index.py`

A deterministic, fully transparent risk index. **No training, no labels, no model.**
It lives in `exposure/`, not `ml/`, because it is a calculation, not a prediction.

### Construction

Three components, each normalised to 0–100 from values the exposure engine already
computes:

| Component | Source | Normalisation |
|---|---|---|
| Hearing | Noise dose % (both criteria) and 7-day cumulative dose | 100% dose maps to 100 |
| Heat | WBGT maximum and time above the ISO 7243 limit | Documented piecewise ramp |
| Respiratory | PM2.5 and PM10 TWA, 7-day cumulative TWA-hours | Relative to the cohort distribution, since no regulatory anchor is available |

Combine with **explicit weights in `config.yaml`**, defaulted to equal, and return:

- `overall_index` (0–100) and a band (Low / Medium / High) with thresholds in config
- `contributions`: each component's absolute and percentage contribution to the total
- `drivers`: the top two contributing factors in plain language
- `confidence`: derived from `data_coverage_pct` and `unaccounted_pct`
- `label: "deterministic_index"` — **not** `predicted_risk`, because nothing is
  predicted here

### Four rules

1. **The weights are a design choice of this project, not a validated clinical
   model.** Say so in the docstring, in `exposure_methods.md`, and in the dashboard
   where the number appears.
2. **PPE status is displayed alongside but never reduces the index**, consistent with
   `CLAUDE.md`.
3. **The respiratory component is cohort-relative**, so it ranks workers against each
   other rather than against a health threshold. State that explicitly — it is the
   component most open to misreading.
4. **Low confidence must suppress the band, not the number.** A worker with 45%
   coverage shows the index with an explicit "insufficient data" state rather than a
   confident Medium.

Write results to a `worker_risk_index` table through `db/repo.py`, one row per worker
per day, storing the weights and version used.

`tests/test_risk_index.py`: contributions sum to 100% of the overall index; a worker
at exactly 100% dose with nothing else elevated produces the expected hearing-driven
value by hand calculation; changing one weight in config changes the output in the
expected direction; low coverage suppresses the band; the index is monotonic in each
component with the others held fixed.

That monotonicity test matters — a risk index that can *fall* when an exposure rises
is indefensible, and it is an easy bug to introduce while normalising.

**Run tests. Commit.**

---

## Task 5 — Results pack

Everything the dashboard and the slides need, written to disk once.

Write `scripts/build_results_pack.py` producing:

- `docs/results_forecast.md` — completed, with metrics, split dates, lead-time
  results and the SHAP interpretation
- `docs/results_exposure.md` — exposure distribution across workers over 7 days;
  worker-days above 100% under each criterion; mean data coverage; mean unaccounted
  time; the zones driving highest dose; the risk-index distribution
- `docs/figures/` as PNG, each legible at slide size:
  1. Noise time series for one zone over one shift, with machine state shaded and
     break periods marked
  2. Forecast predicted versus actual for the test day, with the persistence
     baseline on the same axes
  3. SHAP summary and SHAP bar
  4. Per-worker dose bar chart, sorted, with the 100% line marked under both criteria
  5. Zone heat map of mean noise by zone and hour
  6. Localisation confusion matrix from Day 2
- `docs/results_summary.md` — a one-page table of **every headline number** the
  project has produced: test counts, data-loss reconciliation, ingest throughput,
  localisation accuracy and latency, consistency-test residual, forecast skill
  scores, exceedance lead time, and exposure distribution

Make the script idempotent and rerunnable, because numbers will change if anything
upstream is recomputed.

**Chart conventions** — keep one colour per hazard across every figure (noise, dust,
heat, CO2), never encode meaning in colour alone, label units on every axis, and
check each figure is readable when projected.

Update `docs/review_notes.md`. Run the full suite. **Final commit.**

---

## Definition of done for Day 4

- [ ] `pytest` passes; exit code checked directly, not through a pipe
- [ ] Leakage check passes as a test
- [ ] Persistence and seasonal baselines computed and reported
- [ ] Skill scores reported per target and horizon
- [ ] Exceedance lead time, hit rate and false-alarm rate reported
- [ ] SHAP figures saved and interpreted in prose
- [ ] Risk index contributions sum correctly; monotonicity test passes
- [ ] Risk index labelled `deterministic_index`, with the weights-are-a-design-choice
      caveat in code, docs and UI text
- [ ] `docs/results_summary.md` contains every headline number on one page
- [ ] All figures legible at slide size
- [ ] All work committed

---

## Working instructions

Tasks in order. Task 1's leakage check before any model is trained — a leaked
feature produces a beautiful result that collapses the moment someone asks how it
was validated.

Report the real numbers. If a model fails to beat its baseline, that is the result;
write it down and move on. A negative finding that is honestly reported survives
review. A flattering number that cannot be explained does not.

Do not start the dashboard today. Day 5 covers it, and starting it early is how both
days end up half-finished.