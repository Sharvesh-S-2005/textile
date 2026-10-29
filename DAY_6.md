# DAY 6 — ML Depth: Classification, the Ablation Study, and Anomaly Detection

**Read `CLAUDE.md`, `docs/results_forecast.md` and `docs/results_summary.md` first.**
Day 5 must be committed.

**Why this phase:** the panel asked for ML as the priority, and the largest gap
against the project's own objectives is ML — specifically Objective 6, the
Model 1 / 2 / 3 comparison, which is the stated novelty and is currently unbuilt.
This phase also produces the classification metrics the evaluation section promises
(Accuracy, Precision, Recall, F1, ROC-AUC) which the project does not yet have.

**Two reframings that make this honest. Do not deviate from them.**

**Reframing A — the ablation is on hazard forecasting, not risk prediction.** The
project document frames Models 1/2/3 as risk prediction, which cannot be done
honestly without clinical or questionnaire labels. The same research question — does
worker-specific and machine data improve prediction over environmental data alone? —
is fully answerable on hazard forecasting, which is already built. State this
reframing explicitly in the results document and at the review.

**Reframing B — exceedance classification is not circular.** The label is *"did the
measured noise at t+30 exceed 90 dBA?"*, which comes from a **future measurement**,
not from a rule applied to the same features that predict it. This is categorically
different from a dose-threshold risk label, which is circular. Say so in the
docstring and the results doc, because the distinction is the whole justification.

---

## Priority bands

| Band | Tasks | Why |
|---|---|---|
| **MUST** | 1, 2, 5 | Classification metrics, the stated novelty, and a consolidated results document |
| **SHOULD** | 3 | Second ML result; validates the machine-health link |
| **IF TIME** | 4 | Deep learning for completeness; likely to lose to XGBoost |

If you reach roughly two and a half hours, stop and finish Task 5.

---

## Task 1 (MUST) — `ml/classify_exceedance.py`

Binary classification: **will this zone's L_Aeq over the 1-minute bin at t+h exceed
the 90 dBA India criterion?** Horizons 30 and 60 minutes.

Reuse `ml/features.py` unchanged. The only new thing is the label, derived from
measured future noise.

### Guardrails that decide whether this result is worth anything

**Report the base rate first, before any metric.** In Z5 (weaving) almost every bin
is already above 90 dBA, so a classifier that always predicts "yes" would score
above 95% accuracy and mean nothing. Print prevalence per zone per horizon at the top
of every results table. **Accuracy without prevalence beside it is not reportable.**

**Build two label variants and report both:**
- `above`: is the level above 90 dBA at t+h? (dominated by always-loud zones)
- `crossing`: does the level transition from below to above between t and t+h? This
  is the operationally useful event — a warning before a quiet zone becomes
  hazardous — and it is genuinely rare, so it is the harder and more honest task.

**Use a persistence classifier as the baseline.** "If it is above now, it will be
above at t+h." This will be strong on the `above` variant. Report the model's gain
over it; a model that cannot beat it has no result on that variant.

**Precision-recall AUC, not just ROC-AUC.** With imbalanced classes ROC-AUC is
flattering and misleading. Report both, and lead with PR-AUC.

**Chronological split**, identical date ranges to the forecasting work so results are
comparable. Group by zone where relevant; never shuffle.

### Models

Persistence baseline, Logistic Regression (gives the paper a linear baseline),
Random Forest, XGBoost. Tune each on the validation split with a documented search
grid; record the grid and the chosen parameters.

### Metrics, per zone, per horizon, per label variant

Prevalence, Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, confusion matrix, and
the operating threshold chosen. Choose the threshold to **maximise recall subject to
false-alarm rate below a configured ceiling**, not to maximise accuracy — missing a
worker's hazardous exposure is costlier than a false alarm, and that asymmetry should
be an explicit, stated choice in `config.yaml`.

### Outputs

- `docs/results_classification.md` — full tables, the prevalence caveat stated
  up front, the non-circularity argument, the threshold rationale
- `docs/figures/` — ROC curves, PR curves, and confusion matrices as PNG
- `data/classification_predictions.parquet`

`tests/test_classify.py`: labels are derived only from future measurements (assert
no label depends on data at or before t in a way that leaks); prevalence is computed
correctly on a fixture; the persistence classifier reproduces a hand-computed F1 on a
small fixture; threshold selection respects the false-alarm ceiling.

**Run tests. Commit.**

---

## Task 2 (MUST) — `ml/ablation.py` — the Model 1 / 2 / 3 study

This is Objective 6 and the research novelty. It must be done carefully, because a
sloppy ablation is worse than none — it invites the objection that the differences
are noise.

### The three nested feature sets

| Model | Features |
|---|---|
| **Model 1** | Environmental only: noise, PM, temp, RH, CO2, their lags and rolling statistics, and time-of-day |
| **Model 2** | Model 1 + worker-specific: `workers_present_count`, shift ID, `is_break` |
| **Model 3** | Model 2 + machine and history: `machines_running_count`, `mean_current_a`, `max_vib_rms_g`, cumulative exposure features |

Strictly nested — each set contains the previous one. Define them in one place as
explicit column lists so there is no ambiguity about what went into which.

### Fairness requirements

- **Same model class, same split, same random seed** across all three.
- **Tune hyperparameters separately for each feature set.** Reusing Model 3's
  parameters for Model 1 handicaps Model 1 and inflates the apparent gain. This is
  the most common way an ablation gets rigged, usually accidentally.
- Run on **both** tasks: forecasting (MAE, RMSE, skill score) and exceedance
  classification (F1, PR-AUC).

### Significance — do not skip this

With a single test day, metric differences may well be within noise. Compute
**bootstrap confidence intervals** on the *difference* between models: resample the
test set with replacement a few hundred times, recompute the metric difference, and
report the 95% interval. Report per-zone results too, so a gain that exists only in
one zone is visible rather than hidden in an average.

If an interval straddles zero, **say that the improvement is not statistically
distinguishable on this dataset.** That is an honest and publishable finding, and
being the one to point it out is far stronger than having a panel member find it.

### What to expect, so you are not tempted to tune toward a result

Model 3 will probably beat Model 1, mostly because machine state drives zone noise
directly. Worker presence (Model 2) may add little, since it is partly collinear with
machine operation. If Model 2 adds nothing, report that — it is a specific, useful
finding about which data sources actually matter, which is exactly what the objective
asked you to evaluate.

### Outputs

`docs/results_ablation.md` with: the feature-set definitions; a results table per
task, horizon and model; bootstrap intervals on differences; per-zone breakdown; a
grouped bar figure of skill by model; and a prose conclusion that answers the
objective's question directly in one or two sentences.

`tests/test_ablation.py`: feature sets are strictly nested (assert set containment);
each set's tuning ran independently; the bootstrap returns a wider interval on a
smaller resample, as a sanity check.

**Run tests. Commit.**

---

## Task 3 (SHOULD) — `ml/anomaly.py` — machine condition monitoring

Task C from the project document, and it validates the machine-health-to-noise claim
with something measurable rather than an assertion.

**Detector:** per machine, on vibration features (`vib_rms_g`, `vib_peak_g`, their
rolling means and the ratio of peak to RMS). Use **both** an unsupervised
Isolation Forest and a simple EWMA control chart, and compare them. The control chart
is the one a maintenance engineer would actually trust, so if it performs comparably,
say so.

**Ground truth is available:** M5 is the deliberately degrading machine and the
others are healthy. So report:
- **Detection day** — on which simulated day each detector first flags M5
- **False positive rate** on the seven healthy machines
- Whether any healthy machine is ever flagged

**Second hypothesis, with a caveat you must state:** do vibration anomalies on M5
precede increases in Z5 zone noise? Compute the lead-lag cross-correlation and report
the lead time. Then state plainly that **the simulator couples vibration to zone
noise by construction**, so this result demonstrates the detection pipeline works, not
that the physical relationship holds in a real mill. Same caveat already applied to
the SHAP machine-vibration finding — keep the wording consistent between them.

`docs/results_anomaly.md` plus a figure of M5 vibration over 7 days with the
detection point marked and the healthy-machine band shaded.

`tests/test_anomaly.py`: the detector flags a synthetic step change and a synthetic
ramp; it does not flag stationary noise; detection day on the real M5 series is
stable across reruns with a fixed seed.

**Run tests. Commit.**

---

## Task 4 (IF TIME) — `ml/deep_forecast.py` — LSTM / GRU

Only if Tasks 1, 2, 3 and 5 are done. **Hard cap: if it is not training usefully
within 45 minutes, stop and record it as attempted-and-abandoned with the reason.**

Keep it modest: a sequence of the last 60 minutes of the Model 3 feature set, one or
two LSTM or GRU layers, early stopping on the validation split, same metrics and same
split as everything else.

**Expect it to tie with or lose to XGBoost.** On tabular data with well-engineered
lag features, gradient boosting usually wins, and your lags are good. A negative
result here is a legitimate finding — report it as "deep sequence models offered no
advantage over gradient boosting on this dataset, likely because the engineered lag
features already capture the temporal structure." That sentence is more useful to a
reader than a marginal win would be.

Do not spend time on architecture search. Do not let this eat Task 5.

**Commit.**

---

## Task 5 (MUST) — Consolidation, model cards, and two small fixes

### 5a — `docs/results_ml.md`

One document consolidating every ML result, because the panel asked for ML and
should not have to open five files:

- The task taxonomy: what is predicted, from what, with which label source, and why
  each label is legitimate
- Forecasting results (from Day 4), with the break-free-window analysis called out
- Classification results (Task 1), prevalence first
- The ablation study (Task 2) with the objective's question answered in one sentence
- Anomaly detection (Task 3)
- Deep learning (Task 4) or a line stating it was not attempted
- A single **model comparison table** across every model and task — this is step 9 of
  the project document's ML methodology
- **A limitations section**, stating: simulated data; one test day; the seasonal
  baseline is unrealistically strong because the simulator is perfectly periodic;
  machine-vibration-to-noise coupling is built into the simulator; the risk
  classifier remains deferred pending questionnaire labels

### 5b — Model cards

`docs/model_cards.md` — for each deployed model: purpose, inputs, label source,
training window, split dates, hyperparameters and search grid, metrics, known
failure modes, and intended use with an explicit statement of what it must **not**
be used for. Short, one block each. This is cheap and it is exactly the artifact a
reviewer asking about responsible ML wants to see.

### 5c — Two small fixes

**Risk index worst-component rule.** Add the maximum single component alongside the
weighted composite, and show both on the drilldown. This closes the known defect
where a standards-anchored hearing scale is averaged with a cohort-relative
respiratory scale. The framing: *we report both, because the components are not on
commensurable scales.* Roughly fifteen minutes, and it removes the one open
methodological criticism in the risk layer.

**Dashboard Models page.** Add the new figures and the model comparison table. Read
from saved artifacts; run no inference in the dashboard.

Update `docs/review_notes.md` and `docs/numbers_sheet.md`. Regenerate the results
pack. Run the full suite. **Final commit.**

---

## Definition of done

- [ ] `pytest` passes; exit code checked directly, never chained with `;` or piped
- [ ] Classification reports prevalence before every metric table
- [ ] Both `above` and `crossing` label variants reported
- [ ] Persistence classifier baseline reported for both variants
- [ ] PR-AUC reported and led with, not only ROC-AUC
- [ ] Threshold chosen by the recall-subject-to-false-alarm rule, with the choice stated
- [ ] Feature sets strictly nested; hyperparameters tuned per set
- [ ] Bootstrap confidence intervals on model differences; non-significant results
      labelled as such
- [ ] Per-zone breakdown present for the ablation
- [ ] Anomaly detection reports detection day and healthy-machine false positive rate
- [ ] Simulator-coupling caveat stated consistently wherever the machine-noise link
      appears
- [ ] `docs/results_ml.md` and `docs/model_cards.md` written
- [ ] Risk index shows worst-component alongside composite
- [ ] All work committed

---

## Working instructions

Tasks in order. Task 1's leakage and prevalence checks before any metric is quoted.

**Report what you measure.** If Model 2 adds nothing, if the ablation gains are not
significant, if the LSTM loses — those are the results. A phase that produces three
honest negative findings and one solid positive one is a better review than four
flattering numbers nobody can defend.

Do not tune any model after seeing test-set performance. Tune on validation, report
on test, once.