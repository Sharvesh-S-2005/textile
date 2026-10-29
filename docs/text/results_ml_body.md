## 1. What is predicted, from what, and why each label is legitimate

| Task | Module | Predicts | From | Label source | Why the label is legitimate |
|---|---|---|---|---|---|
| A. Hazard forecasting | `ml/forecast.py` | Zone noise (dBA), PM2.5 (µg/m³), WBGT (°C) at t+30 and t+60 min | 60 zone features at t: lags, rolling statistics, rate of change, temperature/RH/CO2, machine state, occupancy, time of day, shift, break, zone | The **measured** value at t+h | A future measurement; nothing at t determines it. Legitimate on simulated data because the model can be wrong. |
| B. Exceedance classification | `ml/classify_exceedance.py` | Will zone noise at t+h be ≥ 90 dBA (`above`), or cross from below to above (`crossing`)? | The same 60 features | The **measured** level at t+h vs the criterion | Same as A, made binary. Not circular: unlike a dose-threshold risk label, it is not a formula of the inputs. |
| C. Machine condition monitoring | `ml/anomaly.py` | Is a machine's vibration out of control? | Vibration RMS, peak, crest factor, 1-h rolling means | Unsupervised; scored against the configured degrading machine (M5) | Detectors never see the label; it is used only to score them. |
| Objective 6 ablation | `ml/ablation.py` | Tasks A and B with three nested feature sets | Model 1 ⊂ Model 2 ⊂ Model 3 | As A and B | Reframed from risk prediction (no honest label exists yet) to A and B. |
| Risk index (not ML) | `exposure/risk_index.py` | A 0–100 ordering of worker-days | Exposure results | None: deterministic formula | It is a calculation, labelled `deterministic_index`. **The ML risk classifier is deferred**: training it on labels produced by a formula would only re-learn the formula. |

All ML results use one chronological split, never shuffled: 5 training days, 1
validation day and 1 test day (2026-10-26). Every hyperparameter and threshold was
fixed on the validation day; the test day was scored once.

## 2. Forecasting (Task A, Day 4) — `results_forecast.md`

- **Noise:** random forest skill +0.81 at 30 min, but at 60 min **no ML model beats the
  seasonal baseline** ("same minute yesterday", +0.77). Most noise skill is the
  timetable.
- **Break-free windows** are the honest measure: on rows with no break or shift change
  inside the horizon, RF noise skill is +0.56 / +0.62 against +0.39 / +0.51 for the
  seasonal baseline. That is skill beyond the timetable.
- **Dust** is the clearest ML result: XGBoost +0.38 / +0.43, against at most +0.30 for
  either baseline.
- **Heat** is barely predictable (XGBoost +0.13 / +0.21), and the seasonal baseline is
  worse than persistence.
- SHAP (noise t+30): recent noise 37%, schedule 29%, machine state 21% of mean |SHAP|.

## 3. Exceedance classification (Task B) — `results_classification.md`

**Prevalence first:** 40% of test rows are above 90 dBA (78–88% in Z1/Z4/Z5, 0% in
Z7); 16% of below-criterion rows cross within the horizon.

- `above`: XGBoost PR-AUC **0.97** against **0.67** for the persistence classifier;
  F1 0.89 against 0.76.
- `crossing`: persistence cannot predict a crossing (PR-AUC = prevalence 0.16, recall 0).
  XGBoost reaches PR-AUC **0.83–0.84**, recall 0.78–0.81 at a test-day false-positive
  rate of 0.09–0.11. On **steady** rows this falls to **0.69 / 0.77**. In the borderline
  carding zone (Z2), the models barely beat chance (PR-AUC 0.20–0.23 against a
  prevalence of 0.18–0.19).
- Operating threshold: highest recall subject to FPR ≤ 10%, chosen on the validation
  day. That is a stated policy that favours not missing hazardous exposure.

## 4. The Model 1 / 2 / 3 ablation (Objective 6) — `results_ablation.md`

**Answer to the objective:** on this dataset, machine and zone-history data (Model 3)
improve noise forecasting over environmental data alone by a small but statistically
distinguishable margin (skill +0.017 at 30 min and +0.028 at 60 min; 95%
block-bootstrap intervals exclude zero). Worker-specific data (Model 2) adds nothing
of practical size: +0.007 on noise at 30 min (under 0.1 dB of MAE), and elsewhere
nothing distinguishable. Neither set improves dust or heat forecasting or exceedance
classification in any practical sense.

The machine-data gain is bounded by the simulator-coupling caveat (Section 8).

## 5. Machine condition monitoring (Task C) — `results_anomaly.md`

- M5 (degrading) alarmed on **day 2.08** (EWMA chart) and **day 2.12** (isolation
  forest), about 2% above its baseline. **No healthy machine ever alarmed.**
- Healthy per-bin false-positive rate: **0.02%** (EWMA) against **1.03%** (isolation
  forest). The simple control chart performs at least as well and is the one to deploy.
- M5 → Z5 noise: no resolvable lead or lag in the hourly cross-correlation. The
  vibration alarm came 27.8 h before a noise alarm from the same chart: vibration is
  the more sensitive signal, which is not evidence of causation.

## 6. Deep learning (Task 4)

{{DEEP_LEARNING}}

## 7. Model comparison table (test day 2026-10-26)

One row per model and task, the step-9 comparison of the project's ML methodology.
Baselines are listed in the same table as the models they must beat.

{{COMPARISON_TABLE}}

## 8. Limitations

1. **Simulated data.** Every number here comes from the simulator. No result has been
   checked against a real mill.
2. **One test day.** The block bootstrap in the ablation accounts for autocorrelation
   *within* the day, not for variation *between* days. No other result carries a
   confidence interval.
3. **The seasonal baseline is unrealistically strong.** The simulator repeats an
   identical schedule every day, so "same minute yesterday" is nearly perfect at
   breaks and shift changes. Real breaks drift. The steady-row figures are the better
   guide.
4. **Machine-vibration-to-noise coupling is built into the simulator.** A running
   machine adds a fixed share of zone sound energy, and the degrading machine's share
   grows with its vibration. So the SHAP machine-state share, the Model 3 gain and the
   M5–Z5 analysis show that the pipeline uses this link, not that it holds in a real
   mill.
5. **The risk classifier remains deferred** until questionnaire or audiometry labels
   exist. The risk index is a deterministic, standards-anchored ordering, and its
   weights are a design choice.
6. **PM readings are indicative.** Dust forecasts predict an optical-sensor signal,
   not a gravimetric concentration, and are never compared with legal limits.
