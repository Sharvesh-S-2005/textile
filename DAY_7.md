# DAY 7 — Redundancy Analysis, Rehearsal, and Presentation Assembly

Final phase before the review. Three tasks, roughly four hours with buffer.

**Read `docs/results_ml.md` and `docs/results_ablation.md` first.**

---

## Task 1 (MUST) — Ablation redundancy analysis

**Why:** the Model 1 → Model 3 gain of +0.017 / +0.028 skill is almost certainly an
artifact of feature design rather than a finding about machine data. Model 1 already
contains noise lags at t, t−1 … t−30 and rolling statistics. When machines stop,
recent noise is already low, so `machines_running_count` is close to a deterministic
function of information Model 1 already holds. The experiment measured redundancy,
not informativeness.

This task measures the thing the objective actually asked about.

### Two new feature sets, no target lags

| Set | Features |
|---|---|
| **Model 1-NL** | Environmental context only: `temp_c`, `rh_pct`, `co2_ppm`, their rolling means, time-of-day sine/cosine, shift ID, `is_break`. **No lags or rolling statistics of the forecast target.** |
| **Model 3-NL** | Model 1-NL plus `machines_running_count`, `mean_current_a`, `max_vib_rms_g`, `workers_present_count` |

Strictly nested, same as before. Same model class, same split, same seed, and
**hyperparameters tuned independently for each set.**

Run on noise forecasting at 30 and 60 minutes. Add dust if it is quick; skip heat.

Report skill score, MAE, and bootstrap confidence intervals on the Model 1-NL →
Model 3-NL difference, exactly as in Day 6.

### Expected result and how to frame it

The gain should be large — machine state is the physical driver of zone noise, and
with the target's autocorrelation removed there is nothing else carrying that
information. If it is not large, that is itself a finding worth reporting.

Write the conclusion in `docs/results_ablation.md` as a two-condition statement:

> With dense target history available, machine and worker features add little
> (+0.017 to +0.028 skill), because their information is largely redundant with the
> target's own autocorrelation. With target history removed, the same features carry
> [X] of the predictive signal. The practical implication is sensor-failure
> resilience: when a noise or PM sensor clogs, drifts or fails — a documented failure
> mode for optical sensors in lint-heavy environments — a machine-state model
> continues to forecast zone hazard levels.

That last sentence connects the ablation to the project's sensor-health layer and to
the feasibility analysis. Say it at the review.

Add one figure: grouped bars of skill for Model 1, Model 3, Model 1-NL, Model 3-NL,
so the redundancy effect is visible in a single glance.

`tests/test_ablation.py`: assert the NL sets contain no column derived from the
forecast target; assert nesting.

**Run tests. Commit.**

---

## Task 2 (MUST) — Promote two buried findings

No new modelling. Restructure how two existing results are reported.

**2a — Per-zone classification, not averaged.** The `crossing` label is roughly 100%
positive in Z4 and Z5, because the only sub-90 dBA minutes there are breaks and the
machines always restart. So the task is trivially predictable precisely where a
warning adds no value, and genuinely hard where it does.

In `docs/results_classification.md` and on the dashboard Models page, restructure to:

| Zone group | Character | Result |
|---|---|---|
| Z4, Z5 (always loud) | `crossing` ~100% positive; prediction trivial | state the prevalence, note the metric is uninformative here |
| Z2, Z3, Z6 (borderline, hover near 90 dBA) | the operationally meaningful case | report the real numbers, including Z2 barely beating chance |
| Z7 (office) | never exceeds | excluded |

**Lead with Z2.** "Our classifier barely beats chance in carding, which is the zone
where a warning would matter most" is a far stronger thing to say than an average
that hides it. Then give the reason: a zone hovering at the threshold has crossings
driven by small fluctuations rather than by schedule, which is exactly the
unpredictable case.

**2b — The EWMA-over-Isolation-Forest conclusion.** Promote it in
`docs/results_ml.md` from a line to a short paragraph: both methods detect the
degrading machine within a day of each other (day 2.08 versus 2.12), neither
false-alarms on a healthy machine, but the EWMA control chart produces a 50× lower
false positive rate (0.02% versus 1.03%) and is the method a maintenance engineer
would already recognise. The simpler method was chosen on evidence.

Regenerate the results pack and the numbers sheet. **Commit.**

---

## Task 3 (MUST) — Offline rehearsal and final state

**3a — The offline rehearsal, still open from Day 5.** This needs a human and cannot
be done from the container.

1. Disconnect Wi-Fi or unplug the cable — not merely stopping the broker
2. `streamlit run dashboard/app.py`
3. Click every page and every sidebar selection
4. Confirm the W105 drilldown, the Live Zones chart, and every Models-page figure render
5. Reconnect

The failure being hunted is a page reaching for something remote — a CDN font, a
Plotly asset, a map tile. It renders perfectly with network and breaks without, and
the projector is a poor place to discover which.

Record the outcome in `docs/review_notes.md` either way.

**3b — Final housekeeping.**
- Commit `DAY_6.md` and `DAY_7.md` alongside the earlier briefs
- Confirm `docs/numbers_sheet.md` regenerates and carries the Day 6 and Day 7 figures
- Confirm `docs/review_qa.md` Q7 and Q9 point at `docs/feasibility_table.md`
- Add two Q&A entries:
  - *"Your ablation gains are tiny."* — Correct, and we investigated why: the
    features are redundant with target autocorrelation. Here is the no-lag analysis
    and what it implies for sensor-failure resilience.
  - *"Why does classification do worse in carding?"* — Because Z2 hovers at the
    threshold, so crossings are fluctuation-driven rather than schedule-driven. It is
    the hard case and also the one that matters most.
- Run the full suite, exit code checked directly. **Final commit.**

---

## Definition of done

- [ ] `pytest` passes; exit code checked directly
- [ ] Model 1-NL and Model 3-NL results with bootstrap intervals
- [ ] No-lag sets verified by test to contain no target-derived column
- [ ] Two-condition ablation conclusion written, including the sensor-failure
      resilience implication
- [ ] Classification results restructured per zone group, leading with Z2
- [ ] EWMA-versus-Isolation-Forest conclusion promoted to a paragraph
- [ ] **Offline rehearsal performed by a human and the outcome recorded**
- [ ] Day briefs committed; numbers sheet and Q&A current
- [ ] All work committed

---

## Working instructions

Task 1 first — it changes the headline ablation conclusion, which Tasks 2 and 3 then
report.

Do not retune anything on the test set. Tune on validation, report on test, once.

If the no-lag gain comes out small, report it as it is and say that the objective's
question has a negative answer on this dataset. That conclusion is defensible. An
inflated one is not.