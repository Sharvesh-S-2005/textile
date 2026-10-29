## Conclusion: the answer to Objective 6 (revised on Day 7)

**With dense target history available, machine and worker features add little: +0.017
to +0.028 noise skill. The Day 7 redundancy analysis tested whether that was because
their information is redundant with the target's own autocorrelation, and the answer
is: only partly. With every target-derived column removed, the same features still
add only +0.014 noise skill at both horizons (95% CIs exclude 0), and nothing
distinguishable for dust. With the timetable also removed, they add nothing
distinguishable at all (+0.001 to +0.008).**

**The information is redundant with the timetable.** In this simulator, machines run
exactly when the shift schedule says. A context-only model (temperature, RH, CO2, time
of day, shift, break and zone, with **no noise input at all**) reaches noise skill
**+0.76 / +0.77**. That is as high as the Day 6 model with dense noise history
(+0.74 / +0.76; the two were not compared with a paired interval). Machine state at t
describes what is happening *now*. Over a 30–60-minute horizon, what changes zone noise
is the next break or shift change, and the timetable already knows that.

**So the objective's question has a negative answer on this dataset:** worker and
machine data do not improve 30–60-minute hazard forecasting beyond what the
environmental context and the timetable already provide. The improvement that exists
is statistically detectable but small: about 0.1 dB of MAE.

**The practical implication is sensor-failure resilience, carried mainly by the
timetable.** When a noise or PM sensor clogs, drifts or fails (a documented failure
mode for optical sensors in lint-heavy air), a model with no hazard-sensor input
continues to forecast zone hazard levels: noise skill +0.76 / +0.77 and dust
+0.38 / +0.42, close to the full models. Machine state adds a small, significant
+0.014 on top for noise. In this simulator, the timetable, not machine state, carries
the fallback signal. On a real mill, machines also stop for unscheduled reasons, and
then machine state would carry information the timetable cannot. That is a hypothesis
for the hardware phase, not a result here. The direct test, estimating *current* zone
noise from machine state alone (a nowcast), was not run.

### What the Day 6 numbers show

- **Noise forecasting: Model 3 > Model 1, significant at both horizons.** Skill rises
  +0.017 at 30 min (95% CI +0.011 to +0.025) and +0.028 at 60 min (+0.020 to +0.037).
  That is MAE 2.07 → 1.94 dB and 1.96 → 1.73 dB. The gain is modest, and it is real
  on the test day. Per zone, the 60-minute gain is distinguishable in six of seven
  zones; at 30 minutes, in four. So it is not one zone carrying the average.
- **Model 2 (workers present, shift, break) adds nothing measurable.** Its difference
  from Model 1 is distinguishable only for noise at 30 min (+0.007, CI +0.002 to
  +0.013), which is under 0.1 dB of MAE. Everywhere else the interval straddles zero,
  except a negligible *drop* of 0.002 PR-AUC on `above` at 60 min.
  This was expected: worker presence is largely collinear with the schedule and with
  machine operation, and time of day is already in Model 1. It is a specific finding
  about which data sources matter: **zone occupancy from BLE is valuable for exposure
  attribution, not for hazard prediction.**
- **Dust and heat: no distinguishable difference between any pair of sets.** Every
  interval contains zero (dust at 30 min even has Model 3 slightly lower,
  −0.015, CI −0.041 to +0.011). Machine state does not drive these zone signals in
  the data, so adding it cannot help.
- **Exceedance classification: no practical difference.** PR-AUC is 0.968–0.972
  (`above`) and 0.834–0.839 (`crossing`) for all three sets. The only intervals that
  exclude zero are on `above` at 60 min (Model 3 − Model 1 = +0.002; Model 2 − Model 1
  = −0.002). Those are statistically detectable on ~9,600 correlated rows but far too
  small to matter. Environmental lags plus time of day already carry what the
  classifier needs.

### Reading the figure

The 95% bars on the skill *levels* overlap almost completely, yet the differences
are significant. This is not a contradiction. The level intervals are wide because
the test day contains easy and hard hours; both models move up and down together
across them. The difference is computed on the same resampled rows each time
(paired), so that shared variation cancels. The table of differences, not the
overlap of bars, is the test.

### Caveats that bound the claim

1. **The Model 3 noise gain comes mostly from machine state, and the simulator couples
   machine state to zone noise by construction.** A running machine adds a fixed
   share of zone sound energy, and the degrading machine's share grows with its
   vibration. So this result shows that the pipeline uses machine data when it is
   informative; it does not show that the physical relationship holds in a real
   mill. The same caveat applies to the SHAP finding (`results_forecast.md`) and to
   the M5–Z5 lead-lag analysis (`results_anomaly.md`).
2. **One test day, simulated.** The block bootstrap accounts for autocorrelation
   within the day, not for day-to-day variation, which one test day cannot
   measure. A second week of data would be the cheapest way to strengthen this.
3. **Fairness was enforced, not assumed.** The same grid was searched separately per
   set, and the chosen parameters differ between sets (e.g. noise t+30:
   Model 1 lr 0.1 with 335 trees, Model 3 lr 0.05 with 379 trees). Nothing was tuned
   after the test day was seen.
