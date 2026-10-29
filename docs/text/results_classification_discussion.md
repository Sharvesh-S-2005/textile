## What the classification results say

**Prevalence first.** On `above`, 40% of test rows are positive overall, but 78–88% in
Z1, Z4 and Z5 and 0% in Z7. The persistence classifier ("loud now, loud later")
already reaches accuracy 0.81 and F1 0.76 on `above`. So any accuracy figure here
reflects the base rate as much as the model.

**`above`: the models beat persistence clearly.** XGBoost reaches PR-AUC 0.97 against
0.67 for persistence, and F1 0.89 against 0.76 (+0.13), at both horizons. The gain
comes almost entirely from rows whose horizon spans a scheduled break or shift
change: there persistence collapses to PR-AUC 0.38–0.58, while the models stay at
0.96–0.97 because they have learned the timetable. On steady rows persistence is
already strong (PR-AUC 0.81–0.82), and the model still improves on it (0.97). Logistic
regression, the linear baseline, is clearly behind the tree models (PR-AUC 0.89–0.90).

**`crossing`: the pooled number is not the result; the zone groups are (Section 2).**
Persistence cannot predict a crossing at all (recall 0, PR-AUC = prevalence, 0.16),
because every row in this population is below 90 dBA now. Pooled over all zones,
XGBoost reaches PR-AUC 0.83–0.84. That figure is mostly zone identity:

1. **Always-loud zones (Z1, Z4, Z5): trivial and useless.** The only quiet minutes
   there are breaks and changeovers, and the machines always restart, so 91–93% of
   them are followed by a crossing. Two-thirds of the pooled positives are these
   restarts. Predicting them is easy, and warning about them adds nothing.
2. **Office (Z7): certain negatives.** It contributes a quarter of the rows and never
   crosses.
3. **Borderline zones (Z2, Z3, Z6): the case that matters, and the result is weak.**
   PR-AUC is 0.18 against a base rate of 0.08, about twice chance, with precision about
   0.2. **In Z2 (carding) it barely beats chance**: PR-AUC 0.22 against 0.18, and the
   pooled operating threshold raises false alarms on 44–53% of Z2's quiet minutes. Z2
   sits at the criterion, and its crossings are fluctuations, not schedule events, so
   these features cannot predict them.

*Correction (Day 7).* An earlier version of this discussion called the steady-row PR-AUC
(0.69 / 0.77) "the best guide to what will survive". That figure still includes the
office's certain negatives and is superseded by the borderline-group figures above.

**Operating threshold.** The recall-subject-to-FPR rule (ceiling 10%, fixed before the
test day) held on the test day *pooled*: FPR was 0.09–0.12 for every model it was
applied to. Per zone it did not: one threshold for all zones lets Z2 run at FPR
0.44–0.53 while the office sits at 0. Zone-specific thresholds, chosen on validation,
are the obvious next step; they were not tried, to avoid any tuning after the test day
had been seen.

**Non-circularity holds, and the leakage tests enforce it.** The label is looked up at
(zone, t + h) and checked against the forecasting target column at runtime; a
mismatch stops the run. `tests/test_classify.py` shows that changing any value at or
before t leaves the `above` label unchanged, while changing the value at t + h flips it.

**Limits.** The data are simulated, there is one test day, and the schedule is
perfectly regular. The same caveat as the forecasting results applies: skill on
transitions should be expected to fall on real data.
