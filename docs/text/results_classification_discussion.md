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

**`crossing`: the honest, harder task.** Persistence cannot predict a crossing at all
(recall 0, PR-AUC = prevalence, 0.16), because every row in this population is below
90 dBA now. XGBoost reaches PR-AUC 0.83–0.84, recall 0.78–0.81 at a test-day FPR of
0.09–0.11, with precision 0.59–0.61: roughly two correct warnings for every false
one. Two qualifications decide how far this should be trusted:

1. **Much of it is the timetable again.** In Z4 and Z5 the only minutes below 90 dBA
   are breaks and shift changeovers, and the machines always restart, so the
   crossing prevalence there is 99–100%. On rows whose horizon spans a scheduled
   transition the model reaches PR-AUC 0.87–0.93. On **steady** rows (no break or
   shift boundary in the horizon), the PR-AUC is **0.69 at 30 min and 0.77 at 60 min**
   (prevalence 12–14%). Those steady-row figures are the best guide to what will
   survive on a real mill, where breaks drift.
2. **Where the level hovers near the criterion, there is almost no skill.** Z2
   (carding) sits around 90 dBA and crosses it at random. There XGBoost's PR-AUC is
   0.20–0.23 against a prevalence of 0.18–0.19 (per-zone appendix). That is barely
   above chance. A crossing that is a random fluctuation is not predictable from
   these features, and the report says so rather than averaging it away.

**Operating threshold.** The recall-subject-to-FPR rule (ceiling 10%, fixed before the
test day) held on the test day: FPR was 0.09–0.11 for every model it was applied to.
The slight overshoots (0.106–0.122) come from applying a validation-day threshold to a
different day, which is expected.

**Non-circularity holds, and the leakage tests enforce it.** The label is looked up at
(zone, t + h) and checked against the forecasting target column at runtime; a
mismatch stops the run. `tests/test_classify.py` shows that changing any value at or
before t leaves the `above` label unchanged, while changing the value at t + h flips it.

**Limits.** The data are simulated, there is one test day, and the schedule is
perfectly regular. The same caveat as the forecasting results applies: skill on
transitions should be expected to fall on real data.
