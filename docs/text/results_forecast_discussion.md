## What the models learned, and what they did not

**The timetable is most of the noise skill, and the seasonal baseline shows it.**
Persistence fails badly for noise: its test MAE is ~8 dB, because at every break
and shift changeover the hall swings between ~60 and ~95–100 dBA. "Same clock minute
yesterday" already gets noise skill **+0.77** at both horizons, because the mill runs
the same schedule every day. Random forest improves on that at 30 min (+0.81). At
60 min **neither random forest (+0.74) nor XGBoost (+0.75) beats the seasonal
baseline (+0.77)**. That is a negative finding, and it is kept: at an hour ahead, the
models add nothing over "the timetable repeats".

**There is genuine skill beyond the timetable.** On *steady* rows, with no scheduled
break or shift boundary inside the horizon, the ML models still reach noise skill
≈ +0.56–0.62, against +0.39–0.51 for the seasonal baseline. Their dust skill is
≈ +0.30–0.39, against +0.07–0.14. So they forecast changes in machine load and dust
that the schedule alone does not explain.

**Dust is the clearest ML result.** XGBoost beats every baseline at both horizons
(skill +0.38 at 30 min, +0.43 at 60 min). The seasonal baseline helps less because
blowdown and cleaning spikes are not tied to the clock.

**Heat is barely predictable from these inputs, and the seasonal baseline is
worse than persistence.** That is a second negative finding: WBGT follows a slow
diurnal curve plus sensor noise, not a daily repeat. XGBoost's +0.13 / +0.21 skill
over persistence is real but small, about 0.1–0.2 °C of MAE.

**Exceedance warnings come mainly from the schedule.** All 36 sustained 90 dBA
crossings on the test day are caught by the ML models, 17–20 minutes ahead (median).
Most of those crossings are machines restarting after a break, so the lead time
largely reflects the known timetable. The models' real advantage over the seasonal
baseline, which also catches all 36, is precision: a false-alarm ratio of 5–14%,
against 36–42% for the baseline. Persistence cannot warn ahead at all under this
definition.

**Limits of this result.** The data are simulated with a rigid, identical schedule
every day; a real mill's breaks drift and its machines stop for unscheduled
reasons. The skill on transitions should be expected to fall on real data. The
steady-state figures are the better guide to what will survive. One week of data
also gives a single test day, so no confidence interval is reported.
