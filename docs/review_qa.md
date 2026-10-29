# Review Q&A — first review

Short answers for the panel. Numbers come from `docs/numbers_sheet.md`, which is
generated from the results pack; if a number here disagrees with the sheet, the sheet wins.

**1. Is this real data?**
No — it is simulated, and every dashboard page says so. The zone profiles in
`simulator/profiles.py` follow the published ranges for each mill section (noise,
PM2.5, heat, humidity), with machine coupling, breaks, sensor faults and an outage added.
The simulator publishes contract v1.1 over MQTT, so the ESP32 nodes replace it without
any backend change.

**2. How do you know the exposure maths is right?**
Twelve noise cases are hand-calculated in the tests, and four rows of Schedule XXVIII
(Rule 95, Tamil Nadu Factories Rules) are reproduced exactly. For a worker who stays in
one zone, the personal L_EX,8h equals the zone L_Aeq with a residual of 0.0 dB. Levels
are always energy-averaged, never arithmetically.

**3. Why two noise criteria?**
NIOSH 98-126 (85 dBA, 3 dB exchange rate, 80 dBA threshold) and the Tamil Nadu rules
(90 dBA, 5 dB exchange rate, 90 dBA threshold) differ in all three parameters, so the
doses differ a lot: 90% of worker-days exceed 100% dose under NIOSH and 40% under India.
We show both and do not choose between them.

**4. Didn't your model just learn the break schedule?**
Partly, and we measured how much. On break-free windows the random forest still has
skill +0.56 / +0.62 over persistence (t+30 / t+60), against +0.39 / +0.51 for the
seasonal baseline. So it has skill beyond the timetable.

**5. Why does the seasonal baseline beat ML at 60 minutes?**
The simulator repeats an identical daily timetable, which is the best case for a
"same time yesterday" baseline. A real mill varies from day to day, so this result
understates the ML advantage rather than overstating it. We report it as a loss, not
hide it.

**6. How accurate is the worker tracking?**
96.4% of on-site worker-minutes are in the correct zone, with a median zone-change
latency of 30 s. Every error is between adjacent zones, and about 95% of the wrong
windows are transition lag. Real-world accuracy is untested until the hardware arrives.

**7. Why not the better-scoring localisation settings?**
`debounce_windows=1` scores 98.6% but produces 223 zone changes against about 181 real
ones — roughly 40 false zone changes that split a worker's exposure across zones.
Choosing the best row would also tune on the scored data. Retuning waits for
calibration walks on real hardware.

**8. Is the risk score medically validated?**
No. It is a transparent weighted index (noise, dust, heat) whose weights are a design
choice and are visible in the dashboard. It estimates exposure-related risk and
diagnoses nothing. An ML risk classifier is deferred until questionnaire-based labels
exist, because training on simulator-generated labels would only re-learn the generator.

**9. Why drop the heart-rate and respiratory sensors?**
Lead with the feasibility table from the project report: wearability over a full
shift, cost per worker, and the fact that the system estimates exposure, not
physiology. Personal exposure is estimated from zone sensors and time in zone
(`estimated_personal`); a wearable stays an optional prototype (`measured_personal`).

**10. What happens when the network fails?**
The node buffers to microSD and replays with the original timestamps; the backend
deduplicates and never assumes arrival order. The demo shows it: Z3 goes offline 90
simulated minutes in for 5 minutes, reconnects announcing `buffered_records: 40`, and
the replayed records land in their correct time slots with zero loss.

**11. Can the dust readings be compared to legal limits?**
No, and no function in the codebase does so. Regulatory cotton dust is the
elutriated fraction measured gravimetrically; low-cost optical PM2.5/PM10 is indicative
only. We use it for trends, zone comparison and as an ML feature, after humidity
correction.

**12. "Your ablation gains are tiny."**
Correct, and we investigated why. With the noise sensor's own history in the model,
machine and worker data add +0.017 / +0.028 skill. We then removed every column derived
from any hazard sensor and repeated the comparison: the gain is still only +0.014
(significant, about 0.1 dB), and with the timetable also removed it disappears. So the
information is redundant with the **timetable**: in the simulator, machines run exactly
when the schedule says. A model with **no noise input at all** (temperature, RH, CO2,
time of day, shift, break, zone) forecasts zone noise with skill +0.76 / +0.77, as well
as the full model. That is the practical result: **sensor-failure resilience.** When a
noise or PM sensor clogs, drifts or fails, as optical sensors do in lint-heavy air,
zone hazard forecasting continues without it. On a real mill with unscheduled stops,
machine state should matter more; that is a hypothesis for the hardware phase. Show
`docs/figures/ablation_redundancy.png`.

**13. "Why does classification do worse in carding?"**
Because Z2 hovers at the 90 dBA threshold, its crossings are driven by small
fluctuations rather than by the schedule, and fluctuations are not predictable from
these features. PR-AUC there is 0.22 against a base rate of 0.18: it barely beats
chance, and we lead with that. It is the hard case and also the one that matters most.
In the always-loud zones the same task is trivial (machines restart after every break)
and a warning adds nothing, which is why the pooled PR-AUC of 0.83 is not the result we
quote. Zone-specific thresholds and features that see fluctuations coming are the next
step.

**14. "Did you change the simulator until the result came out right?"**
No. The stoppage parameters and four predictions, including the falsification condition
for the main one, were written in `docs/preregistration_day8.md` and committed (commit
2adad2b) **before** any evaluation ran on the new data. The git history shows the order.
The parameters come from mill characteristics. Only the weaving band (78–92% loom
efficiency) is from the literature; the rest are labelled project assumptions. The one
calibration step (carding's stop rate, to bring its *measured availability* into its
band) is disclosed, and it happened before any model was trained. Both datasets are
reported side by side, and a failed prediction is reported as failed.

**15. "Why was the first simulator wrong?"**
It was unrealistically clean: machines ran exactly on the timetable, with 0% unplanned
downtime, where real looms stop 8–22% of the time. That made machine state, worker
presence and the clock the same signal, so Objective 6 could not be answered. We found
this by measuring it (the Day 7 redundancy analysis: a model with no noise input at all
forecast as well as the full model), not by assuming it.

**16. What is left?**
Hardware deployment and firmware testing on real boards, questionnaire labels for a
risk model, calibration against a handheld sound-level meter and a reference dust
sampler, and the TimescaleDB migration.
