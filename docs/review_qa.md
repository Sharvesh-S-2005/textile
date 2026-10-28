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

**12. What is left?**
Hardware deployment and firmware testing on real boards, questionnaire labels for a
risk model, calibration against a handheld sound-level meter and a reference dust
sampler, and the TimescaleDB migration.
