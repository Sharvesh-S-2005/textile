## What the anomaly results say

**Both detectors find the degrading machine early, and neither raises an alarm on a
healthy one.** M5's EWMA alarm comes on day 2.08, two hours after its reference day
ends; the isolation forest follows on day 2.12. At that point M5's vibration is about
2% above its own reference mean, and by the last day it is 21% above (0.508 →
0.613 g). None of the seven healthy machines produces an alarm under either detector.

**The control chart is the better tool here, and that matters for deployment.** Its
per-bin false-positive rate on healthy machines is 0.02% against 1.03% for the
isolation forest, and it alarms slightly earlier. The isolation forest's flags are
held in check only by the three-consecutive-bins rule. An EWMA chart is also what a
maintenance engineer already knows how to read: one statistic, one limit, and a
documented in-control run length. The isolation forest adds no detection power on
this fault. Its advantage would be on faults that change the *shape* of vibration
(crest factor, peak-to-RMS) without moving the mean, which the simulator does not
produce.

**"Earliest possible" detection is limited by the reference day.** Each machine is
calibrated on its own first day. M5 was already wearing during that day (its
reference contains about 3% of drift), which inflates its reference spread and makes
the chart slightly less sensitive for that machine. A real deployment would
calibrate on a machine known to be healthy (after maintenance), not on whatever the
first day happens to be.

**Does M5 vibration precede Z5 noise? Not in a way this data can resolve.** The
hourly cross-correlation of levels peaks at a lag of −6 h (noise leading, r = 0.41).
On hour-to-hour changes it is near zero at every lag (peak r = 0.16), so there is no
resolvable lead or lag. That is consistent with the simulator, where the coupling is
instantaneous. What *is* clear is the difference in sensitivity: Z5 running-hours
noise rose only 0.45 dB over the week (99.43 → 99.88 dBA), and the same EWMA chart
on that noise alarmed on 2026-10-22, **27.8 h after the vibration alarm**. The
vibration channel sees the wear about a day before the zone noise does. That is an
argument for monitoring the machine directly, not evidence that vibration *causes*
the later noise rise.

**Caveat, stated consistently.** The simulator couples machine vibration to zone noise
by construction (a worn machine's share of zone sound energy scales with its
vibration), so this shows that the pipeline detects the link, not that the physical
relationship holds in a real mill. The same caveat applies to the SHAP machine-vibration
finding (`results_forecast.md`) and to the Model 3 gain in the ablation
(`results_ablation.md`).

**Other limits.** The fault is a single, smooth exponential wear curve on one machine
over one week. Real faults include bearing defects that show first in peak and crest
factor, and step changes after maintenance or remounting a sensor. The 10-minute
running-only bins also hide start-up transients, which a real condition monitor
might want to keep.
