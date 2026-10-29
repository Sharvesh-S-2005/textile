# Pre-registration — Day 8: stochastic machine behaviour

**Status: written and committed BEFORE any evaluation on the stochastic dataset.** The
commit that adds this file precedes the commit containing any stochastic-dataset result.
No ML model, feature, split, seed, grid or threshold is changed in this phase; the
simulator is the only variable.

## 1. Why

Day 7 showed that machine state, worker presence and the shift schedule are the same
signal in the deterministic simulator: machines run exactly when the timetable says.
Real looms run at 78–92% efficiency; this simulator stopped them 0% of the time outside
breaks. This phase adds realistic stoppages and asks Objective 6 again.

## 2. What was added (`simulator/machine_stochastics.py`)

1. **Unplanned stops** per machine. Inter-arrival times ~ Exponential(mean gap), counted
   in *running* time; durations ~ Lognormal(median, σ).
2. **Uneven shift starts.** Every machine, every shift: an independent start offset after
   the changeover, plus a chance of a late start or no start at all.
3. **Variable load.** A reflected random walk scales the clamp-meter **current only**.
   Zone noise follows *stops* (as the brief specifies), not load. A load–noise coupling
   would be an invention that pushes the generator toward an informative machine signal,
   so it was deliberately not added.
4. **One correlated dependency.** A blowroom (M1) stop longer than the chute buffer
   starves carding (M2).

Zone noise and dust already follow each machine's physical running state
(`MachineModel.running_now`), so every stop lowers zone noise and every restart raises it
(with the existing current inrush). This was measured (Section 4), not assumed.

The stochastic layer is **off by default** (`machine_stochastics.enabled: false`). Its
random draws use a generator spawned after every existing stream. A 3-hour dry run with
the layer off was byte-identical to the pre-Day-8 simulator (18,777 messages compared).
The deterministic dataset (`data/week/textile_ohs.db`) is kept untouched.

## 3. Parameters and their sources

`literature` = from published mill practice. `project assumption` = a stated assumption of
this project, **not a citation**. Only the weaving availability band is
literature-anchored. Within it, the split into stop rate × stop duration is still an
assumption.

| Type (zone, machines) | Target availability band | Source of band | Mean running gap between stops | Stop duration: median, σ (lognormal) | Source of rate/duration |
|---|---|---|---|---|---|
| Weaving (Z5: M5, M6) — warp/weft break, pick fault | 0.78–0.92 | **literature** (modern loom efficiency) | 25 min | 3.0 min, 0.8 | project assumption |
| Ring spinning (Z4: M3, M7) — doffing (end breaks pieced without stopping) | 0.85–0.95 | project assumption | 150 min | 10.0 min, 0.4 | project assumption |
| Blowroom (Z1: M1) — lap change, choke, cleaning | 0.80–0.93 | project assumption | 90 min | 8.0 min, 0.9 | project assumption |
| Carding (Z2: M2) — flat cleaning, sliver break; plus starvation | 0.80–0.93 | project assumption | **180 min** (see §3.1) | 8.0 min, 0.9 | project assumption |
| Drawing (Z3: M4) — can change, sliver break | 0.85–0.95 | project assumption | 60 min | 4.0 min, 0.7 | project assumption |
| Winding (Z6: M8) — package doff, yarn break, splicer | 0.85–0.95 | project assumption | 45 min | 3.0 min, 0.7 | project assumption |

| Other parameter | Value | Source |
|---|---|---|
| Shift-start offset after changeover | uniform 0–8 min, per machine per shift | project assumption |
| Late start | probability 0.10; extra delay uniform 15–60 min | project assumption |
| No start (order gap, planned maintenance) | probability 0.02 per machine per shift | project assumption |
| Load random walk | bounds 0.70–1.00 of rated current; step sd 0.01 per √min | project assumption |
| Dependency M1 → M2 | starves after a 5-min chute buffer; restarts 2 min after M1 | project assumption |
| Z3 and Z6 given stops at all | not in the Day 8 brief's table; added so no zone stays clock-driven | project assumption |

### 3.1 The one calibration step, disclosed

The first measurement put carding at **0.797** (28-day run) and **0.794** (dataset
window): just below its 0.80 floor, because the starvation loss was underestimated.
Carding's own mean gap was raised **once, 120 → 180 min**, to bring the *measured
availability* into its band. That is calibration to a mill characteristic, as the brief
requires ("set the rate and duration parameters so each machine's realised availability
lands in its target band"). It was done before any evaluation, and no ML number had been
computed on stochastic data. No other parameter was changed after measurement.

## 4. Measured behaviour of the simulator (verification runs)

`python -m simulator.verify_stochastics` (raw output: `docs/preregistration/verification_*.json`) drives the simulator's own `MachineModel`
objects (the construction the MQTT run uses) at the real 10-s machine interval, without
MQTT. Two runs: a general **28-day run (seed 1)**, and the **exact dataset window**
(2026-10-20T00:10Z → 2026-10-27T03:50Z, seed 42). The second is the same machine
trajectory the MQTT run will publish. Availability = running time ÷ deterministic
scheduled time (late and skipped starts count against it).

| Machine | Zone | Type | Band | Availability, 28 days | Availability, dataset window | In band (both) | Stops in 28 days: unplanned / starved |
|---|---|---|---|---|---|---|---|
| M1 | Z1 | blowroom | 0.80–0.93 | 0.852 | 0.829 | yes | 315 / 0 |
| M2 | Z2 | carding | 0.80–0.93 | 0.843 | 0.843 | yes | 178 / 226 |
| M3 | Z4 | ring_spinning | 0.85–0.95 | 0.873 | 0.873 | yes | 207 / 0 |
| M4 | Z3 | drawing | 0.85–0.95 | 0.911 | 0.908 | yes | 504 / 0 |
| M5 | Z5 | weaving | 0.78–0.92 | 0.826 | 0.839 | yes | 1,175 / 0 |
| M6 | Z5 | weaving | 0.78–0.92 | 0.823 | 0.811 | yes | 1,168 / 0 |
| M7 | Z4 | ring_spinning | 0.85–0.95 | 0.909 | 0.923 | yes | 207 / 0 |
| M8 | Z6 | winding | 0.85–0.95 | 0.902 | 0.856 | yes | 694 / 0 |

| Check | 28 days | Dataset window |
|---|---|---|
| Unplanned stop onsets | 4,448 | 1,144 |
| Pearson r, onsets per minute vs "within 10 min of a schedule boundary" | −0.014 | −0.013 |
| Share of onsets near a boundary ÷ share of running minutes near one | 0.92 | 0.93 |
| Blowroom → carding dependency fired | 226 | 65 |
| Z5 noise (energy mean, scheduled time) with 2 / 1 / 0 looms running | 100.9 / 98.5 / 77.2 dBA | 100.8 / 97.7 / 75.3 dBA |
| Z5 stops ≥ 3 min: median drop on stop / median rise on restart | 3.3 / 3.1 dB (n = 999) | 3.1 / 3.1 dB (n = 236) |
| Share of those stops that lowered Z5 noise | 91% | 90% |

So stop onsets are **not** aligned to the shift clock (in the deterministic simulator,
every stop is a schedule boundary: tested), and **an unplanned stop moves zone noise by
about 3 dB**, which is the mechanism this phase depends on. Tests:
`tests/test_machine_stochastics.py`.

## 5. Plan for the evaluation (fixed now)

- **Dataset:** a fresh 7-day run through the full MQTT and ingest path:
  `python -m simulator.publisher --machine-stochastics --speed 90 --start 2026-10-20T06:00:00 --duration-h 168 --seed 42`
  (same window, seed and alignment as the deterministic dataset) into
  `data/week_stochastic/textile_ohs.db`, then the exposure engine with `--resolve`.
- **ML:** identical code and configuration: `ml/features.py`, `ml/forecast.py`,
  `ml/classify_exceedance.py` (with zone groups), `ml/ablation.py` (Model 1/2/3, then
  `--nl-only` for 1-NL/3-NL and 1-NS/3-NS), `ml/anomaly.py`. Same split rule (5 / 1 / 1
  whole UTC days, giving the same dates), same seeds, same grids, same thresholds
  policy, same bootstrap. Outputs go to `data/week_stochastic/`; the deterministic outputs
  are not overwritten.
- `tests/test_results_parity.py` will assert both result sets came from the same model
  configuration.

## 6. Predictions (written before evaluation), with the exact test for each

Deterministic reference values are from the committed Day 6/7 results (commit 0b8a6ce).

| # | Prediction | Deterministic value | **Held if** (stochastic dataset, test day) |
|---|---|---|---|
| 1 | Forecasting skill falls, because the schedule no longer determines everything. | Saved noise model, t+30: **+0.805** | Saved (validation-chosen) noise model's skill at t+30 **< +0.805**. (Secondary, not part of the verdict: t+60 < +0.741.) |
| 2 | The daily "same minute yesterday" baseline falls **further**, since days now differ. | Seasonal noise skill +0.765 (t+30), +0.766 (t+60) | At **both** horizons, the seasonal baseline's skill drop is larger than the saved model's drop: (0.765 − seasonal) > (0.805 − model) at t+30, and (0.766 − seasonal) > (0.741 − model) at t+60. |
| 3 | **(Primary.)** The gain from machine and worker data with no target history and no schedule rises. | Model 3-NS − Model 1-NS, noise: **+0.008** [−0.007, +0.019] (t+30), **+0.002** [−0.008, +0.013] (t+60) | At **both** horizons: point estimate above the deterministic value **and** 95% block-bootstrap CI excluding 0. One horizon only = **partly held**. Neither = **failed**. |
| 4 | Exceedance crossings in the borderline zones (Z2, Z3, Z6) become more frequent and less clock-driven. | `crossing` prevalence, borderline group: 8.3% (t+30) | **(a)** borderline-group `crossing` prevalence at t+30 > 8.3%, **and (b)** among sustained upward 90 dBA crossings in Z2, Z3 and Z6 over all 7 days (below 90 for 5 min, then at or above for 5 min, as in `ml/forecast.py`), the share within ±10 min of a schedule boundary is **lower** than on the deterministic dataset (same function applied to both). Both = held; one = partly held. |

Secondary results, reported but not part of any verdict: the 1-NL → 3-NL gain (deterministic
+0.014 / +0.014), the Day 6 Model 3 − Model 1 gain, pooled and per-group classification, and
anomaly detection.

## 7. Falsification condition

**If prediction 3 does not hold, machine state adds little to zone-hazard forecasting even
under realistic stoppages, and that is the answer to Objective 6 we report.** It will not
be followed by a change of stoppage parameters, features or models in this phase. Every
prediction that fails is reported as failed, with the best available explanation.

---

## Addendum A — 2026-10-29: restart of the generation run (appended; nothing above was edited)

The first generation run was stopped by Claude Code at simulated 2026-10-22T11:00Z
(about 2.4 of 7.15 days) because the machine was low on memory. It lost no data up
to that point, but it cannot be resumed, so it was discarded **unevaluated**. No
model was trained on it and no metric was computed from it. Day 8A measured the
memory before anything was changed (`docs/review_notes.md`, Day 8A): no project
process grows; the pressure came from other applications on the machine.

Changes to how the run is executed. **None changes a parameter, a prediction or the
evaluation plan:**

1. **Speed 60 instead of 90.** The speed only paces the wall clock: every payload is a
   function of simulated time and the seed. The only fields that differ are the
   ingest receipt times, which no feature uses. Section 5 says 90; the run uses 60.
2. **`--discard-session` on the subscriber.** The interrupted run left 4 messages queued
   at the broker for the ingest client's persistent session. Without the purge they
   would arrive first, move the stream watermark 2.4 days ahead, and flag every later
   record `late_upload`. The deterministic dataset has no such flags. Purging restores
   the same clean start the deterministic run had.
3. **Memory caps and a WAL checkpoint** (Day 8A Task 2). The output is byte-identical:
   the 48 h stochastic dry run and its ground truth, and the deterministic dry run, have
   the same SHA-256 before and after these changes.
4. **`OPENBLAS_NUM_THREADS=1` for the publisher.** The simulator uses no BLAS. This only
   avoids about 480 MB of reserved buffers.

The window, seed, start time, duration, outage, stochastic parameters and the
unchunked single run are as pre-registered.
`scripts/run_stochastic_generation.sh` is the exact command.
