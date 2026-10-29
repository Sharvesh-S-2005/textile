# DAY 8 — Breaking the Collinearity: Stochastic Machine Behaviour

**Read `docs/results_ablation.md`, `docs/results_classification.md` and
`CLAUDE.md` first.** Day 7 must be committed.

**Why this phase:** Day 7 established that machine state, worker presence and the
shift schedule are collinear **by construction** — in the simulator, machines run
exactly when the schedule says. That single property explains every weak ML result:
Objective 6 cannot be answered, the daily baseline is unrealistically strong, and
exceedance classification is trivial in the loud zones.

Real mills do not behave this way. Modern looms run at **78–92% efficiency**, meaning
machines are stopped 8–22% of scheduled time by unplanned events. Our simulator stops
them 0% of the time outside breaks.

Fixing that makes the research question answerable. It is software-only, and it
reuses every piece of infrastructure already built.

---

## The integrity rule that governs this entire phase

There is an obvious way to cheat here: adjust the stochasticity until the ablation
shows a gain. That would be fitting the data generator to the desired conclusion, and
it would invalidate every number in the paper.

**So this phase is pre-registered.** Task 1 writes the parameters and the predicted
outcomes to a document and **commits it before any evaluation runs**. Task 2 then
reports whatever comes out.

Three hard rules:

1. **Parameters are chosen from mill characteristics, never from ablation results.**
2. **The pre-registration is committed before the evaluation is run.** The commit
   order is the evidence; do not reorder it.
3. **The deterministic dataset is kept.** Both are reported side by side. The
   comparison *is* the finding.

If the ablation gain stays small even with realistic stoppages, report that. It would
mean machine state genuinely adds little to zone-hazard forecasting, which is also a
real answer to Objective 6.

---

## Task 1 — Stochastic machine behaviour

Extend `simulator/machine_model.py`. Everything configurable goes in `config.yaml`
under a new `machine_stochastics` block.

### Behaviours to add

**1. Unplanned stops.** Each machine stops at random intervals for a random
duration. Draw inter-arrival times from an exponential distribution and durations
from a lognormal, per machine type:

| Machine type | Cause | Target availability |
|---|---|---|
| Weaving (Z5) | warp break, weft break, pick fault | 0.78–0.92, from the loom efficiency range |
| Ring spinning (Z4) | end breakage, doffing | state as a project assumption |
| Blowroom / carding (Z1, Z2) | lap change, choke, cleaning | state as a project assumption |

Set the rate and duration parameters so each machine's **realised availability**
lands in its target band, then verify that by measurement rather than assuming it.

**Only the weaving band is literature-anchored.** The spinning and preparation bands
are stated assumptions of this project — label them that way in the config comments
and in the pre-registration. Do not present an assumption as a citation.

**2. Staggered, partial shift starts.** Machines currently all start together. Give
each an independent random offset of a few minutes, and let a configurable fraction
start late or not at all on a given shift (an order gap, planned maintenance).

**3. Variable load.** Machine current should vary slowly over a shift — different
yarn counts, different fabric — rather than sitting at a fixed fraction of rated
current. A slow random walk within bounds is sufficient.

**4. Correlated stops.** A blowroom choke upstream starves carding downstream. Give
one configurable dependency so stops are not fully independent, which is what makes
`machines_running_count` carry information the clock does not.

### Zone noise must follow

The existing coupling already lowers zone noise when machines stop. Verify it holds
for unplanned stops too, and that noise recovers on restart with the inrush
behaviour already implemented. **This is the mechanism the whole phase depends on:**
if an unplanned stop does not move zone noise, machine data cannot carry
non-schedule information.

### Pre-registration — write and commit before Task 2

Write `docs/preregistration_day8.md` containing:

- every parameter value, with its source marked `literature` or `project assumption`
- the measured realised availability per machine, from a short verification run
- the predicted outcomes, written **before** the evaluation:
  1. Forecasting skill should **fall** from +0.805, because the schedule no longer
     determines everything
  2. The daily "same minute yesterday" baseline should fall **further**, since days
     now differ
  3. The gain from machine data in the no-history, no-schedule condition should
     **rise** above the +0.008 / +0.002 measured on the deterministic data
  4. Exceedance crossings in the borderline zones should become more frequent and
     less clock-driven
- the falsification condition, stated plainly: if prediction 3 does not hold, machine
  state adds little to zone-hazard forecasting even under realistic stoppages, and
  that is the answer we report

`tests/test_machine_stochastics.py`: realised availability falls inside each target
band over a long run; stops are not aligned to the shift clock (assert low
correlation between stop onsets and schedule boundaries); a stop materially lowers
zone noise; the correlated dependency fires; the seeded run is reproducible.

**Commit Task 1 and the pre-registration together, before running Task 2.**

---

## Task 2 — Regenerate and re-evaluate

**Start the data generation first**, then do Task 3 while it runs.

1. Generate a fresh 7-day dataset through the full MQTT and ingest path, into a
   **separate database** so the deterministic dataset survives intact.
2. Re-run the exposure engine, then the full ML suite with no parameter changes:
   features, forecasting with both baselines, exceedance classification per zone
   group, the four-way ablation (Model 1, Model 3, Model 1-NL, Model 3-NL, plus the
   no-schedule condition), bootstrap intervals, and anomaly detection.
3. **Change nothing in the models.** Same splits, same tuning procedure, same seeds.
   The only difference between the two result sets must be the simulator.

### Reporting

Rewrite `docs/results_ablation.md` and `docs/results_classification.md` to present
**both datasets side by side**, with a column for each, and keep the deterministic
numbers rather than replacing them.

Then answer Objective 6 in one paragraph, using whichever result you actually got.
Both of these are publishable:

> Under a deterministic schedule, machine and worker data add +0.008 to +0.002 skill
> and cannot be separated from the clock. With realistic stoppage rates, the same
> features add [X], because machine state now carries information the schedule does
> not. The research question is answerable only where machine operation is stochastic
> — which is the condition in a real mill, and not in a naive simulation.

or

> Even with realistic stoppage rates, machine and worker data add only [X]. Zone
> hazard levels appear to be predictable from environmental history and the schedule
> alone, and the additional data sources proposed in Objective 6 do not materially
> improve forecasting.

Also report what happened to forecasting skill and to the baselines, and say whether
each pre-registered prediction held. **A prediction that failed is reported as
failed.**

`tests/test_results_parity.py`: assert both result sets were produced by the same
model configuration, so the comparison is valid.

**Commit.**

---

## Task 3 — Firmware compile check

Do this while the dataset generates. It closes a stated open item cheaply.

The firmware stub has never been compiled. `arduino-cli` can compile for an ESP32
target **with no board attached**, which verifies the sketch is syntactically valid,
that the library APIs are used correctly, and that the payload-building code builds.

1. Install `arduino-cli`, add the ESP32 core, install the libraries named in
   `firmware/README.md`
2. Compile `firmware/zone_node/zone_node.ino` for an `esp32:esp32:esp32` target
3. Fix whatever breaks — API mismatches in PubSubClient, ArduinoJson or NimBLE are
   the likely candidates
4. Record the exact commands in `firmware/README.md` so it is reproducible, with the
   core and library versions
5. Report the compiled binary size and the free-sketch-space margin

**Update the honesty line precisely.** It becomes: *compiles clean for ESP32; never
flashed or run on hardware; sensor reads are stubs.* Do not let it become "tested."

If the toolchain install proves slow or fails, stop after 45 minutes, record the
attempt and the reason, and move on. This is a nice-to-have.

**Commit.**

---

## Task 4 — Consolidate

- Update `docs/results_ml.md` with both datasets and the Objective 6 answer
- Regenerate the results pack and `docs/numbers_sheet.md`
- Update `docs/model_cards.md` with the dataset each model was trained on
- Add Q&A entries:
  - *"Did you change the simulator until the result came out right?"* — No. The
    parameters were pre-registered and committed before the evaluation ran, with the
    predictions and the falsification condition written down. Both datasets are
    reported.
  - *"Why was the first simulator wrong?"* — It was unrealistically clean. Machines
    ran exactly on schedule, where real looms stop 8–22% of the time. We found this
    by measuring, not by assuming it.
- Update `docs/review_notes.md`

Run the full suite, exit code checked directly. **Final commit.**

---

## Task 5 — Still yours, not Claude Code's

**The offline rehearsal.** Network physically off, `streamlit run dashboard/app.py`,
click every page and selection, confirm every figure renders, record the result in
`docs/review_notes.md`. The code audit lowered the risk but cannot catch a runtime
fetch.

**Save `docs/feasibility_table.md` into the repo** and point the relevant Q&A entries
at it.

---

## Definition of done

- [ ] `pytest` passes; exit code checked directly
- [ ] Realised machine availability measured and inside its target band
- [ ] Stop onsets shown to be uncorrelated with shift boundaries
- [ ] An unplanned stop demonstrably lowers zone noise
- [ ] `docs/preregistration_day8.md` committed **before** the evaluation commit
- [ ] Literature-anchored and assumed parameters clearly distinguished
- [ ] Fresh 7-day dataset in a separate database; deterministic dataset intact
- [ ] Full ML suite re-run with zero model changes
- [ ] Both result sets reported side by side
- [ ] Objective 6 answered in one paragraph, whichever way it came out
- [ ] Each pre-registered prediction marked held or failed
- [ ] Firmware compile result recorded, with the honesty line scoped precisely
- [ ] All work committed

---

## Working instructions

Task 1, then commit the pre-registration, **then** Task 2. The commit order is the
evidence that the parameters were not fitted to the outcome.

Start the dataset generation before Task 3 so the toolchain install overlaps with it.

Do not touch any model hyperparameter in this phase. The only variable is the
simulator.

If a pre-registered prediction fails, that is the result. Write it down, say which
one failed, and offer your best explanation — exactly as you did with the collinearity
finding, which is the strongest piece of reasoning in the project so far.