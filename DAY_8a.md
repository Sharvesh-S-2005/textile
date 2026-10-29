# DAY 8A — Make the Long Run Survive, Then Finish Task 2

The 7-day stochastic generation was killed at ~2.4 days by **memory pressure (RAM)**.
Disk was not the limit, so relocating the data alone will not fix it.

The run is only about 1.9 hours of wall clock at `--speed 90`. It does not need a
bigger machine — it needs to stop growing. Work the fixes in order, then restart.

**The pre-registration stands.** Parameters, predictions and the evaluation plan are
unchanged. Any deviation forced by this restart is appended to
`docs/preregistration_day8.md` as a dated addendum, not edited into the original text.

---

## Task 1 — Find what grows

Before changing anything, measure. Run the publisher and subscriber for ~10 simulated
hours while sampling each process's resident set size every 15 seconds, and record
the two curves.

A flat curve means the kill came from elsewhere on the machine. A rising curve names
the leak. Do not guess — a 20-minute measurement decides which of the fixes below
actually matter, and tells you afterwards whether the fix worked.

The most likely causes, in order:

1. **The SQLite WAL is never checkpointed.** In WAL mode with continuous writes and
   no checkpoint, the WAL file grows unboundedly and its pages stay mapped. Over
   50,000+ inserts this is the prime suspect.
2. **Ground-truth JSONL accumulating in memory** instead of streaming to disk.
3. **An unbounded list in the publisher** — any per-message log, sent-message history
   or metrics array that appends and never trims.
4. **Ring buffers in the subscriber** that are keyed per `(device_id, field)` but
   never bounded in count, or that hold more than `stuck_value_count` entries.

Record the finding in `docs/review_notes.md`. **Commit.**

---

## Task 2 — Fix the growth

**2a — Checkpoint the WAL.** In `db/repo.py`, run
`PRAGMA wal_checkpoint(TRUNCATE)` every N commits (N in `config.yaml`, start at 200).
Log the WAL file size at each checkpoint so the effect is visible.

**2b — Stream the ground truth.** Open the JSONL once, write line by line, flush
every N lines. Never build the list in memory.

**2c — Bound every buffer.** Audit the publisher and subscriber for any collection
that appends without a cap. Replace with a `collections.deque(maxlen=...)` sized from
config. Add a test asserting the deque length never exceeds its cap.

**2d — Counters, not records.** Anything kept only for the periodic summary line
should be an integer counter, not a list of what happened.

`tests/test_memory_bounds.py`: run the publisher in `--dry-run` for a long simulated
window and assert resident set size at the end is within a configured margin of the
value at 10% through. That turns "it leaked" into a regression test.

**Commit.**

---

## Task 3 — Reduce peak pressure

**3a — Move the data directory to the D: drive.** Put the database, WAL and
ground-truth file there via the `paths` block in `config.yaml`. This does not fix RAM
but it removes disk as a second constraint and keeps the C: drive clear.

**3b — Lower the speed to 60.** Less in flight at once, and 168 simulated hours still
completes in about 2.8 hours. Throughput is not the goal here; finishing is.

**3c — Close other applications on the machine before starting.** Browsers and VS
Code extensions are usually the largest consumers, and this is a laptop, not a server.

---

## Task 4 — Chunk the run, if Tasks 2 and 3 are not enough

This is the robust answer: seven separate one-day runs. Each process **exits and
releases everything**, so peak memory is capped regardless of any leak that remains.
Ingest appends to the same database across chunks.

**One trap that must not be missed.** If each chunk starts a fresh publisher with the
same `--seed 42`, every chunk replays the *same* random sequence, and all seven days
come out with identical draws. That would silently destroy the dataset while looking
fine.

So derive the per-chunk seed: `effective_seed = base_seed + chunk_index`. Add
`--chunk-index N` to the publisher, log the effective seed on every run, and write a
test asserting two chunks with different indices produce different stop timings.

**The boundary discontinuity, disclosed.** The autocorrelated noise processes and the
stop processes restart at each chunk boundary rather than continuing. The AR(1)
processes re-converge within minutes, and the machine degradation trend is a function
of absolute simulated time so it is unaffected. Append this to the pre-registration
as a known deviation, with the reason.

Verify after all seven chunks: continuous coverage with no gaps or overlaps at the
boundaries, and the degrading machine's vibration trend still rising monotonically
across the whole week.

**Commit.**

---

## Task 5 — Fallback: a 5-day window

If Task 4 still cannot complete in the time available, **reduce the window to 5 days**
— 3 train, 1 validation, 1 test. That is a real deviation from the pre-registered
7-day plan, so:

- append it to `docs/preregistration_day8.md` with the reason and the date
- report it in `docs/results_ablation.md` wherever the comparison appears
- note that the deterministic dataset was 7 days, so **the two are not matched in
  length**, and say what that does to comparability: shorter training data generally
  depresses the stochastic result, which works *against* the pre-registered
  prediction rather than for it

That last point matters. A deviation that makes your own prediction harder to confirm
is a safe deviation. One that makes it easier would not be.

---

## Task 6 — Finish Task 2 of DAY_8

Once the dataset exists, run `scripts/run_stochastic_evaluation.sh` and complete
Tasks 2 and 4 of DAY_8.md exactly as written: full ML suite, no model changes, both
datasets side by side, Objective 6 answered in one paragraph, each pre-registered
prediction marked held or failed.

---

## Definition of done

- [ ] The memory growth is measured and named, not assumed
- [ ] WAL checkpointing in place with sizes logged
- [ ] Every unbounded collection bounded, with a test
- [ ] A memory regression test exists
- [ ] Data directory on the D: drive
- [ ] Full stochastic dataset generated, coverage verified continuous
- [ ] If chunked: per-chunk seeds verified distinct, discontinuity disclosed
- [ ] Any deviation appended to the pre-registration, dated, with its reason
- [ ] DAY_8 Tasks 2 and 4 complete
- [ ] All work committed

---

## Working instructions

Measure first (Task 1). A fix for the wrong cause costs an hour and teaches nothing.

Prefer Tasks 2 and 3 over chunking, because an unchunked run keeps the data
continuous and avoids the disclosure. Chunk only if the measurement says the leak
cannot be closed quickly.

Log the effective seed on every publisher start, in every mode. Identical draws
across chunks is the one failure here that would not announce itself.

Approved from the last phase: the single carding stop-rate adjustment, made before
any evaluation and disclosed in the pre-registration, is exactly the right handling.
The arduino-cli block is fine too — a manual API review recorded as "a review, not a
compile" is the honest wording. Leave it there.