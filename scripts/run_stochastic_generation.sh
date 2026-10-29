#!/usr/bin/env bash
# Day 8 / 8A: generate the 7-day stochastic dataset through the real MQTT -> ingest path.
#
# Same command as the pre-registered plan (docs/preregistration_day8.md section 5), with the
# Day 8A restart settings appended there as a dated addendum:
#   * --speed 60 instead of 90: pacing only, every payload is a function of simulated time
#     and seed, so the data are identical; less in flight at once (about 2.8 h of wall clock)
#   * --discard-session: drop messages an interrupted run left queued at the broker, which
#     would otherwise move the stream watermark and flag every record late_upload
#   * OPENBLAS_NUM_THREADS=1: numpy's BLAS is not used by the simulator; this avoids
#     about 480 MB of per-thread buffer commit (measured in Day 8A Task 1)
# The database, WAL and ground truth are in data/week_stochastic/, inside the repository
# on the D: drive.
#
# Usage: bash scripts/run_stochastic_generation.sh   (from the repository root)
set -euo pipefail

PY=${PY:-.venv/Scripts/python.exe}
DIR=data/week_stochastic
DB=$DIR/textile_ohs.db
SPEED=${SPEED:-60}
mkdir -p "$DIR" logs

"$PY" -m db.init_db --db "$DB" --reset
"$PY" -m ingest.subscriber --db "$DB" --time-reference stream --idle-exit-s 60 --discard-session \
    > logs/stochastic_subscriber.log 2>&1 &
SUB=$!
until grep -q "subscribing" logs/stochastic_subscriber.log; do
    kill -0 "$SUB" 2>/dev/null || { echo "subscriber exited early; see logs/stochastic_subscriber.log"; exit 1; }
    sleep 1
done

OPENBLAS_NUM_THREADS=1 "$PY" -m simulator.publisher --machine-stochastics --speed "$SPEED" \
    --start 2026-10-20T06:00:00 --duration-h 168 --seed 42 \
    --ground-truth-out "$DIR/truth.jsonl" > logs/stochastic_publisher.log 2>&1
echo "publisher finished; waiting for ingest to drain (idle exit 60 s)"
wait "$SUB"
tail -1 logs/stochastic_subscriber.log
