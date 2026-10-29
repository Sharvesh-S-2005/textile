#!/usr/bin/env bash
# Day 8 Task 2: evaluate the stochastic dataset with the IDENTICAL pipeline.
#
# Pre-registered in docs/preregistration_day8.md (committed before this ran). Every
# command below is the same module, configuration, seed and split as the deterministic
# results; only the input database and the output locations differ, so the
# deterministic outputs in data/ and docs/ are never overwritten.
#
# Prerequisite (already done; ~2 h at 90x through the real broker and ingest):
#   python -m db.init_db --db data/week_stochastic/textile_ohs.db
#   python -m ingest.subscriber --db data/week_stochastic/textile_ohs.db --time-reference stream --idle-exit-s 60
#   python -m simulator.publisher --machine-stochastics --speed 90 --start 2026-10-20T06:00:00 \
#       --duration-h 168 --seed 42 --ground-truth-out data/week_stochastic/truth.jsonl
#
# Usage: bash scripts/run_stochastic_evaluation.sh   (from the repository root)
set -euo pipefail

PY=${PY:-.venv/Scripts/python.exe}
DB=data/week_stochastic/textile_ohs.db
OUT=data/week_stochastic
MODELS=models/stochastic
DOCS=docs/stochastic
FIGS=docs/figures/stochastic
NONE=docs/text/__no_discussion__.md     # per-dataset reports carry no hand-written discussion
mkdir -p "$OUT" "$MODELS" "$DOCS" "$FIGS" logs

echo "== exposure engine (localisation + sessions + exposure)"
"$PY" -m exposure.engine --db "$DB" --resolve > logs/stochastic_engine.log 2>&1

echo "== features"
"$PY" -m ml.features --db "$DB" --out "$OUT/features.parquet" 2> logs/stochastic_features.log

echo "== forecasting (persistence, seasonal, RF, XGBoost)"
"$PY" -m ml.forecast --features "$OUT/features.parquet" --model-dir "$MODELS" \
    --predictions "$OUT/forecast_predictions.parquet" --metrics "$OUT/forecast_metrics.json" \
    --report "$DOCS/results_forecast.md" --discussion "$NONE" > logs/stochastic_forecast.log 2>&1

echo "== exceedance classification (with zone groups)"
"$PY" -m ml.classify_exceedance --features "$OUT/features.parquet" --model-dir "$MODELS" \
    --predictions "$OUT/classification_predictions.parquet" --metrics "$OUT/classification_metrics.json" \
    --figures "$FIGS" --report "$DOCS/results_classification.md" --discussion "$NONE" \
    > logs/stochastic_classify.log 2>&1

echo "== ablation: Model 1/2/3, then no-lag (1-NL/3-NL) and no-lag-no-timetable (1-NS/3-NS)"
"$PY" -m ml.ablation --features "$OUT/features.parquet" --metrics "$OUT/ablation_metrics.json" \
    --figure "$FIGS/ablation_skill.png" --report "$DOCS/results_ablation.md" --discussion "$NONE" \
    > logs/stochastic_ablation.log 2>&1
"$PY" -m ml.ablation --nl-only --features "$OUT/features.parquet" --metrics "$OUT/ablation_metrics.json" \
    --figure "$FIGS/ablation_skill.png" --report "$DOCS/results_ablation.md" --discussion "$NONE" \
    > logs/stochastic_ablation_nl.log 2>&1

echo "== anomaly detection"
"$PY" -m ml.anomaly --db "$DB" --features "$OUT/features.parquet" --metrics "$OUT/anomaly_metrics.json" \
    --figure "$FIGS/anomaly_m5_vibration.png" --report "$DOCS/results_anomaly.md" --discussion "$NONE" \
    > logs/stochastic_anomaly.log 2>&1

echo "done: outputs in $OUT, $DOCS, $FIGS"
