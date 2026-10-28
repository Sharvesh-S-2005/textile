"""Hazard forecasting (Task A): noise, PM2.5 and WBGT at t+30 and t+60 minutes.

    python -m ml.forecast [--features data/features.parquet]

Four models per target and horizon, in order; each should beat the one before:
  1. persistence  - y(t+h) = y(t)
  2. seasonal     - y(t+h) = y at the same clock minute the previous day
  3. random forest
  4. XGBoost (early stopping on the validation day)

Split: chronological by whole UTC days, never shuffled (5 train / 1 validation /
1 test; 60/20/20 by time if fewer days). Rows whose target time falls beyond the end
of their own split are purged, so no label straddles a split boundary.

Metrics per target/horizon/model: skill vs persistence (1 - MAE/MAE_persistence),
MAE, RMSE, R^2, all on the SAME test rows (target and both baselines present).

Exceedance lead time (noise vs the India 90 dBA criterion):
  * event  = a sustained upward crossing: below the threshold for the previous
    `sustain` minutes and at/above it for the next `sustain` minutes;
  * warning = a forecast >= threshold issued at t while the measured level is
    still below it;
  * hit = an event preceded by >= 1 warning within the h minutes before it;
    lead time = event time - first such warning;
  * false-alarm ratio = warnings with no event within (t, t+h] / all warnings.
Persistence can never warn ahead under this definition (its forecast is the
current, sub-threshold value), so its hit rate is 0 by construction.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor

import config
from db.repo import from_epoch_ms
from ml.features import feature_columns

log = logging.getLogger("ml.forecast")

DAY_MS = 86_400_000
MIN_MS = 60_000
MODEL_ORDER = ("persistence", "seasonal", "random_forest", "xgboost")
EVENT_SUSTAIN_MIN = 5


# ---------------------------------------------------------------- split


@dataclass(frozen=True)
class Split:
    """One chronological split: [start_ms, end_ms)."""

    name: str
    start_ms: int
    end_ms: int

    def describe(self) -> str:
        return f"{self.name}: {from_epoch_ms(self.start_ms)} -> {from_epoch_ms(self.end_ms)}"


def chronological_splits(t_ms: pd.Series, days: dict[str, int], fallback: dict[str, float]) -> tuple[list[Split], str]:
    """Train / validation / test by whole UTC days (or 60/20/20 by time if too few days).

    Returns the splits and a sentence describing which rule was used. Only complete
    days are used for the day split; a trailing partial day is left out.
    """
    first_day = int(t_ms.min()) // DAY_MS * DAY_MS
    last_full_day_end = (int(t_ms.max()) + MIN_MS) // DAY_MS * DAY_MS   # exclusive end of last complete day
    full_days = (last_full_day_end - first_day) // DAY_MS
    needed = days["train"] + days["validation"] + days["test"]
    if full_days >= needed:
        b1 = first_day + days["train"] * DAY_MS
        b2 = b1 + days["validation"] * DAY_MS
        b3 = b2 + days["test"] * DAY_MS
        rule = (f"whole UTC days: {days['train']} train / {days['validation']} validation / {days['test']} test"
                + (" (later partial data left out)" if b3 < int(t_ms.max()) else ""))
        return [Split("train", first_day, b1), Split("validation", b1, b2), Split("test", b2, b3)], rule
    lo, hi = int(t_ms.min()), int(t_ms.max()) + MIN_MS
    span = hi - lo
    b1 = lo + int(span * fallback["train"])
    b2 = b1 + int(span * fallback["validation"])
    rule = (f"only {full_days} complete days (< {needed}): chronological "
            f"{fallback['train']:.0%}/{fallback['validation']:.0%}/{fallback['test']:.0%} by time")
    return [Split("train", lo, b1), Split("validation", b1, b2), Split("test", b2, hi)], rule


def split_mask(t_ms: pd.Series, split: Split, horizon_min: int) -> pd.Series:
    """Rows of `split` whose target time t + h is also inside it (boundary purge)."""
    return (t_ms >= split.start_ms) & (t_ms < split.end_ms) & (t_ms + horizon_min * MIN_MS < split.end_ms)


# ---------------------------------------------------------------- metrics


def skill_score(mae_model: float, mae_persistence: float) -> float:
    """1 - MAE_model / MAE_persistence: 0 = no better than persistence, 1 = perfect."""
    if mae_persistence <= 0:
        raise ValueError("persistence MAE must be positive")
    return 1.0 - mae_model / mae_persistence


def regression_metrics(y: np.ndarray, yhat: np.ndarray, mae_persistence: float) -> dict[str, float]:
    """Skill, MAE, RMSE, R^2 for one model on one target/horizon."""
    mae = float(mean_absolute_error(y, yhat))
    return {"skill_vs_persistence": skill_score(mae, mae_persistence), "mae": mae,
            "rmse": float(math.sqrt(mean_squared_error(y, yhat))), "r2": float(r2_score(y, yhat))}


# ---------------------------------------------------------------- exceedance lead time


def crossing_events(series: pd.Series, threshold: float, sustain: int = EVENT_SUSTAIN_MIN) -> list[int]:
    """Indices (positions) of sustained upward crossings of `threshold` in a 1-min series.

    Position i is an event if the previous `sustain` values are all < threshold and
    values i .. i+sustain-1 are all >= threshold (missing values break a run).
    """
    v = series.to_numpy(dtype=float)
    above = v >= threshold
    below = v < threshold                      # NaN is neither above nor below
    events = []
    for i in range(sustain, len(v) - sustain + 1):
        if below[i - sustain:i].all() and above[i:i + sustain].all():
            events.append(i)
    return events


def lead_time_analysis(measured: pd.Series, forecast: pd.Series, threshold: float, horizon_min: int,
                       sustain: int = EVENT_SUSTAIN_MIN) -> dict[str, Any]:
    """Hits, lead times and false alarms for one zone's 1-minute series.

    `measured` and `forecast` share a contiguous 1-minute index; forecast[i] is the
    value predicted at time i for time i + horizon_min.
    """
    v = measured.to_numpy(dtype=float)
    f = forecast.to_numpy(dtype=float)
    events = crossing_events(measured, threshold, sustain)
    warning = (f >= threshold) & (v < threshold)
    leads, hits = [], 0
    for e in events:
        window = [i for i in range(max(0, e - horizon_min), e) if warning[i]]
        if window:
            hits += 1
            leads.append(e - window[0])
    event_set = set(events)
    warn_idx = np.flatnonzero(warning)
    false = sum(1 for i in warn_idx if not any(e in event_set for e in range(i + 1, i + horizon_min + 1)))
    return {"events": len(events), "hits": hits, "lead_times_min": leads, "warnings": int(len(warn_idx)),
            "false_alarms": int(false)}


def summarise_lead(parts: list[dict[str, Any]]) -> dict[str, Any]:
    """Combine per-zone lead-time results."""
    events = sum(p["events"] for p in parts)
    hits = sum(p["hits"] for p in parts)
    warnings = sum(p["warnings"] for p in parts)
    false = sum(p["false_alarms"] for p in parts)
    leads = [x for p in parts for x in p["lead_times_min"]]
    return {"events": events, "hits": hits, "hit_rate": hits / events if events else None,
            "median_lead_min": float(np.median(leads)) if leads else None,
            "warnings": warnings, "false_alarms": false,
            "false_alarm_ratio": false / warnings if warnings else None}


# ---------------------------------------------------------------- models


def make_random_forest(params: dict[str, Any], seed: int) -> Any:
    """Random forest with median imputation (fitted on training data only)."""
    return make_pipeline(SimpleImputer(strategy="median"),
                         RandomForestRegressor(random_state=seed, **params))


def make_xgboost(params: dict[str, Any], seed: int) -> XGBRegressor:
    """XGBoost regressor (handles missing values natively)."""
    return XGBRegressor(random_state=seed, early_stopping_rounds=30, eval_metric="mae", **params)


def save_model(model: Any, path: Path, meta: dict[str, Any]) -> None:
    """Persist a model with its metadata (features, target, horizon, parameters)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "meta": meta}, path)


def load_model(path: Path) -> tuple[Any, dict[str, Any]]:
    """Load a model saved by save_model."""
    blob = joblib.load(path)
    return blob["model"], blob["meta"]


# ---------------------------------------------------------------- pipeline


def run(features_path: Path, model_dir: Path, predictions_path: Path, metrics_path: Path) -> dict[str, Any]:
    """Train and evaluate every model; write models, test predictions and metrics."""
    c = config.get_ml_config()
    seed = int(c["random_seed"])
    df = pd.read_parquet(features_path)
    feats = feature_columns()
    splits, rule = chronological_splits(df["t_ms"], c["split_days"], c["split_fallback_fractions"])
    threshold = float(config.get_exposure_limits()["criteria"][c["exceedance_criterion"]]["criterion_db"])
    results: dict[str, Any] = {"split_rule": rule, "splits": [s.describe() for s in splits], "targets": {},
                               "hyperparameters": {"random_forest": c["random_forest"], "xgboost": c["xgboost"],
                                                   "xgboost_early_stopping_rounds": 30, "random_seed": seed},
                               "n_features": len(feats), "exceedance_threshold_db": threshold,
                               "event_sustain_min": EVENT_SUSTAIN_MIN}
    predictions = []
    for target in c["targets"]:
        for h in c["horizons_min"]:
            key = f"{target}_h{h}"
            y_col, s_col = f"target_{target}_h{h}", f"seasonal_{target}_h{h}"
            masks = {s.name: split_mask(df["t_ms"], s, h) & df[y_col].notna() for s in splits}
            train, val = df[masks["train"]], df[masks["validation"]]
            test = df[masks["test"] & df[target].notna() & df[s_col].notna()]   # same rows for all models
            rf = make_random_forest(c["random_forest"], seed).fit(train[feats], train[y_col])
            xgb = make_xgboost(c["xgboost"], seed).fit(train[feats], train[y_col],
                                                       eval_set=[(val[feats], val[y_col])], verbose=False)
            preds = {"persistence": test[target].to_numpy(), "seasonal": test[s_col].to_numpy(),
                     "random_forest": rf.predict(test[feats]), "xgboost": xgb.predict(test[feats])}
            val_mae = {"random_forest": float(mean_absolute_error(val[y_col], rf.predict(val[feats]))),
                       "xgboost": float(mean_absolute_error(val[y_col], xgb.predict(val[feats])))}
            chosen = min(val_mae, key=val_mae.get)
            y = test[y_col].to_numpy()
            mae_p = float(mean_absolute_error(y, preds["persistence"]))
            metrics = {m: regression_metrics(y, preds[m], mae_p) for m in MODEL_ORDER}
            beats = {MODEL_ORDER[i]: metrics[MODEL_ORDER[i]]["mae"] < metrics[MODEL_ORDER[i - 1]]["mae"]
                     for i in range(1, len(MODEL_ORDER))}
            entry = {"n_train": int(len(train)), "n_validation": int(len(val)), "n_test": int(len(test)),
                     "metrics": metrics, "beats_previous": beats, "validation_mae": val_mae, "chosen": chosen,
                     "xgboost_best_iteration": int(xgb.best_iteration)}
            model = rf if chosen == "random_forest" else xgb
            save_model(model, model_dir / f"forecast_{target}_{h}.pkl",
                       {"target": target, "horizon_min": h, "model": chosen, "features": feats,
                        "split_rule": rule, "splits": [s.describe() for s in splits]})
            out = pd.DataFrame({"t_ms": test["t_ms"].to_numpy(), "ts": test["ts"].to_numpy(),
                                "zone_id": test["zone_id"].astype(str).to_numpy(), "target": target,
                                "horizon_min": h, "actual": y, **{m: preds[m] for m in MODEL_ORDER},
                                "chosen_model": chosen})
            out["chosen"] = out[chosen]
            predictions.append(out)
            if target == "noise_dba":
                entry["exceedance"] = exceedance_for(df, test, preds, h, threshold, splits[-1])
            results["targets"][key] = entry
            log.info("%s: chosen %s, skill RF %.3f XGB %.3f (n_test %d)", key, chosen,
                     metrics["random_forest"]["skill_vs_persistence"], metrics["xgboost"]["skill_vs_persistence"],
                     len(test))
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(predictions, ignore_index=True).to_parquet(predictions_path, index=False)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def exceedance_for(df: pd.DataFrame, test: pd.DataFrame, preds: dict[str, np.ndarray], h: int,
                   threshold: float, test_split: Split) -> dict[str, Any]:
    """Lead-time analysis per model on the test day, per zone on a contiguous 1-minute grid."""
    out = {}
    grid = np.arange(test_split.start_ms, test_split.end_ms, MIN_MS)
    for model in MODEL_ORDER:
        parts = []
        for zone, zdf in df[(df["t_ms"] >= test_split.start_ms) & (df["t_ms"] < test_split.end_ms)].groupby(
                "zone_id", observed=True):
            measured = zdf.set_index("t_ms")["noise_dba"].reindex(grid)
            ztest = test["zone_id"].astype(str) == str(zone)
            forecast = pd.Series(preds[model][ztest.to_numpy()], index=test.loc[ztest, "t_ms"].to_numpy()).reindex(grid)
            parts.append(lead_time_analysis(measured, forecast, threshold, h))
        out[model] = summarise_lead(parts)
    return out


def schedule_mask(pred: pd.DataFrame, schedule: pd.DataFrame, horizon_min: int) -> np.ndarray:
    """True where no break or shift boundary lies in (t, t+h] for that zone ("steady").

    `schedule` is indexed by (zone_id, t_ms) with is_break and shift_code columns.
    A row whose schedule is unknown at some minute in the horizon counts as not steady.
    """
    now = schedule.reindex(pd.MultiIndex.from_arrays([pred["zone_id"], pred["t_ms"]]))
    steady = np.ones(len(pred), dtype=bool)
    for k in range(1, horizon_min + 1):
        mid = schedule.reindex(pd.MultiIndex.from_arrays([pred["zone_id"], pred["t_ms"] + k * MIN_MS]))
        steady &= ((mid["is_break"].to_numpy() == now["is_break"].to_numpy())
                   & (mid["shift_code"].to_numpy() == now["shift_code"].to_numpy()))
    return steady


def schedule_subset_metrics(predictions: pd.DataFrame, features: pd.DataFrame) -> dict[str, Any]:
    """Skill per model on 'steady' rows vs rows spanning a scheduled break/shift change.

    Separates forecasting skill from learning the (fixed) timetable.
    """
    schedule = features.assign(zone_id=features["zone_id"].astype(str)).set_index(["zone_id", "t_ms"])[
        ["is_break", "shift_code"]]
    out: dict[str, Any] = {}
    for (target, h), g in predictions.groupby(["target", "horizon_min"]):
        steady = schedule_mask(g, schedule, int(h))
        part = {}
        for name, mask in (("steady", steady), ("schedule_transition", ~steady)):
            sub = g[mask]
            if not len(sub):
                continue
            mae_p = float(np.mean(np.abs(sub["actual"] - sub["persistence"])))
            part[name] = {"n": int(len(sub)), "mae_persistence": mae_p,
                          "skill": {m: skill_score(float(np.mean(np.abs(sub["actual"] - sub[m]))), mae_p)
                                    for m in MODEL_ORDER}}
        out[f"{target}_h{h}"] = part
    return out


def write_report(results: dict[str, Any], path: Path, discussion_path: Path | None = None) -> None:
    """docs/results_forecast.md: split, hyperparameters, metrics, lead time, discussion."""
    c = config.get_ml_config()
    L = ["# Forecasting results (Task A)", "",
         "**SIMULATED DATA.** Generated by `ml/forecast.py` (`--report-only` regenerates it from saved "
         "predictions). Skill = 1 − MAE_model / MAE_persistence on the same test rows; > 0 beats persistence.", "",
         "## Split", "", f"Chronological, never shuffled — {results['split_rule']}. Rows whose target time "
         "crosses the end of their own split are purged.", ""]
    L += [f"- {s}" for s in results["splits"]]
    L += ["", "| Target | Horizon | n train | n validation | n test |", "|---|---|---|---|---|"]
    for key, e in results["targets"].items():
        t, h = key.rsplit("_h", 1)
        L.append(f"| {t} | {h} min | {e['n_train']:,} | {e['n_validation']:,} | {e['n_test']:,} |")
    hp = results["hyperparameters"]
    L += ["", "## Hyperparameters (fixed in config.yaml, not tuned on the test day)", "",
          f"- Random forest: {hp['random_forest']} (median imputation fitted on train)",
          f"- XGBoost: {hp['xgboost']}, early stopping {hp['xgboost_early_stopping_rounds']} rounds on validation MAE",
          f"- Seed {hp['random_seed']}; {results['n_features']} features (`ml/features.py`). The model saved per "
          "target/horizon is the one with the lower **validation** MAE.", "",
          "## Metrics on the test day", "",
          "| Target | h | Model | **Skill** | MAE | RMSE | R² | Beats previous? |", "|---|---|---|---|---|---|---|---|"]
    for key, e in results["targets"].items():
        t, h = key.rsplit("_h", 1)
        for m in MODEL_ORDER:
            x = e["metrics"][m]
            beats = "—" if m == "persistence" else ("yes" if e["beats_previous"][m] else "**NO**")
            chosen = " (saved)" if m == e["chosen"] else ""
            L.append(f"| {t} | {h} | {m}{chosen} | **{x['skill_vs_persistence']:+.3f}** | {x['mae']:.2f} | "
                     f"{x['rmse']:.2f} | {x['r2']:.3f} | {beats} |")
    if any("by_schedule" in e for e in results["targets"].values()):
        L += ["", "## Skill with and without scheduled transitions", "",
              "*Steady* = no break or shift boundary between t and t+h. *Transition* = the horizon spans one "
              "(break start/end, shift changeover), where the fixed timetable makes the answer predictable.", "",
              "| Target | h | Subset | n | MAE persistence | Skill seasonal | Skill RF | Skill XGBoost |",
              "|---|---|---|---|---|---|---|---|"]
        for key, e in results["targets"].items():
            t, h = key.rsplit("_h", 1)
            for name, part in e.get("by_schedule", {}).items():
                s = part["skill"]
                L.append(f"| {t} | {h} | {name} | {part['n']:,} | {part['mae_persistence']:.2f} | "
                         f"{s['seasonal']:+.3f} | {s['random_forest']:+.3f} | {s['xgboost']:+.3f} |")
    L += ["", f"## Exceedance lead time (noise vs {results['exceedance_threshold_db']:.0f} dBA, "
          f"{c['exceedance_criterion']} criterion)", "",
          f"Event = sustained upward crossing (below for {results['event_sustain_min']} min, then at/above for "
          f"{results['event_sustain_min']} min). Warning = forecast ≥ threshold issued while still below it. "
          "Hit = event with a warning in the h minutes before it. False-alarm ratio = warnings with no event "
          "within h minutes / all warnings. Operating threshold = the criterion level itself.", "",
          "| h | Model | Events | Hit rate | Median lead (min) | Warnings | False-alarm ratio |",
          "|---|---|---|---|---|---|---|"]
    for key, e in results["targets"].items():
        if "exceedance" not in e:
            continue
        h = key.rsplit("_h", 1)[1]
        for m in MODEL_ORDER:
            x = e["exceedance"][m]
            hit = "n/a" if x["hit_rate"] is None else f"{x['hit_rate']:.0%}"
            lead = "—" if x["median_lead_min"] is None else f"{x['median_lead_min']:.1f}"
            far = "—" if x["false_alarm_ratio"] is None else f"{x['false_alarm_ratio']:.1%}"
            L.append(f"| {h} | {m} | {x['events']} | {hit} | {lead} | {x['warnings']} | {far} |")
    if discussion_path is not None and discussion_path.exists():
        L += ["", discussion_path.read_text(encoding="utf-8").strip()]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = config.CONFIG_PATH.parent
    c = config.get_ml_config()
    p = argparse.ArgumentParser(prog="python -m ml.forecast", description=__doc__.split("\n")[0])
    p.add_argument("--features", type=Path, default=root / c["features_path"])
    p.add_argument("--model-dir", type=Path, default=root / config.get_paths()["model_dir"])
    p.add_argument("--predictions", type=Path, default=root / c["predictions_path"])
    p.add_argument("--metrics", type=Path, default=root / c["metrics_path"])
    p.add_argument("--report", type=Path, default=root / "docs" / "results_forecast.md")
    p.add_argument("--discussion", type=Path, default=root / "docs" / "text" / "results_forecast_discussion.md")
    p.add_argument("--report-only", action="store_true",
                   help="skip training: recompute the schedule analysis and report from saved outputs")
    args = p.parse_args(argv)
    if args.report_only:
        results = json.loads(args.metrics.read_text(encoding="utf-8"))
    else:
        results = run(args.features, args.model_dir, args.predictions, args.metrics)
    by_schedule = schedule_subset_metrics(pd.read_parquet(args.predictions), pd.read_parquet(args.features))
    for key, part in by_schedule.items():
        results["targets"][key]["by_schedule"] = part
    args.metrics.write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_report(results, args.report, args.discussion)
    print(json.dumps({k: {m: round(v["metrics"][m]["skill_vs_persistence"], 3) for m in MODEL_ORDER}
                      for k, v in results["targets"].items()}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
