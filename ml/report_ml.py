"""Consolidated ML documents from saved metrics (no training, no inference).

    python -m ml.report_ml

Writes docs/results_ml.md (the prose in docs/text/results_ml_body.md with the model
comparison table filled in) and docs/model_cards.md. Every number comes from the
metrics JSON files written by ml/forecast.py, ml/classify_exceedance.py,
ml/ablation.py and ml/anomaly.py, so nothing is copied by hand. The results pack
(scripts/build_results_pack.py) calls this.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import config

ROOT = config.CONFIG_PATH.parent
FORECAST_MODELS = ("persistence", "seasonal", "random_forest", "xgboost")
CLASSIFY_MODELS = ("persistence", "logistic_regression", "random_forest", "xgboost")
NOT_ATTEMPTED = ("Not attempted. It is optional in the Day 6 brief and was to be attempted only after Tasks 1, 2, "
                 "3 and 5.")


def metric_paths() -> dict[str, Path]:
    """Where each ML module writes its metrics (from config.yaml)."""
    ml = config.get_ml_config()
    return {"forecast": ROOT / ml["metrics_path"], "classification": ROOT / ml["classification"]["metrics_path"],
            "ablation": ROOT / ml["ablation"]["metrics_path"], "anomaly": ROOT / ml["anomaly"]["metrics_path"],
            "deep": ROOT / "data" / "deep_forecast_metrics.json"}


def load_metrics(paths: dict[str, Path] | None = None) -> dict[str, Any]:
    """Every metrics file that exists (missing ones are None)."""
    paths = paths or metric_paths()
    return {k: json.loads(p.read_text(encoding="utf-8")) if p.exists() else None for k, p in paths.items()}


def _split(key: str) -> tuple[str, str]:
    name, h = key.rsplit("_h", 1)
    return name, h


def comparison_rows(m: dict[str, Any]) -> list[dict[str, str]]:
    """One row per model per task on the test day, baselines first within each group.

    Columns: task, target, h, model, metric (headline metric name), value, baseline
    (the same metric for the group's baseline), note.
    """
    rows: list[dict[str, str]] = []
    f = m["forecast"]
    for key, e in f["targets"].items():
        target, h = _split(key)
        for model in FORECAST_MODELS:
            x = e["metrics"][model]
            note = "saved model" if model == e["chosen"] else ""
            if model in ("persistence", "seasonal"):
                note = "baseline"
            rows.append({"task": "A. Forecast", "target": target, "h": h, "model": model,
                         "metric": "skill vs persistence", "value": f"{x['skill_vs_persistence']:+.3f}",
                         "secondary": f"MAE {x['mae']:.2f}, R² {x['r2']:.3f}", "note": note})
    a = m["ablation"]
    if a:
        for key, e in a["forecasting"].items():
            target, h = _split(key)
            for s in ("model_1", "model_2", "model_3"):
                x = e["test"][s]
                d = e["differences"].get(f"{s}_vs_model_1")
                note = "" if d is None else (f"Δ vs Model 1 {d['point']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]")
                rows.append({"task": "Ablation (A)", "target": target, "h": h, "model": f"XGBoost {s.replace('_', ' ')}",
                             "metric": "skill vs persistence", "value": f"{x['skill_vs_persistence']:+.3f}",
                             "secondary": f"MAE {x['mae']:.2f}", "note": note})
    c = m["classification"]
    if c:
        for key, e in c["runs"].items():
            variant, h = _split(key)
            for model in CLASSIFY_MODELS:
                x = e["test"][model]
                note = "baseline" if model == "persistence" else ("saved model" if model == e["chosen"] else "")
                rows.append({"task": "B. Exceedance", "target": f"{variant} (prev. {x['prevalence']:.1%})", "h": h,
                             "model": model, "metric": "PR-AUC", "value": f"{x['pr_auc']:.3f}",
                             "secondary": f"F1 {x['f1'] or 0:.3f}, recall {x['recall'] or 0:.3f}, FPR {x['fpr']:.3f}",
                             "note": note})
    if a:
        for key, e in a["classification"].items():
            variant, h = _split(key)
            for s in ("model_1", "model_2", "model_3"):
                x = e["test"][s]
                d = e["differences"].get(f"{s}_vs_model_1", {}).get("pr_auc")
                note = "" if d is None else (f"Δ vs Model 1 {d['point']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]")
                rows.append({"task": "Ablation (B)", "target": variant, "h": h,
                             "model": f"XGBoost {s.replace('_', ' ')}", "metric": "PR-AUC", "value": f"{x['pr_auc']:.3f}",
                             "secondary": f"F1 {x['f1'] or 0:.3f}", "note": note})
    an = m["anomaly"]
    if an:
        bad = [k for k, v in an["machines"].items() if v["configured_degrading"]]
        for name in ("ewma", "isolation_forest"):
            h = an["healthy_summary"][name]
            day = an["machines"][bad[0]][name]["alarm_day"] if bad else None
            rows.append({"task": "C. Condition", "target": f"{bad[0] if bad else '—'} vibration", "h": "—",
                         "model": name, "metric": "detection day",
                         "value": "never" if day is None else f"{day:.2f}",
                         "secondary": f"healthy per-bin FPR {h['flag_rate']:.2%}",
                         "note": f"healthy machines alarmed: {', '.join(h['machines_with_alarm']) or 'none'}"})
    d = m.get("deep")
    if d and d.get("status") == "completed":
        for key, e in d["targets"].items():
            target, h = _split(key)
            x = e["test"]
            rows.append({"task": "A. Forecast (deep)", "target": target, "h": h, "model": d["model"],
                         "metric": "skill vs persistence", "value": f"{x['skill_vs_persistence']:+.3f}",
                         "secondary": f"MAE {x['mae']:.2f}", "note": f"XGBoost Model 3 {e['xgboost_skill']:+.3f}"})
    return rows


def comparison_table(rows: list[dict[str, str]]) -> str:
    """Markdown table of comparison_rows."""
    L = ["| Task | Target | h (min) | Model | Headline metric | **Value** | Secondary | Note |",
         "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['task']} | {r['target']} | {r['h']} | {r['model']} | {r['metric']} | **{r['value']}** | "
                 f"{r['secondary']} | {r['note']} |")
    return "\n".join(L)


def deep_learning_text(d: dict[str, Any] | None) -> str:
    """The Section 6 paragraph: completed, abandoned (with reason) or not attempted."""
    if not d:
        return NOT_ATTEMPTED
    if d.get("status") != "completed":
        return f"Attempted and abandoned: {d.get('reason', 'no reason recorded')} (`results_deep_forecast.md`)."
    return d.get("summary", "See `results_deep_forecast.md`.")


def write_results_ml(m: dict[str, Any], body_path: Path, out_path: Path) -> None:
    """docs/results_ml.md = header + body prose with the generated placeholders filled."""
    body = body_path.read_text(encoding="utf-8")
    for marker in ("{{COMPARISON_TABLE}}", "{{DEEP_LEARNING}}"):
        if marker not in body:
            raise ValueError(f"{body_path} is missing the {marker} marker")
    body = body.replace("{{COMPARISON_TABLE}}", comparison_table(comparison_rows(m)))
    body = body.replace("{{DEEP_LEARNING}}", deep_learning_text(m.get("deep")))
    head = ["# Machine-learning results — consolidated", "",
            "**SIMULATED DATA.** One document for every ML result. Generated by `ml/report_ml.py` from the saved "
            "metrics (prose in `docs/text/results_ml_body.md`); the per-task documents hold the full tables.", ""]
    out_path.write_text("\n".join(head) + body.strip() + "\n", encoding="utf-8")


# ---------------------------------------------------------------- model cards


def _card(title: str, fields: list[tuple[str, str]]) -> list[str]:
    L = [f"## {title}", "", "| | |", "|---|---|"]
    L += [f"| **{k}** | {v} |" for k, v in fields]
    return L + [""]


def write_model_cards(m: dict[str, Any], out_path: Path) -> None:
    """docs/model_cards.md: one short card per deployed model."""
    f, c, an = m["forecast"], m["classification"], m["anomaly"]
    split = "; ".join(f["splits"])
    hp = f["hyperparameters"]
    cfg = config.get_ml_config()
    L = ["# Model cards", "",
         "**SIMULATED DATA.** One card per deployed model. Generated by `ml/report_ml.py` from the saved metrics. "
         "Every model here was trained and evaluated on simulated data only; none is validated for use in a "
         "real mill.", ""]
    names = {"noise_dba": "Noise forecaster", "pm25_ugm3": "PM2.5 forecaster", "wbgt_c": "WBGT forecaster"}
    units = {"noise_dba": "dBA", "pm25_ugm3": "µg/m³", "wbgt_c": "°C"}
    for target in ("noise_dba", "pm25_ugm3", "wbgt_c"):
        keys = [k for k in f["targets"] if k.startswith(target)]
        chosen = ", ".join(f"t+{_split(k)[1]}: {f['targets'][k]['chosen']}" for k in keys)
        metrics = "; ".join(
            f"t+{_split(k)[1]}: skill {f['targets'][k]['metrics'][f['targets'][k]['chosen']]['skill_vs_persistence']:+.3f}, "
            f"MAE {f['targets'][k]['metrics'][f['targets'][k]['chosen']]['mae']:.2f} {units[target]} "
            f"(seasonal baseline {f['targets'][k]['metrics']['seasonal']['skill_vs_persistence']:+.3f})" for k in keys)
        failure = {
            "noise_dba": "Most skill comes from the fixed timetable; on steady rows skill is lower. At t+60 it does not "
                         "beat the seasonal baseline. Will degrade when breaks or machine stops are unscheduled.",
            "pm25_ugm3": "Blowdown and cleaning spikes are only partly predictable; MAE is large in the dustiest zones. "
                         "Predicts an optical-sensor signal, not a gravimetric concentration.",
            "wbgt_c": "Barely beats persistence (about 0.1–0.2 °C of MAE). WBGT is approximated from temperature and "
                      "RH (no globe thermometer in this phase), so the target itself is approximate.",
        }[target]
        L += _card(f"{names[target]} (`models/forecast_{target}_{{30,60}}.pkl`)", [
            ("Purpose", f"Forecast zone {target.split('_')[0].upper().replace('PM25', 'PM2.5')} 30 and 60 min ahead "
                        "for the Live Zones view and early warning."),
            ("Inputs", f"{f['n_features']} zone features at t (`ml/features.py`): lags, rolling statistics, rate of "
                       "change, temperature/RH/CO2, machine state, occupancy, time of day, shift, break, zone."),
            ("Label source", f"Measured zone {units[target]} at t+h (`measured_environmental`)."),
            ("Output label", "`predicted_risk` (an ML output, per the project's labelling rule). It is a forecast "
                             "of a zone hazard level, not of anyone's health."),
            ("Training window / split", split),
            ("Model and hyperparameters", f"Chosen per horizon by validation MAE ({chosen}). RF {hp['random_forest']}; "
                                          f"XGBoost {hp['xgboost']}, early stopping "
                                          f"{hp['xgboost_early_stopping_rounds']}. Fixed in config.yaml, not grid-"
                                          "searched; seed 42."),
            ("Test-day metrics", metrics),
            ("Known failure modes", failure),
            ("Intended use", "Zone-level trend and early warning on the dashboard, always beside the measured value."),
            ("Must NOT be used for", "Any statement about an individual's health; compliance decisions (it is not a "
                                     "measurement); replacing a sound-level meter or gravimetric sampler."),
        ])
    if c:
        cc = cfg["classification"]
        for variant in ("above", "crossing"):
            keys = [k for k in c["runs"] if k.startswith(variant)]
            met = "; ".join(
                f"t+{_split(k)[1]} ({c['runs'][k]['chosen']}): prevalence {c['runs'][k]['test'][c['runs'][k]['chosen']]['prevalence']:.1%}, "
                f"PR-AUC {c['runs'][k]['test'][c['runs'][k]['chosen']]['pr_auc']:.3f} (persistence "
                f"{c['runs'][k]['test']['persistence']['pr_auc']:.3f}), F1 {c['runs'][k]['test'][c['runs'][k]['chosen']]['f1']:.3f}, "
                f"recall {c['runs'][k]['test'][c['runs'][k]['chosen']]['recall']:.3f}, FPR "
                f"{c['runs'][k]['test'][c['runs'][k]['chosen']]['fpr']:.3f}" for k in keys)
            params = "; ".join(f"t+{_split(k)[1]}: {c['runs'][k]['chosen']} "
                               f"{json.dumps(c['runs'][k]['tuning'][c['runs'][k]['chosen']]['chosen'])}, threshold "
                               f"{c['runs'][k]['test'][c['runs'][k]['chosen']]['threshold']:.3f}" for k in keys)
            what = ("noise at t+h ≥ 90 dBA" if variant == "above"
                    else "noise below 90 dBA now and ≥ 90 dBA at t+h (a quiet zone about to become hazardous)")
            failure = ("High accuracy is mostly base rate in always-loud zones (Z1, Z4, Z5); read PR-AUC and prevalence."
                       if variant == "above" else
                       "Much of the skill is restarts after scheduled breaks; on steady rows PR-AUC is lower, and in a "
                       "zone hovering near 90 dBA (Z2) it is barely above chance.")
            L += _card(f"Exceedance classifier, `{variant}` (`models/classify_{variant}_h{{30,60}}.pkl`)", [
                ("Purpose", f"Warn that {what}."),
                ("Inputs", f"The same {c['n_features']} features as the forecasters."),
                ("Label source", f"Measured noise at t+h vs the {c['criterion']} criterion ({c['threshold_db']:.0f} "
                                 "dBA). A future measurement, not a rule on the inputs, so not circular."),
                ("Output label", "`predicted_risk` (an ML output). A zone-level hazard flag, not a health estimate."),
                ("Training window / split", "; ".join(c["splits"])),
                ("Hyperparameters and search grid", f"Grid (validation PR-AUC): `{json.dumps(cc['grids'])}`. Chosen: "
                                                    f"{params}. Threshold: highest validation recall with FPR ≤ "
                                                    f"{cc['max_false_positive_rate']:.0%}."),
                ("Test-day metrics", met),
                ("Known failure modes", failure),
                ("Intended use", "A zone-level early-warning flag for supervisors, shown with its precision (roughly "
                                 "one false alarm per two true ones on `crossing`)."),
                ("Must NOT be used for", "Reducing any worker's calculated dose; judging compliance; any health "
                                         "statement about a person; replacing measured exceedance alerts."),
            ])
    if an:
        s = an["settings"]
        bad = [k for k, v in an["machines"].items() if v["configured_degrading"]]
        L += _card("Machine condition monitor — EWMA control chart (`ml/anomaly.py`, deployed detector)", [
            ("Purpose", "Flag a machine whose vibration has drifted from its own baseline (wear)."),
            ("Inputs", f"{s['bin_min']}-min mean vib_rms_g of running-only, unflagged readings."),
            ("Label source", "None for training (unsupervised). Scored against `machines[].degrading` in config.yaml."),
            ("Training window", f"Each machine's own first {s['reference_days']} day (Phase I reference)."),
            ("Parameters", f"λ = {s['ewma']['lambda']}, limits ±{s['ewma']['limit_sigma']}σ (exact width), alarm = "
                           f"{s['alarm_consecutive_bins']} consecutive flagged bins. Set before results were seen."),
            ("Metrics", f"{bad[0] if bad else '—'} detected on day "
                        f"{an['machines'][bad[0]]['ewma']['alarm_day']:.2f}; healthy machines alarmed: "
                        f"{', '.join(an['healthy_summary']['ewma']['machines_with_alarm']) or 'none'}; healthy per-bin "
                        f"FPR {an['healthy_summary']['ewma']['flag_rate']:.2%}."),
            ("Known failure modes", "Calibrating on a day when the machine is already wearing inflates its baseline "
                                    "and delays detection. Mean-only, so it misses faults that change crest factor "
                                    "without moving RMS. Tested only on one smooth simulated wear curve."),
            ("Intended use", "A maintenance prompt to inspect the machine."),
            ("Must NOT be used for", "Fault diagnosis (it says *that* vibration changed, not *why*); claims about "
                                     "worker health; shutting a machine down without inspection."),
        ])
        L += _card("Machine condition monitor — isolation forest (compared, not deployed)", [
            ("Purpose", "Unsupervised comparison for the EWMA chart."),
            ("Inputs", "RMS, peak, crest factor, 1-h rolling means of RMS and peak."),
            ("Parameters", f"{s['isolation_forest']['n_estimators']} trees, seed 42; flag above the "
                           f"{s['isolation_forest']['reference_quantile']:.0%} quantile of reference scores."),
            ("Metrics", f"Detected on day {an['machines'][bad[0]]['isolation_forest']['alarm_day']:.2f}; healthy per-bin "
                        f"FPR {an['healthy_summary']['isolation_forest']['flag_rate']:.2%}."),
            ("Why not deployed", "No earlier detection than the control chart, a ~50× higher per-bin false-positive "
                                 "rate, and harder for a maintenance engineer to interpret."),
        ])
    L += _card("Risk index (`exposure/risk_index.py`) — listed for completeness; NOT a trained model", [
        ("Purpose", "Order worker-days by exposure for review."),
        ("Label", "`deterministic_index`: a formula, no training and no labels."),
        ("Reported with", "The worst single component beside the weighted composite (components are not on "
                          "commensurable scales), and both noise doses."),
        ("Must NOT be used for", "Diagnosis of any condition; estimating the probability of a health outcome. The ML "
                                 "risk classifier is deferred until questionnaire or audiometry labels exist."),
    ])
    out_path.write_text("\n".join(L), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    p = argparse.ArgumentParser(prog="python -m ml.report_ml", description=__doc__.split("\n")[0])
    p.add_argument("--body", type=Path, default=ROOT / "docs" / "text" / "results_ml_body.md")
    p.add_argument("--out", type=Path, default=ROOT / "docs" / "results_ml.md")
    p.add_argument("--cards", type=Path, default=ROOT / "docs" / "model_cards.md")
    args = p.parse_args(argv)
    m = load_metrics()
    missing = [k for k in ("forecast", "classification", "ablation", "anomaly") if m[k] is None]
    if missing:
        raise SystemExit(f"missing metrics: {missing}; run the corresponding ml modules first")
    write_results_ml(m, args.body, args.out)
    write_model_cards(m, args.cards)
    print(f"wrote {args.out} and {args.cards}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
