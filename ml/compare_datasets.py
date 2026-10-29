"""Day 8: deterministic vs stochastic machine behaviour, side by side.

    python -m ml.compare_datasets

Reads the saved outputs of the identical ML pipeline run on two datasets (paths in
config.yaml `ml.comparison`) and writes:

  docs/results_ablation.md          ablation side by side, the pre-registered verdicts,
                                    the Objective 6 answer
  docs/results_classification.md    classification by zone group, side by side
  docs/results_day8_comparison.json every compared number, plus the configuration each
                                    result set was produced with (the parity record that
                                    tests/test_results_parity.py checks)
  docs/figures/day8_gain_comparison.png

The verdicts apply the criteria written in docs/preregistration_day8.md (committed before
the evaluation) mechanically; nothing here chooses them. Per-dataset detail reports are
written by each ml module to docs/*_deterministic.md and docs/stochastic/.
No model is trained or run here.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import config
from db.repo import Repo, from_epoch_ms
from ml.forecast import crossing_events
from shifts import ShiftSchedule

ROOT = config.CONFIG_PATH.parent
MIN_MS = 60_000


# ---------------------------------------------------------------- inputs


def result_paths(base: Path) -> dict[str, Path]:
    """The saved outputs of one dataset (same file names in each directory)."""
    ml = config.get_ml_config()
    return {"features": base / Path(ml["features_path"]).name,
            "forecast": base / Path(ml["metrics_path"]).name,
            "classification": base / Path(ml["classification"]["metrics_path"]).name,
            "ablation": base / Path(ml["ablation"]["metrics_path"]).name,
            "anomaly": base / Path(ml["anomaly"]["metrics_path"]).name}


def load_results(base: Path) -> dict[str, Any]:
    """Every metrics JSON of one dataset; raises if one is missing (fail loudly)."""
    p = result_paths(base)
    missing = [k for k, v in p.items() if k != "features" and not v.exists()]
    if missing:
        raise FileNotFoundError(f"{base}: missing {missing}")
    return {k: json.loads(v.read_text(encoding="utf-8")) for k, v in p.items() if k != "features"}


# ---------------------------------------------------------------- prediction 4b


def schedule_boundaries_ms(start_ms: int, end_ms: int, changeover_min: float) -> np.ndarray:
    """Shift start + changeover, shift end, break start, break end (epoch ms), inside the window."""
    sched = ShiftSchedule(config.get_shifts())
    start, end = pd.Timestamp(start_ms, unit="ms", tz="UTC"), pd.Timestamp(end_ms, unit="ms", tz="UTC")
    out = []
    for inst in sched.instances_between(start.to_pydatetime(), end.to_pydatetime()):
        out += [inst.start + timedelta(minutes=changeover_min), inst.end]
        out += [b.start for b in inst.breaks] + [b.end for b in inst.breaks]
    ms = np.array(sorted(int(t.timestamp() * 1000) for t in out))
    return ms[(ms >= start_ms) & (ms < end_ms)]


def crossing_clock_share(features: pd.DataFrame, zones: list[str], threshold: float, sustain: int,
                         window_min: float, changeover_min: float) -> dict[str, Any]:
    """Share of sustained upward crossings that fall within +/- window of a schedule boundary.

    Each zone's 1-minute noise series is put on a contiguous grid (gaps break runs), and
    events are found with ml.forecast.crossing_events (below for `sustain` minutes, then
    at or above for `sustain` minutes).
    """
    events_ms = []
    for z in zones:
        g = features[features["zone_id"].astype(str) == z]
        grid = np.arange(g["t_ms"].min(), g["t_ms"].max() + MIN_MS, MIN_MS)
        s = g.set_index("t_ms")["noise_dba"].reindex(grid)
        events_ms += [int(grid[i]) for i in crossing_events(s.reset_index(drop=True), threshold, sustain)]
    if not events_ms:
        return {"n_events": 0, "n_near_boundary": 0, "share_near_boundary": None}
    ev = np.array(events_ms)
    b = schedule_boundaries_ms(int(features["t_ms"].min()), int(features["t_ms"].max()) + MIN_MS, changeover_min)
    near = np.array([np.min(np.abs(b - e)) <= window_min * MIN_MS if len(b) else False for e in ev])
    return {"n_events": int(len(ev)), "n_near_boundary": int(near.sum()), "share_near_boundary": float(near.mean())}


# ---------------------------------------------------------------- availability through the MQTT path


def machine_availability(det_db: Path, sto_db: Path) -> dict[str, Any]:
    """Per machine: share of the deterministic dataset's running readings at which the
    stochastic dataset's machine was also running (same timestamps; both through ingest).

    The deterministic machine runs exactly when scheduled, so this is the realised
    availability of the stochastic machine, measured from the stored telemetry.
    """
    out = {}
    with Repo(det_db, read_only=True) as rd, Repo(sto_db, read_only=True) as rs:
        span = rd.get_data_span()
        end = from_epoch_ms(int(pd.Timestamp(span[1]).value // 1_000_000) + 1)
        for m in config.get_machines():
            d = pd.DataFrame(rd.get_machine_status(m["machine_id"], span[0], end))[["ts", "running"]]
            s = pd.DataFrame(rs.get_machine_status(m["machine_id"], span[0], end))[["ts", "running"]]
            j = d.merge(s, on="ts", suffixes=("_det", "_sto"))
            on = j["running_det"] == 1
            out[m["machine_id"]] = {"zone_id": m["zone_id"], "n_scheduled_readings": int(on.sum()),
                                    "availability": float(j.loc[on, "running_sto"].mean()) if on.any() else None}
    return out


# ---------------------------------------------------------------- verdicts


def _verdict(n_true: int, n: int) -> str:
    return "held" if n_true == n else ("partly held" if n_true > 0 else "FAILED")


def evaluate_predictions(det: dict[str, Any], sto: dict[str, Any], clock: dict[str, dict[str, Any]],
                         lead_group: str) -> list[dict[str, Any]]:
    """Apply the pre-registered criteria (docs/preregistration_day8.md, Section 6)."""
    fd, fs = det["forecast"]["targets"], sto["forecast"]["targets"]

    def chosen_skill(f: dict[str, Any], key: str) -> float:
        return f[key]["metrics"][f[key]["chosen"]]["skill_vs_persistence"]

    def seasonal(f: dict[str, Any], key: str) -> float:
        return f[key]["metrics"]["seasonal"]["skill_vs_persistence"]

    out = []
    m30_d, m30_s = chosen_skill(fd, "noise_dba_h30"), chosen_skill(fs, "noise_dba_h30")
    out.append({"id": 1, "prediction": "Forecasting skill falls (saved noise model, t+30)",
                "deterministic": f"{m30_d:+.3f} ({fd['noise_dba_h30']['chosen']})",
                "stochastic": f"{m30_s:+.3f} ({fs['noise_dba_h30']['chosen']})",
                "verdict": _verdict(int(m30_s < m30_d), 1)})
    parts, ok = [], 0
    for key in ("noise_dba_h30", "noise_dba_h60"):
        drop_seas = seasonal(fd, key) - seasonal(fs, key)
        drop_model = chosen_skill(fd, key) - chosen_skill(fs, key)
        ok += drop_seas > drop_model
        parts.append(f"t+{key[-2:]}: seasonal {seasonal(fd, key):+.3f}→{seasonal(fs, key):+.3f} "
                     f"(drop {drop_seas:+.3f}) vs model drop {drop_model:+.3f}")
    out.append({"id": 2, "prediction": "The seasonal baseline falls further than the model (both horizons)",
                "deterministic": "—", "stochastic": "; ".join(parts), "verdict": _verdict(ok, 2)})
    ad, as_ = det["ablation"]["forecasting_ns"], sto["ablation"]["forecasting_ns"]
    parts_d, parts_s, ok = [], [], 0
    for key in ("noise_dba_h30", "noise_dba_h60"):
        d = ad[key]["differences"]["model_3_ns_vs_model_1_ns"]
        s = as_[key]["differences"]["model_3_ns_vs_model_1_ns"]
        ok += (s["point"] > d["point"]) and (s["lo"] > 0)
        parts_d.append(f"t+{key[-2:]}: {d['point']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]")
        parts_s.append(f"t+{key[-2:]}: {s['point']:+.3f} [{s['lo']:+.3f}, {s['hi']:+.3f}]")
    out.append({"id": 3, "prediction": "PRIMARY: machine + worker gain with no target history and no schedule "
                                       "rises (point above deterministic AND CI excludes 0, both horizons)",
                "deterministic": "; ".join(parts_d), "stochastic": "; ".join(parts_s), "verdict": _verdict(ok, 2)})
    cd = det["classification"]["runs"]["crossing_h30"]
    cs = sto["classification"]["runs"]["crossing_h30"]
    pd_ = cd["by_zone_group"][lead_group][cd["chosen"]]["prevalence"]
    ps = cs["by_zone_group"][lead_group][cs["chosen"]]["prevalence"]
    sd, ss = clock["deterministic"]["share_near_boundary"], clock["stochastic"]["share_near_boundary"]
    ok = int(ps > pd_) + int(ss is not None and sd is not None and ss < sd)
    out.append({"id": 4, "prediction": "Borderline-zone crossings more frequent (a) and less clock-driven (b)",
                "deterministic": f"(a) prevalence {pd_:.1%}; (b) {sd:.1%} of {clock['deterministic']['n_events']} "
                                 "events near a boundary",
                "stochastic": f"(a) prevalence {ps:.1%}; (b) {ss:.1%} of {clock['stochastic']['n_events']} events "
                              "near a boundary",
                "verdict": _verdict(ok, 2)})
    return out


# ---------------------------------------------------------------- parity


def parity_record(det: dict[str, Any], sto: dict[str, Any]) -> dict[str, Any]:
    """The model configuration each result set was produced with (must be identical)."""
    def cfg(r: dict[str, Any]) -> dict[str, Any]:
        f, c, a, an = r["forecast"], r["classification"], r["ablation"], r["anomaly"]
        return {"forecast": {"hyperparameters": f["hyperparameters"], "split_rule": f["split_rule"],
                             "splits": f["splits"], "n_features": f["n_features"],
                             "exceedance_threshold_db": f["exceedance_threshold_db"]},
                "classification": {k: c[k] for k in ("grids", "random_seed", "max_false_positive_rate",
                                                     "threshold_db", "split_rule", "splits", "n_features",
                                                     "zone_groups")},
                "ablation": {k: a[k] for k in ("model", "grid", "random_seed", "bootstrap", "feature_sets",
                                               "nl_feature_sets", "ns_feature_sets", "split_rule", "splits",
                                               "history", "max_false_positive_rate")},
                "anomaly": {k: an[k] for k in ("settings", "random_seed")}}
    d, s = cfg(det), cfg(sto)
    return {"deterministic": d, "stochastic": s, "identical": d == s,
            "differences": sorted(k for k in d if d[k] != s[k])}


# ---------------------------------------------------------------- tables


def _f(v: Any, fmt: str = "+.3f") -> str:
    return "n/a" if v is None else format(v, fmt)


def _ci(iv: dict[str, Any]) -> str:
    if iv.get("lo") is None:
        return "undefined"
    tag = "✓" if iv["lo"] > 0 else ("✗" if iv["hi"] < 0 else "~")
    return f"{iv['point']:+.3f} [{iv['lo']:+.3f}, {iv['hi']:+.3f}] {tag}"


def forecast_table(det: dict[str, Any], sto: dict[str, Any]) -> list[str]:
    L = ["| Target | h | Persistence MAE (det → sto) | Seasonal skill (det → sto) | RF skill (det → sto) | "
         "XGBoost skill (det → sto) | Saved model (det / sto) |", "|---|---|---|---|---|---|---|"]
    for key in det["forecast"]["targets"]:
        d, s = det["forecast"]["targets"][key], sto["forecast"]["targets"][key]
        t, h = key.rsplit("_h", 1)
        cell = lambda m: f"{d['metrics'][m]['skill_vs_persistence']:+.3f} → **{s['metrics'][m]['skill_vs_persistence']:+.3f}**"  # noqa: E731
        L.append(f"| {t} | {h} | {d['metrics']['persistence']['mae']:.2f} → {s['metrics']['persistence']['mae']:.2f} | "
                 f"{cell('seasonal')} | {cell('random_forest')} | {cell('xgboost')} | {d['chosen']} / {s['chosen']} |")
    return L


def ablation_rows(det: dict[str, Any], sto: dict[str, Any]) -> list[str]:
    conds = [("forecasting", "model_3_vs_model_1", "with target lags (Model 3 − Model 1)"),
             ("forecasting_nl", "model_3_nl_vs_model_1_nl", "no target lags (3-NL − 1-NL)"),
             ("forecasting_ns", "model_3_ns_vs_model_1_ns", "no lags, no timetable (3-NS − 1-NS)")]
    L = ["| Target | h | Condition | Base skill (det → sto) | **Gain, deterministic** [95% CI] | "
         "**Gain, stochastic** [95% CI] |", "|---|---|---|---|---|---|"]
    for key in det["ablation"]["forecasting_nl"]:
        t, h = key.rsplit("_h", 1)
        for block, pair, label in conds:
            base = pair.split("_vs_")[1]
            d, s = det["ablation"][block][key], sto["ablation"][block][key]
            L.append(f"| {t} | {h} | {label} | {d['test'][base]['skill_vs_persistence']:+.3f} → "
                     f"{s['test'][base]['skill_vs_persistence']:+.3f} | {_ci(d['differences'][pair])} | "
                     f"**{_ci(s['differences'][pair])}** |")
    return L


def ns_zone_rows(det: dict[str, Any], sto: dict[str, Any]) -> list[str]:
    L = ["| Target | h | Zone | Gain 3-NS − 1-NS, deterministic | Gain, stochastic |", "|---|---|---|---|---|"]
    for key in ("noise_dba_h30", "noise_dba_h60"):
        t, h = key.rsplit("_h", 1)
        dz, sz = det["ablation"]["forecasting_ns"][key]["by_zone"], sto["ablation"]["forecasting_ns"][key]["by_zone"]
        for z in sorted(set(dz) | set(sz)):
            dd = dz.get(z, {}).get("differences", {}).get("model_3_ns_vs_model_1_ns")
            ss = sz.get(z, {}).get("differences", {}).get("model_3_ns_vs_model_1_ns")
            L.append(f"| {t} | {h} | {z} | {_ci(dd) if dd else 'n/a'} | {_ci(ss) if ss else 'n/a'} |")
    return L


def classification_rows(det: dict[str, Any], sto: dict[str, Any]) -> list[str]:
    groups = det["classification"]["zone_groups"]
    lead = next(g for g in groups if g.get("lead_zone"))
    names = [(f"{lead['name']}:{lead['lead_zone']}", f"{lead['lead_zone']} (lead)")] + [
        (g["name"], f"{g['name']} ({', '.join(g['zones'])})") for g in groups]
    L = ["| Variant | h | Group | Prevalence (det → sto) | PR-AUC (det → sto) | Lift (det → sto) | "
         "FPR at threshold (det → sto) |", "|---|---|---|---|---|---|---|"]
    for key in sorted(det["classification"]["runs"], key=lambda k: (k.startswith("above"), k)):
        variant, h = key.rsplit("_h", 1)
        rd, rs = det["classification"]["runs"][key], sto["classification"]["runs"][key]
        for name, label in names:
            d, s = rd["by_zone_group"][name][rd["chosen"]], rs["by_zone_group"][name][rs["chosen"]]
            if d["n"] == 0 and s["n"] == 0:
                continue
            lift = lambda x: "n/a" if x["pr_auc"] is None else f"{x['pr_auc'] / x['prevalence']:.2f}×"  # noqa: E731
            L.append(f"| {variant} | {h} | {label} | {_f(d['prevalence'], '.1%')} → **{_f(s['prevalence'], '.1%')}** | "
                     f"{_f(d['pr_auc'], '.3f')} → **{_f(s['pr_auc'], '.3f')}** | {lift(d)} → **{lift(s)}** | "
                     f"{_f(d['fpr'], '.3f')} → {_f(s['fpr'], '.3f')} |")
        dp, sp = rd["test"][rd["chosen"]], rs["test"][rs["chosen"]]
        L.append(f"| {variant} | {h} | *pooled (all zones)* | {_f(dp['prevalence'], '.1%')} → {_f(sp['prevalence'], '.1%')} | "
                 f"{_f(dp['pr_auc'], '.3f')} → {_f(sp['pr_auc'], '.3f')} | — | {_f(dp['fpr'], '.3f')} → "
                 f"{_f(sp['fpr'], '.3f')} |")
    return L


# ---------------------------------------------------------------- figure


def write_gain_figure(det: dict[str, Any], sto: dict[str, Any], path: Path) -> Path:
    """Machine + worker gain per condition, deterministic vs stochastic, 95% bars (noise)."""
    import matplotlib.pyplot as plt

    import figstyle

    figstyle.apply()
    conds = [("forecasting", "model_3_vs_model_1", "with target lags"),
             ("forecasting_nl", "model_3_nl_vs_model_1_nl", "no target lags"),
             ("forecasting_ns", "model_3_ns_vs_model_1_ns", "no lags,\nno timetable")]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), sharey=True)
    for ax, key in zip(axes, ("noise_dba_h30", "noise_dba_h60")):
        x = np.arange(len(conds))
        for k, (res, label, colour, hatch) in enumerate(((det, "deterministic simulator", "#BDBDBD", "//"),
                                                         (sto, "stochastic stoppages", "#D55E00", ""))):
            iv = [res["ablation"][b][key]["differences"][p] for b, p, _ in conds]
            pts = np.array([v["point"] for v in iv])
            err = np.array([pts - [v["lo"] for v in iv], [v["hi"] for v in iv] - pts])
            ax.bar(x + (k - 0.5) * 0.38, pts, 0.38, yerr=err, capsize=4, color=colour, hatch=hatch,
                   edgecolor="black", linewidth=0.6, label=label)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([c[2] for c in conds])
        ax.set_title(f"Noise forecast, t+{key[-2:]} min")
    axes[0].set_ylabel("Skill gain from machine + worker data\n(95% block-bootstrap CI)")
    axes[0].legend(loc="upper left", fontsize=11)
    figstyle.simulated_badge(axes[1])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------- documents


def _text(name: str) -> str:
    p = ROOT / "docs" / "text" / name
    return p.read_text(encoding="utf-8").strip() if p.exists() else f"*({name} not written yet.)*"


def write_ablation_doc(cmp: dict[str, Any], det: dict[str, Any], sto: dict[str, Any], path: Path) -> None:
    av = cmp["availability_mqtt"]
    L = ["# Model 1 / 2 / 3 ablation — deterministic vs stochastic machine behaviour", "",
         "**SIMULATED DATA.** Generated by `ml/compare_datasets.py` from the saved outputs of the **same** ML "
         "pipeline run on two datasets that differ only in the simulator: the Day 1–7 deterministic dataset "
         "(machines run exactly on the timetable) and the Day 8 stochastic dataset (unplanned stops, uneven shift "
         "starts, variable load, a starvation dependency). Parameters and predictions were **pre-registered** in "
         "`docs/preregistration_day8.md` and committed before this evaluation ran. Model configuration identical "
         f"for both: **{'yes' if cmp['parity']['identical'] else 'NO — ' + ', '.join(cmp['parity']['differences'])}** "
         "(`tests/test_results_parity.py`).", "",
         "Per-dataset detail: `docs/results_ablation_deterministic.md`, `docs/stochastic/results_ablation.md`.", "",
         "## Objective 6 — the answer", "", _text("day8_objective6.md"), "",
         "## Pre-registered predictions", "",
         "Criteria exactly as written in `docs/preregistration_day8.md` §6, applied mechanically.", "",
         "| # | Prediction | Deterministic | Stochastic | **Verdict** |", "|---|---|---|---|---|"]
    for p in cmp["predictions"]:
        L.append(f"| {p['id']} | {p['prediction']} | {p['deterministic']} | {p['stochastic']} | **{p['verdict']}** |")
    L += ["", _text("day8_predictions.md"), "",
          "## Machine availability, measured through the full MQTT and ingest path", "",
          "Share of the deterministic dataset's running readings (= scheduled time) at which the same machine was "
          "running in the stochastic dataset, same timestamps. Cross-checks the pre-registration's simulator-only "
          "measurement.", "", "| Machine | Zone | Readings | Availability |", "|---|---|---|---|"]
    for m, a in av.items():
        L.append(f"| {m} | {a['zone_id']} | {a['n_scheduled_readings']:,} | {a['availability']:.3f} |")
    L += ["", "## Forecasting skill and baselines (test day)", "",
          "Skill = 1 − MAE/MAE_persistence on that dataset's own test rows; persistence itself changes between "
          "datasets, so its MAE is shown too.", ""] + forecast_table(det, sto)
    L += ["", "## Machine + worker gain, every condition, side by side", "",
          "✓ = 95% interval excludes 0 in favour of adding machine and worker data; ~ = not distinguishable; "
          "✗ = excludes 0 against.", ""] + ablation_rows(det, sto)
    L += ["", "Figure: `docs/figures/day8_gain_comparison.png`.", "",
          "### Per zone, no lags and no timetable (noise)", ""] + ns_zone_rows(det, sto) + [""]
    path.write_text("\n".join(L), encoding="utf-8")


def write_classification_doc(cmp: dict[str, Any], det: dict[str, Any], sto: dict[str, Any], path: Path) -> None:
    clock = cmp["crossing_clock"]
    L = ["# Noise exceedance classification — deterministic vs stochastic machine behaviour", "",
         "**SIMULATED DATA.** Generated by `ml/compare_datasets.py`. The same classifier pipeline "
         "(`ml/classify_exceedance.py`, same grids, seed, split rule and threshold policy) was run on both datasets. "
         "Label: the measured 1-minute L_Aeq of the same zone at t+h vs 90 dBA, a future measurement, not "
         "circular. Prevalence is shown beside every metric; PR-AUC leads.", "",
         "Per-dataset detail (pooled tables, tuning, per-zone appendix, PR/ROC curves): "
         "`docs/results_classification_deterministic.md`, `docs/stochastic/results_classification.md`.", "",
         "## Discussion", "", _text("day8_classification.md"), "",
         "## By zone group, side by side (lead zone first, `crossing` first)", "",
         "Groups from `ml.classification.zone_groups`: borderline Z2/Z3/Z6 (hover near 90 dBA; the case that "
         "matters), always-loud Z1/Z4/Z5, office Z7 (never exceeds). Lift = PR-AUC ÷ prevalence (1.0 = chance).",
         ""] + classification_rows(det, sto)
    L += ["", "## How clock-driven are the crossings? (pre-registered prediction 4b)", "",
          f"Sustained upward 90 dBA crossings in {', '.join(cmp['borderline_zones'])} over all 7 days (below for "
          f"{cmp['settings']['event_sustain_min']} min, then at or above for {cmp['settings']['event_sustain_min']} "
          f"min), and the share within ±{cmp['settings']['clock_window_min']} min of a schedule boundary.", "",
          "| Dataset | Crossing events | Near a boundary | Share |", "|---|---|---|---|"]
    for name in ("deterministic", "stochastic"):
        c = clock[name]
        L.append(f"| {name} | {c['n_events']:,} | {c['n_near_boundary']:,} | {_f(c['share_near_boundary'], '.1%')} |")
    L.append("")
    path.write_text("\n".join(L), encoding="utf-8")


def run(det_dir: Path, sto_dir: Path, det_db: Path, sto_db: Path) -> dict[str, Any]:
    """Compute every comparison and verdict; write the documents, figure and JSON record."""
    c = config.get_ml_config()["comparison"]
    det, sto = load_results(det_dir), load_results(sto_dir)
    groups = config.get_ml_config()["classification"]["zone_groups"]
    lead = next(g for g in groups if g.get("lead_zone"))
    threshold = float(det["classification"]["threshold_db"])
    clock = {}
    for name, base in (("deterministic", det_dir), ("stochastic", sto_dir)):
        feats = pd.read_parquet(result_paths(base)["features"], columns=["t_ms", "zone_id", "noise_dba"])
        clock[name] = crossing_clock_share(feats, lead["zones"], threshold, int(c["event_sustain_min"]),
                                           float(c["clock_window_min"]), float(c["changeover_min"]))
    cmp = {"settings": {k: c[k] for k in ("clock_window_min", "changeover_min", "event_sustain_min")},
           "borderline_zones": lead["zones"], "crossing_clock": clock,
           "availability_mqtt": machine_availability(det_db, sto_db),
           "predictions": evaluate_predictions(det, sto, clock, lead["name"]),
           "parity": parity_record(det, sto)}
    (ROOT / "docs" / "results_day8_comparison.json").write_text(json.dumps(cmp, indent=2), encoding="utf-8")
    write_ablation_doc(cmp, det, sto, ROOT / "docs" / "results_ablation.md")
    write_classification_doc(cmp, det, sto, ROOT / "docs" / "results_classification.md")
    write_gain_figure(det, sto, ROOT / config.get_paths()["figures_dir"] / "day8_gain_comparison.png")
    return cmp


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    c = config.get_ml_config()["comparison"]
    p = argparse.ArgumentParser(prog="python -m ml.compare_datasets", description=__doc__.split("\n")[0])
    p.add_argument("--det-dir", type=Path, default=ROOT / c["deterministic_dir"])
    p.add_argument("--sto-dir", type=Path, default=ROOT / c["stochastic_dir"])
    p.add_argument("--det-db", type=Path, default=ROOT / c["deterministic_db"])
    p.add_argument("--sto-db", type=Path, default=ROOT / c["stochastic_db"])
    args = p.parse_args(argv)
    cmp = run(args.det_dir, args.sto_dir, args.det_db, args.sto_db)
    for pr in cmp["predictions"]:
        print(f"P{pr['id']}: {pr['verdict']}")
    print("parity identical:", cmp["parity"]["identical"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
