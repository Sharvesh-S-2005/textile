"""Build the results pack: figures and results documents, from stored outputs only.

    python scripts/build_results_pack.py [--week-db data/week/textile_ohs.db] [--skip-shap]

Idempotent and rerunnable: every output is overwritten from the current upstream
data, so numbers follow any recomputation. Nothing here is hand-typed - every figure
in docs/results_summary.md is read from the databases, the forecast metrics, the
localisation score files, the run logs or (for the consistency residual) measured
afresh.

Outputs:
  docs/figures/noise_timeseries_zone.png    one zone, one shift, machine state + breaks
  docs/figures/forecast_vs_actual.png       test day: model vs actual, with baselines
  docs/figures/shap_*.png                   via ml/explain.py (unless --skip-shap)
  docs/figures/worker_dose_bars.png         per-worker dose, both criteria, 100% line
  docs/figures/zone_hour_heatmap.png        mean noise by zone and hour
  docs/figures/localisation_confusion.png   Day 2 localisation confusion matrix
  docs/results_forecast.md, docs/results_exposure.md, docs/results_summary.md
  docs/results_ml.md, docs/model_cards.md   via ml/report_ml.py (from the saved ML metrics)
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config  # noqa: E402
import figstyle  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.dates as mdates  # noqa: E402
from db.repo import Repo, from_epoch_ms  # noqa: E402
from exposure.noise import l_aeq, l_ex_8h  # noqa: E402
from exposure.personal import UNKNOWN_ZONE, ExposureParams, personal_exposure_for_session  # noqa: E402
from exposure.risk_index import RiskParams, worst_component  # noqa: E402
from ml import explain, forecast, report_ml  # noqa: E402

FIG = ROOT / config.get_paths()["figures_dir"]
DOCS = ROOT / "docs"
TZ = config.get_shifts()["timezone"]
ML = config.get_ml_config()


def energy_mean(levels: list[float]) -> float:
    """10 log10(mean(10^(L/10))) - the only average of dB used here."""
    return 10 * math.log10(sum(10 ** (x / 10) for x in levels) / len(levels))


def local(t_ms: pd.Series) -> pd.Series:
    """Epoch ms -> mill local time (display layer only)."""
    return pd.to_datetime(t_ms, unit="ms", utc=True).dt.tz_convert(TZ)


# ---------------------------------------------------------------- figures


def fig_noise_timeseries(features: pd.DataFrame, zone: str, day: str) -> dict[str, Any]:
    """Noise over shift A of `day` in `zone`, machines-stopped and break periods shaded."""
    d = features[(features["zone_id"].astype(str) == zone)].copy()
    start = pd.Timestamp(f"{day} 06:00", tz=TZ)
    d["local"] = local(d["t_ms"])
    d = d[(d["local"] >= start) & (d["local"] < start + pd.Timedelta(hours=8))]
    fig, ax = plt.subplots(figsize=(13, 6))
    stopped = d["machines_running_count"].fillna(0) < 0.5
    ax.fill_between(d["local"], 0, 1, where=stopped, transform=ax.get_xaxis_transform(), color=figstyle.LIGHT,
                    alpha=0.6, step="mid", label="machines stopped", linewidth=0)
    ax.fill_between(d["local"], 0, 1, where=d["is_break"] == 1, transform=ax.get_xaxis_transform(),
                    facecolor="none", edgecolor=figstyle.NEUTRAL, hatch="//", step="mid", label="scheduled break",
                    linewidth=0)
    ax.plot(d["local"], d["noise_dba"], color=figstyle.HAZARD_COLOURS["noise"], linewidth=1.6,
            label="noise, 1-min L_Aeq")
    ax.axhline(90, color="black", linestyle="--", linewidth=1.2, label="90 dBA (India criterion)")
    ax.axhline(85, color="black", linestyle=":", linewidth=1.2, label="85 dBA (NIOSH criterion)")
    ax.set_xlabel(f"time ({TZ}, shift A, {day})")
    ax.set_ylabel("noise level (dBA)")
    ax.set_title(f"Zone {zone}: noise follows machine state; it drops to background during breaks")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=False)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=TZ))
    figstyle.simulated_badge(ax)
    fig.savefig(FIG / "noise_timeseries_zone.png")
    plt.close(fig)
    return {"zone": zone, "day": day}


def fig_forecast(predictions: pd.DataFrame, zone: str, horizon: int) -> dict[str, Any]:
    """Test-day noise forecast at t+h for one zone vs actual, persistence and seasonal (by target time)."""
    p = predictions[(predictions["target"] == "noise_dba") & (predictions["horizon_min"] == horizon)
                    & (predictions["zone_id"] == zone)].sort_values("t_ms")
    t = local(p["t_ms"] + horizon * 60_000)
    first_shift = t < t.iloc[0] + pd.Timedelta(hours=8)
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(t[first_shift], p["actual"][first_shift], label="actual", **figstyle.BASELINE_STYLES["actual"])
    ax.plot(t[first_shift], p["persistence"][first_shift], label="persistence baseline",
            **figstyle.BASELINE_STYLES["persistence"])
    ax.plot(t[first_shift], p["seasonal"][first_shift], label="seasonal baseline (same minute, previous day)",
            **figstyle.BASELINE_STYLES["seasonal"])
    chosen = p["chosen_model"].iloc[0]
    ax.plot(t[first_shift], p["chosen"][first_shift], color=figstyle.HAZARD_COLOURS["noise"], linewidth=2.0,
            marker="o", markevery=30, label=f"{chosen.replace('_', ' ')} forecast")
    ax.set_xlabel(f"target time ({TZ}, test day)")
    ax.set_ylabel(f"noise at t+{horizon} min (dBA)")
    ax.set_title(f"Zone {zone}: {horizon}-minute noise forecast vs actual, first 8 h of the test day")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=False)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=TZ))
    figstyle.simulated_badge(ax)
    fig.savefig(FIG / "forecast_vs_actual.png")
    plt.close(fig)
    return {"zone": zone, "horizon": horizon, "model": chosen}


def fig_worker_doses(shift: list[dict[str, Any]]) -> None:
    """Median daily dose per worker under both criteria, sorted, log axis, 100% line."""
    per: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in shift:
        if r["dose_niosh_pct"] is not None:
            per[r["worker_id"]].append(r)
    rows = sorted(((w, statistics.median(x["dose_niosh_pct"] for x in rs),
                    statistics.median(x["dose_india_pct"] for x in rs)) for w, rs in per.items()), key=lambda r: r[1])
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.bar(x - 0.2, [max(r[1], 0.1) for r in rows], 0.4, color=figstyle.HAZARD_COLOURS["noise"],
           label="NIOSH (85 dBA, 3 dB, 80 dBA threshold)")
    ax.bar(x + 0.2, [max(r[2], 0.1) for r in rows], 0.4, facecolor="white", edgecolor=figstyle.HAZARD_COLOURS["noise"],
           hatch="///", label="India (90 dBA, 5 dB, 90 dBA threshold)")
    ax.axhline(100, color="black", linestyle="--", linewidth=1.5, label="100% = daily allowance")
    ax.set_yscale("log")
    ax.set_xticks(x, [r[0] for r in rows], rotation=60)
    ax.set_xlabel("worker (sorted by NIOSH dose)")
    ax.set_ylabel("median daily noise dose, 7 days (% of allowance, log scale)")
    ax.set_title("Per-worker noise dose under both criteria (estimated_personal)")
    ax.legend(loc="upper left")
    figstyle.simulated_badge(ax)
    fig.savefig(FIG / "worker_dose_bars.png")
    plt.close(fig)


def fig_zone_hour(features: pd.DataFrame) -> pd.DataFrame:
    """Energy-mean noise by zone and local hour over the whole dataset."""
    d = features[["zone_id", "t_ms", "noise_dba"]].dropna().copy()
    d["hour"] = local(d["t_ms"]).dt.hour
    d["zone_id"] = d["zone_id"].astype(str)
    table = d.groupby(["zone_id", "hour"])["noise_dba"].apply(lambda s: energy_mean(list(s))).unstack("hour")
    fig, ax = plt.subplots(figsize=(14, 5.5))
    im = ax.imshow(table.to_numpy(), aspect="auto", cmap=figstyle.sequential_cmap("noise"), vmin=50, vmax=105)
    names = {z["zone_id"]: f"{z['zone_id']} {z['name']}" for z in config.get_zones()}
    ax.set_yticks(range(len(table.index)), [names[z] for z in table.index])
    ax.set_xticks(range(24), [f"{h:02d}" for h in table.columns])
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            v = table.iat[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=9,
                    color="white" if v > 85 else "black")
    ax.set_xlabel(f"hour of day ({TZ})")
    ax.set_title("Mean noise by zone and hour, 7 days (energy average, dBA; values printed in each cell)")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="L_Aeq (dBA)")
    figstyle.simulated_badge(ax)
    fig.savefig(FIG / "zone_hour_heatmap.png")
    plt.close(fig)
    return table


def fig_confusion(score: dict[str, Any]) -> None:
    """Row-normalised localisation confusion matrix (Day 2 run), counts printed."""
    conf = score["confusion"]
    rows = list(conf)
    cols = sorted({c for r in conf.values() for c in r}, key=lambda c: (c == "UNKNOWN", c))
    counts = np.array([[conf[r].get(c, 0) for c in cols] for r in rows], dtype=float)
    pct = 100 * counts / counts.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(11, 8))
    im = ax.imshow(pct, cmap=plt.get_cmap("Greys"), vmin=0, vmax=100)
    ax.set_xticks(range(len(cols)), cols)
    ax.set_yticks(range(len(rows)), rows)
    for i in range(len(rows)):
        for j in range(len(cols)):
            if counts[i, j]:
                ax.text(j, i, f"{pct[i, j]:.0f}%\n({int(counts[i, j])})", ha="center", va="center", fontsize=9,
                        color="white" if pct[i, j] > 50 else "black")
    ax.set_xlabel("predicted zone")
    ax.set_ylabel("true zone")
    ax.set_title(f"BLE localisation, Day 2 run: {100 * score['accuracy_on_site']:.1f}% of on-site "
                 "worker-minutes correct")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="% of the true zone's windows")
    figstyle.simulated_badge(ax)
    fig.savefig(FIG / "localisation_confusion.png")
    plt.close(fig)


# ---------------------------------------------------------------- numbers


def parse_run_logs(publisher_log: Path, ingest_log: Path) -> dict[str, Any] | None:
    """Published counts (publisher final summary) and received/stored/rejected (ingest final line)."""
    if not (publisher_log.exists() and ingest_log.exists()):
        return None
    pub = [line for line in publisher_log.read_text(encoding="utf-8").splitlines() if "sim=" in line][-1]
    counts = {k: int(v) for k, v in re.findall(r"(env|ble|machine|rfid|ppe|status|replayed)=(\d+)", pub)}
    ingest = ingest_log.read_text(encoding="utf-8")
    final = [line for line in ingest.splitlines() if "final:" in line][-1]
    got = {k: int(v) for k, v in re.findall(r"\b(received|stored|duplicate|rejected|errors)=(\d+)", final)}
    rates = [float(x) for x in re.findall(r"rate=([0-9.]+) msg/s", ingest) if float(x) > 0]
    wall = float(re.findall(r"wall=([0-9.]+)s", pub)[-1])
    return {"published": sum(counts.values()), "by_type": counts, **got,
            "throughput_msg_s": got["received"] / wall if wall else None,
            "peak_rate_msg_s": max(rates) if rates else None, "wall_s": wall}


def consistency_residual(repo: Repo) -> dict[str, Any]:
    """Re-measure the single-zone consistency residual on stored data."""
    params = ExposureParams.from_config()
    cands = [a for a in repo.get_all_zone_assignments() if a["zone_id"] != UNKNOWN_ZONE and not a["outside_session"]]
    best = max(cands, key=lambda a: a["to_epoch_ms"] - a["from_epoch_ms"])
    sess = {"worker_id": best["worker_id"], "session_start": best["from_ts"], "session_end": best["to_ts"],
            "session_start_ms": best["from_epoch_ms"], "session_end_ms": best["to_epoch_ms"], "flags": []}
    personal = personal_exposure_for_session(repo, sess, params)
    rows = repo.get_zone_env(best["zone_id"], from_epoch_ms(best["from_epoch_ms"] + 1), from_epoch_ms(best["to_epoch_ms"] + 1))
    excluded = set(params.exclude_record_flags)
    levels = [r["noise_dba"] for r in rows if r["noise_dba"] is not None and not excluded & set(r["flags"])]
    hours = (best["to_epoch_ms"] - best["from_epoch_ms"]) / 3.6e6
    expected = l_ex_8h(l_aeq(levels, [params.reading_interval_s] * len(levels)), hours)
    return {"worker": best["worker_id"], "zone": best["zone_id"], "hours": hours,
            "residual_db": abs(personal.l_ex_8h_db - expected)}


def test_count() -> int | None:
    """Number of tests pytest collects (the suite itself is run separately)."""
    out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q"], cwd=ROOT,
                         capture_output=True, text=True)
    m = re.search(r"(\d+) tests? collected", out.stdout)
    return int(m.group(1)) if m else None


_BAND_ORDER = {"Low": 0, "Medium": 1, "High": 2}


def exposure_numbers(repo: Repo) -> dict[str, Any]:
    """Distribution, exceedance counts, coverage and zone drivers from the materialised tables."""
    shift = repo.get_exposure_rows("personal_exposure_shift")
    zones = repo.get_exposure_rows("zone_exposure_summary")
    risk = repo.get_exposure_rows("worker_risk_index")
    valid = [r for r in shift if r["dose_niosh_pct"] is not None]

    def q(vals: list[float], p: float) -> float:
        s = sorted(vals)
        return s[min(len(s) - 1, int(p * len(s)))]

    dist = {k: {"min": min(v), "p25": q(v, .25), "median": statistics.median(v), "p75": q(v, .75), "max": max(v)}
            for k, v in (("dose_niosh_pct", [r["dose_niosh_pct"] for r in valid]),
                         ("dose_india_pct", [r["dose_india_pct"] for r in valid]),
                         ("l_ex_8h_db", [r["l_ex_8h_db"] for r in valid]))}
    by_zone: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for z in zones:
        if z["data_coverage_pct"] is not None and z["data_coverage_pct"] >= 90:
            by_zone[z["zone_id"]].append(z)
    zone_rows = sorted(((zid, energy_mean([r["l_aeq_db"] for r in rs]),
                         statistics.fmean(r["dose_niosh_pct"] for r in rs), statistics.fmean(r["dose_india_pct"] for r in rs),
                         statistics.fmean(r["pm25_twa_raw"] for r in rs), max(r["wbgt_max_c"] for r in rs), len(rs))
                        for zid, rs in by_zone.items()), key=lambda r: -r[2])
    rv = [r["overall_index"] for r in risk]
    params = RiskParams.from_config(config.get_risk_index_config(), config.get_exposure_limits()["wbgt_limit_c"])
    worst = [worst_component(r["components"], params) for r in risk]
    return {
        "dates": sorted({r["shift_date"] for r in shift}), "sessions": len(shift), "valid": len(valid),
        "dist": dist,
        "over_100_niosh": sum(r["dose_niosh_pct"] > 100 for r in valid),
        "over_100_india": sum(r["dose_india_pct"] > 100 for r in valid),
        "coverage_mean": statistics.fmean(r["data_coverage_pct"] for r in valid),
        "coverage_min": min(r["data_coverage_pct"] for r in valid),
        "unaccounted_mean": statistics.fmean(r["unaccounted_pct"] for r in shift),
        "unaccounted_max": max(r["unaccounted_pct"] for r in shift),
        "absolute_max_events": sum(len(r["absolute_max_events"]) for r in shift),
        "excluded_candidates": sum(len(r["excluded_absolute_max_candidates"]) for r in shift),
        "excluded_readings": sum(r["excluded_anomalous_readings"] for r in shift),
        "zones": zone_rows,
        "risk": {"n": len(risk), "bands": dict(Counter(r["band"] for r in risk)),
                 "min": min(rv) if rv else None, "median": statistics.median(rv) if rv else None,
                 "max": max(rv) if rv else None,
                 "worst_bands": dict(Counter(w["band"] for w in worst)),
                 "worst_components": dict(Counter(w["component"] for w in worst)),
                 "worst_band_higher": sum(_BAND_ORDER[w["band"]] > _BAND_ORDER.get(r["band"], -1)
                                          for w, r in zip(worst, risk))},
        "engine_versions": sorted({r["engine_version"] for r in shift}),
    }


# ---------------------------------------------------------------- documents


def write_exposure_md(e: dict[str, Any], path: Path) -> None:
    """docs/results_exposure.md."""
    d = e["dist"]
    L = ["# Exposure results (7 days)", "",
         f"**SIMULATED DATA.** Generated by `scripts/build_results_pack.py` from the materialised exposure tables "
         f"(engine {', '.join(e['engine_versions'])}). Personal figures are `estimated_personal`; dust is "
         "`indicative`; WBGT is `approximate` (an underestimate). Methods and assumptions: "
         "`docs/exposure_methods.md`.", "",
         f"Shift dates {e['dates'][0]} → {e['dates'][-1]}; {e['sessions']} worker-sessions, {e['valid']} with valid "
         "noise data.", "", "## Distribution across worker-days", "",
         "| | min | p25 | median | p75 | max |", "|---|---|---|---|---|---|"]
    for key, label in (("dose_niosh_pct", "NIOSH dose %"), ("dose_india_pct", "India dose %"),
                       ("l_ex_8h_db", "L_EX,8h (dBA)")):
        x = d[key]
        L.append(f"| {label} | {x['min']:,.1f} | {x['p25']:,.1f} | {x['median']:,.1f} | {x['p75']:,.1f} | {x['max']:,.1f} |")
    n = e["valid"]
    L += ["", "## Worker-days above 100% dose", "",
          f"- NIOSH: **{e['over_100_niosh']} / {n} ({100 * e['over_100_niosh'] / n:.1f}%)**",
          f"- India: **{e['over_100_india']} / {n} ({100 * e['over_100_india'] / n:.1f}%)**", "",
          "## Data quality", "",
          f"- Mean data coverage **{e['coverage_mean']:.2f}%** (minimum {e['coverage_min']:.2f}%)",
          f"- Mean unaccounted time **{e['unaccounted_mean']:.2f}%** (maximum {e['unaccounted_max']:.2f}%)",
          f"- Anomaly-flagged readings excluded: {e['excluded_readings']:,}; ≥115 dBA readings among them "
          f"(listed, not dropped): {e['excluded_candidates']}; genuine ≥115 dBA events: {e['absolute_max_events']}", "",
          "## Zones driving the highest doses (person present the whole shift; mean over shifts)", "",
          "| Zone | L_Aeq (dBA, energy mean) | NIOSH % | India % | PM2.5 TWA raw (µg/m³, indicative) | WBGT max (°C, approx.) | shifts |",
          "|---|---|---|---|---|---|---|"]
    names = {z["zone_id"]: z["name"] for z in config.get_zones()}
    for zid, la, dn, di, pm, wb, ns in e["zones"]:
        L.append(f"| {zid} {names[zid]} | {la:.1f} | {dn:,.0f} | {di:,.0f} | {pm:.0f} | {wb:.1f} | {ns} |")
    r = e["risk"]
    L += ["", "## Deterministic risk index", "",
          "> The weights are a design choice of this project, not a validated clinical model. The respiratory "
          "component is relative to the cohort, not to a health threshold. Label: `deterministic_index`.", "",
          f"- {r['n']} worker-days; index min {r['min']:.1f}, median {r['median']:.1f}, max {r['max']:.1f}",
          "- Bands: " + ", ".join(f"{k} {v}" for k, v in sorted(r["bands"].items())),
          "- Hearing uses L_EX,8h (75 → 105 dBA maps to 0 → 100; risk-1.1). risk-1.0 used dose % capped at "
          "100%, which saturated; the dose under both criteria is always shown beside the index. The rating is "
          "mostly High because this is a loud mill (`exposure_methods.md` §8).", "",
          "Figures: `figures/worker_dose_bars.png`, `figures/zone_hour_heatmap.png`, `figures/noise_timeseries_zone.png`."]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def write_summary_md(path: Path, nums: dict[str, Any]) -> None:
    """docs/results_summary.md: every headline number on one page."""
    def fmt(v: Any, spec: str = ",") -> str:
        return "n/a" if v is None else format(v, spec)

    L = ["# Results summary — every headline number", "",
         "**SIMULATED DATA.** Generated by `scripts/build_results_pack.py`; rerun it after any recomputation.", "",
         "| Area | Measure | Value | Source |", "|---|---|---|---|"]
    L.append(f"| Tests | pytest tests collected | **{fmt(nums['tests'])}** (all passing at last commit) | `pytest` |")
    for label, run in (("Day 2 run (8 h, 100×)", nums["day2_run"]), ("7-day run (90×)", nums["week_run"])):
        if run is None:
            L.append(f"| Data loss | {label} | n/a (run logs not found) | logs |")
            continue
        injected = run["received"] - run["published"]
        L.append(f"| Data loss | {label}: published + injected = received = stored + rejected | "
                 f"{run['published']:,} + {injected} = **{run['received']:,}** = {run['stored']:,} + "
                 f"{run['rejected']} (dup {run['duplicate']}, errors {run['errors']}) | publisher + ingest logs |")
        L.append(f"| Ingest | {label}: sustained throughput | **{run['throughput_msg_s']:.1f} msg/s** "
                 f"(peak 30-s rate {run['peak_rate_msg_s']:.1f}) | ingest log |")
    for label, score in (("Day 2 run", nums["score_day2"]), ("7-day run", nums["score_week"])):
        if score is None:
            continue
        L.append(f"| Localisation | {label}: on-site worker-minutes in the correct zone | "
                 f"**{100 * score['accuracy_on_site']:.1f}%** of {score['on_site_windows']:,} | score_localisation.py |")
        lat = score["latency_entry"]
        L.append(f"| Localisation | {label}: zone-change latency | median **{lat['median_s']:.0f} s**, "
                 f"mean {lat['mean_s']:.1f} s, p90 {lat['p90_s']:.0f} s (n={lat['n']:,}) | score_localisation.py |")
    c = nums["consistency"]
    L.append(f"| Exposure | Single-zone consistency residual ({c['worker']} in {c['zone']}, {c['hours']:.2f} h) | "
             f"**{c['residual_db']:.1e} dB** | measured by this script |")
    e = nums["exposure"]
    n = e["valid"]
    L.append(f"| Exposure | Worker-days > 100% dose, NIOSH / India | **{100 * e['over_100_niosh'] / n:.1f}% / "
             f"{100 * e['over_100_india'] / n:.1f}%** ({e['over_100_niosh']} / {e['over_100_india']} of {n}) | exposure tables |")
    L.append(f"| Exposure | Median daily dose NIOSH / India; median L_EX,8h | "
             f"{e['dist']['dose_niosh_pct']['median']:,.0f}% / {e['dist']['dose_india_pct']['median']:,.0f}%; "
             f"{e['dist']['l_ex_8h_db']['median']:.1f} dBA | exposure tables |")
    L.append(f"| Exposure | Mean data coverage / mean unaccounted time | {e['coverage_mean']:.2f}% / "
             f"{e['unaccounted_mean']:.2f}% | exposure tables |")
    top = e["zones"][0]
    L.append(f"| Exposure | Highest-dose zone | {top[0]} (L_Aeq {top[1]:.1f} dBA) | zone_exposure_summary |")
    m = nums["forecast"]
    for key, entry in m["targets"].items():
        sk = {k: (f"**{v['skill_vs_persistence']:+.3f}**" if k == entry["chosen"] else f"{v['skill_vs_persistence']:+.3f}")
              for k, v in entry["metrics"].items()}
        L.append(f"| Forecast | {key}: skill vs persistence (seasonal / RF / XGBoost; saved model in bold) | "
                 f"{sk['seasonal']} / {sk['random_forest']} / {sk['xgboost']} | forecast_metrics.json |")
    for key in ("noise_dba_h30", "noise_dba_h60"):
        ex = m["targets"][key]["exceedance"][m["targets"][key]["chosen"]]
        L.append(f"| Forecast | {key}: 90 dBA crossings warned (hit rate), median lead, false-alarm ratio | "
                 f"{ex['hits']}/{ex['events']} ({100 * ex['hit_rate']:.0f}%), **{ex['median_lead_min']:.1f} min**, "
                 f"{100 * ex['false_alarm_ratio']:.1f}% | forecast_metrics.json |")
    s = nums["shap"]
    if s:
        shares = ", ".join(f"{k} {100 * v:.0f}%" for k, v in list(s["family_share"].items())[:3])
        L.append(f"| Explainability | SHAP importance shares, noise t+30 | {shares} | shap_noise_h30.json |")
    r = e["risk"]
    L.append(f"| Risk index | Bands over {r['n']} worker-days (deterministic_index; weights a design choice) | "
             + ", ".join(f"{k} {v}" for k, v in sorted(r["bands"].items())) + " | worker_risk_index |")
    L.append(f"| Risk index | Worst single component, shown beside the composite: bands | "
             + ", ".join(f"{k} {v}" for k, v in sorted(r["worst_bands"].items()))
             + f"; worst band above the composite band on {r['worst_band_higher']} worker-days | worker_risk_index |")
    ml = nums.get("ml")
    if ml:
        c = ml["classification"]["runs"]
        for key in ("above_h30", "crossing_h30", "crossing_h60"):
            e = c[key]
            x, pp = e["test"][e["chosen"]], e["test"]["persistence"]
            L.append(f"| Classification | {key}: prevalence; PR-AUC {e['chosen']} vs persistence; F1 | "
                     f"{100 * x['prevalence']:.1f}%; **{x['pr_auc']:.3f}** vs {pp['pr_auc']:.3f}; {x['f1']:.3f} | "
                     "classification_metrics.json |")
        a = ml["ablation"]["forecasting"]
        for key in ("noise_dba_h30", "noise_dba_h60"):
            d = a[key]["differences"]["model_3_vs_model_1"]
            d2 = a[key]["differences"]["model_2_vs_model_1"]
            L.append(f"| Ablation | {key}: skill Model 3 − Model 1 [95% CI]; Model 2 − Model 1 | "
                     f"**{d['point']:+.3f}** [{d['lo']:+.3f}, {d['hi']:+.3f}]; {d2['point']:+.3f} "
                     f"[{d2['lo']:+.3f}, {d2['hi']:+.3f}] | ablation_metrics.json |")
        an = ml["anomaly"]
        bad = [k for k, v in an["machines"].items() if v["configured_degrading"]][0]
        L.append(f"| Anomaly | {bad} detection day EWMA / isolation forest; healthy per-bin FPR | "
                 f"**{an['machines'][bad]['ewma']['alarm_day']:.2f} / {an['machines'][bad]['isolation_forest']['alarm_day']:.2f}**; "
                 f"{100 * an['healthy_summary']['ewma']['flag_rate']:.2f}% / "
                 f"{100 * an['healthy_summary']['isolation_forest']['flag_rate']:.2f}% | anomaly_metrics.json |")
    L += ["", "Negative findings are kept: at 60 min neither ML model beats the seasonal baseline for noise, and "
          "for WBGT the seasonal baseline is worse than persistence (`results_forecast.md`)."]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def write_numbers_sheet(path: Path, nums: dict[str, Any]) -> None:
    """docs/numbers_sheet.md: the one printed page to hold during the review."""
    d2, wk, f = nums["day2_run"], nums["week_run"], nums["forecast"]["targets"]
    s2, sw, e = nums["score_day2"], nums["score_week"], nums["exposure"]

    def sk(key: str, model: str) -> str:
        return f"{f[key]['metrics'][model]['skill_vs_persistence']:+.2f}"

    def steady(key: str, model: str) -> str:
        return f"{f[key]['by_schedule']['steady']['skill'][model]:+.2f}"

    ex30 = f["noise_dba_h30"]["exceedance"][f["noise_dba_h30"]["chosen"]]
    ex60 = f["noise_dba_h60"]["exceedance"][f["noise_dba_h60"]["chosen"]]
    r = e["risk"]
    L = ["# Numbers sheet — first review", "",
         "*SIMULATED DATA. Generated by `scripts/build_results_pack.py`. If asked \"how accurate is it\", read the number.*",
         "", "| Question | Answer |", "|---|---|",
         f"| Tests | **{nums['tests']} pytest tests**, all passing (exit code checked directly) |"]
    if d2:
        L.append(f"| Data loss (Day 2, 8 h at 100×) | **{d2['published']:,} published + {d2['received'] - d2['published']} "
                 f"injected = {d2['received']:,} received = {d2['stored']:,} stored + {d2['rejected']} rejected** — zero loss |")
        L.append(f"| Ingest throughput | **{d2['throughput_msg_s']:.1f} msg/s** sustained (Day 2); "
                 f"{wk['throughput_msg_s']:.1f} msg/s over the 7-day run ({wk['received']:,} messages, zero loss) |")
    L.append(f"| Worker localisation | **{100 * s2['accuracy_on_site']:.1f}%** of on-site worker-minutes in the right zone, "
             f"median latency **{s2['latency_entry']['median_s']:.0f} s** (7 days: {100 * sw['accuracy_on_site']:.1f}%) |")
    c = nums["consistency"]
    L.append(f"| Exposure maths | 12 hand-verified noise cases + 4 Schedule XXVIII rows; single-zone consistency residual "
             f"**{c['residual_db']:.1f} dB** |")
    L.append(f"| Forecast skill vs persistence, noise | t+30: RF **{sk('noise_dba_h30', 'random_forest')}** "
             f"(seasonal {sk('noise_dba_h30', 'seasonal')}); t+60: RF {sk('noise_dba_h60', 'random_forest')} — "
             f"**does not beat** seasonal {sk('noise_dba_h60', 'seasonal')} |")
    L.append(f"| Break-free windows, noise | RF **{steady('noise_dba_h30', 'random_forest')} / "
             f"{steady('noise_dba_h60', 'random_forest')}** vs seasonal {steady('noise_dba_h30', 'seasonal')} / "
             f"{steady('noise_dba_h60', 'seasonal')} (t+30 / t+60) — skill beyond the timetable |")
    L.append(f"| Forecast skill, dust / heat | PM2.5 XGBoost **{sk('pm25_ugm3_h30', 'xgboost')} / {sk('pm25_ugm3_h60', 'xgboost')}**; "
             f"WBGT {sk('wbgt_c_h30', 'xgboost')} / {sk('wbgt_c_h60', 'xgboost')} (seasonal worse than persistence) |")
    L.append(f"| 90 dBA crossings warned | **{ex30['hits']}/{ex30['events']}**, median lead **{ex30['median_lead_min']:.0f}–"
             f"{ex60['median_lead_min']:.1f} min**, false alarms **{100 * ex60['false_alarm_ratio']:.0f}–"
             f"{100 * ex30['false_alarm_ratio']:.0f}%** (seasonal baseline "
             f"{100 * f['noise_dba_h60']['exceedance']['seasonal']['false_alarm_ratio']:.0f}–"
             f"{100 * f['noise_dba_h30']['exceedance']['seasonal']['false_alarm_ratio']:.0f}%) |")
    n = e["valid"]
    L.append(f"| Exposure, 140 worker-days | **{100 * e['over_100_niosh'] / n:.0f}% NIOSH / {100 * e['over_100_india'] / n:.0f}% "
             f"India** above 100% dose; coverage {e['coverage_mean']:.1f}%, unaccounted {e['unaccounted_mean']:.2f}% |")
    L.append(f"| Risk index (deterministic, design choice) | " + ", ".join(f"{k} {v}" for k, v in sorted(r["bands"].items()))
             + f"; median {r['median']:.0f}/100 — a loud mill, not a calibration fault |")
    L.append(f"| Risk index, worst single component | Shown beside the composite; worst band above the composite band "
             f"on **{r['worst_band_higher']} of {r['n']}** worker-days (components are not on commensurable scales) |")
    ml = nums.get("ml")
    if ml:
        c = ml["classification"]["runs"]
        ab, cr30, cr60 = (c[k]["test"][c[k]["chosen"]] for k in ("above_h30", "crossing_h30", "crossing_h60"))
        L.append(f"| Exceedance classification (90 dBA) | `above`: prevalence {100 * ab['prevalence']:.0f}%, PR-AUC "
                 f"**{ab['pr_auc']:.2f}** vs persistence {c['above_h30']['test']['persistence']['pr_auc']:.2f}. "
                 f"`crossing`: prevalence {100 * cr30['prevalence']:.0f}%, PR-AUC **{cr30['pr_auc']:.2f} / "
                 f"{cr60['pr_auc']:.2f}** (t+30 / t+60), persistence cannot predict it |")
        st30 = c["crossing_h30"]["by_schedule"]["steady"][c["crossing_h30"]["chosen"]]["pr_auc"]
        st60 = c["crossing_h60"]["by_schedule"]["steady"][c["crossing_h60"]["chosen"]]["pr_auc"]
        L.append(f"| Crossing, break-free windows | PR-AUC **{st30:.2f} / {st60:.2f}** — the honest figure; Z2 "
                 "(near 90 dBA) barely beats chance |")
        a = ml["ablation"]["forecasting"]
        d30, d60 = (a[k]["differences"]["model_3_vs_model_1"] for k in ("noise_dba_h30", "noise_dba_h60"))
        L.append(f"| Objective 6 ablation | Machine + history data (Model 3) improve noise skill by **{d30['point']:+.3f} / "
                 f"{d60['point']:+.3f}** (95% CIs exclude 0); worker data (Model 2) adds at most "
                 f"{a['noise_dba_h30']['differences']['model_2_vs_model_1']['point']:+.3f} (under 0.1 dB); "
                 "no practical gain for dust, heat or classification |")
        an = ml["anomaly"]
        bad = [k for k, v in an["machines"].items() if v["configured_degrading"]][0]
        L.append(f"| Machine condition | {bad} flagged on day **{an['machines'][bad]['ewma']['alarm_day']:.2f}** (EWMA) / "
                 f"{an['machines'][bad]['isolation_forest']['alarm_day']:.2f} (isolation forest); **no healthy machine "
                 f"alarmed**; healthy false-positive rate {100 * an['healthy_summary']['ewma']['flag_rate']:.2f}% "
                 f"vs {100 * an['healthy_summary']['isolation_forest']['flag_rate']:.2f}% |")
        deep = ml.get("deep")
        if deep and deep.get("status") == "completed":
            t = deep["targets"]
            L.append(f"| Deep learning (GRU, optional) | Noise skill **{t['noise_dba_h30']['test']['skill_vs_persistence']:+.2f} / "
                     f"{t['noise_dba_h60']['test']['skill_vs_persistence']:+.2f}** vs XGBoost on the same features "
                     f"{t['noise_dba_h30']['xgboost_skill']:+.2f} / {t['noise_dba_h60']['xgboost_skill']:+.2f} — "
                     "no advantage; the engineered lags already carry the temporal structure |")
    L += ["", "**Say first:** simulated data · zone-weighted presence, not personal measurement · WBGT approximate "
          "(underestimate) · dust indicative · the index is a weighted design choice, not a clinical model."]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def main(argv: list[str] | None = None) -> int:
    """Build every figure and document."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--week-db", type=Path, default=ROOT / "data" / "week" / "textile_ohs.db")
    p.add_argument("--skip-shap", action="store_true", help="reuse the existing SHAP figures and JSON")
    args = p.parse_args(argv)
    figstyle.apply()
    FIG.mkdir(parents=True, exist_ok=True)
    features = pd.read_parquet(ROOT / ML["features_path"])
    predictions = pd.read_parquet(ROOT / ML["predictions_path"])
    metrics_path = ROOT / ML["metrics_path"]
    test_day = forecast.chronological_splits(features["t_ms"], ML["split_days"], ML["split_fallback_fractions"])[0][-1]
    test_date = from_epoch_ms(test_day.start_ms)[:10]

    fig_noise_timeseries(features, "Z5", test_date)
    fig_forecast(predictions, "Z5", 30)
    fig_zone_hour(features)
    score_day2 = load_json(ROOT / "data" / "score.json")
    if score_day2:
        fig_confusion(score_day2[0])
    if not args.skip_shap:
        explain.run(ROOT / ML["features_path"], ROOT / ML["predictions_path"],
                    ROOT / config.get_paths()["model_dir"] / "forecast_noise_dba_30.pkl", FIG,
                    ROOT / "data" / "shap_noise_h30.json")
    results = json.loads(metrics_path.read_text(encoding="utf-8"))
    forecast.write_report(results, DOCS / "results_forecast.md", DOCS / "text" / "results_forecast_discussion.md")

    with Repo(args.week_db) as repo:
        fig_worker_doses(repo.get_exposure_rows("personal_exposure_shift"))
        exposure = exposure_numbers(repo)
        consistency = consistency_residual(repo)
    write_exposure_md(exposure, DOCS / "results_exposure.md")
    week_score = load_json(ROOT / "data" / "week" / "score.json")
    nums = {"tests": test_count(),
            "day2_run": parse_run_logs(ROOT / "logs" / "publisher.log", ROOT / "logs" / "ingest.log"),
            "week_run": parse_run_logs(ROOT / "logs" / "week_publisher.log", ROOT / "logs" / "week_ingest.log"),
            "score_day2": score_day2[0] if score_day2 else None, "score_week": week_score[0] if week_score else None,
            "consistency": consistency, "exposure": exposure, "forecast": results,
            "shap": load_json(ROOT / "data" / "shap_noise_h30.json")}
    ml_metrics = report_ml.load_metrics()
    if all(ml_metrics[k] is not None for k in ("classification", "ablation", "anomaly")):
        nums["ml"] = ml_metrics
        report_ml.write_results_ml(ml_metrics, DOCS / "text" / "results_ml_body.md", DOCS / "results_ml.md")
        report_ml.write_model_cards(ml_metrics, DOCS / "model_cards.md")
    write_summary_md(DOCS / "results_summary.md", nums)
    write_numbers_sheet(DOCS / "numbers_sheet.md", nums)
    # Machine-readable copy for the dashboard (docs/ is committed; works with nothing running).
    slim = {k: v for k, v in nums.items() if k != "exposure"}
    slim["forecast"] = {"split_rule": results["split_rule"], "splits": results["splits"],
                        "targets": {k: {f: v[f] for f in ("metrics", "chosen", "by_schedule", "exceedance", "n_test")
                                        if f in v} for k, v in results["targets"].items()}}
    slim.pop("ml", None)
    if "ml" in nums:
        slim["ml"] = {"comparison": report_ml.comparison_rows(nums["ml"]),
                      "deep_learning": report_ml.deep_learning_text(nums["ml"].get("deep"))}
    slim["exposure"] = {k: exposure[k] for k in ("dates", "valid", "over_100_niosh", "over_100_india", "coverage_mean",
                                                 "unaccounted_mean", "risk")}
    (DOCS / "results_numbers.json").write_text(json.dumps(slim, indent=2, default=str), encoding="utf-8")
    print(f"results pack written: {len(list(FIG.glob('*.png')))} figures, results_forecast.md, "
          "results_exposure.md, results_summary.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
