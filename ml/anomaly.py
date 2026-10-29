"""Machine condition monitoring (Task C): vibration anomaly detection per machine.

    python -m ml.anomaly [--db data/week/textile_ohs.db] [--report-only]

Input: machine_status readings through db/repo.py. Only readings taken while the
machine is running, with an "ok" vibration flag and no anomaly record flag, are
used; they are averaged into 10-minute bins (a bin counts only if the machine ran
for at least half of it). Features per bin: mean vib_rms_g, mean vib_peak_g, the
peak/RMS ratio (crest factor), and 1-hour rolling means of RMS and peak.

Two detectors, both calibrated per machine on its own first day (Phase I) and then
run on the rest of the week (Phase II):
  * EWMA control chart on the standardised bin RMS (Montgomery, Introduction to
    Statistical Quality Control, ch. 9): z_t = lambda x_t + (1 - lambda) z_{t-1},
    limits +/- L sigma sqrt(lambda / (2 - lambda) (1 - (1 - lambda)^(2t))).
  * Isolation forest on all five features, fitted on the reference bins; a bin is
    flagged when its anomaly score exceeds a high quantile of the reference scores.
An ALARM is `alarm_consecutive_bins` consecutive flagged bins, for both detectors.

Evaluation uses the configured machine condition only as the scoring label: M5 is the
degrading machine (machines[].degrading) and the seven others are healthy. The
detectors never see that field.

Second analysis: do vibration anomalies on M5 precede increases in Z5 zone noise?
Lead-lag cross-correlation of hourly series, plus the time between the vibration
alarm and an alarm from the same EWMA chart run on Z5 running-hours noise.
COUPLING_CAVEAT applies, and is repeated wherever the machine-noise link appears.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

import config
from db.repo import Repo, from_epoch_ms

log = logging.getLogger("ml.anomaly")

DAY_MS = 86_400_000
MIN_MS = 60_000
HOUR_MS = 3_600_000
FEATURES = ("rms", "peak", "crest", "rms_roll", "peak_roll")
COUPLING_CAVEAT = ("The simulator couples machine vibration to zone noise by construction (a worn machine's share "
                   "of zone sound energy scales with its vibration), so this shows that the pipeline detects the "
                   "link, not that the physical relationship holds in a real mill.")


# ---------------------------------------------------------------- binning


def bin_vibration(readings: pd.DataFrame, bin_min: int, min_running_fraction: float) -> pd.DataFrame:
    """10-minute (or `bin_min`) bins of running-only vibration.

    Args:
        readings: columns ts_ms, running (0/1), vib_rms_g, vib_peak_g, usable (bool:
            vibration flag ok and no anomaly record flag).
        bin_min: bin width in minutes.
        min_running_fraction: minimum share of the bin's readings taken while running.

    Returns:
        DataFrame indexed by bin start (ms) with rms, peak, crest, n; invalid bins dropped.
    """
    bin_ms = bin_min * MIN_MS
    r = readings.assign(bin=readings["ts_ms"] // bin_ms * bin_ms)
    run_frac = r.groupby("bin")["running"].mean()
    use = r[(r["running"] == 1) & r["usable"] & r["vib_rms_g"].notna() & r["vib_peak_g"].notna()]
    g = use.groupby("bin")
    out = pd.DataFrame({"rms": g["vib_rms_g"].mean(), "peak": g["vib_peak_g"].mean(), "n": g.size()})
    out["crest"] = out["peak"] / out["rms"]
    out = out[run_frac.reindex(out.index) >= min_running_fraction]
    out.index.name = "bin_ms"
    return out


def add_rolling(bins: pd.DataFrame, k: int) -> pd.DataFrame:
    """Rolling means of RMS and peak over the last k valid bins (past and current only)."""
    return bins.assign(rms_roll=bins["rms"].rolling(k, min_periods=1).mean(),
                       peak_roll=bins["peak"].rolling(k, min_periods=1).mean())


# ---------------------------------------------------------------- detectors


def ewma_chart(x: np.ndarray, mu0: float, sigma0: float, lam: float, limit_sigma: float) -> dict[str, np.ndarray]:
    """Two-sided EWMA chart with exact (time-varying) control limits.

    Returns the EWMA statistic z (in sigma units), the half-width of the limits,
    and the per-point out-of-control flag.
    """
    if sigma0 <= 0:
        raise ValueError("reference standard deviation must be positive")
    s = (np.asarray(x, dtype=float) - mu0) / sigma0
    z = np.empty(len(s))
    prev = 0.0
    for i, v in enumerate(s):
        prev = lam * v + (1.0 - lam) * prev
        z[i] = prev
    t = np.arange(1, len(s) + 1)
    half = limit_sigma * np.sqrt(lam / (2.0 - lam) * (1.0 - (1.0 - lam) ** (2 * t)))
    return {"z": z, "limit": half, "flag": np.abs(z) > half}


def first_alarm(flags: np.ndarray, consecutive: int) -> int | None:
    """Position at which `consecutive` flagged points in a row are first completed."""
    run = 0
    for i, f in enumerate(flags):
        run = run + 1 if f else 0
        if run >= consecutive:
            return i
    return None


def ewma_flags(bins: pd.DataFrame, reference: np.ndarray, lam: float, limit_sigma: float) -> np.ndarray:
    """EWMA flags for the monitoring bins (reference statistics from the reference bins)."""
    ref = bins.loc[reference, "rms"].to_numpy()
    return ewma_chart(bins.loc[~reference, "rms"].to_numpy(), float(ref.mean()), float(ref.std(ddof=1)),
                      lam, limit_sigma)["flag"]


def isolation_forest_flags(bins: pd.DataFrame, reference: np.ndarray, n_estimators: int, quantile: float,
                           seed: int) -> np.ndarray:
    """Isolation-forest flags for the monitoring bins (fitted on the reference bins)."""
    X = bins[list(FEATURES)].to_numpy()
    model = IsolationForest(n_estimators=n_estimators, random_state=seed).fit(X[reference])
    score = -model.score_samples(X)                          # higher = more anomalous
    return score[~reference] > np.quantile(score[reference], quantile)


def detect(bins: pd.DataFrame, reference: np.ndarray, c: dict[str, Any], seed: int) -> dict[str, dict[str, Any]]:
    """Run both detectors on one machine's bins. Times are bin starts (ms)."""
    t = bins.index.to_numpy()[~reference]
    k = int(c["alarm_consecutive_bins"])
    flags = {"ewma": ewma_flags(bins, reference, float(c["ewma"]["lambda"]), float(c["ewma"]["limit_sigma"])),
             "isolation_forest": isolation_forest_flags(bins, reference, int(c["isolation_forest"]["n_estimators"]),
                                                        float(c["isolation_forest"]["reference_quantile"]), seed)}
    out = {}
    for name, f in flags.items():
        pos = first_alarm(f, k)
        out[name] = {"alarm_ms": int(t[pos]) if pos is not None else None, "n_monitored": int(len(f)),
                     "n_flagged": int(f.sum()), "flag_rate": float(f.mean()) if len(f) else None,
                     "flags": f}
    return out


# ---------------------------------------------------------------- lead-lag


def cross_correlation(a: pd.Series, b: pd.Series, max_lag: int) -> dict[int, float]:
    """corr(a[t], b[t + k]) for k = -max_lag .. max_lag on a shared regular index.

    Positive k with a high correlation means `a` leads `b` by k steps.
    """
    out = {}
    for k in range(-max_lag, max_lag + 1):
        pair = pd.concat([a, b.shift(-k)], axis=1).dropna()
        out[k] = float(pair.iloc[:, 0].corr(pair.iloc[:, 1])) if len(pair) > 2 else float("nan")
    return out


def peak_lag(xc: dict[int, float]) -> tuple[int, float]:
    """Lag with the highest correlation, and that correlation."""
    k = max((k for k in xc if not np.isnan(xc[k])), key=lambda k: xc[k])
    return k, xc[k]


def hourly_energy_mean(noise: pd.Series, min_minutes: int = 30) -> pd.Series:
    """Hourly L_Aeq (energy average) of a 1-minute dBA series indexed by t_ms."""
    hours = noise.index.to_numpy() // HOUR_MS * HOUR_MS
    e = (10.0 ** (noise / 10.0)).groupby(hours)
    mean = e.mean().where(e.count() >= min_minutes)
    return 10.0 * np.log10(mean)


def lead_lag(m5_bins: pd.DataFrame, z5_noise: pd.Series, start_ms: int, alarm_vib_ms: int | None,
             c: dict[str, Any]) -> dict[str, Any]:
    """Does M5 vibration lead Z5 noise? Cross-correlation and alarm-to-alarm lead time."""
    step = int(c["lead_lag_bin_min"]) * MIN_MS
    vib = m5_bins["rms"].groupby(m5_bins.index.to_numpy() // step * step).mean()
    noise = hourly_energy_mean(z5_noise)
    grid = np.arange(min(vib.index.min(), noise.index.min()), max(vib.index.max(), noise.index.max()) + step, step)
    vib, noise = vib.reindex(grid), noise.reindex(grid)
    max_lag = int(c["lead_lag_max_lag_h"] * 60 // c["lead_lag_bin_min"])
    raw = cross_correlation(vib, noise, max_lag)
    diff = cross_correlation(vib.diff(), noise.diff(), max_lag)
    ref_end = start_ms + int(c["reference_days"]) * DAY_MS
    ref = (noise.index < ref_end) & noise.notna().to_numpy()
    mon = (noise.index >= ref_end) & noise.notna().to_numpy()
    chart = ewma_chart(noise[mon].to_numpy(), float(noise[ref].mean()), float(noise[ref].std(ddof=1)),
                       float(c["ewma"]["lambda"]), float(c["ewma"]["limit_sigma"]))
    pos = first_alarm(chart["flag"] & (chart["z"] > 0), int(c["alarm_consecutive_bins"]))
    noise_alarm = int(noise.index[mon][pos]) if pos is not None else None
    k_raw, r_raw = peak_lag(raw)
    k_diff, r_diff = peak_lag(diff)
    return {"step_min": int(c["lead_lag_bin_min"]), "xcorr_raw": raw, "xcorr_diff": diff,
            "peak_lag_raw_steps": k_raw, "peak_corr_raw": r_raw, "corr_raw_lag0": raw[0],
            "peak_lag_diff_steps": k_diff, "peak_corr_diff": r_diff, "corr_diff_lag0": diff[0],
            "noise_reference_db": float(noise[ref].mean()), "noise_last_day_db":
                float(noise[noise.index >= noise.index.max() - DAY_MS].mean()),
            "noise_alarm_ms": noise_alarm, "vibration_alarm_ms": alarm_vib_ms,
            "alarm_lead_h": (noise_alarm - alarm_vib_ms) / HOUR_MS
            if noise_alarm is not None and alarm_vib_ms is not None else None}


# ---------------------------------------------------------------- pipeline


def load_readings(repo: Repo, machine_id: str, start_ts: str, end_ts: str, exclude: set[str]) -> pd.DataFrame:
    """One machine's readings as a DataFrame with a `usable` column."""
    rows = repo.get_machine_status(machine_id, start_ts, end_ts)
    df = pd.DataFrame([{"ts_ms": r["ts_epoch_ms"], "running": int(r["running"]), "vib_rms_g": r["vib_rms_g"],
                        "vib_peak_g": r["vib_peak_g"],
                        "usable": r["flag_vibration"] == "ok" and not (set(r["flags"]) & exclude)} for r in rows])
    return df.astype({"vib_rms_g": float, "vib_peak_g": float})


def day_number(t_ms: int, start_ms: int) -> float:
    """Simulated day as a 1-based fractional number (day 1 = the first 24 h of data)."""
    return 1.0 + (t_ms - start_ms) / DAY_MS


def run(db_path: Path, features_path: Path, metrics_path: Path) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    """Detect per machine, evaluate against the degrading flag, run the lead-lag analysis."""
    ml = config.get_ml_config()
    c = ml["anomaly"]
    seed = int(ml["random_seed"])
    exclude = set(config.get_exposure_methods()["personal"]["exclude_record_flags"])
    with Repo(db_path, read_only=True) as repo:
        span = repo.get_data_span()
        if span is None:
            raise ValueError(f"no data in {db_path}")
        start_ms = int(pd.Timestamp(span[0]).value // 1_000_000)
        bins_by_machine = {}
        for m in config.get_machines():
            readings = load_readings(repo, m["machine_id"], span[0], from_epoch_ms(
                int(pd.Timestamp(span[1]).value // 1_000_000) + 1), exclude)
            bins_by_machine[m["machine_id"]] = add_rolling(
                bin_vibration(readings, int(c["bin_min"]), float(c["min_running_fraction"])), int(c["rolling_bins"]))
    ref_end = start_ms + int(c["reference_days"]) * DAY_MS
    results: dict[str, Any] = {"db": str(db_path), "start": span[0], "end": span[1], "settings": {
        k: c[k] for k in ("bin_min", "min_running_fraction", "reference_days", "rolling_bins", "ewma",
                          "isolation_forest", "alarm_consecutive_bins")}, "random_seed": seed, "machines": {}}
    for m in config.get_machines():
        mid = m["machine_id"]
        bins = bins_by_machine[mid]
        reference = bins.index.to_numpy() < ref_end
        det = detect(bins, reference, c, seed)
        entry = {"zone_id": m["zone_id"], "configured_degrading": bool(m["degrading"]),
                 "n_bins": int(len(bins)), "n_reference": int(reference.sum()),
                 "reference_rms_mean": float(bins.loc[reference, "rms"].mean()),
                 "last_day_rms_mean": float(bins.loc[bins.index >= bins.index.max() - DAY_MS, "rms"].mean())}
        for name, d in det.items():
            bins.loc[~reference, f"flag_{name}"] = d.pop("flags")
            d["alarm_ts"] = from_epoch_ms(d["alarm_ms"]) if d["alarm_ms"] is not None else None
            d["alarm_day"] = day_number(d["alarm_ms"], start_ms) if d["alarm_ms"] is not None else None
            entry[name] = d
        results["machines"][mid] = entry
        log.info("%s: EWMA alarm %s, IF alarm %s", mid, entry["ewma"]["alarm_ts"], entry["isolation_forest"]["alarm_ts"])
    healthy = [k for k, v in results["machines"].items() if not v["configured_degrading"]]
    results["healthy_summary"] = {
        name: {"machines_with_alarm": [k for k in healthy if results["machines"][k][name]["alarm_ms"] is not None],
               "flag_rate": float(sum(results["machines"][k][name]["n_flagged"] for k in healthy)
                                  / sum(results["machines"][k][name]["n_monitored"] for k in healthy))}
        for name in ("ewma", "isolation_forest")}
    degrading = [k for k, v in results["machines"].items() if v["configured_degrading"]]
    feats = pd.read_parquet(features_path, columns=["t_ms", "zone_id", "noise_dba", "machines_running_count"])
    if degrading:
        mid = degrading[0]
        zone = results["machines"][mid]["zone_id"]
        n_machines = sum(1 for m in config.get_machines() if m["zone_id"] == zone)
        z = feats[(feats["zone_id"].astype(str) == zone) & (feats["machines_running_count"] == n_machines)]
        noise = z.set_index("t_ms")["noise_dba"].dropna()
        results["lead_lag"] = dict(lead_lag(bins_by_machine[mid], noise, start_ms,
                                            results["machines"][mid]["ewma"]["alarm_ms"], c),
                                   machine_id=mid, zone_id=zone)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(results, indent=2, default=_json_default), encoding="utf-8")
    bins_dir = metrics_path.with_suffix("")
    bins_dir.mkdir(parents=True, exist_ok=True)
    for mid, b in bins_by_machine.items():
        b.to_parquet(bins_dir / f"{mid}.parquet")
    return results, bins_by_machine


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    raise TypeError(type(o))


# ---------------------------------------------------------------- figure and report


def write_figure(results: dict[str, Any], bins_by_machine: dict[str, pd.DataFrame], path: Path) -> Path:
    """Degrading machine's hourly vib_rms (relative to its reference) over the week,
    healthy-machine band shaded, detection points marked."""
    import matplotlib.pyplot as plt

    import figstyle

    figstyle.apply()
    start_ms = int(pd.Timestamp(results["start"]).value // 1_000_000)
    ref_days = int(results["settings"]["reference_days"])

    def hourly_rel(mid: str) -> pd.Series:
        b = bins_by_machine[mid]
        s = (b["rms"] / results["machines"][mid]["reference_rms_mean"])
        return s.groupby(b.index.to_numpy() // HOUR_MS * HOUR_MS).mean()

    healthy = [k for k, v in results["machines"].items() if not v["configured_degrading"]]
    degrading = [k for k, v in results["machines"].items() if v["configured_degrading"]]
    band = pd.concat({k: hourly_rel(k) for k in healthy}, axis=1)
    x = lambda idx: 1.0 + (np.asarray(idx) - start_ms) / DAY_MS          # noqa: E731
    fig, ax = plt.subplots(figsize=(13, 5.8))
    ax.axvspan(1, 1 + ref_days, color="#F2F2F2", zorder=0)
    ax.text(1.05, 0.45, "reference day\n(each machine's\nown baseline)", transform=ax.get_xaxis_transform(),
            fontsize=10, color=figstyle.NEUTRAL)
    ax.fill_between(x(band.index), band.min(axis=1), band.max(axis=1), color=figstyle.LIGHT, alpha=0.7,
                    label=f"healthy machines ({len(healthy)}): hourly min–max")
    styles = {"ewma": ("--", "#D55E00", "EWMA alarm"), "isolation_forest": (":", "#009E73", "Isolation-forest alarm")}
    for mid in degrading:
        s = hourly_rel(mid)
        ax.plot(x(s.index), s.to_numpy(), color=figstyle.HAZARD_COLOURS["noise"], linewidth=1.6,
                label=f"{mid} (degrading), hourly mean")
        for name, (linestyle, colour, label) in styles.items():
            a = results["machines"][mid][name]
            if a["alarm_ms"] is None:
                continue
            ax.axvline(x(a["alarm_ms"]), color=colour, linestyle=linestyle, linewidth=2.0,
                       label=f"{label} on {mid}: day {a['alarm_day']:.2f}")
    ax.set(xlabel="Simulated day", ylabel="vib RMS / reference-day mean", title="Machine vibration over the week")
    ax.legend(loc="upper left", fontsize=10)
    figstyle.simulated_badge(ax)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def _day(v: float | None) -> str:
    return "never" if v is None else f"day {v:.2f}"


def write_report(results: dict[str, Any], path: Path, discussion_path: Path | None = None) -> None:
    """docs/results_anomaly.md."""
    s = results["settings"]
    L = ["# Machine condition monitoring (Task C)", "",
         "**SIMULATED DATA.** Generated by `ml/anomaly.py` (`--report-only` rebuilds this file and the figure).",
         "", "## Method", "",
         f"- Data: `machine_status` for {len(results['machines'])} machines, {results['start']} → {results['end']}. "
         "Readings are used only while the machine runs, with an `ok` vibration flag and no anomaly record flag.",
         f"- Bins: {s['bin_min']}-min means of `vib_rms_g` and `vib_peak_g` (bin kept if the machine ran ≥ "
         f"{s['min_running_fraction']:.0%} of it); features RMS, peak, crest factor (peak/RMS), and "
         f"{s['rolling_bins']}-bin rolling means of RMS and peak.",
         f"- Reference (Phase I): each machine's own first {s['reference_days']} day(s). Monitoring (Phase II): the "
         "rest of the week.",
         f"- **EWMA chart** on standardised bin RMS, λ = {s['ewma']['lambda']}, limits ±{s['ewma']['limit_sigma']}σ "
         "(exact time-varying width). Two-sided.",
         f"- **Isolation forest** ({s['isolation_forest']['n_estimators']} trees, seed {results['random_seed']}) on "
         "all five features, fitted on the reference bins; flag = score above the "
         f"{s['isolation_forest']['reference_quantile']:.0%} quantile of reference scores.",
         f"- **Alarm** = {s['alarm_consecutive_bins']} consecutive flagged bins, for both detectors.",
         "- Scoring label, used only to evaluate: `machines[].degrading` in `config.yaml` (M5 degrades; the rest are "
         "healthy). All settings were fixed in `config.yaml` before the results were seen.", "",
         "## Detection", "",
         "| Machine | Zone | Configured condition | Ref. RMS (g) | Last-day RMS (g) | EWMA alarm | EWMA flag rate | IF alarm | "
         "IF flag rate |", "|---|---|---|---|---|---|---|---|---|"]
    for mid, e in results["machines"].items():
        condition = "**degrading**" if e["configured_degrading"] else "healthy"
        L.append(f"| {mid} | {e['zone_id']} | {condition} | {e['reference_rms_mean']:.3f} | {e['last_day_rms_mean']:.3f} | "
                 f"{_day(e['ewma']['alarm_day'])} | {e['ewma']['flag_rate']:.1%} | "
                 f"{_day(e['isolation_forest']['alarm_day'])} | {e['isolation_forest']['flag_rate']:.1%} |")
    hs = results["healthy_summary"]
    L += ["", "Flag rate = share of monitored bins flagged (before the consecutive-bin rule). On healthy machines it "
          "is the per-bin false-positive rate.", "",
          "| Detector | Healthy machines with any alarm | Healthy per-bin false-positive rate |", "|---|---|---|"]
    for name, h in hs.items():
        L.append(f"| {name} | {', '.join(h['machines_with_alarm']) or 'none'} | {h['flag_rate']:.2%} |")
    if "lead_lag" in results:
        ll = results["lead_lag"]
        st = ll["step_min"]
        lead = "no noise alarm" if ll["alarm_lead_h"] is None else f"{ll['alarm_lead_h']:+.1f} h"
        L += ["", f"## Does {ll['machine_id']} vibration precede {ll['zone_id']} noise?", "",
              f"Hourly {ll['machine_id']} RMS against hourly {ll['zone_id']} L_Aeq over the minutes when every "
              f"{ll['zone_id']} machine was running (so the schedule does not dominate). Positive lag = vibration "
              "leads noise.", "",
              "| Measure | Value |", "|---|---|",
              f"| Cross-correlation, levels: peak lag | {ll['peak_lag_raw_steps'] * st / 60:+.0f} h (r = "
              f"{ll['peak_corr_raw']:.2f}; r at lag 0 = {ll['corr_raw_lag0']:.2f}) |",
              f"| Cross-correlation, hour-to-hour changes: peak lag | {ll['peak_lag_diff_steps'] * st / 60:+.0f} h "
              f"(r = {ll['peak_corr_diff']:.2f}; r at lag 0 = {ll['corr_diff_lag0']:.2f}) |",
              f"| {ll['zone_id']} running-hours noise, reference day → last day | {ll['noise_reference_db']:.2f} → "
              f"{ll['noise_last_day_db']:.2f} dBA |",
              f"| Vibration EWMA alarm | {ll['vibration_alarm_ms'] and from_epoch_ms(ll['vibration_alarm_ms'])} |",
              f"| Noise EWMA alarm (same chart settings, upward) | "
              f"{ll['noise_alarm_ms'] and from_epoch_ms(ll['noise_alarm_ms'])} |",
              f"| Lead of the vibration alarm over the noise alarm | {lead} |", "",
              f"**Caveat.** {COUPLING_CAVEAT}", ""]
    L += ["Figure: `docs/figures/anomaly_m5_vibration.png`.", ""]
    if discussion_path is not None and discussion_path.exists():
        L += [discussion_path.read_text(encoding="utf-8").strip(), ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = config.CONFIG_PATH.parent
    ml = config.get_ml_config()
    c = ml["anomaly"]
    p = argparse.ArgumentParser(prog="python -m ml.anomaly", description=__doc__.split("\n")[0])
    p.add_argument("--db", type=Path, default=root / c["db_path"])
    p.add_argument("--features", type=Path, default=root / ml["features_path"])
    p.add_argument("--metrics", type=Path, default=root / c["metrics_path"])
    p.add_argument("--figure", type=Path,
                   default=root / config.get_paths()["figures_dir"] / "anomaly_m5_vibration.png")
    p.add_argument("--report", type=Path, default=root / "docs" / "results_anomaly.md")
    p.add_argument("--discussion", type=Path, default=root / "docs" / "text" / "results_anomaly_discussion.md")
    p.add_argument("--report-only", action="store_true", help="skip detection: rebuild figure and report")
    args = p.parse_args(argv)
    if args.report_only:
        results = json.loads(args.metrics.read_text(encoding="utf-8"))
        bins_dir = args.metrics.with_suffix("")
        bins = {m: pd.read_parquet(bins_dir / f"{m}.parquet") for m in results["machines"]}
    else:
        results, bins = run(args.db, args.features, args.metrics)
    write_figure(results, bins, args.figure)
    write_report(results, args.report, args.discussion)
    print(json.dumps({m: {"ewma": e["ewma"]["alarm_day"], "isolation_forest": e["isolation_forest"]["alarm_day"]}
                      for m, e in results["machines"].items()}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
