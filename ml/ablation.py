"""Model 1 / 2 / 3 feature ablation (Objective 6): does worker and machine data help?

    python -m ml.ablation [--features data/features.parquet] [--report-only]

Reframing (stated in the results document too): the project document frames
Models 1/2/3 as *risk* prediction, which cannot be evaluated honestly without
clinical or questionnaire labels. The same research question - does worker-specific
and machine data improve prediction over environmental data alone? - is answered
here on *hazard forecasting* and *exceedance classification*, whose labels are
future measurements.

Feature sets, strictly nested (each contains the previous one):
  Model 1  environmental: noise, PM2.5, WBGT with lags / rolling stats / rate of
           change; temperature, RH, CO2 and their rolling means; time of day; zone
           identity (all three models pool every zone, so all three need it).
  Model 2  Model 1 + worker-specific: workers present, shift, break.
  Model 3  Model 2 + machine and history: machines running, mean current, max
           vibration, and zone-history features over trailing windows (8 h zone
           L_Aeq and mean PM2.5 - the zone's own cumulative exposure so far - and
           the fraction of the last hour machines were running).

Per-worker cumulative exposure (exposure/cumulative.py) is not used: it is
materialised per worker-day, after the day ends, so at minute t it would contain
the future. The zone-history features are its leak-free zone-level counterpart.

Fairness: one model class (XGBoost), one split (ml/forecast.py), one seed, and the
SAME grid searched SEPARATELY for each feature set on the validation day. Reusing
one set's parameters for another would handicap it.

Significance: a moving-block bootstrap on the test day. (zone, 60-minute block)
groups are resampled with replacement; for each resample the metric DIFFERENCE
between two models is recomputed on identical rows. Blocks keep autocorrelated
minutes together; resampling single minutes would give falsely narrow intervals.
If the 95% interval contains zero, the difference is reported as not statistically
distinguishable on this dataset.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from xgboost import XGBRegressor

import config
from ml.classify_exceedance import (choose_threshold, classification_metrics, expand_grid, future_value, make_labels,
                                    prevalence, tune)
from ml.features import feature_columns
from ml.forecast import MIN_MS, chronological_splits, regression_metrics, split_mask

log = logging.getLogger("ml.ablation")

# ---------------------------------------------------------------- feature sets (the single definition)

_LAGGED = ["", "_lag1", "_lag2", "_lag3", "_lag5", "_lag10", "_lag15", "_lag30", "_rmean5", "_rstd5", "_rmean15",
           "_rstd15", "_rmean30", "_rstd30", "_roc5"]
ENVIRONMENTAL: list[str] = (
    [f"noise_dba{s}" for s in _LAGGED] + [f"pm25_ugm3{s}" for s in _LAGGED] + [f"wbgt_c{s}" for s in _LAGGED]
    + ["temp_c", "temp_c_rmean15", "rh_pct", "rh_pct_rmean15", "co2_ppm", "co2_ppm_rmean15",
       "minute_sin", "minute_cos", "zone_code"])
WORKER: list[str] = ["workers_present_count", "shift_code", "is_break"]
HISTORY: list[str] = ["noise_dba_laeq_long", "pm25_ugm3_rmean_long", "machines_running_frac"]
MACHINE_HISTORY: list[str] = ["machines_running_count", "mean_current_a", "max_vib_rms_g"] + HISTORY
FEATURE_SETS: dict[str, list[str]] = {
    "model_1": ENVIRONMENTAL,
    "model_2": ENVIRONMENTAL + WORKER,
    "model_3": ENVIRONMENTAL + WORKER + MACHINE_HISTORY,
}
SET_LABELS = {"model_1": "Model 1 (environmental)", "model_2": "Model 2 (+ worker)",
              "model_3": "Model 3 (+ machine & history)"}
PAIRS = (("model_2", "model_1"), ("model_3", "model_1"), ("model_3", "model_2"))


def check_feature_sets(sets: dict[str, list[str]] = FEATURE_SETS) -> None:
    """Raise unless the sets are strictly nested and cover exactly the available columns.

    Model 3 must equal every ml/features.py feature plus the history features, so no
    column is silently left out of the ablation.
    """
    names = list(sets)
    for a, b in zip(names, names[1:]):
        if not set(sets[a]) < set(sets[b]):
            raise ValueError(f"{a} is not a strict subset of {b}")
    for name, cols in sets.items():
        if len(cols) != len(set(cols)):
            raise ValueError(f"{name} has duplicate columns")
    if set(sets[names[-1]]) != set(feature_columns()) | set(HISTORY):
        raise ValueError("the largest set must be exactly the feature table plus the history features")


# ---------------------------------------------------------------- history features


def add_history_features(df: pd.DataFrame, long_min: int, machine_min: int, min_coverage: float) -> pd.DataFrame:
    """Add the trailing-window zone-history columns (HISTORY). Uses data at or before t only.

    Each zone is put on a contiguous 1-minute grid first, so a window counts minutes,
    not rows. Noise is energy-averaged (10^(L/10)), never arithmetically averaged.
    A value is NaN unless `min_coverage` of its window has data.
    """
    out = df.copy()
    for col in HISTORY:
        out[col] = np.nan
    for _zone, idx in out.groupby("zone_id", observed=True).groups.items():
        z = out.loc[idx].set_index("t_ms")
        grid = np.arange(z.index.min(), z.index.max() + MIN_MS, MIN_MS)
        z = z.reindex(grid)
        energy = 10.0 ** (z["noise_dba"] / 10.0)
        long_e = energy.rolling(long_min, min_periods=int(long_min * min_coverage)).mean()
        hist = pd.DataFrame({
            "noise_dba_laeq_long": 10.0 * np.log10(long_e),
            "pm25_ugm3_rmean_long": z["pm25_ugm3"].rolling(long_min, min_periods=int(long_min * min_coverage)).mean(),
            "machines_running_frac": (z["machines_running_count"] > 0).astype(float).where(
                z["machines_running_count"].notna()).rolling(machine_min,
                                                             min_periods=int(machine_min * min_coverage)).mean(),
        }, index=grid)
        out.loc[idx, HISTORY] = hist.reindex(out.loc[idx, "t_ms"].to_numpy()).to_numpy()
    return out


# ---------------------------------------------------------------- bootstrap


def block_groups(zone: np.ndarray, t_ms: np.ndarray, block_min: int) -> np.ndarray:
    """Integer group id per row: one group per (zone, block of `block_min` minutes)."""
    keys = pd.Series(zone).astype(str) + ":" + pd.Series(t_ms // (block_min * MIN_MS)).astype(str)
    return pd.factorize(keys)[0]


def bootstrap_interval(groups: np.ndarray, stat: Callable[[np.ndarray], float], n_resamples: int, seed: int,
                       ci: float) -> dict[str, Any]:
    """Percentile interval of `stat` over group-resampled row indices.

    Args:
        groups: group id per row; whole groups are drawn with replacement.
        stat: function of a row-index array returning a float (NaN = undefined).
        n_resamples, seed, ci: resample count, RNG seed, interval width (0.95).

    Returns:
        {"point", "lo", "hi", "n_groups", "n_defined"}; point is stat on all rows.
    """
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    members = [np.flatnonzero(groups == g) for g in uniq]
    values = []
    for _ in range(n_resamples):
        pick = rng.integers(0, len(uniq), len(uniq))
        values.append(stat(np.concatenate([members[i] for i in pick])))
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    a = (1.0 - ci) / 2.0
    return {"point": float(stat(np.arange(len(groups)))),
            "lo": float(np.quantile(v, a)) if len(v) else None,
            "hi": float(np.quantile(v, 1.0 - a)) if len(v) else None,
            "n_groups": int(len(uniq)), "n_defined": int(len(v))}


def verdict(interval: dict[str, Any]) -> str:
    """'better' / 'worse' if the interval excludes 0, else 'not distinguishable'."""
    if interval["lo"] is None:
        return "undefined"
    if interval["lo"] > 0:
        return "better"
    if interval["hi"] < 0:
        return "worse"
    return "not distinguishable"


# ---------------------------------------------------------------- tuning


def tune_regressor(grid: dict[str, list[Any]], X_train: pd.DataFrame, y_train: np.ndarray, X_val: pd.DataFrame,
                   y_val: np.ndarray, seed: int, early_stopping: int) -> tuple[XGBRegressor, dict[str, Any],
                                                                               list[dict[str, Any]]]:
    """XGBoost regressor grid search on the validation set (selection: validation MAE)."""
    best, best_params, best_mae, search = None, {}, np.inf, []
    for params in expand_grid(grid):
        model = XGBRegressor(random_state=seed, n_jobs=-1, eval_metric="mae", early_stopping_rounds=early_stopping,
                             **params).fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        mae = float(np.mean(np.abs(model.predict(X_val) - y_val)))
        search.append({"params": params, "val_mae": mae})
        if mae < best_mae:
            best, best_params, best_mae = model, params, mae
    return best, dict(best_params, best_iteration=int(best.best_iteration)), search


def tune_per_set(kind: str, sets: dict[str, list[str]], grid: dict[str, list[Any]], train: pd.DataFrame,
                 y_train: np.ndarray, val: pd.DataFrame, y_val: np.ndarray, seed: int,
                 early_stopping: int) -> dict[str, dict[str, Any]]:
    """Run a separate grid search for each feature set ("regression" or "classification").

    Returns {set: {"model", "features", "chosen", "search"}}. Every set gets the full
    grid and only its own columns.
    """
    out = {}
    for name, cols in sets.items():
        if kind == "regression":
            model, params, search = tune_regressor(grid, train[cols], y_train, val[cols], y_val, seed, early_stopping)
        else:
            model, params, search = tune("xgboost", grid, train[cols], y_train, val[cols], y_val, seed,
                                         early_stopping)
        out[name] = {"model": model, "features": list(cols), "chosen": params, "search": search}
    return out


# ---------------------------------------------------------------- the two tasks


def _skill_stat(y: np.ndarray, pred: np.ndarray, persist: np.ndarray) -> Callable[[np.ndarray], float]:
    return lambda i: 1.0 - np.mean(np.abs(y[i] - pred[i])) / np.mean(np.abs(y[i] - persist[i]))


def _skill_diff_stat(y: np.ndarray, a: np.ndarray, b: np.ndarray, persist: np.ndarray) -> Callable[[np.ndarray], float]:
    return lambda i: (np.mean(np.abs(y[i] - b[i])) - np.mean(np.abs(y[i] - a[i]))) / np.mean(np.abs(y[i] - persist[i]))


def _pr_auc(y: np.ndarray, s: np.ndarray) -> float:
    return float(average_precision_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def _f1(y: np.ndarray, pred: np.ndarray) -> float:
    tp = int(np.sum((pred == 1) & (y == 1)))
    denom = 2 * tp + int(np.sum((pred == 1) & (y == 0))) + int(np.sum((pred == 0) & (y == 1)))
    return 2 * tp / denom if denom else float("nan")


def forecasting_ablation(df: pd.DataFrame, splits: list[Any], c: dict[str, Any], horizons: list[int],
                         seed: int) -> dict[str, Any]:
    """Skill, MAE, RMSE per feature set, with bootstrap intervals on skill differences."""
    out = {}
    bs = (int(c["bootstrap_resamples"]), seed, float(c["bootstrap_ci"]))
    for target in c["forecast_targets"]:
        for h in horizons:
            key = f"{target}_h{h}"
            y_col = f"target_{target}_h{h}"
            m = {s.name: split_mask(df["t_ms"], s, h) & df[y_col].notna() for s in splits}
            train, val = df[m["train"]], df[m["validation"]]
            test = df[m["test"] & df[target].notna()]
            fitted = tune_per_set("regression", FEATURE_SETS, c["grid"], train, train[y_col].to_numpy(), val,
                                  val[y_col].to_numpy(), seed, int(c["early_stopping_rounds"]))
            y, persist = test[y_col].to_numpy(), test[target].to_numpy()
            mae_p = float(np.mean(np.abs(y - persist)))
            preds = {n: f["model"].predict(test[f["features"]]) for n, f in fitted.items()}
            zones = test["zone_id"].astype(str).to_numpy()
            groups = block_groups(zones, test["t_ms"].to_numpy(), int(c["bootstrap_block_min"]))
            entry: dict[str, Any] = {
                "n_train": int(len(train)), "n_validation": int(len(val)), "n_test": int(len(test)),
                "tuning": {n: {"n_features": len(f["features"]), "chosen": f["chosen"], "search": f["search"]}
                           for n, f in fitted.items()},
                "test": {n: regression_metrics(y, p, mae_p) for n, p in preds.items()},
                "skill_interval": {n: bootstrap_interval(groups, _skill_stat(y, p, persist), *bs)
                                   for n, p in preds.items()},
                "differences": {f"{a}_vs_{b}": bootstrap_interval(groups, _skill_diff_stat(y, preds[a], preds[b],
                                                                                           persist), *bs)
                                for a, b in PAIRS},
                "by_zone": {}}
            for z in sorted(np.unique(zones)):
                zi = zones == z
                zy, zp = y[zi], persist[zi]
                if np.mean(np.abs(zy - zp)) == 0:
                    continue
                zg = groups[zi]
                entry["by_zone"][z] = {
                    "n": int(zi.sum()),
                    "skill": {n: float(_skill_stat(zy, p[zi], zp)(np.arange(zi.sum()))) for n, p in preds.items()},
                    "differences": {f"{a}_vs_{b}": bootstrap_interval(
                        zg, _skill_diff_stat(zy, preds[a][zi], preds[b][zi], zp), *bs) for a, b in PAIRS}}
            out[key] = entry
            log.info("forecast %s skill %s", key, {n: round(v["skill_vs_persistence"], 3)
                                                   for n, v in entry["test"].items()})
    return out


def classification_ablation(df: pd.DataFrame, splits: list[Any], c: dict[str, Any], horizons: list[int],
                            seed: int, threshold_db: float, max_fpr: float) -> dict[str, Any]:
    """PR-AUC and F1 per feature set, with bootstrap intervals on their differences."""
    out = {}
    bs = (int(c["bootstrap_resamples"]), seed, float(c["bootstrap_ci"]))
    now = df["noise_dba"].to_numpy(dtype=float)
    for h in horizons:
        future = future_value(df, "noise_dba", h)
        for variant in c["classification_variants"]:
            key = f"{variant}_h{h}"
            y_all, pop = make_labels(now, future, threshold_db, variant)
            idx = {s.name: np.flatnonzero(split_mask(df["t_ms"], s, h).to_numpy() & pop) for s in splits}
            part = {k: df.iloc[v] for k, v in idx.items()}
            yy = {k: y_all[v] for k, v in idx.items()}
            fitted = tune_per_set("classification", FEATURE_SETS, c["grid"], part["train"], yy["train"],
                                  part["validation"], yy["validation"], seed, int(c["early_stopping_rounds"]))
            y = yy["test"]
            scores, preds, thresholds = {}, {}, {}
            for n, f in fitted.items():
                thresholds[n] = choose_threshold(yy["validation"],
                                                 f["model"].predict_proba(part["validation"][f["features"]])[:, 1],
                                                 max_fpr)
                scores[n] = f["model"].predict_proba(part["test"][f["features"]])[:, 1]
                preds[n] = (scores[n] >= thresholds[n]).astype(int)
            zones = part["test"]["zone_id"].astype(str).to_numpy()
            groups = block_groups(zones, part["test"]["t_ms"].to_numpy(), int(c["bootstrap_block_min"]))
            entry: dict[str, Any] = {
                "n_train": int(len(idx["train"])), "n_test": int(len(y)), "prevalence_test": prevalence(y),
                "tuning": {n: {"n_features": len(f["features"]), "chosen": f["chosen"], "search": f["search"],
                               "threshold": thresholds[n]} for n, f in fitted.items()},
                "test": {n: classification_metrics(y, scores[n], thresholds[n]) for n in fitted},
                "differences": {}, "by_zone": {}}
            for a, b in PAIRS:
                entry["differences"][f"{a}_vs_{b}"] = {
                    "pr_auc": bootstrap_interval(groups, lambda i, a=a, b=b: _pr_auc(y[i], scores[a][i])
                                                 - _pr_auc(y[i], scores[b][i]), *bs),
                    "f1": bootstrap_interval(groups, lambda i, a=a, b=b: _f1(y[i], preds[a][i]) - _f1(y[i], preds[b][i]),
                                             *bs)}
            for z in sorted(np.unique(zones)):
                zi = zones == z
                zy = y[zi]
                if not 0 < zy.sum() < len(zy):
                    continue
                zg = groups[zi]
                entry["by_zone"][z] = {
                    "n": int(zi.sum()), "prevalence": prevalence(zy),
                    "pr_auc": {n: _pr_auc(zy, s[zi]) for n, s in scores.items()},
                    "differences": {f"{a}_vs_{b}": bootstrap_interval(
                        zg, lambda i, a=a, b=b, zi=zi, zy=zy: _pr_auc(zy[i], scores[a][zi][i])
                        - _pr_auc(zy[i], scores[b][zi][i]), *bs) for a, b in PAIRS}}
            out[key] = entry
            log.info("classify %s PR-AUC %s", key, {n: round(v["pr_auc"], 3) for n, v in entry["test"].items()})
    return out


def run(features_path: Path, metrics_path: Path) -> dict[str, Any]:
    """Both ablations end to end; writes the metrics JSON."""
    check_feature_sets()
    ml = config.get_ml_config()
    c = ml["ablation"]
    seed = int(ml["random_seed"])
    df = pd.read_parquet(features_path)
    df = add_history_features(df, int(c["history_long_window_min"]), int(c["history_machine_window_min"]),
                              float(c["history_min_coverage"]))
    splits, rule = chronological_splits(df["t_ms"], ml["split_days"], ml["split_fallback_fractions"])
    cc = ml["classification"]
    threshold_db = float(config.get_exposure_limits()["criteria"][cc["criterion"]]["criterion_db"])
    results = {"split_rule": rule, "splits": [s.describe() for s in splits], "model": c["model"], "grid": c["grid"],
               "random_seed": seed, "feature_sets": FEATURE_SETS, "bootstrap": {
                   "resamples": c["bootstrap_resamples"], "block_min": c["bootstrap_block_min"],
                   "ci": c["bootstrap_ci"]},
               "history": {"long_window_min": c["history_long_window_min"],
                           "machine_window_min": c["history_machine_window_min"]},
               "threshold_db": threshold_db, "max_false_positive_rate": cc["max_false_positive_rate"],
               "forecasting": forecasting_ablation(df, splits, c, ml["horizons_min"], seed),
               "classification": classification_ablation(df, splits, c, cc["horizons_min"], seed, threshold_db,
                                                         float(cc["max_false_positive_rate"]))}
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


# ---------------------------------------------------------------- figure and report


def write_figure(results: dict[str, Any], path: Path) -> Path:
    """Grouped bars: forecast skill and classification PR-AUC by feature set, 95% bootstrap bars."""
    import matplotlib.pyplot as plt

    import figstyle

    figstyle.apply()
    sets = list(FEATURE_SETS)
    colours = {"model_1": "#BDBDBD", "model_2": "#56B4E9", "model_3": "#0072B2"}
    hatches = {"model_1": "", "model_2": "//", "model_3": ".."}
    fc, cl = results["forecasting"], results["classification"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={"width_ratios": [len(fc), len(cl)]})
    width = 0.27
    for ax, block, metric in ((ax1, fc, "skill"), (ax2, cl, "pr_auc")):
        keys = list(block)
        x = np.arange(len(keys))
        for k, s in enumerate(sets):
            if metric == "skill":
                vals = [block[key]["test"][s]["skill_vs_persistence"] for key in keys]
                lo = [block[key]["skill_interval"][s]["lo"] for key in keys]
                hi = [block[key]["skill_interval"][s]["hi"] for key in keys]
                err = np.array([np.subtract(vals, lo), np.subtract(hi, vals)])
            else:
                vals = [block[key]["test"][s]["pr_auc"] for key in keys]
                err = None
            ax.bar(x + (k - 1) * width, vals, width, yerr=err, capsize=3, color=colours[s], hatch=hatches[s],
                   edgecolor="black", linewidth=0.6, label=SET_LABELS[s])
        ax.set_xticks(x)
        ax.set_xticklabels([key.replace("_h", "\nt+").replace("_dba", "").replace("_ugm3", "").replace("_c\n", "\n")
                            for key in keys], fontsize=11)
        ax.axhline(0, color="black", linewidth=0.8)
    ax1.set(ylabel="Skill vs persistence (1 − MAE/MAE_pers.)", title="Forecasting: skill by feature set")
    ax2.set(ylabel="PR-AUC", ylim=(0, 1.05), title="Exceedance classification: PR-AUC")
    ax1.legend(loc="upper right", fontsize=10)
    figstyle.simulated_badge(ax2)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def _ci(iv: dict[str, Any], fmt: str = "+.3f") -> str:
    if iv["lo"] is None:
        return "undefined"
    return f"{iv['point']:{fmt}} [{iv['lo']:{fmt}}, {iv['hi']:{fmt}}]"


def write_report(results: dict[str, Any], path: Path, discussion_path: Path | None = None) -> None:
    """docs/results_ablation.md."""
    fs = results["feature_sets"]
    b = results["bootstrap"]
    L = ["# Model 1 / 2 / 3 ablation (Objective 6)", "",
         "**SIMULATED DATA.** Generated by `ml/ablation.py` (`--report-only` rebuilds this file and the figure "
         "from `data/ablation_metrics.json`).", "",
         "**Reframing.** The project document frames Models 1/2/3 as *risk* prediction. That cannot be evaluated "
         "honestly without clinical or questionnaire labels: any risk label available now is a formula of the "
         "inputs, so the comparison would only measure how well each set reproduces the formula. The same "
         "research question — *does worker-specific and machine data improve prediction over environmental data "
         "alone?* — is answered here on **hazard forecasting** and **exceedance classification**, whose labels "
         "are future measurements.", "",
         "## Feature sets (strictly nested)", "",
         "Defined once, as explicit column lists, in `ml/ablation.py` (`FEATURE_SETS`); "
         "`tests/test_ablation.py` asserts strict containment.", "",
         "| Set | n features | Adds |", "|---|---|---|",
         f"| Model 1 — environmental | {len(fs['model_1'])} | noise, PM2.5, WBGT with lags 1–30 min, rolling "
         "mean/std 5/15/30 min, 5-min rate of change; temperature, RH, CO2 and 15-min means; time of day "
         "(sin/cos); zone identity |",
         f"| Model 2 — + worker | {len(fs['model_2'])} | `workers_present_count`, `shift_code`, `is_break` |",
         f"| Model 3 — + machine & history | {len(fs['model_3'])} | `machines_running_count`, `mean_current_a`, "
         f"`max_vib_rms_g`; zone history over trailing windows: {results['history']['long_window_min']}-min "
         "energy-averaged noise (`noise_dba_laeq_long`), mean PM2.5 (`pm25_ugm3_rmean_long`), fraction of the "
         f"last {results['history']['machine_window_min']} min with machines running (`machines_running_frac`) |",
         "", "**Why zone history, not per-worker cumulative exposure.** The per-worker 7/30-day exposure tables are "
         "materialised per worker-day after the day ends; at minute t they would contain the future. The "
         "trailing-window zone history is their leak-free counterpart.", "",
         "**Why zone identity is in Model 1.** All three models pool every zone; without zone identity Model 1 "
         "could not tell the weaving hall from the office, and the comparison would measure that, not the value "
         "of worker or machine data.", "",
         "## Fairness and significance", "",
         f"- Same model class ({results['model']}), same split, same seed ({results['random_seed']}) for all "
         "three sets.", f"- {results['split_rule']}:"]
    L += [f"  - {s}" for s in results["splits"]]
    L += [f"- The same grid `{json.dumps(results['grid'])}` is searched **separately for each set** on the "
          "validation day (forecasting: validation MAE; classification: validation PR-AUC). The chosen parameters "
          "are listed per set below; they differ where the search said they should.",
          f"- Classification thresholds: highest validation recall with FPR ≤ "
          f"{results['max_false_positive_rate']:.0%}, chosen per set (as in `results_classification.md`).",
          f"- **Block bootstrap** on the test day: {b['resamples']} resamples of (zone, {b['block_min']}-min) blocks, "
          f"{b['ci']:.0%} percentile interval on the **difference** between two sets, recomputed on identical "
          "rows. Blocks keep autocorrelated minutes together; resampling single minutes would make the interval "
          "falsely narrow. **If the interval contains 0, the difference is not statistically distinguishable on "
          "this dataset.**", "",
          "## Forecasting (skill vs persistence; higher is better)", "",
          "| Target | h | Set | Chosen params | **Skill** [95% CI] | MAE | RMSE |", "|---|---|---|---|---|---|---|"]
    for key, e in results["forecasting"].items():
        t, h = key.rsplit("_h", 1)
        for s in fs:
            x, p = e["test"][s], e["tuning"][s]["chosen"]
            iv = e["skill_interval"][s]
            L.append(f"| {t} | {h} | {s} | depth {p['max_depth']}, lr {p['learning_rate']}, {p['best_iteration']} "
                     f"trees | **{x['skill_vs_persistence']:+.3f}** [{iv['lo']:+.3f}, {iv['hi']:+.3f}] | "
                     f"{x['mae']:.2f} | {x['rmse']:.2f} |")
    L += ["", "### Differences in skill (95% block-bootstrap CI)", "",
          "| Target | h | Model 2 − Model 1 | Model 3 − Model 1 | Model 3 − Model 2 |", "|---|---|---|---|---|"]
    for key, e in results["forecasting"].items():
        t, h = key.rsplit("_h", 1)
        cells = [f"{_ci(e['differences'][f'{a}_vs_{b}'])} — {verdict(e['differences'][f'{a}_vs_{b}'])}"
                 for a, b in PAIRS]
        L.append(f"| {t} | {h} | " + " | ".join(cells) + " |")
    L += ["", "## Exceedance classification (noise ≥ 90 dBA at t+h)", "",
          "| Variant | h | Prevalence | Set | Chosen params | **PR-AUC** | F1 | Recall | Precision | FPR |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for key, e in results["classification"].items():
        v, h = key.rsplit("_h", 1)
        for s in fs:
            x, p = e["test"][s], e["tuning"][s]["chosen"]
            L.append(f"| {v} | {h} | {e['prevalence_test']:.1%} | {s} | depth {p['max_depth']}, lr "
                     f"{p['learning_rate']}, {p['best_iteration']} trees | **{x['pr_auc']:.3f}** | "
                     f"{x['f1'] or 0:.3f} | {x['recall'] or 0:.3f} | {x['precision'] or 0:.3f} | {x['fpr']:.3f} |")
    L += ["", "### Differences (95% block-bootstrap CI)", "",
          "| Variant | h | Metric | Model 2 − Model 1 | Model 3 − Model 1 | Model 3 − Model 2 |",
          "|---|---|---|---|---|---|"]
    for key, e in results["classification"].items():
        v, h = key.rsplit("_h", 1)
        for metric in ("pr_auc", "f1"):
            cells = [f"{_ci(e['differences'][f'{a}_vs_{b}'][metric])} — "
                     f"{verdict(e['differences'][f'{a}_vs_{b}'][metric])}" for a, b in PAIRS]
            L.append(f"| {v} | {h} | {metric.upper().replace('_', '-')} | " + " | ".join(cells) + " |")
    L += ["", "## Per-zone breakdown", "",
          "A gain that exists in only one zone is visible here rather than hidden in the pooled average. Zones "
          "where persistence is perfect (skill undefined) or with one class only (PR-AUC undefined) are omitted.",
          "", "### Forecasting skill per zone", "",
          "| Target | h | Zone | n | Skill M1 | Skill M2 | Skill M3 | M2 − M1 [CI] | M3 − M1 [CI] |",
          "|---|---|---|---|---|---|---|---|---|"]
    for key, e in results["forecasting"].items():
        t, h = key.rsplit("_h", 1)
        for z, zz in e["by_zone"].items():
            sk = zz["skill"]
            d21, d31 = zz["differences"]["model_2_vs_model_1"], zz["differences"]["model_3_vs_model_1"]
            L.append(f"| {t} | {h} | {z} | {zz['n']:,} | {sk['model_1']:+.3f} | {sk['model_2']:+.3f} | "
                     f"{sk['model_3']:+.3f} | {_ci(d21)} {_mark(d21)} | {_ci(d31)} {_mark(d31)} |")
    L += ["", "### Classification PR-AUC per zone", "",
          "| Variant | h | Zone | n | Prevalence | PR-AUC M1 | PR-AUC M2 | PR-AUC M3 | M2 − M1 [CI] | M3 − M1 [CI] |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for key, e in results["classification"].items():
        v, h = key.rsplit("_h", 1)
        for z, zz in e["by_zone"].items():
            pa = zz["pr_auc"]
            d21, d31 = zz["differences"]["model_2_vs_model_1"], zz["differences"]["model_3_vs_model_1"]
            L.append(f"| {v} | {h} | {z} | {zz['n']:,} | {zz['prevalence']:.1%} | {pa['model_1']:.3f} | "
                     f"{pa['model_2']:.3f} | {pa['model_3']:.3f} | {_ci(d21)} {_mark(d21)} | "
                     f"{_ci(d31)} {_mark(d31)} |")
    L += ["", "✓ = interval excludes 0 in favour of the larger set; ✗ = excludes 0 against it; "
          "~ = not distinguishable.", "", "Figure: `docs/figures/ablation_skill.png` (forecast skill with 95% "
          "bootstrap bars, and classification PR-AUC, by feature set).", ""]
    if discussion_path is not None and discussion_path.exists():
        L += [discussion_path.read_text(encoding="utf-8").strip(), ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L), encoding="utf-8")


def _mark(iv: dict[str, Any]) -> str:
    return {"better": "✓", "worse": "✗", "not distinguishable": "~"}.get(verdict(iv), "")


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = config.CONFIG_PATH.parent
    ml = config.get_ml_config()
    p = argparse.ArgumentParser(prog="python -m ml.ablation", description=__doc__.split("\n")[0])
    p.add_argument("--features", type=Path, default=root / ml["features_path"])
    p.add_argument("--metrics", type=Path, default=root / ml["ablation"]["metrics_path"])
    p.add_argument("--figure", type=Path, default=root / config.get_paths()["figures_dir"] / "ablation_skill.png")
    p.add_argument("--report", type=Path, default=root / "docs" / "results_ablation.md")
    p.add_argument("--discussion", type=Path, default=root / "docs" / "text" / "results_ablation_discussion.md")
    p.add_argument("--report-only", action="store_true", help="skip training: rebuild figure and report")
    args = p.parse_args(argv)
    if args.report_only:
        results = json.loads(args.metrics.read_text(encoding="utf-8"))
    else:
        results = run(args.features, args.metrics)
    write_figure(results, args.figure)
    write_report(results, args.report, args.discussion)
    summary = {k: {s: round(v["test"][s]["skill_vs_persistence"], 3) for s in FEATURE_SETS}
               for k, v in results["forecasting"].items()}
    summary.update({k: {s: round(v["test"][s]["pr_auc"], 3) for s in FEATURE_SETS}
                    for k, v in results["classification"].items()})
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
