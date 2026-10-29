"""Noise exceedance classification: will a zone's noise at t+h exceed the 90 dBA criterion?

    python -m ml.classify_exceedance [--features data/features.parquet] [--report-only]

What is predicted, and why the label is not circular
----------------------------------------------------
The label is the MEASURED 1-minute L_Aeq of the same zone at t+h (h = 30, 60 min),
compared with the India criterion level (exposure_limits.criteria.india, 90 dBA). It
comes from a future measurement, not from a rule applied to the features that
predict it. A dose-threshold "risk" label is circular (the label is a formula of the
inputs, so a model only re-learns the formula); this label is not, because nothing
known at t determines the measured level at t+h. It is the same kind of label as the
regression target in ml/forecast.py, made binary.

Two label variants:
  * above    - L(t+h) >= 90. Dominated by always-loud zones (Z4, Z5), where
               "always yes" scores high accuracy. Population: every row whose
               t+h measurement exists.
  * crossing - L(t) < 90 and L(t+h) >= 90: a quiet zone becomes hazardous within
               the horizon. Population: rows that are below the criterion at t (the
               only rows where a warning is meaningful). L(t) is used only to select
               the population; it is known at t, so this is not leakage.

Baseline: the persistence classifier ("above now -> above at t+h"). On `crossing`
it never predicts yes (every row is below now), so its recall is 0 by construction.

Models: persistence, logistic regression (the linear baseline), random forest and
XGBoost, on the 60 features of ml/features.py (unchanged). Each is tuned on the
validation day over the grid in config.yaml (ml.classification.grids), selected by
validation PR-AUC. The operating threshold is chosen on the validation day to
maximise recall subject to false-positive rate <= the configured ceiling.

Split: the chronological whole-day split of ml/forecast.py, with the same boundary
purge. Never shuffled. Prevalence is reported before every metric, and PR-AUC is
reported first, because on imbalanced classes accuracy and ROC-AUC flatter.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import config
from ml.features import feature_columns
from ml.forecast import MIN_MS, chronological_splits, save_model, schedule_mask, split_mask

log = logging.getLogger("ml.classify_exceedance")

MODELS = ("persistence", "logistic_regression", "random_forest", "xgboost")
LEARNED = MODELS[1:]
VARIANTS = ("above", "crossing")


# ---------------------------------------------------------------- labels


def future_value(df: pd.DataFrame, column: str, horizon_min: int) -> np.ndarray:
    """Value of `column` in the same zone at t + h (NaN where that bin is absent).

    Looks the value up by (zone_id, t_ms + h); it never reads a row at or before t.
    """
    keyed = pd.Series(df[column].to_numpy(dtype=float),
                      index=pd.MultiIndex.from_arrays([df["zone_id"].astype(str), df["t_ms"]]))
    want = pd.MultiIndex.from_arrays([df["zone_id"].astype(str), df["t_ms"] + horizon_min * MIN_MS])
    return keyed.reindex(want).to_numpy()


def make_labels(now: np.ndarray, future: np.ndarray, threshold: float, variant: str) -> tuple[np.ndarray, np.ndarray]:
    """Binary label and population mask for one label variant.

    Args:
        now: measured level at t (used only to select the `crossing` population).
        future: measured level at t + h (the label source).
        threshold: criterion level in dBA; "exceed" means >= threshold.
        variant: "above" or "crossing".

    Returns:
        (label, in_population). Rows outside the population have label 0 and must
        be ignored.
    """
    known = ~np.isnan(future)
    label = known & (np.nan_to_num(future, nan=-np.inf) >= threshold)
    if variant == "above":
        population = known
    elif variant == "crossing":
        population = known & ~np.isnan(now) & (np.nan_to_num(now, nan=np.inf) < threshold)
    else:
        raise ValueError(f"unknown label variant {variant!r}")
    return (label & population).astype(int), population


def prevalence(y: np.ndarray) -> float | None:
    """Fraction of positive labels (None for an empty set)."""
    return float(np.mean(y)) if len(y) else None


def persistence_scores(now: np.ndarray, threshold: float) -> np.ndarray:
    """Persistence classifier: 1.0 if the level at t is already >= threshold, else 0.0."""
    return (np.nan_to_num(now, nan=-np.inf) >= threshold).astype(float)


# ---------------------------------------------------------------- threshold and metrics


def choose_threshold(y: np.ndarray, scores: np.ndarray, max_fpr: float) -> float:
    """Score threshold with the highest recall subject to false-positive rate <= max_fpr.

    "Predict yes" means score >= threshold. Ties in recall go to the lowest false-
    positive rate (the higher threshold). If no finite threshold meets the ceiling,
    returns +inf (never alarm). Needs both classes in `y`.
    """
    if len(np.unique(y)) < 2:
        raise ValueError("threshold selection needs both classes")
    fpr, tpr, thr = roc_curve(y, scores, drop_intermediate=False)
    ok = fpr <= max_fpr + 1e-12
    best = np.flatnonzero(ok & (tpr == tpr[ok].max()))
    return float(thr[best[np.argmin(fpr[best])]])


def classification_metrics(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    """Prevalence, PR-AUC, ROC-AUC, F1, precision, recall, FPR, accuracy and confusion matrix.

    AUCs are None when only one class is present (they are undefined); precision is
    None when nothing is predicted positive.
    """
    y = np.asarray(y, dtype=int)
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y, pred, labels=[0, 1]).ravel())
    both = len(np.unique(y)) == 2
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else None
    return {"n": int(len(y)), "prevalence": prevalence(y),
            "pr_auc": float(average_precision_score(y, scores)) if both else None,
            "roc_auc": float(roc_auc_score(y, scores)) if both else None,
            "f1": f1, "precision": precision, "recall": recall,
            "fpr": fp / (fp + tn) if fp + tn else None,
            "accuracy": (tp + tn) / len(y) if len(y) else None,
            "threshold": threshold if np.isfinite(threshold) else "never",
            "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp}}


# ---------------------------------------------------------------- models and tuning


def expand_grid(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Every combination of a parameter grid, in a fixed order."""
    keys = list(grid)
    return [dict(zip(keys, combo)) for combo in itertools.product(*(grid[k] for k in keys))]


def make_classifier(name: str, params: dict[str, Any], seed: int, early_stopping: int) -> Any:
    """Untrained classifier. Imputation and scaling are fitted on training data only."""
    if name == "logistic_regression":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(max_iter=2000, random_state=seed, **params))
    if name == "random_forest":
        return make_pipeline(SimpleImputer(strategy="median"),
                             RandomForestClassifier(random_state=seed, n_jobs=-1, **params))
    if name == "xgboost":
        return XGBClassifier(random_state=seed, n_jobs=-1, eval_metric="aucpr",
                             early_stopping_rounds=early_stopping, **params)
    raise ValueError(f"unknown model {name!r}")


def fit_classifier(model: Any, name: str, X_train: pd.DataFrame, y_train: np.ndarray,
                   X_val: pd.DataFrame, y_val: np.ndarray) -> Any:
    """Fit; XGBoost early-stops on the validation set (the other models ignore it)."""
    if name == "xgboost":
        return model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model.fit(X_train, y_train)


def tune(name: str, grid: dict[str, list[Any]], X_train: pd.DataFrame, y_train: np.ndarray,
         X_val: pd.DataFrame, y_val: np.ndarray, seed: int, early_stopping: int) -> tuple[Any, dict[str, Any],
                                                                                        list[dict[str, Any]]]:
    """Grid search on the validation set; selection metric PR-AUC (average precision).

    Returns the best fitted model, its parameters, and the full search log. The test
    set is never seen here.
    """
    best, best_params, best_score, search = None, {}, -np.inf, []
    for params in expand_grid(grid):
        model = fit_classifier(make_classifier(name, params, seed, early_stopping), name,
                               X_train, y_train, X_val, y_val)
        score = float(average_precision_score(y_val, model.predict_proba(X_val)[:, 1]))
        search.append({"params": params, "val_pr_auc": score})
        if score > best_score:
            best, best_params, best_score = model, params, score
    if name == "xgboost":
        best_params = dict(best_params, best_iteration=int(best.best_iteration))
    return best, best_params, search


# ---------------------------------------------------------------- pipeline


def _per_zone(zones: np.ndarray, y: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    return {str(z): classification_metrics(y[zones == z], scores[zones == z], threshold)
            for z in sorted(np.unique(zones))}


def run(features_path: Path, model_dir: Path, predictions_path: Path, metrics_path: Path) -> dict[str, Any]:
    """Tune, threshold and evaluate every model for every horizon and label variant."""
    ml = config.get_ml_config()
    c = ml["classification"]
    seed = int(ml["random_seed"])
    threshold_db = float(config.get_exposure_limits()["criteria"][c["criterion"]]["criterion_db"])
    max_fpr = float(c["max_false_positive_rate"])
    es = int(c["xgboost_early_stopping_rounds"])
    df = pd.read_parquet(features_path)
    feats = feature_columns()
    splits, rule = chronological_splits(df["t_ms"], ml["split_days"], ml["split_fallback_fractions"])
    now = df["noise_dba"].to_numpy(dtype=float)
    results: dict[str, Any] = {"split_rule": rule, "splits": [s.describe() for s in splits],
                               "criterion": c["criterion"], "threshold_db": threshold_db,
                               "max_false_positive_rate": max_fpr, "grids": c["grids"], "random_seed": seed,
                               "n_features": len(feats), "runs": {}}
    predictions = []
    for h in c["horizons_min"]:
        future = future_value(df, "noise_dba", h)
        check_against_target_column(future, df[f"target_noise_dba_h{h}"].to_numpy(dtype=float))
        for variant in c["variants"]:
            key = f"{variant}_h{h}"
            y_all, pop = make_labels(now, future, threshold_db, variant)
            idx = {s.name: np.flatnonzero(split_mask(df["t_ms"], s, h).to_numpy() & pop) for s in splits}
            X = {k: df.iloc[v][feats] for k, v in idx.items()}
            y = {k: y_all[v] for k, v in idx.items()}
            zones = {k: df.iloc[v]["zone_id"].astype(str).to_numpy() for k, v in idx.items()}
            entry: dict[str, Any] = {"n": {k: int(len(v)) for k, v in idx.items()},
                                     "prevalence": {k: prevalence(v) for k, v in y.items()},
                                     "prevalence_by_zone": {k: {z: prevalence(y[k][zones[k] == z])
                                                                for z in sorted(np.unique(zones[k]))} for k in y},
                                     "tuning": {}, "test": {}, "test_by_zone": {}, "validation_pr_auc": {}}
            scores_test = {"persistence": persistence_scores(now[idx["test"]], threshold_db)}
            thresholds = {"persistence": 0.5}
            fitted = {}
            for name in LEARNED:
                model, params, search = tune(name, c["grids"][name], X["train"], y["train"],
                                             X["validation"], y["validation"], seed, es)
                val_scores = model.predict_proba(X["validation"])[:, 1]
                thresholds[name] = choose_threshold(y["validation"], val_scores, max_fpr)
                entry["tuning"][name] = {"chosen": params, "search": search,
                                         "validation_fpr_at_threshold":
                                             classification_metrics(y["validation"], val_scores,
                                                                    thresholds[name])["fpr"]}
                entry["validation_pr_auc"][name] = float(average_precision_score(y["validation"], val_scores))
                scores_test[name] = model.predict_proba(X["test"])[:, 1]
                fitted[name] = (model, params)
                log.info("%s %s: chosen %s, val PR-AUC %.3f", key, name, params, entry["validation_pr_auc"][name])
            chosen = max(entry["validation_pr_auc"], key=entry["validation_pr_auc"].get)
            entry["chosen"] = chosen
            model, params = fitted[chosen]
            save_model(model, model_dir / f"classify_{key}.pkl",
                       {"task": "noise exceedance classification", "variant": variant, "horizon_min": h,
                        "model": chosen, "params": params, "threshold": thresholds[chosen],
                        "criterion_db": threshold_db, "features": feats, "split_rule": rule,
                        "splits": [s.describe() for s in splits]})
            for name in MODELS:
                entry["test"][name] = classification_metrics(y["test"], scores_test[name], thresholds[name])
                entry["test_by_zone"][name] = _per_zone(zones["test"], y["test"], scores_test[name], thresholds[name])
            results["runs"][key] = entry
            test_rows = df.iloc[idx["test"]]
            predictions.append(pd.DataFrame({
                "t_ms": test_rows["t_ms"].to_numpy(), "ts": test_rows["ts"].to_numpy(), "zone_id": zones["test"],
                "horizon_min": h, "variant": variant, "label": y["test"], "noise_now": now[idx["test"]],
                "noise_future": future[idx["test"]], **{f"score_{m}": scores_test[m] for m in MODELS},
                **{f"pred_{m}": (scores_test[m] >= thresholds[m]).astype(int) for m in MODELS},
                "chosen_model": chosen}))
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(predictions, ignore_index=True).to_parquet(predictions_path, index=False)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def check_against_target_column(future: np.ndarray, target: np.ndarray) -> None:
    """Fail loudly if the label source disagrees with the forecasting target column.

    Both must be the measured zone noise at t + h; a mismatch means one of them is
    reading the wrong time.
    """
    both = ~np.isnan(future) & ~np.isnan(target)
    if not both.any() or not np.allclose(future[both], target[both]):
        raise ValueError("future noise lookup disagrees with target_noise_dba_h* in the feature table")


def schedule_breakdown(predictions: pd.DataFrame, features: pd.DataFrame,
                       thresholds: dict[str, dict[str, float]]) -> dict[str, Any]:
    """Test metrics on "steady" rows (no break or shift boundary in (t, t+h]) and on rows
    whose horizon spans one. Evaluation-only slicing; no model or threshold changes.

    `thresholds` maps run key -> model -> operating threshold (from the saved results).
    """
    schedule = features.assign(zone_id=features["zone_id"].astype(str)).set_index(["zone_id", "t_ms"])[
        ["is_break", "shift_code"]]
    out: dict[str, Any] = {}
    for (variant, h), g in predictions.groupby(["variant", "horizon_min"]):
        key = f"{variant}_h{h}"
        steady = schedule_mask(g, schedule, int(h))
        out[key] = {}
        for name, mask in (("steady", steady), ("schedule_transition", ~steady)):
            sub = g[mask]
            y = sub["label"].to_numpy()
            out[key][name] = {m: classification_metrics(y, sub[f"score_{m}"].to_numpy(), thresholds[key][m])
                              for m in MODELS}
    return out


# ---------------------------------------------------------------- figures


def write_figures(predictions: pd.DataFrame, results: dict[str, Any], figures_dir: Path) -> list[Path]:
    """PR and ROC curves per variant/horizon, and confusion matrices of the chosen models."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve

    import figstyle

    figstyle.apply()
    figures_dir.mkdir(parents=True, exist_ok=True)
    styles = {"persistence": {"color": figstyle.NEUTRAL},
              "logistic_regression": {"color": "#009E73", "linestyle": ":"},
              "random_forest": {"color": "#E69F00", "linestyle": "-."},
              "xgboost": {"color": "#0072B2", "linestyle": "-"}}
    out = []
    for key, entry in results["runs"].items():
        variant, h = key.rsplit("_h", 1)
        g = predictions[(predictions["variant"] == variant) & (predictions["horizon_min"] == int(h))]
        y = g["label"].to_numpy()
        fig, (ax_pr, ax_roc) = plt.subplots(1, 2, figsize=(13, 5.5))
        for m in MODELS:
            s = g[f"score_{m}"].to_numpy()
            met = entry["test"][m]
            if met["pr_auc"] is None:
                continue
            if m == "persistence":
                # A yes/no rule has one operating point, not a curve; a line would be an interpolation artefact.
                ax_pr.plot([met["recall"]], [met["precision"] if met["precision"] is not None else met["prevalence"]],
                           linestyle="none", markersize=9, color=styles[m]["color"], marker="o",
                           label=f"{m}: one operating point (PR-AUC {met['pr_auc']:.2f})")
                ax_roc.plot([met["fpr"]], [met["recall"]], linestyle="none", markersize=9, color=styles[m]["color"],
                            marker="o", label=f"{m}: one operating point (ROC-AUC {met['roc_auc']:.2f})")
                continue
            p, r, _ = precision_recall_curve(y, s)
            fpr, tpr, _ = roc_curve(y, s)
            ax_pr.plot(r, p, label=f"{m} (PR-AUC {met['pr_auc']:.2f})", **styles[m])
            ax_roc.plot(fpr, tpr, label=f"{m} (ROC-AUC {met['roc_auc']:.2f})", **styles[m])
        prev = entry["prevalence"]["test"]
        ax_pr.axhline(prev, color=figstyle.LIGHT, linewidth=1.5)
        ax_pr.text(0.98, prev - 0.06, f"prevalence {prev:.1%}", color=figstyle.NEUTRAL, fontsize=11, ha="right")
        ax_roc.plot([0, 1], [0, 1], color=figstyle.LIGHT, linewidth=1.5)
        ax_roc.axvline(results["max_false_positive_rate"], color=figstyle.NEUTRAL, linewidth=1, linestyle=":")
        ax_pr.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02),
                  title=f"Precision-recall: {variant}, t+{h} min")
        ax_roc.set(xlabel="False-positive rate", ylabel="True-positive rate (recall)", xlim=(0, 1), ylim=(0, 1.02),
                   title=f"ROC: {variant}, t+{h} min")
        ax_pr.legend(loc="center left", fontsize=10)
        ax_roc.legend(loc="lower right", fontsize=10)
        figstyle.simulated_badge(ax_roc)
        path = figures_dir / f"classification_pr_roc_{variant}_h{h}.png"
        fig.savefig(path)
        plt.close(fig)
        out.append(path)

    keys = list(results["runs"])
    ncol = 2
    nrow = (len(keys) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(11, 4.8 * nrow), gridspec_kw={"wspace": 0.45, "hspace": 0.45})
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[len(keys):]:
        ax.set_visible(False)
    for ax, key in zip(axes, keys):
        entry = results["runs"][key]
        m = entry["chosen"]
        cm = entry["test"][m]["confusion"]
        grid = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
        ax.imshow(grid, cmap=figstyle.sequential_cmap("noise"))
        for (i, j), v in np.ndenumerate(grid):
            ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=13,
                    color="white" if v > grid.max() / 2 else "black")
        variant, h = key.rsplit("_h", 1)
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["pred no", "pred yes"],
               yticklabels=["actual no", "actual yes"], title=f"{variant}, t+{h} min ({m})")
        ax.grid(False)
    figstyle.simulated_badge(axes[ncol - 1])
    path = figures_dir / "classification_confusion.png"
    fig.savefig(path)
    plt.close(fig)
    out.append(path)
    return out


# ---------------------------------------------------------------- report


def _f(v: Any, fmt: str = ".3f") -> str:
    if v is None:
        return "n/a"
    if isinstance(v, str):
        return v
    return format(v, fmt)


def write_report(results: dict[str, Any], path: Path, discussion_path: Path | None = None) -> None:
    """docs/results_classification.md: prevalence first, then metrics; PR-AUC leads."""
    thr = results["threshold_db"]
    L = ["# Noise exceedance classification results", "",
         "**SIMULATED DATA.** Generated by `ml/classify_exceedance.py` (`--report-only` rebuilds this file "
         "and the figures from saved outputs).", "",
         f"**Question.** Will a zone's measured 1-minute L_Aeq at t+h be at or above {thr:.0f} dBA "
         f"({results['criterion']} criterion)? h = 30 and 60 min.", "",
         "**Why this label is not circular.** The label is a *future measurement* of the same zone, not a "
         "rule applied to the features that predict it. A dose-threshold risk label (e.g. \"dose > 100% ⇒ at "
         "risk\") is a formula of the inputs, so a model trained on it only re-learns the formula and scores "
         "~99% for no reason. Nothing known at t fixes the level at t+h, so a model here can be wrong, and its "
         "score measures real predictive skill.", "",
         "**Two label variants.** `above`: L(t+h) ≥ 90 on every row. `crossing`: L(t) < 90 *and* L(t+h) ≥ 90, "
         "evaluated only on rows below the criterion at t — the operationally useful warning (a quiet zone "
         "about to become hazardous) and a much rarer event.", "",
         "## 1. Prevalence (read this before any metric)", "",
         "Base rate of the positive class. On `above`, zones Z1, Z4 and Z5 are above 90 dBA most of the time, so "
         "an \"always yes\" classifier scores high accuracy there and means nothing. **No accuracy figure below "
         "is reportable without the prevalence beside it.** Z7 (office) never reaches 90 dBA, so its metrics "
         "are undefined (n/a).", "",
         "| Variant | h | Split | n | All zones | " + " | ".join(f"Z{i}" for i in range(1, 8)) + " |",
         "|---|---|---|---|---|" + "---|" * 7]
    for key, e in results["runs"].items():
        variant, h = key.rsplit("_h", 1)
        for split in ("train", "validation", "test"):
            pz = e["prevalence_by_zone"][split]
            L.append(f"| {variant} | {h} | {split} | {e['n'][split]:,} | {_f(e['prevalence'][split], '.1%')} | "
                     + " | ".join(_f(pz.get(f"Z{i}"), ".1%") for i in range(1, 8)) + " |")
    L += ["", "## 2. Split, tuning and operating threshold", "",
          f"Chronological, never shuffled, identical to the forecasting work — {results['split_rule']}. Rows "
          "whose t+h crosses the end of their split are purged.", ""]
    L += [f"- {s}" for s in results["splits"]]
    L += ["", f"One pooled model per variant and horizon over all zones ({results['n_features']} features from "
          f"`ml/features.py`, unchanged; zone identity is a feature). Seed {results['random_seed']}.", "",
          "**Tuning.** Each learned model is grid-searched on the validation day; selection metric validation "
          "PR-AUC. The test day is scored once, after every choice is fixed.", ""]
    for name, grid in results["grids"].items():
        L.append(f"- {name} grid: `{json.dumps(grid)}`")
    L += ["", "| Variant | h | Model | Chosen parameters | Validation PR-AUC |", "|---|---|---|---|---|"]
    for key, e in results["runs"].items():
        variant, h = key.rsplit("_h", 1)
        for name in LEARNED:
            mark = " (saved)" if name == e["chosen"] else ""
            L.append(f"| {variant} | {h} | {name}{mark} | `{json.dumps(e['tuning'][name]['chosen'])}` | "
                     f"{e['validation_pr_auc'][name]:.3f} |")
    L += ["", f"**Operating threshold.** Chosen on the validation day: the score threshold with the **highest "
          f"recall subject to false-positive rate ≤ {results['max_false_positive_rate']:.0%}** "
          "(`ml.classification.max_false_positive_rate`). Missing a worker's hazardous exposure costs more than a "
          "false alarm, so recall is what is maximised; the ceiling stops the classifier from alarming "
          "constantly. This is a stated policy choice, not a tuned value. The persistence classifier has no "
          "score to threshold (it is yes/no). The test-day FPR can differ from the ceiling because the "
          "threshold is fixed before the test day is seen.", "",
          "## 3. Test-day results (all zones)", "",
          "PR-AUC leads. Gain = model minus persistence on the same rows. A model that does not beat "
          "persistence has no result on that variant.", "",
          "| Variant | h | Prevalence | Model | **PR-AUC** | ROC-AUC | F1 | ΔF1 vs persistence | Precision | "
          "Recall | FPR | Accuracy | Threshold | TN | FP | FN | TP |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for key, e in results["runs"].items():
        variant, h = key.rsplit("_h", 1)
        f1_p = e["test"]["persistence"]["f1"] or 0.0
        for m in MODELS:
            x = e["test"][m]
            cm = x["confusion"]
            gain = "—" if m == "persistence" else _f((x["f1"] or 0.0) - f1_p, "+.3f")
            mark = " (saved)" if m == e["chosen"] else ""
            L.append(f"| {variant} | {h} | {_f(x['prevalence'], '.1%')} | {m}{mark} | **{_f(x['pr_auc'])}** | "
                     f"{_f(x['roc_auc'])} | {_f(x['f1'])} | {gain} | {_f(x['precision'])} | {_f(x['recall'])} | "
                     f"{_f(x['fpr'])} | {_f(x['accuracy'])} | {_f(x['threshold'], '.3f')} | {cm['tn']:,} | "
                     f"{cm['fp']:,} | {cm['fn']:,} | {cm['tp']:,} |")
    if any("by_schedule" in e for e in results["runs"].values()):
        L += ["", "### Steady rows vs rows spanning a scheduled break or shift change", "",
              "*Steady* = no break or shift boundary in (t, t+h]. *Transition* = the horizon spans one, where the "
              "fixed timetable makes the answer predictable (machines restart after every break). Same models and "
              "thresholds; evaluation-only split.", "",
              "| Variant | h | Subset | n | Prevalence | Model | **PR-AUC** | F1 | Recall | Precision | FPR |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for key, e in results["runs"].items():
            variant, h = key.rsplit("_h", 1)
            for subset, part in e.get("by_schedule", {}).items():
                for m in ("persistence", e["chosen"]):
                    x = part[m]
                    L.append(f"| {variant} | {h} | {subset} | {x['n']:,} | {_f(x['prevalence'], '.1%')} | {m} | "
                             f"**{_f(x['pr_auc'])}** | {_f(x['f1'])} | {_f(x['recall'])} | {_f(x['precision'])} | "
                             f"{_f(x['fpr'])} |")
    L += ["", "Figures: `docs/figures/classification_pr_roc_<variant>_h<h>.png` (PR and ROC curves, prevalence "
          "line on the PR panel, FPR ceiling on the ROC panel) and `docs/figures/classification_confusion.png` "
          "(confusion matrices of the saved model per variant/horizon).", ""]
    if discussion_path is not None and discussion_path.exists():
        L += [discussion_path.read_text(encoding="utf-8").strip(), ""]
    L += ["## Appendix: per-zone test-day results", "",
          "Same thresholds as above (chosen on the pooled validation day). n/a = undefined (one class only, or "
          "no positive predictions).", ""]
    for key, e in results["runs"].items():
        variant, h = key.rsplit("_h", 1)
        L += [f"### {variant}, t+{h} min", "",
              "| Zone | n | Prevalence | Model | **PR-AUC** | ROC-AUC | F1 | Precision | Recall | FPR | Accuracy | "
              "TN | FP | FN | TP |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for z in sorted(e["test_by_zone"]["persistence"]):
            for m in MODELS:
                x = e["test_by_zone"][m][z]
                cm = x["confusion"]
                L.append(f"| {z} | {x['n']:,} | {_f(x['prevalence'], '.1%')} | {m} | **{_f(x['pr_auc'])}** | "
                         f"{_f(x['roc_auc'])} | {_f(x['f1'])} | {_f(x['precision'])} | {_f(x['recall'])} | "
                         f"{_f(x['fpr'])} | {_f(x['accuracy'])} | {cm['tn']:,} | {cm['fp']:,} | {cm['fn']:,} | "
                         f"{cm['tp']:,} |")
        L.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = config.CONFIG_PATH.parent
    ml = config.get_ml_config()
    c = ml["classification"]
    p = argparse.ArgumentParser(prog="python -m ml.classify_exceedance", description=__doc__.split("\n")[0])
    p.add_argument("--features", type=Path, default=root / ml["features_path"])
    p.add_argument("--model-dir", type=Path, default=root / config.get_paths()["model_dir"])
    p.add_argument("--predictions", type=Path, default=root / c["predictions_path"])
    p.add_argument("--metrics", type=Path, default=root / c["metrics_path"])
    p.add_argument("--figures", type=Path, default=root / config.get_paths()["figures_dir"])
    p.add_argument("--report", type=Path, default=root / "docs" / "results_classification.md")
    p.add_argument("--discussion", type=Path,
                   default=root / "docs" / "text" / "results_classification_discussion.md")
    p.add_argument("--report-only", action="store_true", help="skip training: rebuild figures and report")
    args = p.parse_args(argv)
    if args.report_only:
        results = json.loads(args.metrics.read_text(encoding="utf-8"))
    else:
        results = run(args.features, args.model_dir, args.predictions, args.metrics)
    predictions = pd.read_parquet(args.predictions)
    thresholds = {k: {m: (np.inf if e["test"][m]["threshold"] == "never" else e["test"][m]["threshold"])
                      for m in MODELS} for k, e in results["runs"].items()}
    for key, part in schedule_breakdown(predictions, pd.read_parquet(args.features), thresholds).items():
        results["runs"][key]["by_schedule"] = part
    args.metrics.write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_figures(predictions, results, args.figures)
    write_report(results, args.report, args.discussion)
    print(json.dumps({k: {m: {"prevalence": round(e["test"][m]["prevalence"], 3),
                              "pr_auc": None if e["test"][m]["pr_auc"] is None else round(e["test"][m]["pr_auc"], 3),
                              "f1": None if e["test"][m]["f1"] is None else round(e["test"][m]["f1"], 3)}
                          for m in MODELS} for k, e in results["runs"].items()}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
