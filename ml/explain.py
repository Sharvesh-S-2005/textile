"""SHAP explanations of the noise t+30 forecaster (shap.TreeExplainer).

    python -m ml.explain

Writes to docs/figures/:
  shap_summary_noise_h30.png   beeswarm over a sample of the test day
  shap_bar_noise_h30.png       mean |SHAP| per feature
  shap_waterfall_hit.png       a correctly predicted 90 dBA crossing
  shap_waterfall_miss.png      a missed exceedance (see `pick_miss`)
and the per-family importance shares to data/shap_noise_h30.json.

TreeExplainer is used (fast and exact for tree ensembles). The model-agnostic
explainer is not. For a scikit-learn pipeline, the explainer runs on the forest
with the pipeline's (train-fitted) imputer applied to the inputs.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap

import config
import figstyle
import matplotlib.pyplot as plt
from ml.forecast import load_model

log = logging.getLogger("ml.explain")
TARGET, HORIZON = "noise_dba", 30
SAMPLE_ROWS = 2000

VAR_LABEL = {"noise_dba": "noise", "pm25_ugm3": "PM2.5", "wbgt_c": "WBGT", "temp_c": "temp", "rh_pct": "RH",
             "co2_ppm": "CO2"}
FIXED_LABEL = {"machines_running_count": "machines running", "mean_current_a": "mean machine current",
               "max_vib_rms_g": "max machine vibration", "workers_present_count": "workers present",
               "minute_sin": "time of day (sin)", "minute_cos": "time of day (cos)", "shift_code": "shift",
               "is_break": "break period", "zone_code": "zone"}


def readable(name: str) -> str:
    """Plain-language feature label for slides."""
    if name in FIXED_LABEL:
        return FIXED_LABEL[name]
    m = re.fullmatch(r"(\w+?)_(lag|rmean|rstd|roc)(\d+)", name)
    if m:
        var, kind, n = VAR_LABEL.get(m.group(1), m.group(1)), m.group(2), m.group(3)
        return {"lag": f"{var}, {n} min ago", "rmean": f"{var}, {n}-min mean", "rstd": f"{var}, {n}-min std",
                "roc": f"{var}, {n}-min change"}[kind]
    return f"{VAR_LABEL.get(name, name)} now"


def family(name: str) -> str:
    """Feature family, for the importance-share summary."""
    if name.startswith(("noise_dba", "pm25_ugm3", "wbgt_c")):
        return "recent hazard values (lags, rolling stats)"
    if name in ("machines_running_count", "mean_current_a", "max_vib_rms_g"):
        return "machine state"
    if name in ("minute_sin", "minute_cos", "shift_code", "is_break"):
        return "schedule (time of day, shift, break)"
    if name == "workers_present_count":
        return "occupancy"
    if name == "zone_code":
        return "zone"
    return "ambient (temp, RH, CO2)"


def explainer_inputs(model: Any, X: pd.DataFrame) -> tuple[Any, np.ndarray]:
    """(tree model, matrix it sees) for a bare tree model or an imputer+forest pipeline."""
    if hasattr(model, "steps"):
        *pre, (_, tree) = model.steps
        Xt = X
        for _, step in pre:
            Xt = step.transform(Xt)
        return tree, np.asarray(Xt, dtype=float)
    return model, X.to_numpy(dtype=float)


def pick_hit(test: pd.DataFrame, threshold: float, pred_col: str) -> int | None:
    """A row where the noise was below threshold and the model correctly forecast >= threshold."""
    ok = test[(test["noise_dba"] < threshold) & (test[pred_col] >= threshold) & (test["actual"] >= threshold)]
    return None if ok.empty else int(ok.index[np.argmax((ok[pred_col] - threshold).to_numpy())])


def pick_miss(test: pd.DataFrame, threshold: float, pred_col: str) -> int | None:
    """A missed exceedance: below threshold now, actual >= threshold at t+h, forecast < threshold.

    (On the test day every sustained crossing was warned, so a 'miss' is the issue
    time with the largest under-forecast among exceedances not foreseen at that time.)
    """
    bad = test[(test["noise_dba"] < threshold) & (test["actual"] >= threshold) & (test[pred_col] < threshold)]
    return None if bad.empty else int(bad.index[np.argmax((bad["actual"] - bad[pred_col]).to_numpy())])


def run(features_path: Path, predictions_path: Path, model_path: Path, figures_dir: Path,
        out_json: Path) -> dict[str, Any]:
    """Compute SHAP values for the noise t+30 model and write the figures."""
    figstyle.apply()
    model, meta = load_model(model_path)
    feats = meta["features"]
    threshold = float(config.get_exposure_limits()["criteria"][config.get_ml_config()["exceedance_criterion"]][
        "criterion_db"])
    preds = pd.read_parquet(predictions_path)
    preds = preds[(preds["target"] == TARGET) & (preds["horizon_min"] == HORIZON)]
    features = pd.read_parquet(features_path)
    features["zone_id"] = features["zone_id"].astype(str)
    test = preds.merge(features[["zone_id", "t_ms", *feats]], on=["zone_id", "t_ms"], how="left",
                       validate="one_to_one").reset_index(drop=True)
    X_test = test[feats]
    tree, Xt = explainer_inputs(model, X_test)
    explainer = shap.TreeExplainer(tree)
    rng = np.random.default_rng(int(config.get_ml_config()["random_seed"]))
    sample = np.sort(rng.choice(len(test), size=min(SAMPLE_ROWS, len(test)), replace=False))
    labels = [readable(f) for f in feats]
    values = explainer.shap_values(Xt[sample], check_additivity=False)
    figures_dir.mkdir(parents=True, exist_ok=True)
    noise_cmap = figstyle.sequential_cmap("noise")

    shap.summary_plot(values, Xt[sample], feature_names=labels, max_display=15, cmap=noise_cmap, show=False,
                      plot_size=(11, 7))
    ax = plt.gca()
    ax.set_xlabel("SHAP value: effect on forecast noise at t+30 (dBA)")
    ax.set_title(f"What drives the 30-minute noise forecast ({meta['model'].replace('_', ' ')}, test day, "
                 f"{len(sample):,} rows)")
    figstyle.simulated_badge(ax)
    plt.savefig(figures_dir / "shap_summary_noise_h30.png")
    plt.close("all")

    mean_abs = np.abs(values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:15][::-1]
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh([labels[i] for i in order], mean_abs[order], color=figstyle.HAZARD_COLOURS["noise"])
    ax.set_xlabel("mean |SHAP value| (dBA)")
    ax.set_title("Mean absolute importance, noise forecast at t+30")
    figstyle.simulated_badge(ax)
    fig.savefig(figures_dir / "shap_bar_noise_h30.png")
    plt.close(fig)

    picks = {"hit": pick_hit(test, threshold, "chosen"), "miss": pick_miss(test, threshold, "chosen")}
    waterfalls = {}
    for name, idx in picks.items():
        if idx is None:
            log.warning("no %s case found for a waterfall plot", name)
            continue
        row = Xt[idx:idx + 1]
        sv = explainer.shap_values(row, check_additivity=False)[0]
        base = float(np.ravel(explainer.expected_value)[0])
        exp = shap.Explanation(values=sv, base_values=base, data=row[0], feature_names=labels)
        plt.figure(figsize=(11, 7))
        shap.plots.waterfall(exp, max_display=12, show=False)
        r = test.iloc[idx]
        outcome = "correctly forecast crossing" if name == "hit" else "missed exceedance"
        plt.title(f"{outcome}: {r['zone_id']} at {r['ts']} — now {r['noise_dba']:.1f}, forecast "
                  f"{r['chosen']:.1f}, actual {r['actual']:.1f} dBA (threshold {threshold:.0f})", fontsize=13)
        figstyle.simulated_badge(plt.gca())
        plt.savefig(figures_dir / f"shap_waterfall_{name}.png")
        plt.close("all")
        top = np.argsort(np.abs(sv))[::-1][:4]
        waterfalls[name] = {"zone_id": r["zone_id"], "ts": r["ts"], "now_db": float(r["noise_dba"]),
                            "forecast_db": float(r["chosen"]), "actual_db": float(r["actual"]), "base_db": base,
                            "top_features": [{"feature": labels[i], "shap_db": float(sv[i])} for i in top]}

    shares: dict[str, float] = {}
    for f, v in zip(feats, mean_abs):
        shares[family(f)] = shares.get(family(f), 0.0) + float(v)
    total = sum(shares.values())
    result = {"model": meta["model"], "target": TARGET, "horizon_min": HORIZON, "sample_rows": int(len(sample)),
              "top_features": [{"feature": labels[i], "mean_abs_shap_db": float(mean_abs[i])}
                               for i in np.argsort(mean_abs)[::-1][:10]],
              "family_share": {k: v / total for k, v in sorted(shares.items(), key=lambda kv: -kv[1])},
              "waterfalls": waterfalls}
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = config.CONFIG_PATH.parent
    c = config.get_ml_config()
    p = argparse.ArgumentParser(prog="python -m ml.explain", description=__doc__.split("\n")[0])
    p.add_argument("--features", type=Path, default=root / c["features_path"])
    p.add_argument("--predictions", type=Path, default=root / c["predictions_path"])
    p.add_argument("--model", type=Path, default=root / config.get_paths()["model_dir"] / f"forecast_{TARGET}_{HORIZON}.pkl")
    p.add_argument("--figures", type=Path, default=root / config.get_paths()["figures_dir"])
    p.add_argument("--out", type=Path, default=root / "data" / "shap_noise_h30.json")
    args = p.parse_args(argv)
    result = run(args.features, args.predictions, args.model, args.figures, args.out)
    print(json.dumps({"family_share": result["family_share"], "top": result["top_features"][:6]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
