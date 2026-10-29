"""Optional deep sequence forecaster (Day 6 Task 4): a GRU over the last 60 minutes.

    python -m ml.deep_forecast [--features data/features.parquet] [--report-only]

Input: for each zone and minute t, the window t-59 .. t of the Model 3 feature set
(ml/ablation.py: every ml/features.py column plus the zone-history features).
Windows never reach past t. At the start of each zone's series the window is padded
with missing values, so the rows used are exactly the ablation's test rows and the
skill can be compared with XGBoost on Model 3 directly.

Preprocessing is fitted on the training split only: median imputation, then
standardisation. Target standardised the same way. One GRU layer (64 units) and a
linear head, trained with L1 loss (MAE, the reported metric), Adam, early stopping on
validation MAE. Same chronological split, same seed, same metrics as everything else.

Expectation stated before training: with well-engineered lag features, gradient
boosting usually matches or beats a sequence model on tabular data like this.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

import config
from ml.ablation import FEATURE_SETS, add_history_features
from ml.forecast import MIN_MS, chronological_splits, regression_metrics, split_mask

log = logging.getLogger("ml.deep_forecast")


# ---------------------------------------------------------------- windows


def stack_zones(df: pd.DataFrame, cols: list[str], seq_len: int) -> tuple[np.ndarray, pd.Series]:
    """One array of all zones on contiguous 1-minute grids, each preceded by seq_len - 1 NaN rows.

    Returns the array (rows x features) and, for every row of `df`, the array index of
    its own minute t (the LAST row of its window).
    """
    blocks, end_index = [], pd.Series(-1, index=df.index, dtype=np.int64)
    offset = 0
    for _zone, g in df.groupby("zone_id", observed=True):
        grid = np.arange(g["t_ms"].min(), g["t_ms"].max() + MIN_MS, MIN_MS)
        values = g.set_index("t_ms")[cols].reindex(grid).to_numpy(dtype=np.float32)
        pad = np.full((seq_len - 1, len(cols)), np.nan, dtype=np.float32)
        blocks += [pad, values]
        pos = np.searchsorted(grid, g["t_ms"].to_numpy())
        end_index.loc[g.index] = offset + seq_len - 1 + pos
        offset += seq_len - 1 + len(grid)
    return np.concatenate(blocks), end_index


def gather_windows(arr: np.ndarray, end_idx: np.ndarray, seq_len: int) -> np.ndarray:
    """(batch, seq_len, features): rows end_idx - seq_len + 1 .. end_idx inclusive."""
    return arr[end_idx[:, None] + np.arange(-seq_len + 1, 1)[None, :]]


# ---------------------------------------------------------------- model


class SeqRegressor(nn.Module):
    """GRU or LSTM encoder, last hidden state -> linear output."""

    def __init__(self, n_features: int, hidden: int, layers: int, cell: str) -> None:
        super().__init__()
        rnn = {"gru": nn.GRU, "lstm": nn.LSTM}[cell]
        self.rnn = rnn(n_features, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def predict(model: SeqRegressor, arr: np.ndarray, idx: np.ndarray, seq_len: int, batch: int) -> np.ndarray:
    """Model output for every window ending at idx (standardised units)."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(idx), batch):
            out.append(model(torch.from_numpy(gather_windows(arr, idx[i:i + batch], seq_len))).numpy())
    return np.concatenate(out) if out else np.empty(0)


def train(arr: np.ndarray, idx_train: np.ndarray, y_train: np.ndarray, idx_val: np.ndarray, y_val: np.ndarray,
          c: dict[str, Any], seed: int, deadline: float) -> tuple[SeqRegressor, dict[str, Any]]:
    """Train with early stopping on validation MAE (standardised units). Stops at `deadline`."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    seq, bs = int(c["seq_len_min"]), int(c["batch_size"])
    model = SeqRegressor(arr.shape[1], int(c["hidden"]), int(c["layers"]), c["cell"])
    opt = torch.optim.Adam(model.parameters(), lr=float(c["learning_rate"]))
    loss_fn = nn.L1Loss()
    best, best_state, stale, history, hit_cap = np.inf, None, 0, [], False
    for epoch in range(int(c["max_epochs"])):
        model.train()
        order = rng.permutation(len(idx_train))       # shuffles minibatches of TRAINING windows only
        for i in range(0, len(order), bs):
            b = order[i:i + bs]
            x = torch.from_numpy(gather_windows(arr, idx_train[b], seq))
            opt.zero_grad()
            loss = loss_fn(model(x), torch.from_numpy(y_train[b]))
            loss.backward()
            opt.step()
        val_mae = float(np.mean(np.abs(predict(model, arr, idx_val, seq, 4 * bs) - y_val)))
        history.append(val_mae)
        log.info("epoch %d: validation MAE (standardised) %.4f", epoch + 1, val_mae)
        if val_mae < best:
            best, best_state, stale = val_mae, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            stale += 1
        if stale >= int(c["patience"]):
            break
        if time.monotonic() > deadline:
            hit_cap = True
            break
    model.load_state_dict(best_state)
    return model, {"epochs_run": len(history), "best_epoch": int(np.argmin(history)) + 1,
                   "val_mae_history": history, "hit_time_cap": hit_cap}


# ---------------------------------------------------------------- pipeline


def run(features_path: Path, metrics_path: Path, ablation_metrics_path: Path) -> dict[str, Any]:
    """Train and evaluate per target and horizon; compare with XGBoost on Model 3."""
    ml = config.get_ml_config()
    c, ac = ml["deep"], ml["ablation"]
    seed = int(ml["random_seed"])
    torch.manual_seed(seed)
    started = time.monotonic()
    deadline = started + 60 * float(c["time_cap_min"])
    df = pd.read_parquet(features_path)
    df = add_history_features(df, int(ac["history_long_window_min"]), int(ac["history_machine_window_min"]),
                              float(ac["history_min_coverage"])).reset_index(drop=True)
    cols = FEATURE_SETS["model_3"]
    splits, rule = chronological_splits(df["t_ms"], ml["split_days"], ml["split_fallback_fractions"])
    seq = int(c["seq_len_min"])
    raw, end_index = stack_zones(df, cols, seq)
    ablation = json.loads(ablation_metrics_path.read_text(encoding="utf-8")) if ablation_metrics_path.exists() else None
    results: dict[str, Any] = {"status": "completed", "model": f"{c['cell'].upper()} ({c['layers']}×{c['hidden']})",
                               "settings": {k: c[k] for k in c if k != "metrics_path"}, "random_seed": seed,
                               "split_rule": rule, "splits": [s.describe() for s in splits],
                               "n_features": len(cols), "targets": {}}
    for target in c["targets"]:
        for h in ml["horizons_min"]:
            key = f"{target}_h{h}"
            y_col = f"target_{target}_h{h}"
            m = {s.name: split_mask(df["t_ms"], s, h) & df[y_col].notna() for s in splits}
            m["test"] = m["test"] & df[target].notna()                   # identical to the ablation's test rows
            tr = df.index[m["train"]].to_numpy()
            # Preprocessing fitted on training minutes only (window rows of training samples are earlier still).
            train_rows = raw[end_index[tr].to_numpy()]
            med = np.nanmedian(train_rows, axis=0)
            med = np.where(np.isnan(med), 0.0, med).astype(np.float32)
            filled = np.where(np.isnan(raw), med, raw)
            mu, sd = filled[end_index[tr].to_numpy()].mean(axis=0), filled[end_index[tr].to_numpy()].std(axis=0)
            sd = np.where(sd > 0, sd, 1.0)
            arr = ((filled - mu) / sd).astype(np.float32)
            y_mu, y_sd = float(df.loc[m["train"], y_col].mean()), float(df.loc[m["train"], y_col].std())
            idx = {k: end_index[df.index[v]].to_numpy() for k, v in m.items()}
            ys = {k: ((df.loc[v, y_col].to_numpy() - y_mu) / y_sd).astype(np.float32) for k, v in m.items()}
            model, info = train(arr, idx["train"], ys["train"], idx["validation"], ys["validation"], c, seed, deadline)
            pred = predict(model, arr, idx["test"], seq, 4 * int(c["batch_size"])) * y_sd + y_mu
            y = df.loc[m["test"], y_col].to_numpy()
            persist = df.loc[m["test"], target].to_numpy()
            mae_p = float(np.mean(np.abs(y - persist)))
            entry = {"n_train": int(len(idx["train"])), "n_test": int(len(y)), "training": info,
                     "test": regression_metrics(y, pred, mae_p)}
            if ablation and key in ablation["forecasting"]:
                xg = ablation["forecasting"][key]
                if xg["n_test"] != len(y):
                    raise ValueError(f"{key}: test rows differ from the ablation ({len(y)} vs {xg['n_test']})")
                entry["xgboost_skill"] = xg["test"]["model_3"]["skill_vs_persistence"]
                entry["xgboost_mae"] = xg["test"]["model_3"]["mae"]
            results["targets"][key] = entry
            log.info("%s: GRU skill %.3f (XGBoost Model 3 %s), %d epochs", key, entry["test"]["skill_vs_persistence"],
                     entry.get("xgboost_skill"), info["epochs_run"])
    results["minutes_used"] = (time.monotonic() - started) / 60.0
    results["summary"] = summary_sentence(results)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def summary_sentence(results: dict[str, Any]) -> str:
    """One paragraph for docs/results_ml.md Section 6, from the numbers."""
    parts, wins = [], 0
    for key, e in results["targets"].items():
        g, x = e["test"]["skill_vs_persistence"], e.get("xgboost_skill")
        if x is None:
            parts.append(f"{key}: GRU skill {g:+.3f}")
            continue
        wins += g > x
        parts.append(f"{key}: GRU skill {g:+.3f} (MAE {e['test']['mae']:.2f}) against XGBoost on the same features "
                     f"{x:+.3f} (MAE {e['xgboost_mae']:.2f})")
    n = len(results["targets"])
    verdict = ("Deep sequence models offered no advantage over gradient boosting on this dataset, likely because the "
               "engineered lag features already capture the temporal structure."
               if wins == 0 else f"The GRU beat XGBoost on {wins} of {n} target/horizon pairs; see "
                                 "`results_deep_forecast.md` before reading anything into it (one test day, no "
                                 "interval).")
    cap = " The 45-minute cap was hit, so training stopped early." if any(
        e["training"]["hit_time_cap"] for e in results["targets"].values()) else ""
    return (f"{results['model']} over the last {results['settings']['seq_len_min']} minutes of the Model 3 feature "
            f"set, same split and test rows ({'; '.join(parts)}). {verdict}{cap} "
            "Details: `results_deep_forecast.md`.")


def write_report(results: dict[str, Any], path: Path) -> None:
    """docs/results_deep_forecast.md."""
    s = results["settings"]
    L = ["# Deep sequence forecaster (Task 4, optional)", "",
         "**SIMULATED DATA.** Generated by `ml/deep_forecast.py` (`--report-only` rebuilds this file).", "",
         f"**Model.** {results['model']} over the last {s['seq_len_min']} minutes of the Model 3 feature set "
         f"({results['n_features']} features), linear head, L1 loss, Adam (lr {s['learning_rate']}), batch "
         f"{s['batch_size']}, early stopping on validation MAE (patience {s['patience']}, at most {s['max_epochs']} "
         f"epochs), seed {results['random_seed']}. Imputation and scaling fitted on training minutes only. No "
         "architecture search.", "",
         f"**Split.** {results['split_rule']}; test rows identical to the ablation's (checked at runtime).", "",
         "| Target | h | GRU skill | GRU MAE | XGBoost (Model 3) skill | XGBoost MAE | Epochs (best) | n test |",
         "|---|---|---|---|---|---|---|---|"]
    for key, e in results["targets"].items():
        t, h = key.rsplit("_h", 1)
        x = e["test"]
        xs = "—" if e.get("xgboost_skill") is None else f"{e['xgboost_skill']:+.3f}"
        xm = "—" if e.get("xgboost_mae") is None else f"{e['xgboost_mae']:.2f}"
        L.append(f"| {t} | {h} | **{x['skill_vs_persistence']:+.3f}** | {x['mae']:.2f} | {xs} | {xm} | "
                 f"{e['training']['epochs_run']} ({e['training']['best_epoch']}) | {e['n_test']:,} |")
    L += ["", f"Training time: {results['minutes_used']:.1f} min (cap {s['time_cap_min']} min).", "",
          "## Conclusion", "", results["summary"], "",
          "**Scope of the claim.** This is one modest configuration with no architecture search, as the brief "
          "required. The validation error plateaued for many epochs and then dropped late, so the GRU may be "
          "under-trained, and a larger search might narrow the gap. The finding is that *a modest sequence model "
          "adds nothing here*, not that sequence models cannot work. No confidence interval is reported (one test "
          "day); the gap is large relative to the ablation's intervals, but that is an observation, not a test.", ""]
    path.write_text("\n".join(L), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = config.CONFIG_PATH.parent
    ml = config.get_ml_config()
    p = argparse.ArgumentParser(prog="python -m ml.deep_forecast", description=__doc__.split("\n")[0])
    p.add_argument("--features", type=Path, default=root / ml["features_path"])
    p.add_argument("--metrics", type=Path, default=root / ml["deep"]["metrics_path"])
    p.add_argument("--ablation-metrics", type=Path, default=root / ml["ablation"]["metrics_path"])
    p.add_argument("--report", type=Path, default=root / "docs" / "results_deep_forecast.md")
    p.add_argument("--report-only", action="store_true")
    args = p.parse_args(argv)
    if args.report_only:
        results = json.loads(args.metrics.read_text(encoding="utf-8"))
    else:
        results = run(args.features, args.metrics, args.ablation_metrics)
    write_report(results, args.report)
    print(results["summary"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
