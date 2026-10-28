"""Localisation parameter sensitivity: vary one parameter at a time, score each.

    python scripts/sweep_localisation.py --truth data/truth.jsonl [--db PATH]

Resolves the stored BLE sightings in memory with each parameter set and scores the
result against the ground truth. Nothing is written to the database. One parameter
is varied at a time around the configured baseline, so each row shows that
parameter's effect alone.

Caveat for reporting: these are scores on simulated data, from the same run that
is being scored. Picking the best row and reporting it would overfit; the baseline
values come from physical reasoning (config.yaml comments) and the sweep is
reported as sensitivity, not as tuning.
"""

from __future__ import annotations

import argparse
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from db.repo import Repo  # noqa: E402
from ingest.localisation import Params, resolve  # noqa: E402
from ingest.validation import parse_ts  # noqa: E402
from scripts.score_localisation import load_truth, score  # noqa: E402

GRID: dict[str, list[Any]] = {
    "ewma_alpha": [0.3, 0.5, 0.8, 1.0],
    "rssi_margin_db": [0.0, 3.0, 6.0, 9.0],
    "debounce_windows": [1, 2, 3],
    "absence_windows": [2, 4, 6],
}


def sweep(sightings: list[tuple], start: Any, end: Any, truth: list[dict]) -> list[dict[str, Any]]:
    """Score the baseline and every one-at-a-time variation."""
    tag_to_worker = {w["tag_id"]: w["worker_id"] for w in config.get_workers()}
    base = Params.from_config()
    runs = [("baseline", None, base)]
    for name, values in GRID.items():
        for v in values:
            if v != getattr(base, name):
                runs.append((name, v, Params.from_config({name: v})))
    out = []
    for name, value, params in runs:
        rows = resolve(sightings, tag_to_worker, start, end, params)
        r = score(truth, rows, params.window_s)
        changes = sum(1 for a, b in zip(rows, rows[1:]) if a["worker_id"] == b["worker_id"])
        out.append({"param": name, "value": value, "accuracy_on_site": r["accuracy_on_site"],
                    "incorrectly_unknown": r["unknown"]["incorrectly_unknown"],
                    "off_site_assigned": r["unknown"]["off_site_assigned_a_zone"],
                    "latency_median_s": r["latency_entry"]["median_s"],
                    "latency_p90_s": r["latency_entry"]["p90_s"],
                    "not_followed": r["changes_not_followed"], "zone_changes": changes})
    return out


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--truth", type=Path, required=True)
    p.add_argument("--db", type=Path, default=config.CONFIG_PATH.parent / config.get_paths()["db_path"])
    args = p.parse_args(argv)
    truth = load_truth(args.truth)
    with Repo(args.db) as repo:
        span = repo.get_ble_time_range()
        if span is None:
            print("no BLE data", file=sys.stderr)
            return 1
        window = timedelta(seconds=int(config.get_localisation_config()["window_s"]))
        end = parse_ts(span[1]) + window
        sightings = [(parse_ts(s["ts"]), s["zone_id"], s["tag_id"], float(s["rssi_dbm"]))
                     for s in repo.get_ble_sightings(span[0], end.strftime("%Y-%m-%dT%H:%M:%SZ"))]
    results = sweep(sightings, parse_ts(span[0]), end, truth)
    print(f"{'parameter':<18}{'value':>7}{'acc on-site':>13}{'wrong UNK':>11}{'offsite->zone':>15}"
          f"{'lat med':>9}{'lat p90':>9}{'missed':>8}{'changes':>9}")
    for r in results:
        value = "" if r["value"] is None else r["value"]
        print(f"{r['param']:<18}{value!s:>7}{100 * r['accuracy_on_site']:>12.1f}%{r['incorrectly_unknown']:>11}"
              f"{r['off_site_assigned']:>15}{r['latency_median_s']!s:>9}{r['latency_p90_s']!s:>9}"
              f"{r['not_followed']:>8}{r['zone_changes']:>9}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
