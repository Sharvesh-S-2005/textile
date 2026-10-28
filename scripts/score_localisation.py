"""Score BLE localisation against the simulator's ground truth.

    python scripts/score_localisation.py --truth data/truth.jsonl [--db PATH] [--json-out PATH]

The ground-truth file is written by `simulator/publisher.py --ground-truth-out`. It
exists only for simulated runs (real hardware has no ground truth), never travels
over MQTT, and never enters the database: this script is the only thing that reads it.

Alignment: every ground-truth row is one worker at one BLE scan tick (a 30 s
window). The prediction is the stored assignment interval covering that instant.
Each row is one worker-window = window_s / 60 worker-minutes.

Reported:
  * accuracy over on-site worker-windows (true zone predicted), the headline figure
  * accuracy over all worker-windows (off-site counts as correct only if UNKNOWN)
  * confusion matrix: true zone (or OFF_SITE) x predicted zone (or UNKNOWN)
  * UNKNOWN split into correctly unknown (really off site) and incorrectly unknown,
    plus off-site windows wrongly given a zone
  * per-zone recall
  * latency: time from a real zone change until the prediction first matches it
  * the localisation parameters used
Both the raw assignments and the session-clipped view (outside_session -> UNKNOWN)
are scored.
"""

from __future__ import annotations

import argparse
import bisect
import json
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from db.repo import Repo  # noqa: E402

UNKNOWN = "UNKNOWN"
OFF_SITE = "OFF_SITE"


def _ts(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def load_truth(path: Path) -> list[dict[str, Any]]:
    """Read the ground-truth JSONL file."""
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _predictor(assignments: list[dict[str, Any]], clip: bool):
    """Return f(worker_id, ts) -> predicted zone, UNKNOWN, or None if not covered."""
    by_worker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for a in assignments:
        by_worker[a["worker_id"]].append(a)
    starts = {}
    for w, rows in by_worker.items():
        rows.sort(key=lambda r: r["from_ts"])
        starts[w] = [r["from_ts"] for r in rows]

    def predict(worker_id: str, ts: str) -> str | None:
        rows = by_worker.get(worker_id)
        if not rows:
            return None
        i = bisect.bisect_right(starts[worker_id], ts) - 1
        if i < 0 or not (rows[i]["from_ts"] <= ts < rows[i]["to_ts"]):
            return None
        if clip and rows[i].get("outside_session"):
            return UNKNOWN
        return rows[i]["zone_id"]

    return predict


def _latencies(series: list[tuple[datetime, str, str]]) -> tuple[list[float], list[float], int]:
    """Entry and exit latencies (s) for one worker's (ts, true, pred) series, plus misses.

    Entry: the true zone changes to an on-site zone b; latency until pred == b.
    Exit: the worker leaves the site; latency until pred == UNKNOWN.
    A change the prediction never follows before the next true change is a miss.
    """
    entry, exit_, missed = [], [], 0
    for k in range(1, len(series)):
        if series[k][1] == series[k - 1][1]:
            continue
        target = series[k][1]
        want = UNKNOWN if target == OFF_SITE else target
        followed = None
        for j in range(k, len(series)):
            if j > k and series[j][1] != target:
                break
            if series[j][2] == want:
                followed = (series[j][0] - series[k][0]).total_seconds()
                break
        if followed is None:
            missed += 1
        elif target == OFF_SITE:
            exit_.append(followed)
        else:
            entry.append(followed)
    return entry, exit_, missed


def _summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"n": 0, "mean_s": None, "median_s": None, "p90_s": None}
    ordered = sorted(values)
    return {"n": len(values), "mean_s": round(statistics.fmean(values), 1),
            "median_s": round(statistics.median(values), 1),
            "p90_s": round(ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))], 1)}


def score(truth: list[dict[str, Any]], assignments: list[dict[str, Any]], window_s: int,
          clip: bool = False) -> dict[str, Any]:
    """Score assignments against ground truth. Pure.

    Args:
        truth: rows with ts, worker_id, zone_id (None = off site).
        assignments: rows with worker_id, zone_id, from_ts, to_ts (and outside_session).
        window_s: length of one ground-truth window, for worker-minute totals.
        clip: treat assignments marked outside_session as UNKNOWN.
    """
    predict = _predictor(assignments, clip)
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    uncovered = 0
    series: dict[str, list[tuple[datetime, str, str]]] = defaultdict(list)
    for row in truth:
        pred = predict(row["worker_id"], row["ts"])
        if pred is None:
            uncovered += 1
            continue
        true = row["zone_id"] or OFF_SITE
        confusion[true][pred] += 1
        series[row["worker_id"]].append((_ts(row["ts"]), true, pred))

    on_site = sum(n for t, preds in confusion.items() if t != OFF_SITE for n in preds.values())
    on_site_correct = sum(confusion[t][t] for t in confusion if t != OFF_SITE)
    off_row = confusion.get(OFF_SITE, Counter())      # .get: do not create an empty row
    off_site = sum(off_row.values())
    correctly_unknown = off_row[UNKNOWN]
    incorrectly_unknown = sum(confusion[t][UNKNOWN] for t in confusion if t != OFF_SITE)
    total = on_site + off_site

    entry, exit_, missed = [], [], 0
    for s in series.values():
        s.sort()
        e, x, m = _latencies(s)
        entry += e
        exit_ += x
        missed += m

    zones = sorted(t for t in confusion if t != OFF_SITE)
    params = sorted({json.dumps(a["params"], sort_keys=True) for a in assignments if "params" in a})
    minutes = window_s / 60.0
    return {
        "clipped_to_sessions": clip,
        "windows": total,
        "uncovered_windows": uncovered,
        "worker_minutes": round(total * minutes, 1),
        "accuracy_on_site": round(on_site_correct / on_site, 4) if on_site else None,
        "accuracy_all": round((on_site_correct + correctly_unknown) / total, 4) if total else None,
        "on_site_windows": on_site,
        "off_site_windows": off_site,
        "unknown": {
            "correctly_unknown": correctly_unknown,
            "incorrectly_unknown": incorrectly_unknown,
            "off_site_assigned_a_zone": off_site - correctly_unknown,
        },
        "per_zone_recall": {z: round(confusion[z][z] / sum(confusion[z].values()), 4) for z in zones},
        "confusion": {t: dict(sorted(p.items())) for t, p in sorted(confusion.items())},
        "latency_entry": _summary(entry),
        "latency_exit": _summary(exit_),
        "changes_not_followed": missed,
        "params": [json.loads(p) for p in params],
    }


def format_report(result: dict[str, Any]) -> str:
    """Human-readable report."""
    lines = []
    title = "SESSION-CLIPPED (outside_session -> UNKNOWN)" if result["clipped_to_sessions"] else "RAW ASSIGNMENTS"
    lines.append(f"=== Localisation score: {title} ===")
    pct = lambda v: "n/a" if v is None else f"{100 * v:.1f}%"  # noqa: E731
    lines.append(f"worker-windows scored: {result['windows']} ({result['worker_minutes']} worker-minutes); "
                 f"not covered by any assignment: {result['uncovered_windows']}")
    lines.append(f"ACCURACY (on-site worker-minutes in the correct zone): {pct(result['accuracy_on_site'])} "
                 f"of {result['on_site_windows']} windows")
    lines.append(f"accuracy incl. off-site (off-site correct only if UNKNOWN): {pct(result['accuracy_all'])}")
    u = result["unknown"]
    lines.append(f"UNKNOWN: correctly unknown (off site) {u['correctly_unknown']}, incorrectly unknown "
                 f"(on site) {u['incorrectly_unknown']}; off-site windows given a zone {u['off_site_assigned_a_zone']}")
    lines.append("per-zone recall: " + ", ".join(f"{z} {pct(r)}" for z, r in result["per_zone_recall"].items()))
    for name in ("latency_entry", "latency_exit"):
        lat = result[name]
        label = "zone-change latency" if name == "latency_entry" else "exit latency (to UNKNOWN)"
        if lat["n"]:
            lines.append(f"{label}: n={lat['n']} mean {lat['mean_s']} s, median {lat['median_s']} s, "
                         f"p90 {lat['p90_s']} s")
        else:
            lines.append(f"{label}: n=0")
    lines.append(f"true changes never followed: {result['changes_not_followed']}")
    cols = sorted({p for row in result["confusion"].values() for p in row})
    lines.append("confusion matrix (rows = true, columns = predicted):")
    lines.append(f"{'':>9}" + "".join(f"{c:>9}" for c in cols))
    for true, row in result["confusion"].items():
        lines.append(f"{true:>9}" + "".join(f"{row.get(c, 0):>9}" for c in cols))
    for p in result["params"]:
        lines.append("parameters: " + ", ".join(f"{k}={v}" for k, v in sorted(p.items())))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--truth", type=Path, required=True, help="JSONL from publisher --ground-truth-out")
    p.add_argument("--db", type=Path, default=config.CONFIG_PATH.parent / config.get_paths()["db_path"])
    p.add_argument("--method", default=config.get_localisation_config()["method"])
    p.add_argument("--json-out", type=Path, default=None, help="also write both results as JSON")
    args = p.parse_args(argv)
    truth = load_truth(args.truth)
    with Repo(args.db) as repo:
        assignments = repo.get_all_zone_assignments(args.method)
    if not assignments:
        print(f"no zone_assignments for method {args.method!r} in {args.db}", file=sys.stderr)
        return 1
    window_s = int(config.get_localisation_config()["window_s"])
    results = [score(truth, assignments, window_s, clip=False), score(truth, assignments, window_s, clip=True)]
    print("\n\n".join(format_report(r) for r in results))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
