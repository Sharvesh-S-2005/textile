"""BLE zone resolution: raw sightings -> one zone per worker per scan window.

This runs as a separate pass over stored sightings (not inside the subscriber), so
it can be re-run with different parameters without replaying any data:

    python -m ingest.localisation [--db PATH]

Every assignment is an INFERENCE about where a worker was, labelled
`estimated_personal` (CLAUDE.md integrity rules). It is not a measurement.

Algorithm (`ble_ewma_hysteresis`), per tag, per scan window of `window_s`:

1. Weak-signal floor: sightings below `min_rssi_dbm` are ignored.
2. Smooth: an EWMA of RSSI per (tag, zone). A zone heard this window updates with
   its RSSI (strongest sighting if a node reported more than one); a zone not heard
   decays toward `missing_rssi_dbm` and is dropped once below the floor.
3. Rank candidate zones by smoothed RSSI.
4. Hysteresis: if the best zone is not the current zone but beats it by less than
   `rssi_margin_db`, the current zone stays the challenger.
5. Debounce: a new zone must win `debounce_windows` consecutive windows before the
   assignment changes (this includes leaving UNKNOWN).
6. Absence: unseen everywhere for `absence_windows` consecutive windows -> UNKNOWN,
   and the smoothing state is cleared. The last zone is never carried forward
   indefinitely: a worker who left must not keep accruing exposure.

A tag seen by three nodes at once is handled the same way: all three zones are
smoothed, ranked, and the leader must clear the margin and the debounce before the
worker moves.

confidence = clip((s_assigned - s_runner_up) / confidence_full_margin_db, 0, 1),
where s_runner_up is the next-best smoothed RSSI (or the floor if none). UNKNOWN
has confidence 0. Consecutive identical windows are merged into one interval whose
confidence is the mean over its windows.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import config
from db.repo import Repo
from ingest.validation import parse_ts

log = logging.getLogger("ingest.localisation")

UNKNOWN = "UNKNOWN"
LABEL = "estimated_personal"


@dataclass(frozen=True)
class Params:
    """Localisation parameters (the `localisation` section of config.yaml)."""

    method: str
    window_s: int
    ewma_alpha: float
    rssi_margin_db: float
    debounce_windows: int
    absence_windows: int
    min_rssi_dbm: float
    missing_rssi_dbm: float
    confidence_full_margin_db: float

    @classmethod
    def from_config(cls, overrides: dict[str, Any] | None = None) -> Params:
        """Parameters from config.yaml, optionally with some values replaced."""
        values = {**config.get_localisation_config(), **(overrides or {})}
        return cls(**{f: values[f] for f in cls.__dataclass_fields__})

    def as_dict(self) -> dict[str, Any]:
        """Parameters as a plain dict (stored with every assignment)."""
        return {f: getattr(self, f) for f in self.__dataclass_fields__}


class TagTracker:
    """Zone decision state for one tag. Call step() once per window, in time order."""

    def __init__(self, params: Params) -> None:
        """Start with no zone (UNKNOWN) and no smoothing state."""
        self.p = params
        self.current = UNKNOWN
        self.ewma: dict[str, float] = {}
        self.absent_windows = 0
        self.pending: str | None = None
        self.pending_count = 0

    def _smooth(self, heard: dict[str, float]) -> None:
        """Update the per-zone EWMA from this window's sightings."""
        a = self.p.ewma_alpha
        for zone in set(self.ewma) | set(heard):
            obs = heard.get(zone, self.p.missing_rssi_dbm)
            prev = self.ewma.get(zone)
            s = obs if prev is None else a * obs + (1.0 - a) * prev
            if zone not in heard and s < self.p.min_rssi_dbm:
                del self.ewma[zone]
            else:
                self.ewma[zone] = s

    def confidence(self) -> float:
        """Margin of the current zone over the runner-up, scaled to [0, 1]."""
        if self.current == UNKNOWN:
            return 0.0
        own = self.ewma.get(self.current, self.p.missing_rssi_dbm)
        others = [s for z, s in self.ewma.items() if z != self.current]
        runner_up = max(others) if others else self.p.min_rssi_dbm
        return min(max((own - runner_up) / self.p.confidence_full_margin_db, 0.0), 1.0)

    def step(self, heard: dict[str, float]) -> tuple[str, float]:
        """Advance one window. `heard` maps zone_id -> strongest RSSI this window.

        Returns (zone_id or UNKNOWN, confidence).
        """
        heard = {z: r for z, r in heard.items() if r >= self.p.min_rssi_dbm}
        self._smooth(heard)

        self.absent_windows = 0 if heard else self.absent_windows + 1
        if self.absent_windows >= self.p.absence_windows:
            self.current, self.pending, self.pending_count = UNKNOWN, None, 0
            self.ewma.clear()
            return UNKNOWN, 0.0

        if self.ewma:
            best, best_s = sorted(self.ewma.items(), key=lambda kv: (-kv[1], kv[0]))[0]
            challenger = best
            if (best != self.current and self.current in self.ewma
                    and best_s - self.ewma[self.current] < self.p.rssi_margin_db):
                challenger = self.current
            if challenger == self.current:
                self.pending, self.pending_count = None, 0
            else:
                if challenger == self.pending:
                    self.pending_count += 1
                else:
                    self.pending, self.pending_count = challenger, 1
                if self.pending_count >= self.p.debounce_windows:
                    self.current, self.pending, self.pending_count = challenger, None, 0
        return self.current, self.confidence()


# ---------------------------------------------------------------- pure resolution


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def window_origin(ts: datetime, window_s: int) -> datetime:
    """Start of the window containing `ts`, on a grid aligned to the Unix epoch."""
    epoch = int(ts.timestamp())
    return datetime.fromtimestamp(epoch - epoch % window_s, tz=timezone.utc)


def resolve(
    sightings: Iterable[tuple[datetime, str, str, float]],
    tag_to_worker: dict[str, str],
    start: datetime,
    end: datetime,
    params: Params,
) -> list[dict[str, Any]]:
    """Resolve sightings into merged assignment intervals for every configured tag.

    Args:
        sightings: (ts, zone_id, tag_id, rssi_dbm) tuples, in any order.
        tag_to_worker: tag_id -> worker_id for every tag to resolve.
        start, end: the time span to cover; windows run from the one containing
            `start` up to `end` (exclusive).
        params: algorithm parameters.

    Returns:
        Interval rows: worker_id, tag_id, zone_id, from_ts, to_ts, confidence, params,
        sorted by worker then from_ts. Intervals of one worker never overlap.
    """
    w = params.window_s
    origin = window_origin(start, w)
    n_windows = max(1, math.ceil((end - origin).total_seconds() / w))
    heard: dict[str, dict[int, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    unknown_tags: set[str] = set()
    for ts, zone_id, tag_id, rssi in sightings:
        if tag_id not in tag_to_worker:
            unknown_tags.add(tag_id)
            continue
        i = int((ts - origin).total_seconds() // w)
        if 0 <= i < n_windows:
            per_zone = heard[tag_id][i]
            per_zone[zone_id] = max(rssi, per_zone.get(zone_id, -math.inf))
    if unknown_tags:
        log.warning("ignored sightings of %d tags not in config: %s", len(unknown_tags), sorted(unknown_tags))

    rows: list[dict[str, Any]] = []
    for tag_id, worker_id in sorted(tag_to_worker.items(), key=lambda kv: kv[1]):
        tracker = TagTracker(params)
        decisions = [tracker.step(heard[tag_id].get(i, {})) for i in range(n_windows)]
        rows.extend(_merge(worker_id, tag_id, decisions, origin, w, params))
    return rows


def _merge(worker_id: str, tag_id: str, decisions: list[tuple[str, float]], origin: datetime,
           window_s: int, params: Params) -> list[dict[str, Any]]:
    """Merge consecutive windows with the same zone into intervals."""
    rows = []
    run_start, confidences = 0, []
    for i, (zone, conf) in enumerate(decisions):
        confidences.append(conf)
        last = i == len(decisions) - 1
        if last or decisions[i + 1][0] != zone:
            rows.append({
                "worker_id": worker_id,
                "tag_id": tag_id,
                "zone_id": zone,
                "from_ts": _iso(origin + timedelta(seconds=run_start * window_s)),
                "to_ts": _iso(origin + timedelta(seconds=(i + 1) * window_s)),
                "confidence": round(sum(confidences) / len(confidences), 3),
                "params": params.as_dict(),
            })
            run_start, confidences = i + 1, []
    return rows


def count_changes(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Number of zone changes per worker (intervals - 1)."""
    per_worker: dict[str, int] = defaultdict(int)
    for r in rows:
        per_worker[r["worker_id"]] += 1
    return {w: n - 1 for w, n in per_worker.items()}


# ---------------------------------------------------------------- database pass


def run_localisation(repo: Repo, params: Params | None = None) -> dict[str, Any]:
    """Resolve every stored BLE sighting and replace this method's zone_assignments.

    Returns a summary (rows written, windows, UNKNOWN share). Re-run the session
    builder afterwards: replacing assignments resets their outside_session marks.
    """
    params = params or Params.from_config()
    span = repo.get_ble_time_range()
    if span is None:
        log.warning("no BLE scans stored; nothing to resolve")
        return {"rows": 0}
    start, last = parse_ts(span[0]), parse_ts(span[1])
    end = last + timedelta(seconds=params.window_s)
    sightings = [(parse_ts(s["ts"]), s["zone_id"], s["tag_id"], float(s["rssi_dbm"]))
                 for s in repo.get_ble_sightings(span[0], _iso(end))]
    tag_to_worker = {w["tag_id"]: w["worker_id"] for w in config.get_workers()}
    rows = resolve(sightings, tag_to_worker, start, end, params)
    written = repo.replace_zone_assignments(params.method, rows)
    repo.commit()
    unknown_s = sum((parse_ts(r["to_ts"]) - parse_ts(r["from_ts"])).total_seconds()
                    for r in rows if r["zone_id"] == UNKNOWN)
    total_s = sum((parse_ts(r["to_ts"]) - parse_ts(r["from_ts"])).total_seconds() for r in rows)
    summary = {"rows": written, "sightings": len(sightings), "from": span[0], "to": _iso(end),
               "unknown_share": round(unknown_s / total_s, 4) if total_s else 0.0,
               "zone_changes": sum(count_changes(rows).values()), "params": params.as_dict()}
    log.info("wrote %d assignment intervals from %d sightings (%s -> %s), UNKNOWN share %.1f%%",
             written, len(sightings), span[0], summary["to"], 100 * summary["unknown_share"])
    return summary


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(prog="python -m ingest.localisation", description=__doc__.split("\n")[0])
    p.add_argument("--db", type=Path, default=config.CONFIG_PATH.parent / config.get_paths()["db_path"])
    args = p.parse_args(argv)
    with Repo(args.db) as repo:
        run_localisation(repo)
    return 0


if __name__ == "__main__":
    sys.exit(main())
