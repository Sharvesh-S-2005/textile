"""Worker sessions from RFID gate events, and clipping of zone assignments to them.

    python -m ingest.sessions [--db PATH]

Run after ingest/localisation.py (re-running localisation resets the clipping).

Pairing, per worker in ts order:
  check_in  -> opens a session. If one is already open, the session is flagged
               `duplicate_check_in` and keeps its first check-in as the start
               (a double tap is the usual cause).
  check_out -> closes the open session. With none open, the session's start is
               inferred as the start of its shift, and it is flagged
               `orphan_check_out` (the start is an inference, not a tap).
  still open at the end -> the end is inferred as the end of its shift, and it
               is flagged `unmatched_check_in`.
Any session longer than `sessions.max_session_h` is also flagged
`implausible_duration`.

The two inferred cases are deliberately symmetric. Treating an orphan check-out as
covering nothing would erase the real end of a shift (for example, the night shift
when data starts mid-shift), and under-reporting exposure is the unsafe direction
for an occupational-health system. The flags keep every inference visible.

Clipping: a worker who is not checked in cannot be accruing exposure, whatever a
stray BLE sighting suggests. Assignments are split at session boundaries, and
pieces outside every session are kept but marked `outside_session`.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any

import config
from db.repo import Repo
from ingest.validation import parse_ts
from shifts import ShiftInstance, ShiftSchedule

log = logging.getLogger("ingest.sessions")


class SessionFlag(StrEnum):
    """Closed set of session problems (the dashboard groups by these)."""

    UNMATCHED_CHECK_IN = "unmatched_check_in"
    ORPHAN_CHECK_OUT = "orphan_check_out"
    DUPLICATE_CHECK_IN = "duplicate_check_in"
    IMPLAUSIBLE_DURATION = "implausible_duration"


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _nearest_shift(schedule: ShiftSchedule, shift_id: str, ts: datetime, by: str) -> ShiftInstance:
    """The instance of `shift_id` whose start (by='start') or end (by='end') is nearest `ts`."""
    window = timedelta(hours=24)
    candidates = schedule.instances_between(ts - window, ts + window, shift_id)
    return min(candidates, key=lambda inst: abs((getattr(inst, by) - ts).total_seconds()))


def build_sessions(events: list[dict[str, Any]], schedule: ShiftSchedule,
                   worker_shift: dict[str, str], max_session_h: float) -> list[dict[str, Any]]:
    """Pair gate events into sessions. Pure: no database access.

    Args:
        events: gate event rows (worker_id, event, ts, event_id), in any order.
        schedule: shift calendar, used to close an unmatched check-in at shift end.
        worker_shift: worker_id -> configured shift ID.
        max_session_h: sessions longer than this are flagged implausible_duration.

    Returns:
        Session rows: worker_id, session_start, session_end, duration_s, shift,
        flags, check_in_event_id, check_out_event_id.
    """
    by_worker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for e in events:
        if e["worker_id"] is None:
            log.warning("gate event %s has unknown card %s; not paired", e["event_id"], e.get("card_id"))
            continue
        by_worker[e["worker_id"]].append(e)

    sessions: list[dict[str, Any]] = []
    for worker_id, worker_events in sorted(by_worker.items()):
        shift_id = worker_shift[worker_id]
        open_session: dict[str, Any] | None = None
        for e in sorted(worker_events, key=lambda ev: (ev["ts"], ev["event_id"])):
            ts = parse_ts(e["ts"])
            if e["event"] == "check_in":
                if open_session is not None:
                    open_session["flags"].add(SessionFlag.DUPLICATE_CHECK_IN)
                    continue
                inst = _nearest_shift(schedule, shift_id, ts, "start")
                open_session = {"worker_id": worker_id, "start": ts, "shift": inst.shift_id,
                                "shift_end": inst.end, "flags": set(), "check_in_event_id": e["event_id"]}
            elif open_session is not None:
                sessions.append(_close(open_session, ts, e["event_id"], max_session_h))
                open_session = None
            else:
                inst = _nearest_shift(schedule, shift_id, ts, "end")
                start = min(inst.start, ts)          # never after the check-out itself
                orphan = {"worker_id": worker_id, "start": start, "shift": inst.shift_id,
                          "flags": {SessionFlag.ORPHAN_CHECK_OUT}, "check_in_event_id": None}
                sessions.append(_close(orphan, ts, e["event_id"], max_session_h))
        if open_session is not None:
            open_session["flags"].add(SessionFlag.UNMATCHED_CHECK_IN)
            sessions.append(_close(open_session, open_session["shift_end"], None, max_session_h))
    return sessions


def _close(s: dict[str, Any], end: datetime, check_out_event_id: str | None,
           max_session_h: float) -> dict[str, Any]:
    """Finish an open session as a row."""
    duration_s = int((end - s["start"]).total_seconds())
    flags = set(s["flags"])
    if duration_s > max_session_h * 3600:
        flags.add(SessionFlag.IMPLAUSIBLE_DURATION)
    return {"worker_id": s["worker_id"], "session_start": _iso(s["start"]), "session_end": _iso(end),
            "duration_s": duration_s, "shift": s["shift"], "flags": sorted(str(f) for f in flags),
            "check_in_event_id": s["check_in_event_id"], "check_out_event_id": check_out_event_id}


def clip_assignments(assignments: list[dict[str, Any]], sessions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Split assignments at session boundaries and mark pieces outside every session.

    Pure. Returns new assignment rows (same fields, plus `outside_session`), still
    contiguous and non-overlapping per worker.
    """
    spans: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for s in sessions:
        spans[s["worker_id"]].append((s["session_start"], s["session_end"]))

    out: list[dict[str, Any]] = []
    for a in assignments:
        worker_spans = sorted(spans.get(a["worker_id"], []))
        cuts = {a["from_ts"], a["to_ts"]}
        for start, end in worker_spans:
            for t in (start, end):
                if a["from_ts"] < t < a["to_ts"]:
                    cuts.add(t)
        edges = sorted(cuts)
        for lo, hi in zip(edges, edges[1:]):
            inside = any(start <= lo and hi <= end for start, end in worker_spans)
            out.append({**a, "from_ts": lo, "to_ts": hi, "outside_session": not inside})
    return out


def run_sessions(repo: Repo) -> dict[str, Any]:
    """Rebuild worker_sessions from gate events, then clip every stored assignment."""
    schedule = ShiftSchedule(config.get_shifts())
    worker_shift = {w["worker_id"]: w["shift"] for w in config.get_workers()}
    sessions = build_sessions(repo.get_gate_events(), schedule, worker_shift,
                              float(config.get_sessions_config()["max_session_h"]))
    repo.replace_worker_sessions(sessions)

    assignments = repo.get_all_zone_assignments()
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for a in assignments:
        by_method[a["method"]].append(a)
    outside = 0
    for method, rows in by_method.items():
        clipped = clip_assignments(rows, sessions)
        outside += sum(1 for r in clipped if r["outside_session"])
        repo.replace_zone_assignments(method, clipped)
    repo.commit()
    flag_counts: dict[str, int] = defaultdict(int)
    for s in sessions:
        for f in s["flags"]:
            flag_counts[f] += 1
    summary = {"sessions": len(sessions), "flags": dict(flag_counts), "outside_session_rows": outside}
    log.info("built %d sessions (flags: %s); %d assignment pieces outside any session",
             len(sessions), dict(flag_counts) or "none", outside)
    return summary


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(prog="python -m ingest.sessions", description=__doc__.split("\n")[0])
    p.add_argument("--db", type=Path, default=config.CONFIG_PATH.parent / config.get_paths()["db_path"])
    args = p.parse_args(argv)
    with Repo(args.db) as repo:
        run_sessions(repo)
    return 0


if __name__ == "__main__":
    sys.exit(main())
