"""Exposure engine: compute and store personal, zone and cumulative exposure.

    python -m exposure.engine --date 2026-10-21
    python -m exposure.engine --from 2026-10-20 --to 2026-10-26 --recompute
    python -m exposure.engine --worker W105 --dry-run
    python -m exposure.engine --resolve            # re-run localisation + sessions first

For each shift date: per-session personal exposure (exposure/personal.py) ->
personal_exposure_shift; per-zone per-shift summaries -> zone_exposure_summary; then
7- and 30-day rolling cumulatives (exposure/cumulative.py) ->
personal_exposure_cumulative. Existing rows for a date are kept unless
--recompute is given; recomputing a date replaces only that date's rows.

One bad worker or session never aborts the batch: it is logged loudly and skipped.
A session with no readings is still written, as a null result with its flag.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import config
from db.repo import Repo, from_epoch_ms, to_epoch_ms
from exposure.cumulative import daily_exposures, rolling_window
from exposure.personal import ExposureParams, Interval, compute_session_exposure, personal_exposure_for_session
from ingest.localisation import run_localisation
from ingest.sessions import run_sessions
from shifts import ShiftSchedule

log = logging.getLogger("exposure.engine")

WINDOWS_DAYS = (7, 30)
ZONE_LABEL = "measured_environmental"


def shift_date_of(epoch_ms: int) -> str:
    """Shift date = UTC date of a session/shift start ('YYYY-MM-DD')."""
    return from_epoch_ms(epoch_ms)[:10]


def _params_dict(params: ExposureParams) -> dict[str, Any]:
    return params.as_dict()


def personal_rows(repo: Repo, day: str, params: ExposureParams, worker_id: str | None,
                  report: dict[str, Any]) -> list[dict[str, Any]]:
    """personal_exposure_shift rows for every session starting on `day`."""
    sessions = [s for s in repo.get_all_worker_sessions()
                if shift_date_of(s["session_start_ms"]) == day and (worker_id is None or s["worker_id"] == worker_id)]
    workers = [worker_id] if worker_id else [w["worker_id"] for w in config.get_workers()]
    for wid in workers:
        if not any(s["worker_id"] == wid for s in sessions):
            log.warning("%s: no session on %s - skipped", wid, day)
            report["no_session"].append((wid, day))
    rows = []
    for s in sessions:
        try:
            result = personal_exposure_for_session(repo, s, params)
        except Exception:
            log.exception("%s: exposure failed for session starting %s - skipped", s["worker_id"], s["session_start"])
            report["errors"].append((s["worker_id"], s["session_start"]))
            continue
        if "no_valid_readings" in result.flags:
            log.warning("%s: session starting %s has no valid readings - stored as a null result",
                        s["worker_id"], s["session_start"])
        row = result.as_dict()
        row.update(shift_date=day, shift=s["shift"], session_start_ms=s["session_start_ms"],
                   session_end_ms=s["session_end_ms"], session_flags=s["flags"],
                   engine_version=params.engine_version, params=_params_dict(params))
        rows.append(row)
    return rows


def zone_rows(repo: Repo, day: str, params: ExposureParams) -> list[dict[str, Any]]:
    """zone_exposure_summary rows for every zone and every shift starting on `day`.

    Uses the same integration as a worker present in the zone for the whole shift,
    so the zone figures and the personal figures are computed identically.
    """
    span = repo.get_data_span()
    if span is None:
        return []
    data_lo, data_hi = to_epoch_ms(span[0]), to_epoch_ms(span[1])
    schedule = ShiftSchedule(config.get_shifts())
    d = date.fromisoformat(day)
    dt_ms = int(params.reading_interval_s * 1000)
    rows = []
    for inst in sorted((schedule.instance(sid, local_day) for sid in schedule.shift_ids
                        for local_day in (d - timedelta(days=1), d, d + timedelta(days=1))),
                       key=lambda i: i.start):
        start_ms, end_ms = int(inst.start.timestamp() * 1000), int(inst.end.timestamp() * 1000)
        if shift_date_of(start_ms) != day or end_ms <= data_lo or start_ms >= data_hi:
            continue
        for zone in config.get_zones():
            readings = repo.get_zone_env(zone["zone_id"], from_epoch_ms(start_ms), from_epoch_ms(end_ms + dt_ms))
            r = compute_session_exposure(zone["zone_id"], start_ms, end_ms,
                                         [Interval(zone["zone_id"], start_ms, end_ms, readings)], params)
            rows.append({
                "zone_id": zone["zone_id"], "shift_date": day, "shift": inst.shift_id,
                "shift_start": from_epoch_ms(start_ms), "shift_end": from_epoch_ms(end_ms),
                "shift_start_ms": start_ms, "shift_end_ms": end_ms,
                "l_aeq_db": r.l_aeq_db, "l_ex_8h_db": r.l_ex_8h_db,
                "dose_niosh_pct": r.dose_niosh_pct, "dose_india_pct": r.dose_india_pct,
                "absolute_max_events": len(r.absolute_max_events),
                "pm25_twa_raw": r.pm25_twa_raw, "pm25_twa_corrected": r.pm25_twa_corrected,
                "pm10_twa_raw": r.pm10_twa_raw, "pm10_twa_corrected": r.pm10_twa_corrected,
                "wbgt_max_c": r.wbgt_max_c, "wbgt_time_above_limit_s": r.wbgt_time_above_limit_s,
                "co2_twa_ppm": r.co2_twa_ppm, "data_coverage_pct": r.data_coverage_pct,
                "coverage_by_metric": r.coverage_by_metric,
                "flags": r.flags,
                "label": ZONE_LABEL, "dust_label": r.dust_label, "wbgt_t_nw_method": r.wbgt_t_nw_method,
                "wbgt_approximate": r.wbgt_approximate, "engine_version": params.engine_version,
                "params": _params_dict(params),
            })
    return rows


def cumulative_rows(repo: Repo, day: str, params: ExposureParams, worker_id: str | None) -> list[dict[str, Any]]:
    """personal_exposure_cumulative rows (7 and 30 days) for every worker as of `day`."""
    span = repo.get_data_span()
    if span is None:
        return []
    first, last = date.fromisoformat(span[0][:10]), date.fromisoformat(span[1][:10])
    as_of = date.fromisoformat(day)
    lookback = (as_of - timedelta(days=max(WINDOWS_DAYS) - 1)).isoformat()
    daily = daily_exposures(repo.get_exposure_rows("personal_exposure_shift", lookback, day, worker_id))
    workers = [worker_id] if worker_id else [w["worker_id"] for w in config.get_workers()]
    rows = []
    for wid in workers:
        for window in WINDOWS_DAYS:
            row = rolling_window(daily, wid, as_of, window, first, last)
            row.update(engine_version=params.engine_version, params=_params_dict(params))
            rows.append(row)
    return rows


def run_engine(repo: Repo, days: list[str], worker_id: str | None = None, recompute: bool = False,
               dry_run: bool = False) -> dict[str, Any]:
    """Compute and (unless dry_run) store all three tables for each day. Returns a report."""
    params = ExposureParams.from_config()
    report: dict[str, Any] = {"days": days, "personal": [], "zone_rows": 0, "cumulative_rows": 0,
                              "skipped_existing": [], "no_session": [], "errors": [], "dry_run": dry_run}
    existing = set(repo.get_exposure_dates("personal_exposure_shift"))
    for day in days:
        if day in existing and not recompute and not dry_run:
            log.info("%s: rows exist - skipped (use --recompute to replace)", day)
            report["skipped_existing"].append(day)
            continue
        p_rows = personal_rows(repo, day, params, worker_id, report)
        z_rows = zone_rows(repo, day, params) if worker_id is None else []
        report["personal"].extend(p_rows)
        report["zone_rows"] += len(z_rows)
        if not dry_run:
            repo.replace_exposure_rows("personal_exposure_shift", day, p_rows, worker_id)
            if worker_id is None:
                repo.replace_exposure_rows("zone_exposure_summary", day, z_rows)
            repo.commit()
    for day in days:
        if day in report["skipped_existing"]:
            continue
        c_rows = cumulative_rows(repo, day, params, worker_id)
        report["cumulative_rows"] += len(c_rows)
        report.setdefault("cumulative", []).extend(c_rows)
        if not dry_run:
            repo.replace_exposure_rows("personal_exposure_cumulative", day, c_rows, worker_id)
            repo.commit()
    return report


def format_summary(report: dict[str, Any]) -> str:
    """Per-session summary table: worker, dose under each criterion, coverage, unaccounted, flags."""
    head = f"{'date':<11}{'worker':<7}{'NIOSH %':>9}{'India %':>9}{'L_EX,8h':>9}{'cover %':>9}{'unacc %':>9}  flags"
    lines = [head, "-" * len(head)]

    def fmt(v: float | None, spec: str) -> str:
        return "null" if v is None else format(v, spec)

    for r in sorted(report["personal"], key=lambda x: (x["shift_date"], x["worker_id"])):
        lines.append(f"{r['shift_date']:<11}{r['worker_id']:<7}{fmt(r['dose_niosh_pct'], '9.1f'):>9}"
                     f"{fmt(r['dose_india_pct'], '9.1f'):>9}{fmt(r['l_ex_8h_db'], '9.1f'):>9}"
                     f"{fmt(r['data_coverage_pct'], '9.1f'):>9}{r['unaccounted_pct']:>9.1f}  {','.join(r['flags']) or '-'}")
    lines.append(f"{len(report['personal'])} sessions, {report['zone_rows']} zone-shift rows, "
                 f"{report['cumulative_rows']} cumulative rows; skipped (existing) {len(report['skipped_existing'])} "
                 f"days; errors {len(report['errors'])}" + ("  [DRY RUN: nothing written]" if report["dry_run"] else ""))
    return "\n".join(lines)


def _dates(args: argparse.Namespace, repo: Repo) -> list[str]:
    if args.date:
        return [args.date]
    if args.date_from or args.date_to:
        if not (args.date_from and args.date_to):
            raise SystemExit("--from and --to must be given together")
        lo, hi = date.fromisoformat(args.date_from), date.fromisoformat(args.date_to)
        return [(lo + timedelta(days=i)).isoformat() for i in range((hi - lo).days + 1)]
    days = sorted({shift_date_of(s["session_start_ms"]) for s in repo.get_all_worker_sessions()})
    if not days:
        log.error("no worker sessions stored; run with --resolve (localisation + sessions) first")
    return days


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code (1 if any session failed)."""
    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(prog="python -m exposure.engine", description=__doc__.split("\n")[0])
    p.add_argument("--db", type=Path, default=config.CONFIG_PATH.parent / config.get_paths()["db_path"])
    p.add_argument("--date", help="one shift date, YYYY-MM-DD")
    p.add_argument("--from", dest="date_from", help="first shift date (with --to)")
    p.add_argument("--to", dest="date_to", help="last shift date (with --from)")
    p.add_argument("--worker", help="restrict to one worker")
    p.add_argument("--recompute", action="store_true", help="replace existing rows for the range")
    p.add_argument("--dry-run", action="store_true", help="print results without writing")
    p.add_argument("--resolve", action="store_true", help="run localisation and sessions first")
    args = p.parse_args(argv)
    with Repo(args.db) as repo:
        repo.init_schema()
        if args.resolve:
            run_localisation(repo)
            run_sessions(repo)
        report = run_engine(repo, _dates(args, repo), args.worker, args.recompute, args.dry_run)
    print(format_summary(report))
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
