"""Rolling 7-day and 30-day cumulative exposure per worker.

Built from the per-session results (personal_exposure_shift). Pure functions: no
database, no config reads.

Definitions (stated in docs/exposure_methods.md):
* A "day" is a shift date (UTC date of the session start; for this mill it equals
  the local shift date). A worker's sessions on one date are combined first.
* Cumulative noise dose = sum of daily dose percentages, per criterion.
* Daily L_EX,8h of several sessions combines on an energy basis:
  10 log10(sum_i 10^(L_EX,8h,i / 10)). The mean daily L_EX,8h over a window is the
  ENERGY mean, 10 log10(mean(10^(L/10))), the same form as the ISO 9612 /
  EU 2003/10/EC weekly exposure level - never an arithmetic mean of dB.
* Dust TWA-hours = TWA (ug/m3) x hours of accounted exposure (ug.h/m3), summed;
  indicative only.
* Days without valid data are counted separately and excluded from every sum and
  mean - never added as zero.

Partial windows. `window_coverage_pct` is the share of the window's days that lie
inside the period the system was collecting data. A 30-day window over 7 days of
data reports ~23% and `partial = True`, and its totals are the totals over the
days that exist. Nothing is ever scaled up to a notional full window.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

LABEL = "estimated_personal"


@dataclass(frozen=True)
class DailyExposure:
    """One worker's combined exposure for one shift date."""

    worker_id: str
    day: date
    has_data: bool
    dose_niosh_pct: float | None
    dose_india_pct: float | None
    l_ex_8h_db: float | None
    pm25_twa_hours_raw: float | None
    pm25_twa_hours_corrected: float | None
    pm10_twa_hours_raw: float | None
    pm10_twa_hours_corrected: float | None
    wbgt_hours_above_limit: float | None
    data_coverage_pct: float | None
    unaccounted_pct: float | None


def _sum_or_none(values: Iterable[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return math.fsum(present) if present else None


def _twa_hours(rows: list[dict[str, Any]], key: str) -> float | None:
    """sum of TWA x accounted hours over the rows that have the TWA."""
    return _sum_or_none(r[key] * r["accounted_time_s"] / 3600.0 if r[key] is not None else None for r in rows)


def _weighted_mean(rows: list[dict[str, Any]], key: str, weight: str) -> float | None:
    pairs = [(r[key], r[weight]) for r in rows if r[key] is not None and r[weight] > 0]
    total = math.fsum(w for _, w in pairs)
    return None if total <= 0 else math.fsum(v * w for v, w in pairs) / total


def daily_exposures(shift_rows: Iterable[dict[str, Any]]) -> list[DailyExposure]:
    """Combine per-session rows into one DailyExposure per worker per shift date."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in shift_rows:
        groups[(r["worker_id"], r["shift_date"])].append(r)
    out = []
    for (worker_id, day), rows in sorted(groups.items()):
        with_noise = [r for r in rows if r["dose_niosh_pct"] is not None]
        lex = None
        if with_noise:
            lex = 10.0 * math.log10(math.fsum(10.0 ** (r["l_ex_8h_db"] / 10.0) for r in with_noise))
        wbgt = _sum_or_none(r["wbgt_time_above_limit_s"] for r in rows)
        out.append(DailyExposure(
            worker_id=worker_id,
            day=date.fromisoformat(day),
            has_data=bool(with_noise),
            dose_niosh_pct=_sum_or_none(r["dose_niosh_pct"] for r in rows),
            dose_india_pct=_sum_or_none(r["dose_india_pct"] for r in rows),
            l_ex_8h_db=lex,
            pm25_twa_hours_raw=_twa_hours(rows, "pm25_twa_raw"),
            pm25_twa_hours_corrected=_twa_hours(rows, "pm25_twa_corrected"),
            pm10_twa_hours_raw=_twa_hours(rows, "pm10_twa_raw"),
            pm10_twa_hours_corrected=_twa_hours(rows, "pm10_twa_corrected"),
            wbgt_hours_above_limit=None if wbgt is None else wbgt / 3600.0,
            data_coverage_pct=_weighted_mean(rows, "data_coverage_pct", "accounted_time_s"),
            unaccounted_pct=_weighted_mean(rows, "unaccounted_pct", "session_duration_s"),
        ))
    return out


def energy_mean_db(levels_db: list[float]) -> float:
    """10 log10(mean(10^(L/10))): the only way levels are averaged here."""
    if not levels_db:
        raise ValueError("no levels to average")
    return 10.0 * math.log10(math.fsum(10.0 ** (level / 10.0) for level in levels_db) / len(levels_db))


def window_coverage_pct(as_of: date, window_days: int, data_first: date, data_last: date) -> float:
    """Share (%) of the window [as_of - window_days + 1, as_of] inside [data_first, data_last]."""
    start = as_of - timedelta(days=window_days - 1)
    lo, hi = max(start, data_first), min(as_of, data_last)
    days_inside = max((hi - lo).days + 1, 0)
    return 100.0 * days_inside / window_days


def rolling_window(daily: Iterable[DailyExposure], worker_id: str, as_of: date, window_days: int,
                   data_first: date, data_last: date) -> dict[str, Any]:
    """Cumulative exposure for one worker over the window ending at `as_of` (inclusive).

    Totals are over the days that exist in the window; a partial window is reported
    as partial (window_coverage_pct < 100) and never scaled up.
    """
    if window_days < 1:
        raise ValueError("window must be at least one day")
    start = as_of - timedelta(days=window_days - 1)
    days = [d for d in daily if d.worker_id == worker_id and start <= d.day <= as_of]
    worked = [d for d in days if d.has_data]
    coverage = window_coverage_pct(as_of, window_days, data_first, data_last)
    lex = [d.l_ex_8h_db for d in worked if d.l_ex_8h_db is not None]
    return {
        "worker_id": worker_id,
        "as_of_date": as_of.isoformat(),
        "window_days": window_days,
        "window_start_date": start.isoformat(),
        "window_coverage_pct": round(coverage, 3),
        "partial": coverage < 100.0,
        "days_worked": len(worked),
        "days_without_data": len(days) - len(worked),
        "cumulative_dose_niosh_pct": _sum_or_none(d.dose_niosh_pct for d in worked),
        "cumulative_dose_india_pct": _sum_or_none(d.dose_india_pct for d in worked),
        "mean_daily_l_ex_8h_db": energy_mean_db(lex) if lex else None,
        "max_daily_l_ex_8h_db": max(lex) if lex else None,
        "pm25_twa_hours_raw": _sum_or_none(d.pm25_twa_hours_raw for d in worked),
        "pm25_twa_hours_corrected": _sum_or_none(d.pm25_twa_hours_corrected for d in worked),
        "pm10_twa_hours_raw": _sum_or_none(d.pm10_twa_hours_raw for d in worked),
        "pm10_twa_hours_corrected": _sum_or_none(d.pm10_twa_hours_corrected for d in worked),
        "wbgt_hours_above_limit": _sum_or_none(d.wbgt_hours_above_limit for d in worked),
        "days_dose_niosh_over_100": sum(1 for d in worked if d.dose_niosh_pct is not None and d.dose_niosh_pct > 100.0),
        "days_dose_india_over_100": sum(1 for d in worked if d.dose_india_pct is not None and d.dose_india_pct > 100.0),
        "mean_data_coverage_pct": _mean([d.data_coverage_pct for d in worked]),
        "mean_unaccounted_pct": _mean([d.unaccounted_pct for d in worked]),
        "label": LABEL,
    }


def _mean(values: list[float | None]) -> float | None:
    """Arithmetic mean of percentages (linear quantities; never used for dB)."""
    present = [v for v in values if v is not None]
    return math.fsum(present) / len(present) if present else None
