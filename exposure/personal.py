"""Personal exposure for one worker over one shift session.

Joins WHERE the worker was (session-clipped zone_assignments) to WHAT the zone
sensors measured there (zone_env). Every result is labelled `estimated_personal`:
zone data weighted by inferred presence, not a personal measurement. A wearable
reading, if one ever exists, belongs in the separate `measured_personal` field and
never overwrites this one.

Rules implemented here (CLAUDE.md and the Day 3 brief):

* UNKNOWN zone time is never zero-filled and never carried forward. It is reported
  as `unaccounted_time_s` / `unaccounted_pct`. All figures cover ACCOUNTED time
  only, so where unaccounted time exists they are lower bounds, and the result is
  flagged `incomplete_presence` above `unaccounted_flag_pct`.
* A NULL reading is absent data, not silence (or clean air, or cool air). Within
  each assignment interval the valid readings stand for the whole interval - the
  ISO 9612 task-based principle that a sample represents the task duration - so
  missing readings lower `data_coverage_pct` but do not depress the dose. An
  interval with no valid reading at all is NOT imputed: it is excluded and shows
  up as lost coverage (flagged `low_data_coverage` below `coverage_flag_pct`).
* A session with no valid noise reading returns a null result flagged
  `no_valid_readings`, never a zero dose.
* Records flagged as detected sensor anomalies (`exclude_record_flags`: spike,
  stuck_value) are excluded from integration and counted in
  `excluded_anomalous_readings` (they reduce coverage). An excluded reading at or
  above the 115 dBA absolute maximum is NOT silently dropped: it is listed in
  `excluded_absolute_max_candidates`, because a real event cannot be told apart
  from a glitch by the data alone.

Timing: each zone_env reading is the L_Aeq (and the sample) over the 10 s BEFORE
its ts (contract section 2/4), and is weighted by its exact overlap with the
assignment interval.

The `compute_session_exposure` core is pure; `personal_exposure_for_session` reads
the inputs through db.repo.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

import config
from exposure.dust import correct_humidity
from exposure.heat import T_NW_METHOD, wbgt_estimate
from db.repo import from_epoch_ms
from exposure.noise import NoiseCriterion, absolute_max_exceeded, l_ex_8h, noise_dose_pct

LABEL = "estimated_personal"
UNKNOWN_ZONE = "UNKNOWN"


class ResultFlag(StrEnum):
    """Closed set of personal-exposure result flags."""

    NO_VALID_READINGS = "no_valid_readings"
    INCOMPLETE_PRESENCE = "incomplete_presence"
    LOW_DATA_COVERAGE = "low_data_coverage"
    ABSOLUTE_MAX_EXCEEDED = "absolute_max_exceeded"
    ANOMALOUS_READINGS_EXCLUDED = "anomalous_readings_excluded"
    EXCLUDED_ABSOLUTE_MAX_CANDIDATE = "excluded_absolute_max_candidate"
    SESSION_BOUNDARY_INFERRED = "session_boundary_inferred"
    WBGT_OUTSIDE_APPROXIMATION_RANGE = "wbgt_outside_approximation_range"


@dataclass(frozen=True)
class ExposureParams:
    """Everything the calculation depends on; stored with every result."""

    niosh: NoiseCriterion
    india: NoiseCriterion
    wbgt_limit_c: float
    wbgt_category: str
    pm_correction: dict[str, Any]
    reading_interval_s: float
    unaccounted_flag_pct: float
    coverage_flag_pct: float
    engine_version: str
    exclude_record_flags: tuple[str, ...] = ()

    @classmethod
    def from_config(cls) -> ExposureParams:
        """Parameters from config.yaml."""
        limits, methods = config.get_exposure_limits(), config.get_exposure_methods()
        pm = methods["pm_humidity_correction"]
        personal = methods["personal"]
        return cls(
            niosh=NoiseCriterion.from_dict("niosh", limits["criteria"]["niosh"]),
            india=NoiseCriterion.from_dict("india", limits["criteria"]["india"]),
            wbgt_limit_c=float(limits["wbgt_limit_c"]),
            wbgt_category=str(limits["wbgt_work_category"]),
            pm_correction={"kappa": pm["kappa"], "rh_cap_pct": pm["rh_cap_pct"], "method": pm["method"],
                           "threshold_pct": limits["rh_correction_threshold_pct"]},
            reading_interval_s=float(personal["env_reading_interval_s"]),
            unaccounted_flag_pct=float(personal["unaccounted_flag_pct"]),
            coverage_flag_pct=float(personal["coverage_flag_pct"]),
            engine_version=str(personal["engine_version"]),
            exclude_record_flags=tuple(personal["exclude_record_flags"]),
        )

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly copy (stored with every materialised row)."""
        return asdict(self)


@dataclass(frozen=True)
class Interval:
    """One assignment interval [start_ms, end_ms) in one zone, with that zone's readings."""

    zone_id: str
    start_ms: int
    end_ms: int
    readings: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class PersonalExposure:
    """Result for one worker-session. None means 'not computable', never zero."""

    worker_id: str
    session_start: str
    session_end: str
    session_duration_s: float
    l_aeq_db: float | None
    l_ex_8h_db: float | None
    dose_niosh_pct: float | None
    dose_india_pct: float | None
    absolute_max_events: list[dict[str, Any]]
    excluded_anomalous_readings: int
    excluded_absolute_max_candidates: list[dict[str, Any]]
    pm25_twa_raw: float | None
    pm25_twa_corrected: float | None
    pm10_twa_raw: float | None
    pm10_twa_corrected: float | None
    wbgt_max_c: float | None
    wbgt_time_above_limit_s: float | None
    co2_twa_ppm: float | None
    data_coverage_pct: float | None
    coverage_by_metric: dict[str, float | None]
    accounted_time_s: float
    exposure_duration_s: float
    unaccounted_time_s: float
    unaccounted_pct: float
    zones_visited: dict[str, float]
    flags: list[str]
    label: str = LABEL
    measured_personal: dict[str, Any] | None = None      # wearable path; never merged into the above
    dust_label: str = "indicative"
    wbgt_t_nw_method: str = T_NW_METHOD
    wbgt_approximate: bool = True
    wbgt_metabolic_category: str = ""
    criteria_sources: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Plain dict for storage and printing."""
        return asdict(self)


# ---------------------------------------------------------------- helpers


def _iso(ms: int) -> str:
    return from_epoch_ms(ms)


def _overlap_s(lo_ms: int, hi_ms: int, a_ms: int, b_ms: int) -> float:
    """Seconds of overlap between [lo, hi) and [a, b)."""
    return max(0, min(hi_ms, b_ms) - max(lo_ms, a_ms)) / 1000.0


class _Accumulator:
    """Integrates one metric over intervals, imputing missing readings per interval."""

    def __init__(self) -> None:
        self.weighted = 0.0        # sum over intervals of (interval mean * interval duration)
        self.duration_s = 0.0      # duration of intervals that had valid data
        self.valid_s = 0.0         # time actually covered by valid readings

    def add_interval(self, values: list[tuple[float, float]], interval_s: float) -> float | None:
        """Add one interval's (value, overlap seconds) pairs. Returns the interval
        scale factor (interval duration / valid time), or None if no valid data."""
        valid = math.fsum(t for _, t in values)
        if valid <= 0:
            return None
        mean = math.fsum(v * t for v, t in values) / valid
        self.weighted += mean * interval_s
        self.duration_s += interval_s
        self.valid_s += valid
        return interval_s / valid

    def mean(self) -> float | None:
        return None if self.duration_s <= 0 else self.weighted / self.duration_s


# ---------------------------------------------------------------- pure core


def compute_session_exposure(worker_id: str, session_start_ms: int, session_end_ms: int,
                             intervals: list[Interval], params: ExposureParams,
                             session_flags: list[str] | None = None) -> PersonalExposure:
    """Personal exposure over one session from assignment intervals and their readings.

    Args:
        worker_id: worker.
        session_start_ms, session_end_ms: the session [start, end).
        intervals: in-session assignment intervals (UNKNOWN ones included, so that
            their time is reported), each with its zone's zone_env rows.
        params: limits and method parameters.
        session_flags: flags carried by the session (e.g. unmatched_check_in).

    Returns:
        PersonalExposure. Numeric fields are None when not computable.
    """
    session_s = (session_end_ms - session_start_ms) / 1000.0
    if session_s <= 0:
        raise ValueError("session must have positive duration")
    dt_ms = int(params.reading_interval_s * 1000)
    zones: dict[str, float] = defaultdict(float)
    accounted_s = 0.0

    noise_energy = _Accumulator()      # mean of 10^(L/10)
    co2, pm25, pm10, pm25c, pm10c = (_Accumulator() for _ in range(5))
    dose = {"niosh": 0.0, "india": 0.0}
    noise_intervals_with_data = 0.0
    wbgt_above_s, wbgt_max, wbgt_valid_s, wbgt_any_invalid = 0.0, None, 0.0, False
    abs_events: list[dict[str, Any]] = []
    excluded = 0
    excluded_loud: list[dict[str, Any]] = []
    exclude = set(params.exclude_record_flags)
    loud_db = params.india.absolute_max_db

    for iv in intervals:
        a, b = max(iv.start_ms, session_start_ms), min(iv.end_ms, session_end_ms)
        if b <= a:
            continue
        span_s = (b - a) / 1000.0
        if iv.zone_id == UNKNOWN_ZONE:
            continue                                   # stays unaccounted
        zones[iv.zone_id] += span_s
        accounted_s += span_s

        noise_vals, co2_vals, pm25_vals, pm10_vals, pm25c_vals, pm10c_vals, wbgt_vals = [], [], [], [], [], [], []
        for r in iv.readings:
            t = _overlap_s(r["ts_epoch_ms"] - dt_ms, r["ts_epoch_ms"], a, b)
            if t <= 0:
                continue
            if exclude & set(r.get("flags") or ()):
                # A detected sensor anomaly: excluded (lost coverage). A loud one is
                # still reported, because a real event cannot be told apart from a glitch.
                excluded += 1
                if loud_db is not None and r["noise_dba"] is not None and r["noise_dba"] >= loud_db:
                    excluded_loud.append({"ts": r["ts"], "level_db": r["noise_dba"], "zone_id": iv.zone_id,
                                          "flags": sorted(r.get("flags") or ())})
                continue
            if r["noise_dba"] is not None:
                noise_vals.append((r["noise_dba"], t, r["ts"]))
            if r["co2_ppm"] is not None:
                co2_vals.append((float(r["co2_ppm"]), t))
            for name, raw_vals, cor_vals in (("pm25_ugm3", pm25_vals, pm25c_vals),
                                             ("pm10_ugm3", pm10_vals, pm10c_vals)):
                if r[name] is not None:
                    raw_vals.append((r[name], t))
                    if r["rh_pct"] is not None:
                        cor_vals.append((correct_humidity(r[name], r["rh_pct"], params.pm_correction).corrected, t))
            if r["temp_c"] is not None and r["rh_pct"] is not None and r["globe_temp_c"] is not None:
                w = wbgt_estimate(r["temp_c"], r["rh_pct"], r["globe_temp_c"])
                wbgt_vals.append((w.wbgt_c, t))
                wbgt_any_invalid = wbgt_any_invalid or not w.within_validity

        scale = noise_energy.add_interval([(10.0 ** (lv / 10.0), t) for lv, t, _ in noise_vals], span_s)
        if scale is not None:
            noise_intervals_with_data += span_s
            for crit_name, crit in (("niosh", params.niosh), ("india", params.india)):
                dose[crit_name] += noise_dose_pct([lv for lv, _, _ in noise_vals],
                                                  [t * scale for _, t, _ in noise_vals], crit)
            for ev in absolute_max_exceeded([lv for lv, _, _ in noise_vals], [ts for _, _, ts in noise_vals],
                                            params.india):
                abs_events.append({**ev, "zone_id": iv.zone_id})
        co2.add_interval(co2_vals, span_s)
        pm25.add_interval(pm25_vals, span_s)
        pm10.add_interval(pm10_vals, span_s)
        pm25c.add_interval(pm25c_vals, span_s)
        pm10c.add_interval(pm10c_vals, span_s)
        valid_w = math.fsum(t for _, t in wbgt_vals)
        if valid_w > 0:
            wbgt_scale = span_s / valid_w
            wbgt_valid_s += valid_w
            wbgt_above_s += math.fsum(t for w, t in wbgt_vals if w > params.wbgt_limit_c) * wbgt_scale
            top = max(w for w, _ in wbgt_vals)
            wbgt_max = top if wbgt_max is None else max(wbgt_max, top)

    unaccounted_s = max(session_s - accounted_s, 0.0)
    unaccounted_pct = 100.0 * unaccounted_s / session_s

    def cov(acc_valid_s: float) -> float | None:
        return None if accounted_s <= 0 else round(100.0 * min(acc_valid_s / accounted_s, 1.0), 3)

    coverage = {"noise": cov(noise_energy.valid_s), "pm25": cov(pm25.valid_s), "pm10": cov(pm10.valid_s),
                "co2": cov(co2.valid_s), "wbgt": cov(wbgt_valid_s)}

    flags: set[str] = set()
    if session_flags:
        if {"unmatched_check_in", "orphan_check_out"} & set(session_flags):
            flags.add(ResultFlag.SESSION_BOUNDARY_INFERRED)
    if unaccounted_pct > params.unaccounted_flag_pct:
        flags.add(ResultFlag.INCOMPLETE_PRESENCE)
    if wbgt_any_invalid:
        flags.add(ResultFlag.WBGT_OUTSIDE_APPROXIMATION_RANGE)

    mean_energy = noise_energy.mean()
    if mean_energy is None:
        flags.add(ResultFlag.NO_VALID_READINGS)
        l_aeq_db = l_ex = None
        doses: dict[str, float | None] = {"niosh": None, "india": None}
    else:
        l_aeq_db = 10.0 * math.log10(mean_energy)
        l_ex = l_ex_8h(l_aeq_db, noise_intervals_with_data / 3600.0)
        doses = dict(dose)
        if coverage["noise"] is not None and coverage["noise"] < params.coverage_flag_pct:
            flags.add(ResultFlag.LOW_DATA_COVERAGE)
    if abs_events:
        flags.add(ResultFlag.ABSOLUTE_MAX_EXCEEDED)
    if excluded:
        flags.add(ResultFlag.ANOMALOUS_READINGS_EXCLUDED)
    if excluded_loud:
        flags.add(ResultFlag.EXCLUDED_ABSOLUTE_MAX_CANDIDATE)

    return PersonalExposure(
        worker_id=worker_id,
        session_start=_iso(session_start_ms),
        session_end=_iso(session_end_ms),
        session_duration_s=session_s,
        l_aeq_db=l_aeq_db,
        l_ex_8h_db=l_ex,
        dose_niosh_pct=doses["niosh"],
        dose_india_pct=doses["india"],
        absolute_max_events=abs_events,
        excluded_anomalous_readings=excluded,
        excluded_absolute_max_candidates=excluded_loud,
        pm25_twa_raw=pm25.mean(),
        pm25_twa_corrected=pm25c.mean(),
        pm10_twa_raw=pm10.mean(),
        pm10_twa_corrected=pm10c.mean(),
        wbgt_max_c=wbgt_max,
        wbgt_time_above_limit_s=None if wbgt_valid_s <= 0 else wbgt_above_s,
        co2_twa_ppm=co2.mean(),
        data_coverage_pct=coverage["noise"],
        coverage_by_metric=coverage,
        accounted_time_s=accounted_s,
        exposure_duration_s=noise_intervals_with_data,
        unaccounted_time_s=unaccounted_s,
        unaccounted_pct=round(unaccounted_pct, 3),
        zones_visited=dict(sorted(zones.items())),
        flags=sorted(str(f) for f in flags),
        wbgt_metabolic_category=params.wbgt_category,
        criteria_sources={"niosh": params.niosh.source, "india": params.india.source},
    )


# ---------------------------------------------------------------- database adapter


def personal_exposure_for_session(repo: Any, session: dict[str, Any], params: ExposureParams) -> PersonalExposure:
    """Fetch a session's assignments and readings through db.repo and compute the result.

    Assignments marked outside_session are excluded; the rest are clipped to the
    session in compute_session_exposure.
    """
    start_ms, end_ms = session["session_start_ms"], session["session_end_ms"]
    assignments = [a for a in repo.get_zone_assignments(session["worker_id"], session["session_start"],
                                                        session["session_end"])
                   if not a["outside_session"]]
    dt_ms = int(params.reading_interval_s * 1000)
    intervals = []
    for a in assignments:
        readings: list[dict[str, Any]] = []
        if a["zone_id"] != UNKNOWN_ZONE:
            lo, hi = max(a["from_epoch_ms"], start_ms), min(a["to_epoch_ms"], end_ms)
            # A reading covers the 10 s before its ts, so look up to one interval past hi.
            readings = repo.get_zone_env(a["zone_id"], _iso(lo), _iso(hi + dt_ms))
        intervals.append(Interval(a["zone_id"], a["from_epoch_ms"], a["to_epoch_ms"], readings))
    return compute_session_exposure(session["worker_id"], start_ms, end_ms, intervals, params,
                                    session.get("flags") or [])
