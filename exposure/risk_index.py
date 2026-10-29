"""Deterministic, transparent exposure-related risk index. No training, no labels, no model.

THE WEIGHTS AND RAMPS ARE A DESIGN CHOICE OF THIS PROJECT, NOT A VALIDATED CLINICAL
MODEL. The index orders exposure situations; it does not estimate the probability of
any health outcome and does not diagnose anything. It lives in exposure/ (it is a
calculation, not a prediction) and is labelled `deterministic_index`, not
`predicted_risk`.

Components, each 0-100, from values the exposure engine already computed:

* Hearing: L_EX,8h (ISO 9612), blended from today's value and the 7-day energy-mean
  daily value, mapped linearly from `l_ex_floor_db` (-> 0) to `l_ex_ceiling_db`
  (-> 100), clamped at both ends. dB is already logarithmic in acoustic energy, so
  this needs no ad-hoc transform, and it is the unit the standards use. (risk-1.0
  used dose % capped at 100%; it saturated and could not tell 235% from 3,378%.)
  The dose under both criteria is always reported beside the index: the index
  ranks, the dose is the regulated quantity.
* Heat: mean of two linear ramps relative to the ISO 7243 reference value:
  WBGT max (0 at limit-2 C, 100 at limit+6 C) and hours above the limit
  (0 at 0 h, 100 at `hours_above_full_h`).
* Respiratory: COHORT-RELATIVE. The mean, over PM2.5 TWA, PM10 TWA and 7-day PM2.5
  TWA-hours, of the worker's percentile rank among the workers of the same day,
  (count strictly lower) / (N - 1) x 100. There is no regulatory anchor for
  optical PM, so this ranks workers against each other; it says nothing about
  whether any dust level is safe. This is the component most open to misreading.

overall = sum_i w_i c_i / sum_i w_i (weights from config.yaml, equal by default).
Every component, and so the overall index, is non-decreasing in every exposure input.

Confidence = data_coverage_pct x (1 - unaccounted_pct/100). Below
`min_confidence_pct` the BAND is suppressed ("insufficient data") but the number is
still reported. PPE status is carried alongside for display and never changes the
index (CLAUDE.md).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

LABEL = "deterministic_index"
CAVEAT = ("Weights are a design choice of this project, not a validated clinical model. "
          "The respiratory component is relative to the cohort, not to a health threshold.")
COMPONENTS = ("hearing", "heat", "respiratory")


@dataclass(frozen=True)
class RiskParams:
    """Index parameters (config.yaml `risk_index`) plus the WBGT reference value."""

    version: str
    weights: dict[str, float]
    low_max: float
    medium_max: float
    min_confidence_pct: float
    hearing: dict[str, float]
    heat: dict[str, float]
    respiratory_metrics: tuple[str, ...]
    wbgt_limit_c: float

    @classmethod
    def from_config(cls, cfg: dict[str, Any], wbgt_limit_c: float) -> RiskParams:
        """Build from the `risk_index` section."""
        return cls(version=cfg["version"], weights={k: float(v) for k, v in cfg["weights"].items()},
                   low_max=float(cfg["bands"]["low_max"]), medium_max=float(cfg["bands"]["medium_max"]),
                   min_confidence_pct=float(cfg["min_confidence_pct"]), hearing=dict(cfg["hearing"]),
                   heat=dict(cfg["heat"]), respiratory_metrics=tuple(cfg["respiratory"]["metrics"]),
                   wbgt_limit_c=float(wbgt_limit_c))

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly copy (stored with every row)."""
        return {"version": self.version, "weights": self.weights, "low_max": self.low_max,
                "medium_max": self.medium_max, "min_confidence_pct": self.min_confidence_pct,
                "hearing": self.hearing, "heat": self.heat, "respiratory_metrics": list(self.respiratory_metrics),
                "wbgt_limit_c": self.wbgt_limit_c}


def _ramp(x: float, lo: float, hi: float) -> float:
    """0 at or below lo, 100 at or above hi, linear between."""
    if x <= lo:
        return 0.0
    if x >= hi:
        return 100.0
    return 100.0 * (x - lo) / (hi - lo)


def hearing_component(l_ex_8h_db: float, mean_daily_l_ex_7d_db: float, p: RiskParams) -> float:
    """Blend of today's and the 7-day energy-mean L_EX,8h, mapped floor..ceiling -> 0..100.

    The two levels are blended after mapping (each ramp value is linear in dB); the
    map is clamped at both ends.
    """
    h = p.hearing
    lo, hi = h["l_ex_floor_db"], h["l_ex_ceiling_db"]
    blended = (h["daily_weight"] * _ramp(l_ex_8h_db, lo, hi)
               + h["cumulative_weight"] * _ramp(mean_daily_l_ex_7d_db, lo, hi))
    return blended / (h["daily_weight"] + h["cumulative_weight"])


def heat_component(wbgt_max_c: float | None, hours_above: float | None, p: RiskParams) -> float:
    """Mean of the WBGT-max ramp and the time-above ramp (missing heat data counts as 0 here
    and lowers confidence through coverage, never raises the index)."""
    h = p.heat
    w = 0.0 if wbgt_max_c is None else _ramp(wbgt_max_c, p.wbgt_limit_c + h["wbgt_ramp_start_offset_c"],
                                              p.wbgt_limit_c + h["wbgt_ramp_end_offset_c"])
    t = 0.0 if hours_above is None else _ramp(hours_above, 0.0, h["hours_above_full_h"])
    return (h["wbgt_weight"] * w + h["time_weight"] * t) / (h["wbgt_weight"] + h["time_weight"])


def percentile_rank(value: float, cohort: list[float]) -> float:
    """(cohort values strictly below `value`) / (N - 1) x 100; 0 when there is nobody to rank against."""
    others = len(cohort) - 1
    if others <= 0:
        return 0.0
    return 100.0 * sum(1 for v in cohort if v < value) / others


def respiratory_component(values: dict[str, float | None], cohort: dict[str, list[float]]) -> float:
    """Mean percentile rank over the respiratory metrics present (cohort-relative)."""
    ranks = [percentile_rank(v, cohort[k]) for k, v in values.items() if v is not None and cohort.get(k)]
    return sum(ranks) / len(ranks) if ranks else 0.0


def band_for(index: float, p: RiskParams) -> str:
    """Low / Medium / High from the configured thresholds."""
    if index < p.low_max:
        return "Low"
    return "Medium" if index < p.medium_max else "High"


def combine(components: dict[str, float], p: RiskParams) -> tuple[float, dict[str, dict[str, float]]]:
    """Weighted overall index and each component's absolute and % contribution (sums to 100%)."""
    total_w = sum(p.weights[c] for c in COMPONENTS)
    absolute = {c: p.weights[c] * components[c] / total_w for c in COMPONENTS}
    overall = sum(absolute.values())
    contributions = {c: {"absolute": absolute[c],
                         "percent": 100.0 * absolute[c] / overall if overall > 0 else 0.0} for c in COMPONENTS}
    return overall, contributions


def worst_component(components: dict[str, float], p: RiskParams) -> dict[str, Any]:
    """The single highest component, reported ALONGSIDE the weighted composite.

    The components are not on commensurable scales: hearing is anchored to a standard
    (L_EX,8h, ISO 9612), respiratory is cohort-relative. Averaging them can hide one
    clearly high exposure behind two low ones, so both figures are reported, and
    neither replaces the other. Derived from stored components; nothing recomputed.
    Ties go to the first component in COMPONENTS order.
    """
    name = max(COMPONENTS, key=lambda c: components[c])
    return {"component": name, "value": float(components[name]), "band": band_for(components[name], p)}


def compute_index(worker: dict[str, Any], cohort: dict[str, list[float]], p: RiskParams) -> dict[str, Any]:
    """Risk index for one worker-day.

    `worker` holds: worker_id, date, l_ex_8h_db (today), mean_daily_l_ex_7d_db,
    dose_niosh_pct, dose_india_pct (shown beside the index, not inputs),
    wbgt_max_c, wbgt_hours_above, the respiratory metric values,
    data_coverage_pct, unaccounted_pct, ppe_status. `cohort` maps each respiratory
    metric to the same-day values of every worker (including this one).
    """
    comps = {
        "hearing": hearing_component(worker["l_ex_8h_db"], worker["mean_daily_l_ex_7d_db"], p),
        "heat": heat_component(worker["wbgt_max_c"], worker["wbgt_hours_above"], p),
        "respiratory": respiratory_component({m: worker.get(m) for m in p.respiratory_metrics}, cohort),
    }
    overall, contributions = combine(comps, p)
    coverage = worker["data_coverage_pct"] if worker["data_coverage_pct"] is not None else 0.0
    confidence = coverage * (1.0 - worker["unaccounted_pct"] / 100.0)
    insufficient = confidence < p.min_confidence_pct
    describe = {
        "hearing": (f"noise L_EX,8h {worker['l_ex_8h_db']:.1f} dBA (dose {worker['dose_niosh_pct']:,.0f}% NIOSH, "
                    f"{worker['dose_india_pct']:,.0f}% India)"),
        "heat": ("heat: WBGT max {:.1f} C, {:.1f} h above the {:.0f} C reference (approximate)".format(
            worker["wbgt_max_c"] or 0.0, worker["wbgt_hours_above"] or 0.0, p.wbgt_limit_c)),
        "respiratory": "dust higher than {:.0f}% of co-workers today (cohort-relative, indicative)".format(
            comps["respiratory"]),
    }
    ranked = sorted(COMPONENTS, key=lambda c: -contributions[c]["absolute"])
    drivers = [describe[c] for c in ranked[:2] if contributions[c]["absolute"] > 0]
    return {
        "worker_id": worker["worker_id"], "date": worker["date"], "overall_index": overall,
        "band": "insufficient data" if insufficient else band_for(overall, p),
        "insufficient_data": insufficient, "components": comps, "contributions": contributions,
        "drivers": drivers, "confidence_pct": confidence, "ppe_status": worker.get("ppe_status") or {},
        "l_ex_8h_db": worker["l_ex_8h_db"], "dose_niosh_pct": worker["dose_niosh_pct"],
        "dose_india_pct": worker["dose_india_pct"],
        "label": LABEL, "weights": p.weights, "version": p.version, "params": {**p.as_dict(), "caveat": CAVEAT},
    }


def worker_day_inputs(shift_rows: list[dict[str, Any]], cumulative_7d: dict[str, dict[str, Any]],
                      ppe: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Combine one day's personal_exposure_shift rows per worker into index inputs."""
    by_worker: dict[str, list[dict[str, Any]]] = {}
    for r in shift_rows:
        by_worker.setdefault(r["worker_id"], []).append(r)
    out = []
    for wid, rows in sorted(by_worker.items()):
        valid = [r for r in rows if r["dose_niosh_pct"] is not None]
        if not valid:
            continue
        niosh = sum(r["dose_niosh_pct"] for r in valid)
        india = sum(r["dose_india_pct"] for r in valid)
        # Daily L_EX,8h of several sessions combines on an energy basis.
        lex = 10.0 * math.log10(sum(10.0 ** (r["l_ex_8h_db"] / 10.0) for r in valid))
        cum = cumulative_7d.get(wid, {})
        acc = sum(r["accounted_time_s"] for r in valid)
        sess = sum(r["session_duration_s"] for r in valid)
        cov = sum((r["data_coverage_pct"] or 0.0) * r["accounted_time_s"] for r in valid) / acc if acc else None
        wbgt = [r["wbgt_max_c"] for r in valid if r["wbgt_max_c"] is not None]
        above = [r["wbgt_time_above_limit_s"] for r in valid if r["wbgt_time_above_limit_s"] is not None]

        def twa(key: str) -> float | None:
            pairs = [(r[key], r["accounted_time_s"]) for r in valid if r[key] is not None]
            w = sum(t for _, t in pairs)
            return sum(v * t for v, t in pairs) / w if w else None

        out.append({"worker_id": wid, "date": rows[0]["shift_date"],
                    "l_ex_8h_db": lex, "dose_niosh_pct": niosh, "dose_india_pct": india,
                    "mean_daily_l_ex_7d_db": cum.get("mean_daily_l_ex_8h_db") or lex,
                    "wbgt_max_c": max(wbgt) if wbgt else None,
                    "wbgt_hours_above": sum(above) / 3600.0 if above else None,
                    "pm25_twa_raw": twa("pm25_twa_raw"), "pm10_twa_raw": twa("pm10_twa_raw"),
                    "pm25_twa_hours_raw_7d": cum.get("pm25_twa_hours_raw"),
                    "data_coverage_pct": cov,
                    "unaccounted_pct": 100.0 * (sess - acc) / sess if sess else 100.0,
                    "ppe_status": ppe.get(wid, {})})
    return out


def compute_day(inputs: list[dict[str, Any]], p: RiskParams) -> list[dict[str, Any]]:
    """Index rows for every worker of one day (the cohort for the respiratory ranks)."""
    cohort = {m: [w[m] for w in inputs if w.get(m) is not None] for m in p.respiratory_metrics}
    return [compute_index(w, cohort, p) for w in inputs]
