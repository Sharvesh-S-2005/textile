"""Dust exposure from optical PM sensors: time-weighted average and humidity correction.

INDICATIVE ONLY. Regulatory cotton-dust limits are defined for the lint-free
respirable fraction collected by vertical elutriator and weighed (gravimetric). A
low-cost optical PM2.5/PM10 sensor does not measure that fraction, so nothing in
this module - or anywhere in this codebase - compares a PM reading with a legal
limit. Dust outputs are for trends, zone comparison and ML features, and every one
is labelled "indicative".

Pure functions: no database, no config reads. Parameters are passed in.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

LABEL = "indicative"


def twa(concentrations: Sequence[float], durations_s: Sequence[float]) -> float:
    """Time-weighted average concentration.

        TWA = sum_i (C_i * t_i) / sum_i t_i

    This is a LINEAR average. Dust mass concentration (ug/m3) is not logarithmic,
    so unlike noise (exposure/noise.py, energy-averaged dB) a plain weighted mean is
    correct here. Do not carry the dB instinct across.

    Raises ValueError on empty, mismatched, missing or non-positive input.
    """
    if len(concentrations) == 0:
        raise ValueError("no concentrations given: empty input has no TWA, not zero")
    if len(concentrations) != len(durations_s):
        raise ValueError(f"{len(concentrations)} concentrations but {len(durations_s)} durations")
    for c in concentrations:
        if c is None or not math.isfinite(c) or c < 0:
            raise ValueError(f"invalid concentration {c!r}: missing readings must be excluded by the caller")
    for d in durations_s:
        if d is None or not math.isfinite(d) or d <= 0:
            raise ValueError(f"invalid duration {d!r}")
    return math.fsum(c * t for c, t in zip(concentrations, durations_s)) / math.fsum(durations_s)


@dataclass(frozen=True)
class PMCorrection:
    """A humidity-corrected PM value that always keeps the raw reading beside it."""

    raw: float
    corrected: float
    correction_applied: bool
    kappa: float
    rh_pct: float
    method: str
    label: str = LABEL

    def as_dict(self) -> dict[str, Any]:
        """Plain dict for storage."""
        return {"raw": self.raw, "corrected": self.corrected, "correction_applied": self.correction_applied,
                "kappa": self.kappa, "rh_pct": self.rh_pct, "method": self.method, "label": self.label}


def growth_factor(rh_pct: float, kappa: float, rh_cap_pct: float) -> float:
    """Hygroscopic mass growth factor 1 + kappa * a_w / (1 - a_w), a_w = RH/100 (capped)."""
    a_w = min(rh_pct, rh_cap_pct) / 100.0
    return 1.0 + kappa * a_w / (1.0 - a_w)


def correct_humidity(pm_raw: float, rh_pct: float, params: dict[str, Any]) -> PMCorrection:
    """Hygroscopic-growth correction of an optical PM reading (Crilley et al. 2018).

        C_corrected = C_raw / (1 + kappa * (a_w / (1 - a_w))),   a_w = RH / 100

    Source of the form: Crilley, L. R. et al. (2018), "Evaluation of a low-cost
    optical particle counter (Alphasense OPC-N2) for ambient air monitoring",
    Atmospheric Measurement Techniques 11, 709-720 (kappa-Koehler growth).

    Applied only when RH is above `params["threshold_pct"]` (75% in config.yaml);
    below it the corrected value equals the raw value. Above the threshold the
    corrected value falls monotonically as RH rises. Note the step at the
    threshold: the correction switches on, it does not fade in.

    ASSUMPTION: `kappa` was derived for ambient aerosol. Cotton dust is fibrous
    cellulose and far less hygroscopic, so kappa here is an assumption of this
    project, not a measurement (set low; see config.yaml). The RH cap
    (`rh_cap_pct`) is also a project assumption, for numerical stability.

    Args:
        pm_raw: raw sensor value (ug/m3), never modified.
        rh_pct: relative humidity at the same instant (%).
        params: {"kappa", "rh_cap_pct", "threshold_pct", "method"}.

    Returns:
        PMCorrection with raw, corrected, correction_applied and the kappa used.
    """
    if pm_raw is None or not math.isfinite(pm_raw) or pm_raw < 0:
        raise ValueError(f"invalid raw PM value {pm_raw!r}")
    if rh_pct is None or not 0.0 <= rh_pct <= 100.0:
        raise ValueError(f"invalid RH {rh_pct!r}")
    kappa = float(params["kappa"])
    applied = rh_pct > float(params["threshold_pct"])
    corrected = pm_raw / growth_factor(rh_pct, kappa, float(params["rh_cap_pct"])) if applied else pm_raw
    return PMCorrection(raw=pm_raw, corrected=corrected, correction_applied=applied, kappa=kappa,
                        rh_pct=rh_pct, method=str(params["method"]))
