"""Noise exposure: equivalent level, daily exposure level, dose, absolute maximum.

Standards:
  * ISO 9612:2009, Acoustics - Determination of occupational noise exposure -
    Engineering method (L_Aeq,T and L_EX,8h, energy averaging).
  * NIOSH 98-126 (1998), Criteria for a Recommended Standard: Occupational Noise
    Exposure (85 dBA criterion, 3 dB exchange rate, 80 dBA threshold, dose formula).
  * Schedule XXVIII under Rule 95 of the Tamil Nadu Factories Rules (90 dBA for
    8 h, 5 dB exchange rate, levels below 90 dBA not counted, 115 dBA absolute
    maximum).

Pure functions: no database, no config. Limits are passed in, so every function
can be checked by hand in isolation.

dB is logarithmic. Levels are only ever combined on an energy basis
(10 ** (L / 10)); nothing in this file averages decibels arithmetically.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

# ISO 9612 normalises daily exposure to a reference duration T0 of 8 hours. This is
# part of the definition of L_EX,8h, not a tunable limit.
REFERENCE_DURATION_H = 8.0


@dataclass(frozen=True)
class NoiseCriterion:
    """One noise exposure criterion (values come from config.yaml exposure_limits)."""

    name: str
    criterion_db: float          # level allowed for `duration_h` (100% dose)
    exchange_rate_db: float      # level change that halves / doubles the allowed time
    duration_h: float            # reference duration of the criterion
    threshold_db: float          # levels below this contribute nothing to the dose
    source: str
    absolute_max_db: float | None = None   # hard ceiling (alarm), not a dose term

    @classmethod
    def from_dict(cls, name: str, spec: dict[str, Any]) -> NoiseCriterion:
        """Build from an `exposure_limits.criteria.<name>` mapping."""
        return cls(name=name, criterion_db=float(spec["criterion_db"]),
                   exchange_rate_db=float(spec["exchange_rate_db"]), duration_h=float(spec["duration_h"]),
                   threshold_db=float(spec["threshold_db"]), source=str(spec["source"]),
                   absolute_max_db=None if spec.get("absolute_max_db") is None else float(spec["absolute_max_db"]))


def _check(levels_db: Sequence[float], durations_s: Sequence[float]) -> None:
    """Fail loudly on empty, mismatched, missing or non-positive input."""
    if len(levels_db) == 0:
        raise ValueError("no noise levels given: an empty input has no exposure, not zero exposure")
    if len(levels_db) != len(durations_s):
        raise ValueError(f"{len(levels_db)} levels but {len(durations_s)} durations")
    for level in levels_db:
        if level is None or not math.isfinite(level):
            raise ValueError(f"invalid level {level!r}: missing readings must be excluded by the caller")
    for d in durations_s:
        if d is None or not math.isfinite(d) or d <= 0:
            raise ValueError(f"invalid duration {d!r}: durations must be positive seconds")


def l_aeq(levels_db: Sequence[float], durations_s: Sequence[float]) -> float:
    """Energy-equivalent continuous A-weighted level L_Aeq,T (ISO 9612:2009).

        L_Aeq,T = 10 * log10( (1/T) * sum_i t_i * 10^(L_i / 10) ),   T = sum_i t_i

    Supports unequal durations; with equal durations it reduces to
    10 * log10(mean(10 ** (L / 10))). Never an arithmetic mean of dB.

    Raises ValueError on empty or invalid input.
    """
    _check(levels_db, durations_s)
    total_s = math.fsum(durations_s)
    energy = math.fsum(t * 10.0 ** (level / 10.0) for level, t in zip(levels_db, durations_s))
    return 10.0 * math.log10(energy / total_s)


def l_ex_8h(l_aeq_db: float, exposure_duration_h: float) -> float:
    """Daily noise exposure level L_EX,8h (ISO 9612:2009, eq. for L_EX,8h).

        L_EX,8h = L_Aeq,Te + 10 * log10(T_e / T_0),   T_0 = 8 h

    `exposure_duration_h` is the effective duration T_e of the working day.
    """
    if not math.isfinite(l_aeq_db):
        raise ValueError(f"invalid L_Aeq {l_aeq_db!r}")
    if exposure_duration_h <= 0:
        raise ValueError(f"exposure duration must be positive, got {exposure_duration_h!r} h")
    return l_aeq_db + 10.0 * math.log10(exposure_duration_h / REFERENCE_DURATION_H)


def allowed_time_h(level_db: float, criterion_db: float, exchange_rate_db: float, duration_h: float) -> float:
    """Permitted exposure time at a level (NIOSH 98-126; TN Factories Rules Sch. XXVIII).

        T_i = duration_h / 2 ** ((L_i - criterion_db) / exchange_rate_db)

    e.g. NIOSH (85, 3, 8 h): 91 dBA -> 2 h. India (90, 5, 8 h): 95 dBA -> 4 h.
    """
    if exchange_rate_db <= 0 or duration_h <= 0:
        raise ValueError("exchange rate and duration must be positive")
    return duration_h / 2.0 ** ((level_db - criterion_db) / exchange_rate_db)


def noise_dose_pct(levels_db: Sequence[float], durations_s: Sequence[float], criterion: NoiseCriterion) -> float:
    """Noise dose in percent (NIOSH 98-126; Schedule XXVIII, Rule 95, TN Factories Rules).

        Dose% = 100 * sum_i (C_i / T_i)

    C_i is the time at level L_i and T_i the permitted time at that level
    (allowed_time_h). Levels below `criterion.threshold_db` contribute zero - 80 dBA
    for NIOSH, 90 dBA for India, so an hour at 85 dBA counts under NIOSH and not
    under India. A level exactly at the threshold counts (the conservative reading).

    Raises ValueError on empty or invalid input.
    """
    _check(levels_db, durations_s)
    total = 0.0
    for level, t_s in zip(levels_db, durations_s):
        if level < criterion.threshold_db:
            continue
        allowed_h = allowed_time_h(level, criterion.criterion_db, criterion.exchange_rate_db, criterion.duration_h)
        total += (t_s / 3600.0) / allowed_h
    return 100.0 * total


def absolute_max_exceeded(levels_db: Sequence[float], timestamps: Sequence[datetime | str],
                          criterion: NoiseCriterion) -> list[dict[str, Any]]:
    """Readings at or above the criterion's absolute maximum (Schedule XXVIII: 115 dBA).

    This is a hard alarm condition, reported separately - never a dose term. The
    schedule permits no exposure above 115 dBA; a reading AT 115 dBA is reported
    too (the conservative reading of an ambiguous boundary). Criteria without an
    absolute maximum (NIOSH here) return an empty list.

    Returns [{"ts": ..., "level_db": ...}, ...] in input order.
    """
    if len(levels_db) != len(timestamps):
        raise ValueError(f"{len(levels_db)} levels but {len(timestamps)} timestamps")
    if criterion.absolute_max_db is None:
        return []
    return [{"ts": ts, "level_db": level} for level, ts in zip(levels_db, timestamps)
            if level is not None and level >= criterion.absolute_max_db]
