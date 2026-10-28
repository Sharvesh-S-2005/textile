"""Heat stress: indoor WBGT (ISO 7243) from air temperature, humidity and globe temperature.

THE KNOWN WEAK POINT. ISO 7243 needs the NATURAL wet-bulb temperature T_nw (a
wetted sensor exposed to the room's actual air movement and radiation). The zone
nodes measure air temperature and RH instead, so T_nw is replaced by Stull's
(2011) PSYCHROMETRIC wet-bulb approximation. Psychrometric wet-bulb is not natural
wet-bulb: in still air T_nw is generally HIGHER than T_pwb, so the substitute
UNDERESTIMATES WBGT - it errs toward reporting conditions as safer than they are.
Every WBGT value produced here therefore carries
`t_nw_method = "stull_psychrometric_approximation"` and `approximate = True`, and
those flags must travel with the value all the way to the dashboard.

Pure functions: no database, no config reads.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

T_NW_METHOD = "stull_psychrometric_approximation"
APPROXIMATION_NOTE = ("T_nw approximated by Stull (2011) psychrometric wet-bulb; natural wet-bulb is "
                      "generally higher in still air, so this WBGT is likely an UNDERESTIMATE")
# Stull (2011) states the fit is valid for these ranges, with error -1 to +0.65 degC.
STULL_VALID_T_C = (-20.0, 50.0)
STULL_VALID_RH_PCT = (5.0, 99.0)


def wbgt_indoor(t_nw_c: float, t_g_c: float) -> float:
    """Indoor WBGT, no solar load (ISO 7243:2017).

        WBGT = 0.7 * T_nw + 0.3 * T_g

    The formula alone. Use `wbgt_estimate` for values that carry their
    approximation flags.
    """
    return 0.7 * t_nw_c + 0.3 * t_g_c


def stull_wet_bulb(temp_c: float, rh_pct: float) -> float:
    """Psychrometric wet-bulb temperature (Stull 2011, J. Appl. Meteor. Climatol. 50, 2267).

        T_w = T atan[0.151977 (RH + 8.313659)^(1/2)] + atan(T + RH) - atan(RH - 1.676331)
              + 0.00391838 RH^(3/2) atan(0.023101 RH) - 4.686035

    T in degC, RH in percent, sea-level pressure. Stated validity: RH 5-99%,
    T -20 to 50 degC, error -1 to +0.65 degC. Worked example in the paper:
    T = 20 degC, RH = 50% -> T_w = 13.7 degC.
    """
    t, rh = temp_c, rh_pct
    return (t * math.atan(0.151977 * math.sqrt(rh + 8.313659)) + math.atan(t + rh) - math.atan(rh - 1.676331)
            + 0.00391838 * rh ** 1.5 * math.atan(0.023101 * rh) - 4.686035)


@dataclass(frozen=True)
class WetBulbEstimate:
    """An estimated wet-bulb temperature with the metadata that must travel with it."""

    value_c: float
    t_nw_method: str
    approximate: bool
    within_validity: bool
    note: str


def estimate_natural_wet_bulb(temp_c: float, rh_pct: float) -> WetBulbEstimate:
    """Stand-in for natural wet-bulb T_nw, from air temperature and RH.

    PSYCHROMETRIC WET-BULB IS NOT NATURAL WET-BULB. This returns Stull's (2011)
    psychrometric approximation; T_nw is generally higher in still air, so any
    WBGT built on it underestimates heat stress (the unsafe direction). The result
    always says so: t_nw_method = "stull_psychrometric_approximation",
    approximate = True. `within_validity` is False outside Stull's stated range.
    """
    if temp_c is None or rh_pct is None or not (math.isfinite(temp_c) and math.isfinite(rh_pct)):
        raise ValueError(f"invalid temperature/RH {temp_c!r}/{rh_pct!r}")
    valid = (STULL_VALID_T_C[0] <= temp_c <= STULL_VALID_T_C[1]
             and STULL_VALID_RH_PCT[0] <= rh_pct <= STULL_VALID_RH_PCT[1])
    return WetBulbEstimate(value_c=stull_wet_bulb(temp_c, rh_pct), t_nw_method=T_NW_METHOD,
                           approximate=True, within_validity=valid, note=APPROXIMATION_NOTE)


@dataclass(frozen=True)
class WbgtEstimate:
    """A WBGT value with its approximation flags (never strip these)."""

    wbgt_c: float
    t_nw_c: float
    t_g_c: float
    t_nw_method: str
    approximate: bool
    within_validity: bool

    def as_dict(self) -> dict[str, Any]:
        """Plain dict, flags included."""
        return {"wbgt_c": self.wbgt_c, "t_nw_c": self.t_nw_c, "t_g_c": self.t_g_c,
                "t_nw_method": self.t_nw_method, "approximate": self.approximate,
                "within_validity": self.within_validity}


def wbgt_estimate(temp_c: float, rh_pct: float, globe_c: float) -> WbgtEstimate:
    """Indoor WBGT (ISO 7243) from zone-node readings, with the T_nw approximation flagged."""
    if globe_c is None or not math.isfinite(globe_c):
        raise ValueError(f"invalid globe temperature {globe_c!r}")
    wet = estimate_natural_wet_bulb(temp_c, rh_pct)
    return WbgtEstimate(wbgt_c=wbgt_indoor(wet.value_c, globe_c), t_nw_c=wet.value_c, t_g_c=globe_c,
                        t_nw_method=wet.t_nw_method, approximate=True, within_validity=wet.within_validity)


def wbgt_exceedance(wbgt_series: Sequence[float], durations_s: Sequence[float], limit_c: float,
                    metabolic_category: str) -> dict[str, Any]:
    """Time above the ISO 7243 reference value, and the maximum WBGT reached.

    `limit_c` is the ISO 7243 reference value for `metabolic_category` (moderate
    work, acclimatised: 28 degC in config.yaml). The category assumption is
    recorded in the result, as are the approximation flags. Time is counted only
    while WBGT is strictly above the limit.

    Raises ValueError on empty or invalid input (missing readings must be excluded
    by the caller, never treated as cool).
    """
    if len(wbgt_series) == 0:
        raise ValueError("no WBGT values given")
    if len(wbgt_series) != len(durations_s):
        raise ValueError(f"{len(wbgt_series)} values but {len(durations_s)} durations")
    for w, d in zip(wbgt_series, durations_s):
        if w is None or not math.isfinite(w) or d is None or d <= 0:
            raise ValueError(f"invalid WBGT/duration {w!r}/{d!r}")
    above = math.fsum(d for w, d in zip(wbgt_series, durations_s) if w > limit_c)
    return {"time_above_limit_s": above, "max_wbgt_c": max(wbgt_series), "limit_c": limit_c,
            "metabolic_category": metabolic_category, "t_nw_method": T_NW_METHOD, "approximate": True}
