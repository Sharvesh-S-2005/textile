"""Hand-verified tests for exposure/heat.py."""

import math

import pytest

import config
from exposure.heat import (T_NW_METHOD, estimate_natural_wet_bulb, stull_wet_bulb, wbgt_estimate, wbgt_exceedance,
                           wbgt_indoor)

LIMIT = config.get_exposure_limits()["wbgt_limit_c"]
CATEGORY = config.get_exposure_limits()["wbgt_work_category"]


def psychrometric_wet_bulb(temp_c: float, rh_pct: float, p_hpa: float = 1013.25) -> float:
    """Independent reference: solve the psychrometer equation e = e_s(Tw) - A P (T - Tw)
    (Magnus saturation curve, Alduchov-Eskridge coefficients; ventilated psychrometer
    constant A = 6.6e-4 (1 + 0.00115 Tw) /K) by bisection."""
    def es(t: float) -> float:
        return 6.112 * math.exp(17.62 * t / (243.12 + t))
    e = rh_pct / 100 * es(temp_c)
    lo, hi = -40.0, temp_c
    for _ in range(200):
        mid = (lo + hi) / 2
        f = es(mid) - 6.6e-4 * (1 + 0.00115 * mid) * p_hpa * (temp_c - mid) - e
        lo, hi = (mid, hi) if f < 0 else (lo, mid)
    return mid


def test_wbgt_indoor_hand_value() -> None:
    """0.7 * 26 + 0.3 * 35 = 18.2 + 10.5 = 28.7 degC."""
    assert wbgt_indoor(26.0, 35.0) == pytest.approx(28.7)


def test_stull_reproduces_the_papers_worked_example() -> None:
    """Stull (2011): T = 20 degC, RH = 50% -> Tw = 13.7 degC."""
    assert stull_wet_bulb(20.0, 50.0) == pytest.approx(13.7, abs=0.05)


@pytest.mark.parametrize("t,rh", [(30.0, 50.0), (33.0, 72.5), (38.0, 85.0), (28.0, 60.0)])
def test_stull_agrees_with_psychrometric_equations_within_stated_error(t: float, rh: float) -> None:
    """Second reference: the standard sea-level psychrometric relation (independent
    solve). Stull states his error as -1 to +0.65 degC; mill conditions are checked."""
    diff = stull_wet_bulb(t, rh) - psychrometric_wet_bulb(t, rh)
    assert -1.0 <= diff <= 0.65, diff


def test_estimate_always_carries_the_approximation_flag() -> None:
    est = estimate_natural_wet_bulb(33.0, 72.5)
    assert est.t_nw_method == T_NW_METHOD == "stull_psychrometric_approximation"
    assert est.approximate is True and est.within_validity is True
    assert "UNDERESTIMATE" in est.note


def test_outside_stull_validity_is_flagged_not_hidden() -> None:
    est = estimate_natural_wet_bulb(55.0, 50.0)
    assert est.within_validity is False and est.approximate is True


def test_wbgt_estimate_hand_value_and_flags() -> None:
    """T 20, RH 50 -> Tw 13.70 (Stull); globe 30 -> 0.7*13.70 + 0.3*30 = 18.59 degC."""
    w = wbgt_estimate(20.0, 50.0, 30.0)
    assert w.wbgt_c == pytest.approx(0.7 * 13.70 + 9.0, abs=0.05)
    d = w.as_dict()
    assert d["t_nw_method"] == T_NW_METHOD and d["approximate"] is True


def test_wbgt_exceedance_hand_values() -> None:
    """Values 27, 29, 31 for 10, 20, 30 min; limit 28 -> 20 + 30 = 50 min above; max 31."""
    r = wbgt_exceedance([27.0, 29.0, 31.0], [600.0, 1200.0, 1800.0], 28.0, "moderate")
    assert r["time_above_limit_s"] == 3000.0 and r["max_wbgt_c"] == 31.0
    assert r["metabolic_category"] == "moderate" and r["approximate"] is True
    assert r["t_nw_method"] == T_NW_METHOD


def test_exactly_at_limit_is_not_above() -> None:
    assert wbgt_exceedance([28.0], [600.0], 28.0, "moderate")["time_above_limit_s"] == 0.0


def test_config_limit_is_moderate_work() -> None:
    assert LIMIT == 28.0 and CATEGORY == "moderate"


def test_missing_values_raise_rather_than_count_as_cool() -> None:
    with pytest.raises(ValueError):
        wbgt_exceedance([30.0, None], [600.0, 600.0], 28.0, "moderate")
    with pytest.raises(ValueError):
        wbgt_exceedance([], [], 28.0, "moderate")
    with pytest.raises(ValueError):
        wbgt_estimate(30.0, 70.0, None)
