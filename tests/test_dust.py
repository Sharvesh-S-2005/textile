"""Hand-verified tests for exposure/dust.py."""

import re
from pathlib import Path

import pytest

import config
from exposure.dust import LABEL, correct_humidity, growth_factor, twa

METHOD = config.get_exposure_methods()["pm_humidity_correction"]
PARAMS = {"kappa": METHOD["kappa"], "rh_cap_pct": METHOD["rh_cap_pct"], "method": METHOD["method"],
          "threshold_pct": config.get_exposure_limits()["rh_correction_threshold_pct"]}
H = 3600.0


def test_twa_equal_durations() -> None:
    """(100 + 200 + 300) / 3 = 200 ug/m3."""
    assert twa([100.0, 200.0, 300.0], [H, H, H]) == pytest.approx(200.0)


def test_twa_unequal_durations() -> None:
    """6 h @ 400 + 2 h @ 100: (2400 + 200) / 8 = 325 ug/m3."""
    assert twa([400.0, 100.0], [6 * H, 2 * H]) == pytest.approx(325.0)


def test_twa_is_linear_not_energy_averaged() -> None:
    """Linear: (10 + 1000)/2 = 505. An energy-style average would give ~700."""
    assert twa([10.0, 1000.0], [H, H]) == pytest.approx(505.0)


def test_twa_empty_or_missing_raises() -> None:
    with pytest.raises(ValueError):
        twa([], [])
    with pytest.raises(ValueError):
        twa([100.0, None], [H, H])


def test_correction_is_a_no_op_at_and_below_threshold() -> None:
    for rh in (40.0, 60.0, PARAMS["threshold_pct"]):
        c = correct_humidity(250.0, rh, PARAMS)
        assert c.corrected == 250.0 and c.raw == 250.0 and c.correction_applied is False


def test_correction_hand_value_at_85_percent() -> None:
    """kappa 0.05, RH 85: a_w/(1-a_w) = 0.85/0.15 = 5.667; factor 1 + 0.05*5.667 = 1.2833;
    300 / 1.2833 = 233.77 ug/m3."""
    c = correct_humidity(300.0, 85.0, {**PARAMS, "kappa": 0.05})
    assert c.correction_applied and c.kappa == 0.05
    assert c.corrected == pytest.approx(233.77, abs=0.01)


def test_correction_is_monotonic_above_threshold() -> None:
    rhs = [76.0, 80.0, 85.0, 90.0, 94.0, 95.0]
    values = [correct_humidity(300.0, rh, PARAMS).corrected for rh in rhs]
    assert all(a > b for a, b in zip(values, values[1:]))
    assert all(v < 300.0 for v in values)


def test_rh_cap_prevents_divergence() -> None:
    capped = correct_humidity(300.0, 100.0, PARAMS)
    at_cap = correct_humidity(300.0, PARAMS["rh_cap_pct"], PARAMS)
    assert capped.corrected == pytest.approx(at_cap.corrected) and capped.corrected > 0


def test_raw_value_survives_correction_and_is_labelled() -> None:
    c = correct_humidity(300.0, 90.0, PARAMS)
    d = c.as_dict()
    assert d["raw"] == 300.0 and d["corrected"] < 300.0
    assert d["kappa"] == PARAMS["kappa"] and d["correction_applied"] is True
    assert d["label"] == LABEL == "indicative"


def test_growth_factor_is_one_at_zero_kappa() -> None:
    assert growth_factor(90.0, 0.0, 95.0) == 1.0


def test_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError):
        correct_humidity(None, 80.0, PARAMS)
    with pytest.raises(ValueError):
        correct_humidity(100.0, 120.0, PARAMS)


def test_no_regulatory_dust_limit_comparison_exists() -> None:
    """Optical PM is never compared with a legal cotton-dust limit (CLAUDE.md)."""
    root = Path(__file__).resolve().parents[1]
    pattern = re.compile(r"(cotton_dust_limit|dust_limit|pel_|osha|permissible_dust|regulatory_limit)", re.I)
    for path in list((root / "exposure").rglob("*.py")) + [root / "config.yaml"]:
        assert not pattern.search(path.read_text(encoding="utf-8")), path.name
