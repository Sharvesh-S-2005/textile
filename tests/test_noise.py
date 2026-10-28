"""Hand-verified tests for exposure/noise.py.

Every expected value below was computed by hand from the published formulas
(the working is in each docstring). Tolerances: 0.1 dB for levels, 0.5% for doses.
"""

import math

import pytest

import config
from exposure.noise import NoiseCriterion, absolute_max_exceeded, allowed_time_h, l_aeq, l_ex_8h, noise_dose_pct

H = 3600.0
CRITERIA = config.get_exposure_limits()["criteria"]
NIOSH = NoiseCriterion.from_dict("niosh", CRITERIA["niosh"])
INDIA = NoiseCriterion.from_dict("india", CRITERIA["india"])
DB, PCT = 0.1, 0.5


def dose(level: float, hours: float, criterion: NoiseCriterion) -> float:
    return noise_dose_pct([level], [hours * H], criterion)


# ---------------------------------------------------------------- the twelve cases

def test_case_01_8h_at_85_niosh() -> None:
    """8 h @ 85 dBA, NIOSH. L_EX,8h = 85 + 10 log10(8/8) = 85.0. T = 8/2^0 = 8 h -> 8/8 = 100%."""
    assert l_ex_8h(l_aeq([85.0], [8 * H]), 8.0) == pytest.approx(85.0, abs=DB)
    assert dose(85.0, 8, NIOSH) == pytest.approx(100.0, abs=PCT)


def test_case_02_2h_at_91_niosh() -> None:
    """2 h @ 91 dBA, NIOSH. T = 8 / 2^((91-85)/3) = 8/4 = 2 h -> 2/2 = 100%."""
    assert dose(91.0, 2, NIOSH) == pytest.approx(100.0, abs=PCT)


def test_case_03_4h_at_91_niosh_is_200_percent() -> None:
    """4 h @ 91 dBA, NIOSH. T = 2 h (case 2), so 4/2 = 200%. (Not 100%: that figure,
    stated in an early planning message, was wrong.)"""
    assert dose(91.0, 4, NIOSH) == pytest.approx(200.0, abs=PCT)


def test_case_04_1h_at_94_niosh() -> None:
    """1 h @ 94 dBA, NIOSH. T = 8 / 2^(9/3) = 8/8 = 1 h -> 100%."""
    assert dose(94.0, 1, NIOSH) == pytest.approx(100.0, abs=PCT)


def test_case_05_8h_at_90_india() -> None:
    """8 h @ 90 dBA, India. Schedule XXVIII row: 90 dBA permitted 8 h -> 100%.
    Agreement with the published Tamil Nadu Factories Rules table."""
    assert dose(90.0, 8, INDIA) == pytest.approx(100.0, abs=PCT)


def test_case_06_4h_at_95_india() -> None:
    """4 h @ 95 dBA, India. Schedule XXVIII row: 95 dBA permitted 4 h -> 100%.
    Agreement with the published Tamil Nadu Factories Rules table."""
    assert dose(95.0, 4, INDIA) == pytest.approx(100.0, abs=PCT)


def test_case_07_2h_at_100_india() -> None:
    """2 h @ 100 dBA, India. Schedule XXVIII row: 100 dBA permitted 2 h -> 100%.
    Agreement with the published Tamil Nadu Factories Rules table."""
    assert dose(100.0, 2, INDIA) == pytest.approx(100.0, abs=PCT)


def test_case_08_15min_at_115_india_and_absolute_max_alarm() -> None:
    """0.25 h @ 115 dBA, India. Schedule XXVIII row: 115 dBA permitted 15 min ->
    8 / 2^(25/5) = 0.25 h -> 100%. 115 dBA is also the absolute maximum, so the
    alarm fires (reported separately from the dose). Agreement with the
    published Tamil Nadu Factories Rules table."""
    assert dose(115.0, 0.25, INDIA) == pytest.approx(100.0, abs=PCT)
    events = absolute_max_exceeded([115.0], ["2026-10-28T04:00:00Z"], INDIA)
    assert events == [{"ts": "2026-10-28T04:00:00Z", "level_db": 115.0}]


def test_case_09_8h_at_85_india_is_zero() -> None:
    """8 h @ 85 dBA, India. 85 < 90 dBA threshold -> contributes nothing -> 0%.
    (Under NIOSH the same exposure is 100%: the thresholds differ.)"""
    assert dose(85.0, 8, INDIA) == 0.0
    assert dose(85.0, 8, NIOSH) == pytest.approx(100.0, abs=PCT)


def test_case_10_energy_average_not_arithmetic() -> None:
    """4 h @ 80 + 4 h @ 90 dBA. L_Aeq = 10 log10((4*10^8 + 4*10^9)/8) = 10 log10(5.5e8)
    = 87.40 dBA. An arithmetic mean would give 85.0 - this case exists to catch that."""
    value = l_aeq([80.0, 90.0], [4 * H, 4 * H])
    assert value == pytest.approx(87.4, abs=DB)
    assert abs(value - 85.0) > 2.0


def test_case_11_criteria_diverge_at_92() -> None:
    """8 h @ 92 dBA. NIOSH: T = 8/2^(7/3) = 1.587 h -> 8/1.587 = 504%.
    India: T = 8/2^(2/5) = 6.063 h -> 8/6.063 = 132%. Why both are always shown."""
    assert dose(92.0, 8, NIOSH) == pytest.approx(504.0, abs=PCT)
    assert dose(92.0, 8, INDIA) == pytest.approx(132.0, abs=PCT)


def test_case_12_empty_input_raises() -> None:
    """No readings is no information, never 0% dose or 0 dB."""
    with pytest.raises(ValueError):
        l_aeq([], [])
    with pytest.raises(ValueError):
        noise_dose_pct([], [], NIOSH)


# ---------------------------------------------------------------- supporting checks

def test_criteria_come_from_config_with_their_thresholds() -> None:
    assert (NIOSH.criterion_db, NIOSH.exchange_rate_db, NIOSH.threshold_db, NIOSH.absolute_max_db) == (85, 3, 80, None)
    assert (INDIA.criterion_db, INDIA.exchange_rate_db, INDIA.threshold_db, INDIA.absolute_max_db) == (90, 5, 90, 115)
    assert "Rule 95" in INDIA.source and "98-126" in NIOSH.source


def test_l_aeq_unequal_durations() -> None:
    """1 h @ 100 + 7 h @ 80. 10 log10((1e10 + 7e8)/8) = 10 log10(1.3375e9) = 91.26 dBA."""
    assert l_aeq([100.0, 80.0], [1 * H, 7 * H]) == pytest.approx(91.26, abs=0.01)


def test_l_aeq_equal_durations_matches_mean_energy_form() -> None:
    levels = [88.0, 93.5, 79.0, 101.2]
    expected = 10 * math.log10(sum(10 ** (x / 10) for x in levels) / len(levels))
    assert l_aeq(levels, [10.0] * 4) == pytest.approx(expected, abs=1e-9)


def test_l_ex_8h_normalises_shorter_and_longer_days() -> None:
    """4 h at L_Aeq 91: 91 + 10 log10(4/8) = 87.99. 12 h at 85: 85 + 10 log10(1.5) = 86.76."""
    assert l_ex_8h(91.0, 4.0) == pytest.approx(87.99, abs=0.01)
    assert l_ex_8h(85.0, 12.0) == pytest.approx(86.76, abs=0.01)


def test_niosh_dose_agrees_with_l_ex_8h_to_within_the_exchange_rate_rounding() -> None:
    """Cross-check two routes for 4 h @ 91 dBA.

    Dose formula: 4 h / 2 h allowed = 200.0%.
    Via L_EX,8h: 91 + 10 log10(4/8) = 87.990 dBA, and 100 * 2^((87.990 - 85)/3) = 199.5%.
    They differ by 0.24% because NIOSH's 3 dB exchange rate rounds the exact
    energy-doubling step 10 log10(2) = 3.0103 dB. With that exact step the routes agree.
    """
    levels, durations = [91.0], [4 * H]
    lex = l_ex_8h(l_aeq(levels, durations), 4.0)
    via_lex = 100 * 2 ** ((lex - 85) / 3)
    assert noise_dose_pct(levels, durations, NIOSH) == pytest.approx(200.0, abs=1e-9)
    assert via_lex == pytest.approx(199.52, abs=0.01)
    assert abs(200.0 - via_lex) < PCT
    exact_step = 10 * math.log10(2)
    assert 100 * 2 ** ((lex - 85) / exact_step) * 2 ** ((91 - 85) / 3 - (91 - 85) / exact_step) == pytest.approx(
        200.0, rel=1e-9)


def test_allowed_time_table_rows() -> None:
    for level, hours in ((90, 8), (95, 4), (100, 2), (105, 1), (110, 0.5), (115, 0.25)):
        assert allowed_time_h(level, 90, 5, 8) == pytest.approx(hours)       # Schedule XXVIII, all six rows
    assert allowed_time_h(91, 85, 3, 8) == pytest.approx(2.0)


def test_threshold_is_inclusive_and_applied_per_reading() -> None:
    # Exactly at the threshold counts; just below does not.
    assert dose(90.0, 1, INDIA) == pytest.approx(12.5, abs=PCT)
    assert dose(89.9, 1, INDIA) == 0.0
    assert dose(79.9, 8, NIOSH) == 0.0
    # Mixed readings: only the loud part counts under India.
    mixed = noise_dose_pct([85.0, 95.0], [4 * H, 2 * H], INDIA)
    assert mixed == pytest.approx(50.0, abs=PCT)          # 2 h of the 4 h allowed at 95


def test_absolute_max_only_reports_levels_at_or_above_115() -> None:
    ts = ["t0", "t1", "t2", "t3"]
    events = absolute_max_exceeded([114.9, 115.0, 120.0, 96.0], ts, INDIA)
    assert [e["ts"] for e in events] == ["t1", "t2"]
    assert absolute_max_exceeded([130.0], ["t"], NIOSH) == []


@pytest.mark.parametrize("levels,durations", [([90.0, None], [10, 10]), ([90.0], [0]), ([90.0], [10, 10]),
                                              ([float("nan")], [10])])
def test_invalid_input_raises(levels: list, durations: list) -> None:
    with pytest.raises(ValueError):
        l_aeq(levels, durations)


def test_noise_module_has_no_arithmetic_mean() -> None:
    """dB must never be averaged arithmetically. Checks real code tokens only (strings
    and comments ignored): no mean/average/statistics names and no division by len()."""
    import io
    import tokenize
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "exposure" / "noise.py").read_text(encoding="utf-8")
    tokens = [t for t in tokenize.generate_tokens(io.StringIO(source).readline)
              if t.type not in (tokenize.STRING, tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE)]
    names = {t.string for t in tokens if t.type == tokenize.NAME}
    assert not names & {"mean", "fmean", "average", "statistics", "median"}, names & {"mean", "fmean", "average"}
    for a, b in zip(tokens, tokens[1:]):
        assert not (a.string == "/" and b.string == "len"), "division by len() found"
