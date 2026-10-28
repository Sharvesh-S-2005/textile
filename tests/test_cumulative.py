"""Tests for exposure/cumulative.py and idempotent exposure-table recomputation."""

import copy
import math
from datetime import date
from pathlib import Path

import pytest

from db.repo import Repo
from exposure.cumulative import daily_exposures, energy_mean_db, rolling_window, window_coverage_pct

D0 = date(2026, 10, 20)
# A worker's 7 consecutive shift days (one session each) - hand-chosen values.
DOSES_NIOSH = [100.0, 200.0, 50.0, 0.0, 150.0, 100.0, 300.0]
DOSES_INDIA = [20.0, 120.0, 5.0, 0.0, 90.0, 20.0, 150.0]
LEX = [85.0, 88.0, 82.0, 70.0, 86.76, 85.0, 89.77]


def shift_row(i: int, worker: str = "W101", **over: object) -> dict:
    d = date.fromordinal(D0.toordinal() + i).isoformat()
    row = {"worker_id": worker, "shift_date": d, "dose_niosh_pct": DOSES_NIOSH[i], "dose_india_pct": DOSES_INDIA[i],
           "l_ex_8h_db": LEX[i], "pm25_twa_raw": 200.0, "pm25_twa_corrected": 180.0, "pm10_twa_raw": 400.0,
           "pm10_twa_corrected": 360.0, "wbgt_time_above_limit_s": 1800.0, "accounted_time_s": 8 * 3600.0,
           "session_duration_s": 8 * 3600.0, "data_coverage_pct": 99.0, "unaccounted_pct": 2.0}
    row.update(over)
    return row


WEEK = [shift_row(i) for i in range(7)]
FIRST, LAST = D0, date.fromordinal(D0.toordinal() + 6)


def test_seven_day_window_hand_totals() -> None:
    """Sum of NIOSH doses 100+200+50+0+150+100+300 = 900%; India 405%.
    Days over 100%: NIOSH 3 (200, 150, 300); India 2 (120, 150).
    Dust: 200 ug/m3 x 8 h x 7 days = 11200 ug.h/m3. WBGT: 0.5 h x 7 = 3.5 h."""
    r = rolling_window(daily_exposures(WEEK), "W101", LAST, 7, FIRST, LAST)
    assert r["cumulative_dose_niosh_pct"] == pytest.approx(900.0)
    assert r["cumulative_dose_india_pct"] == pytest.approx(405.0)
    assert (r["days_dose_niosh_over_100"], r["days_dose_india_over_100"]) == (3, 2)
    assert r["pm25_twa_hours_raw"] == pytest.approx(11200.0)
    assert r["pm25_twa_hours_corrected"] == pytest.approx(180.0 * 8 * 7)
    assert r["wbgt_hours_above_limit"] == pytest.approx(3.5)
    assert r["days_worked"] == 7 and r["window_coverage_pct"] == 100.0 and r["partial"] is False
    assert r["max_daily_l_ex_8h_db"] == pytest.approx(89.77)


def test_mean_daily_l_ex_is_an_energy_mean() -> None:
    """Energy mean of the 7 daily L_EX,8h values, hand-computed independently."""
    expected = 10 * math.log10(sum(10 ** (x / 10) for x in LEX) / 7)
    r = rolling_window(daily_exposures(WEEK), "W101", LAST, 7, FIRST, LAST)
    assert r["mean_daily_l_ex_8h_db"] == pytest.approx(expected, abs=1e-9)
    assert r["mean_daily_l_ex_8h_db"] > sum(LEX) / 7 + 1.0    # an arithmetic mean would be ~1.8 dB low


def test_energy_mean_hand_value() -> None:
    """85 and 95: 10 log10((10^8.5 + 10^9.5)/2) = 92.40 dBA (arithmetic mean 90)."""
    assert energy_mean_db([85.0, 95.0]) == pytest.approx(92.40, abs=0.01)


def test_thirty_day_window_over_seven_days_is_partial_and_not_scaled() -> None:
    r30 = rolling_window(daily_exposures(WEEK), "W101", LAST, 30, FIRST, LAST)
    r7 = rolling_window(daily_exposures(WEEK), "W101", LAST, 7, FIRST, LAST)
    assert r30["partial"] is True
    assert r30["window_coverage_pct"] == pytest.approx(100 * 7 / 30, abs=1e-3)   # 23.333% (stored to 3 dp)
    assert r30["cumulative_dose_niosh_pct"] == r7["cumulative_dose_niosh_pct"] == pytest.approx(900.0)
    assert r30["cumulative_dose_niosh_pct"] != pytest.approx(900.0 * 30 / 7)  # never scaled up


def test_window_ending_mid_data_counts_only_its_days() -> None:
    """3-day window ending on day index 3: doses 200 + 50 + 0 = 250%."""
    r = rolling_window(daily_exposures(WEEK), "W101", date.fromordinal(D0.toordinal() + 3), 3, FIRST, LAST)
    assert r["cumulative_dose_niosh_pct"] == pytest.approx(250.0) and r["days_worked"] == 3


def test_window_coverage_hand_values() -> None:
    assert window_coverage_pct(LAST, 7, FIRST, LAST) == 100.0
    assert window_coverage_pct(FIRST, 7, FIRST, LAST) == pytest.approx(100 / 7)      # only day 1 is in the data
    assert window_coverage_pct(date(2026, 11, 30), 7, FIRST, LAST) == 0.0


def test_days_without_data_are_counted_not_zeroed() -> None:
    rows = copy.deepcopy(WEEK)
    rows[1].update(dose_niosh_pct=None, dose_india_pct=None, l_ex_8h_db=None)
    r = rolling_window(daily_exposures(rows), "W101", LAST, 7, FIRST, LAST)
    assert r["days_worked"] == 6 and r["days_without_data"] == 1
    assert r["cumulative_dose_niosh_pct"] == pytest.approx(700.0)


def test_two_sessions_on_one_date_combine_on_energy_basis() -> None:
    """Two 4-h sessions with L_EX,8h 85 and 85: daily L_EX = 85 + 10 log10(2) = 88.01; doses add."""
    rows = [shift_row(0, l_ex_8h_db=85.0, dose_niosh_pct=50.0), shift_row(0, l_ex_8h_db=85.0, dose_niosh_pct=50.0)]
    (day,) = daily_exposures(rows)
    assert day.l_ex_8h_db == pytest.approx(88.01, abs=0.01) and day.dose_niosh_pct == pytest.approx(100.0)


def test_other_workers_do_not_leak_in() -> None:
    rows = WEEK + [shift_row(i, worker="W102", dose_niosh_pct=999.0) for i in range(7)]
    r = rolling_window(daily_exposures(rows), "W101", LAST, 7, FIRST, LAST)
    assert r["cumulative_dose_niosh_pct"] == pytest.approx(900.0)


# ---------------------------------------------------------------- idempotent recomputation

def _cum_row(day: str, worker: str, dose: float) -> dict:
    return {"worker_id": worker, "as_of_date": day, "window_days": 7, "window_start_date": "2026-10-14",
            "window_coverage_pct": 100.0, "partial": False, "days_worked": 7, "days_without_data": 0,
            "cumulative_dose_niosh_pct": dose, "cumulative_dose_india_pct": dose / 2,
            "mean_daily_l_ex_8h_db": 88.0, "max_daily_l_ex_8h_db": 90.0, "pm25_twa_hours_raw": 1.0,
            "pm25_twa_hours_corrected": 1.0, "pm10_twa_hours_raw": 1.0, "pm10_twa_hours_corrected": 1.0,
            "wbgt_hours_above_limit": 0.5, "days_dose_niosh_over_100": 1, "days_dose_india_over_100": 0,
            "mean_data_coverage_pct": 99.0, "mean_unaccounted_pct": 1.0, "label": "estimated_personal",
            "engine_version": "exposure-1.0", "params": {"kappa": 0.05}}


def test_recomputing_one_date_leaves_adjacent_dates_byte_identical(tmp_path: Path) -> None:
    repo = Repo(tmp_path / "c.db")
    repo.init_schema()
    days = ["2026-10-20", "2026-10-21", "2026-10-22"]
    for i, d in enumerate(days):
        repo.replace_exposure_rows("personal_exposure_cumulative", d,
                                   [_cum_row(d, "W101", 100.0 + i), _cum_row(d, "W102", 200.0 + i)])
    repo.commit()
    before = {d: repo.get_exposure_rows("personal_exposure_cumulative", d, d) for d in days}

    # Recompute the middle date with different values, twice (idempotent).
    for _ in range(2):
        repo.replace_exposure_rows("personal_exposure_cumulative", "2026-10-21",
                                   [_cum_row("2026-10-21", "W101", 555.0), _cum_row("2026-10-21", "W102", 666.0)])
        repo.commit()
    after = {d: repo.get_exposure_rows("personal_exposure_cumulative", d, d) for d in days}

    strip = lambda rows: [{k: v for k, v in r.items() if k != "id"} for r in rows]  # noqa: E731
    assert after["2026-10-20"] == before["2026-10-20"]          # untouched, ids included
    assert after["2026-10-22"] == before["2026-10-22"]
    assert [r["cumulative_dose_niosh_pct"] for r in after["2026-10-21"]] == [555.0, 666.0]
    assert len(after["2026-10-21"]) == 2                        # replaced, not appended
    assert strip(after["2026-10-21"])[0]["params"] == {"kappa": 0.05}
    repo.close()


def test_replace_refuses_rows_for_another_date(tmp_path: Path) -> None:
    repo = Repo(tmp_path / "c.db")
    repo.init_schema()
    with pytest.raises(ValueError):
        repo.replace_exposure_rows("personal_exposure_cumulative", "2026-10-20", [_cum_row("2026-10-21", "W1", 1.0)])
    repo.close()


def test_replace_for_one_worker_leaves_other_workers(tmp_path: Path) -> None:
    repo = Repo(tmp_path / "c.db")
    repo.init_schema()
    d = "2026-10-20"
    repo.replace_exposure_rows("personal_exposure_cumulative", d, [_cum_row(d, "W101", 1.0), _cum_row(d, "W102", 2.0)])
    repo.replace_exposure_rows("personal_exposure_cumulative", d, [_cum_row(d, "W101", 9.0)], worker_id="W101")
    rows = repo.get_exposure_rows("personal_exposure_cumulative", d, d)
    assert {r["worker_id"]: r["cumulative_dose_niosh_pct"] for r in rows} == {"W101": 9.0, "W102": 2.0}
    repo.close()
