"""Tests for exposure/personal.py."""

import math
from pathlib import Path

import pytest

from db.repo import Repo, from_epoch_ms, to_epoch_ms
from exposure.noise import l_aeq, l_ex_8h
from exposure.personal import (LABEL, UNKNOWN_ZONE, ExposureParams, Interval, ResultFlag,
                               compute_session_exposure, personal_exposure_for_session)

P = ExposureParams.from_config()
H_MS = 3_600_000
DT_MS = int(P.reading_interval_s * 1000)
S0 = to_epoch_ms("2026-10-28T00:30:00Z")


def readings(start_ms: int, end_ms: int, noise: float | None = 90.0, every_other_null: bool = False,
             **over: object) -> list[dict]:
    """Synthetic zone_env rows covering [start, end): one per 10 s, ts at the END of its window."""
    rows = []
    k = 0
    for ts_ms in range(start_ms + DT_MS, end_ms + 1, DT_MS):
        row = {"ts": from_epoch_ms(ts_ms), "ts_epoch_ms": ts_ms, "noise_dba": noise, "pm25_ugm3": 200.0,
               "pm10_ugm3": 400.0, "temp_c": 30.0, "rh_pct": 60.0, "globe_temp_c": 32.0, "co2_ppm": 800}
        row.update(over)
        if every_other_null and k % 2:
            row["noise_dba"] = None
        rows.append(row)
        k += 1
    return rows


def compute(intervals: list[Interval], start: int = S0, end: int | None = None, flags: list | None = None):
    end = end if end is not None else max(iv.end_ms for iv in intervals)
    return compute_session_exposure("W999", start, end, intervals, P, flags)


# ---------------------------------------------------------------- hand-computed synthetic cases

def test_split_between_loud_and_quiet_zone_is_dominated_by_the_loud_one() -> None:
    """4 h @ 95 + 4 h @ 65. L_Aeq = 10 log10(0.5*10^9.5 + 0.5*10^6.5) = 91.99 dBA.
    NIOSH: 4 h / 0.7937 h = 504% (65 < 80 contributes 0). India: 4 h / 4 h = 100%."""
    mid, end = S0 + 4 * H_MS, S0 + 8 * H_MS
    r = compute([Interval("Z5", S0, mid, readings(S0, mid, 95.0)), Interval("Z7", mid, end, readings(mid, end, 65.0))])
    assert r.l_aeq_db == pytest.approx(91.994, abs=0.01)
    assert r.l_ex_8h_db == pytest.approx(91.994, abs=0.01)
    assert r.dose_niosh_pct == pytest.approx(504.0, abs=0.5)
    assert r.dose_india_pct == pytest.approx(100.0, abs=0.5)
    assert r.zones_visited == {"Z5": 4 * 3600.0, "Z7": 4 * 3600.0}
    assert r.unaccounted_time_s == 0.0 and r.data_coverage_pct == pytest.approx(100.0)


def test_unaccounted_time_is_reported_not_absorbed() -> None:
    """8 h session: 6 h in Z5 @ 95 dBA, 2 h UNKNOWN. Unaccounted 2 h = 25%. Figures cover
    the 6 h: L_EX,8h = 95 + 10 log10(6/8) = 93.75; NIOSH 6 / 0.7937 = 756%."""
    six, end = S0 + 6 * H_MS, S0 + 8 * H_MS
    r = compute([Interval("Z5", S0, six, readings(S0, six, 95.0)), Interval(UNKNOWN_ZONE, six, end)])
    assert r.unaccounted_time_s == 2 * 3600.0 and r.unaccounted_pct == pytest.approx(25.0)
    assert ResultFlag.INCOMPLETE_PRESENCE in r.flags
    assert r.l_ex_8h_db == pytest.approx(93.75, abs=0.01)
    assert r.dose_niosh_pct == pytest.approx(756.0, abs=0.5)
    assert r.accounted_time_s == 6 * 3600.0
    assert UNKNOWN_ZONE not in r.zones_visited


def test_session_time_with_no_assignment_at_all_is_unaccounted() -> None:
    end = S0 + 8 * H_MS
    r = compute([Interval("Z5", S0, S0 + 4 * H_MS, readings(S0, S0 + 4 * H_MS, 95.0))], end=end)
    assert r.unaccounted_pct == pytest.approx(50.0)


def test_null_readings_reduce_coverage_not_dose() -> None:
    """Every other reading NULL in a constant 95 dBA zone: dose and L_Aeq unchanged,
    coverage halves and the result is flagged."""
    end = S0 + 4 * H_MS
    full = compute([Interval("Z5", S0, end, readings(S0, end, 95.0))])
    half = compute([Interval("Z5", S0, end, readings(S0, end, 95.0, every_other_null=True))])
    assert half.dose_niosh_pct == pytest.approx(full.dose_niosh_pct, rel=1e-9)
    assert half.dose_india_pct == pytest.approx(full.dose_india_pct, rel=1e-9)
    assert half.l_aeq_db == pytest.approx(full.l_aeq_db, abs=1e-9)
    assert half.data_coverage_pct == pytest.approx(50.0, abs=0.1)
    assert ResultFlag.LOW_DATA_COVERAGE in half.flags and ResultFlag.LOW_DATA_COVERAGE not in full.flags


def test_session_with_zero_valid_readings_is_null_not_zero() -> None:
    end = S0 + 2 * H_MS
    r = compute([Interval("Z5", S0, end, readings(S0, end, None))])
    assert r.l_aeq_db is None and r.l_ex_8h_db is None
    assert r.dose_niosh_pct is None and r.dose_india_pct is None
    assert ResultFlag.NO_VALID_READINGS in r.flags
    r2 = compute([Interval("Z5", S0, end, [])])
    assert r2.dose_niosh_pct is None and ResultFlag.NO_VALID_READINGS in r2.flags


def test_reading_windows_are_weighted_by_overlap_with_the_interval() -> None:
    """A 15 s interval [S0, S0+15 s) with readings at S0+10 s and S0+20 s: the second
    reading's window [S0+10, S0+20) overlaps only 5 s. Energy-weighted:
    10 log10((10*10^9 + 5*10^8)/15) = 88.45 dBA."""
    end = S0 + 15_000
    rows = [{**readings(S0, S0 + 10_000, 90.0)[0]},
            {**readings(S0 + 10_000, S0 + 20_000, 80.0)[0]}]
    r = compute([Interval("Z5", S0, end, rows)], end=end)
    assert r.l_aeq_db == pytest.approx(10 * math.log10((10 * 1e9 + 5 * 1e8) / 15), abs=1e-9)


def test_absolute_max_events_are_reported_with_zone() -> None:
    end = S0 + 3 * DT_MS
    rows = readings(S0, end, 96.0)
    rows[1]["noise_dba"] = 116.0
    r = compute([Interval("Z5", S0, end, rows)], end=end)
    assert [(e["level_db"], e["zone_id"]) for e in r.absolute_max_events] == [(116.0, "Z5")]
    assert ResultFlag.ABSOLUTE_MAX_EXCEEDED in r.flags


def test_dust_raw_and_corrected_are_both_kept() -> None:
    """RH 85 > 75: corrected = 200 / (1 + 0.05 * 0.85/0.15) = 155.84; raw stays 200."""
    end = S0 + H_MS
    r = compute([Interval("Z2", S0, end, readings(S0, end, 90.0, rh_pct=85.0))])
    assert r.pm25_twa_raw == pytest.approx(200.0)
    assert r.pm25_twa_corrected == pytest.approx(200.0 / (1 + 0.05 * 0.85 / 0.15), abs=0.01)
    assert r.dust_label == "indicative"


def test_wbgt_flags_and_time_above_limit() -> None:
    """T 30, RH 60, globe 32: Tw (Stull) ~ 23.7 -> WBGT ~ 26.2 < 28 for 1 h; then
    T 34, RH 75, globe 38: WBGT ~ 0.7*30.2 + 0.3*38 = 32.5 > 28 for 1 h -> 3600 s above."""
    one, two = S0 + H_MS, S0 + 2 * H_MS
    r = compute([Interval("Z4", S0, one, readings(S0, one, 90.0)),
                 Interval("Z5", one, two, readings(one, two, 90.0, temp_c=34.0, rh_pct=75.0, globe_temp_c=38.0))])
    assert r.wbgt_time_above_limit_s == pytest.approx(3600.0)
    assert 32.0 < r.wbgt_max_c < 33.0
    assert r.wbgt_approximate is True and r.wbgt_t_nw_method == "stull_psychrometric_approximation"
    assert r.wbgt_metabolic_category == "moderate"


def test_every_result_is_labelled_estimated_personal() -> None:
    end = S0 + H_MS
    r = compute([Interval("Z5", S0, end, readings(S0, end))])
    d = r.as_dict()
    assert d["label"] == LABEL == "estimated_personal" and d["measured_personal"] is None
    assert "Rule 95" in d["criteria_sources"]["india"]


def test_inferred_session_boundary_is_carried_as_a_flag() -> None:
    end = S0 + H_MS
    r = compute([Interval("Z5", S0, end, readings(S0, end))], flags=["unmatched_check_in"])
    assert ResultFlag.SESSION_BOUNDARY_INFERRED in r.flags


# ---------------------------------------------------------------- the internal consistency test


def test_single_zone_personal_l_ex_equals_zone_l_aeq_normalised(pipeline_db: Path) -> None:
    """THE cross-check of three pipelines. Take a worker's longest single-zone stretch
    inside a real session (zone pipeline + assignment pipeline output) and treat it as
    the exposure period. The personal L_EX,8h must equal the zone's own L_Aeq over that
    period, computed straight from zone_env, normalised to the period's duration."""
    with Repo(pipeline_db) as repo:
        candidates = [a for a in repo.get_all_zone_assignments()
                      if a["zone_id"] != UNKNOWN_ZONE and not a["outside_session"]]
        best = max(candidates, key=lambda a: a["to_epoch_ms"] - a["from_epoch_ms"])
        assert best["to_epoch_ms"] - best["from_epoch_ms"] >= H_MS     # a substantial stretch
        session = {"worker_id": best["worker_id"], "session_start": best["from_ts"], "session_end": best["to_ts"],
                   "session_start_ms": best["from_epoch_ms"], "session_end_ms": best["to_epoch_ms"], "flags": []}
        personal = personal_exposure_for_session(repo, session, P)

        # Independent route: the zone's readings whose 10 s windows lie in the stretch.
        rows = repo.get_zone_env(best["zone_id"], from_epoch_ms(best["from_epoch_ms"] + 1),
                                 from_epoch_ms(best["to_epoch_ms"] + 1))
        # Same definition of a valid reading as the engine: non-null and not a detected
        # anomaly (spike / stuck_value records are excluded on both routes).
        excluded = set(P.exclude_record_flags)
        levels = [r["noise_dba"] for r in rows if r["noise_dba"] is not None and not excluded & set(r["flags"])]
        zone_l_aeq = l_aeq(levels, [P.reading_interval_s] * len(levels))
        hours = (best["to_epoch_ms"] - best["from_epoch_ms"]) / H_MS
        expected = l_ex_8h(zone_l_aeq, hours)

    assert personal.zones_visited == {best["zone_id"]: hours * 3600}
    assert personal.unaccounted_time_s == 0.0
    assert personal.l_ex_8h_db == pytest.approx(expected, abs=1e-9)
    assert personal.l_aeq_db == pytest.approx(zone_l_aeq, abs=1e-9)


def test_real_session_results_carry_accounting(pipeline_db: Path) -> None:
    with Repo(pipeline_db) as repo:
        sessions = repo.get_all_worker_sessions()
        assert sessions
        for s in sessions:
            r = personal_exposure_for_session(repo, s, P)
            assert r.label == LABEL
            assert r.unaccounted_pct is not None and r.data_coverage_pct is not None
            assert r.dose_niosh_pct is not None and r.dose_niosh_pct >= r.dose_india_pct
            assert 0.0 <= r.unaccounted_pct <= 100.0


def test_anomaly_flagged_readings_are_excluded_but_loud_ones_still_reported() -> None:
    """A spike-flagged 135 dBA glitch (Day 3 finding) must not enter the dose, but must
    stay visible: excluded count, lower coverage, and listed as an excluded >=115 candidate."""
    end = S0 + 10 * DT_MS
    clean = readings(S0, end, 90.0)
    glitched = [dict(r) for r in clean]
    glitched[4].update(noise_dba=135.0, flags=["spike"])
    glitched[5].update(flags=["spike"])                   # the jump back down is flagged too
    base = compute([Interval("Z5", S0, end, clean)], end=end)
    r = compute([Interval("Z5", S0, end, glitched)], end=end)
    assert r.dose_niosh_pct == pytest.approx(base.dose_niosh_pct)         # not inflated
    assert r.absolute_max_events == []
    assert r.excluded_anomalous_readings == 2
    assert [c["level_db"] for c in r.excluded_absolute_max_candidates] == [135.0]
    assert r.data_coverage_pct == pytest.approx(80.0)
    assert {ResultFlag.ANOMALOUS_READINGS_EXCLUDED, ResultFlag.EXCLUDED_ABSOLUTE_MAX_CANDIDATE} <= set(r.flags)
    # Not excluded: sensor_not_ok (per-field nulls handle it) and late_upload (valid data).
    kept = [dict(x, flags=["late_upload"]) for x in clean]
    assert compute([Interval("Z5", S0, end, kept)], end=end).excluded_anomalous_readings == 0
