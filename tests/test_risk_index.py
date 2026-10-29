"""Tests for exposure/risk_index.py - hand-computed cases and the monotonicity property."""

import copy
from pathlib import Path

import numpy as np
import pytest

import config
from db.repo import Repo
from exposure import engine
from exposure.risk_index import (LABEL, RiskParams, compute_day, compute_index, heat_component, hearing_component,
                                 percentile_rank, worst_component)

P = RiskParams.from_config(config.get_risk_index_config(), config.get_exposure_limits()["wbgt_limit_c"])
METRICS = P.respiratory_metrics


def worker(**over: object) -> dict:
    w = {"worker_id": "W1", "date": "2026-10-26", "l_ex_8h_db": 85.0, "mean_daily_l_ex_7d_db": 85.0,
         "dose_niosh_pct": 100.0, "dose_india_pct": 0.0, "wbgt_max_c": 25.0, "wbgt_hours_above": 0.0,
         "pm25_twa_raw": 10.0, "pm10_twa_raw": 20.0, "pm25_twa_hours_raw_7d": 80.0,
         "data_coverage_pct": 99.0, "unaccounted_pct": 0.5, "ppe_status": {"hearing_ppe": "worn"}}
    w.update(over)
    return w


COHORT = {"pm25_twa_raw": [10.0, 50.0, 100.0, 200.0], "pm10_twa_raw": [20.0, 90.0, 180.0, 400.0],
          "pm25_twa_hours_raw_7d": [80.0, 400.0, 800.0, 1600.0]}


def test_contributions_sum_to_100_percent_of_the_overall_index() -> None:
    r = compute_index(worker(wbgt_max_c=31.0, wbgt_hours_above=2.0, pm25_twa_raw=100.0), COHORT, P)
    c = r["contributions"]
    assert sum(x["percent"] for x in c.values()) == pytest.approx(100.0)
    assert sum(x["absolute"] for x in c.values()) == pytest.approx(r["overall_index"])


def test_exactly_100_percent_dose_with_nothing_else_elevated() -> None:
    """100% NIOSH dose over 8 h is L_EX,8h = 85 dBA (today and 7-day mean).
    Hearing: (85 - 75) / (105 - 75) x 100 = 33.33. Heat: WBGT 25 C is below the ramp start
    (28 - 2 = 26 C), 0 h above -> 0. Respiratory: lowest of the cohort -> 0.
    Overall = 33.33 / 3 = 11.11 -> Low (< 33). The dose is reported beside it."""
    r = compute_index(worker(), COHORT, P)
    assert r["components"] == {"hearing": pytest.approx(100 / 3), "heat": 0.0, "respiratory": 0.0}
    assert r["overall_index"] == pytest.approx(100 / 9)
    assert r["band"] == "Low"
    assert r["contributions"]["hearing"]["percent"] == pytest.approx(100.0)
    assert r["drivers"] == ["noise L_EX,8h 85.0 dBA (dose 100% NIOSH, 0% India)"]
    assert (r["dose_niosh_pct"], r["dose_india_pct"], r["l_ex_8h_db"]) == (100.0, 0.0, 85.0)


def test_hand_values_for_heat_and_ranks() -> None:
    """Heat ramp 26 -> 34 C: WBGT 30 -> 50; time ramp 0 -> 4 h: 2 h -> 50; heat = 50.
    Rank of 100 in [10, 50, 100, 200]: 2 values below / (4 - 1) = 66.7."""
    assert heat_component(30.0, 2.0, P) == pytest.approx(50.0)
    assert heat_component(40.0, 9.0, P) == pytest.approx(100.0)
    assert percentile_rank(100.0, COHORT["pm25_twa_raw"]) == pytest.approx(200 / 3)
    assert percentile_rank(10.0, COHORT["pm25_twa_raw"]) == 0.0
    assert percentile_rank(200.0, COHORT["pm25_twa_raw"]) == 100.0
    assert percentile_rank(5.0, [5.0]) == 0.0                     # nobody to rank against
    assert hearing_component(90.0, 90.0, P) == pytest.approx(50.0)   # (90 - 75) / 30
    assert hearing_component(105.0, 110.0, P) == 100.0              # clamped at the ceiling
    assert hearing_component(70.0, 60.0, P) == 0.0                  # clamped at the floor
    assert hearing_component(95.0, 85.0, P) == pytest.approx((200 / 3 + 100 / 3) / 2)


def test_changing_a_weight_moves_the_output_in_the_expected_direction() -> None:
    """Hearing-only worker, hearing 33.33. Doubling the hearing weight: 2*33.33/4 = 16.67 > 11.11.
    Doubling the heat weight instead lowers it: 33.33/4 = 8.33."""
    base = compute_index(worker(), COHORT, P)["overall_index"]
    heavy_hearing = copy.deepcopy(P.weights) | {"hearing": 2.0}
    heavy_heat = copy.deepcopy(P.weights) | {"heat": 2.0}
    up = compute_index(worker(), COHORT, RiskParams(**{**P.__dict__, "weights": heavy_hearing}))["overall_index"]
    down = compute_index(worker(), COHORT, RiskParams(**{**P.__dict__, "weights": heavy_heat}))["overall_index"]
    assert up == pytest.approx(100 / 6) and up > base
    assert down == pytest.approx(100 / 12) and down < base


def test_low_confidence_suppresses_the_band_not_the_number() -> None:
    confident = compute_index(worker(), COHORT, P)
    weak = compute_index(worker(data_coverage_pct=45.0), COHORT, P)
    assert weak["band"] == "insufficient data" and weak["insufficient_data"] is True
    assert weak["overall_index"] == pytest.approx(confident["overall_index"])   # number still shown
    assert weak["confidence_pct"] == pytest.approx(45.0 * 0.995)
    mostly_unknown = compute_index(worker(unaccounted_pct=30.0), COHORT, P)       # 99 * 0.7 = 69.3 < 80
    assert mostly_unknown["band"] == "insufficient data"


@pytest.mark.parametrize("field,low,high", [
    ("l_ex_8h_db", 70.0, 110.0), ("mean_daily_l_ex_7d_db", 70.0, 110.0), ("wbgt_max_c", 27.0, 31.0),
    ("wbgt_hours_above", 0.5, 3.0), ("pm25_twa_raw", 10.0, 150.0), ("pm10_twa_raw", 20.0, 300.0),
    ("pm25_twa_hours_raw_7d", 80.0, 1000.0)])
def test_index_is_monotonic_in_each_input_with_the_others_fixed(field: str, low: float, high: float) -> None:
    """An index that can fall when an exposure rises is indefensible. For random
    baselines, raising one input (others fixed, cohort fixed) never lowers any output."""
    rng = np.random.default_rng(hash(field) % 2**32)
    for _ in range(200):
        base = worker(l_ex_8h_db=rng.uniform(70, 110), mean_daily_l_ex_7d_db=rng.uniform(70, 110),
                      wbgt_max_c=rng.uniform(20, 40), wbgt_hours_above=rng.uniform(0, 8),
                      pm25_twa_raw=rng.uniform(0, 300), pm10_twa_raw=rng.uniform(0, 600),
                      pm25_twa_hours_raw_7d=rng.uniform(0, 2000))
        a, b = sorted(rng.uniform(low, high, size=2))
        lo = compute_index({**base, field: a}, COHORT, P)
        hi = compute_index({**base, field: b}, COHORT, P)
        assert hi["overall_index"] >= lo["overall_index"] - 1e-12, (field, a, b)
        for c in lo["components"]:
            assert hi["components"][c] >= lo["components"][c] - 1e-12, (field, c)


def test_ppe_status_never_changes_the_index() -> None:
    worn = compute_index(worker(ppe_status={"hearing_ppe": "worn", "respiratory_ppe": "worn"}), COHORT, P)
    none = compute_index(worker(ppe_status={"hearing_ppe": "not_worn", "respiratory_ppe": "not_worn"}), COHORT, P)
    assert worn["overall_index"] == none["overall_index"] and worn["components"] == none["components"]
    assert none["ppe_status"]["hearing_ppe"] == "not_worn"           # carried for display


def test_label_weights_and_caveat_are_stored() -> None:
    r = compute_index(worker(), COHORT, P)
    assert r["label"] == LABEL == "deterministic_index"
    assert "design choice" in r["params"]["caveat"] and "cohort" in r["params"]["caveat"]
    assert r["weights"] == P.weights and r["version"] == P.version


def test_compute_day_ranks_within_the_day_cohort() -> None:
    rows = compute_day([worker(worker_id="W1", pm25_twa_raw=10.0, pm10_twa_raw=10.0, pm25_twa_hours_raw_7d=10.0),
                        worker(worker_id="W2", pm25_twa_raw=90.0, pm10_twa_raw=90.0, pm25_twa_hours_raw_7d=90.0)], P)
    assert [r["components"]["respiratory"] for r in rows] == [0.0, 100.0]


def test_engine_writes_risk_rows(engine_db: Path) -> None:
    with Repo(engine_db) as repo:
        engine.run_engine(repo, ["2026-10-28"])
        rows = repo.get_exposure_rows("worker_risk_index", "2026-10-28", "2026-10-28")
    assert len(rows) == 8 and all(r["label"] == "deterministic_index" for r in rows)
    for r in rows:
        assert sum(c["percent"] for c in r["contributions"].values()) == pytest.approx(100.0)
        assert r["band"] in ("Low", "Medium", "High", "insufficient data")
        assert "never reduces" in r["ppe_status"].get("note", "never reduces")



def test_regression_higher_l_ex_never_scores_below_lower_l_ex() -> None:
    """Regression for the risk-1.0 bug: dose % capped at 100% could not separate 235% from
    3,378%, so a far louder worker could rank lower. With other components equal, a
    materially higher L_EX,8h must never score below a lower one, including well above
    the 100%-dose point (85 dBA) where 1.0 saturated."""
    for low, high in ((88.8, 100.4), (85.0, 86.0), (90.0, 104.0), (95.0, 99.0)):
        quieter = compute_index(worker(l_ex_8h_db=low, mean_daily_l_ex_7d_db=low), COHORT, P)
        louder = compute_index(worker(l_ex_8h_db=high, mean_daily_l_ex_7d_db=high), COHORT, P)
        assert louder["components"]["hearing"] > quieter["components"]["hearing"], (low, high)
        assert louder["overall_index"] > quieter["overall_index"], (low, high)


def test_worst_component_is_reported_beside_the_composite() -> None:
    """Hearing 90, heat 0, respiratory 0: the equal-weight composite is 30 (Low), but the
    worst single component is hearing at 90 (High). Both must be available."""
    comps = {"hearing": 90.0, "heat": 0.0, "respiratory": 0.0}
    w = worst_component(comps, P)
    assert w == {"component": "hearing", "value": 90.0, "band": "High"}
    from exposure.risk_index import combine
    overall, _ = combine(comps, P)
    assert overall == pytest.approx(30.0) and overall < w["value"]


def test_worst_component_never_below_the_composite() -> None:
    rng = np.random.default_rng(0)
    from exposure.risk_index import combine
    for _ in range(200):
        comps = dict(zip(("hearing", "heat", "respiratory"), rng.uniform(0, 100, 3)))
        assert worst_component(comps, P)["value"] >= combine(comps, P)[0] - 1e-9
