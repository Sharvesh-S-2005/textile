"""Tests for config.py and the contents of config.yaml."""

from pathlib import Path

import pytest
import yaml

import config


def test_config_loads() -> None:
    cfg = config.get_config()
    assert cfg["simulation"]["schema_version"] == "1.1"


def test_entity_counts() -> None:
    assert len(config.get_zones()) == 7
    assert len(config.get_machines()) == 8
    assert len(config.get_workers()) == 20


def test_zone_ids_unique() -> None:
    ids = [z["zone_id"] for z in config.get_zones()]
    assert len(ids) == len(set(ids))


def test_machine_zone_references_exist() -> None:
    zone_ids = {z["zone_id"] for z in config.get_zones()}
    for m in config.get_machines():
        assert m["zone_id"] in zone_ids, m["machine_id"]


def test_worker_zone_and_shift_references_exist() -> None:
    zone_ids = {z["zone_id"] for z in config.get_zones()}
    shift_ids = {s["shift_id"] for s in config.get_shifts()["definitions"]}
    for w in config.get_workers():
        assert w["home_zone"] in zone_ids, w["worker_id"]
        assert w["shift"] in shift_ids, w["worker_id"]


def test_tag_and_card_ids_unique() -> None:
    workers = config.get_workers()
    tags = [w["tag_id"] for w in workers]
    cards = [w["card_id"] for w in workers]
    assert len(tags) == len(set(tags))
    assert len(cards) == len(set(cards))


def test_exactly_one_degrading_machine_is_m5_in_z5() -> None:
    degrading = [m for m in config.get_machines() if m["degrading"]]
    assert [(m["machine_id"], m["zone_id"]) for m in degrading] == [("M5", "Z5")]


def test_ppe_compliance_in_unit_interval() -> None:
    for w in config.get_workers():
        assert 0.0 <= w["ppe_compliance"] <= 1.0


def test_get_zone_unknown_raises() -> None:
    with pytest.raises(config.ConfigError):
        config.get_zone("Z99")


def test_missing_key_raises_clear_error(tmp_path: Path) -> None:
    cfg = yaml.safe_load(config.CONFIG_PATH.read_text(encoding="utf-8"))
    del cfg["mqtt"]["port"]
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(config.ConfigError, match="port"):
        config.load_config(bad)


def test_missing_section_raises(tmp_path: Path) -> None:
    cfg = yaml.safe_load(config.CONFIG_PATH.read_text(encoding="utf-8"))
    del cfg["workers"]
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(config.ConfigError, match="workers"):
        config.load_config(bad)


def test_adjacency_is_symmetric_and_covers_all_zones() -> None:
    adjacency = config.get_adjacency()
    zone_ids = {z["zone_id"] for z in config.get_zones()}
    assert set(adjacency) == zone_ids
    for zone, neighbours in adjacency.items():
        for n in neighbours:
            assert n in zone_ids
            assert zone in adjacency[n], (zone, n)


def test_high_exposure_workers_exist() -> None:
    ids = {w["worker_id"] for w in config.get_workers()}
    dyn = config.get_simulation_config()["worker_dynamics"]
    assert set(dyn["high_exposure_workers"]) <= ids
    assert dyn["break_zone"] in {z["zone_id"] for z in config.get_zones()}


def test_noise_criteria_constants() -> None:
    criteria = config.get_exposure_limits()["criteria"]
    india, niosh = criteria["india"], criteria["niosh"]
    assert (india["criterion_db"], india["exchange_rate_db"], india["duration_h"]) == (90, 5, 8)
    assert india["threshold_db"] == 90 and india["absolute_max_db"] == 115
    assert "Rule 95" in india["source"]
    assert (niosh["criterion_db"], niosh["exchange_rate_db"], niosh["threshold_db"]) == (85, 3, 80)


def test_same_second_tap_pairs_share_a_shift() -> None:
    shift = {w["worker_id"]: w["shift"] for w in config.get_workers()}
    order = [w["worker_id"] for w in config.get_workers()]
    for first, second in config.get_simulation_config()["worker_dynamics"]["same_second_tap_pairs"]:
        assert shift[first] == shift[second]
        assert order.index(first) < order.index(second)


def test_align_margins_cover_check_in_and_overtime() -> None:
    sim = config.get_simulation_config()
    dyn, margins = sim["worker_dynamics"], sim["align_shifts"]
    assert margins["margin_before_min"] >= dyn["check_in_before_min"]["max"]
    longest_overtime = max(dyn["overtime"]["duration_min"]["max"], dyn["high_exposure"]["overtime_min"]["max"])
    assert margins["margin_after_min"] >= longest_overtime + dyn["check_out_after_min"]["max"]
