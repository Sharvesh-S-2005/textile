"""Tests for scripts/build_results_pack.py helpers."""

from pathlib import Path

import pytest

from scripts.build_results_pack import energy_mean, parse_run_logs


def test_energy_mean_is_not_arithmetic() -> None:
    assert energy_mean([80.0, 90.0]) == pytest.approx(87.40, abs=0.01)


def test_parse_run_logs_reconciles_published_and_received(tmp_path: Path) -> None:
    pub, ing = tmp_path / "p.log", tmp_path / "i.log"
    pub.write_text("x sim=2026-10-28T08:00:00Z wall=100.0s env=10 ble=5 machine=8 rfid=1 ppe=1 status=4 "
                   "buffered=2 replayed=2\n", encoding="utf-8")
    ing.write_text("x received=31 stored=30 rejected=1 rate=12.5 msg/s\n"
                   "x final: received=33 stored=31 duplicate=0 rejected=2 errors=0\n", encoding="utf-8")
    r = parse_run_logs(pub, ing)
    assert r["published"] == 10 + 5 + 8 + 1 + 1 + 4 + 2           # buffered is not counted twice
    assert (r["received"], r["stored"], r["rejected"]) == (33, 31, 2)
    assert r["throughput_msg_s"] == pytest.approx(0.33) and r["peak_rate_msg_s"] == 12.5
    assert parse_run_logs(tmp_path / "missing.log", ing) is None
