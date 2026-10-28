"""Tests for exposure/engine.py over a real simulated shift (shift A crew, 2026-10-28)."""

import contextlib
import io
from pathlib import Path

import pytest

from db.repo import Repo
from exposure import engine

DAY = "2026-10-28"
A_CREW = {f"W10{i}" for i in range(1, 9)}


def strip_ids(rows: list[dict]) -> list[dict]:
    return [{k: v for k, v in r.items() if k != "id"} for r in rows]


def test_engine_writes_all_three_tables(engine_db: Path) -> None:
    with Repo(engine_db) as repo:
        report = engine.run_engine(repo, [DAY])
        personal = repo.get_exposure_rows("personal_exposure_shift", DAY, DAY)
        zones = repo.get_exposure_rows("zone_exposure_summary", DAY, DAY)
        cumulative = repo.get_exposure_rows("personal_exposure_cumulative", DAY, DAY)
    assert {r["worker_id"] for r in personal} == A_CREW
    assert all(r["label"] == "estimated_personal" and r["wbgt_approximate"] is True for r in personal)
    assert all(r["engine_version"] == "exposure-1.0" and "niosh" in r["params"] for r in personal)
    assert all(r["dose_niosh_pct"] is not None and r["unaccounted_pct"] is not None for r in personal)
    # Shifts starting on the date that overlap the data: A (whole) and B (partly).
    assert {(r["shift"]) for r in zones} == {"A", "B"} and len(zones) == 2 * 7
    assert all(r["label"] == "measured_environmental" for r in zones)
    b = [r for r in zones if r["shift"] == "B"]
    assert all("low_data_coverage" in r["flags"] for r in b)          # data ends mid-shift: said, not hidden
    assert len(cumulative) == 20 * 2
    assert report["errors"] == []


def test_cumulative_rows_are_partial_and_unscaled(engine_db: Path) -> None:
    with Repo(engine_db) as repo:
        engine.run_engine(repo, [DAY])
        personal = {r["worker_id"]: r for r in repo.get_exposure_rows("personal_exposure_shift", DAY, DAY)}
        cumulative = repo.get_exposure_rows("personal_exposure_cumulative", DAY, DAY)
    for r in cumulative:
        assert r["partial"] is True
        assert r["window_coverage_pct"] == pytest.approx(100.0 / r["window_days"], abs=1e-3)   # one day of data
        if r["worker_id"] in A_CREW:
            assert r["days_worked"] == 1
            assert r["cumulative_dose_niosh_pct"] == pytest.approx(personal[r["worker_id"]]["dose_niosh_pct"])
        else:
            assert r["days_worked"] == 0 and r["cumulative_dose_niosh_pct"] is None   # not worked: null, not 0


def test_existing_rows_are_kept_unless_recompute(engine_db: Path) -> None:
    with Repo(engine_db) as repo:
        engine.run_engine(repo, [DAY])
        first = strip_ids(repo.get_exposure_rows("personal_exposure_shift", DAY, DAY))
        again = engine.run_engine(repo, [DAY])
        assert again["skipped_existing"] == [DAY]
        engine.run_engine(repo, [DAY], recompute=True)
        recomputed = strip_ids(repo.get_exposure_rows("personal_exposure_shift", DAY, DAY))
        cum = strip_ids(repo.get_exposure_rows("personal_exposure_cumulative", DAY, DAY))
        engine.run_engine(repo, [DAY], recompute=True)
        assert strip_ids(repo.get_exposure_rows("personal_exposure_cumulative", DAY, DAY)) == cum
    assert recomputed == first                                         # deterministic, no duplicates


def test_dry_run_writes_nothing(engine_db: Path) -> None:
    with Repo(engine_db) as repo:
        before = repo.table_counts()
        report = engine.run_engine(repo, [DAY], dry_run=True)
        assert repo.table_counts() == before
    assert len(report["personal"]) == 8 and report["dry_run"] is True


def test_worker_filter_touches_only_that_worker(engine_db: Path) -> None:
    with Repo(engine_db) as repo:
        engine.run_engine(repo, [DAY])
        others = strip_ids([r for r in repo.get_exposure_rows("personal_exposure_shift", DAY, DAY)
                            if r["worker_id"] != "W105"])
        report = engine.run_engine(repo, [DAY], worker_id="W105", recompute=True)
        assert {r["worker_id"] for r in report["personal"]} == {"W105"}
        assert strip_ids([r for r in repo.get_exposure_rows("personal_exposure_shift", DAY, DAY)
                          if r["worker_id"] != "W105"]) == others
        assert len(repo.get_exposure_rows("personal_exposure_shift", DAY, DAY)) == 8


def test_one_failing_worker_does_not_abort_the_batch(engine_db: Path, monkeypatch: pytest.MonkeyPatch,
                                                     caplog: pytest.LogCaptureFixture) -> None:
    real = engine.personal_exposure_for_session

    def flaky(repo, session, params):
        if session["worker_id"] == "W103":
            raise RuntimeError("corrupt session")
        return real(repo, session, params)

    monkeypatch.setattr(engine, "personal_exposure_for_session", flaky)
    with Repo(engine_db) as repo:
        report = engine.run_engine(repo, [DAY])
        written = {r["worker_id"] for r in repo.get_exposure_rows("personal_exposure_shift", DAY, DAY)}
    assert written == A_CREW - {"W103"}
    assert [w for w, _ in report["errors"]] == ["W103"]
    assert "W103: exposure failed" in caplog.text


def test_workers_without_sessions_are_logged(engine_db: Path, caplog: pytest.LogCaptureFixture) -> None:
    with Repo(engine_db) as repo:
        report = engine.run_engine(repo, [DAY])
    absent = {w for w, _ in report["no_session"]}
    assert absent == {f"W{i}" for i in range(109, 121)}
    assert "W115: no session on 2026-10-28" in caplog.text


def test_cli_prints_summary_table(engine_db: Path) -> None:
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = engine.main(["--db", str(engine_db), "--date", DAY])
    text = out.getvalue()
    assert code == 0
    assert "NIOSH %" in text and "India %" in text and "cover %" in text and "unacc %" in text
    assert "W105" in text and "8 sessions" in text


def test_cli_rejects_half_a_range(engine_db: Path) -> None:
    with pytest.raises(SystemExit):
        engine.main(["--db", str(engine_db), "--from", DAY])
