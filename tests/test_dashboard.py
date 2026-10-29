"""Headless render tests for the Streamlit dashboard (streamlit.testing AppTest)."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from db.repo import Repo
from exposure import engine

ROOT = Path(__file__).resolve().parents[1]
PAGES = ["worker_drilldown.py", "live_zones.py", "system_results.py", "models.py"]


@pytest.fixture
def dashboard_db(engine_db: Path) -> Path:
    with Repo(engine_db) as repo:
        engine.run_engine(repo, ["2026-10-28"])
    return engine_db


def render(page: str, db: Path, monkeypatch: pytest.MonkeyPatch) -> AppTest:
    monkeypatch.setenv("OHS_DB", str(db))
    at = AppTest.from_file(str(ROOT / "dashboard" / "views" / page), default_timeout=60)
    at.run()
    return at


def text_of(at: AppTest) -> str:
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption] + [i.value for i in at.info]
    parts += [e.value for e in at.error] + [w.value for w in at.warning] + [s.value for s in at.subheader]
    return "\n".join(str(p) for p in parts)


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders_with_the_simulated_badge(page: str, dashboard_db: Path,
                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    at = render(page, dashboard_db, monkeypatch)
    assert not at.exception, at.exception
    assert "SIMULATED DATA" in text_of(at)
    assert "read-only" in text_of(at)


@pytest.mark.parametrize("page", PAGES)
def test_every_page_has_a_graceful_empty_state(page: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    empty_db = tmp_path / "empty.db"
    with Repo(empty_db) as r:
        r.init_schema()
    at = render(page, empty_db, monkeypatch)
    assert not at.exception, at.exception
    assert at.info or at.error                         # a plain message, never a traceback


@pytest.mark.parametrize("page", PAGES)
def test_missing_database_shows_a_message(page: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    at = render(page, tmp_path / "nope.db", monkeypatch)
    assert not at.exception and at.error


def test_drilldown_shows_the_full_value_chain(dashboard_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    at = render("worker_drilldown.py", dashboard_db, monkeypatch)
    text = text_of(at)
    labels = [m.label for m in at.metric]
    assert "L_EX,8h (ISO 9612)" in labels and "Noise dose — NIOSH | India" in labels
    assert "Data coverage" in labels and "Unaccounted time" in labels
    assert "lower bound" in text                                   # the credibility sentence
    assert "not a validated clinical model" in text                # the design-choice caveat
    assert "APPROXIMATE" in text and "INDICATIVE" in text
    assert "Rule 95" in text and "98-126" in text


def test_drilldown_shows_the_worst_component_beside_the_composite(dashboard_db: Path,
                                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    at = render("worker_drilldown.py", dashboard_db, monkeypatch)
    text = text_of(at)
    assert "Risk index" in text and "Worst single component" in text
    assert "not on commensurable scales" in text


def test_models_page_shows_the_comparison_table_and_caveats(dashboard_db: Path,
                                                             monkeypatch: pytest.MonkeyPatch) -> None:
    at = render("models.py", dashboard_db, monkeypatch)
    assert not at.exception, at.exception
    text = text_of(at)
    assert "no model runs on this page" in text
    assert "by construction" in text                               # simulator-coupling caveat
    assert "risk classifier is deferred" in text
    assert at.dataframe and len(at.dataframe[0].value) > 0
    assert "barely beats chance" in text                           # Day 7: lead with the zone that matters
    assert len(at.dataframe) >= 2                                  # comparison table + zone-group table
    assert "50× lower" in text                                     # EWMA chosen on evidence


def test_drilldown_defaults_to_the_highest_index_worker(dashboard_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with Repo(dashboard_db) as r:
        risk = r.get_exposure_rows("worker_risk_index", "2026-10-28", "2026-10-28")
    top = max(risk, key=lambda x: x["overall_index"])["worker_id"]
    at = render("worker_drilldown.py", dashboard_db, monkeypatch)
    assert at.sidebar.selectbox[-1].value == top


def test_dashboard_contains_no_sql() -> None:
    import re
    verbs = "|".join(["SEL" + "ECT", "INS" + "ERT", "UPD" + "ATE", "DEL" + "ETE"])   # split: not SQL itself
    pattern = re.compile(rf"\b({verbs})\b[^\n]*\b(FROM|INTO|SET|WHERE)\b")
    for path in (ROOT / "dashboard").rglob("*.py"):
        assert not pattern.search(path.read_text(encoding="utf-8")), path.name


def test_demo_copy_keeps_results_and_one_day_of_raw(dashboard_db: Path, tmp_path: Path,
                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    """The offline demo copy: every derived table in full, raw telemetry for one day,
    no raw BLE; the source is untouched and every page still renders from the copy."""
    with Repo(dashboard_db, read_only=True) as src:
        before = src.table_counts()
        demo = tmp_path / "demo.db"
        counts = src.export_demo_copy(demo, "2026-10-28T02:00:00Z", "2026-10-28T03:00:00Z")
        assert src.table_counts() == before                                   # source never modified
    for t in ("personal_exposure_shift", "worker_risk_index", "zone_assignments", "worker_sessions",
              "personal_exposure_cumulative", "device_status"):
        assert counts[t] == before[t], t
    assert 0 < counts["zone_env"] < before["zone_env"] and counts["ble_sightings"] == 0
    for page in PAGES:
        at = render(page, demo, monkeypatch)
        assert not at.exception, (page, at.exception)
