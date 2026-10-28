"""Tests for scripts/score_localisation.py and the ground-truth isolation rule."""

import contextlib
import io
import json
import re
from pathlib import Path

import pytest

from scripts.score_localisation import OFF_SITE, UNKNOWN, format_report, score
from simulator import publisher

W = 30


def truth(rows: list[tuple[str, str, str | None]]) -> list[dict]:
    return [{"ts": ts, "worker_id": w, "zone_id": z} for ts, w, z in rows]


def ts(i: int) -> str:
    return f"2026-10-28T00:{(i * 30) // 60:02d}:{(i * 30) % 60:02d}Z"


def assign(worker: str, zone: str, i_from: int, i_to: int, outside: bool = False) -> dict:
    return {"worker_id": worker, "zone_id": zone, "from_ts": ts(i_from), "to_ts": ts(i_to),
            "outside_session": outside, "params": {"ewma_alpha": 0.5}}


def test_perfect_match_scores_100_percent() -> None:
    t = truth([(ts(i), "W101", "Z4") for i in range(10)] + [(ts(i), "W101", None) for i in range(10, 20)])
    a = [assign("W101", "Z4", 0, 10), assign("W101", UNKNOWN, 10, 20)]
    r = score(t, a, W)
    assert r["accuracy_on_site"] == 1.0 and r["accuracy_all"] == 1.0
    assert r["unknown"] == {"correctly_unknown": 10, "incorrectly_unknown": 0, "off_site_assigned_a_zone": 0}
    assert r["per_zone_recall"] == {"Z4": 1.0}
    assert r["latency_exit"]["mean_s"] == 0.0
    assert r["worker_minutes"] == 10.0 and r["uncovered_windows"] == 0


def test_one_deliberate_error_gives_a_known_lower_value() -> None:
    # 10 on-site windows in Z4; the algorithm says Z5 for exactly one of them.
    t = truth([(ts(i), "W101", "Z4") for i in range(10)])
    a = [assign("W101", "Z4", 0, 4), assign("W101", "Z5", 4, 5), assign("W101", "Z4", 5, 10)]
    r = score(t, a, W)
    assert r["accuracy_on_site"] == 0.9
    assert r["confusion"] == {"Z4": {"Z4": 9, "Z5": 1}}
    assert r["per_zone_recall"] == {"Z4": 0.9}


def test_unknown_split_and_off_site_assigned() -> None:
    t = truth([(ts(0), "W101", "Z4"), (ts(1), "W101", None), (ts(2), "W101", None)])
    a = [assign("W101", UNKNOWN, 0, 2), assign("W101", "Z7", 2, 3)]
    r = score(t, a, W)
    assert r["unknown"] == {"correctly_unknown": 1, "incorrectly_unknown": 1, "off_site_assigned_a_zone": 1}
    assert r["accuracy_on_site"] == 0.0 and r["accuracy_all"] == pytest.approx(1 / 3, abs=1e-4)
    assert r["confusion"][OFF_SITE] == {UNKNOWN: 1, "Z7": 1}


def test_latency_counts_windows_until_the_prediction_follows() -> None:
    # True move Z4 -> Z5 at window 5; the algorithm follows two windows later.
    t = truth([(ts(i), "W101", "Z4" if i < 5 else "Z5") for i in range(12)])
    a = [assign("W101", "Z4", 0, 7), assign("W101", "Z5", 7, 12)]
    r = score(t, a, W)
    assert r["latency_entry"] == {"n": 1, "mean_s": 60.0, "median_s": 60.0, "p90_s": 60.0}
    assert r["changes_not_followed"] == 0


def test_change_never_followed_is_counted_as_missed() -> None:
    t = truth([(ts(i), "W101", "Z4" if i < 5 or i >= 7 else "Z5") for i in range(10)])
    a = [assign("W101", "Z4", 0, 10)]                      # never noticed the 1-minute trip
    r = score(t, a, W)
    assert r["changes_not_followed"] == 1


def test_clipped_view_treats_outside_session_as_unknown() -> None:
    t = truth([(ts(0), "W101", None), (ts(1), "W101", "Z7")])
    a = [assign("W101", "Z7", 0, 1, outside=True), assign("W101", "Z7", 1, 2)]
    assert score(t, a, W)["unknown"]["off_site_assigned_a_zone"] == 1
    clipped = score(t, a, W, clip=True)
    assert clipped["unknown"]["off_site_assigned_a_zone"] == 0 and clipped["accuracy_all"] == 1.0


def test_uncovered_windows_are_reported_not_scored() -> None:
    t = truth([(ts(i), "W101", "Z4") for i in range(4)])
    r = score(t, [assign("W101", "Z4", 0, 2)], W)
    assert r["windows"] == 2 and r["uncovered_windows"] == 2


def test_report_prints_accuracy_confusion_and_parameters() -> None:
    t = truth([(ts(i), "W101", "Z4") for i in range(10)])
    text = format_report(score(t, [assign("W101", "Z4", 0, 4), assign("W101", "Z5", 4, 5),
                                   assign("W101", "Z4", 5, 10)], W))
    assert "ACCURACY" in text and "90.0%" in text and "confusion matrix" in text and "ewma_alpha=0.5" in text


# ---------------------------------------------------------------- ground truth isolation

def test_ground_truth_goes_to_a_file_never_to_mqtt(tmp_path: Path) -> None:
    out = tmp_path / "truth.jsonl"
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert publisher.main(["--dry-run", "--no-align-shifts", "--speed", "1e9", "--duration-h", "0.1", "--start",
                               "2026-10-28T05:30:00", "--ground-truth-out", str(out)]) == 0
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert rows and set(rows[0]) == {"ts", "worker_id", "tag_id", "zone_id"}
    assert len({r["worker_id"] for r in rows}) == 20
    published = buf.getvalue()
    assert "truth" not in published
    topics = {line.split(" ", 1)[0] for line in published.splitlines()}
    assert all(not t.startswith("textile/truth") for t in topics)
    # Same number of payloads as without the flag: nothing extra is published.
    buf2 = io.StringIO()
    with contextlib.redirect_stdout(buf2):
        publisher.main(["--dry-run", "--no-align-shifts", "--speed", "1e9", "--duration-h", "0.1", "--start", "2026-10-28T05:30:00"])
    assert len(buf2.getvalue().splitlines()) == len(published.splitlines())


def test_backend_code_never_reads_ground_truth() -> None:
    root = Path(__file__).resolve().parents[1]
    pattern = re.compile(r"truth", re.IGNORECASE)
    for package in ("ingest", "db", "exposure", "ml", "api", "dashboard"):
        for path in (root / package).rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            assert not pattern.search(text), f"{path.relative_to(root)} mentions ground truth"


def test_no_backend_module_imports_the_simulator() -> None:
    root = Path(__file__).resolve().parents[1]
    pattern = re.compile(r"^\s*(from|import)\s+simulator\b", re.MULTILINE)
    for package in ("ingest", "db", "exposure", "ml", "api", "dashboard", "scripts"):
        for path in (root / package).rglob("*.py"):
            assert not pattern.search(path.read_text(encoding="utf-8")), path.relative_to(root)
