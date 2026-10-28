"""Tests for scripts/replay_capture.py."""

import contextlib
import io
from pathlib import Path

from db.repo import Repo
from scripts.replay_capture import read_capture, replay
from simulator import publisher


def test_replaying_a_capture_twice_changes_no_counts(tmp_path: Path) -> None:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        publisher.main(["--dry-run", "--no-align-shifts", "--speed", "1e9", "--duration-h", "0.3", "--start", "2026-10-28T05:30:00",
                        "--outage-start-min", "5", "--outage-duration-min", "3"])
    capture = tmp_path / "capture.txt"
    capture.write_bytes(buf.getvalue().encode() + b"textile/zone/Z1/env {broken\n")
    messages = read_capture(capture)
    repo = Repo(tmp_path / "r.db")
    repo.init_schema()
    first = replay(repo, messages, "stream")
    assert not first["unchanged"] and first["after"]["rejected_records"] == 1
    second = replay(repo, messages, "stream")
    assert second["unchanged"]
    assert second["outcomes"].get("stored", 0) == 0 and second["outcomes"].get("error", 0) == 0
    assert second["outcomes"]["duplicate"] + second["outcomes"]["rejected_duplicate"] == len(messages)
    repo.close()
