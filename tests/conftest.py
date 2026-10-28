"""Shared fixtures."""

import contextlib
import io
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pytest

from db.repo import Repo
from ingest.localisation import run_localisation
from ingest.sessions import run_sessions
from ingest.subscriber import Ingestor, Message, TimeReference
from simulator import publisher


@pytest.fixture(scope="session")
def pipeline_db(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A real aligned 8 h shift (06:00 IST, shift-A crew) through the real ingest,
    localisation and session pipelines."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert publisher.main(["--dry-run", "--speed", "1e9", "--duration-h", "8", "--start",
                               "2026-10-28T06:00:00", "--no-outage", "--seed", "3"]) == 0
    db = tmp_path_factory.mktemp("pipeline") / "p.db"
    repo = Repo(db)
    repo.init_schema()
    ingestor = Ingestor(repo, TimeReference("stream"), 500, 1.0)
    for line in buf.getvalue().splitlines():
        topic, _, body = line.partition(" ")
        ingestor.handle(Message(topic, body.encode(), datetime.now(timezone.utc)))
        if ingestor.due():
            ingestor.flush()
    ingestor.flush()
    run_localisation(repo)
    run_sessions(repo)
    repo.close()
    return db


@pytest.fixture
def engine_db(pipeline_db: Path, tmp_path: Path) -> Path:
    """A private copy of the pipeline database that a test may write to."""
    copy = tmp_path / "engine.db"
    shutil.copy(pipeline_db, copy)
    return copy
