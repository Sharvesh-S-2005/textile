"""Create the database from db/schema.sql.

    python -m db.init_db                 # create tables if missing (keeps existing data)
    python -m db.init_db --reset         # delete the database file first, then create
    python -m db.init_db --db other.db

Safe to re-run: without --reset it never drops or alters existing data.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import config
from db.repo import Repo

log = logging.getLogger("db.init_db")


def default_db_path() -> Path:
    """`paths.db_path` from config.yaml, resolved against the project root."""
    return config.CONFIG_PATH.parent / config.get_paths()["db_path"]


def init_db(db_path: Path, reset: bool = False) -> Path:
    """Create (or, with reset, recreate) the database at `db_path`. Returns the path."""
    if reset:
        for suffix in ("", "-wal", "-shm"):
            Path(str(db_path) + suffix).unlink(missing_ok=True)
    with Repo(db_path) as repo:
        repo.init_schema()
        counts = repo.table_counts()
    log.info("database ready at %s (%d tables, %d rows)", db_path, len(counts), sum(counts.values()))
    return db_path


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(prog="python -m db.init_db", description=__doc__.split("\n")[0])
    p.add_argument("--db", type=Path, default=default_db_path())
    p.add_argument("--reset", action="store_true", help="delete the existing database first")
    args = p.parse_args(argv)
    init_db(args.db, args.reset)
    return 0


if __name__ == "__main__":
    sys.exit(main())
