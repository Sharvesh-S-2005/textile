"""Re-ingest a `mosquitto_sub -v` capture through the real validation + storage path.

    python scripts/replay_capture.py --capture capture.txt [--db PATH]

Used to prove idempotency: replaying every message of a run into the database it
already populated must change no row counts. Each capture line is `topic payload`
(as printed by `mosquitto_sub -v`); payloads are replayed byte-for-byte.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from db.repo import Repo  # noqa: E402
from ingest.subscriber import Ingestor, Message, TimeReference  # noqa: E402


def read_capture(path: Path) -> list[tuple[str, bytes]]:
    """(topic, payload bytes) for each line of a mosquitto_sub -v capture."""
    out = []
    with path.open("rb") as fh:
        for line in fh:
            line = line.rstrip(b"\r\n")
            if not line:
                continue
            topic, _, payload = line.partition(b" ")
            out.append((topic.decode("utf-8"), payload))
    return out


def replay(repo: Repo, messages: list[tuple[str, bytes]], time_reference: str) -> dict[str, Any]:
    """Push every message through an Ingestor; returns counts before/after and outcomes."""
    ingest_cfg = config.get_ingest_config()
    ingestor = Ingestor(repo, TimeReference(time_reference), int(ingest_cfg["batch_size"]),
                        float(ingest_cfg["batch_interval_s"]))
    before = repo.table_counts()
    for topic, payload in messages:
        ingestor.handle(Message(topic, payload, datetime.now(timezone.utc)))
        if ingestor.due():
            ingestor.flush()
    ingestor.flush()
    after = repo.table_counts()
    return {"messages": len(messages), "before": before, "after": after,
            "unchanged": before == after, "outcomes": dict(ingestor.counts)}


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns 0 if no row count changed, 1 otherwise."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--capture", type=Path, required=True)
    p.add_argument("--db", type=Path, default=config.CONFIG_PATH.parent / config.get_paths()["db_path"])
    p.add_argument("--time-reference", choices=["wall", "stream"], default="stream")
    args = p.parse_args(argv)
    with Repo(args.db) as repo:
        result = replay(repo, read_capture(args.capture), args.time_reference)
    print(json.dumps(result, indent=2))
    return 0 if result["unchanged"] else 1


if __name__ == "__main__":
    sys.exit(main())
