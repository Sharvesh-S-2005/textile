"""Build the committed offline demo database from a full run.

    python scripts/make_demo_db.py [--source data/week/textile_ohs.db] [--day 2026-10-26]

Copies every derived table in full (7 days of exposure, risk, cumulative, sessions,
assignments, device status, rejections) and raw zone and machine telemetry for one
demo day only, dropping raw BLE, so the file stays small enough to commit (the full
week is 365 MB). The source database is opened read-only and never modified.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from db.repo import Repo  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--source", type=Path, default=ROOT / "data" / "week" / "textile_ohs.db")
    p.add_argument("--dest", type=Path, default=ROOT / "demo" / "textile_ohs_demo.db")
    p.add_argument("--day", default="2026-10-26", help="shift date whose raw telemetry is kept")
    args = p.parse_args(argv)
    # Keep the day's three shifts in full: C ends at 06:00 IST the next day (00:30 UTC),
    # plus the aligned run's post-shift margin.
    end = (date.fromisoformat(args.day) + timedelta(days=1)).isoformat()
    with Repo(args.source, read_only=True) as repo:
        counts = repo.export_demo_copy(args.dest, f"{args.day}T00:00:00Z", f"{end}T04:00:00Z")
    size_mb = args.dest.stat().st_size / 1e6
    print(f"wrote {args.dest} ({size_mb:.1f} MB): " + ", ".join(f"{k} {v:,}" for k, v in counts.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
