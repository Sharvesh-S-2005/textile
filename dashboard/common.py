"""Shared dashboard setup: database path, cached reads, header band, sidebar, empty states.

All data comes through db/repo.py on a READ-ONLY connection (no SQL in dashboard/).
Every read is cached for `dashboard.cache_ttl_s` seconds, so widget interactions do
not re-query, while a live ingest run still shows up within seconds.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

import config  # noqa: E402
from db.repo import Repo  # noqa: E402

DASH = config.get_dashboard_config()
TTL = int(DASH["cache_ttl_s"])
TZ = config.get_shifts()["timezone"]
BAND_COLOURS: dict[str, str] = DASH["band_colours"]
BAND_ORDER = ("Low", "Medium", "High", "insufficient data")
# One colour per hazard, identical to the results-pack figures (figstyle.HAZARD_COLOURS).
HAZARD = {"noise": "#0072B2", "dust": "#E69F00", "heat": "#D55E00", "co2": "#009E73"}
NEUTRAL, LIGHT = "#4D4D4D", "#BDBDBD"
INDEX_CAVEAT = "This index is a weighted design choice of this project, not a validated clinical model."


def db_path() -> str:
    """Database to show: `-- --db PATH` on the streamlit command line, else OHS_DB, else config."""
    argv = sys.argv
    if "--db" in argv and argv.index("--db") + 1 < len(argv):
        chosen = argv[argv.index("--db") + 1]
    else:
        chosen = os.environ.get("OHS_DB", DASH["default_db"])
    path = Path(chosen)
    return str(path if path.is_absolute() else ROOT / path)


def _repo(path: str) -> Repo:
    return Repo(path, read_only=True)


# ---------------------------------------------------------------- cached reads (all via db/repo.py)


@st.cache_data(ttl=TTL, show_spinner=False)
def exposure_rows(path: str, table: str, start: str | None = None, end: str | None = None,
                  worker: str | None = None) -> list[dict[str, Any]]:
    """Rows of a materialised exposure table (dates inclusive)."""
    with _repo(path) as r:
        return r.get_exposure_rows(table, start, end, worker)


@st.cache_data(ttl=TTL, show_spinner=False)
def exposure_dates(path: str) -> list[str]:
    """Shift dates with personal exposure results."""
    with _repo(path) as r:
        return r.get_exposure_dates("personal_exposure_shift")


@st.cache_data(ttl=TTL, show_spinner=False)
def data_span(path: str) -> tuple[str, str] | None:
    """(first, last) zone_env timestamp."""
    with _repo(path) as r:
        return r.get_data_span()


@st.cache_data(ttl=TTL, show_spinner=False)
def zone_env(path: str, zone_id: str, start_ts: str, end_ts: str) -> pd.DataFrame:
    """Zone readings in [start, end), time-ordered."""
    with _repo(path) as r:
        return pd.DataFrame(r.get_zone_env(zone_id, start_ts, end_ts))


@st.cache_data(ttl=TTL, show_spinner=False)
def machine_status(path: str, machine_id: str, start_ts: str, end_ts: str) -> pd.DataFrame:
    """Machine readings in [start, end), time-ordered."""
    with _repo(path) as r:
        return pd.DataFrame(r.get_machine_status(machine_id, start_ts, end_ts))


@st.cache_data(ttl=TTL, show_spinner=False)
def assignments(path: str, worker_id: str, start_ts: str, end_ts: str) -> list[dict[str, Any]]:
    """A worker's zone assignment intervals overlapping [start, end)."""
    with _repo(path) as r:
        return r.get_zone_assignments(worker_id, start_ts, end_ts)


@st.cache_data(ttl=TTL, show_spinner=False)
def all_assignments(path: str) -> list[dict[str, Any]]:
    """Every zone assignment."""
    with _repo(path) as r:
        return r.get_all_zone_assignments()


@st.cache_data(ttl=TTL, show_spinner=False)
def ingest_stats(path: str) -> dict[str, Any]:
    """Accepted / flagged / rejected counts and per-device last-seen over all data."""
    with _repo(path) as r:
        return r.get_ingest_stats("2000-01-01T00:00:00Z", "2100-01-01T00:00:00Z")


@st.cache_data(ttl=TTL, show_spinner=False)
def table_counts(path: str) -> dict[str, int]:
    """Row count per table."""
    with _repo(path) as r:
        return r.table_counts()


# ---------------------------------------------------------------- display helpers


def to_local(ts: Any) -> pd.Series | pd.Timestamp:
    """UTC ISO string(s) or epoch ms -> mill local time (display layer only)."""
    if isinstance(ts, pd.Series):
        if pd.api.types.is_numeric_dtype(ts):
            return pd.to_datetime(ts, unit="ms", utc=True).dt.tz_convert(TZ)
        return pd.to_datetime(ts, utc=True, format="ISO8601").dt.tz_convert(TZ)
    return pd.Timestamp(ts).tz_convert(TZ)


def badge_html(text: str, colour: str, fg: str = "white") -> str:
    """An inline label (always carries text: meaning is never colour alone)."""
    border = "1px solid #333;" if colour.upper() == "#FFFFFF" else ""
    return (f"<span style='background:{colour};color:{fg};padding:2px 10px;border-radius:6px;"
            f"font-weight:700;{border}'>{text}</span>")


def band_badge(band: str) -> str:
    """Risk band label in its fixed colour."""
    fg = "black" if band in ("Low", "insufficient data") else "white"
    return badge_html(band.upper() if band != "insufficient data" else "INSUFFICIENT DATA", BAND_COLOURS[band], fg)


def header(page: str) -> bool:
    """Header band on every page: title, SIMULATED DATA badge, date range, versions.

    Returns False (after a plain message) if the database cannot be opened.
    """
    path = db_path()
    st.markdown(f"<div style='display:flex;align-items:center;gap:14px;flex-wrap:wrap'>"
                f"<span style='font-size:1.6rem;font-weight:700'>Textile OHS monitor — {page}</span>"
                f"{badge_html('SIMULATED DATA', '#B2182B')}</div>", unsafe_allow_html=True)
    try:
        dates = exposure_dates(path)
        span = data_span(path)
        rows = exposure_rows(path, "personal_exposure_shift", dates[-1], dates[-1]) if dates else []
        risk = exposure_rows(path, "worker_risk_index", dates[-1], dates[-1]) if dates else []
    except Exception as exc:                      # graceful: never raise in front of the panel
        st.error(f"Cannot read the database at `{path}`: {exc}")
        return False
    engine = rows[0]["engine_version"] if rows else "—"
    index_v = risk[0]["version"] if risk else "—"
    shown = f"{dates[0]} → {dates[-1]}" if dates else "no exposure results"
    raw = f"{span[0]} → {span[1]}" if span else "no telemetry"
    st.caption(f"Loaded shift dates: **{shown}** · telemetry {raw} (UTC; times shown in {TZ}) · "
               f"exposure engine **{engine}** · risk index **{index_v}** · database `{Path(path).name}` (read-only)")
    return True


def sidebar_filters(dates: list[str]) -> tuple[str, str]:
    """Shared date and shift selectors, kept in session state across pages."""
    st.sidebar.header("Selection")
    if "date" not in st.session_state or st.session_state["date"] not in dates:
        st.session_state["date"] = dates[-1]
    if "shift" not in st.session_state:
        st.session_state["shift"] = "All"
    st.sidebar.selectbox("Shift date", dates, key="date")
    st.sidebar.selectbox("Shift", ["All", "A", "B", "C"], key="shift")
    st.sidebar.caption("All values are SIMULATED. Personal figures are estimated_personal.")
    return st.session_state["date"], st.session_state["shift"]


def empty(message: str) -> None:
    """Plain empty state, then stop rendering this page."""
    st.info(message)
    st.stop()
