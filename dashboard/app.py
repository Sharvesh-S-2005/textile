"""Textile OHS dashboard - routing and shared setup only.

    streamlit run dashboard/app.py                         # default database (config.yaml)
    streamlit run dashboard/app.py -- --db demo/textile_ohs_demo.db

Reads SQLite through db/repo.py on a read-only connection; there is no API tier and
no SQL in dashboard/. Every page carries the SIMULATED DATA badge in its header.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="Textile OHS monitor (SIMULATED)", layout="wide")

PAGES = Path(__file__).resolve().parent / "pages"
navigation = st.navigation([
    st.Page(str(PAGES / "worker_drilldown.py"), title="Worker Drilldown", default=True),
    st.Page(str(PAGES / "live_zones.py"), title="Live Zones"),
    st.Page(str(PAGES / "system_results.py"), title="System & Results"),
])
navigation.run()
