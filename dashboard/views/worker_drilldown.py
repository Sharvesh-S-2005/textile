"""Worker Drilldown: sensors -> localisation -> exposure -> index, for one worker and day."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import config
from exposure.risk_index import RiskParams, worst_component

from dashboard.common import (HAZARD, INDEX_CAVEAT, LIGHT, NEUTRAL, TZ, all_assignments, band_badge, db_path,
                              empty, exposure_dates, exposure_rows, header, sidebar_filters, to_local, zone_env)

EXCLUDED = set(config.get_exposure_methods()["personal"]["exclude_record_flags"])
RISK_PARAMS = RiskParams.from_config(config.get_risk_index_config(), config.get_exposure_limits()["wbgt_limit_c"])
COMPONENT_COLOUR = {"hearing": HAZARD["noise"], "heat": HAZARD["heat"], "respiratory": HAZARD["dust"]}


def local_naive(epoch_ms: int) -> pd.Timestamp:
    """Mill local wall-clock time without tz info (plotly shapes, annotations and traces then agree)."""
    return to_local(pd.Series([epoch_ms]))[0].tz_localize(None)


def fmt(v: float | None, spec: str, unit: str = "") -> str:
    return "n/a" if v is None else f"{format(v, spec)}{unit}"


if not header("Worker Drilldown"):
    st.stop()
path = db_path()
dates = exposure_dates(path)
if not dates:
    empty("No exposure results in this database yet. Run the exposure engine (`python -m exposure.engine`).")
day, shift_filter = sidebar_filters(dates)

risk = {r["worker_id"]: r for r in exposure_rows(path, "worker_risk_index", day, day)}
shifts = [r for r in exposure_rows(path, "personal_exposure_shift", day, day)
          if shift_filter == "All" or r["shift"] == shift_filter]
if not shifts:
    empty(f"No worker sessions on {day}" + ("" if shift_filter == "All" else f" for shift {shift_filter}") + ".")

workers = sorted({r["worker_id"] for r in shifts}, key=lambda w: -(risk[w]["overall_index"] if w in risk else -1))
if st.session_state.get("worker") not in workers:
    st.session_state["worker"] = workers[0]          # default: highest index that day
worker = st.sidebar.selectbox(
    "Worker (sorted by index, highest first)", workers, key="worker",
    format_func=lambda w: f"{w} — index {risk[w]['overall_index']:.0f}" if w in risk else w)
session = sorted((r for r in shifts if r["worker_id"] == worker), key=lambda r: r["session_start_ms"])[0]
r_index = risk.get(worker)

st.subheader(f"{worker} · shift {session['shift']} · {to_local(session['session_start']):%d %b %H:%M} – "
             f"{to_local(session['session_end']):%H:%M} ({TZ})")

# ---------------------------------------------------------------- tiles
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("L_EX,8h (ISO 9612)", fmt(session["l_ex_8h_db"], ".1f", " dBA"))
    st.caption(f"exposure duration {session['exposure_duration_s'] / 3600:.2f} h · L_Aeq "
               f"{fmt(session['l_aeq_db'], '.1f', ' dBA')}")
with c2:
    st.metric("Noise dose — NIOSH | India", f"{fmt(session['dose_niosh_pct'], ',.0f', '%')} | "
              f"{fmt(session['dose_india_pct'], ',.0f', '%')}")
    st.caption("NIOSH 98-126 (85 dBA, 3 dB, 80 dBA threshold) · Schedule XXVIII, Rule 95, TN Factories Rules "
               "(90 dBA, 5 dB, 90 dBA threshold)")
with c3:
    above = session["wbgt_time_above_limit_s"]
    st.metric("WBGT max · time above 28 °C", f"{fmt(session['wbgt_max_c'], '.1f', ' °C')} · "
              f"{fmt(None if above is None else above / 3600, '.1f', ' h')}")
    st.caption("**APPROXIMATE** — Stull psychrometric wet-bulb stands in for natural wet-bulb, so WBGT is likely an "
               "underestimate · ISO 7243, moderate work")
with c4:
    st.metric("PM2.5 TWA raw | corrected", f"{fmt(session['pm25_twa_raw'], '.0f')} | "
              f"{fmt(session['pm25_twa_corrected'], '.0f')} µg/m³")
    st.caption("**INDICATIVE** — optical sensor; not comparable with regulatory cotton-dust limits")

# ---------------------------------------------------------------- data quality band
q1, q2, q3 = st.columns([1, 1, 3])
q1.metric("Data coverage", fmt(session["data_coverage_pct"], ".1f", "%"))
q2.metric("Unaccounted time", f"{session['unaccounted_pct']:.2f}%",
          help="Share of the session whose zone is UNKNOWN or unassigned")
q2.caption(f"{session['unaccounted_time_s'] / 60:.0f} min of the session")
q3.warning("Where unaccounted (UNKNOWN-zone) time exists, every exposure figure on this page is a **lower bound**: "
           "it covers only the time the worker's zone is known. Missing readings lower coverage; they are never "
           "treated as zero." + (f" Result flags: {', '.join(session['flags'])}." if session["flags"] else ""))

# ---------------------------------------------------------------- exposure timeline
start_ms, end_ms = session["session_start_ms"], session["session_end_ms"]
intervals = [a for a in all_assignments(path) if a["worker_id"] == worker and not a["outside_session"]
             and a["from_epoch_ms"] < end_ms and a["to_epoch_ms"] > start_ms]
st.markdown("**Zone occupied** (top: hatched = UNKNOWN, i.e. missing, not quiet; dark grey = break zone Z7) "
            "and **the noise there** (bottom; spike/stuck-flagged readings excluded, as in the exposure engine)")
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.2, 0.8], vertical_spacing=0.04)
break_zone = config.get_simulation_config()["worker_dynamics"]["break_zone"]
noise_frames = []
for a in intervals:
    lo, hi = max(a["from_epoch_ms"], start_ms), min(a["to_epoch_ms"], end_ms)
    unknown = a["zone_id"] == "UNKNOWN"
    x0, x1 = local_naive(lo), local_naive(hi)
    fill = "#FFFFFF" if unknown else ("#7A7A7A" if a["zone_id"] == break_zone else LIGHT)
    fig.add_shape(type="rect", x0=x0, x1=x1, y0=0, y1=1, fillcolor=fill,
                  line={"color": NEUTRAL, "width": 1, "dash": "dash" if unknown else "solid"}, row=1, col=1)
    if hi - lo >= 12 * 60_000 or unknown:            # label wide enough segments (and every UNKNOWN)
        fig.add_annotation(x=x0 + (x1 - x0) / 2, y=0.5, showarrow=False,
                           text="UNKNOWN" if unknown else a["zone_id"], font={"color": "black", "size": 12},
                           row=1, col=1)
    if not unknown:
        env = zone_env(path, a["zone_id"], a["from_ts"], a["to_ts"])
        if not env.empty:
            # Same rule as the exposure engine: detected sensor anomalies are not data.
            env = env[(env["ts_epoch_ms"] > lo) & (env["ts_epoch_ms"] <= hi)
                      & ~env["flags"].apply(lambda f: bool(EXCLUDED & set(f)))]
            noise_frames.append(env[["ts_epoch_ms", "noise_dba"]])
if noise_frames:
    noise = pd.concat(noise_frames).sort_values("ts_epoch_ms")
    fig.add_trace(go.Scatter(x=to_local(noise["ts_epoch_ms"]).dt.tz_localize(None), y=noise["noise_dba"], mode="lines",
                             line={"color": HAZARD["noise"], "width": 1.2}, name="noise where the worker was (dBA)"),
                  row=2, col=1)
for level, dash, label in ((90, "dash", "90 dBA India criterion"), (85, "dot", "85 dBA NIOSH criterion")):
    fig.add_hline(y=level, line_dash=dash, line_color="#9E9E9E", annotation_text=label, row=2, col=1)
fig.update_yaxes(range=[0, 1], showticklabels=False, showgrid=False, row=1, col=1)
fig.update_yaxes(title_text="noise (dBA)", row=2, col=1)
fig.update_xaxes(title_text=f"time ({TZ})", row=2, col=1)
fig.update_xaxes(type="date")       # row 1 holds only shapes; without this its axis is inferred as linear
fig.update_layout(height=420, margin={"t": 10, "b": 10}, legend={"orientation": "h", "y": -0.18})
st.plotly_chart(fig, use_container_width=True)
if not noise_frames:
    st.info("No raw zone telemetry for this date in this database (the offline demo copy keeps raw readings for "
            "its demo day only). The exposure figures above come from the stored results and are complete.")

# ---------------------------------------------------------------- risk index panel
st.markdown(f"**{INDEX_CAVEAT}**")
left, right = st.columns([3, 2])
with left:
    if r_index is None:
        st.info("No risk index computed for this worker and day.")
    else:
        worst = worst_component(r_index["components"], RISK_PARAMS)
        st.markdown(f"### Risk index {r_index['overall_index']:.1f} / 100 &nbsp; {band_badge(r_index['band'])}",
                    unsafe_allow_html=True)
        st.markdown(f"**Worst single component: {worst['component']} {worst['value']:.1f} / 100** &nbsp; "
                    f"{band_badge(worst['band'])}", unsafe_allow_html=True)
        st.caption("Both are reported because the components are not on commensurable scales (hearing is "
                   "standards-anchored, respiratory is cohort-relative); the weighted composite can hide one high "
                   "component behind two low ones.")
        st.caption(f"label `{r_index['label']}` · confidence {r_index['confidence_pct']:.1f}% "
                   + ("— below 80%: band suppressed, number shown" if r_index["insufficient_data"] else "")
                   + f" · weights {r_index['weights']} · {r_index['version']}")
        contrib = r_index["contributions"]
        bars = go.Figure()
        for comp in ("hearing", "heat", "respiratory"):
            c = contrib[comp]
            bars.add_trace(go.Bar(y=["index"], x=[c["absolute"]], orientation="h", name=comp,
                                  marker_color=COMPONENT_COLOUR[comp], text=[f"{comp} {c['percent']:.0f}%"],
                                  textposition="inside", insidetextanchor="middle"))
        bars.update_layout(barmode="stack", height=130, margin={"t": 5, "b": 5}, showlegend=False,
                           xaxis={"range": [0, 100], "title": "contribution to the index (points of 100)"})
        st.plotly_chart(bars, use_container_width=True)
        comps = r_index["components"]
        st.caption("Components (0–100): " + " · ".join(f"{k} {v:.0f}" for k, v in comps.items())
                   + " · respiratory is **cohort-relative** (ranks workers; not a health threshold)")
        st.markdown("**Top drivers**\n" + "\n".join(f"- {d}" for d in r_index["drivers"]))
        ppe = r_index["ppe_status"]
        if ppe:
            st.caption(f"PPE recorded: hearing {ppe.get('hearing_ppe')}, respiratory {ppe.get('respiratory_ppe')} "
                       "— shown only; it never reduces the index or any dose.")
with right:
    st.markdown("#### The regulated quantities")
    st.metric("NIOSH dose", fmt(session["dose_niosh_pct"], ",.0f", "%"))
    st.metric("India dose (TN Schedule XXVIII)", fmt(session["dose_india_pct"], ",.0f", "%"))
    st.metric("L_EX,8h", fmt(session["l_ex_8h_db"], ".1f", " dBA"))
    st.caption("The index ranks; the dose is the regulated quantity. 100% = the daily allowance.")

# ---------------------------------------------------------------- cumulative panel
st.markdown("#### Cumulative exposure")
cum = [r for r in exposure_rows(path, "personal_exposure_cumulative", day, day, worker)]
if not cum:
    st.info("No cumulative rows for this worker and day.")
else:
    table = pd.DataFrame([{
        "window": f"{r['window_days']} days" + (" (PARTIAL)" if r["partial"] else ""),
        "window coverage %": round(r["window_coverage_pct"], 1),
        "days worked": r["days_worked"],
        "cumulative dose NIOSH %": None if r["cumulative_dose_niosh_pct"] is None else round(r["cumulative_dose_niosh_pct"]),
        "cumulative dose India %": None if r["cumulative_dose_india_pct"] is None else round(r["cumulative_dose_india_pct"]),
        "mean daily L_EX,8h (dBA, energy mean)": None if r["mean_daily_l_ex_8h_db"] is None else round(r["mean_daily_l_ex_8h_db"], 1),
        "days > 100% NIOSH / India": f"{r['days_dose_niosh_over_100']} / {r['days_dose_india_over_100']}",
    } for r in sorted(cum, key=lambda r: r["window_days"])])
    st.dataframe(table, hide_index=True, use_container_width=True)
    st.caption("A partial window covers fewer days than its length; its totals are over the days that exist and are "
               "never scaled up.")
