"""Live Zones: the newest reading of every zone, and noise vs machine state for one zone and shift."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config
from dashboard.common import (HAZARD, LIGHT, NEUTRAL, TZ, all_assignments, badge_html, data_span, db_path, empty,
                              exposure_dates, header, machine_status, sidebar_filters, to_local, zone_env)
from db.repo import from_epoch_ms, to_epoch_ms
from exposure.heat import wbgt_estimate
from shifts import ShiftSchedule

EXCLUDED = set(config.get_exposure_methods()["personal"]["exclude_record_flags"])
LOOKBACK_MS = 10 * 60_000
SENSORS = ("noise", "pm", "temp_rh", "globe", "co2")


def naive(epoch_ms: pd.Series) -> pd.Series:
    """Mill local wall-clock time without tz (consistent plotly axes)."""
    return to_local(epoch_ms).dt.tz_localize(None)


if not header("Live Zones"):
    st.stop()
path = db_path()
span = data_span(path)
dates = exposure_dates(path)
if span is None:
    empty("No zone telemetry in this database yet. Start ingest and the simulator (or hardware).")
day, shift_filter = sidebar_filters(dates) if dates else (span[1][:10], "All")

newest_ms = to_epoch_ms(span[1])
wall_age_s = (datetime.now(timezone.utc).timestamp() * 1000 - newest_ms) / 1000
live = wall_age_s < 3600
st.caption(f"Newest reading in the database: **{to_local(span[1]):%d %b %H:%M:%S} ({TZ})** — "
           + (f"{wall_age_s:.0f} s ago (live)" if live else "stored data, not live (ages below are relative to the "
                                                               "newest reading)"))

zones = config.get_zones()
machines_by_zone: dict[str, list[str]] = {}
for m in config.get_machines():
    machines_by_zone.setdefault(m["zone_id"], []).append(m["machine_id"])
present = [a for a in all_assignments(path) if a["zone_id"] != "UNKNOWN" and not a["outside_session"]
           and a["from_epoch_ms"] <= newest_ms < a["to_epoch_ms"]]

lo, hi = from_epoch_ms(newest_ms - LOOKBACK_MS), from_epoch_ms(newest_ms + 1)
cols = st.columns(4)
for i, z in enumerate(zones):
    zid = z["zone_id"]
    env = zone_env(path, zid, lo, hi)
    with cols[i % 4].container(border=True):
        st.markdown(f"**{zid} · {z['name']}**")
        if env.empty:
            st.info("no reading in the last 10 minutes of data")
            continue
        last = env.iloc[-1]
        valid = env[~env["flags"].apply(lambda f: bool(EXCLUDED & set(f)))]
        cur = valid.iloc[-1] if not valid.empty else last
        wbgt = None
        if all(cur[k] is not None and pd.notna(cur[k]) for k in ("temp_c", "rh_pct", "globe_temp_c")):
            wbgt = wbgt_estimate(cur["temp_c"], cur["rh_pct"], cur["globe_temp_c"]).wbgt_c

        def show(v: object, spec: str, unit: str) -> str:
            return "n/a" if v is None or pd.isna(v) else f"{format(v, spec)} {unit}"

        swatch = "<span style='color:{c};font-size:1.2rem'>&#9632;</span>"
        rows = [("noise", "Noise", show(cur["noise_dba"], ".1f", "dBA"), ""),
                ("dust", "PM2.5", show(cur["pm25_ugm3"], ".0f", "µg/m³"), " (indicative)"),
                ("heat", "WBGT", show(wbgt, ".1f", "°C"), " (approx.)"),
                ("co2", "CO2", show(cur["co2_ppm"], ".0f", "ppm"), "")]
        st.markdown("<br>".join(f"{swatch.format(c=HAZARD[h])} {label}: <b style='font-size:1.25rem'>{value}</b>"
                                f"<span style='font-size:0.8rem'>{note}</span>" for h, label, value, note in rows),
                    unsafe_allow_html=True)
        running = 0
        for mid in machines_by_zone.get(zid, []):
            ms = machine_status(path, mid, lo, hi)
            running += int(bool(ms.iloc[-1]["running"])) if not ms.empty else 0
        n_machines = len(machines_by_zone.get(zid, []))
        workers = sum(1 for p in present if p["zone_id"] == zid)
        st.caption(f"machines running **{running} / {n_machines}** · workers present **{workers}**")
        bad = [f"{s}: {last[f'flag_{s}']}" for s in SENSORS if last[f"flag_{s}"] != "ok"]
        health = ("all sensors ok" if not bad and not last["flags"] else
                  "; ".join(bad + [f for f in last["flags"] if f not in ("sensor_not_ok",)]))
        colour = "#2E7D32" if health == "all sensors ok" else ("#B2182B" if any("needs_clean" in x for x in bad)
                                                               else "#E69F00")
        st.markdown(f"Sensor health: {badge_html(health, colour, 'white')}", unsafe_allow_html=True)
        age = (newest_ms - int(last["ts_epoch_ms"])) / 1000
        st.caption(f"last reading {to_local(last['ts']):%H:%M:%S} · "
                   + (f"{wall_age_s + age:.0f} s old" if live else f"{age:.0f} s before the newest reading"))

# ---------------------------------------------------------------- the coupling chart
st.markdown("### Does noise follow the machines?")
zone_names = {z["zone_id"]: f"{z['zone_id']} · {z['name']}" for z in zones}
machine_zones = [z for z in zone_names if machines_by_zone.get(z)]
default = machine_zones.index("Z5") if "Z5" in machine_zones else 0
zone = st.selectbox("Zone", machine_zones, index=default, format_func=zone_names.get)
shift_id = shift_filter if shift_filter != "All" else "A"
inst = ShiftSchedule(config.get_shifts()).instance(shift_id, date.fromisoformat(day))
s_ms, e_ms = int(inst.start.timestamp() * 1000), int(inst.end.timestamp() * 1000)
env = zone_env(path, zone, from_epoch_ms(s_ms), from_epoch_ms(e_ms))
if env.empty:
    empty(f"No readings for {zone} during shift {shift_id} of {day}.")
env = env[~env["flags"].apply(lambda f: bool(EXCLUDED & set(f)))]

frames = []
for mid in machines_by_zone[zone]:
    ms = machine_status(path, mid, from_epoch_ms(s_ms), from_epoch_ms(e_ms))
    if not ms.empty:
        frames.append(ms[["ts_epoch_ms", "running"]].assign(machine=mid))
fig = go.Figure()
if frames:
    run = (pd.concat(frames).pivot_table(index="ts_epoch_ms", columns="machine", values="running", aggfunc="last")
           .sort_index().ffill().fillna(0).sum(axis=1))
    stopped = run < 0.5
    # Shade each contiguous stretch with every machine stopped.
    block = (stopped != stopped.shift()).cumsum()
    for _, seg in run[stopped].groupby(block[stopped]):
        fig.add_vrect(x0=naive(pd.Series([seg.index[0]]))[0], x1=naive(pd.Series([seg.index[-1] + 10_000]))[0],
                      fillcolor=LIGHT, opacity=0.35, line_width=0, layer="below")
for brk in inst.breaks:
    fig.add_vrect(x0=naive(pd.Series([int(brk.start.timestamp() * 1000)]))[0],
                  x1=naive(pd.Series([int(brk.end.timestamp() * 1000)]))[0], fillcolor="rgba(0,0,0,0)",
                  line={"color": "#9E9E9E", "dash": "dash", "width": 1.5},
                  annotation_text=f"{brk.kind} break", annotation_position="top left")
fig.add_trace(go.Scatter(x=naive(env["ts_epoch_ms"]), y=env["noise_dba"], mode="lines",
                         line={"color": HAZARD["noise"], "width": 1.2}, name="noise (10-s L_Aeq, dBA)"))
fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker={"size": 14, "color": LIGHT, "symbol": "square"},
                         name="all machines stopped (shaded)"))
fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line={"color": "#9E9E9E", "dash": "dash"},
                         name="scheduled break (dashed box)"))
fig.add_hline(y=90, line_dash="dash", line_color=NEUTRAL, annotation_text="90 dBA India criterion")
fig.update_layout(height=430, margin={"t": 20, "b": 10}, legend={"orientation": "h", "y": -0.2},
                  yaxis_title="noise (dBA)", xaxis_title=f"time ({TZ}), shift {shift_id}, {day}")
st.plotly_chart(fig, use_container_width=True)
st.caption("Machine state comes from the machine nodes' clamp-current readings (`running`); the shading is where "
           "every machine in the zone reported stopped. Breaks come from the shift calendar. Spike/stuck-flagged "
           "readings are excluded, as in the exposure engine.")
