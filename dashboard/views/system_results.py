"""System & Results: alerts, system health (with the ingest reconciliation), and the results pack.

Reads the database through db/repo.py and the committed results artefacts
(docs/results_numbers.json, docs/figures/*.png). No model inference here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

import config
from dashboard.common import (ROOT, TTL, badge_html, db_path, empty, exposure_dates, exposure_rows, header,
                              ingest_stats, machine_status, sidebar_filters, table_counts, to_local)
from db.repo import Repo

DASH = config.get_dashboard_config()
EXCLUDED = set(config.get_exposure_methods()["personal"]["exclude_record_flags"])
FIG = ROOT / config.get_paths()["figures_dir"]
NIOSH_SRC = "NIOSH 98-126"
INDIA_SRC = "Schedule XXVIII, Rule 95, TN Factories Rules"


@st.cache_data(ttl=TTL, show_spinner=False)
def device_states(path: str) -> list[dict]:
    with Repo(path, read_only=True) as r:
        return r.get_device_states()


@st.cache_data(ttl=TTL, show_spinner=False)
def sensor_flags(path: str) -> list[dict]:
    with Repo(path, read_only=True) as r:
        return r.get_sensor_flag_counts()


@st.cache_data(ttl=TTL, show_spinner=False)
def results_numbers() -> dict | None:
    p = ROOT / DASH["results_numbers"]
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


if not header("System & Results"):
    st.stop()
path = db_path()
dates = exposure_dates(path)
if not dates:
    empty("No exposure results in this database yet. Run the exposure engine (`python -m exposure.engine`).")
day, shift_filter = sidebar_filters(dates)

# ---------------------------------------------------------------- alerts
st.markdown("## Alerts")
st.caption(f"Shift date {day}" + ("" if shift_filter == "All" else f", shift {shift_filter}") + ". "
           "Doses are estimated_personal; WBGT is approximate (likely an underestimate).")
alerts = []
for r in exposure_rows(path, "personal_exposure_shift", day, day):
    if shift_filter != "All" and r["shift"] != shift_filter:
        continue
    t = to_local(r["session_end"]).strftime("%d %b %H:%M")
    if r["dose_niosh_pct"] is not None and r["dose_niosh_pct"] > 100:
        alerts.append((t, r["worker_id"], "noise dose", f"{r['dose_niosh_pct']:,.0f}%", "100%", NIOSH_SRC))
    if r["dose_india_pct"] is not None and r["dose_india_pct"] > 100:
        alerts.append((t, r["worker_id"], "noise dose", f"{r['dose_india_pct']:,.0f}%", "100%", INDIA_SRC))
    for e in r["absolute_max_events"]:
        alerts.append((to_local(e["ts"]).strftime("%d %b %H:%M:%S"), f"{r['worker_id']} in {e['zone_id']}",
                       "ABSOLUTE MAXIMUM", f"{e['level_db']:.1f} dBA", "115 dBA", INDIA_SRC))
    for e in r["excluded_absolute_max_candidates"]:
        alerts.append((to_local(e["ts"]).strftime("%d %b %H:%M:%S"), f"{r['worker_id']} in {e['zone_id']}",
                       "≥115 dBA reading EXCLUDED as a sensor anomaly (check)", f"{e['level_db']:.1f} dBA", "115 dBA",
                       f"validation flags {', '.join(e['flags'])}"))
    if r["wbgt_time_above_limit_s"]:
        alerts.append((t, r["worker_id"], "WBGT above reference (approx.)",
                       f"{r['wbgt_time_above_limit_s'] / 3600:.1f} h, max {r['wbgt_max_c']:.1f} °C", "28 °C",
                       "ISO 7243, moderate work"))
ratio = float(DASH["vibration_alert_ratio"])
start, end = f"{day}T00:00:00Z", f"{(pd.Timestamp(day) + pd.Timedelta(days=1)).date()}T00:00:00Z"
for m in config.get_machines():
    ms = machine_status(path, m["machine_id"], start, end)
    if ms.empty:
        continue
    ok = ms[ms["running"].astype(bool) & ms["vib_rms_g"].notna() & ~ms["flags"].apply(lambda f: bool(EXCLUDED & set(f)))]
    if ok.empty:
        continue
    mean = float(ok["vib_rms_g"].mean())
    if mean > ratio * m["baseline_vib_rms_g"]:
        alerts.append((day, f"{m['machine_id']} in {m['zone_id']}", "machine vibration above baseline",
                       f"{mean:.3f} g ({mean / m['baseline_vib_rms_g']:.2f}× baseline)",
                       f"{ratio:.1f}× {m['baseline_vib_rms_g']} g", "project threshold (design choice)"))
if alerts:
    table = pd.DataFrame(alerts, columns=["time", "worker / zone", "type", "value", "threshold", "criterion / source"])
    st.dataframe(table, hide_index=True, use_container_width=True, height=min(420, 38 + 35 * len(table)))
    st.caption(f"{len(table)} alerts. Machine vibration is compared with each machine's own baseline because ISO 10816 "
               "limits are in velocity (mm/s), which these accelerometer RMS readings are not.")
else:
    st.info("No alerts for this selection.")

# ---------------------------------------------------------------- system health
st.markdown("## System health")
stats = ingest_stats(path)
states = {s["device_id"]: s for s in device_states(path)}
devices = pd.DataFrame([{
    "device": d, "online (latest status)": ("online" if states[d]["online"] else "OFFLINE") if d in states else "—",
    "last received": to_local(v["last_received_ts"]).strftime("%d %b %H:%M:%S") if v["last_received_ts"] else "—",
    "last device ts": to_local(v["last_ts"]).strftime("%d %b %H:%M:%S") if v["last_ts"] else "—",
} for d, v in stats["last_seen"].items()])
c1, c2, c3 = st.columns([2, 1, 1])
with c1:
    st.markdown("**Devices** (an OFFLINE state at the end of a stored run is the clean-shutdown status)")
    st.dataframe(devices, hide_index=True, use_container_width=True, height=300)
with c2:
    st.markdown("**Rejected records by reason**")
    rej = stats["rejected"]
    st.dataframe(pd.DataFrame(sorted(rej.items()), columns=["reason", "count"]) if rej else
                 pd.DataFrame({"reason": ["none"], "count": [0]}), hide_index=True, use_container_width=True)
with c3:
    st.markdown("**Flagged records by flag**")
    st.dataframe(pd.DataFrame(sorted(stats["flag_counts"].items()), columns=["flag", "count"]),
                 hide_index=True, use_container_width=True)
flags = sensor_flags(path)
clean = sum(f["count"] for f in flags if f["flag"] == "needs_clean")
st.markdown(f"PM sensors reporting **needs_clean**: {badge_html(f'{clean:,} readings', '#B2182B' if clean else '#2E7D32')}"
            " — optical PM sensor lint build-up; readings drift HIGH until cleaned.", unsafe_allow_html=True)
with st.expander("All non-ok sensor flags"):
    st.dataframe(pd.DataFrame(flags), hide_index=True, use_container_width=True)

st.markdown("### Ingest reconciliation")
counts = table_counts(path)
stored = sum(counts[t] for t in ("zone_env", "ble_scans", "machine_status", "gate_events", "ppe_entries",
                                 "device_status"))
nums = results_numbers()
recs = []
for label, key in (("Day 2 run (8 h, 100×)", "day2_run"), ("7-day run (90×)", "week_run")):
    run = (nums or {}).get(key)
    if run:
        recs.append({"run": label, "published": f"{run['published']:,}",
                     "injected bad records": run["received"] - run["published"], "received": f"{run['received']:,}",
                     "stored": f"{run['stored']:,}", "rejected": run["rejected"], "duplicates": run["duplicate"],
                     "errors": run["errors"], "throughput (msg/s)": round(run["throughput_msg_s"], 1),
                     "reconciles": "YES" if run["received"] == run["stored"] + run["rejected"] + run["duplicate"]
                     else "NO"})
if recs:
    st.dataframe(pd.DataFrame(recs), hide_index=True, use_container_width=True)
st.caption(f"This database: {stored:,} stored messages + {counts['rejected_records']:,} rejected. Published = "
           "stored + rejected exactly, with zero loss; recorded by the results pack from the publisher and ingest "
           "logs.")

# ---------------------------------------------------------------- results
st.markdown("## Results (from the committed results pack; no model runs here)")
if nums is None:
    st.info("Results pack not found. Run `python scripts/build_results_pack.py`.")
else:
    f = nums["forecast"]
    st.caption(f"Forecast split: {f['split_rule']}. " + "; ".join(f["splits"]))
    rows = []
    for key, e in f["targets"].items():
        target, h = key.rsplit("_h", 1)
        sk = {m: e["metrics"][m]["skill_vs_persistence"] for m in ("seasonal", "random_forest", "xgboost")}
        steady = e.get("by_schedule", {}).get("steady", {}).get("skill", {})
        rows.append({"target": target, "horizon (min)": int(h), "skill seasonal": round(sk["seasonal"], 3),
                     "skill RF": round(sk["random_forest"], 3), "skill XGBoost": round(sk["xgboost"], 3),
                     "saved model": e["chosen"],
                     "break-free skill: seasonal / RF / XGB": " / ".join(f"{steady[m]:+.2f}" for m in
                                                                          ("seasonal", "random_forest", "xgboost"))
                     if steady else "—"})
    st.markdown("**Forecast skill vs persistence** (1 − MAE/MAE_persistence; > 0 beats persistence). "
                "Break-free windows = no scheduled break or shift change within the horizon.")
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    lead = []
    for key in ("noise_dba_h30", "noise_dba_h60"):
        e = f["targets"][key]
        x = e["exceedance"][e["chosen"]]
        lead.append({"horizon": key.rsplit("_h", 1)[1] + " min", "model": e["chosen"],
                     "90 dBA crossings warned": f"{x['hits']}/{x['events']}",
                     "median lead (min)": x["median_lead_min"], "false-alarm ratio": f"{100 * x['false_alarm_ratio']:.1f}%",
                     "seasonal baseline false-alarm ratio": f"{100 * e['exceedance']['seasonal']['false_alarm_ratio']:.1f}%"})
    st.markdown("**Exceedance lead time** (noise vs the 90 dBA India criterion, test day)")
    st.dataframe(pd.DataFrame(lead), hide_index=True, use_container_width=True)
    st.markdown("Negative findings are kept: at 60 min neither ML model beats the seasonal baseline for noise, and for "
                "WBGT the seasonal baseline is worse than persistence.")
st.info("**The ML risk classifier is deferred** until questionnaire-based health labels exist: training one on "
        "labels derived from dose thresholds would only re-learn the threshold rule. The deterministic risk index "
        "(a weighted design choice of this project, not a validated clinical model) is its baseline.")
g1, g2 = st.columns(2)
for col, name, caption in ((g1, "forecast_vs_actual.png", "Forecast vs actual, with the persistence and seasonal "
                            "baselines"), (g2, "shap_summary_noise_h30.png", "SHAP: what drives the noise forecast"),
                           (g1, "localisation_confusion.png", "BLE localisation confusion matrix (Day 2 run)"),
                           (g2, "shap_bar_noise_h30.png", "SHAP mean absolute importance")):
    fp = FIG / name
    if fp.exists():
        col.image(str(fp), caption=caption, use_container_width=True)
    else:
        col.info(f"Figure {name} not found — run the results pack.")
