"""Models: every ML result in one place, read from the saved results artefacts.

Reads docs/results_numbers.json (written by scripts/build_results_pack.py) and the
committed figures in docs/figures/. No model is loaded and no inference runs here.
"""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

import config
from dashboard.common import ROOT, TTL, header

DASH = config.get_dashboard_config()
FIG = ROOT / config.get_paths()["figures_dir"]
COUPLING = ("The simulator couples machine vibration to zone noise by construction, so machine-to-noise results show "
            "that the pipeline uses the link, not that it holds in a real mill.")


@st.cache_data(ttl=TTL, show_spinner=False)
def ml_results() -> dict | None:
    p = ROOT / DASH["results_numbers"]
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8")).get("ml")


def figure(col: st.delta_generator.DeltaGenerator, name: str, caption: str) -> None:
    fp = FIG / name
    if fp.exists():
        col.image(str(fp), caption=caption, use_container_width=True)
    else:
        col.info(f"Figure {name} not found — run the results pack.")


if not header("Models"):
    st.stop()
st.info("Read from saved results (`docs/results_numbers.json`, `docs/figures/`); no model runs on this page. "
        "Every number is from **simulated data** and one test day (2026-10-26). Full write-up: `docs/results_ml.md`; "
        "model cards: `docs/model_cards.md`.")
ml = ml_results()
if ml is None:
    st.error("No ML results found. Run `python scripts/build_results_pack.py` after the ML modules.")
    st.stop()

st.markdown("## Model comparison (test day)")
st.caption("Baselines sit in the same table as the models that must beat them. Classification rows carry the "
           "prevalence (base rate) in the target column: read PR-AUC against it, not accuracy.")
table = pd.DataFrame(ml["comparison"]).rename(columns={
    "task": "Task", "target": "Target", "h": "h (min)", "model": "Model", "metric": "Headline metric",
    "value": "Value", "secondary": "Secondary", "note": "Note"})
tasks = list(dict.fromkeys(table["Task"]))
chosen = st.multiselect("Tasks", tasks, default=tasks)
st.dataframe(table[table["Task"].isin(chosen)], hide_index=True, use_container_width=True)

st.markdown("## Exceedance classification (noise ≥ 90 dBA at t+h)")
st.caption("Label = the measured level at t+h: a future measurement, not a rule on the inputs, so not circular. "
           "Threshold = highest validation recall with false-positive rate ≤ 10% (a stated policy).")
c1, c2 = st.columns(2)
figure(c1, "classification_pr_roc_above_h30.png", "`above`, t+30: PR and ROC curves")
figure(c2, "classification_pr_roc_crossing_h30.png", "`crossing`, t+30: PR and ROC curves")
figure(c1, "classification_pr_roc_crossing_h60.png", "`crossing`, t+60: PR and ROC curves")
figure(c2, "classification_confusion.png", "Confusion matrices of the saved classifiers")

st.markdown("## Objective 6: does worker and machine data help? (Model 1 / 2 / 3)")
st.caption("Reframed onto hazard forecasting and exceedance classification, because no honest risk label exists "
           "yet. Same model class, split and seed; each feature set tuned separately; 95% block-bootstrap intervals "
           "on paired differences.")
figure(st, "ablation_skill.png", "Forecast skill and classification PR-AUC by feature set")
st.markdown("**Finding:** machine and zone-history data give a small, statistically distinguishable gain in noise "
            "forecasting; worker data adds nothing of practical size; no set helps dust, heat or classification.")
st.warning(COUPLING)

st.markdown("## Machine condition monitoring")
figure(st, "anomaly_m5_vibration.png", "M5 vibration over the week, healthy-machine band, detection points")
st.warning(COUPLING)

st.markdown("## Deep learning")
st.markdown(ml["deep_learning"])
st.info("**The ML risk classifier is deferred** until questionnaire-based health labels exist. The risk index on "
        "the Worker Drilldown is deterministic (a weighted design choice of this project), not an ML model.")
