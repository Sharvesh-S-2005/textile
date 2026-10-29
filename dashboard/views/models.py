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
groups = ml.get("classification_groups") or []
lead = next((r for r in groups if r["variant"] == "crossing" and r["h"] == 30 and "," not in r["zones"]
             and r["pr_auc"] is not None), None)
if lead:
    st.warning(f"**Where a warning matters most, the classifier barely beats chance.** In {lead['zones']} "
               f"(hovering at 90 dBA), `crossing` PR-AUC is {lead['pr_auc']:.2f} against a base rate of "
               f"{lead['prevalence']:.2f}, with false alarms on {lead['fpr']:.0%} of quiet minutes. Its crossings are "
               "small fluctuations, not schedule events. In the always-loud zones the label is trivially predictable "
               "(machines restart after every break), so the pooled PR-AUC overstates the useful skill.")
    table_g = pd.DataFrame([{
        "Variant": r["variant"], "h (min)": r["h"], "Group": r["group"], "Zones": r["zones"], "n": r["n"],
        "Prevalence": None if r["prevalence"] is None else round(r["prevalence"], 3),
        "PR-AUC": None if r["pr_auc"] is None else round(r["pr_auc"], 3),
        "Lift (PR-AUC ÷ prevalence)": None if r["lift"] is None else round(r["lift"], 2),
        "Recall": None if r["recall"] is None else round(r["recall"], 3),
        "FPR": None if r["fpr"] is None else round(r["fpr"], 3), "Character": r["character"]} for r in groups])
    st.dataframe(table_g, hide_index=True, use_container_width=True)
c1, c2 = st.columns(2)
figure(c1, "classification_pr_roc_above_h30.png", "`above`, t+30: PR and ROC curves")
figure(c2, "classification_pr_roc_crossing_h30.png", "`crossing`, t+30: PR and ROC curves")
figure(c1, "classification_pr_roc_crossing_h60.png", "`crossing`, t+60: PR and ROC curves")
figure(c2, "classification_confusion.png", "Confusion matrices of the saved classifiers")

st.markdown("## Objective 6: does worker and machine data help? (Model 1 / 2 / 3)")
st.caption("Reframed onto hazard forecasting and exceedance classification, because no honest risk label exists "
           "yet. Same model class, split and seed; each feature set tuned separately; 95% block-bootstrap intervals "
           "on paired differences.")
figure(st, "ablation_redundancy.png", "Skill with and without target history (and without the timetable)")
st.markdown("**Finding (negative on this dataset):** machine and worker data add +0.017 / +0.028 noise skill with "
            "target lags and only about +0.014 without them; nothing once the timetable is also removed. The "
            "information is redundant with the timetable: a model with **no noise input** already reaches skill "
            "+0.76 / +0.77. **Implication:** if a noise or PM sensor fails, zone forecasting continues from context "
            "and the timetable (sensor-failure resilience).")
figure(st, "ablation_skill.png", "Day 6 ablation: forecast skill and classification PR-AUC by feature set")
st.warning(COUPLING)

st.markdown("## Machine condition monitoring")
figure(st, "anomaly_m5_vibration.png", "M5 vibration over the week, healthy-machine band, detection points")
st.markdown("**The simpler method was chosen on evidence.** Both detectors flag the degrading machine within a day "
            "of each other (day 2.08 EWMA, 2.12 isolation forest), and neither alarms on a healthy machine, but the "
            "EWMA control chart's false-positive rate is more than 50× lower (0.02% vs 1.03%), and it is the method a "
            "maintenance engineer already recognises.")
st.warning(COUPLING)

st.markdown("## Deep learning")
st.markdown(ml["deep_learning"])
st.info("**The ML risk classifier is deferred** until questionnaire-based health labels exist. The risk index on "
        "the Worker Drilldown is deterministic (a weighted design choice of this project), not an ML model.")
