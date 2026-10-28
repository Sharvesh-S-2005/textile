# Project: AI-IoT Textile Occupational Health Monitoring

B.Tech final-year project + research project intended for journal publication.

The system monitors noise, cotton dust (PM2.5/PM10), heat stress (WBGT), temperature,
humidity and CO2 across textile mill zones; tracks which worker is in which zone and
for how long (BLE zone-level localisation + RFID shift check-in); monitors machine
status and vibration non-invasively; calculates each worker's shift and cumulative
exposure using occupational hygiene standards; and uses machine learning to forecast
hazard levels and estimate personalised exposure-related occupational health risk.

---

## CURRENT PHASE — READ THIS FIRST

**Hardware has been ordered but has NOT arrived.** All development runs against
`simulator/`, which publishes synthetic but realistic data over MQTT using the exact
schema defined in `docs/data_contract.md`.

**The single most important architectural rule:**

> Nothing downstream of MQTT may know that data is simulated.
> When the ESP32 firmware arrives, it replaces `simulator/publisher.py` and
> NOTHING ELSE CHANGES.

Never let the simulator write to the database directly. Never import simulator code
from `ingest/`, `exposure/`, `ml/`, `api/` or `dashboard/`. Data always flows:

```
simulator/publisher.py  ->  MQTT broker  ->  ingest/subscriber.py  ->  SQLite
                                                                        |
                        dashboard/  <-  api/  <-  ml/  <-  exposure/  <-+
```

---

## Tech stack

- Python 3.11+
- `paho-mqtt` for MQTT; Eclipse Mosquitto as the local broker
- **SQLite** for storage in this phase (migrate to TimescaleDB after the review;
  keep all DB access inside `db/repo.py` so the swap touches one file)
- `pandas`, `numpy`, `scipy` for processing
- `scikit-learn`, `xgboost` for ML; `shap` for explainability
- `fastapi` + `uvicorn` for the API
- `streamlit` + `plotly` for the dashboard
- `pytest` for tests
- `PyYAML` for config
- Arduino / PlatformIO for ESP32 firmware (later phase)

Do NOT introduce Docker, Kubernetes, React, Grafana, Node-RED, Kafka, InfluxDB,
Redis or any cloud service in this phase. Time is short. If you believe one is
genuinely needed, say so and wait for approval rather than adding it.

---

## Repository layout

```
textile-ohs/
|-- CLAUDE.md
|-- DAY_1.md, DAY_2.md          # task briefs - follow the current one
|-- requirements.txt
|-- config.py                   # loader for config.yaml (raises on missing keys)
|-- shifts.py                   # shift and break calendar (simulator AND backend)
|-- config.yaml                 # single source of truth: zones, workers, machines, limits
|-- docs/
|   |-- data_contract.md        # MQTT topics + JSON schema - AUTHORITATIVE
|   |-- exposure_methods.md     # formulas, standards, worked examples
|   +-- review_notes.md         # running log of what is built, for the review
|-- simulator/
|   |-- faults.py               # sensor-fault episodes for machine nodes
|   |-- profiles.py             # per-zone hazard distributions
|   |-- worker_model.py         # shifts, zone movement, breaks, PPE
|   |-- machine_model.py        # ON/OFF cycles, vibration, slow degradation
|   +-- publisher.py            # the ONLY file in simulator/ that talks to MQTT
|-- ingest/
|   |-- subscriber.py           # MQTT -> validate -> db.repo
|   |-- localisation.py         # BLE sightings -> worker zone intervals (separate pass)
|   |-- sessions.py             # gate events -> worker sessions; clips assignments
|   +-- validation.py           # range, stuck-value, spike, staleness checks
|-- db/
|   |-- schema.sql
|   +-- repo.py                 # ALL SQL lives here; no SQL anywhere else
|-- exposure/
|   |-- noise.py                # L_Aeq, L_EX,8h, dose %
|   |-- dust.py                 # TWA, RH correction
|   |-- heat.py                 # WBGT
|   +-- cumulative.py           # rolling 7-day / 30-day per worker
|-- ml/
|   |-- features.py
|   |-- forecast.py             # Task A - hazard forecasting (HEADLINE RESULT)
|   |-- risk.py                 # Task B - risk scoring
|   +-- explain.py              # SHAP
|-- api/
|   +-- main.py
|-- dashboard/
|   +-- app.py
|-- tests/
|-- firmware/                   # ESP32 sketches (stub for now)
+-- scripts/
    |-- score_localisation.py   # localisation accuracy vs simulator ground truth
    +-- run_all.sh
```

---

## Non-negotiable engineering rules

1. **dB is logarithmic.** Never average sound levels arithmetically. Always use
   energy averaging: `L_Aeq = 10 * log10(mean(10 ** (L / 10)))`. A reviewer will
   check this specific thing.
2. **Every function in `exposure/` cites its standard in the docstring** (ISO 9612
   for noise, ISO 7243 for WBGT, NIOSH 98-126 for dose criteria) and has a unit test
   in `tests/` with a hand-calculated expected value.
3. **All SQL lives in `db/repo.py`.** No raw SQL in dashboards, API or ML code.
4. **All constants live in `config.yaml`.** No magic numbers for zone names, exposure
   limits, sampling rates or thresholds anywhere else in the code.
5. **Type hints on every function.** Docstrings on every public function.
6. **Timestamps are UTC ISO-8601 with a `Z` suffix internally.** Convert to
   Asia/Kolkata only at the display layer.
7. **Never install a package that is not in `requirements.txt`.** Add it there first.
8. **Fail loudly on bad data.** Log and reject invalid records; never silently
   substitute defaults.
9. **No network calls at runtime** other than MQTT to the local broker.

---

## Scientific integrity rules (these protect the paper)

Every value the system handles must be labelled as exactly one of:

| Label | Meaning |
|---|---|
| `measured_environmental` | Directly measured by a fixed zone sensor |
| `estimated_personal` | Zone value weighted by the worker's time in that zone |
| `measured_personal` | From a wearable sensor (optional prototype) |
| `predicted_risk` | ML model output |

- **The system never diagnoses disease.** It estimates *exposure-related risk*.
  No function, variable, log line, UI string or docstring may say or imply that the
  system detects hearing loss, byssinosis or any medical condition.
- **PM readings from low-cost optical sensors are indicative, not regulatory.**
  Never compare them directly against legal cotton dust limits (those require
  gravimetric sampling). Use them for trends, zone comparison and ML features.
- **Do not train an ML risk model on labels generated by the same formula the
  simulator uses.** That only re-learns the generator and produces a meaningless
  ~99% accuracy. Forecasting (Task A) is the headline ML result because it is
  legitimate on simulated data. For risk scoring (Task B), label the output
  explicitly as standards-based and state the limitation in the UI and the report.
- **Time-series splits are chronological, never randomly shuffled**, and are grouped
  by worker so the same worker never appears in both train and test.
- Anywhere the dashboard shows data in this phase, it must carry a visible
  `SIMULATED DATA` badge. Do not hide this.

---

## Working style for Claude Code

- Work **one module at a time**. Finish it, test it, commit it, then move on.
- After writing any module, **run its tests and show the output**.
- **Commit after every working piece** with a clear message.
- When something is ambiguous, prefer the simplest option that satisfies
  `docs/data_contract.md`; if it is still ambiguous, ask rather than assume.
- Keep `docs/review_notes.md` updated with what was built and what is pending.
- Prefer small, readable functions over clever ones. Examiners read this code.
- Never present placeholder or TODO stubs as finished work. If something is
  incomplete, say so explicitly in your response.
- Do not refactor or rewrite modules that already pass their tests unless asked.

---

## Domain reference (textile mill zones)

| Zone ID | Zone | Noise (dBA) | PM2.5 (ug/m3) | Notes |
|---|---|---|---|---|
| Z1 | Blowroom | 88-95 | 150-400 | Very dusty |
| Z2 | Carding | 85-92 | 200-500 | Highest dust |
| Z3 | Drawing/Combing | 82-88 | 80-200 | |
| Z4 | Ring Spinning | 92-99 | 60-150 | High noise, high RH |
| Z5 | Weaving/Power Loom | 95-102 | 50-120 | Highest noise |
| Z6 | Winding/Packing | 85-90 | 40-100 | |
| Z7 | Office/Store | 50-60 | 10-30 | Low-exposure control zone |

Mills deliberately run high humidity (50-85% RH) for yarn quality, at 28-38 C.
These conditions are exactly why DHT11 and bare optical PM sensors were replaced
in the hardware design, and why WBGT matters more than temperature alone.

---

## Exposure limits (keep these in config.yaml, not in code)

- NIOSH: 85 dBA criterion, 8 h, **3 dB exchange rate**
- India (Factories Act / state rules): 90 dBA criterion, 8 h. Verify the exchange
  rate against the current Tamil Nadu Factories Rules before final submission.
- The dashboard shows **both** dose figures side by side.
- WBGT action limits per ISO 7243 depend on metabolic rate. Use the moderate-work
  category and state the assumption wherever the value is shown.
- PPE use is recorded as a model feature but **must not** automatically reduce the
  calculated dose, because real-world earplug attenuation is unreliable.

---

## What this phase must produce (first review deliverable)

A live demo that runs end to end without any hardware:

1. Simulator replays a full 8-hour shift in about 5 minutes over MQTT
2. Data lands in SQLite through the real ingest path
3. Dashboard shows live zones, factory heat map, worker drilldown, alerts
4. `pytest` passes, with hand-verified exposure calculations
5. Forecast chart: predicted vs actual noise, compared against a persistence baseline
6. SHAP explanation for one worker's risk score
7. `docs/data_contract.md` plus an ESP32 firmware stub, showing hardware drop-in
   readiness