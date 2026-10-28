# Demo script — first review

Total: about **15 minutes**. Every command is run from the repository root
(`D:\textile-ohs`) in a terminal with the venv active (`.venv\Scripts\activate`).
Open `docs/numbers_sheet.md` (printed) beside the laptop before starting.

Two databases are involved:

| Database | Used for | Needs broker / simulator? |
|---|---|---|
| `demo/textile_ohs_demo.db` (committed, 44 MB) | Worker Drilldown, System & Results — 7 days of derived results, raw telemetry for 2026-10-26 | **No** |
| `data/textile_ohs.db` (created live) | Live Zones while the simulator is running | Yes |

## 0. Before the panel arrives (5 min, not shown)

```
del data\textile_ohs.db data\textile_ohs.db-wal data\textile_ohs.db-shm
streamlit run dashboard/app.py
```

Expected: browser opens on **Worker Drilldown**, header shows the red `SIMULATED DATA`
badge and "(read-only)", default worker **W102** on 2026-10-26. Leave this tab open — it
is the fallback for everything below (step F).

## 1. Broker (30 s)

```
sc query mosquitto
```

Expected: `STATE : 4 RUNNING`. If stopped: `net start mosquitto` (admin terminal).
Say: "This is a standard MQTT broker. The ESP32 nodes will connect to it exactly like the simulator does."

## 2. Ingest (30 s) — terminal A

```
python -m ingest.subscriber --time-reference stream
```

Expected: `connected; subscribing to …`, then a stats line every few seconds
(received / stored / rejected). It creates `data/textile_ohs.db` if missing.
Say: "Ingest validates every message against the contract and rejects bad records loudly."

## 3. Simulator (5 min) — terminal B

```
python -m simulator.publisher --speed 100 --align-shifts
```

Expected: one 8-hour shift replayed in ~5 minutes (~170 msg/s). Terminal A's
`stored` count climbs; `rejected` stays at 0 unless `python scripts/inject_bad_records.py`
is run in terminal C (13 records, all rejected, reasons logged).
Midway the configured zone-node outage fires: the device goes `offline` via its LWT.
Say: "Nothing downstream of MQTT knows this is simulated. When the hardware arrives,
this process is replaced by the ESP32s and nothing else changes."

## 4. Worker Drilldown (3 min) — demo DB tab from step 0

Walk through W102, then W105:
- tiles: L_EX,8h (ISO 9612), NIOSH | India dose side by side, coverage, unaccounted time;
- the "lower bound" data-quality sentence;
- zone timeline with energy-averaged noise overlay (glitch readings excluded);
- risk index panel — say the caveat aloud: *a weighted design choice, not a validated clinical model*;
- cumulative 7-day / 30-day table.

## 5. Live Zones (2 min) — live DB, second dashboard — terminal C

```
streamlit run dashboard/app.py --server.port 8502 -- --db data/textile_ohs.db
```

Open "Live Zones" in the new tab. Expected: seven zone tiles updating every ~10 s,
the coupling chart with machines-stopped shading and dashed break bands. The outage
zone shows as offline during the outage window.
If the live run is not producing data, use the Live Zones page of the demo tab instead
(2026-10-26 has raw telemetry).

## 6. Tests (1 min) — any terminal, after the simulator finishes

```
python -m pytest -q
```

Expected: `380 passed` (number from the numbers sheet), exit code 0. Point at
`tests/test_noise.py` hand-calculated cases if asked.

## 7. Hardware readiness (2 min)

Open `docs/data_contract.md` (topics, JSON schema, contract v1.1: `event_id`, LWT without `ts`)
and `firmware/`. Say: "The firmware's only job is to publish these exact messages.
The firmware is a stub until the hardware arrives — it has not been run on a board."

## Optional — derive a live shift (1 min)

```
python -m exposure.engine --db data/textile_ohs.db --resolve --date <shift date from terminal B>
```

Then reload the step-5 tab and open Worker Drilldown on that date.

## F. Fallback — no broker, no simulator, no network

If anything in steps 1–3 or 5 fails, stop it (Ctrl+C) and continue on the step-0 tab:

```
streamlit run dashboard/app.py
```

All three pages read only `demo/textile_ohs_demo.db`. The dashboard makes no network
calls (`.streamlit/config.toml` disables usage stats and binds to localhost).

### Rehearsal with the network disabled (do this once before the review)

1. Turn Wi-Fi off / unplug Ethernet (aeroplane mode).
2. Make sure no `python` processes are running (Task Manager).
3. `streamlit run dashboard/app.py`, then open each of the three pages.
4. Expected: every page renders with the `SIMULATED DATA` badge; no error banners.

This rehearsal has to be done by the presenter: the development session could not
turn the network off. The dashboard-only path (no broker, no simulator, no other
Python process) was rehearsed on 2026-10-28.
