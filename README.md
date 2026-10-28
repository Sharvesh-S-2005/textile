# AI-IoT Textile Occupational Health Monitoring

An IoT and machine-learning system that monitors noise, cotton dust (PM2.5/PM10),
heat stress (WBGT), temperature, humidity and CO2 across textile mill zones, tracks
which worker is in which zone (BLE zone-level localisation plus RFID shift
check-in), monitors machine status and vibration, and estimates each worker's shift
and cumulative exposure against occupational hygiene standards (ISO 9612, ISO 7243,
NIOSH 98-126). It forecasts hazard levels and estimates exposure-related
occupational health risk; it does **not** diagnose any medical condition. In the
current phase all data comes from a simulator that publishes over MQTT in the format
defined in `docs/data_contract.md`; ESP32 firmware will later replace it with no
backend changes.

## 1. Install the MQTT broker (Eclipse Mosquitto)

- **Windows:** download the installer from <https://mosquitto.org/download/>, install,
  then run `mosquitto -v` from the install folder (or start the "Mosquitto Broker"
  service).
- **Ubuntu/Debian:** `sudo apt install mosquitto mosquitto-clients`
- **macOS:** `brew install mosquitto`

The broker listens on `localhost:1883` by default.

## 2. Create a virtual environment (Python 3.11+)

```bash
python -m venv .venv
# Windows:      .venv\Scripts\activate
# Linux/macOS:  source .venv/bin/activate
```

## 3. Install requirements

```bash
pip install -r requirements.txt
```

## 4. Run the simulator

```bash
# Print payloads without a broker:
python -m simulator.publisher --dry-run --duration-h 0.1

# Publish a full 8-hour shift in about 5 minutes:
python -m simulator.publisher --speed 100 --duration-h 8

# In another terminal, watch the messages:
mosquitto_sub -h localhost -t "textile/#" -v
```

## 5. Ingest into SQLite, resolve zones, score

```bash
python -m db.init_db                                   # create data/textile_ohs.db
# Start ingest first; "stream" time reference is for accelerated/replayed data
# (real hardware uses the default, wall clock):
python -m ingest.subscriber --time-reference stream --idle-exit-s 20
# Then, in another terminal:
python -m simulator.publisher --speed 100 --duration-h 8 --start 2026-10-28T05:30:00 \
    --ground-truth-out data/truth.jsonl
# After it finishes:
python -m ingest.localisation                          # BLE sightings -> zone intervals
python -m ingest.sessions                              # gate events -> sessions, clip
python scripts/score_localisation.py --truth data/truth.jsonl
```

## 6. Run the tests

```bash
pytest
```
