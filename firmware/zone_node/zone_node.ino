// zone_node.ino - ESP32 zone node, data contract v1.1 (docs/data_contract.md)
//
// STUB. Not compiled and not run on hardware. Every sensor read returns a fixed
// value; the comment on each read names the part and library it will use. What this
// sketch does show is that the contract is implementable on the node: the same
// topics, the same JSON fields, the same LWT (no "ts"), the same status message.
// The backend requires no changes when this replaces simulator/publisher.py.
//
// Board: ESP32 DevKit V1 (WROOM-32). Pin assignments are in firmware/README.md.
//
// Pin rules for this board:
//   - Analog inputs on ADC1 only. ADC2 cannot be read while Wi-Fi is active.
//   - Never use GPIO6-11 (SPI flash) or the strapping pins GPIO0/2/12/15.
//
// Libraries: PubSubClient, ArduinoJson (v7), Adafruit_SHT31, OneWire,
// DallasTemperature, MH-Z19, PMS, RTClib, SD, NimBLE-Arduino.

#include <WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>
#include <Wire.h>
#include <RTClib.h>
// #include <Adafruit_SHT31.h>      // temp / RH
// #include <OneWire.h>             // globe probe
// #include <DallasTemperature.h>
// #include <MHZ19.h>               // CO2
// #include <PMS.h>                 // PM2.5 / PM10
// #include <SD.h>                  // microSD buffer
// #include <NimBLEDevice.h>        // worker tag scan

// ---------------------------------------------------------------- identity
// One sketch per zone: change these two lines per node. Client ID = device ID (contract 1).
#define ZONE_ID   "Z4"
#define DEVICE_ID "ZN-Z4"
#define FIRMWARE  "zn-0.1.0"
#define SCHEMA_VERSION "1.1"

// Placeholders only. Never commit real credentials.
const char* WIFI_SSID = "CHANGE_ME";
const char* WIFI_PASS = "CHANGE_ME";
const char* MQTT_HOST = "192.168.1.10";   // laptop running Mosquitto
const uint16_t MQTT_PORT = 1883;

// ---------------------------------------------------------------- pins
#define PIN_SOUND      34   // SEN0232 analog out, ADC1_CH6 (input-only pin)
#define PIN_PMS_RX     16   // UART2 <- PMS7003 TX
#define PIN_PMS_TX     17   // UART2 -> PMS7003 RX
#define PIN_CO2_RX     27   // UART1 <- MH-Z19C TX
#define PIN_CO2_TX     14   // UART1 -> MH-Z19C RX
#define PIN_I2C_SDA    21   // SHT31 0x44, DS3231 0x68, SSD1306 0x3C
#define PIN_I2C_SCL    22
#define PIN_ONEWIRE     4   // DS18B20 in black globe, 4.7 kOhm pull-up
#define PIN_SD_CS       5   // microSD on VSPI: MOSI 23, MISO 19, SCK 18
#define PIN_BUZZER     13
#define PIN_STATUS_LED 33

// ---------------------------------------------------------------- rates (contract 2)
const uint32_t ENV_PERIOD_MS = 10000;   // textile/zone/{zone_id}/env, 1 msg / 10 s
const uint32_t BLE_PERIOD_MS = 30000;   // textile/zone/{zone_id}/ble, 1 msg / 30 s
const uint8_t  BLE_SCAN_WINDOW_S = 30;

// ---------------------------------------------------------------- topics (contract 2)
const char* TOPIC_ENV    = "textile/zone/" ZONE_ID "/env";
const char* TOPIC_BLE    = "textile/zone/" ZONE_ID "/ble";
const char* TOPIC_STATUS = "textile/status/" DEVICE_ID;

// LWT (contract 1, 3): retained, and deliberately WITHOUT "ts". It is registered at
// connect time and cannot know when the disconnect will happen; the backend stores
// it with ts NULL plus its own received_ts.
const char* LWT_PAYLOAD =
    "{\"schema_version\":\"" SCHEMA_VERSION "\",\"device_id\":\"" DEVICE_ID "\",\"online\":false}";

WiFiClient wifi;
PubSubClient mqtt(wifi);
RTC_DS3231 rtc;

uint32_t lastEnvMs = 0;
uint32_t lastBleMs = 0;
uint32_t lastReconnectMs = 0;
const uint32_t RECONNECT_PERIOD_MS = 5000;
uint32_t bufferedRecords = 0;   // records on microSD awaiting upload (contract 9)

// ---------------------------------------------------------------- timestamp
// ts is UTC ISO-8601 with Z, from the DS3231. The RTC is the authority for ts
// (contract 3) and is set to UTC once, at commissioning - never to local time.
void isoTimestamp(char* out, size_t len) {
  DateTime now = rtc.now();
  snprintf(out, len, "%04d-%02d-%02dT%02d:%02d:%02dZ",
           now.year(), now.month(), now.day(), now.hour(), now.minute(), now.second());
}

// ---------------------------------------------------------------- sensor stubs
// Each returns a FIXED value. Replace the body, keep the signature. A failed read
// sets its flag to "error" or "stale" and the field is published as null (contract 4).

struct Reading {
  float value;
  const char* flag;   // "ok", "stale", "error", "uncalibrated", "needs_clean"
};

// DFRobot Gravity SEN0232, analogRead(PIN_SOUND). Output 0.6-2.6 V for 30-130 dBA,
// i.e. dBA = volts * 50. The ESP32 ADC is non-linear: calibration against the
// handheld sound-level meter is mandatory before any reading is used.
// noise_dba is L_Aeq over the 10 s interval: energy-average the samples,
// 10*log10(mean(10^(L/10))), never an arithmetic mean of dB.
Reading readNoiseLaeq() { return {96.4f, "ok"}; }
Reading readNoisePeak() { return {103.1f, "ok"}; }   // max sample in the interval

// Plantower PMS7003 on UART2 (PMS library). Raw, UNCORRECTED values: humidity
// correction is done in the backend (exposure/dust.py), never on the device.
// Needs a 5 V supply (3.3 V logic), ~100 mA peak.
Reading readPm25() { return {118.2f, "ok"}; }
Reading readPm10() { return {240.7f, "ok"}; }

// Sensirion SHT31 at 0x44 on I2C (Adafruit_SHT31).
Reading readTempC() { return {33.8f, "ok"}; }
Reading readRhPct() { return {72.5f, "ok"}; }

// DS18B20 inside a black globe on 1-Wire GPIO4 (OneWire + DallasTemperature).
Reading readGlobeC() { return {36.2f, "ok"}; }

// Winsen MH-Z19C on UART1 (MH-Z19 library). 5 V supply, ~150 mA peak.
// Use a proper 5 V adapter for the node, not laptop USB.
Reading readCo2Ppm() { return {980.0f, "ok"}; }

// ---------------------------------------------------------------- JSON helpers
// Publish the value rounded to one decimal, or null when the sensor is not ok.
void setField(JsonDocument& doc, const char* key, const Reading& r) {
  if (strcmp(r.flag, "ok") == 0) {
    doc[key] = serialized(String(r.value, 1));
  } else {
    doc[key] = nullptr;
  }
}

// ---------------------------------------------------------------- microSD buffer (stub)
// Contract 11: when the broker is unreachable, append each record to microSD and
// replay it on reconnect with its ORIGINAL ts. The backend deduplicates on
// (device_id, ts, topic) and never assumes arrival order.
void bufferToSd(const char* topic, const char* payload) {
  // File f = SD.open("/buffer.jsonl", FILE_APPEND);
  // f.print(topic); f.print('\t'); f.println(payload);   // one record per line
  // f.close();
  bufferedRecords++;
}

void replaySdBuffer() {
  // File f = SD.open("/buffer.jsonl", FILE_READ);
  // while (f.available()) {
  //   String line = f.readStringUntil('\n');
  //   int tab = line.indexOf('\t');
  //   mqtt.publish(line.substring(0, tab).c_str(), line.substring(tab + 1).c_str());
  //   bufferedRecords--;
  // }
  // f.close();
  // SD.remove("/buffer.jsonl");
}

void publishOrBuffer(const char* topic, const char* payload) {
  if (mqtt.connected() && mqtt.publish(topic, payload, false)) {
    return;
  }
  bufferToSd(topic, payload);
}

// ---------------------------------------------------------------- messages
// Contract 9: retained birth message, WITH ts.
void publishBirth() {
  JsonDocument doc;
  char ts[24];
  isoTimestamp(ts, sizeof(ts));
  doc["schema_version"] = SCHEMA_VERSION;
  doc["device_id"] = DEVICE_ID;
  doc["ts"] = ts;
  doc["online"] = true;
  doc["firmware"] = FIRMWARE;
  doc["uptime_s"] = (uint32_t)(millis() / 1000);
  doc["buffered_records"] = bufferedRecords;
  char out[256];
  serializeJson(doc, out, sizeof(out));
  mqtt.publish(TOPIC_STATUS, out, true);   // retained
}

// Contract 4: zone environmental telemetry.
void publishEnv() {
  Reading noise = readNoiseLaeq(), peak = readNoisePeak();
  Reading pm25 = readPm25(), pm10 = readPm10();
  Reading temp = readTempC(), rh = readRhPct();
  Reading globe = readGlobeC(), co2 = readCo2Ppm();

  JsonDocument doc;
  char ts[24];
  isoTimestamp(ts, sizeof(ts));
  doc["schema_version"] = SCHEMA_VERSION;
  doc["device_id"] = DEVICE_ID;
  doc["ts"] = ts;
  doc["zone_id"] = ZONE_ID;
  setField(doc, "noise_dba", noise);
  setField(doc, "noise_peak_dba", peak);
  setField(doc, "pm25_ugm3", pm25);
  setField(doc, "pm10_ugm3", pm10);
  setField(doc, "temp_c", temp);
  setField(doc, "rh_pct", rh);
  setField(doc, "globe_temp_c", globe);
  if (strcmp(co2.flag, "ok") == 0) {
    doc["co2_ppm"] = (int)lroundf(co2.value);   // integer field
  } else {
    doc["co2_ppm"] = nullptr;
  }

  JsonObject flags = doc["sensor_flags"].to<JsonObject>();
  flags["noise"] = noise.flag;   // one flag covers noise_dba and noise_peak_dba
  flags["pm"] = pm25.flag;
  flags["temp_rh"] = temp.flag;
  flags["globe"] = globe.flag;
  flags["co2"] = co2.flag;

  char out[512];
  serializeJson(doc, out, sizeof(out));
  publishOrBuffer(TOPIC_ENV, out);
}

// Contract 5: every worker tag seen in the scan window, with RSSI. Zone assignment
// is a backend decision; the node only reports what it hears.
void publishBle() {
  JsonDocument doc;
  char ts[24];
  isoTimestamp(ts, sizeof(ts));
  doc["schema_version"] = SCHEMA_VERSION;
  doc["device_id"] = DEVICE_ID;
  doc["ts"] = ts;
  doc["zone_id"] = ZONE_ID;
  doc["scan_window_s"] = BLE_SCAN_WINDOW_S;
  JsonArray tags = doc["tags"].to<JsonArray>();   // an empty array is valid

  // STUB. Real path (NimBLE-Arduino): passive scan for BLE_SCAN_WINDOW_S, keep
  // advertisements whose name matches "BLE-nnnn", and per tag record the mean
  // RSSI (int dBm) and the number of advertisements seen.
  JsonObject tag = tags.add<JsonObject>();
  tag["tag_id"] = "BLE-0007";
  tag["rssi_dbm"] = -58;
  tag["seen_count"] = 11;

  char out[1024];
  serializeJson(doc, out, sizeof(out));
  publishOrBuffer(TOPIC_BLE, out);
}

// ---------------------------------------------------------------- connection
void connectWifi() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
  }
}

// Note: PubSubClient publishes at QoS 0 only, while the contract asks for QoS 1 on
// telemetry. Either swap to a QoS 1 client or accept QoS 0 with the microSD buffer
// as the delivery guarantee - an open decision, see firmware/README.md.
bool connectMqtt() {
  // connect(clientId, willTopic, willQos, willRetain, willMessage)
  if (!mqtt.connect(DEVICE_ID, TOPIC_STATUS, 1, true, LWT_PAYLOAD)) {
    return false;
  }
  publishBirth();
  replaySdBuffer();
  publishBirth();   // second birth reports buffered_records after the replay
  return true;
}

// ---------------------------------------------------------------- main
void setup() {
  Serial.begin(115200);
  pinMode(PIN_STATUS_LED, OUTPUT);
  Wire.begin(PIN_I2C_SDA, PIN_I2C_SCL);
  rtc.begin();
  // Serial2.begin(9600, SERIAL_8N1, PIN_PMS_RX, PIN_PMS_TX);   // PMS7003
  // Serial1.begin(9600, SERIAL_8N1, PIN_CO2_RX, PIN_CO2_TX);   // MH-Z19C
  // SD.begin(PIN_SD_CS);
  // NimBLEDevice::init("");

  connectWifi();
  mqtt.setServer(MQTT_HOST, MQTT_PORT);
  mqtt.setBufferSize(1024);   // default 256 B is smaller than one env payload
  connectMqtt();
}

void loop() {
  if (!mqtt.connected()) {
    digitalWrite(PIN_STATUS_LED, LOW);
    // Never block here: sampling and microSD buffering must continue offline.
    if (millis() - lastReconnectMs >= RECONNECT_PERIOD_MS) {
      lastReconnectMs = millis();
      if (WiFi.status() != WL_CONNECTED) {
        WiFi.reconnect();
      } else {
        connectMqtt();
      }
    }
  } else {
    digitalWrite(PIN_STATUS_LED, HIGH);
  }
  mqtt.loop();

  uint32_t now = millis();
  if (now - lastEnvMs >= ENV_PERIOD_MS) {
    lastEnvMs = now;
    publishEnv();
  }
  if (now - lastBleMs >= BLE_PERIOD_MS) {
    lastBleMs = now;
    publishBle();
  }
}
