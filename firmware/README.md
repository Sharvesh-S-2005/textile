# firmware/

ESP32 sketches (Arduino). **The backend requires no changes when this replaces the
simulator, because both publish contract v1.1** (`docs/data_contract.md`).

**Status: stub. Not compiled; never flashed or run on hardware; sensor reads are
stubs.** `zone_node/zone_node.ino` has not been compiled or run on a board (see
"Compile check" below for the attempt and how to run it). Sensor reads return fixed values. The sketch shows that the
contract can be implemented on the node: same topics, same JSON fields, the LWT
registered retained with no `ts`, the birth/status message, and the microSD
buffer-and-replay path (commented stub). The wiring below is the planned assignment
from the component selection and has **not been verified on hardware**.

## Zone node — ESP32 DevKit V1 (WROOM-32)

| Function | Part | Interface | Pins |
|---|---|---|---|
| Sound level | DFRobot Gravity SEN0232 | Analog | GPIO34 (ADC1_CH6, input-only) |
| PM2.5 / PM10 | Plantower PMS7003 | UART2 | RX=GPIO16, TX=GPIO17 |
| CO2 | Winsen MH-Z19C | UART1 | RX=GPIO27, TX=GPIO14 |
| Temp / RH | Sensirion SHT31 (0x44) | I2C | SDA=GPIO21, SCL=GPIO22 |
| Globe temp (WBGT) | DS18B20 in black globe | 1-Wire | GPIO4, 4.7 kΩ pull-up |
| RTC | DS3231 (0x68) | I2C | shared GPIO21/22 |
| OLED | SSD1306 0.96" (0x3C) | I2C | shared GPIO21/22 |
| Data buffer | microSD module | VSPI | MOSI=23, MISO=19, SCK=18, CS=5 |
| Alarm | Active buzzer | GPIO | GPIO13 |
| Status LED | LED | GPIO | GPIO33 |
| Worker scan | ESP32 internal BLE | — | — |

## Machine node

ADXL345 accelerometer (I2C 0x53) and ADS1115 ADC (I2C 0x48) reading an SCT-013 clamp,
both on the GPIO21/22 bus. No sketch yet; it publishes contract §6
(`textile/machine/{machine_id}/status`).

## Constraints

- **ADC1 only.** ADC2 cannot be read while Wi-Fi is active on the ESP32.
- Avoid GPIO6–11 (SPI flash) and the strapping pins GPIO0/2/12/15.
- The SEN0232 outputs 0.6–2.6 V for 30–130 dBA, but the ESP32 ADC is non-linear, so
  **calibration against the handheld sound-level meter is mandatory**, not optional.
- The PMS7003 and MH-Z19C need a 5 V supply with 3.3 V logic and draw about 100 mA
  and 150 mA at peak. Use a proper 5 V adapter, not laptop USB.
- PM values are published raw. Humidity correction happens in the backend
  (`exposure/dust.py`), never on the device.
- The DS3231 is set to UTC; it is the authority for `ts`.

## Libraries

PubSubClient, ArduinoJson (v7), Adafruit_SHT31, OneWire + DallasTemperature, MH-Z19,
PMS, RTClib, SD, Adafruit_SSD1306 + Adafruit_GFX, NimBLE-Arduino.

## Compile check (Day 8 Task 3): attempted, NOT done

**2026-10-29: blocked.** The development session was not permitted to download and run
the `arduino-cli` toolchain, so the sketch has **not** been compiled. A manual review of
the sketch against the library APIs found no mismatch. That covers PubSubClient
`connect(id, willTopic, willQos, willRetain, willMessage)`, `publish(topic, payload,
retained)` and `setBufferSize`; ArduinoJson v7 `JsonDocument`, `doc[k].to<JsonObject>()`,
`JsonArray::add<JsonObject>()`, `serialized()` and `serializeJson(doc, buf, size)`; and
RTClib `RTC_DS3231` / `DateTime`. **A review is not a compile.**

The sketch includes only `WiFi.h`, `PubSubClient.h`, `ArduinoJson.h`, `Wire.h` and
`RTClib.h`; the sensor libraries are commented out until real reads replace the stubs.
To run the check (no board needed), from the repository root:

```bash
arduino-cli version
arduino-cli config init
arduino-cli config add board_manager.additional_urls https://espressif.github.io/arduino-esp32/package_esp32_index.json
arduino-cli core update-index
arduino-cli core install esp32:esp32
arduino-cli lib install PubSubClient ArduinoJson RTClib     # RTClib pulls in Adafruit BusIO
arduino-cli compile --fqbn esp32:esp32:esp32 firmware/zone_node
arduino-cli core list && arduino-cli lib list               # record these versions here
```

Then record here: the `arduino-cli`, `esp32:esp32` core and library versions; the
"Sketch uses N bytes (P%) of program storage space. Maximum is M bytes." line; and the
free-space margin (M − N). If it compiles, the status line above becomes exactly:
*compiles clean for ESP32; never flashed or run on hardware; sensor reads are stubs.*
Not "tested".

## Open items

- **QoS.** PubSubClient publishes at QoS 0 only, but the contract specifies QoS 1 for
  telemetry. Either use a client that supports QoS 1 or accept QoS 0 with the microSD
  buffer as the delivery guarantee. This needs a decision before deployment.
- Real sensor reads, BLE scanning, microSD buffering and the machine-node sketch.
- First compile (commands above), then a bench test against the local broker and `ingest/subscriber.py`,
  checking that the rejected-records count stays at zero.
