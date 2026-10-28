-- Rows for a Day-2 (schema v1) database, used to test the v2 migration.
-- Deliberately inserted out of time order, including a fractional-second ts that
-- sorts wrongly as text ('.' < 'Z').
INSERT INTO zone_env (topic, device_id, ts, received_ts, quality, flags, schema_version, zone_id,
    noise_dba, noise_peak_dba, pm25_ugm3, pm10_ugm3, temp_c, rh_pct, globe_temp_c, co2_ppm,
    flag_noise, flag_pm, flag_temp_rh, flag_globe, flag_co2) VALUES
 ('textile/zone/Z1/env', 'ZN-Z1', '2026-10-28T00:00:11Z',     '2026-10-28T06:00:00.003Z', 'ok', '[]', '1.1', 'Z1', 92.0, 96.0, 250.0, 500.0, 33.0, 65.0, 35.5, 760, 'ok', 'ok', 'ok', 'ok', 'ok'),
 ('textile/zone/Z1/env', 'ZN-Z1', '2026-10-28T00:00:10.500Z', '2026-10-28T06:00:00.002Z', 'ok', '[]', '1.1', 'Z1', 91.5, 95.5, 250.0, 500.0, 33.0, 65.0, 35.5, 760, 'ok', 'ok', 'ok', 'ok', 'ok'),
 ('textile/zone/Z1/env', 'ZN-Z1', '2026-10-28T00:00:10Z',     '2026-10-28T06:00:00.001Z', 'ok', '[]', '1.1', 'Z1', 91.0, 95.0, 250.0, 500.0, 33.0, 65.0, 35.5, 760, 'ok', 'ok', 'ok', 'ok', 'ok');
INSERT INTO device_status (topic, device_id, ts, received_ts, quality, flags, schema_version, online) VALUES
 ('textile/status/ZN-Z1', 'ZN-Z1', NULL, '2026-10-28T06:00:01.000Z', 'ok', '[]', '1.1', 0);
INSERT INTO zone_assignments (worker_id, tag_id, zone_id, from_ts, to_ts, confidence, method, label, outside_session, params) VALUES
 ('W101', 'BLE-0001', 'Z1', '2026-10-28T00:00:00Z', '2026-10-28T00:30:00Z', 0.9, 'ble_ewma_hysteresis', 'estimated_personal', 0, '{}');
INSERT INTO worker_sessions (worker_id, session_start, session_end, duration_s, shift, flags) VALUES
 ('W101', '2026-10-28T00:20:00Z', '2026-10-28T08:35:00Z', 29700, 'A', '[]');
