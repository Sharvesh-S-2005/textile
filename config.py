"""Loader for config.yaml, the single source of truth for all constants.

The file is read once and cached. Every accessor raises ConfigError on a missing
key; nothing ever falls back to a silent default.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

CONFIG_PATH: Path = Path(__file__).resolve().parent / "config.yaml"

# Required keys per top-level section. List sections give the keys each item needs.
_REQUIRED_SECTIONS: dict[str, list[str]] = {
    "mqtt": ["host", "port", "keepalive_s", "qos", "topic_prefix", "client_id_prefix"],
    "simulation": [
        "schema_version", "env_interval_s", "ble_interval_s", "machine_interval_s",
        "speed_factor", "random_seed", "firmware_tag", "zone_dynamics",
        "machine_dynamics", "worker_dynamics", "ble_model", "outage",
    ],
    "shifts": ["timezone", "definitions"],
    "exposure_limits": [
        "criteria", "wbgt_limit_c", "wbgt_work_category", "rh_correction_threshold_pct",
    ],
    "validation": [
        "fields", "allowed_sensor_flags", "allowed_ppe_values", "allowed_rfid_events",
        "stuck_value_count", "future_ts_tolerance_s", "late_upload_threshold_s",
    ],
    "paths": ["db_path", "model_dir", "log_dir"],
    "ingest": [
        "client_id_suffix", "clean_session", "batch_size", "batch_interval_s", "max_unacked",
        "queue_poll_s",
        "reconnect_min_s", "reconnect_max_s", "stats_interval_s", "payload_excerpt_chars",
        "time_reference",
    ],
    "devices": ["gate", "ppe_app"],
    "layout": ["adjacency"],
}
_REQUIRED_LIST_ITEMS: dict[str, list[str]] = {
    "zones": ["zone_id", "name", "device_id", "profile"],
    "machines": [
        "machine_id", "zone_id", "device_id", "rated_current_a",
        "baseline_vib_rms_g", "degrading",
    ],
    "workers": [
        "worker_id", "name", "tag_id", "card_id", "home_zone", "shift", "ppe_compliance",
    ],
}
_REQUIRED_PROFILE_FIELDS: list[str] = [
    "noise_dba", "pm25_ugm3", "pm10_ugm3", "temp_c", "rh_pct", "co2_ppm",
]
_REQUIRED_CRITERION_KEYS: list[str] = [
    "criterion_db", "exchange_rate_db", "duration_h", "threshold_db", "source",
]
_REQUIRED_VALIDATION_FIELD_KEYS: list[str] = ["min", "max", "spike_threshold"]


class ConfigError(Exception):
    """Raised when config.yaml is missing a required key or is malformed."""


def _require(mapping: Any, keys: list[str], where: str) -> None:
    """Raise ConfigError unless `mapping` is a dict containing every key in `keys`."""
    if not isinstance(mapping, dict):
        raise ConfigError(f"{where}: expected a mapping, got {type(mapping).__name__}")
    missing = [k for k in keys if k not in mapping]
    if missing:
        raise ConfigError(f"{where}: missing required key(s) {missing}")


def validate_config(cfg: dict[str, Any]) -> None:
    """Check that every required section and key is present. Raises ConfigError."""
    _require(cfg, list(_REQUIRED_SECTIONS) + list(_REQUIRED_LIST_ITEMS), "config.yaml")
    for section, keys in _REQUIRED_SECTIONS.items():
        _require(cfg[section], keys, section)
    for section, keys in _REQUIRED_LIST_ITEMS.items():
        items = cfg[section]
        if not isinstance(items, list) or not items:
            raise ConfigError(f"{section}: expected a non-empty list")
        for i, item in enumerate(items):
            _require(item, keys, f"{section}[{i}]")

    for zone in cfg["zones"]:
        where = f"zones[{zone['zone_id']}].profile"
        _require(zone["profile"], _REQUIRED_PROFILE_FIELDS + ["globe_offset_c"], where)
        for field in _REQUIRED_PROFILE_FIELDS:
            _require(zone["profile"][field], ["mean", "sd"], f"{where}.{field}")

    for shift in cfg["shifts"]["definitions"]:
        _require(shift, ["shift_id", "start", "end", "breaks"], "shifts.definitions[]")
        for brk in shift["breaks"]:
            _require(brk, ["kind", "start", "duration_min"], f"shift {shift['shift_id']} break")

    for name, crit in cfg["exposure_limits"]["criteria"].items():
        _require(crit, _REQUIRED_CRITERION_KEYS, f"exposure_limits.criteria.{name}")

    for name, spec in cfg["validation"]["fields"].items():
        _require(spec, _REQUIRED_VALIDATION_FIELD_KEYS, f"validation.fields.{name}")


def load_config(path: Path | str = CONFIG_PATH) -> dict[str, Any]:
    """Read and validate a config file. Use `get_config()` for the cached project config."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    validate_config(cfg)
    return cfg


@lru_cache(maxsize=1)
def get_config() -> dict[str, Any]:
    """Return the project config, loaded and validated once per process."""
    return load_config(CONFIG_PATH)


def get_mqtt_config() -> dict[str, Any]:
    """Return the `mqtt` section (host, port, keepalive_s, qos, topic_prefix, ...)."""
    return get_config()["mqtt"]


def get_simulation_config() -> dict[str, Any]:
    """Return the `simulation` section (intervals, seed, schema_version, ...)."""
    return get_config()["simulation"]


def get_zones() -> list[dict[str, Any]]:
    """Return the list of zone definitions, each with its hazard `profile`."""
    return get_config()["zones"]


def get_zone(zone_id: str) -> dict[str, Any]:
    """Return one zone by ID. Raises ConfigError if it does not exist."""
    for zone in get_zones():
        if zone["zone_id"] == zone_id:
            return zone
    raise ConfigError(f"unknown zone_id: {zone_id}")


def get_machines() -> list[dict[str, Any]]:
    """Return the list of machine definitions."""
    return get_config()["machines"]


def get_workers() -> list[dict[str, Any]]:
    """Return the list of worker definitions."""
    return get_config()["workers"]


def get_shifts() -> dict[str, Any]:
    """Return the `shifts` section (timezone and per-shift definitions with breaks)."""
    return get_config()["shifts"]


def get_exposure_limits() -> dict[str, Any]:
    """Return the `exposure_limits` section (NIOSH/India criteria, WBGT limit, RH threshold)."""
    return get_config()["exposure_limits"]


def get_validation_config() -> dict[str, Any]:
    """Return the `validation` section (per-field ranges and spike thresholds)."""
    return get_config()["validation"]


def get_devices() -> dict[str, Any]:
    """Return the `devices` section (gate reader and supervisor PPE app IDs)."""
    return get_config()["devices"]


def get_adjacency() -> dict[str, list[str]]:
    """Return the zone adjacency map from the `layout` section."""
    return get_config()["layout"]["adjacency"]


def get_ingest_config() -> dict[str, Any]:
    """Return the `ingest` section (batching, reconnect backoff, time reference)."""
    return get_config()["ingest"]


def get_paths() -> dict[str, Any]:
    """Return the `paths` section (db_path, model_dir, log_dir), relative to the project root."""
    return get_config()["paths"]
