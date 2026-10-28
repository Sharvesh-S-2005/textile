"""Zone hazard model: environmental readings for one zone at one simulated instant.

Each quantity follows a discrete Ornstein-Uhlenbeck (AR(1)) update toward a target
that depends on machine load and time of day:

    value_next = value + theta * (target - value) + sigma * N(0, 1)
    theta = 1 - exp(-dt / tau)
    sigma = sd * sqrt(1 - (1 - theta) ** 2)     # keeps the stationary sd equal to `sd`

Noise targets are combined on an energy basis (dB is logarithmic), so with every
machine stopped the zone falls to its background level, not to some arithmetic
average.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np

import config
from simulator.faults import SilentFaults

SENSORS: tuple[str, ...] = ("noise", "pm", "temp_rh", "globe", "co2")
SENSOR_FIELDS: dict[str, tuple[str, ...]] = {
    "noise": ("noise_dba", "noise_peak_dba"),
    "pm": ("pm25_ugm3", "pm10_ugm3"),
    "temp_rh": ("temp_c", "rh_pct"),
    "globe": ("globe_temp_c",),
    "co2": ("co2_ppm",),
}


def _theta(dt_s: float, tau_s: float) -> float:
    """Fraction of the gap to target closed in dt_s for time constant tau_s."""
    return 1.0 - math.exp(-dt_s / tau_s)


def _event_probability(rate_per_h: float, dt_s: float) -> float:
    """Probability that a Poisson event with the given hourly rate occurs within dt_s."""
    return 1.0 - math.exp(-rate_per_h * dt_s / 3600.0)


def _db_to_energy(level_db: float) -> float:
    """Convert a sound level in dB to relative energy."""
    return 10.0 ** (level_db / 10.0)


def _energy_to_db(energy: float) -> float:
    """Convert relative sound energy to a level in dB."""
    return 10.0 * math.log10(energy)


class ZoneProfile:
    """Stateful generator of environmental readings for one zone.

    Call `reading()` with strictly increasing timestamps. The model keeps its own
    state between calls, so consecutive readings are autocorrelated.
    """

    def __init__(
        self,
        zone_cfg: dict[str, Any],
        dynamics: dict[str, Any],
        valid_ranges: dict[str, dict[str, float]],
        n_machines: int,
        timezone: str,
        default_dt_s: float,
        rng: np.random.Generator,
        start_ts: datetime,
    ) -> None:
        """Build a profile from a zone's config entry.

        Args:
            zone_cfg: one entry of `zones` in config.yaml.
            dynamics: `simulation.zone_dynamics` from config.yaml.
            valid_ranges: `validation.fields` from config.yaml (for output clamping).
            n_machines: number of machines located in this zone.
            timezone: local timezone name for the diurnal curves.
            default_dt_s: interval assumed before the first reading.
            rng: random generator owned by this profile.
            start_ts: simulated start time (UTC, timezone-aware).
        """
        self.zone_id: str = zone_cfg["zone_id"]
        self._p: dict[str, Any] = zone_cfg["profile"]
        self._d: dict[str, Any] = dynamics
        self._ranges = valid_ranges
        self._n_machines = n_machines
        self._tz = ZoneInfo(timezone)
        self._default_dt_s = default_dt_s
        self._rng = rng

        self._last_ts: datetime | None = None
        self._state: dict[str, float] = {}
        self._noise_transient_left = 0
        self._noise_boost_db = 0.0
        self._peak_boost_db = 0.0
        self._faults: dict[str, tuple[str, datetime] | None] = {s: None for s in SENSORS}

        drift = dynamics["pm_drift"]
        days_since_clean = rng.uniform(0.0, drift["initial_days_since_clean_max"])
        self._last_clean_ts: datetime = start_ts - timedelta(days=days_since_clean)
        self._needs_clean_since: datetime | None = None
        # Own stream, so silent-fault draws do not shift the measurement sequence.
        self._silent = SilentFaults(SENSOR_FIELDS, dynamics["silent_faults"], valid_ranges, rng.spawn(1)[0])

    @classmethod
    def from_config(
        cls, zone_id: str, rng: np.random.Generator, start_ts: datetime
    ) -> ZoneProfile:
        """Build the profile for `zone_id` using the project config.yaml."""
        n_machines = sum(1 for m in config.get_machines() if m["zone_id"] == zone_id)
        sim = config.get_simulation_config()
        return cls(
            zone_cfg=config.get_zone(zone_id),
            dynamics=sim["zone_dynamics"],
            valid_ranges=config.get_validation_config()["fields"],
            n_machines=n_machines,
            timezone=config.get_shifts()["timezone"],
            default_dt_s=float(sim["env_interval_s"]),
            rng=rng,
            start_ts=start_ts,
        )

    # ------------------------------------------------------------------ targets

    def _running_fraction(self, running_factors: list[float]) -> float:
        """Share of this zone's machines that are running (1.0 for machine-free zones)."""
        if self._n_machines == 0:
            return 1.0
        return len(running_factors) / self._n_machines

    def _noise_target(self, running_factors: list[float]) -> tuple[float, float]:
        """Return (target dBA, stationary sd) from background plus running machines.

        Each machine contributes an equal share of the zone's full-load sound energy,
        multiplied by its noise energy factor (above 1.0 for a degrading machine).
        """
        mean, sd = self._p["noise_dba"]["mean"], self._p["noise_dba"]["sd"]
        if self._n_machines == 0:
            return mean, sd
        bg = self._d["background_noise_dba"]
        e_bg = _db_to_energy(bg["mean"])
        e_machines = max(_db_to_energy(mean) - e_bg, 0.0)
        energy_share = sum(running_factors) / self._n_machines
        target = _energy_to_db(e_bg + e_machines * energy_share)
        f = self._running_fraction(running_factors)
        return target, bg["sd"] + f * (sd - bg["sd"])

    def _diurnal_cos(self, ts: datetime) -> float:
        """Cosine of the daily cycle: +1 at the hottest local hour, -1 twelve hours later."""
        local = ts.astimezone(self._tz)
        hour = local.hour + local.minute / 60.0 + local.second / 3600.0
        peak = self._d["diurnal"]["temp_peak_hour_local"]
        return math.cos(2.0 * math.pi * (hour - peak) / 24.0)

    # ------------------------------------------------------------------ steps

    def _ou(self, key: str, target: float, sd: float, tau_key: str, dt_s: float) -> float:
        """Advance state `key` one OU step toward `target` and return the new value."""
        theta = _theta(dt_s, self._d["tau_s"][tau_key])
        sigma = sd * math.sqrt(1.0 - (1.0 - theta) ** 2)
        value = self._state[key]
        value = value + theta * (target - value) + sigma * self._rng.standard_normal()
        self._state[key] = value
        return value

    def _init_state(self, targets: dict[str, float]) -> None:
        """Start every quantity at its current target (steady state for the current load)."""
        self._state = dict(targets)

    def _step_noise(self, f: float, dt_s: float, target: float, sd: float) -> tuple[float, float]:
        """Return (noise_dba, noise_peak_dba) including any active transient."""
        level = self._ou("noise", target, sd, "noise", dt_s)
        tr = self._d["transients"]
        if self._noise_transient_left == 0 and f > 0.0:
            if self._rng.random() < _event_probability(tr["noise_rate_per_h"], dt_s):
                lo, hi = tr["noise_duration_readings"]["min"], tr["noise_duration_readings"]["max"]
                self._noise_transient_left = int(self._rng.integers(lo, hi + 1))
                self._noise_boost_db = self._rng.uniform(
                    tr["noise_boost_db"]["min"], tr["noise_boost_db"]["max"])
                self._peak_boost_db = self._rng.uniform(
                    tr["noise_peak_boost_db"]["min"], tr["noise_peak_boost_db"]["max"])
        boost = peak_boost = 0.0
        if self._noise_transient_left > 0:
            boost, peak_boost = self._noise_boost_db, self._peak_boost_db
            self._noise_transient_left -= 1
        crest = self._d["noise_crest_db"]
        crest_db = crest["min"] + self._rng.exponential(crest["mean_extra"])
        noise = level + boost
        return noise, noise + crest_db + peak_boost

    def _step_pm(self, f: float, dt_s: float) -> float:
        """Advance the true PM2.5 concentration, including dust spikes while running."""
        mean, sd = self._p["pm25_ugm3"]["mean"], self._p["pm25_ugm3"]["sd"]
        floor = self._d["pm_floor_fraction"]
        load = floor + (1.0 - floor) * f
        value = self._ou("pm25", mean * load, sd * load, "pm", dt_s)
        tr = self._d["transients"]
        if f > 0.0 and self._rng.random() < _event_probability(tr["dust_rate_per_h"], dt_s):
            imp = tr["dust_impulse_x_mean"]
            value += self._rng.uniform(imp["min"], imp["max"]) * mean
        value = max(value, 0.0)
        self._state["pm25"] = value
        return value

    def _pm_gain(self, ts: datetime) -> float:
        """Sensor gain from lint build-up; resets once maintenance cleans the sensor."""
        drift = self._d["pm_drift"]
        if self._needs_clean_since is not None:
            if ts - self._needs_clean_since >= timedelta(hours=drift["clean_after_h"]):
                self._last_clean_ts = ts
                self._needs_clean_since = None
        days = (ts - self._last_clean_ts).total_seconds() / 86400.0
        gain = 1.0 + drift["gain_per_day"] * days
        if gain >= drift["needs_clean_gain"] and self._needs_clean_since is None:
            self._needs_clean_since = ts
        return gain

    def _update_faults(self, ts: datetime, dt_s: float) -> None:
        """End expired sensor faults and randomly start new ones."""
        cfg = self._d["sensor_faults"]
        p = _event_probability(cfg["rate_per_sensor_per_h"], dt_s)
        for sensor in SENSORS:
            fault = self._faults[sensor]
            if fault is not None and ts >= fault[1]:
                self._faults[sensor] = None
            if self._faults[sensor] is None and self._rng.random() < p:
                kind = str(self._rng.choice(cfg["kinds"]))
                minutes = self._rng.uniform(cfg["duration_min"]["min"], cfg["duration_min"]["max"])
                self._faults[sensor] = (kind, ts + timedelta(minutes=minutes))

    def _clamp(self, field: str, value: float) -> float:
        """Clamp to the contract's valid range for `field`."""
        rng = self._ranges[field]
        return min(max(value, rng["min"]), rng["max"])

    # ------------------------------------------------------------------ public

    @property
    def pm_needs_clean(self) -> bool:
        """True while the PM sensor is flagged for cleaning."""
        return self._needs_clean_since is not None

    def reading(self, ts: datetime, running_factors: list[float]) -> dict[str, Any]:
        """Return one environmental reading for this zone at simulated time `ts`.

        Args:
            ts: simulated time, timezone-aware, strictly later than the previous call.
            running_factors: one noise energy factor per machine in this zone that is
                currently running (1.0 nominal; a degrading machine reports more).

        Returns:
            The measurement fields of a `textile/zone/{zone_id}/env` payload plus
            `sensor_flags`. Fields of a faulted sensor are None.
        """
        if ts.tzinfo is None:
            raise ValueError("ts must be timezone-aware")
        if self._last_ts is None:
            dt_s = self._default_dt_s
        else:
            dt_s = (ts - self._last_ts).total_seconds()
            if dt_s <= 0:
                raise ValueError(f"timestamps must increase: {ts} after {self._last_ts}")
        self._last_ts = ts

        f = self._running_fraction(running_factors)
        diurnal = self._d["diurnal"]
        cos_day = self._diurnal_cos(ts)
        residual = diurnal["residual_sd_fraction"]
        p = self._p

        noise_target, noise_sd = self._noise_target(running_factors)
        temp_target = (p["temp_c"]["mean"] + diurnal["temp_amplitude_c"] * cos_day
                       + diurnal["machine_heat_c"] * f)
        rh_target = p["rh_pct"]["mean"] - diurnal["rh_amplitude_pct"] * cos_day
        outdoor = self._d["outdoor_co2_ppm"]
        co2_target = outdoor + f * (p["co2_ppm"]["mean"] - outdoor)

        if not self._state:
            floor = self._d["pm_floor_fraction"]
            self._init_state({
                "noise": noise_target,
                "pm25": p["pm25_ugm3"]["mean"] * (floor + (1.0 - floor) * f),
                "temp": temp_target,
                "rh": rh_target,
                "globe": temp_target + p["globe_offset_c"],
                "co2": co2_target,
            })

        noise, noise_peak = self._step_noise(f, dt_s, noise_target, noise_sd)
        pm25_true = self._step_pm(f, dt_s)
        temp = self._ou("temp", temp_target, p["temp_c"]["sd"] * residual, "temp", dt_s)
        rh = self._ou("rh", rh_target, p["rh_pct"]["sd"] * residual, "rh", dt_s)
        globe_share = diurnal["globe_idle_fraction"] + (1.0 - diurnal["globe_idle_fraction"]) * f
        globe = self._ou("globe", temp + p["globe_offset_c"] * globe_share,
                         diurnal["globe_sd_c"], "globe", dt_s)
        co2 = self._ou("co2", co2_target, p["co2_ppm"]["sd"] * residual, "co2", dt_s)

        gain = self._pm_gain(ts)
        pm25 = pm25_true * gain
        pm_ratio = p["pm10_ugm3"]["mean"] / p["pm25_ugm3"]["mean"]
        pm10 = pm25 * pm_ratio * math.exp(self._d["pm10_ratio_log_sd"] * self._rng.standard_normal())

        values: dict[str, Any] = {
            "noise_dba": round(self._clamp("noise_dba", noise), 1),
            "noise_peak_dba": round(self._clamp("noise_peak_dba", noise_peak), 1),
            "pm25_ugm3": round(self._clamp("pm25_ugm3", pm25), 1),
            "pm10_ugm3": round(self._clamp("pm10_ugm3", pm10), 1),
            "temp_c": round(self._clamp("temp_c", temp), 1),
            "rh_pct": round(self._clamp("rh_pct", rh), 1),
            "globe_temp_c": round(self._clamp("globe_temp_c", globe), 1),
            "co2_ppm": int(round(self._clamp("co2_ppm", co2))),
        }
        values = self._silent.apply(ts, dt_s, values)

        self._update_faults(ts, dt_s)
        flags: dict[str, str] = {}
        for sensor in SENSORS:
            fault = self._faults[sensor]
            if fault is not None:
                flags[sensor] = fault[0]
                for field in SENSOR_FIELDS[sensor]:
                    values[field] = None
            elif sensor == "pm" and self.pm_needs_clean:
                flags[sensor] = "needs_clean"
            else:
                flags[sensor] = "ok"
        values["sensor_flags"] = flags
        return values
