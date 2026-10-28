"""Machine model: clamp-meter current and vibration for one machine.

A machine runs while its zone's shift is in progress, stops for breaks and for a
short crew changeover at each shift start, and lags every switch by its own fixed
stagger so machines do not all switch at the same second. `running` is decided
from `current_a` crossing a threshold, exactly as the real machine node does it.

The degrading machine's vibration grows exponentially with simulated age:
    factor(days) = ratio_at_30_days ** (days / 30)
Its noise contribution to the zone scales with the same factor (sound energy taken
as proportional to vibration RMS - a deliberately mild coupling; see
`noise_energy_factor`).

The current and vibration channels have occasional `stale`/`error` episodes
(contract v1.1d). A faulted channel publishes `null`; with the current channel down,
`running` repeats the node's last decision because the node cannot measure it.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import numpy as np

import config
from simulator.faults import SensorFaults
from shifts import ShiftSchedule

CHANNELS: tuple[str, ...] = ("current", "vibration")


class MachineModel:
    """Stateful generator of `textile/machine/{machine_id}/status` readings."""

    def __init__(
        self,
        machine_cfg: dict[str, Any],
        dynamics: dict[str, Any],
        valid_ranges: dict[str, dict[str, float]],
        schedule: ShiftSchedule,
        rng: np.random.Generator,
        start_ts: datetime,
    ) -> None:
        """Build the model from a machine's config entry.

        Args:
            machine_cfg: one entry of `machines` in config.yaml.
            dynamics: `simulation.machine_dynamics` from config.yaml.
            valid_ranges: `validation.fields` from config.yaml (for output clamping).
            schedule: shift calendar deciding when machines should run.
            rng: random generator owned by this machine.
            start_ts: simulated start time (UTC, timezone-aware); age is measured from here.
        """
        self.machine_id: str = machine_cfg["machine_id"]
        self.zone_id: str = machine_cfg["zone_id"]
        self.device_id: str = machine_cfg["device_id"]
        self._rated_a = float(machine_cfg["rated_current_a"])
        self._baseline_vib = float(machine_cfg["baseline_vib_rms_g"])
        self.degrading: bool = bool(machine_cfg["degrading"])
        self._d = dynamics
        self._ranges = valid_ranges
        self._schedule = schedule
        self._rng = rng
        self._start_ts = start_ts
        self._lag = timedelta(seconds=float(rng.uniform(0.0, dynamics["stagger_max_s"])))
        self._on_since: datetime | None = None
        self._first_reading = True
        # Own stream, so fault draws do not shift the measurement sequence.
        self._faults = SensorFaults(CHANNELS, dynamics["sensor_faults"], rng.spawn(1)[0])
        self._last_ts: datetime | None = None
        self._last_reported_running = False
        self.running_now = False  # physical state, for the zone noise coupling

    @classmethod
    def from_config(
        cls, machine_id: str, schedule: ShiftSchedule, rng: np.random.Generator,
        start_ts: datetime,
    ) -> MachineModel:
        """Build the model for `machine_id` using the project config.yaml."""
        matches = [m for m in config.get_machines() if m["machine_id"] == machine_id]
        if not matches:
            raise config.ConfigError(f"unknown machine_id: {machine_id}")
        dynamics = config.get_simulation_config()["machine_dynamics"]
        valid_ranges = config.get_validation_config()["fields"]
        return cls(matches[0], dynamics, valid_ranges, schedule, rng, start_ts)

    def scheduled_on(self, ts: datetime) -> bool:
        """True if the shift calendar says this machine should be running at `ts`."""
        t = ts - self._lag
        shift = self._schedule.active_at(t)
        if shift is None or shift.break_at(t) is not None:
            return False
        return t >= shift.start + timedelta(minutes=self._d["changeover_min"])

    def degradation_factor(self, ts: datetime) -> float:
        """Expected vib_rms multiplier from wear at `ts` (always 1.0 if not degrading)."""
        if not self.degrading:
            return 1.0
        deg = self._d["degradation"]
        days = deg["initial_days"] + (ts - self._start_ts).total_seconds() / 86400.0
        return float(deg["ratio_at_30_days"] ** (max(days, 0.0) / 30.0))

    def noise_energy_factor(self, ts: datetime) -> float:
        """Multiplier on this machine's share of zone sound energy, for ZoneProfile.

        Uses the expected (noise-free) vibration ratio, so a worn machine is
        audibly louder in its zone: 2.5x vibration -> 2.5x this machine's energy share.
        """
        return self.degradation_factor(ts)

    def _current(self, ts: datetime, on: bool) -> float:
        """Clamp-meter current at `ts`, including an inrush spike right after start-up."""
        if not on:
            self._on_since = None
            standby = self._d["standby_current_a"]
            return abs(standby["mean"] + standby["sd"] * self._rng.standard_normal())
        if self._on_since is None:
            # Already running when the simulation starts: no start-up, so no inrush.
            self._on_since = ts - timedelta(days=1) if self._first_reading else ts
        current = self._rated_a * (1.0 + self._d["current_noise_fraction"] * self._rng.standard_normal())
        inrush = self._d["inrush"]
        if (ts - self._on_since).total_seconds() < inrush["duration_s"]:
            current *= self._rng.uniform(inrush["multiplier"]["min"], inrush["multiplier"]["max"])
        return max(current, 0.0)

    def _clamp(self, field: str, value: float) -> float:
        """Clamp to the contract's valid range for `field`."""
        rng = self._ranges[field]
        return min(max(value, rng["min"]), rng["max"])

    def reading(self, ts: datetime) -> dict[str, Any]:
        """Return the measurement fields of a machine status payload at simulated `ts`.

        Also updates `running_now`, the machine's true physical state, which the
        publisher uses for the zone noise coupling whatever the sensors report.
        """
        current = self._current(ts, self.scheduled_on(ts))
        self._first_reading = False
        running = current > self._d["running_threshold_fraction"] * self._rated_a
        self.running_now = bool(running)
        if running:
            expected = self._baseline_vib * self.degradation_factor(ts)
            rms = expected * (1.0 + self._d["vib_noise_fraction"] * self._rng.standard_normal())
        else:
            idle = self._d["idle_vib_rms_g"]
            rms = idle["mean"] + idle["sd"] * self._rng.standard_normal()
        rms = max(rms, 0.0)
        crest = self._d["vib_crest_factor"]
        peak = rms * self._rng.uniform(crest["min"], crest["max"])

        dt_s = 0.0 if self._last_ts is None else (ts - self._last_ts).total_seconds()
        self._last_ts = ts
        self._faults.update(ts, dt_s)
        flags = {c: self._faults.flag(c) for c in CHANNELS}

        reading: dict[str, Any] = {
            "zone_id": self.zone_id,
            "current_a": round(self._clamp("current_a", current), 2),
            "running": bool(running),
            "vib_rms_g": round(self._clamp("vib_rms_g", rms), 3),
            "vib_peak_g": round(self._clamp("vib_peak_g", peak), 3),
            "sensor_flags": flags,
        }
        if flags["current"] != "ok":
            reading["current_a"] = None
            reading["running"] = self._last_reported_running
        if flags["vibration"] != "ok":
            reading["vib_rms_g"] = None
            reading["vib_peak_g"] = None
        self._last_reported_running = reading["running"]
        return reading
