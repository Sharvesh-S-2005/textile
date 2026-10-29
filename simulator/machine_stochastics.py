"""Stochastic machine behaviour (Day 8): unplanned stops, uneven shift starts,
variable load and one correlated stop dependency.

Pre-registered in docs/preregistration_day8.md; every parameter is in config.yaml
(`simulation.machine_stochastics`) and marked as literature or project assumption.

Without this module a machine runs exactly when the shift calendar says, so machine
state, worker presence and the clock are the same signal. With it:

* Unplanned stops: inter-arrival times ~ Exponential(mean gap), counted in RUNNING
  time (a stopped machine cannot break again); durations ~ Lognormal(median, sigma).
* Shift starts: each machine gets an independent offset after the changeover on every
  shift, and may start late or not at all on that shift.
* Load: a reflected random walk scaling the clamp-meter current (not zone noise).
* Dependency: a blowroom stop longer than the chute buffer starves carding.

A machine's physical state still drives zone noise through MachineModel.running_now,
so every stop here lowers zone noise and every restart raises it again (with the
existing inrush on the current channel).

The draws use their own generator, so turning this off leaves the deterministic
simulator's random sequences exactly as they were.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np


@dataclass(frozen=True)
class StopParams:
    """Unplanned-stop parameters for one machine type."""

    mean_gap_min: float
    duration_median_min: float
    duration_sigma: float
    target_availability: tuple[float, float]

    @property
    def expected_unplanned_availability(self) -> float:
        """Availability from unplanned stops alone: gap / (gap + mean duration)."""
        mean_dur = self.duration_median_min * math.exp(self.duration_sigma ** 2 / 2.0)
        return self.mean_gap_min / (self.mean_gap_min + mean_dur)


@dataclass
class StopRecord:
    """One stop, for verification: unplanned (own) or forced (starved by upstream)."""

    start: datetime
    end: datetime
    kind: str


@dataclass
class MachineStochastics:
    """Per-machine stochastic state. Ask it, once per reading, whether the machine runs."""

    machine_id: str
    params: StopParams
    shift_cfg: dict[str, Any]
    load_cfg: dict[str, Any]
    rng: np.random.Generator
    plant: Plant | None = None
    stops: deque[StopRecord] = field(default_factory=deque)   # Plant passes a bounded deque
    _forced: list[tuple[datetime, datetime]] = field(default_factory=list)
    _stop_until: datetime | None = None
    _run_since_stop_min: float = 0.0
    _next_gap_min: float = 0.0
    _last_ts: datetime | None = None
    _shift_plan: dict[datetime, float | None] = field(default_factory=dict)
    _load: float = 0.0

    def __post_init__(self) -> None:
        self._next_gap_min = float(self.rng.exponential(self.params.mean_gap_min))
        lo, hi = self.load_cfg["bounds"]
        self._load = float(self.rng.uniform(lo, hi))

    # ------------------------------------------------------------ shift starts

    def start_delay_min(self, shift_start: datetime) -> float | None:
        """Minutes after the changeover before this machine starts on that shift (None = not at all).

        Drawn once per shift instance, the first time the shift is seen.
        """
        if shift_start not in self._shift_plan:
            # Readings arrive in time order, so plans older than a day are never asked
            # for again; dropping them keeps the dict bounded without changing any draw.
            horizon = shift_start - timedelta(days=1)
            for old in [k for k in self._shift_plan if k < horizon]:
                del self._shift_plan[old]
            s = self.shift_cfg
            if self.rng.random() < s["skip_probability"]:
                delay: float | None = None
            else:
                delay = float(self.rng.uniform(0.0, s["offset_max_min"]))
                if self.rng.random() < s["late_probability"]:
                    delay += float(self.rng.uniform(*s["late_extra_min"]))
            self._shift_plan[shift_start] = delay
        return self._shift_plan[shift_start]

    # ------------------------------------------------------------ stops

    def force_stop(self, start: datetime, end: datetime) -> None:
        """A stop imposed from outside (starvation by an upstream machine)."""
        self._forced.append((start, end))
        self.stops.append(StopRecord(start, end, "forced"))

    def in_stop(self, ts: datetime) -> bool:
        """True during an own unplanned stop or a forced stop."""
        if self._stop_until is not None and ts < self._stop_until:
            return True
        return any(a <= ts < b for a, b in self._forced)

    def running(self, ts: datetime, scheduled: bool, since_changeover_min: float | None,
                shift_start: datetime | None) -> bool:
        """Physical running state at `ts`, given the deterministic calendar.

        Args:
            ts: simulated time (readings arrive in time order).
            scheduled: the deterministic calendar says the machine should run.
            since_changeover_min: minutes since this shift's changeover ended (None off-shift).
            shift_start: start of the active shift instance (key for its start plan).
        """
        dt_min = 0.0 if self._last_ts is None else (ts - self._last_ts).total_seconds() / 60.0
        self._last_ts = ts
        self._step_load(dt_min)
        if self._stop_until is not None and ts >= self._stop_until:
            self._stop_until = None
        self._forced = [(a, b) for a, b in self._forced if b > ts]
        if not scheduled or shift_start is None:
            return False
        delay = self.start_delay_min(shift_start)
        if delay is None or since_changeover_min is None or since_changeover_min < delay:
            return False
        if self.in_stop(ts):
            return False
        self._run_since_stop_min += dt_min
        if self._run_since_stop_min >= self._next_gap_min:
            self._begin_stop(ts)
            return False
        return True

    def _begin_stop(self, ts: datetime) -> None:
        p = self.params
        dur_min = float(p.duration_median_min * math.exp(p.duration_sigma * self.rng.standard_normal()))
        end = ts + timedelta(minutes=dur_min)
        self._stop_until = end
        self._run_since_stop_min = 0.0
        self._next_gap_min = float(self.rng.exponential(p.mean_gap_min))
        self.stops.append(StopRecord(ts, end, "unplanned"))
        if self.plant is not None:
            self.plant.on_stop(self.machine_id, ts, end)

    # ------------------------------------------------------------ load

    def _step_load(self, dt_min: float) -> None:
        if dt_min <= 0:
            return
        lo, hi = self.load_cfg["bounds"]
        x = self._load + self.load_cfg["step_sd_per_min"] * math.sqrt(dt_min) * float(self.rng.standard_normal())
        while x < lo or x > hi:                      # reflect into [lo, hi]
            x = 2 * lo - x if x < lo else 2 * hi - x
        self._load = x

    @property
    def load(self) -> float:
        """Current load fraction of rated current."""
        return self._load


class Plant:
    """All machines' stochastic state plus the stop dependencies between them."""

    def __init__(self, machines_cfg: list[dict[str, Any]], cfg: dict[str, Any], rng: np.random.Generator,
                 stop_log_max: int) -> None:
        """Args: config `machines`, `simulation.machine_stochastics`, a generator owned by the
        plant, and the cap on each machine's stop log (`simulation.memory_bounds`)."""
        self.stop_log_max = stop_log_max
        self.cfg = cfg
        self.machines: dict[str, MachineStochastics] = {}
        for m, child in zip(machines_cfg, rng.spawn(len(machines_cfg))):
            mtype = cfg["zone_machine_type"][m["zone_id"]]
            t = cfg["types"][mtype]
            params = StopParams(float(t["mean_run_min_between_stops"]), float(t["stop_duration_median_min"]),
                                float(t["stop_duration_sigma"]), tuple(t["target_availability"]))
            self.machines[m["machine_id"]] = MachineStochastics(m["machine_id"], params, cfg["shift_start"],
                                                                cfg["load"], child, plant=self,
                                                                stops=deque(maxlen=stop_log_max))
        self.dependencies = list(cfg.get("dependencies", []))
        self.dependency_fired = 0
        for d in self.dependencies:
            if d["upstream"] not in self.machines or d["downstream"] not in self.machines:
                raise ValueError(f"dependency names an unknown machine: {d}")

    def on_stop(self, machine_id: str, start: datetime, end: datetime) -> None:
        """Propagate an unplanned stop to dependent machines (starvation after the buffer empties)."""
        for d in self.dependencies:
            if d["upstream"] != machine_id:
                continue
            if (end - start) > timedelta(minutes=d["buffer_min"]):
                self.machines[d["downstream"]].force_stop(start + timedelta(minutes=d["buffer_min"]),
                                                          end + timedelta(minutes=d["restart_lag_min"]))
                self.dependency_fired += 1

    def for_machine(self, machine_id: str) -> MachineStochastics:
        """The stochastic state of one machine."""
        return self.machines[machine_id]
