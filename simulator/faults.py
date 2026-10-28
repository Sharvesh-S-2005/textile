"""Random sensor-fault episodes for simulated nodes.

Each channel independently enters a fault state (e.g. `stale` or `error`) as a
Poisson process, stays in it for a random duration, then recovers. While a channel
is faulted the node publishes `null` for that channel's values with the fault kind
as its sensor flag (data contract, sections 4 and 6).

Used by machine_model.py. (profiles.py has the same behaviour built in from Day 1;
it was left as is because it already passes its tests.)
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Any

import numpy as np


class SensorFaults:
    """Fault state for a fixed set of named channels on one node."""

    def __init__(self, channels: tuple[str, ...], cfg: dict[str, Any], rng: np.random.Generator) -> None:
        """Args: channel names, a `sensor_faults` config block, and the node's generator."""
        self._channels = channels
        self._cfg = cfg
        self._rng = rng
        self._faults: dict[str, tuple[str, datetime] | None] = {c: None for c in channels}

    def update(self, ts: datetime, dt_s: float) -> None:
        """End expired faults and randomly start new ones for the interval ending at `ts`."""
        p = 1.0 - math.exp(-self._cfg["rate_per_sensor_per_h"] * dt_s / 3600.0)
        for channel in self._channels:
            fault = self._faults[channel]
            if fault is not None and ts >= fault[1]:
                self._faults[channel] = None
            if self._faults[channel] is None and self._rng.random() < p:
                kind = str(self._rng.choice(self._cfg["kinds"]))
                bounds = self._cfg["duration_min"]
                minutes = self._rng.uniform(bounds["min"], bounds["max"])
                self._faults[channel] = (kind, ts + timedelta(minutes=minutes))

    def flag(self, channel: str) -> str:
        """Current sensor flag for `channel`: the fault kind, or "ok"."""
        fault = self._faults[channel]
        return "ok" if fault is None else fault[0]
