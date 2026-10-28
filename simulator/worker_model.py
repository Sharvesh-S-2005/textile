"""Worker model: where each worker is, plus RFID gate and PPE events.

For the simulated window the model builds each worker's timeline up front as a list
of contiguous (start, end, zone) segments running from gate check-in to check-out:

    check-in -> break zone until shift start
             -> work periods (home zone, with occasional short trips to an adjacent
                zone) separated by breaks in the break zone
             -> optional overtime in the home zone
             -> break zone until check-out

Outputs are the *true* zones. Turning them into noisy BLE sightings is the
publisher's job; resolving those sightings back to zones is the backend's job.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np

import config
from shifts import ShiftInstance, ShiftSchedule


@dataclass(frozen=True)
class Segment:
    """A worker is in `zone_id` during [start, end)."""

    start: datetime
    end: datetime
    zone_id: str


@dataclass(frozen=True)
class RfidEvent:
    """A gate check-in or check-out."""

    ts: datetime
    worker_id: str
    card_id: str
    event: str


@dataclass(frozen=True)
class PpeEvent:
    """A supervisor-recorded PPE observation."""

    ts: datetime
    worker_id: str
    hearing_ppe: str
    respiratory_ppe: str
    recorded_by: str


def _minutes(rng: np.random.Generator, bounds: dict[str, float]) -> timedelta:
    """Uniform random duration between bounds["min"] and bounds["max"] minutes."""
    return timedelta(minutes=float(rng.uniform(bounds["min"], bounds["max"])))


def _merge(segments: list[Segment]) -> list[Segment]:
    """Drop empty segments and join neighbours that share a zone."""
    merged: list[Segment] = []
    for seg in segments:
        if seg.end <= seg.start:
            continue
        if merged and merged[-1].zone_id == seg.zone_id and merged[-1].end == seg.start:
            merged[-1] = Segment(merged[-1].start, seg.end, seg.zone_id)
        else:
            merged.append(seg)
    return merged


class WorkerModel:
    """Timelines, gate events and PPE events for every worker over a simulated window."""

    def __init__(
        self,
        workers: list[dict[str, Any]],
        dynamics: dict[str, Any],
        adjacency: dict[str, list[str]],
        schedule: ShiftSchedule,
        rng: np.random.Generator,
        start: datetime,
        end: datetime,
        whole_sessions_only: bool = False,
    ) -> None:
        """Build every worker's timeline for shifts overlapping [start, end).

        Args:
            workers: `workers` from config.yaml.
            dynamics: `simulation.worker_dynamics` from config.yaml.
            adjacency: `layout.adjacency` from config.yaml.
            schedule: shift calendar.
            rng: random generator; one child stream is spawned per worker.
            start, end: simulated window (UTC, timezone-aware).
            whole_sessions_only: keep only shift instances whose whole session
                (check-in to check-out) lies inside [start, end). Crews cut by the
                window edge are then simply not on site. Shifts overlap at every
                handover, so no cut point avoids truncating someone otherwise.
        """
        self._d = dynamics
        self._adjacency = adjacency
        self._workers = {w["worker_id"]: w for w in workers}
        self.timelines: dict[str, list[Segment]] = {}
        self.rfid_events: list[RfidEvent] = []
        self.ppe_events: list[PpeEvent] = []
        self._tap_partner = {second: first for first, second in dynamics["same_second_tap_pairs"]}
        self._check_ins: dict[tuple[str, datetime], datetime] = {}

        # Widen the query so a shift ending at `start` (its check-out) and one starting
        # just after `end` (its check-in) are both included.
        margin = timedelta(hours=12)
        child_rngs = rng.spawn(len(workers))
        for worker, w_rng in zip(workers, child_rngs):
            segments: list[Segment] = []
            for shift in schedule.instances_between(start - margin, end + margin, worker["shift"]):
                n_rfid, n_ppe = len(self.rfid_events), len(self.ppe_events)
                shift_segments = self._shift_timeline(worker, shift, w_rng)
                whole = shift_segments[0].start >= start and shift_segments[-1].end <= end
                if whole_sessions_only and not whole:
                    del self.rfid_events[n_rfid:]       # drop this cut session's events too
                    del self.ppe_events[n_ppe:]
                    continue
                segments.extend(shift_segments)
            self.timelines[worker["worker_id"]] = _merge(segments)
        self._starts = {wid: [s.start for s in segs] for wid, segs in self.timelines.items()}
        self.rfid_events.sort(key=lambda e: e.ts)
        self.ppe_events.sort(key=lambda e: e.ts)

    @classmethod
    def from_config(
        cls, schedule: ShiftSchedule, rng: np.random.Generator, start: datetime, end: datetime,
        whole_sessions_only: bool = False,
    ) -> WorkerModel:
        """Build the model using the project config.yaml."""
        return cls(
            workers=config.get_workers(),
            dynamics=config.get_simulation_config()["worker_dynamics"],
            adjacency=config.get_adjacency(),
            schedule=schedule,
            rng=rng,
            start=start,
            end=end,
            whole_sessions_only=whole_sessions_only,
        )

    # ------------------------------------------------------------ generation

    def _is_high_exposure(self, worker_id: str) -> bool:
        """True for the configured deliberate high-exposure workers."""
        return worker_id in self._d["high_exposure_workers"]

    def _work_period(
        self, home: str, start: datetime, end: datetime, dwell_mean_min: float,
        rng: np.random.Generator,
    ) -> list[Segment]:
        """Home-zone time with occasional short trips to an adjacent zone."""
        segments = []
        t, zone = start, home
        while t < end:
            if zone == home:
                dwell = timedelta(minutes=float(rng.exponential(dwell_mean_min)))
            else:
                dwell = _minutes(rng, self._d["away_dwell_min"])
            seg_end = min(t + dwell, end)
            segments.append(Segment(t, seg_end, zone))
            t = seg_end
            zone = str(rng.choice(self._adjacency[home])) if zone == home else home
        return segments

    def _ppe_choice(self, compliance: float, rng: np.random.Generator) -> str:
        """Draw worn / partial / not_worn / unknown from the worker's PPE compliance."""
        ppe = self._d["ppe"]
        if rng.random() < ppe["unknown_probability"]:
            return "unknown"
        if rng.random() < compliance:
            return "worn"
        return "partial" if rng.random() < ppe["partial_share_of_noncompliance"] else "not_worn"

    def _shift_timeline(
        self, worker: dict[str, Any], shift: ShiftInstance, rng: np.random.Generator
    ) -> list[Segment]:
        """Segments for one worker on one shift instance; also records gate and PPE events."""
        wid, home = worker["worker_id"], worker["home_zone"]
        break_zone = self._d["break_zone"]
        high = self._is_high_exposure(wid)

        check_in = shift.start - _minutes(rng, self._d["check_in_before_min"])
        partner = self._tap_partner.get(wid)
        if partner is not None:
            if (partner, shift.start) not in self._check_ins:
                raise config.ConfigError(
                    f"same_second_tap_pairs: {partner} must come before {wid} in workers")
            # Arrives with the partner: same whole second, half a second later.
            partner_in = self._check_ins[(partner, shift.start)]
            check_in = partner_in.replace(microsecond=0) + timedelta(milliseconds=500)
        self._check_ins[(wid, shift.start)] = check_in
        if high:
            overtime = _minutes(rng, self._d["high_exposure"]["overtime_min"])
        elif rng.random() < self._d["overtime"]["probability"]:
            overtime = _minutes(rng, self._d["overtime"]["duration_min"])
        else:
            overtime = timedelta(0)
        work_end = shift.end + overtime
        check_out = work_end + _minutes(rng, self._d["check_out_after_min"])

        dwell_mean = (self._d["high_exposure"]["home_dwell_mean_min"] if high
                      else self._d["home_dwell_mean_min"])
        breaks = [b for b in shift.breaks
                  if not (high and b.kind == "tea" and self._d["high_exposure"]["skips_tea_breaks"])]

        segments = [Segment(check_in, shift.start, break_zone)]
        t = shift.start
        for brk in breaks:
            segments.extend(self._work_period(home, t, brk.start, dwell_mean, rng))
            segments.append(Segment(brk.start, brk.end, break_zone))
            t = brk.end
        segments.extend(self._work_period(home, t, work_end, dwell_mean, rng))
        segments.append(Segment(work_end, check_out, break_zone))

        self.rfid_events.append(RfidEvent(check_in, wid, worker["card_id"], "check_in"))
        self.rfid_events.append(RfidEvent(check_out, wid, worker["card_id"], "check_out"))
        ppe_ts = shift.start + _minutes(rng, self._d["ppe"]["record_delay_min"])
        self.ppe_events.append(PpeEvent(
            ppe_ts, wid,
            hearing_ppe=self._ppe_choice(worker["ppe_compliance"], rng),
            respiratory_ppe=self._ppe_choice(worker["ppe_compliance"], rng),
            recorded_by=self._d["supervisors"][shift.shift_id],
        ))
        return segments

    # ------------------------------------------------------------ queries

    def zone_at(self, worker_id: str, ts: datetime) -> str | None:
        """True zone of `worker_id` at `ts`, or None if the worker is off site."""
        segments = self.timelines[worker_id]
        i = bisect.bisect_right(self._starts[worker_id], ts) - 1
        if i >= 0 and segments[i].start <= ts < segments[i].end:
            return segments[i].zone_id
        return None

    def zones_at(self, ts: datetime) -> dict[str, str]:
        """Map of worker_id -> true zone for every worker on site at `ts`."""
        found = {}
        for wid in self.timelines:
            zone = self.zone_at(wid, ts)
            if zone is not None:
                found[wid] = zone
        return found

    def tag_id(self, worker_id: str) -> str:
        """BLE tag carried by `worker_id`."""
        return self._workers[worker_id]["tag_id"]
