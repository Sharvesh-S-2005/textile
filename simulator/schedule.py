"""Shift and break calendar shared by the machine and worker models.

Shift times in config.yaml are local (Asia/Kolkata); everything returned here is
timezone-aware UTC.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class Break:
    """One break inside a shift instance."""

    kind: str
    start: datetime
    end: datetime


@dataclass(frozen=True)
class ShiftInstance:
    """One concrete occurrence of a shift on a given day."""

    shift_id: str
    start: datetime
    end: datetime
    breaks: tuple[Break, ...]

    def contains(self, ts: datetime) -> bool:
        """True if `ts` falls in [start, end)."""
        return self.start <= ts < self.end

    def break_at(self, ts: datetime) -> Break | None:
        """Return the break covering `ts`, or None."""
        for brk in self.breaks:
            if brk.start <= ts < brk.end:
                return brk
        return None


def _parse_hhmm(text: str) -> time:
    """Parse "HH:MM" into a time."""
    hours, minutes = text.split(":")
    return time(int(hours), int(minutes))


class ShiftSchedule:
    """Resolve shift and break instances from the `shifts` section of config.yaml."""

    def __init__(self, shifts_cfg: dict[str, Any]) -> None:
        """Build the schedule from config.yaml's `shifts` section."""
        self._tz = ZoneInfo(shifts_cfg["timezone"])
        self._defs: dict[str, dict[str, Any]] = {
            d["shift_id"]: d for d in shifts_cfg["definitions"]
        }

    @property
    def shift_ids(self) -> list[str]:
        """All configured shift IDs."""
        return list(self._defs)

    def _local(self, day: date, at: time) -> datetime:
        """Local wall-clock time on `day`, converted to UTC."""
        return datetime.combine(day, at, tzinfo=self._tz).astimezone(timezone.utc)

    def instance(self, shift_id: str, local_day: date) -> ShiftInstance:
        """The occurrence of `shift_id` that starts on local calendar day `local_day`."""
        d = self._defs[shift_id]
        start_t, end_t = _parse_hhmm(d["start"]), _parse_hhmm(d["end"])
        start = self._local(local_day, start_t)
        end_day = local_day + timedelta(days=1) if end_t <= start_t else local_day
        end = self._local(end_day, end_t)
        breaks = []
        for b in d["breaks"]:
            b_t = _parse_hhmm(b["start"])
            b_day = local_day + timedelta(days=1) if b_t < start_t else local_day
            b_start = self._local(b_day, b_t)
            breaks.append(Break(b["kind"], b_start, b_start + timedelta(minutes=b["duration_min"])))
        return ShiftInstance(shift_id, start, end, tuple(breaks))

    def instances_between(
        self, start: datetime, end: datetime, shift_id: str | None = None
    ) -> list[ShiftInstance]:
        """All shift instances overlapping [start, end), sorted by start time."""
        first_day = start.astimezone(self._tz).date() - timedelta(days=1)
        last_day = end.astimezone(self._tz).date()
        ids = [shift_id] if shift_id is not None else self.shift_ids
        found = []
        day = first_day
        while day <= last_day:
            for sid in ids:
                inst = self.instance(sid, day)
                if inst.start < end and inst.end > start:
                    found.append(inst)
            day += timedelta(days=1)
        return sorted(found, key=lambda i: i.start)

    def active_at(self, ts: datetime) -> ShiftInstance | None:
        """The shift instance in progress at `ts`, or None if no shift covers it."""
        for inst in self.instances_between(ts, ts + timedelta(seconds=1)):
            if inst.contains(ts):
                return inst
        return None
