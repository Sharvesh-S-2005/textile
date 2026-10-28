"""Tests for scripts/inject_bad_records.py: each crafted record is rejected for its stated reason."""

import json
from datetime import datetime, timezone

from ingest.validation import Reason, validate
from scripts.inject_bad_records import MARKER, build_records

REF = datetime(2026, 10, 28, 3, 0, tzinfo=timezone.utc)


def test_one_record_per_rejection_reason() -> None:
    reasons = [r for r, _, _ in build_records(REF)]
    assert sorted(reasons) == sorted(Reason)          # every reason, exactly once


def test_each_record_is_rejected_for_its_stated_reason() -> None:
    for expected, topic, raw in build_records(REF):
        if expected == Reason.MALFORMED_JSON:
            try:
                json.loads(raw)
            except json.JSONDecodeError:
                continue
            raise AssertionError("malformed_json record parsed")
        verdict = validate(topic, json.loads(raw), REF)
        assert not verdict.accepted and verdict.reason == expected, (expected, verdict.reason, verdict.detail)


def test_every_record_carries_the_marker() -> None:
    for _, _, raw in build_records(REF):
        assert MARKER.encode() in raw
