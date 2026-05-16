"""Tests for the Sentry wrappers + PII scrubbing.

CI runs without a SENTRY_DSN; the SDK init is a no-op. The PII
scrubber is exercised independently from the SDK init so the
contract is testable without sending any events to Sentry.
"""

from __future__ import annotations

from typing import Any

import pytest

from cardiorisk.observability.sentry import init_sentry, scrub_patient
from cardiorisk.settings import get_settings


@pytest.fixture(autouse=True)
def _disable_sentry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SENTRY_DSN", raising=False)
    get_settings.cache_clear()


class TestNoOpInit:
    def test_init_returns_without_dsn(self) -> None:
        # Must not raise — Sentry simply does nothing.
        init_sentry(app=None)


class TestPatientScrub:
    def test_strips_top_level_patient_key(self) -> None:
        event: dict[str, Any] = {
            "level": "error",
            "patient": {"Age": 58, "Sex": "M"},
        }
        out = scrub_patient(event)
        assert out["level"] == "error"
        assert out["patient"] == "<scrubbed>"

    def test_strips_nested_patient_key(self) -> None:
        event: dict[str, Any] = {
            "request": {
                "method": "POST",
                "data": {"patient": {"Age": 72, "Sex": "F"}},
            },
        }
        out = scrub_patient(event)
        assert out["request"]["method"] == "POST"
        assert out["request"]["data"]["patient"] == "<scrubbed>"

    def test_strips_patient_inside_list(self) -> None:
        event: dict[str, Any] = {
            "breadcrumbs": [
                {"message": "request started", "patient": {"Age": 40}},
                {"message": "validation passed"},
            ]
        }
        out = scrub_patient(event)
        assert out["breadcrumbs"][0]["patient"] == "<scrubbed>"
        assert out["breadcrumbs"][1] == {"message": "validation passed"}

    def test_case_insensitive_match(self) -> None:
        event: dict[str, Any] = {"extra": {"Patient": {"Age": 50}, "PATIENT": {"Age": 60}}}
        out = scrub_patient(event)
        assert out["extra"]["Patient"] == "<scrubbed>"
        assert out["extra"]["PATIENT"] == "<scrubbed>"

    def test_leaves_other_keys_untouched(self) -> None:
        event: dict[str, Any] = {
            "level": "error",
            "user": {"id": "u-123"},
            "tags": {"environment": "production"},
        }
        out = scrub_patient(event)
        assert out == event

    def test_tuples_are_preserved(self) -> None:
        event: dict[str, Any] = {
            "extra": {
                "path": ("api", "v1", "agents"),
                "patient_tuple": (1, 2, 3),  # not a 'patient' key
            }
        }
        out = scrub_patient(event)
        assert out["extra"]["path"] == ("api", "v1", "agents")
        # 'patient_tuple' is NOT exactly 'patient' so it passes through.
        assert out["extra"]["patient_tuple"] == (1, 2, 3)
