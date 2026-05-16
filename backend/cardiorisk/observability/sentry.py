"""Sentry wrappers — error tracking with PII scrubbing.

Same env-var-gated discipline as :mod:`cardiorisk.observability.langfuse`:
when ``SENTRY_DSN`` is unset, every helper here is a no-op. The SDK
itself is only imported inside the helpers so pytest collection
stays fast and CI doesn't pull a network connection to Sentry just
by running the test suite.

PII scrubbing
-------------
This is a research artefact for synthetic data only (AGENTS §1), but
the discipline is real: any payload key named ``patient`` is stripped
from every Sentry event before it leaves the process. The patient
schema is the only API field whose contents Sentry would otherwise
log (a 422 on a malformed patient body, for example, can include the
raw payload). :func:`scrub_patient` is the dedicated helper; it's
wired in as the SDK's ``before_send`` hook and also exposed for
unit-testing.
"""

from __future__ import annotations

from typing import Any

from cardiorisk.settings import get_settings


def init_sentry(*, app: Any | None = None) -> None:
    """Wire Sentry into the FastAPI process.

    Called from :func:`cardiorisk.api.server.build_app`. Re-entrant —
    calling twice has no extra effect; the SDK's own ``init`` is
    idempotent within a process.

    Args:
        app: The FastAPI app (kept for parity with the SDK API; the
            current sentry-sdk auto-discovers FastAPI without needing
            an explicit handle, but accepting it future-proofs the
            integration if we ever switch to the explicit
            ``FastApiIntegration`` route).
    """
    settings = get_settings()
    if not settings.sentry_enabled:
        return
    try:
        import sentry_sdk
        from sentry_sdk.integrations.fastapi import FastApiIntegration
        from sentry_sdk.integrations.starlette import StarletteIntegration
    except ImportError:  # pragma: no cover - dep is optional at import time
        return
    try:
        sentry_sdk.init(
            dsn=settings.sentry_dsn,
            environment=settings.app_env,
            release=settings.app_release,
            traces_sample_rate=settings.sentry_traces_sample_rate,
            send_default_pii=False,
            integrations=[
                StarletteIntegration(),
                FastApiIntegration(),
            ],
            before_send=_before_send,
        )
    except Exception:  # pragma: no cover - SDK init failure must not break app boot
        return


def scrub_patient(payload: Any) -> Any:
    """Strip every ``patient``-shaped key from a Sentry event payload.

    Walks dicts + lists recursively. Replaces values under the
    ``patient`` key with the literal string ``"<scrubbed>"`` so the
    scrub is visible in the Sentry UI (vs silently dropping the key,
    which can hide bugs in the scrub path itself). Other keys are
    untouched.

    Exposed publicly so the unit tests can exercise the recursion
    without re-deriving Sentry's event schema.
    """
    return _scrub(payload)


def _before_send(event: Any, _hint: Any) -> Any | None:
    """Sentry ``before_send`` hook — scrub PII, then pass through."""
    try:
        return _scrub(event)
    except Exception:  # pragma: no cover - never drop an event because of a scrub bug
        return event


def _scrub(value: Any) -> Any:
    if isinstance(value, dict):
        scrubbed: dict[str, Any] = {}
        for k, v in value.items():
            if isinstance(k, str) and k.lower() == "patient":
                scrubbed[k] = "<scrubbed>"
            else:
                scrubbed[k] = _scrub(v)
        return scrubbed
    if isinstance(value, list):
        return [_scrub(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_scrub(item) for item in value)
    return value


__all__ = ["init_sentry", "scrub_patient"]
