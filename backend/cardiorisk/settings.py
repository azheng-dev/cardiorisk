"""Central env-var-driven settings for the backend.

One :class:`Settings` lives at the top of the process. It reads from
``os.environ`` (and an optional ``.env`` file) at construction time
and is otherwise immutable. Every other module that needs an env-
backed knob calls :func:`get_settings` rather than reaching into
:mod:`os.environ` directly.

The settings deliberately *do not* fail loudly when keys are missing —
every observability + integration credential is optional. CI runs
without any of them; production wires them in via the platform env.
Modules that need a key explicitly check for its presence and fall
back to a no-op.

This module is the home of the Phase 7 observability env contract
(:data:`Settings.langfuse_*`, :data:`Settings.sentry_dsn`) and the
Phase 8 deploy contract (:data:`Settings.supabase_*`,
:data:`Settings.app_env`, :data:`Settings.cors_allow_origins`). See
ADR-024 for the observability decision and ADR-025 for the deploy
decision (placeholder until Phase 8 commits to it).
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-backed settings.

    Field names are lower-case; the underlying env-var names are
    derived case-insensitively (pydantic-settings default) — so
    ``langfuse_public_key`` reads from ``LANGFUSE_PUBLIC_KEY``.
    """

    # ------------------------------------------------------------------
    # App identity
    # ------------------------------------------------------------------
    app_env: str = Field(default="development", description="dev|preview|production")
    app_release: str | None = Field(
        default=None,
        description=(
            "Release identifier — Sentry tags every event with this. "
            "On Vercel/HF Spaces this is the short git SHA."
        ),
    )

    # ------------------------------------------------------------------
    # CORS (Phase 8 deploy)
    # ------------------------------------------------------------------
    cors_allow_origins: str = Field(
        default="*",
        description=(
            "Comma-separated list of origins permitted by CORS. Defaults "
            "to '*' for local dev; production must override with the "
            "Vercel origin."
        ),
    )

    # ------------------------------------------------------------------
    # Langfuse (Phase 7 — observability)
    # ------------------------------------------------------------------
    langfuse_public_key: str | None = Field(default=None)
    langfuse_secret_key: str | None = Field(default=None)
    langfuse_host: str = Field(
        default="https://cloud.langfuse.com",
        description=(
            "Langfuse base URL. Cloud Hobby is the default per ADR-024; self-hosters can override."
        ),
    )

    # ------------------------------------------------------------------
    # Sentry (Phase 7 — error tracking)
    # ------------------------------------------------------------------
    sentry_dsn: str | None = Field(default=None)
    sentry_traces_sample_rate: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description=(
            "Fraction of transactions Sentry samples for performance "
            "tracing. Production default is 0.1; CI is 0.0 (no DSN)."
        ),
    )

    # ------------------------------------------------------------------
    # Supabase (Phase 8 — synthetic case storage)
    # ------------------------------------------------------------------
    supabase_url: str | None = Field(default=None)
    supabase_anon_key: str | None = Field(default=None)
    supabase_service_role_key: str | None = Field(default=None)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # Convenience predicates
    # ------------------------------------------------------------------
    @property
    def langfuse_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)

    @property
    def sentry_enabled(self) -> bool:
        return bool(self.sentry_dsn)

    @property
    def cors_allow_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_allow_origins.split(",") if o.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings constructor.

    Tests can clear the cache with ``get_settings.cache_clear()``
    after monkey-patching environment variables. Production code
    only calls this once at process boot.
    """
    return Settings()


__all__ = ["Settings", "get_settings"]
