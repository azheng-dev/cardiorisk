"""Tests for the central env-var settings module."""

from __future__ import annotations

import pytest

from cardiorisk.settings import Settings, get_settings


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    """Reset the lru_cache around get_settings between tests."""
    get_settings.cache_clear()


def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "APP_ENV",
        "APP_RELEASE",
        "CORS_ALLOW_ORIGINS",
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "LANGFUSE_HOST",
        "SENTRY_DSN",
        "SENTRY_TRACES_SAMPLE_RATE",
        "SUPABASE_URL",
        "SUPABASE_ANON_KEY",
        "SUPABASE_SERVICE_ROLE_KEY",
    ):
        monkeypatch.delenv(name, raising=False)


class TestDefaults:
    def test_defaults_are_sane(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _clear_env(monkeypatch)
        settings = Settings(_env_file=None)  # type: ignore[call-arg]
        assert settings.app_env == "development"
        assert settings.cors_allow_origins == "*"
        assert settings.langfuse_host == "https://cloud.langfuse.com"
        assert settings.sentry_traces_sample_rate == pytest.approx(0.1)

    def test_observability_predicates_are_false_when_keys_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _clear_env(monkeypatch)
        settings = Settings(_env_file=None)  # type: ignore[call-arg]
        assert settings.langfuse_enabled is False
        assert settings.sentry_enabled is False

    def test_cors_origins_list_splits_on_commas(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _clear_env(monkeypatch)
        monkeypatch.setenv(
            "CORS_ALLOW_ORIGINS",
            "https://example.com, https://www.example.com,https://staging.example.com",
        )
        settings = Settings(_env_file=None)  # type: ignore[call-arg]
        assert settings.cors_allow_origins_list == [
            "https://example.com",
            "https://www.example.com",
            "https://staging.example.com",
        ]


class TestLangfuse:
    def test_langfuse_enabled_requires_both_keys(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _clear_env(monkeypatch)
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
        settings = Settings(_env_file=None)  # type: ignore[call-arg]
        assert settings.langfuse_enabled is False  # secret key missing
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
        settings_full = Settings(_env_file=None)  # type: ignore[call-arg]
        assert settings_full.langfuse_enabled is True


class TestSentry:
    def test_sentry_enabled_when_dsn_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _clear_env(monkeypatch)
        monkeypatch.setenv("SENTRY_DSN", "https://abc@sentry.io/123")
        settings = Settings(_env_file=None)  # type: ignore[call-arg]
        assert settings.sentry_enabled is True

    def test_sentry_sample_rate_bounded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _clear_env(monkeypatch)
        monkeypatch.setenv("SENTRY_TRACES_SAMPLE_RATE", "0.5")
        settings = Settings(_env_file=None)  # type: ignore[call-arg]
        assert settings.sentry_traces_sample_rate == pytest.approx(0.5)


class TestCache:
    def test_get_settings_is_cached(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _clear_env(monkeypatch)
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2
