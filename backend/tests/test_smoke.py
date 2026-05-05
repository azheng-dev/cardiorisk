"""Smoke test that proves the test runner works in CI.

Real tests land alongside the modules they cover, starting in Phase 2.
"""

from cardiorisk import __version__


def test_package_import_and_version() -> None:
    assert __version__ == "0.0.0"
