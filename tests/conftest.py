"""Pytest configuration and shared fixtures.

This module provides shared fixtures and configuration for all tests.
"""

import inspect
import os

import pytest
from pydantic_settings import BaseSettings

from pg_mcp.config import settings as settings_module
from pg_mcp.config.settings import reset_settings

_SETTINGS_CLASSES = [
    cls
    for _, cls in inspect.getmembers(settings_module, inspect.isclass)
    if issubclass(cls, BaseSettings)
]


@pytest.fixture(autouse=True)
def reset_config() -> None:
    """Reset global settings before each test."""
    reset_settings()


@pytest.fixture(autouse=True)
def ignore_local_dotenv():
    """Keep tests deterministic: a developer's local .env must not leak in."""
    original = {cls: cls.model_config.get("env_file") for cls in _SETTINGS_CLASSES}
    for cls in _SETTINGS_CLASSES:
        cls.model_config["env_file"] = None
    yield
    for cls, value in original.items():
        cls.model_config["env_file"] = value


@pytest.fixture(autouse=True)
def disable_metrics_for_tests():
    """Disable metrics for tests to avoid port conflicts."""
    os.environ["OBSERVABILITY_METRICS_ENABLED"] = "false"
    yield
    # Clean up
    if "OBSERVABILITY_METRICS_ENABLED" in os.environ:
        del os.environ["OBSERVABILITY_METRICS_ENABLED"]
