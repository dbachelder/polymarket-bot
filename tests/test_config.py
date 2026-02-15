"""Tests for polymarket.config module."""

from __future__ import annotations

import os
from unittest import mock

import pytest

from polymarket.config import PolymarketConfig, load_config


class TestPolymarketConfig:
    """Tests for PolymarketConfig dataclass."""

    def test_default_is_dry_run(self) -> None:
        """Config defaults to dry-run mode for safety."""
        config = PolymarketConfig()
        assert config.dry_run is True

    def test_has_credentials_all_present(self) -> None:
        """has_credentials returns True when all creds present."""
        config = PolymarketConfig(
            api_key="key",
            api_secret="secret",
            api_passphrase="pass",
        )
        assert config.has_credentials is True

    def test_has_credentials_missing_one(self) -> None:
        """has_credentials returns False when any cred missing."""
        config = PolymarketConfig(
            api_key="key",
            api_secret="secret",
            api_passphrase=None,
        )
        assert config.has_credentials is False

    def test_can_trade_requires_credentials_and_not_dry_run(self) -> None:
        """can_trade requires both credentials AND dry_run=False."""
        # Has creds but dry-run
        config = PolymarketConfig(
            api_key="key",
            api_secret="secret",
            api_passphrase="pass",
            dry_run=True,
        )
        assert config.can_trade is False

        # No creds but not dry-run
        config = PolymarketConfig(dry_run=False)
        assert config.can_trade is False

        # Has creds and not dry-run
        config = PolymarketConfig(
            api_key="key",
            api_secret="secret",
            api_passphrase="pass",
            dry_run=False,
        )
        assert config.can_trade is True

    def test_validate_or_raise_dry_run_no_creds_ok(self) -> None:
        """Validation passes in dry-run mode without credentials."""
        config = PolymarketConfig(dry_run=True)
        config.validate_or_raise()  # Should not raise

    def test_validate_or_raise_live_no_creds_raises(self) -> None:
        """Validation fails in live mode without credentials."""
        config = PolymarketConfig(dry_run=False)
        with pytest.raises(ValueError, match="Live trading requested"):
            config.validate_or_raise()


class TestFromEnv:
    """Tests for loading config from environment."""

    def test_loads_from_environment(self) -> None:
        """Config loads from environment variables."""
        env = {
            "POLYMARKET_API_KEY": "test_key",
            "POLYMARKET_API_SECRET": "test_secret",
            "POLYMARKET_API_PASSPHRASE": "test_pass",
            "POLYMARKET_DRY_RUN": "false",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            config = PolymarketConfig.from_env(load_dotenv_file=False)

        assert config.api_key == "test_key"
        assert config.api_secret == "test_secret"
        assert config.api_passphrase == "test_pass"
        assert config.dry_run is False

    def test_dry_run_true_values(self) -> None:
        """Various truthy values for dry-run."""
        for value in ("1", "true", "TRUE", "yes", "YES", "on", "ON"):
            with mock.patch.dict(os.environ, {"POLYMARKET_DRY_RUN": value}, clear=True):
                config = PolymarketConfig.from_env(load_dotenv_file=False)
                assert config.dry_run is True, f"Expected True for {value}"

    def test_dry_run_false_values(self) -> None:
        """Various falsy values for dry-run."""
        for value in ("0", "false", "FALSE", "no", "NO", "off", "OFF", ""):
            with mock.patch.dict(os.environ, {"POLYMARKET_DRY_RUN": value}, clear=True):
                config = PolymarketConfig.from_env(load_dotenv_file=False)
                assert config.dry_run is False, f"Expected False for {value}"

    def test_dry_run_defaults_to_true(self) -> None:
        """When POLYMARKET_DRY_RUN is not set, defaults to True."""
        with mock.patch.dict(os.environ, {}, clear=True):
            config = PolymarketConfig.from_env(load_dotenv_file=False)
            assert config.dry_run is True

    def test_empty_strings_become_none(self) -> None:
        """Empty environment variables become None, not empty strings."""
        env = {
            "POLYMARKET_API_KEY": "",
            "POLYMARKET_API_SECRET": "",
            "POLYMARKET_API_PASSPHRASE": "",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            config = PolymarketConfig.from_env(load_dotenv_file=False)

        assert config.api_key is None
        assert config.api_secret is None
        assert config.api_passphrase is None


class TestLoadConfig:
    """Tests for load_config convenience function."""

    def test_load_config_returns_config(self) -> None:
        """load_config returns a PolymarketConfig instance."""
        with mock.patch.dict(os.environ, {}, clear=True):
            config = load_config(load_dotenv_file=False)
            assert isinstance(config, PolymarketConfig)
