"""Configuration management for Polymarket bot.

Loads credentials from environment or .env file with safe defaults.
All trading operations default to dry-run mode unless explicitly disabled.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from dotenv import load_dotenv

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PolymarketConfig:
    """Configuration for Polymarket CLOB trading.

    Attributes:
        api_key: Polymarket API key
        api_secret: Polymarket API secret for signing
        api_passphrase: API passphrase for authentication
        dry_run: If True, orders are logged but not submitted (safety default)
    """

    api_key: str | None = None
    api_secret: str | None = None
    api_passphrase: str | None = None
    dry_run: bool = True  # Safety: default to dry-run

    @property
    def has_credentials(self) -> bool:
        """Check if all required credentials are present."""
        return all([self.api_key, self.api_secret, self.api_passphrase])

    @property
    def can_trade(self) -> bool:
        """Check if live trading is possible (credentials present AND dry_run disabled)."""
        return self.has_credentials and not self.dry_run

    def validate_or_raise(self) -> None:
        """Validate configuration, raising error if credentials missing for live trading.

        Raises:
            ValueError: If attempting live trading without credentials.
        """
        if not self.dry_run and not self.has_credentials:
            msg = (
                "Live trading requested (POLYMARKET_DRY_RUN=false) "
                "but credentials are missing. "
                "Set POLYMARKET_API_KEY, POLYMARKET_API_SECRET, and "
                "POLYMARKET_API_PASSPHRASE environment variables."
            )
            raise ValueError(msg)

    @classmethod
    def from_env(cls, *, load_dotenv_file: bool = True) -> Self:
        """Load configuration from environment variables.

        Args:
            load_dotenv_file: If True, load .env file from project root first.

        Returns:
            PolymarketConfig instance with loaded values.
        """
        loaded_env_path: Path | None = None

        if load_dotenv_file:
            # Load from .env file if present (silently skip if not found)
            env_path = Path.cwd() / ".env"
            if env_path.exists():
                load_dotenv(env_path)
                loaded_env_path = env_path
            # Also check parent directories for .env
            for parent in Path.cwd().parents:
                env_path = parent / ".env"
                if env_path.exists():
                    load_dotenv(env_path)
                    loaded_env_path = env_path
                    break

            if loaded_env_path:
                logger.info("CONFIG: Loaded .env file from %s", loaded_env_path)
            else:
                logger.info("CONFIG: No .env file found in cwd or parents")

        def _mask_value(value: str | None, visible_chars: int = 4) -> str:
            """Mask a sensitive value for logging, showing only last N characters."""
            if not value:
                return "<not set>"
            if len(value) <= visible_chars:
                return "****" + value[-visible_chars:] if len(value) > 0 else "<not set>"
            return "****" + value[-visible_chars:]

        def _bool_env(key: str, default: bool = False) -> bool:
            """Parse boolean from environment variable."""
            raw = os.getenv(key)
            if raw is None:
                return default
            value = raw.lower().strip()
            if value in ("1", "true", "yes", "on"):
                return True
            if value in ("0", "false", "no", "off", ""):
                return False
            return default

        # Load raw values for logging
        raw_key = os.getenv("POLYMARKET_API_KEY") or None
        raw_secret = os.getenv("POLYMARKET_API_SECRET") or None
        raw_passphrase = os.getenv("POLYMARKET_API_PASSPHRASE") or None

        logger.info(
            "CONFIG: Environment variables - POLYMARKET_API_KEY=%s (len=%d), "
            "POLYMARKET_API_SECRET=%s (len=%d), POLYMARKET_API_PASSPHRASE=%s (len=%d)",
            _mask_value(raw_key),
            len(raw_key) if raw_key else 0,
            _mask_value(raw_secret),
            len(raw_secret) if raw_secret else 0,
            _mask_value(raw_passphrase),
            len(raw_passphrase) if raw_passphrase else 0,
        )

        config = cls(
            api_key=raw_key,
            api_secret=raw_secret,
            api_passphrase=raw_passphrase,
            dry_run=_bool_env("POLYMARKET_DRY_RUN", default=True),  # Safety default
        )

        logger.info(
            "CONFIG: PolymarketConfig loaded - has_credentials=%s, can_trade=%s, dry_run=%s",
            config.has_credentials,
            config.can_trade,
            config.dry_run,
        )

        return config


def load_config(*, load_dotenv_file: bool = True) -> PolymarketConfig:
    """Load and return Polymarket configuration.

    This is the main entry point for getting configuration.
    Always defaults to dry-run mode for safety.

    Args:
        load_dotenv_file: If True, also load from .env file.

    Returns:
        PolymarketConfig with loaded values.

    Example:
        >>> from polymarket.config import load_config
        >>> config = load_config()
        >>> print(f"Dry-run mode: {config.dry_run}")
        >>> if config.can_trade:
        ...     print("Live trading enabled")
    """
    return PolymarketConfig.from_env(load_dotenv_file=load_dotenv_file)


def validate_credentials_diagnostic(config: PolymarketConfig | None = None) -> dict:
    """Run comprehensive credential diagnostic.

    Returns detailed information about credential status and environment.

    Args:
        config: Optional pre-loaded config (loads if not provided)

    Returns:
        Dict with diagnostic information
    """
    if config is None:
        config = load_config()

    result = {
        "has_credentials": config.has_credentials,
        "can_trade": config.can_trade,
        "dry_run": config.dry_run,
        "env_vars": {
            "POLYMARKET_API_KEY": "set" if config.api_key else "missing",
            "POLYMARKET_API_SECRET": "set" if config.api_secret else "missing",
            "POLYMARKET_API_PASSPHRASE": "set" if config.api_passphrase else "missing",
            "POLYMARKET_DRY_RUN": str(config.dry_run),
        },
        "lengths": {
            "api_key": len(config.api_key) if config.api_key else 0,
            "api_secret": len(config.api_secret) if config.api_secret else 0,
            "api_passphrase": len(config.api_passphrase) if config.api_passphrase else 0,
        },
        "working_directory": str(Path.cwd()),
        "env_files_checked": [],
        "recommendations": [],
    }

    # Check for .env files
    env_path = Path.cwd() / ".env"
    result["env_files_checked"].append({"path": str(env_path), "exists": env_path.exists()})
    for parent in Path.cwd().parents:
        env_path = parent / ".env"
        if env_path.exists():
            result["env_files_checked"].append(
                {"path": str(env_path), "exists": True, "found_at": str(parent)}
            )
            break

    # Generate recommendations
    if not config.has_credentials:
        result["recommendations"].append(
            "API credentials are missing. Set POLYMARKET_API_KEY, POLYMARKET_API_SECRET, "
            "and POLYMARKET_API_PASSPHRASE environment variables."
        )
        result["recommendations"].append(
            "Option 1: Export them directly: export POLYMARKET_API_KEY=xxx"
        )
        result["recommendations"].append(
            "Option 2: Use 1Password: source ./scripts/load-env-from-1password.sh"
        )
        result["recommendations"].append(
            "Option 3: Add them to .env file (never commit this file!)"
        )
    elif not config.can_trade:
        result["recommendations"].append(
            "Credentials present but dry_run=True. Set POLYMARKET_DRY_RUN=false for live trading."
        )
    else:
        result["recommendations"].append("Credentials configured. Live trading is enabled.")

    return result
