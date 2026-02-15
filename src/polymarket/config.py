"""Configuration management for Polymarket bot.

Loads credentials from environment or .env file with safe defaults.
All trading operations default to dry-run mode unless explicitly disabled.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from dotenv import load_dotenv


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
        if load_dotenv_file:
            # Load from .env file if present (silently skip if not found)
            env_path = Path.cwd() / ".env"
            if env_path.exists():
                load_dotenv(env_path)
            # Also check parent directories for .env
            for parent in Path.cwd().parents:
                env_path = parent / ".env"
                if env_path.exists():
                    load_dotenv(env_path)
                    break

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

        return cls(
            api_key=os.getenv("POLYMARKET_API_KEY") or None,
            api_secret=os.getenv("POLYMARKET_API_SECRET") or None,
            api_passphrase=os.getenv("POLYMARKET_API_PASSPHRASE") or None,
            dry_run=_bool_env("POLYMARKET_DRY_RUN", default=True),  # Safety default
        )


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
