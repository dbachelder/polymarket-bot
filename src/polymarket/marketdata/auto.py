"""Auto provider with fallback logic for market data.

Tries providers in order: Binance -> Coinbase -> Kraken
Automatically falls back when a provider is unavailable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TypeVar

from polymarket.marketdata import (
    AggTrade,
    Kline,
    MarketDataProvider,
    ProviderUnavailableError,
    Snapshot,
)
from polymarket.marketdata.binance import BinanceProvider
from polymarket.marketdata.coinbase import CoinbaseProvider
from polymarket.marketdata.kraken import KrakenProvider

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class ProviderHealth:
    """Health status for a provider."""

    name: str
    healthy: bool
    error: str | None = None


class AutoProvider(MarketDataProvider):
    """Auto provider with automatic fallback between multiple providers.

    Attempts providers in order:
    1. Binance (primary, if available)
    2. Coinbase (fallback 1)
    3. Kraken (fallback 2)

    On the first successful call, the working provider is cached until
    it fails, at which point fallback resumes.
    """

    name = "auto"

    def __init__(
        self,
        preferred_order: list[str] | None = None,
        binance_base_url: str | None = None,
        timeout: float = 30.0,
    ):
        """Initialize auto provider.

        Args:
            preferred_order: List of provider names in preferred order
            binance_base_url: Optional override for Binance base URL
            timeout: Request timeout in seconds
        """
        self.preferred_order = preferred_order or ["binance", "coinbase", "kraken"]
        self.binance_base_url = binance_base_url
        self.timeout = timeout

        # Provider instances (lazy initialization)
        self._providers: dict[str, MarketDataProvider] = {}
        self._current_provider: MarketDataProvider | None = None
        self._current_provider_name: str | None = None

    def _get_provider(self, name: str) -> MarketDataProvider:
        """Get or create a provider instance."""
        if name not in self._providers:
            if name == "binance":
                self._providers[name] = BinanceProvider(
                    base_url=self.binance_base_url,
                    timeout=self.timeout,
                )
            elif name == "coinbase":
                self._providers[name] = CoinbaseProvider(timeout=self.timeout)
            elif name == "kraken":
                self._providers[name] = KrakenProvider(timeout=self.timeout)
            else:
                raise ValueError(f"Unknown provider: {name}")
        return self._providers[name]

    def _try_providers(
        self,
        operation: str,
        fn: callable[[MarketDataProvider], T],
    ) -> tuple[T, MarketDataProvider]:
        """Try operation on providers in order until one succeeds.

        Args:
            operation: Description of the operation for logging
            fn: Function to call on each provider

        Returns:
            Tuple of (result, successful_provider)

        Raises:
            ProviderUnavailableError: If all providers fail
        """
        errors: list[tuple[str, Exception]] = []

        # Try current provider first if we have one
        if self._current_provider is not None:
            try:
                result = fn(self._current_provider)
                return result, self._current_provider
            except ProviderUnavailableError as e:
                logger.warning(
                    "Current provider %s failed for %s: %s",
                    self._current_provider_name,
                    operation,
                    e,
                )
                self._current_provider = None
                self._current_provider_name = None

        # Try each provider in preferred order
        for name in self.preferred_order:
            try:
                provider = self._get_provider(name)
                result = fn(provider)

                # Success! Cache this provider
                self._current_provider = provider
                self._current_provider_name = name
                logger.info("Using provider: %s for %s", name, operation)

                return result, provider

            except ProviderUnavailableError as e:
                logger.warning("Provider %s unavailable for %s: %s", name, operation, e)
                errors.append((name, e))

        # All providers failed
        error_msg = "; ".join(f"{name}: {e}" for name, e in errors)
        raise ProviderUnavailableError(f"All providers failed for {operation}: {error_msg}")

    def get_agg_trades(
        self,
        symbol: str = "BTCUSDT",
        start_time_ms: int | None = None,
        end_time_ms: int | None = None,
        limit: int = 1000,
    ) -> list[AggTrade]:
        """Fetch aggregated trades from the first available provider."""
        result, _ = self._try_providers(
            "get_agg_trades",
            lambda p: p.get_agg_trades(symbol, start_time_ms, end_time_ms, limit),
        )
        return result

    def get_klines(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "1m",
        start_time_ms: int | None = None,
        end_time_ms: int | None = None,
        limit: int = 1000,
    ) -> list[Kline]:
        """Fetch klines from the first available provider."""
        result, _ = self._try_providers(
            "get_klines",
            lambda p: p.get_klines(symbol, interval, start_time_ms, end_time_ms, limit),
        )
        return result

    def get_snapshot(
        self,
        symbol: str = "BTCUSDT",
        kline_intervals: list[str] | None = None,
        trade_lookback_seconds: int = 60,
    ) -> Snapshot:
        """Fetch snapshot from the first available provider."""
        result, provider = self._try_providers(
            "get_snapshot",
            lambda p: p.get_snapshot(symbol, kline_intervals, trade_lookback_seconds),
        )
        return result

    def health_check(self) -> bool:
        """Check if any provider is healthy."""
        for name in self.preferred_order:
            try:
                provider = self._get_provider(name)
                if provider.health_check():
                    return True
            except Exception:
                continue
        return False

    def check_all_health(self) -> list[ProviderHealth]:
        """Check health of all providers.

        Returns:
            List of health status for each provider
        """
        results = []
        for name in self.preferred_order:
            try:
                provider = self._get_provider(name)
                healthy = provider.health_check()
                results.append(ProviderHealth(name=name, healthy=healthy, error=None))
            except Exception as e:
                results.append(ProviderHealth(name=name, healthy=False, error=str(e)))
        return results

    def close(self) -> None:
        """Close all provider connections."""
        for provider in self._providers.values():
            provider.close()
        self._providers.clear()
        self._current_provider = None
        self._current_provider_name = None


def create_provider(
    name: str,
    binance_base_url: str | None = None,
    timeout: float = 30.0,
) -> MarketDataProvider:
    """Factory function to create a provider by name.

    Args:
        name: Provider name ("binance", "coinbase", "kraken", "auto")
        binance_base_url: Optional override for Binance base URL
        timeout: Request timeout in seconds

    Returns:
        MarketDataProvider instance

    Raises:
        ValueError: If provider name is unknown
    """
    name = name.lower()

    if name == "binance":
        return BinanceProvider(base_url=binance_base_url, timeout=timeout)
    elif name == "coinbase":
        return CoinbaseProvider(timeout=timeout)
    elif name == "kraken":
        return KrakenProvider(timeout=timeout)
    elif name == "auto":
        return AutoProvider(binance_base_url=binance_base_url, timeout=timeout)
    else:
        raise ValueError(f"Unknown provider: {name}. Use: binance, coinbase, kraken, auto")


def get_available_providers() -> list[str]:
    """Get list of available provider names.

    Returns:
        List of provider names
    """
    return ["binance", "coinbase", "kraken", "auto"]
