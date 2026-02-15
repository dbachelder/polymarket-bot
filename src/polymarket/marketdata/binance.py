"""Binance market data provider implementation."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from polymarket.marketdata import (
    AggTrade,
    Kline,
    MarketDataProvider,
    ProviderUnavailableError,
)

logger = logging.getLogger(__name__)

# Default endpoints (can be overridden via env)
DEFAULT_REST_ENDPOINTS = [
    "https://api.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
]

DEFAULT_WS_BASE = "wss://stream.binance.com:9443/ws"


class BinanceProvider(MarketDataProvider):
    """Binance market data provider.

    Uses Binance public API (no authentication required).
    Supports endpoint rotation for resilience.
    """

    name = "binance"

    def __init__(
        self,
        base_url: str | None = None,
        timeout: float = 30.0,
    ):
        """Initialize Binance provider.

        Args:
            base_url: Override base URL (defaults to rotating endpoints)
            timeout: Request timeout in seconds
        """
        self.timeout = timeout
        self.endpoints = [base_url] if base_url else DEFAULT_REST_ENDPOINTS.copy()
        self.current_endpoint_idx = 0
        self._client: httpx.Client | None = None

    def _get_client(self) -> httpx.Client:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.Client(timeout=self.timeout, follow_redirects=True)
        return self._client

    def _get_base_url(self) -> str:
        """Get current endpoint URL."""
        return self.endpoints[self.current_endpoint_idx]

    def _rotate_endpoint(self) -> bool:
        """Rotate to next endpoint. Returns True if there are more endpoints to try."""
        self.current_endpoint_idx += 1
        if self.current_endpoint_idx >= len(self.endpoints):
            self.current_endpoint_idx = 0
            return False
        logger.info("Rotating to Binance endpoint: %s", self._get_base_url())
        return True

    def _make_request(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
        max_retries: int | None = None,
    ) -> dict[str, Any]:
        """Make HTTP request with endpoint rotation on failure.

        Args:
            endpoint: API endpoint path
            params: Query parameters
            max_retries: Maximum number of endpoint rotations

        Returns:
            JSON response

        Raises:
            ProviderUnavailableError: If all endpoints fail
        """
        if max_retries is None:
            max_retries = len(self.endpoints)

        last_error: Exception | None = None
        attempts = 0

        while attempts <= max_retries:
            url = f"{self._get_base_url()}{endpoint}"
            client = self._get_client()

            try:
                response = client.get(url, params=params)

                # Check for HTTP 451 (Unavailable For Legal Reasons)
                if response.status_code == 451:
                    logger.warning("Binance endpoint %s returned HTTP 451", url)
                    if self._rotate_endpoint():
                        attempts += 1
                        continue
                    raise ProviderUnavailableError(
                        "Binance API unavailable (HTTP 451) from all endpoints"
                    )

                response.raise_for_status()
                return response.json()

            except httpx.HTTPStatusError as e:
                last_error = e
                # Don't rotate on 4xx client errors (except 451 handled above)
                if e.response.status_code < 500:
                    raise
                if self._rotate_endpoint():
                    attempts += 1
                    continue
                break

            except (httpx.ConnectError, httpx.TimeoutException) as e:
                last_error = e
                if self._rotate_endpoint():
                    attempts += 1
                    continue
                break

        raise ProviderUnavailableError(
            f"Binance API unavailable after trying {attempts + 1} endpoint(s)"
        ) from last_error

    def get_agg_trades(
        self,
        symbol: str = "BTCUSDT",
        start_time_ms: int | None = None,
        end_time_ms: int | None = None,
        limit: int = 1000,
    ) -> list[AggTrade]:
        """Fetch aggregated trades from Binance.

        Args:
            symbol: Trading pair symbol (default: BTCUSDT)
            start_time_ms: Start time in milliseconds
            end_time_ms: End time in milliseconds
            limit: Maximum number of trades (max 1000)

        Returns:
            List of aggregated trades
        """
        params: dict[str, Any] = {"symbol": symbol, "limit": min(limit, 1000)}
        if start_time_ms is not None:
            params["startTime"] = start_time_ms
        if end_time_ms is not None:
            params["endTime"] = end_time_ms

        raw_data = self._make_request("/api/v3/aggTrades", params)

        return [
            AggTrade(
                timestamp_ms=int(item["T"]),
                price=float(item["p"]),
                quantity=float(item["q"]),
                is_buyer_maker=bool(item["m"]),
                trade_id=int(item["a"]),
            )
            for item in raw_data
        ]

    def get_klines(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "1m",
        start_time_ms: int | None = None,
        end_time_ms: int | None = None,
        limit: int = 1000,
    ) -> list[Kline]:
        """Fetch klines (candlestick data) from Binance.

        Args:
            symbol: Trading pair symbol
            interval: Kline interval (1s, 1m, 3m, 5m, 15m, 30m, 1h, etc.)
            start_time_ms: Start time in milliseconds
            end_time_ms: End time in milliseconds
            limit: Maximum number of klines (max 1000)

        Returns:
            List of klines
        """
        params: dict[str, Any] = {
            "symbol": symbol,
            "interval": interval,
            "limit": min(limit, 1000),
        }
        if start_time_ms is not None:
            params["startTime"] = start_time_ms
        if end_time_ms is not None:
            params["endTime"] = end_time_ms

        raw_data = self._make_request("/api/v3/klines", params)

        return [
            Kline(
                open_time_ms=int(item[0]),
                close_time_ms=int(item[6]),
                open_price=float(item[1]),
                high_price=float(item[2]),
                low_price=float(item[3]),
                close_price=float(item[4]),
                volume=float(item[5]),
                quote_volume=float(item[7]),
                trades_count=int(item[8]),
                taker_buy_volume=float(item[9]),
                taker_buy_quote_volume=float(item[10]),
            )
            for item in raw_data
        ]

    def close(self) -> None:
        """Close HTTP client connection."""
        if self._client and not self._client.is_closed:
            self._client.close()
