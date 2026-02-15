"""Coinbase Exchange market data provider implementation."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

import httpx

from polymarket.marketdata import (
    AggTrade,
    Kline,
    MarketDataProvider,
    ProviderUnavailableError,
)

logger = logging.getLogger(__name__)

# Coinbase Exchange API (not Coinbase Consumer API)
# Public endpoints - no authentication required
COINBASE_REST_BASE = "https://api.exchange.coinbase.com"


class CoinbaseProvider(MarketDataProvider):
    """Coinbase Exchange market data provider.

    Uses Coinbase Exchange public API (not Coinbase Consumer/Pro).
    https://docs.cloud.coinbase.com/exchange/reference/
    """

    name = "coinbase"

    # Interval mapping from standard to Coinbase format
    INTERVAL_MAP = {
        "1m": 60,
        "5m": 300,
        "15m": 900,
        "1h": 3600,
        "6h": 21600,
        "1d": 86400,
    }

    def __init__(self, base_url: str = COINBASE_REST_BASE, timeout: float = 30.0):
        """Initialize Coinbase provider.

        Args:
            base_url: Override base URL
            timeout: Request timeout in seconds
        """
        self.base_url = base_url
        self.timeout = timeout
        self._client: httpx.Client | None = None

    def _get_client(self) -> httpx.Client:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.Client(timeout=self.timeout, follow_redirects=True)
        return self._client

    def _make_request(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
    ) -> Any:
        """Make HTTP request to Coinbase API.

        Args:
            endpoint: API endpoint path
            params: Query parameters

        Returns:
            JSON response

        Raises:
            ProviderUnavailableError: If the API is unavailable
        """
        url = f"{self.base_url}{endpoint}"
        client = self._get_client()

        try:
            response = client.get(url, params=params)

            if response.status_code in (451, 403):
                raise ProviderUnavailableError(
                    f"Coinbase API unavailable (HTTP {response.status_code})"
                )

            response.raise_for_status()
            return response.json()

        except httpx.HTTPStatusError as e:
            raise ProviderUnavailableError(f"Coinbase API error: {e}") from e
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            raise ProviderUnavailableError(f"Coinbase API connection error: {e}") from e

    def _normalize_symbol(self, symbol: str) -> str:
        """Convert symbol to Coinbase format (e.g., BTCUSDT -> BTC-USD).

        Args:
            symbol: Symbol in various formats

        Returns:
            Coinbase product ID format
        """
        # Handle common variations
        symbol = symbol.upper()

        # Already in correct format
        if "-" in symbol:
            return symbol

        # Convert BTCUSDT -> BTC-USD (Coinbase uses USD not USDT for spot)
        if symbol.endswith("USDT"):
            return f"{symbol[:-4]}-USD"
        if symbol.endswith("USD"):
            # Handle BTCUSD -> BTC-USD
            base = symbol[:-3]
            return f"{base}-USD"

        # Default: assume it's a base currency and add USD
        return f"{symbol}-USD"

    def get_agg_trades(
        self,
        symbol: str = "BTCUSDT",
        start_time_ms: int | None = None,
        end_time_ms: int | None = None,
        limit: int = 1000,
    ) -> list[AggTrade]:
        """Fetch recent trades from Coinbase.

        Note: Coinbase doesn't have aggregated trades like Binance.
        We fetch individual trades and aggregate them by timestamp (1-second buckets).

        Args:
            symbol: Trading pair symbol
            start_time_ms: Start time in milliseconds
            end_time_ms: End time in milliseconds
            limit: Maximum number of trades

        Returns:
            List of aggregated trades (1-second aggregation)
        """
        product_id = self._normalize_symbol(symbol)

        # Coinbase uses 'before' cursor for pagination, not time-based
        # We fetch recent trades and filter by time client-side
        params: dict[str, Any] = {"limit": min(limit, 1000)}

        raw_data = self._make_request(f"/products/{product_id}/trades", params)

        # Coinbase returns trades in format:
        # [{"time": "2023-11-14T22:13:20.123456Z", "trade_id": 12345,
        #   "price": "42000.00", "size": "0.5", "side": "buy"}, ...]

        trades = []
        for item in raw_data:
            # Parse timestamp
            time_str = item["time"]
            dt = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
            timestamp_ms = int(dt.timestamp() * 1000)

            # Filter by time range if specified
            if start_time_ms is not None and timestamp_ms < start_time_ms:
                continue
            if end_time_ms is not None and timestamp_ms > end_time_ms:
                continue

            # side: "buy" means buyer is maker (taker is seller)
            # side: "sell" means seller is maker (taker is buyer)
            is_buyer_maker = item["side"] == "buy"

            trades.append(
                AggTrade(
                    timestamp_ms=timestamp_ms,
                    price=float(item["price"]),
                    quantity=float(item["size"]),
                    is_buyer_maker=is_buyer_maker,
                    trade_id=int(item["trade_id"]),
                )
            )

        # Sort by timestamp
        trades.sort(key=lambda t: t.timestamp_ms)

        return trades

    def get_klines(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "1m",
        start_time_ms: int | None = None,
        end_time_ms: int | None = None,
        limit: int = 300,  # Coinbase limits candles
    ) -> list[Kline]:
        """Fetch klines (candlestick data) from Coinbase.

        Args:
            symbol: Trading pair symbol
            interval: Kline interval (1m, 5m, 15m, 1h, etc.)
            start_time_ms: Start time in milliseconds
            end_time_ms: End time in milliseconds
            limit: Maximum number of candles (max 300 for Coinbase)

        Returns:
            List of klines
        """
        product_id = self._normalize_symbol(symbol)

        # Convert interval to granularity in seconds
        granularity = self.INTERVAL_MAP.get(interval, 60)

        params: dict[str, Any] = {
            "granularity": granularity,
        }

        # Add time range if specified
        if start_time_ms is not None:
            # Coinbase uses ISO 8601 format
            start_dt = datetime.fromtimestamp(start_time_ms / 1000, tz=UTC)
            params["start"] = start_dt.isoformat()
        if end_time_ms is not None:
            end_dt = datetime.fromtimestamp(end_time_ms / 1000, tz=UTC)
            params["end"] = end_dt.isoformat()

        raw_data = self._make_request(f"/products/{product_id}/candles", params)

        # Coinbase returns candles in format:
        # [[timestamp, low, high, open, close, volume], ...]
        # Timestamp is in seconds, not milliseconds

        klines = []
        for item in raw_data:
            # item = [time, low, high, open, close, volume]
            time_sec = int(item[0])
            open_time_ms = time_sec * 1000
            # Coinbase candles don't have explicit close time, infer from granularity
            close_time_ms = open_time_ms + (granularity * 1000) - 1

            klines.append(
                Kline(
                    open_time_ms=open_time_ms,
                    close_time_ms=close_time_ms,
                    open_price=float(item[3]),  # index 3 is open
                    high_price=float(item[2]),  # index 2 is high
                    low_price=float(item[1]),   # index 1 is low
                    close_price=float(item[4]), # index 4 is close
                    volume=float(item[5]),
                    quote_volume=0.0,  # Not provided by Coinbase
                    trades_count=0,    # Not provided by Coinbase
                    taker_buy_volume=0.0,  # Not provided
                    taker_buy_quote_volume=0.0,  # Not provided
                )
            )

        # Sort by open_time
        klines.sort(key=lambda k: k.open_time_ms)

        return klines

    def close(self) -> None:
        """Close HTTP client connection."""
        if self._client and not self._client.is_closed:
            self._client.close()
