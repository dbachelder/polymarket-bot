"""Kraken market data provider implementation."""

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

# Kraken public API
KRAKEN_REST_BASE = "https://api.kraken.com"


class KrakenProvider(MarketDataProvider):
    """Kraken market data provider.

    Uses Kraken public API (no authentication required for market data).
    https://docs.kraken.com/rest/
    """

    name = "kraken"

    # Interval mapping from standard to Kraken format (in minutes)
    INTERVAL_MAP = {
        "1m": 1,
        "5m": 5,
        "15m": 15,
        "30m": 30,
        "1h": 60,
        "4h": 240,
        "1d": 1440,
        "1w": 10080,
    }

    def __init__(self, base_url: str = KRAKEN_REST_BASE, timeout: float = 30.0):
        """Initialize Kraken provider.

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
    ) -> dict[str, Any]:
        """Make HTTP request to Kraken API.

        Args:
            endpoint: API endpoint path
            params: Query parameters

        Returns:
            JSON response (the 'result' field)

        Raises:
            ProviderUnavailableError: If the API is unavailable
        """
        url = f"{self.base_url}{endpoint}"
        client = self._get_client()

        try:
            response = client.get(url, params=params)

            if response.status_code in (451, 403):
                raise ProviderUnavailableError(
                    f"Kraken API unavailable (HTTP {response.status_code})"
                )

            response.raise_for_status()
            data = response.json()

            # Kraken returns { "error": [], "result": { ... } }
            if data.get("error"):
                raise ProviderUnavailableError(
                    f"Kraken API error: {data['error']}"
                )

            return data.get("result", {})

        except httpx.HTTPStatusError as e:
            raise ProviderUnavailableError(f"Kraken API error: {e}") from e
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            raise ProviderUnavailableError(f"Kraken API connection error: {e}") from e

    def _normalize_symbol(self, symbol: str) -> str:
        """Convert symbol to Kraken format (e.g., BTCUSDT -> XBTUSD).

        Args:
            symbol: Symbol in various formats

        Returns:
            Kraken pair format
        """
        symbol = symbol.upper()

        # Handle already-normalized format
        if "/" in symbol:
            return symbol.replace("/", "")

        # Convert common quote currencies to Kraken format
        # Kraken uses XXBT for BTC, but recent trades API accepts XBT
        quote_mapping = {
            "USDT": "USD",  # Kraken uses USD for both USD and USDT
            "USDC": "USDC",
            "USD": "USD",
            "EUR": "EUR",
            "GBP": "GBP",
            "JPY": "JPY",
        }

        base_mapping = {
            "BTC": "XBT",
        }

        # Find the quote currency
        for quote in sorted(quote_mapping.keys(), key=len, reverse=True):
            if symbol.endswith(quote):
                base = symbol[:-len(quote)]
                kraken_base = base_mapping.get(base, base)
                kraken_quote = quote_mapping[quote]
                return f"{kraken_base}{kraken_quote}"

        # Default: assume USD quote
        base = symbol
        kraken_base = base_mapping.get(base, base)
        return f"{kraken_base}USD"

    def get_agg_trades(
        self,
        symbol: str = "BTCUSDT",
        start_time_ms: int | None = None,
        end_time_ms: int | None = None,
        limit: int = 1000,
    ) -> list[AggTrade]:
        """Fetch recent trades from Kraken.

        Note: Kraken returns individual trades, not aggregated.
        We return them as-is since Kraken doesn't aggregate by timestamp.

        Args:
            symbol: Trading pair symbol
            start_time_ms: Start time in milliseconds
            end_time_ms: End time in milliseconds
            limit: Maximum number of trades

        Returns:
            List of trades
        """
        pair = self._normalize_symbol(symbol)

        params: dict[str, Any] = {"pair": pair}
        if limit:
            params["count"] = min(limit, 1000)

        result = self._make_request("/0/public/Trades", params)

        # Kraken returns { "XXBTZUSD": [[price, volume, time, side, orderType, misc], ...], "last": ... }
        # Access the pair data
        trades_data = result.get(pair, [])

        trades = []
        for item in trades_data:
            # item format: [price, volume, time, side, orderType, misc]
            # time is in seconds with decimal
            time_sec = float(item[2])
            timestamp_ms = int(time_sec * 1000)

            # Filter by time range if specified
            if start_time_ms is not None and timestamp_ms < start_time_ms:
                continue
            if end_time_ms is not None and timestamp_ms > end_time_ms:
                continue

            price = float(item[0])
            volume = float(item[1])
            side = item[3]  # 'b' = buy, 's' = sell

            # side='b' means buyer is aggressor (taker is buyer)
            # side='s' means seller is aggressor (taker is seller)
            # In Binance terms: is_buyer_maker = (side == 's')
            is_buyer_maker = side == "s"

            # Generate a trade ID from timestamp and index (Kraken doesn't provide IDs)
            trade_id = int(time_sec * 1000) + len(trades)

            trades.append(
                AggTrade(
                    timestamp_ms=timestamp_ms,
                    price=price,
                    quantity=volume,
                    is_buyer_maker=is_buyer_maker,
                    trade_id=trade_id,
                )
            )

        # Sort by timestamp
        trades.sort(key=lambda t: t.timestamp_ms)

        return trades[:limit]

    def get_klines(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "1m",
        start_time_ms: int | None = None,
        end_time_ms: int | None = None,
        limit: int = 720,  # Kraken limits OHLC data
    ) -> list[Kline]:
        """Fetch klines (OHLC data) from Kraken.

        Args:
            symbol: Trading pair symbol
            interval: Kline interval (1m, 5m, 15m, etc.)
            start_time_ms: Start time in milliseconds
            end_time_ms: End time in milliseconds
            limit: Maximum number of candles

        Returns:
            List of klines
        """
        pair = self._normalize_symbol(symbol)
        interval_minutes = self.INTERVAL_MAP.get(interval, 1)

        params: dict[str, Any] = {
            "pair": pair,
            "interval": interval_minutes,
        }

        # Kraken uses seconds since epoch
        if start_time_ms is not None:
            params["since"] = start_time_ms // 1000

        result = self._make_request("/0/public/OHLC", params)

        # Kraken returns { "XXBTZUSD": [[time, open, high, low, close, vwap, volume, count], ...], "last": ... }
        ohlc_data = result.get(pair, [])

        klines = []
        for item in ohlc_data:
            # item format: [time, open, high, low, close, vwap, volume, count]
            time_sec = int(item[0])
            open_time_ms = time_sec * 1000
            # Kraken intervals are in minutes
            granularity_sec = interval_minutes * 60
            close_time_ms = open_time_ms + (granularity_sec * 1000) - 1

            # Filter by end time if specified
            if end_time_ms is not None and open_time_ms > end_time_ms:
                break

            klines.append(
                Kline(
                    open_time_ms=open_time_ms,
                    close_time_ms=close_time_ms,
                    open_price=float(item[1]),
                    high_price=float(item[2]),
                    low_price=float(item[3]),
                    close_price=float(item[4]),
                    volume=float(item[6]),  # index 6 is volume
                    quote_volume=0.0,  # Not directly provided, vwap available at index 5
                    trades_count=int(item[7]),  # index 7 is count
                    taker_buy_volume=0.0,  # Not provided by Kraken
                    taker_buy_quote_volume=0.0,  # Not provided
                )
            )

        # Sort by open_time
        klines.sort(key=lambda k: k.open_time_ms)

        return klines[:limit]

    def close(self) -> None:
        """Close HTTP client connection."""
        if self._client and not self._client.is_closed:
            self._client.close()
