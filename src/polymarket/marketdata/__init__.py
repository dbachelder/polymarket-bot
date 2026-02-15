"""Market data provider abstraction for crypto spot trading data.

Provides a unified interface for fetching BTC spot trades and klines (candlesticks)
from multiple exchanges (Binance, Coinbase, Kraken).

All timestamps are UTC. Data is normalized to a common schema compatible with
existing Snapshot/Kline/AggTrade pipeline.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AggTrade:
    """Aggregated trade - normalized across all providers."""

    timestamp_ms: int
    price: float
    quantity: float
    is_buyer_maker: bool
    trade_id: int
    timestamp: str = field(init=False)

    def __post_init__(self):
        ts = datetime.fromtimestamp(self.timestamp_ms / 1000, tz=UTC).isoformat()
        object.__setattr__(self, "timestamp", ts)

    @property
    def signed_volume(self) -> float:
        """Signed volume: positive for buy aggressor, negative for sell aggressor."""
        return -self.quantity if self.is_buyer_maker else self.quantity

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "timestamp_ms": self.timestamp_ms,
            "price": self.price,
            "quantity": self.quantity,
            "is_buyer_maker": self.is_buyer_maker,
            "trade_id": self.trade_id,
            "signed_volume": self.signed_volume,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AggTrade:
        return cls(
            timestamp_ms=data["timestamp_ms"],
            price=data["price"],
            quantity=data["quantity"],
            is_buyer_maker=data["is_buyer_maker"],
            trade_id=data["trade_id"],
        )


@dataclass(frozen=True)
class Kline:
    """Kline (candlestick) data - normalized across all providers."""

    open_time_ms: int
    close_time_ms: int
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: float
    quote_volume: float
    trades_count: int
    taker_buy_volume: float
    taker_buy_quote_volume: float
    open_time: str = field(init=False)
    close_time: str = field(init=False)

    def __post_init__(self):
        open_ts = datetime.fromtimestamp(self.open_time_ms / 1000, tz=UTC).isoformat()
        close_ts = datetime.fromtimestamp(self.close_time_ms / 1000, tz=UTC).isoformat()
        object.__setattr__(self, "open_time", open_ts)
        object.__setattr__(self, "close_time", close_ts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "open_time": self.open_time,
            "close_time": self.close_time,
            "open_time_ms": self.open_time_ms,
            "close_time_ms": self.close_time_ms,
            "open_price": self.open_price,
            "high_price": self.high_price,
            "low_price": self.low_price,
            "close_price": self.close_price,
            "volume": self.volume,
            "quote_volume": self.quote_volume,
            "trades_count": self.trades_count,
            "taker_buy_volume": self.taker_buy_volume,
            "taker_buy_quote_volume": self.taker_buy_quote_volume,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Kline:
        return cls(
            open_time_ms=data["open_time_ms"],
            close_time_ms=data["close_time_ms"],
            open_price=data["open_price"],
            high_price=data["high_price"],
            low_price=data["low_price"],
            close_price=data["close_price"],
            volume=data["volume"],
            quote_volume=data["quote_volume"],
            trades_count=data["trades_count"],
            taker_buy_volume=data["taker_buy_volume"],
            taker_buy_quote_volume=data["taker_buy_quote_volume"],
        )


@dataclass
class Snapshot:
    """Time-aligned snapshot of market data - normalized across all providers."""

    timestamp: str
    timestamp_ms: int
    symbol: str
    provider: str
    trades: list[AggTrade] = field(default_factory=list)
    klines: dict[str, Kline | None] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "timestamp_ms": self.timestamp_ms,
            "symbol": self.symbol,
            "provider": self.provider,
            "trades": [t.to_dict() for t in self.trades],
            "klines": {k: (v.to_dict() if v else None) for k, v in self.klines.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Snapshot:
        return cls(
            timestamp=data["timestamp"],
            timestamp_ms=data["timestamp_ms"],
            symbol=data["symbol"],
            provider=data.get("provider", "unknown"),
            trades=[AggTrade.from_dict(t) for t in data.get("trades", [])],
            klines={
                k: (Kline.from_dict(v) if v else None)
                for k, v in data.get("klines", {}).items()
            },
        )


class MarketDataProvider(ABC):
    """Abstract base class for market data providers.

    All providers must implement methods for fetching:
    - Recent trades (AggTrade)
    - Klines/candlesticks (Kline)
    - Full snapshots combining both

    The interface is designed to be provider-agnostic, with normalized data types.
    """

    name: str = "unknown"

    @abstractmethod
    def get_agg_trades(
        self,
        symbol: str,
        start_time_ms: int | None = None,
        end_time_ms: int | None = None,
        limit: int = 1000,
    ) -> list[AggTrade]:
        """Fetch aggregated trades.

        Args:
            symbol: Trading pair symbol (e.g., "BTCUSDT", "BTC-USD")
            start_time_ms: Start time in milliseconds since epoch
            end_time_ms: End time in milliseconds since epoch
            limit: Maximum number of trades to fetch

        Returns:
            List of aggregated trades
        """
        raise NotImplementedError

    @abstractmethod
    def get_klines(
        self,
        symbol: str,
        interval: str = "1m",
        start_time_ms: int | None = None,
        end_time_ms: int | None = None,
        limit: int = 1000,
    ) -> list[Kline]:
        """Fetch klines (candlestick data).

        Args:
            symbol: Trading pair symbol
            interval: Kline interval (1m, 5m, 15m, 1h, etc.)
            start_time_ms: Start time in milliseconds
            end_time_ms: End time in milliseconds
            limit: Maximum number of klines to fetch

        Returns:
            List of klines
        """
        raise NotImplementedError

    def get_snapshot(
        self,
        symbol: str,
        kline_intervals: list[str] | None = None,
        trade_lookback_seconds: int = 60,
    ) -> Snapshot:
        """Fetch a complete snapshot including recent trades and klines.

        Args:
            symbol: Trading pair symbol
            kline_intervals: List of kline intervals to fetch
            trade_lookback_seconds: How far back to fetch trades

        Returns:
            Snapshot with all requested data
        """
        now = datetime.now(UTC)
        timestamp_ms = int(now.timestamp() * 1000)

        # Fetch recent trades
        start_time_ms = timestamp_ms - (trade_lookback_seconds * 1000)
        trades = self.get_agg_trades(
            symbol=symbol,
            start_time_ms=start_time_ms,
            end_time_ms=timestamp_ms,
            limit=1000,
        )

        # Fetch klines for each interval
        klines: dict[str, Kline | None] = {}
        for interval in kline_intervals or ["1m", "5m"]:
            try:
                kl = self.get_klines(symbol=symbol, interval=interval, limit=10)
                klines[interval] = kl[-1] if kl else None
            except Exception as e:
                logger.warning("Failed to fetch klines for %s: %s", interval, e)
                klines[interval] = None

        return Snapshot(
            timestamp=now.isoformat(),
            timestamp_ms=timestamp_ms,
            symbol=symbol,
            provider=self.name,
            trades=trades,
            klines=klines,
        )

    def health_check(self) -> bool:
        """Check if the provider is accessible and returning valid data.

        Returns:
            True if healthy, False otherwise
        """
        try:
            # Try to fetch a minimal amount of data
            trades = self.get_agg_trades(symbol="BTCUSDT", limit=1)
            return len(trades) >= 0  # Empty list is OK, just need success
        except Exception as e:
            logger.warning("Health check failed for %s: %s", self.name, e)
            return False

    @abstractmethod
    def close(self) -> None:
        """Close any open connections."""
        raise NotImplementedError

    def __enter__(self) -> MarketDataProvider:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


class ProviderError(Exception):
    """Base exception for provider errors."""
    pass


class ProviderUnavailableError(ProviderError):
    """Raised when a provider is unavailable (e.g., HTTP 451, 503)."""
    pass


class ProviderRateLimitError(ProviderError):
    """Raised when rate limit is hit."""
    pass
