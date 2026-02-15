"""Binance public market data collector for BTC spot trading.

Provides WebSocket and REST API collectors for aggregated trades and klines (candlesticks).
All timestamps are UTC. Data is written to JSON/CSV for time-aligned analysis.
"""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import os
import random
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Callable

import httpx
import websockets
from websockets.exceptions import ConnectionClosed, ConnectionClosedOK

logger = logging.getLogger(__name__)

# Binance public API endpoints (no authentication required)
#
# NOTE: Some hosts/IPs receive HTTP 451 from api.binance.com ("Unavailable For Legal Reasons").
# We support endpoint rotation + env overrides.
BINANCE_REST_BASE = "https://api.binance.com"
BINANCE_REST_FALLBACK_BASES = [
    "https://api.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
    "https://api4.binance.com",
]
BINANCE_WS_BASE = "wss://stream.binance.com:9443/ws"

# Default symbol for BTC spot (USDT pair)
DEFAULT_SYMBOL = "BTCUSDT"


@dataclass(frozen=True)
class AggTrade:
    """Aggregated trade from Binance."""

    timestamp_ms: int
    price: float
    quantity: float
    is_buyer_maker: bool
    trade_id: int
    timestamp: str = field(init=False)

    def __post_init__(self):
        # Use object.__setattr__ since dataclass is frozen
        ts = datetime.fromtimestamp(self.timestamp_ms / 1000, tz=UTC).isoformat()
        object.__setattr__(self, "timestamp", ts)

    @property
    def signed_volume(self) -> float:
        """Signed volume: positive for sell aggressor (buyer maker), negative for buy aggressor."""
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


@dataclass(frozen=True)
class Kline:
    """Kline (candlestick) data from Binance."""

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


@dataclass
class Snapshot:
    """Time-aligned snapshot of Binance market data."""

    timestamp: str
    timestamp_ms: int
    symbol: str
    trades: list[AggTrade] = field(default_factory=list)
    klines: dict[str, Kline | None] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "timestamp_ms": self.timestamp_ms,
            "symbol": self.symbol,
            "trades": [t.to_dict() for t in self.trades],
            "klines": {k: (v.to_dict() if v else None) for k, v in self.klines.items()},
        }


class BinanceRestClient:
    """REST API client for Binance public market data.

    Supports endpoint rotation (HTTP 451 / connectivity issues) and env overrides.

    Env:
      - BINANCE_REST_BASE: single base URL (e.g. https://api1.binance.com)
      - BINANCE_REST_BASE_URLS: comma-separated base URLs (highest priority)
    """

    def __init__(
        self,
        base_url: str = BINANCE_REST_BASE,
        timeout: float = 30.0,
        base_urls: list[str] | None = None,
    ):
        self.timeout = timeout
        self.base_urls = base_urls or self._get_base_urls(default_base=base_url)
        self._base_url_idx = 0
        self.base_url = self.base_urls[self._base_url_idx]
        self.client = httpx.Client(timeout=timeout, follow_redirects=True)

    @staticmethod
    def _get_base_urls(default_base: str) -> list[str]:
        # Highest priority: explicit list
        raw_list = os.environ.get("BINANCE_REST_BASE_URLS")
        if raw_list:
            urls = [u.strip() for u in raw_list.split(",") if u.strip()]
            if urls:
                return urls

        # Next: single override
        raw_single = os.environ.get("BINANCE_REST_BASE")
        if raw_single and raw_single.strip():
            return [raw_single.strip()]

        # Default: include fallbacks (and keep default_base first)
        bases = [default_base]
        for u in BINANCE_REST_FALLBACK_BASES:
            if u not in bases:
                bases.append(u)
        return bases

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> BinanceRestClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _get(self, endpoint: str, params: dict[str, Any] | None = None) -> Any:
        """Make a GET request to Binance API.

        Retries across configured base URLs on common geo/legal blocks (HTTP 451)
        and transient network errors.
        """
        last_exc: Exception | None = None

        for i, base in enumerate(self.base_urls):
            self.base_url = base
            url = f"{base}{endpoint}"
            try:
                response = self.client.get(url, params=params)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                last_exc = e
                status = e.response.status_code
                # 451 often indicates region/legal block on this endpoint.
                if status in (451, 418, 429) and i < len(self.base_urls) - 1:
                    logger.warning(
                        "Binance REST blocked (%s) on %s; trying next endpoint...",
                        status,
                        base,
                    )
                    continue
                raise
            except (httpx.ConnectError, httpx.ReadError, httpx.TimeoutException) as e:
                last_exc = e
                if i < len(self.base_urls) - 1:
                    logger.warning(
                        "Binance REST connectivity error on %s (%s); trying next endpoint...",
                        base,
                        type(e).__name__,
                    )
                    continue
                raise

        if last_exc:
            raise last_exc
        raise RuntimeError("Binance REST request failed without exception")

    def get_agg_trades(
        self,
        symbol: str = DEFAULT_SYMBOL,
        start_time_ms: int | None = None,
        end_time_ms: int | None = None,
        limit: int = 1000,
    ) -> list[AggTrade]:
        """Fetch aggregated trades from REST API.

        Args:
            symbol: Trading pair symbol
            start_time_ms: Start time in milliseconds
            end_time_ms: End time in milliseconds
            limit: Maximum number of trades to fetch (max 1000)

        Returns:
            List of aggregated trades
        """
        params: dict[str, Any] = {"symbol": symbol, "limit": min(limit, 1000)}
        if start_time_ms is not None:
            params["startTime"] = start_time_ms
        if end_time_ms is not None:
            params["endTime"] = end_time_ms

        raw_data: Any = self._get("/api/v3/aggTrades", params)

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
        symbol: str = DEFAULT_SYMBOL,
        interval: str = "1m",
        start_time_ms: int | None = None,
        end_time_ms: int | None = None,
        limit: int = 1000,
    ) -> list[Kline]:
        """Fetch klines (candlestick data) from REST API.

        Args:
            symbol: Trading pair symbol
            interval: Kline interval (1s, 1m, 3m, 5m, 15m, 30m, 1h, etc.)
            start_time_ms: Start time in milliseconds
            end_time_ms: End time in milliseconds
            limit: Maximum number of klines to fetch (max 1000)

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

        raw_data: Any = self._get("/api/v3/klines", params)

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


class BinanceWebSocketCollector:
    """WebSocket collector for Binance real-time market data."""

    def __init__(
        self,
        symbol: str = DEFAULT_SYMBOL,
        ws_base: str = BINANCE_WS_BASE,
        max_reconnect_delay: float = 60.0,
        trade_buffer_size: int = 10000,
    ):
        self.symbol = symbol.lower()
        self.ws_base = ws_base
        self.max_reconnect_delay = max_reconnect_delay
        self.trade_buffer: deque[AggTrade] = deque(maxlen=trade_buffer_size)
        self.klines: dict[str, Kline | None] = {}
        self._running = False
        self._reconnect_delay = 1.0
        self._ws: Any = None

    def _get_stream_url(self, streams: list[str]) -> str:
        """Build WebSocket URL for combined streams."""
        if len(streams) == 1:
            return f"{self.ws_base}/{streams[0]}"
        combined = "/".join(streams)
        return f"{self.ws_base}/stream?streams={combined}"

    def _parse_agg_trade(self, data: dict[str, Any]) -> AggTrade:
        """Parse aggregated trade from WebSocket message."""
        return AggTrade(
            timestamp_ms=data["T"],
            price=float(data["p"]),
            quantity=float(data["q"]),
            is_buyer_maker=data["m"],
            trade_id=data["a"],
        )

    def _parse_kline(self, data: dict[str, Any]) -> tuple[str, Kline]:
        """Parse kline from WebSocket message. Returns (interval, kline)."""
        k = data["k"]
        interval = k["i"]
        return (
            interval,
            Kline(
                open_time_ms=k["t"],
                close_time_ms=k["T"],
                open_price=float(k["o"]),
                high_price=float(k["h"]),
                low_price=float(k["l"]),
                close_price=float(k["c"]),
                volume=float(k["v"]),
                quote_volume=float(k["q"]),
                trades_count=k["n"],
                taker_buy_volume=float(k["V"]),
                taker_buy_quote_volume=float(k["Q"]),
            ),
        )

    async def _connect_and_listen(
        self,
        kline_intervals: list[str] | None = None,
        on_snapshot: Callable[[Snapshot], None] | None = None,
        snapshot_interval_seconds: float = 5.0,
    ) -> None:
        """Connect to WebSocket and listen for messages."""
        streams = [f"{self.symbol}@aggTrade"]
        intervals = kline_intervals or ["1m"]
        for interval in intervals:
            streams.append(f"{self.symbol}@kline_{interval}")

        url = self._get_stream_url(streams)
        logger.info("Connecting to Binance WebSocket: %s", url)

        try:
            async with websockets.connect(url) as ws:
                self._ws = ws
                self._reconnect_delay = 1.0  # Reset on successful connection
                logger.info("Connected to Binance WebSocket")

                last_snapshot = time.time()

                async for message in ws:
                    if not self._running:
                        break

                    try:
                        data = json.loads(message)

                        # Handle combined stream format
                        if "stream" in data:
                            stream = data["stream"]
                            payload = data["data"]
                        else:
                            stream = ""
                            payload = data

                        if "aggTrade" in stream or ("e" in payload and payload["e"] == "aggTrade"):
                            trade = self._parse_agg_trade(payload)
                            self.trade_buffer.append(trade)

                        elif "kline" in stream or ("e" in payload and payload["e"] == "kline"):
                            interval, kline = self._parse_kline(payload)
                            self.klines[interval] = kline

                    except (json.JSONDecodeError, KeyError, ValueError) as e:
                        logger.warning("Failed to parse message: %s", e)
                        continue

                    # Generate periodic snapshots
                    if on_snapshot and time.time() - last_snapshot >= snapshot_interval_seconds:
                        snapshot = self._create_snapshot()
                        on_snapshot(snapshot)
                        last_snapshot = time.time()

        except ConnectionClosedOK:
            logger.info("WebSocket connection closed normally")
        except ConnectionClosed as e:
            logger.warning("WebSocket connection closed: %s", e)
            raise

    def _create_snapshot(self) -> Snapshot:
        """Create a snapshot of current market data."""
        now = datetime.now(UTC)
        return Snapshot(
            timestamp=now.isoformat(),
            timestamp_ms=int(now.timestamp() * 1000),
            symbol=self.symbol.upper(),
            trades=list(self.trade_buffer),
            klines=dict(self.klines),
        )

    async def run(
        self,
        kline_intervals: list[str] | None = None,
        on_snapshot: Callable[[Snapshot], None] | None = None,
        snapshot_interval_seconds: float = 5.0,
    ) -> None:
        """Run the WebSocket collector with automatic reconnection.

        Args:
            kline_intervals: List of kline intervals to subscribe to (e.g., ["1m", "5m"])
            on_snapshot: Callback function called with each snapshot
            snapshot_interval_seconds: How often to generate snapshots
        """
        self._running = True

        while self._running:
            try:
                await self._connect_and_listen(
                    kline_intervals=kline_intervals,
                    on_snapshot=on_snapshot,
                    snapshot_interval_seconds=snapshot_interval_seconds,
                )
            except Exception as e:
                if not self._running:
                    break
                logger.error("WebSocket error: %s", e)

            # Exponential backoff with jitter
            if self._running:
                delay = self._reconnect_delay + random.uniform(0, 1.0)
                logger.info("Reconnecting in %.1f seconds...", delay)
                await asyncio.sleep(delay)
                self._reconnect_delay = min(self.max_reconnect_delay, self._reconnect_delay * 1.5)

    def stop(self) -> None:
        """Stop the collector."""
        self._running = False
        if self._ws:
            asyncio.create_task(self._ws.close())


def collect_snapshot_rest(
    out_dir: Path,
    symbol: str = DEFAULT_SYMBOL,
    kline_intervals: list[str] | None = None,
) -> Path:
    """Collect a single snapshot using REST API.

    Args:
        out_dir: Directory to write output files
        symbol: Trading pair symbol
        kline_intervals: Kline intervals to fetch

    Returns:
        Path to the written JSON file
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    kline_intervals = kline_intervals or ["1m", "5m"]

    now = datetime.now(UTC)
    ts_str = now.strftime("%Y%m%dT%H%M%SZ")

    with BinanceRestClient() as client:
        # Fetch recent trades (last 1000)
        end_time = int(now.timestamp() * 1000)
        start_time = end_time - (60 * 1000)  # Last 60 seconds
        trades = client.get_agg_trades(
            symbol=symbol, start_time_ms=start_time, end_time_ms=end_time, limit=1000
        )

        # Fetch klines for each interval
        klines: dict[str, Kline | None] = {}
        for interval in kline_intervals:
            try:
                kl = client.get_klines(symbol=symbol, interval=interval, limit=10)
                klines[interval] = kl[-1] if kl else None
            except httpx.HTTPError as e:
                logger.warning("Failed to fetch klines for %s: %s", interval, e)
                klines[interval] = None

    snapshot = Snapshot(
        timestamp=now.isoformat(),
        timestamp_ms=int(now.timestamp() * 1000),
        symbol=symbol,
        trades=trades,
        klines=klines,
    )

    out_path = out_dir / f"binance_{symbol.lower()}_{ts_str}.json"
    out_path.write_text(json.dumps(snapshot.to_dict(), indent=2, sort_keys=True))

    # Also write trades to CSV for easy analysis
    csv_path = out_dir / f"binance_{symbol.lower()}_trades_{ts_str}.csv"
    with csv_path.open("w", newline="") as f:
        if trades:
            writer = csv.DictWriter(f, fieldnames=trades[0].to_dict().keys())
            writer.writeheader()
            for trade in trades:
                writer.writerow(trade.to_dict())

    return out_path


async def collect_loop_ws(
    out_dir: Path,
    symbol: str = DEFAULT_SYMBOL,
    kline_intervals: list[str] | None = None,
    snapshot_interval_seconds: float = 5.0,
    max_reconnect_delay: float = 60.0,
    retention_hours: float | None = None,
) -> None:
    """Continuously collect market data via WebSocket.

    Args:
        out_dir: Directory to write output files
        symbol: Trading pair symbol
        kline_intervals: Kline intervals to subscribe to
        snapshot_interval_seconds: How often to write snapshots
        max_reconnect_delay: Maximum reconnection delay in seconds
        retention_hours: If set, delete files older than this many hours
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    kline_intervals = kline_intervals or ["1m", "5m"]

    collector = BinanceWebSocketCollector(symbol=symbol, max_reconnect_delay=max_reconnect_delay)

    def on_snapshot(snapshot: Snapshot) -> None:
        # Write JSON snapshot
        ts_str = datetime.fromtimestamp(snapshot.timestamp_ms / 1000, tz=UTC).strftime(
            "%Y%m%dT%H%M%SZ"
        )
        out_path = out_dir / f"binance_{symbol.lower()}_{ts_str}.json"
        out_path.write_text(json.dumps(snapshot.to_dict(), indent=2, sort_keys=True))

        # Write latest pointer
        latest = out_dir / f"latest_{symbol.lower()}.json"
        latest.write_text(
            json.dumps(
                {
                    "path": str(out_path),
                    "generated_at": datetime.now(UTC).isoformat(),
                    "timestamp_ms": snapshot.timestamp_ms,
                }
            )
        )

        # Prune old files if retention is set
        if retention_hours is not None:
            _prune_old_files(out_dir, symbol.lower(), retention_hours)

    logger.info(
        "Starting Binance collector for %s (intervals: %s)",
        symbol,
        kline_intervals,
    )

    await collector.run(
        kline_intervals=kline_intervals,
        on_snapshot=on_snapshot,
        snapshot_interval_seconds=snapshot_interval_seconds,
    )


def _prune_old_files(out_dir: Path, symbol: str, retention_hours: float) -> int:
    """Delete files older than retention_hours."""
    cutoff = datetime.now(UTC) - timedelta(hours=retention_hours)
    deleted = 0
    prefix = f"binance_{symbol}_"

    for p in out_dir.glob(f"{prefix}*.json"):
        # Skip "latest" files
        if "latest" in p.name:
            continue
        try:
            mtime = datetime.fromtimestamp(p.stat().st_mtime, tz=UTC)
        except FileNotFoundError:
            continue
        if mtime < cutoff:
            try:
                p.unlink()
                deleted += 1
            except FileNotFoundError:
                pass

    return deleted


def run_collector_loop(
    out_dir: Path,
    symbol: str = DEFAULT_SYMBOL,
    kline_intervals: list[str] | None = None,
    snapshot_interval_seconds: float = 5.0,
    max_reconnect_delay: float = 60.0,
    retention_hours: float | None = None,
) -> None:
    """Synchronous entry point to run the WebSocket collector loop."""
    asyncio.run(
        collect_loop_ws(
            out_dir=out_dir,
            symbol=symbol,
            kline_intervals=kline_intervals,
            snapshot_interval_seconds=snapshot_interval_seconds,
            max_reconnect_delay=max_reconnect_delay,
            retention_hours=retention_hours,
        )
    )
