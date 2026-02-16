"""Pricefeed abstraction with Coinbase (primary) and Kraken (fallback) support.

Provides unified interface for collecting BTC spot price data from public APIs
and WebSocket feeds. Used for aligning Polymarket snapshots with reference prices.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Callable

import httpx
import websockets
from websockets.exceptions import ConnectionClosed, ConnectionClosedOK

logger = logging.getLogger(__name__)

# API endpoints
COINBASE_REST_BASE = "https://api.exchange.coinbase.com"
COINBASE_WS_BASE = "wss://ws-feed.exchange.coinbase.com"

KRAKEN_REST_BASE = "https://api.kraken.com"
KRAKEN_WS_BASE = "wss://ws.kraken.com"

# Default trading pair (BTC/USD)
DEFAULT_SYMBOL = "BTC-USD"
COINBASE_PRODUCT = "BTC-USD"
KRAKEN_PAIR = "XBT/USD"
KRAKEN_WS_PAIR = "XBT/USD"


@dataclass(frozen=True)
class PriceTick:
    """Normalized price tick from any venue."""

    timestamp_ms: int
    price: float
    size: float
    side: str  # 'buy' or 'sell'
    venue: str
    raw_data: dict[str, Any] = field(repr=False)

    @property
    def timestamp(self) -> str:
        return datetime.fromtimestamp(self.timestamp_ms / 1000, tz=UTC).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "timestamp_ms": self.timestamp_ms,
            "price": self.price,
            "size": self.size,
            "side": self.side,
            "venue": self.venue,
        }


@dataclass(frozen=True)
class Trade:
    """Normalized trade from any venue."""

    timestamp_ms: int
    price: float
    size: float
    side: str  # 'buy' or 'sell' (taker side)
    trade_id: str
    venue: str
    raw_data: dict[str, Any] = field(repr=False)

    @property
    def timestamp(self) -> str:
        return datetime.fromtimestamp(self.timestamp_ms / 1000, tz=UTC).isoformat()

    @property
    def signed_volume(self) -> float:
        """Signed volume: positive for buy (taker buy), negative for sell."""
        return self.size if self.side == "buy" else -self.size

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "timestamp_ms": self.timestamp_ms,
            "price": self.price,
            "size": self.size,
            "side": self.side,
            "trade_id": self.trade_id,
            "venue": self.venue,
            "signed_volume": self.signed_volume,
        }


@dataclass
class Snapshot:
    """Time-aligned snapshot of pricefeed data."""

    timestamp: str
    timestamp_ms: int
    symbol: str
    venue: str
    trades: list[Trade] = field(default_factory=list)
    current_price: float | None = None
    bid: float | None = None
    ask: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "timestamp_ms": self.timestamp_ms,
            "symbol": self.symbol,
            "venue": self.venue,
            "current_price": self.current_price,
            "bid": self.bid,
            "ask": self.ask,
            "trades": [t.to_dict() for t in self.trades],
        }


class PricefeedClient(ABC):
    """Abstract base class for pricefeed clients."""

    @abstractmethod
    def get_latest_price(self, symbol: str) -> dict[str, Any]:
        """Get latest price for a symbol."""
        raise NotImplementedError

    @abstractmethod
    def get_recent_trades(self, symbol: str, limit: int = 100) -> list[Trade]:
        """Get recent trades for a symbol."""
        raise NotImplementedError

    @property
    @abstractmethod
    def venue_name(self) -> str:
        """Return the venue name."""
        raise NotImplementedError


class CoinbaseClient(PricefeedClient):
    """Coinbase Exchange API client (REST)."""

    def __init__(self, base_url: str = COINBASE_REST_BASE, timeout: float = 30.0):
        self.base_url = base_url
        self.timeout = timeout
        self.client = httpx.Client(timeout=timeout, follow_redirects=True)

    @property
    def venue_name(self) -> str:
        return "coinbase"

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> CoinbaseClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _get(self, endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Make a GET request to Coinbase API."""
        url = f"{self.base_url}{endpoint}"
        response = self.client.get(url, params=params)
        response.raise_for_status()
        return response.json()

    def get_latest_price(self, symbol: str = COINBASE_PRODUCT) -> dict[str, Any]:
        """Get latest ticker data for a symbol."""
        data = self._get(f"/products/{symbol}/ticker")
        return {
            "price": float(data["price"]),
            "bid": float(data["bid"]),
            "ask": float(data["ask"]),
            "volume": float(data["volume"]),
            "timestamp": data["time"],
        }

    def get_recent_trades(self, symbol: str = COINBASE_PRODUCT, limit: int = 100) -> list[Trade]:
        """Get recent trades for a symbol."""
        data = self._get(f"/products/{symbol}/trades", {"limit": min(limit, 1000)})

        trades = []
        for item in data:
            # Parse timestamp
            ts_str = item.get("time", "")
            if ts_str:
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                ts_ms = int(ts.timestamp() * 1000)
            else:
                ts_ms = int(datetime.now(UTC).timestamp() * 1000)

            # Side: Coinbase uses 'side' field with 'buy' or 'sell'
            side = item.get("side", "buy")

            trades.append(
                Trade(
                    timestamp_ms=ts_ms,
                    price=float(item["price"]),
                    size=float(item["size"]),
                    side=side,
                    trade_id=str(item["trade_id"]),
                    venue="coinbase",
                    raw_data=item,
                )
            )

        return trades


class KrakenClient(PricefeedClient):
    """Kraken API client (REST)."""

    def __init__(self, base_url: str = KRAKEN_REST_BASE, timeout: float = 30.0):
        self.base_url = base_url
        self.timeout = timeout
        self.client = httpx.Client(timeout=timeout, follow_redirects=True)

    @property
    def venue_name(self) -> str:
        return "kraken"

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> KrakenClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _get(self, endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Make a GET request to Kraken API."""
        url = f"{self.base_url}{endpoint}"
        response = self.client.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        if data.get("error"):
            raise httpx.HTTPError(f"Kraken API error: {data['error']}")
        return data["result"]

    def get_latest_price(self, symbol: str = KRAKEN_PAIR) -> dict[str, Any]:
        """Get latest ticker data for a symbol."""
        data = self._get("/0/public/Ticker", {"pair": symbol})
        # Kraken returns data keyed by pair ID (e.g., 'XXBTZUSD')
        pair_data = list(data.values())[0]
        return {
            "price": float(pair_data["c"][0]),  # last trade closed price
            "bid": float(pair_data["b"][0]),  # best bid
            "ask": float(pair_data["a"][0]),  # best ask
            "volume": float(pair_data["v"][1]),  # 24h volume
            "timestamp": datetime.now(UTC).isoformat(),
        }

    def get_recent_trades(self, symbol: str = KRAKEN_PAIR, limit: int = 100) -> list[Trade]:
        """Get recent trades for a symbol."""
        data = self._get("/0/public/Trades", {"pair": symbol, "count": min(limit, 1000)})
        # Kraken returns trades in the pair key
        pair_key = list(data.keys())[0]
        trades_data = data[pair_key]

        trades = []
        for item in trades_data:
            # Kraken format: [price, volume, time, side, orderType, misc]
            # time is in seconds with decimals
            ts_ms = int(float(item[2]) * 1000)
            side = "buy" if item[3] == "b" else "sell"

            trades.append(
                Trade(
                    timestamp_ms=ts_ms,
                    price=float(item[0]),
                    size=float(item[1]),
                    side=side,
                    trade_id=f"{ts_ms}_{item[0]}",  # Kraken doesn't provide trade IDs
                    venue="kraken",
                    raw_data={"kraken_format": item},
                )
            )

        return trades


class PricefeedManager:
    """Manager that handles primary/fallback logic across venues."""

    def __init__(
        self,
        primary: str = "coinbase",
        fallback: str | None = "kraken",
        timeout: float = 30.0,
    ):
        self.primary_name = primary
        self.fallback_name = fallback
        self.timeout = timeout
        self._clients: dict[str, PricefeedClient] = {}

    def _get_client(self, venue: str) -> PricefeedClient:
        """Get or create client for a venue."""
        if venue not in self._clients:
            if venue == "coinbase":
                self._clients[venue] = CoinbaseClient(timeout=self.timeout)
            elif venue == "kraken":
                self._clients[venue] = KrakenClient(timeout=self.timeout)
            else:
                raise ValueError(f"Unknown venue: {venue}")
        return self._clients[venue]

    def get_latest_price(self, symbol: str | None = None) -> dict[str, Any]:
        """Get latest price, trying primary then fallback."""
        venues_to_try = [self.primary_name]
        if self.fallback_name:
            venues_to_try.append(self.fallback_name)

        for venue in venues_to_try:
            try:
                client = self._get_client(venue)
                symbol_map = {"coinbase": COINBASE_PRODUCT, "kraken": KRAKEN_PAIR}
                sym = symbol or symbol_map.get(venue, COINBASE_PRODUCT)
                result = client.get_latest_price(sym)
                result["venue"] = venue
                return result
            except Exception as e:
                logger.warning("Failed to get price from %s: %s", venue, e)
                continue

        raise httpx.HTTPError("All pricefeed venues failed")

    def get_recent_trades(self, symbol: str | None = None, limit: int = 100) -> list[Trade]:
        """Get recent trades, trying primary then fallback."""
        venues_to_try = [self.primary_name]
        if self.fallback_name:
            venues_to_try.append(self.fallback_name)

        for venue in venues_to_try:
            try:
                client = self._get_client(venue)
                symbol_map = {"coinbase": COINBASE_PRODUCT, "kraken": KRAKEN_PAIR}
                sym = symbol or symbol_map.get(venue, COINBASE_PRODUCT)
                return client.get_recent_trades(sym, limit)
            except Exception as e:
                logger.warning("Failed to get trades from %s: %s", venue, e)
                continue

        raise httpx.HTTPError("All pricefeed venues failed")

    def close(self) -> None:
        """Close all clients."""
        for client in self._clients.values():
            client.close()
        self._clients.clear()

    def __enter__(self) -> PricefeedManager:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


class CoinbaseWebSocketCollector:
    """WebSocket collector for Coinbase real-time market data."""

    def __init__(
        self,
        product_id: str = COINBASE_PRODUCT,
        ws_base: str = COINBASE_WS_BASE,
        max_reconnect_delay: float = 60.0,
        trade_buffer_size: int = 10000,
    ):
        self.product_id = product_id
        self.ws_base = ws_base
        self.max_reconnect_delay = max_reconnect_delay
        self.trade_buffer: deque[Trade] = deque(maxlen=trade_buffer_size)
        self.current_price: float | None = None
        self.bid: float | None = None
        self.ask: float | None = None
        self._running = False
        self._reconnect_delay = 1.0
        self._ws: Any = None

    def _parse_trade(self, data: dict[str, Any]) -> Trade | None:
        """Parse trade from WebSocket message."""
        if data.get("type") != "match" and data.get("type") != "last_match":
            return None

        # Parse timestamp
        ts_str = data.get("time", "")
        if ts_str:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            ts_ms = int(ts.timestamp() * 1000)
        else:
            ts_ms = int(datetime.now(UTC).timestamp() * 1000)

        # Side: 'side' is maker side, so taker side is opposite
        maker_side = data.get("side", "buy")
        taker_side = "sell" if maker_side == "buy" else "buy"

        return Trade(
            timestamp_ms=ts_ms,
            price=float(data["price"]),
            size=float(data["size"]),
            side=taker_side,
            trade_id=str(data.get("trade_id", data.get("sequence", ts_ms))),
            venue="coinbase",
            raw_data=data,
        )

    def _parse_ticker(self, data: dict[str, Any]) -> None:
        """Parse ticker update."""
        if data.get("type") == "ticker":
            self.current_price = float(data.get("price", 0)) or None
            self.bid = float(data.get("best_bid", 0)) or None
            self.ask = float(data.get("best_ask", 0)) or None

    async def _connect_and_listen(
        self,
        on_snapshot: Callable[[Snapshot], None] | None = None,
        snapshot_interval_seconds: float = 1.0,
    ) -> None:
        """Connect to WebSocket and listen for messages."""
        subscribe_msg = {
            "type": "subscribe",
            "product_ids": [self.product_id],
            "channels": ["matches", "ticker"],
        }

        logger.info("Connecting to Coinbase WebSocket: %s", self.ws_base)

        try:
            async with websockets.connect(self.ws_base) as ws:
                self._ws = ws
                await ws.send(json.dumps(subscribe_msg))
                self._reconnect_delay = 1.0  # Reset on successful connection
                logger.info("Connected to Coinbase WebSocket, subscribed to %s", self.product_id)

                last_snapshot = asyncio.get_event_loop().time()

                async for message in ws:
                    if not self._running:
                        break

                    try:
                        data = json.loads(message)

                        # Parse trade
                        trade = self._parse_trade(data)
                        if trade:
                            self.trade_buffer.append(trade)

                        # Parse ticker
                        self._parse_ticker(data)

                    except (json.JSONDecodeError, KeyError, ValueError) as e:
                        logger.warning("Failed to parse message: %s", e)
                        continue

                    # Generate periodic snapshots
                    now = asyncio.get_event_loop().time()
                    if on_snapshot and now - last_snapshot >= snapshot_interval_seconds:
                        snapshot = self._create_snapshot()
                        on_snapshot(snapshot)
                        last_snapshot = now

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
            symbol=self.product_id,
            venue="coinbase",
            trades=list(self.trade_buffer),
            current_price=self.current_price,
            bid=self.bid,
            ask=self.ask,
        )

    async def run(
        self,
        on_snapshot: Callable[[Snapshot], None] | None = None,
        snapshot_interval_seconds: float = 1.0,
    ) -> None:
        """Run the WebSocket collector with automatic reconnection."""
        self._running = True

        while self._running:
            try:
                await self._connect_and_listen(
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
                self._reconnect_delay = min(
                    self.max_reconnect_delay, self._reconnect_delay * 1.5
                )

    def stop(self) -> None:
        """Stop the collector."""
        self._running = False
        if self._ws:
            asyncio.create_task(self._ws.close())


class KrakenWebSocketCollector:
    """WebSocket collector for Kraken real-time market data."""

    def __init__(
        self,
        pair: str = KRAKEN_WS_PAIR,
        ws_base: str = KRAKEN_WS_BASE,
        max_reconnect_delay: float = 60.0,
        trade_buffer_size: int = 10000,
    ):
        self.pair = pair
        self.ws_base = ws_base
        self.max_reconnect_delay = max_reconnect_delay
        self.trade_buffer: deque[Trade] = deque(maxlen=trade_buffer_size)
        self.current_price: float | None = None
        self.bid: float | None = None
        self.ask: float | None = None
        self._running = False
        self._reconnect_delay = 1.0
        self._ws: Any = None

    def _parse_trade(self, data: list[Any]) -> Trade | None:
        """Parse trade from WebSocket message."""
        # Kraken trade format: [channelID, [[price, volume, time, side, orderType, misc], ...], channelName, pair]
        if not isinstance(data, list) or len(data) < 4:
            return None

        channel_name = data[2] if len(data) > 2 else ""
        if "trade" not in channel_name:
            return None

        trades_data = data[1] if len(data) > 1 else []
        if not trades_data or not isinstance(trades_data, list):
            return None

        # Take the first trade from the batch
        trade_item = trades_data[0]
        if not isinstance(trade_item, list) or len(trade_item) < 4:
            return None

        ts_sec = float(trade_item[2])
        ts_ms = int(ts_sec * 1000)
        side = "buy" if trade_item[3] == "b" else "sell"

        return Trade(
            timestamp_ms=ts_ms,
            price=float(trade_item[0]),
            size=float(trade_item[1]),
            side=side,
            trade_id=f"{ts_ms}_{trade_item[0]}",
            venue="kraken",
            raw_data={"kraken_ws": data},
        )

    def _parse_ticker(self, data: list[Any]) -> None:
        """Parse ticker update."""
        # Kraken ticker format: [channelID, {bid: [...], ask: [...], ...}, channelName, pair]
        if not isinstance(data, list) or len(data) < 4:
            return

        channel_name = data[2] if len(data) > 2 else ""
        if "ticker" not in channel_name:
            return

        ticker_data = data[1] if len(data) > 1 else {}
        if not isinstance(ticker_data, dict):
            return

        # Extract best bid/ask
        bid_data = ticker_data.get("b", [])
        ask_data = ticker_data.get("a", [])

        if bid_data and isinstance(bid_data, list):
            self.bid = float(bid_data[0])
        if ask_data and isinstance(ask_data, list):
            self.ask = float(ask_data[0])

        # Last trade price
        last_trade = ticker_data.get("c", [])
        if last_trade and isinstance(last_trade, list):
            self.current_price = float(last_trade[0])

    async def _connect_and_listen(
        self,
        on_snapshot: Callable[[Snapshot], None] | None = None,
        snapshot_interval_seconds: float = 1.0,
    ) -> None:
        """Connect to WebSocket and listen for messages."""
        subscribe_msg = {
            "event": "subscribe",
            "pair": [self.pair],
            "subscription": {"name": "trade"},
        }
        ticker_msg = {
            "event": "subscribe",
            "pair": [self.pair],
            "subscription": {"name": "ticker"},
        }

        logger.info("Connecting to Kraken WebSocket: %s", self.ws_base)

        try:
            async with websockets.connect(self.ws_base) as ws:
                self._ws = ws
                await ws.send(json.dumps(subscribe_msg))
                await ws.send(json.dumps(ticker_msg))
                self._reconnect_delay = 1.0  # Reset on successful connection
                logger.info("Connected to Kraken WebSocket, subscribed to %s", self.pair)

                last_snapshot = asyncio.get_event_loop().time()

                async for message in ws:
                    if not self._running:
                        break

                    try:
                        data = json.loads(message)

                        # Handle array format (data messages)
                        if isinstance(data, list):
                            trade = self._parse_trade(data)
                            if trade:
                                self.trade_buffer.append(trade)
                            self._parse_ticker(data)
                        # Handle dict format (event messages)
                        elif isinstance(data, dict):
                            event = data.get("event")
                            if event == "heartbeat":
                                pass  # Keepalive
                            elif event in ("subscribed", "unsubscribed", "error"):
                                logger.debug("Kraken event: %s", data)

                    except (json.JSONDecodeError, KeyError, ValueError, IndexError) as e:
                        logger.warning("Failed to parse message: %s", e)
                        continue

                    # Generate periodic snapshots
                    now = asyncio.get_event_loop().time()
                    if on_snapshot and now - last_snapshot >= snapshot_interval_seconds:
                        snapshot = self._create_snapshot()
                        on_snapshot(snapshot)
                        last_snapshot = now

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
            symbol=self.pair,
            venue="kraken",
            trades=list(self.trade_buffer),
            current_price=self.current_price,
            bid=self.bid,
            ask=self.ask,
        )

    async def run(
        self,
        on_snapshot: Callable[[Snapshot], None] | None = None,
        snapshot_interval_seconds: float = 1.0,
    ) -> None:
        """Run the WebSocket collector with automatic reconnection."""
        self._running = True

        while self._running:
            try:
                await self._connect_and_listen(
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
                self._reconnect_delay = min(
                    self.max_reconnect_delay, self._reconnect_delay * 1.5
                )

    def stop(self) -> None:
        """Stop the collector."""
        self._running = False
        if self._ws:
            asyncio.create_task(self._ws.close())


async def collect_loop_ws(
    out_dir: Path,
    venue: str = "coinbase",
    snapshot_interval_seconds: float = 1.0,
    max_reconnect_delay: float = 60.0,
    retention_hours: float | None = None,
) -> None:
    """Continuously collect pricefeed data via WebSocket.

    Args:
        out_dir: Directory to write output files
        venue: Venue to use (coinbase or kraken)
        snapshot_interval_seconds: How often to write snapshots
        max_reconnect_delay: Maximum reconnection delay in seconds
        retention_hours: If set, delete files older than this many hours
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    if venue == "coinbase":
        collector = CoinbaseWebSocketCollector(max_reconnect_delay=max_reconnect_delay)
    elif venue == "kraken":
        collector = KrakenWebSocketCollector(max_reconnect_delay=max_reconnect_delay)
    else:
        raise ValueError(f"Unknown venue: {venue}")

    def on_snapshot(snapshot: Snapshot) -> None:
        # Write JSON snapshot
        ts_str = datetime.fromtimestamp(snapshot.timestamp_ms / 1000, tz=UTC).strftime(
            "%Y%m%dT%H%M%SZ"
        )
        out_path = out_dir / f"pricefeed_{venue}_{ts_str}.json"
        out_path.write_text(json.dumps(snapshot.to_dict(), indent=2, sort_keys=True))

        # Write latest pointer
        latest = out_dir / f"latest_{venue}.json"
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
            _prune_old_files(out_dir, venue, retention_hours)

    logger.info("Starting pricefeed collector for %s", venue)

    await collector.run(
        on_snapshot=on_snapshot,
        snapshot_interval_seconds=snapshot_interval_seconds,
    )


def _prune_old_files(out_dir: Path, venue: str, retention_hours: float) -> int:
    """Delete files older than retention_hours."""
    cutoff = datetime.now(UTC) - timedelta(hours=retention_hours)
    deleted = 0
    prefix = f"pricefeed_{venue}_"

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
    venue: str = "coinbase",
    snapshot_interval_seconds: float = 1.0,
    max_reconnect_delay: float = 60.0,
    retention_hours: float | None = None,
) -> None:
    """Synchronous entry point to run the WebSocket collector loop."""
    asyncio.run(
        collect_loop_ws(
            out_dir=out_dir,
            venue=venue,
            snapshot_interval_seconds=snapshot_interval_seconds,
            max_reconnect_delay=max_reconnect_delay,
            retention_hours=retention_hours,
        )
    )


def collect_snapshot_rest(
    out_dir: Path,
    venue: str = "coinbase",
) -> Path:
    """Collect a single snapshot using REST API.

    Args:
        out_dir: Directory to write output files
        venue: Venue to use (coinbase or kraken)

    Returns:
        Path to the written JSON file
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now(UTC)
    ts_str = now.strftime("%Y%m%dT%H%M%SZ")

    if venue == "coinbase":
        with CoinbaseClient() as client:
            price_data = client.get_latest_price(COINBASE_PRODUCT)
            trades = client.get_recent_trades(COINBASE_PRODUCT, limit=100)
            symbol = COINBASE_PRODUCT
    elif venue == "kraken":
        with KrakenClient() as client:
            price_data = client.get_latest_price(KRAKEN_PAIR)
            trades = client.get_recent_trades(KRAKEN_PAIR, limit=100)
            symbol = KRAKEN_PAIR
    else:
        raise ValueError(f"Unknown venue: {venue}")

    snapshot = Snapshot(
        timestamp=now.isoformat(),
        timestamp_ms=int(now.timestamp() * 1000),
        symbol=symbol,
        venue=venue,
        trades=trades,
        current_price=price_data.get("price"),
        bid=price_data.get("bid"),
        ask=price_data.get("ask"),
    )

    out_path = out_dir / f"pricefeed_{venue}_{ts_str}.json"
    out_path.write_text(json.dumps(snapshot.to_dict(), indent=2, sort_keys=True))

    return out_path
