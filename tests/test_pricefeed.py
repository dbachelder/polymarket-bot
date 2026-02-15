"""Tests for pricefeed module."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from polymarket.pricefeed import (
    CoinbaseClient,
    CoinbaseWebSocketCollector,
    KrakenClient,
    KrakenWebSocketCollector,
    PricefeedManager,
    Snapshot,
    Trade,
    collect_snapshot_rest,
)


class TestTrade:
    """Test Trade dataclass."""

    def test_trade_creation(self):
        """Test creating a Trade."""
        trade = Trade(
            timestamp_ms=1700000000000,
            price=50000.0,
            size=1.5,
            side="buy",
            trade_id="12345",
            venue="coinbase",
            raw_data={},
        )

        assert trade.timestamp_ms == 1700000000000
        assert trade.price == 50000.0
        assert trade.size == 1.5
        assert trade.side == "buy"
        assert trade.venue == "coinbase"
        assert trade.signed_volume == 1.5

    def test_trade_signed_volume_sell(self):
        """Test signed volume for sell trades."""
        trade = Trade(
            timestamp_ms=1700000000000,
            price=50000.0,
            size=1.5,
            side="sell",
            trade_id="12345",
            venue="coinbase",
            raw_data={},
        )

        assert trade.signed_volume == -1.5

    def test_trade_to_dict(self):
        """Test converting Trade to dict."""
        trade = Trade(
            timestamp_ms=1700000000000,
            price=50000.0,
            size=1.5,
            side="buy",
            trade_id="12345",
            venue="coinbase",
            raw_data={"extra": "data"},
        )

        d = trade.to_dict()
        assert d["price"] == 50000.0
        assert d["size"] == 1.5
        assert d["side"] == "buy"
        assert d["signed_volume"] == 1.5
        assert "timestamp" in d


class TestCoinbaseClient:
    """Test Coinbase REST client."""

    @patch("polymarket.pricefeed.httpx.Client")
    def test_get_latest_price(self, mock_client_class):
        """Test fetching latest price."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "price": "50000.00",
            "bid": "49999.00",
            "ask": "50001.00",
            "volume": "1000.00",
            "time": "2024-01-01T00:00:00Z",
        }
        mock_client.get.return_value = mock_response

        with CoinbaseClient() as client:
            result = client.get_latest_price("BTC-USD")

        assert result["price"] == 50000.0
        assert result["bid"] == 49999.0
        assert result["ask"] == 50001.0
        # Venue is added by manager, not client
        assert "venue" not in result

    @patch("polymarket.pricefeed.httpx.Client")
    def test_get_recent_trades(self, mock_client_class):
        """Test fetching recent trades."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.json.return_value = [
            {
                "trade_id": 12345,
                "price": "50000.00",
                "size": "1.5",
                "side": "buy",
                "time": "2024-01-01T00:00:00Z",
            },
            {
                "trade_id": 12346,
                "price": "50001.00",
                "size": "0.5",
                "side": "sell",
                "time": "2024-01-01T00:00:01Z",
            },
        ]
        mock_client.get.return_value = mock_response

        with CoinbaseClient() as client:
            trades = client.get_recent_trades("BTC-USD", limit=100)

        assert len(trades) == 2
        assert trades[0].price == 50000.0
        assert trades[0].size == 1.5
        assert trades[0].side == "buy"
        assert trades[1].side == "sell"


class TestKrakenClient:
    """Test Kraken REST client."""

    @patch("polymarket.pricefeed.httpx.Client")
    def test_get_latest_price(self, mock_client_class):
        """Test fetching latest price from Kraken."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "error": [],
            "result": {
                "XXBTZUSD": {
                    "a": ["50001.0", "1", "1.000"],
                    "b": ["49999.0", "1", "1.000"],
                    "c": ["50000.0", "1.5"],
                    "v": ["100.0", "1000.0"],
                }
            },
        }
        mock_client.get.return_value = mock_response

        with KrakenClient() as client:
            result = client.get_latest_price("XBT/USD")

        assert result["price"] == 50000.0
        assert result["bid"] == 49999.0
        assert result["ask"] == 50001.0

    @patch("polymarket.pricefeed.httpx.Client")
    def test_get_recent_trades(self, mock_client_class):
        """Test fetching recent trades from Kraken."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "error": [],
            "result": {
                "XXBTZUSD": [
                    ["50000.0", "1.5", "1700000000.0", "b", "l", ""],
                    ["50001.0", "0.5", "1700000001.0", "s", "l", ""],
                ]
            },
        }
        mock_client.get.return_value = mock_response

        with KrakenClient() as client:
            trades = client.get_recent_trades("XBT/USD", limit=100)

        assert len(trades) == 2
        assert trades[0].price == 50000.0
        assert trades[0].size == 1.5
        assert trades[0].side == "buy"  # 'b' = buy
        assert trades[1].side == "sell"  # 's' = sell


class TestPricefeedManager:
    """Test PricefeedManager with fallback logic."""

    @patch("polymarket.pricefeed.CoinbaseClient")
    @patch("polymarket.pricefeed.KrakenClient")
    def test_primary_success(self, mock_kraken, mock_coinbase):
        """Test using primary venue when available."""
        mock_cb = MagicMock()
        mock_cb.get_latest_price.return_value = {
            "price": 50000.0,
            "bid": 49999.0,
            "ask": 50001.0,
        }
        mock_coinbase.return_value = mock_cb

        manager = PricefeedManager(primary="coinbase", fallback="kraken")
        result = manager.get_latest_price()

        assert result["price"] == 50000.0
        assert result["venue"] == "coinbase"
        mock_cb.get_latest_price.assert_called_once()

    @patch("polymarket.pricefeed.CoinbaseClient")
    @patch("polymarket.pricefeed.KrakenClient")
    def test_fallback_on_failure(self, mock_kraken, mock_coinbase):
        """Test fallback to secondary venue when primary fails."""
        mock_cb = MagicMock()
        mock_cb.get_latest_price.side_effect = Exception("Coinbase down")
        mock_coinbase.return_value = mock_cb

        mock_kr = MagicMock()
        mock_kr.get_latest_price.return_value = {
            "price": 50000.0,
            "bid": 49999.0,
            "ask": 50001.0,
        }
        mock_kraken.return_value = mock_kr

        manager = PricefeedManager(primary="coinbase", fallback="kraken")
        result = manager.get_latest_price()

        assert result["price"] == 50000.0
        assert result["venue"] == "kraken"


class TestCoinbaseWebSocketCollector:
    """Test Coinbase WebSocket collector."""

    def test_parse_trade(self):
        """Test parsing a trade message."""
        collector = CoinbaseWebSocketCollector()

        data = {
            "type": "match",
            "trade_id": 12345,
            "sequence": 67890,
            "maker_order_id": "abc",
            "taker_order_id": "def",
            "time": "2024-01-01T00:00:00Z",
            "product_id": "BTC-USD",
            "size": "1.5",
            "price": "50000.00",
            "side": "sell",  # Maker side
        }

        trade = collector._parse_trade(data)

        assert trade is not None
        assert trade.price == 50000.0
        assert trade.size == 1.5
        assert trade.side == "buy"  # Taker side is opposite of maker
        assert trade.venue == "coinbase"

    def test_parse_ticker(self):
        """Test parsing a ticker message."""
        collector = CoinbaseWebSocketCollector()

        data = {
            "type": "ticker",
            "sequence": 12345,
            "product_id": "BTC-USD",
            "price": "50000.00",
            "best_bid": "49999.00",
            "best_ask": "50001.00",
        }

        collector._parse_ticker(data)

        assert collector.current_price == 50000.0
        assert collector.bid == 49999.0
        assert collector.ask == 50001.0


class TestKrakenWebSocketCollector:
    """Test Kraken WebSocket collector."""

    def test_parse_trade(self):
        """Test parsing a trade message from Kraken."""
        collector = KrakenWebSocketCollector()

        # Kraken format: [channelID, [[price, volume, time, side, orderType, misc], ...], channelName, pair]
        data = [
            42,
            [
                ["50000.00000", "1.50000000", "1700000000.123456", "b", "l", ""],
                ["50001.00000", "0.50000000", "1700000001.123456", "s", "l", ""],
            ],
            "trade",
            "XBT/USD",
        ]

        trade = collector._parse_trade(data)

        assert trade is not None
        assert trade.price == 50000.0
        assert trade.size == 1.5
        assert trade.side == "buy"  # 'b' = buy
        assert trade.venue == "kraken"

    def test_parse_ticker(self):
        """Test parsing a ticker message from Kraken."""
        collector = KrakenWebSocketCollector()

        # Kraken ticker format: [channelID, data, channelName, pair]
        data = [
            42,
            {
                "a": ["50001.00000", "1", "1.000"],
                "b": ["49999.00000", "1", "1.000"],
                "c": ["50000.00000", "1.50000000"],
            },
            "ticker",
            "XBT/USD",
        ]

        collector._parse_ticker(data)

        assert collector.current_price == 50000.0
        assert collector.bid == 49999.0
        assert collector.ask == 50001.0


class TestSnapshot:
    """Test Snapshot dataclass."""

    def test_snapshot_creation(self):
        """Test creating a Snapshot."""
        trade = Trade(
            timestamp_ms=1700000000000,
            price=50000.0,
            size=1.5,
            side="buy",
            trade_id="12345",
            venue="coinbase",
            raw_data={},
        )

        snapshot = Snapshot(
            timestamp="2024-01-01T00:00:00+00:00",
            timestamp_ms=1700000000000,
            symbol="BTC-USD",
            venue="coinbase",
            trades=[trade],
            current_price=50000.0,
            bid=49999.0,
            ask=50001.0,
        )

        d = snapshot.to_dict()
        assert d["symbol"] == "BTC-USD"
        assert d["venue"] == "coinbase"
        assert d["current_price"] == 50000.0
        assert len(d["trades"]) == 1


class TestCollectSnapshotRest:
    """Test REST snapshot collection."""

    @patch("polymarket.pricefeed.CoinbaseClient")
    def test_collect_snapshot_coinbase(self, mock_client_class, tmp_path):
        """Test collecting a snapshot from Coinbase."""
        mock_client = MagicMock()
        mock_client_class.return_value.__enter__.return_value = mock_client

        mock_client.get_latest_price.return_value = {
            "price": 50000.0,
            "bid": 49999.0,
            "ask": 50001.0,
        }
        mock_client.get_recent_trades.return_value = [
            Trade(
                timestamp_ms=1700000000000,
                price=50000.0,
                size=1.5,
                side="buy",
                trade_id="12345",
                venue="coinbase",
                raw_data={},
            )
        ]

        out_path = collect_snapshot_rest(tmp_path, venue="coinbase")

        assert out_path.exists()
        data = json.loads(out_path.read_text())
        assert data["venue"] == "coinbase"
        assert data["symbol"] == "BTC-USD"
        assert data["current_price"] == 50000.0

    @patch("polymarket.pricefeed.KrakenClient")
    def test_collect_snapshot_kraken(self, mock_client_class, tmp_path):
        """Test collecting a snapshot from Kraken."""
        mock_client = MagicMock()
        mock_client_class.return_value.__enter__.return_value = mock_client

        mock_client.get_latest_price.return_value = {
            "price": 50000.0,
            "bid": 49999.0,
            "ask": 50001.0,
        }
        mock_client.get_recent_trades.return_value = [
            Trade(
                timestamp_ms=1700000000000,
                price=50000.0,
                size=1.5,
                side="buy",
                trade_id="12345",
                venue="kraken",
                raw_data={},
            )
        ]

        out_path = collect_snapshot_rest(tmp_path, venue="kraken")

        assert out_path.exists()
        data = json.loads(out_path.read_text())
        assert data["venue"] == "kraken"
