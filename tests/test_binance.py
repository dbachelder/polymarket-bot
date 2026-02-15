"""Tests for Binance collector and feature builder."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from polymarket.binance_collector import (
    AggTrade,
    BinanceRestClient,
    BinanceWebSocketCollector,
    Kline,
    Snapshot,
    collect_snapshot_rest,
)
from polymarket.binance_features import (
    FeatureBuilder,
    Returns,
    VolumeMetrics,
    align_to_polymarket_snapshots,
)


class TestAggTrade:
    def test_agg_trade_creation(self):
        trade = AggTrade(
            timestamp_ms=1_700_000_000_000,
            price=42_000.50,
            quantity=0.5,
            is_buyer_maker=True,
            trade_id=12345,
        )
        assert trade.timestamp_ms == 1_700_000_000_000
        assert trade.price == 42_000.50
        assert trade.quantity == 0.5
        assert trade.is_buyer_maker is True
        assert trade.trade_id == 12345
        assert "2023" in trade.timestamp  # UTC timestamp string

    def test_signed_volume_buy_aggressor(self):
        # is_buyer_maker=False means buyer is aggressor (positive signed volume)
        trade = AggTrade(
            timestamp_ms=1_700_000_000_000,
            price=42_000.0,
            quantity=1.0,
            is_buyer_maker=False,
            trade_id=1,
        )
        assert trade.signed_volume == 1.0

    def test_signed_volume_sell_aggressor(self):
        # is_buyer_maker=True means seller is aggressor (negative signed volume)
        trade = AggTrade(
            timestamp_ms=1_700_000_000_000,
            price=42_000.0,
            quantity=1.0,
            is_buyer_maker=True,
            trade_id=1,
        )
        assert trade.signed_volume == -1.0

    def test_to_dict(self):
        trade = AggTrade(
            timestamp_ms=1_700_000_000_000,
            price=42_000.0,
            quantity=1.0,
            is_buyer_maker=False,
            trade_id=1,
        )
        d = trade.to_dict()
        assert d["price"] == 42_000.0
        assert d["quantity"] == 1.0
        assert d["signed_volume"] == 1.0


class TestKline:
    def test_kline_creation(self):
        kline = Kline(
            open_time_ms=1_700_000_000_000,
            close_time_ms=1_700_000_060_000,
            open_price=41_000.0,
            high_price=43_000.0,
            low_price=40_000.0,
            close_price=42_000.0,
            volume=100.0,
            quote_volume=4_200_000.0,
            trades_count=50,
            taker_buy_volume=60.0,
            taker_buy_quote_volume=2_520_000.0,
        )
        assert kline.open_price == 41_000.0
        assert kline.high_price == 43_000.0
        assert kline.low_price == 40_000.0
        assert kline.close_price == 42_000.0

    def test_to_dict(self):
        kline = Kline(
            open_time_ms=1_700_000_000_000,
            close_time_ms=1_700_000_060_000,
            open_price=41_000.0,
            high_price=43_000.0,
            low_price=40_000.0,
            close_price=42_000.0,
            volume=100.0,
            quote_volume=4_200_000.0,
            trades_count=50,
            taker_buy_volume=60.0,
            taker_buy_quote_volume=2_520_000.0,
        )
        d = kline.to_dict()
        assert d["open_price"] == 41_000.0
        assert d["close_price"] == 42_000.0


class TestSnapshot:
    def test_snapshot_creation(self):
        trade = AggTrade(
            timestamp_ms=1_700_000_000_000,
            price=42_000.0,
            quantity=1.0,
            is_buyer_maker=False,
            trade_id=1,
        )
        snapshot = Snapshot(
            timestamp="2023-11-14T12:00:00Z",
            timestamp_ms=1_700_000_000_000,
            symbol="BTCUSDT",
            trades=[trade],
            klines={"1m": None},
        )
        assert snapshot.symbol == "BTCUSDT"
        assert len(snapshot.trades) == 1


class TestBinanceRestClient:
    def test_client_context_manager(self):
        with BinanceRestClient() as client:
            assert client.base_url == "https://api.binance.com"

    def test_client_close(self):
        client = BinanceRestClient()
        client.close()
        assert client.client.is_closed


class TestBinanceWebSocketCollector:
    def test_collector_init(self):
        collector = BinanceWebSocketCollector(symbol="BTCUSDT")
        assert collector.symbol == "btcusdt"
        assert collector.max_reconnect_delay == 60.0

    def test_get_stream_url_single(self):
        collector = BinanceWebSocketCollector()
        url = collector._get_stream_url(["btcusdt@aggTrade"])
        assert url == "wss://stream.binance.com:9443/ws/btcusdt@aggTrade"

    def test_get_stream_url_combined(self):
        collector = BinanceWebSocketCollector()
        url = collector._get_stream_url(["btcusdt@aggTrade", "btcusdt@kline_1m"])
        assert "stream?streams=" in url
        assert "btcusdt@aggTrade" in url
        assert "btcusdt@kline_1m" in url


class TestFeatureBuilder:
    def test_builder_init(self):
        builder = FeatureBuilder(horizons=[5, 30])
        assert builder.horizons == [5, 30]

    def test_compute_returns_empty(self):
        builder = FeatureBuilder()
        returns = builder.compute_returns([])
        assert returns == []

    def test_compute_returns_basic(self):
        builder = FeatureBuilder(horizons=[5])
        trades = [
            AggTrade(
                timestamp_ms=1_700_000_000_000,
                price=40_000.0,
                quantity=1.0,
                is_buyer_maker=False,
                trade_id=1,
            ),
            AggTrade(
                timestamp_ms=1_700_000_005_000,
                price=41_000.0,
                quantity=1.0,
                is_buyer_maker=False,
                trade_id=2,
            ),
        ]
        returns = builder.compute_returns(trades, reference_time_ms=1_700_000_005_000)
        assert len(returns) == 1
        assert returns[0].horizon_seconds == 5
        assert returns[0].simple_return == pytest.approx(0.025, rel=1e-2)

    def test_compute_volume_metrics(self):
        builder = FeatureBuilder(horizons=[5])
        trades = [
            AggTrade(
                timestamp_ms=1_700_000_000_000,
                price=40_000.0,
                quantity=1.0,
                is_buyer_maker=False,
                trade_id=1,
            ),
            AggTrade(
                timestamp_ms=1_700_000_005_000,
                price=41_000.0,
                quantity=2.0,
                is_buyer_maker=True,
                trade_id=2,
            ),
        ]
        metrics = builder.compute_volume_metrics(trades, reference_time_ms=1_700_000_005_000)
        assert len(metrics) == 1
        assert metrics[0].total_volume == 3.0
        assert metrics[0].buy_volume == 1.0
        assert metrics[0].sell_volume == 2.0

    def test_build_features(self):
        builder = FeatureBuilder(horizons=[5])
        trades = [
            AggTrade(
                timestamp_ms=1_700_000_000_000,
                price=40_000.0,
                quantity=1.0,
                is_buyer_maker=False,
                trade_id=1,
            ),
        ]
        features = builder.build_features(trades)
        assert features.symbol == "BTCUSDT"
        assert features.reference_price == 40_000.0


class TestAlignToPolymarketSnapshots:
    def test_align_empty_directories(self, tmp_path: Path):
        binance_dir = tmp_path / "binance"
        pm_dir = tmp_path / "polymarket"
        binance_dir.mkdir()
        pm_dir.mkdir()

        result = align_to_polymarket_snapshots(binance_dir, pm_dir)
        assert result == []

    def test_align_with_data(self, tmp_path: Path):
        binance_dir = tmp_path / "binance"
        pm_dir = tmp_path / "polymarket"
        binance_dir.mkdir()
        pm_dir.mkdir()

        # Use consistent timestamps
        ts_ms = 1_700_000_000_000
        ts_iso = "2023-11-14T22:13:20Z"  # Corresponds to 1_700_000_000_000 ms

        # Create a Binance snapshot
        binance_snap = {
            "timestamp": ts_iso,
            "timestamp_ms": ts_ms,
            "symbol": "BTCUSDT",
            "trades": [
                {
                    "timestamp": ts_iso,
                    "timestamp_ms": ts_ms,
                    "price": 42_000.0,
                    "quantity": 1.0,
                    "is_buyer_maker": False,
                    "trade_id": 1,
                    "signed_volume": 1.0,
                }
            ],
            "klines": {},
        }
        (binance_dir / "binance_btcusdt_20231114T221320Z.json").write_text(
            json.dumps(binance_snap)
        )

        # Create a Polymarket snapshot at same time
        pm_snap = {
            "generated_at": ts_iso,
            "markets": [],
        }
        (pm_dir / "snapshot_5m_20231114T221320Z.json").write_text(json.dumps(pm_snap))

        result = align_to_polymarket_snapshots(binance_dir, pm_dir, tolerance_seconds=1.0)
        assert len(result) == 1
        assert result[0]["time_diff_seconds"] == 0.0
