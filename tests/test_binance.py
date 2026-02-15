"""Tests for Binance collector and feature builder."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from polymarket.binance_collector import (
    BinanceRestClient,
    BinanceWebSocketCollector,
    collect_snapshot_rest,
)
from polymarket.binance_features import (
    FeatureBuilder,
    align_to_polymarket_snapshots,
)
from polymarket.marketdata import AggTrade, Kline, Snapshot


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
            provider="binance",
            trades=[trade],
            klines={"1m": None},
        )
        assert snapshot.symbol == "BTCUSDT"
        assert snapshot.provider == "binance"
        assert len(snapshot.trades) == 1

    def test_snapshot_provider_fallback(self):
        """Test that snapshot includes provider info for fallback tracking."""
        snapshot = Snapshot(
            timestamp="2023-11-14T12:00:00Z",
            timestamp_ms=1_700_000_000_000,
            symbol="BTCUSDT",
            provider="coinbase",  # Could be coinbase/kraken if binance fails
            trades=[],
            klines={},
        )
        d = snapshot.to_dict()
        assert d["provider"] == "coinbase"


class TestBinanceRestClient:
    def test_client_context_manager(self):
        with BinanceRestClient() as client:
            # Default should be binance.com or auto-rotate list
            assert "binance.com" in client.base_url

    def test_client_close(self):
        client = BinanceRestClient()
        client.close()
        assert client.client.is_closed

    def test_endpoint_rotation_on_451(self, monkeypatch):
        """Test HTTP 451 triggers endpoint rotation across multiple bases."""
        # First endpoint blocked; second succeeds.
        client = BinanceRestClient(
            base_urls=["https://api.binance.com", "https://api1.binance.com"]
        )

        calls: list[str] = []

        def fake_get(url, params=None):  # noqa: ANN001
            calls.append(url)
            req = httpx.Request("GET", url, params=params)
            if len(calls) == 1:
                return httpx.Response(
                    451, request=req, json={"msg": "Unavailable For Legal Reasons"}
                )
            return httpx.Response(
                200, request=req, json=[{"T": 1, "p": "1", "q": "1", "m": False, "a": 1}]
            )

        monkeypatch.setattr(client.client, "get", fake_get)

        trades = client.get_agg_trades(symbol="BTCUSDT", limit=1)
        assert len(trades) == 1
        assert calls[0].startswith("https://api.binance.com")
        assert calls[1].startswith("https://api1.binance.com")
        client.close()

    def test_endpoint_rotation_all_fail(self, monkeypatch):
        """Test that all endpoints failing raises an exception."""
        client = BinanceRestClient(
            base_urls=["https://api.binance.com", "https://api1.binance.com"]
        )

        def fake_get(url, params=None):  # noqa: ANN001
            req = httpx.Request("GET", url, params=params)
            return httpx.Response(
                451, request=req, json={"msg": "Unavailable For Legal Reasons"}
            )

        monkeypatch.setattr(client.client, "get", fake_get)

        with pytest.raises(httpx.HTTPStatusError):
            client.get_agg_trades(symbol="BTCUSDT", limit=1)

        client.close()

    def test_env_override_base_urls(self):
        """Test that custom base_urls parameter is respected."""
        custom_urls = ["https://api3.binance.com", "https://api4.binance.com"]
        client = BinanceRestClient(base_urls=custom_urls)
        assert client.base_urls == custom_urls
        assert client.base_url == custom_urls[0]
        client.close()


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


class TestFallbackProvider:
    """Tests for fallback provider functionality (auto-provider with coinbase/kraken)."""

    def test_auto_provider_tries_multiple_providers(self, monkeypatch):
        """Test that AutoProvider tries Binance, then Coinbase, then Kraken."""
        from polymarket.marketdata.auto import AutoProvider
        from polymarket.marketdata import ProviderUnavailableError

        provider = AutoProvider()

        # Track which providers were called
        calls = []

        # Mock Binance to fail with 451
        def mock_binance_health(*args, **kwargs):  # noqa: ANN001, ANN002
            calls.append("binance")
            raise ProviderUnavailableError("HTTP 451")

        # Mock Coinbase to succeed
        def mock_coinbase_health(*args, **kwargs):  # noqa: ANN001, ANN002
            calls.append("coinbase")
            return True

        # Apply mocks
        monkeypatch.setattr(
            provider._get_provider("binance"), "health_check", mock_binance_health
        )
        monkeypatch.setattr(
            provider._get_provider("coinbase"), "health_check", mock_coinbase_health
        )

        # Should try binance first, then succeed via coinbase
        result = provider.health_check()
        assert result is True
        assert "binance" in calls
        assert "coinbase" in calls

        provider.close()

    def test_auto_provider_reports_provider_in_snapshot(self, monkeypatch):
        """Test that snapshot includes which provider was used."""
        from polymarket.marketdata.auto import AutoProvider

        provider = AutoProvider(preferred_order=["coinbase"])

        # Mock get_agg_trades to return test data
        def mock_get_agg_trades(*args, **kwargs):  # noqa: ANN001, ANN002
            return [
                AggTrade(
                    timestamp_ms=1_700_000_000_000,
                    price=42000.0,
                    quantity=1.0,
                    is_buyer_maker=False,
                    trade_id=1,
                )
            ]

        # Mock get_klines to return empty list
        def mock_get_klines(*args, **kwargs):  # noqa: ANN001, ANN002
            return []

        coinbase_provider = provider._get_provider("coinbase")
        monkeypatch.setattr(coinbase_provider, "get_agg_trades", mock_get_agg_trades)
        monkeypatch.setattr(coinbase_provider, "get_klines", mock_get_klines)

        snapshot = provider.get_snapshot(symbol="BTCUSDT", kline_intervals=[])
        assert snapshot.provider == "coinbase"
        assert len(snapshot.trades) == 1

        provider.close()

    def test_collect_snapshot_rest_includes_provider(self, tmp_path: Path, monkeypatch):
        """Test that collect_snapshot_rest includes provider in output."""
        from polymarket.marketdata.auto import AutoProvider

        out_dir = tmp_path / "data"

        # Mock get_snapshot to return a test snapshot with provider info
        def mock_get_snapshot(self, *args, **kwargs):  # noqa: ANN001, ANN002
            return Snapshot(
                timestamp="2023-11-14T22:13:20Z",
                timestamp_ms=1_700_000_000_000,
                symbol="BTCUSDT",
                provider="coinbase",  # Simulate fallback to coinbase
                trades=[
                    AggTrade(
                        timestamp_ms=1_700_000_000_000,
                        price=42000.0,
                        quantity=1.0,
                        is_buyer_maker=False,
                        trade_id=1,
                    )
                ],
                klines={},
            )

        monkeypatch.setattr(AutoProvider, "get_snapshot", mock_get_snapshot)

        out_path = collect_snapshot_rest(
            out_dir=out_dir,
            symbol="BTCUSDT",
            kline_intervals=["1m"],
        )

        assert out_path.exists()
        snapshot_data = json.loads(out_path.read_text())
        assert "provider" in snapshot_data
        assert snapshot_data["provider"] == "coinbase"

        # Output filename should include provider
        assert "coinbase" in out_path.name

        # Latest pointer should also include provider
        latest_path = out_dir / "latest_btcusdt.json"
        assert latest_path.exists()
        latest_data = json.loads(latest_path.read_text())
        assert latest_data.get("provider") == "coinbase"
