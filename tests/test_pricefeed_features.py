"""Tests for pricefeed_features module."""

from __future__ import annotations

import json

import pytest

from polymarket.pricefeed import Trade
from polymarket.pricefeed_features import (
    FeatureBuilder,
    FeatureVector,
    Returns,
    align_to_polymarket_snapshots,
    save_aligned_features,
)


class TestFeatureBuilder:
    """Test FeatureBuilder."""

    def test_compute_returns(self):
        """Test computing returns over horizons."""
        trades = [
            Trade(
                timestamp_ms=1700000000000,
                price=50000.0,
                size=1.0,
                side="buy",
                trade_id="1",
                venue="coinbase",
                raw_data={},
            ),
            Trade(
                timestamp_ms=1700000005000,  # 5 seconds later
                price=50100.0,  # 0.2% increase
                size=1.0,
                side="buy",
                trade_id="2",
                venue="coinbase",
                raw_data={},
            ),
        ]

        builder = FeatureBuilder(horizons=[5])
        returns = builder.compute_returns(trades, reference_time_ms=1700000005000)

        assert len(returns) == 1
        assert returns[0].horizon_seconds == 5
        assert returns[0].simple_return == pytest.approx(0.002, rel=1e-3)
        assert returns[0].start_price == 50000.0
        assert returns[0].end_price == 50100.0

    def test_compute_returns_empty(self):
        """Test computing returns with no trades."""
        builder = FeatureBuilder()
        returns = builder.compute_returns([])
        assert returns == []

    def test_compute_volume_metrics(self):
        """Test computing volume metrics."""
        trades = [
            Trade(
                timestamp_ms=1700000000000,
                price=50000.0,
                size=1.0,
                side="buy",
                trade_id="1",
                venue="coinbase",
                raw_data={},
            ),
            Trade(
                timestamp_ms=1700000001000,
                price=50000.0,
                size=2.0,
                side="sell",
                trade_id="2",
                venue="coinbase",
                raw_data={},
            ),
        ]

        builder = FeatureBuilder(horizons=[5])
        metrics = builder.compute_volume_metrics(trades, reference_time_ms=1700000001000)

        assert len(metrics) == 1
        assert metrics[0].total_volume == 3.0
        assert metrics[0].buy_volume == 1.0
        assert metrics[0].sell_volume == 2.0
        assert metrics[0].signed_volume == -1.0  # 1 - 2

    def test_build_features(self):
        """Test building complete feature vector."""
        trades = [
            Trade(
                timestamp_ms=1700000000000,
                price=50000.0,
                size=1.0,
                side="buy",
                trade_id="1",
                venue="coinbase",
                raw_data={},
            ),
            Trade(
                timestamp_ms=1700000005000,
                price=50100.0,
                size=1.0,
                side="buy",
                trade_id="2",
                venue="coinbase",
                raw_data={},
            ),
        ]

        builder = FeatureBuilder(horizons=[5])
        features = builder.build_features(trades, venue="coinbase")

        assert features.venue == "coinbase"
        assert features.reference_price == 50100.0
        assert len(features.returns) == 1
        assert len(features.volume_metrics) == 1

    def test_feature_vector_to_dict(self):
        """Test converting FeatureVector to dict."""
        returns = Returns(
            horizon_seconds=5,
            simple_return=0.002,
            log_return=0.001998,
            start_price=50000.0,
            end_price=50100.0,
            start_time="2024-01-01T00:00:00+00:00",
            end_time="2024-01-01T00:00:05+00:00",
        )

        vector = FeatureVector(
            timestamp="2024-01-01T00:00:05+00:00",
            timestamp_ms=1700000005000,
            symbol="BTC-USD",
            venue="coinbase",
            reference_price=50100.0,
            returns=[returns],
        )

        d = vector.to_dict()
        assert d["symbol"] == "BTC-USD"
        assert d["venue"] == "coinbase"
        assert d["reference_price"] == 50100.0
        assert len(d["returns"]) == 1
        assert d["returns"][0]["simple_return"] == 0.002


class TestAlignToPolymarketSnapshots:
    """Test alignment function."""

    def test_align_snapshots(self, tmp_path):
        """Test aligning pricefeed to Polymarket snapshots."""
        # Use correct timestamp_ms for 2024-01-01T00:00:00+00:00
        ts_ms = 1704067200000

        # Create pricefeed directory
        pricefeed_dir = tmp_path / "pricefeed"
        pricefeed_dir.mkdir()

        # Create a pricefeed snapshot
        pf_snapshot = {
            "timestamp": "2024-01-01T00:00:00+00:00",
            "timestamp_ms": ts_ms,
            "symbol": "BTC-USD",
            "venue": "coinbase",
            "current_price": 50000.0,
            "trades": [
                {
                    "timestamp": "2024-01-01T00:00:00+00:00",
                    "timestamp_ms": ts_ms,
                    "price": 50000.0,
                    "size": 1.0,
                    "side": "buy",
                    "trade_id": "1",
                    "venue": "coinbase",
                }
            ],
        }
        (pricefeed_dir / "pricefeed_coinbase_20240101T000000Z.json").write_text(
            json.dumps(pf_snapshot)
        )

        # Create Polymarket directory
        pm_dir = tmp_path / "polymarket"
        pm_dir.mkdir()

        # Create a Polymarket snapshot (same timestamp as pricefeed)
        pm_snapshot = {
            "generated_at": "2024-01-01T00:00:00+00:00",
            "markets": [
                {
                    "market_slug": "bitcoin-up-or-down-15m",
                    "title": "Will Bitcoin go up or down in 15m?",
                    "clob_token_ids": ["token_yes_123", "token_no_456"],
                    "books": {
                        "yes": {
                            "bids": [{"price": "0.55", "size": "100"}],
                            "asks": [{"price": "0.57", "size": "100"}],
                        },
                        "no": {
                            "bids": [{"price": "0.43", "size": "100"}],
                            "asks": [{"price": "0.45", "size": "100"}],
                        },
                    },
                }
            ],
        }
        (pm_dir / "snapshot_15m_20240101T000000Z.json").write_text(json.dumps(pm_snapshot))

        # Align
        aligned = align_to_polymarket_snapshots(
            pricefeed_data_dir=pricefeed_dir,
            polymarket_data_dir=pm_dir,
            tolerance_seconds=1.0,
        )

        assert len(aligned) == 1
        assert aligned[0]["venue"] == "coinbase"
        assert aligned[0]["time_diff_seconds"] == 0.0
        assert "pricefeed_features" in aligned[0]
        assert "polymarket_data" in aligned[0]

    def test_align_no_match(self, tmp_path):
        """Test alignment when no snapshots match."""
        # Use correct timestamp_ms values
        ts_pf = 1704067200000  # 2024-01-01T00:00:00
        # PM timestamp is 60 seconds later (1704067260000)

        pricefeed_dir = tmp_path / "pricefeed"
        pricefeed_dir.mkdir()
        pm_dir = tmp_path / "polymarket"
        pm_dir.mkdir()

        # Create snapshots that are far apart
        pf_snapshot = {
            "timestamp": "2024-01-01T00:00:00+00:00",
            "timestamp_ms": ts_pf,
            "symbol": "BTC-USD",
            "venue": "coinbase",
            "trades": [],
        }
        (pricefeed_dir / "pricefeed_coinbase_20240101T000000Z.json").write_text(
            json.dumps(pf_snapshot)
        )

        pm_snapshot = {
            "generated_at": "2024-01-01T00:01:00+00:00",  # 60 seconds later
            "markets": [],
        }
        (pm_dir / "snapshot_15m_20240101T000100Z.json").write_text(json.dumps(pm_snapshot))

        aligned = align_to_polymarket_snapshots(
            pricefeed_data_dir=pricefeed_dir,
            polymarket_data_dir=pm_dir,
            tolerance_seconds=1.0,  # Only 1 second tolerance
        )

        assert len(aligned) == 0


class TestSaveAlignedFeatures:
    """Test saving aligned features."""

    def test_save_aligned(self, tmp_path):
        """Test saving aligned features to file."""
        aligned = [
            {
                "polymarket_timestamp": "2024-01-01T00:00:00+00:00",
                "pricefeed_timestamp": "2024-01-01T00:00:00+00:00",
                "time_diff_seconds": 0.0,
                "venue": "coinbase",
                "polymarket_data": {},
                "pricefeed_features": {},
            }
        ]

        out_path = tmp_path / "aligned.json"
        save_aligned_features(aligned, out_path)

        assert out_path.exists()
        data = json.loads(out_path.read_text())
        assert len(data) == 1
        assert data[0]["venue"] == "coinbase"
