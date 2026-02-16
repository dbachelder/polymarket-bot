"""Tests for discounted outcome arbitrage strategy."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from polymarket.strategy_discounted_outcome import (
    DiscountedArbitrageTracker,
    DiscountedArbitrageSignal,
    DiscountedMarket,
    InsiderPosition,
    PaperTrade,
    StrategyPerformance,
    detect_vertical,
    find_discounted_markets,
    qualify_insider,
    run_discounted_arbitrage_scan,
)


class TestDetectVertical:
    """Tests for vertical detection."""

    def test_detect_crypto(self):
        """Test crypto vertical detection."""
        assert detect_vertical("Will Bitcoin go up?") == "crypto"
        assert detect_vertical("BTC price prediction") == "crypto"
        assert detect_vertical("Ethereum to $5000?") == "crypto"

    def test_detect_politics(self):
        """Test politics vertical detection."""
        assert detect_vertical("Will Trump win the election?") == "politics"
        assert detect_vertical("Senate vote on bill") == "politics"

    def test_detect_sports(self):
        """Test sports vertical detection."""
        assert detect_vertical("NBA championship winner") == "sports"
        assert detect_vertical("Who will win the game?") == "sports"

    def test_detect_weather(self):
        """Test weather vertical detection."""
        assert detect_vertical("Temperature above 72°F tomorrow") == "weather"
        assert detect_vertical("Will it rain?") == "weather"

    def test_detect_other(self):
        """Test other vertical detection."""
        assert detect_vertical("Some random question?") == "other"


class TestFindDiscountedMarkets:
    """Tests for finding discounted markets."""

    def test_find_discounted_yes(self, tmp_path: Path):
        """Test finding markets with discounted YES."""
        snapshot = {
            "generated_at": datetime.now(UTC).isoformat(),
            "markets": [
                {
                    "market_id": "test-1",
                    "question": "Will BTC go up?",
                    "clobTokenIds": ["yes1", "no1"],
                    "books": {
                        "yes": {
                            "bids": [{"price": "0.30", "size": "100"}],
                            "asks": [{"price": "0.32", "size": "100"}],
                        },
                        "no": {
                            "bids": [{"price": "0.68", "size": "100"}],
                            "asks": [{"price": "0.70", "size": "100"}],
                        },
                    },
                }
            ],
        }

        snapshot_path = tmp_path / "test_snapshot.json"
        snapshot_path.write_text(json.dumps(snapshot))

        markets = find_discounted_markets(snapshot_path, Decimal("0.35"))

        assert len(markets) == 1
        assert markets[0].discounted_side == "YES"
        assert markets[0].discounted_price == pytest.approx(Decimal("0.31"), abs=Decimal("0.01"))

    def test_find_discounted_no(self, tmp_path: Path):
        """Test finding markets with discounted NO."""
        snapshot = {
            "generated_at": datetime.now(UTC).isoformat(),
            "markets": [
                {
                    "market_id": "test-1",
                    "question": "Will it rain?",
                    "clobTokenIds": ["yes1", "no1"],
                    "books": {
                        "yes": {
                            "bids": [{"price": "0.70", "size": "100"}],
                            "asks": [{"price": "0.72", "size": "100"}],
                        },
                        "no": {
                            "bids": [{"price": "0.28", "size": "100"}],
                            "asks": [{"price": "0.30", "size": "100"}],
                        },
                    },
                }
            ],
        }

        snapshot_path = tmp_path / "test_snapshot.json"
        snapshot_path.write_text(json.dumps(snapshot))

        markets = find_discounted_markets(snapshot_path, Decimal("0.35"))

        assert len(markets) == 1
        assert markets[0].discounted_side == "NO"

    def test_no_discount_found(self, tmp_path: Path):
        """Test that markets without discount are not found."""
        snapshot = {
            "generated_at": datetime.now(UTC).isoformat(),
            "markets": [
                {
                    "market_id": "test-1",
                    "question": "Will BTC go up?",
                    "clobTokenIds": ["yes1", "no1"],
                    "books": {
                        "yes": {
                            "bids": [{"price": "0.48", "size": "100"}],
                            "asks": [{"price": "0.50", "size": "100"}],
                        },
                        "no": {
                            "bids": [{"price": "0.50", "size": "100"}],
                            "asks": [{"price": "0.52", "size": "100"}],
                        },
                    },
                }
            ],
        }

        snapshot_path = tmp_path / "test_snapshot.json"
        snapshot_path.write_text(json.dumps(snapshot))

        markets = find_discounted_markets(snapshot_path, Decimal("0.35"))

        assert len(markets) == 0


class TestQualifyInsider:
    """Tests for insider qualification logic."""

    def test_qualifies_as_insider(self):
        """Test wallet that qualifies as insider."""
        from polymarket.copytrade_profiler import TraderPerformanceMetrics

        metrics = TraderPerformanceMetrics(
            address="0x123",
            total_trades=20,
            winning_trades=12,
            losing_trades=8,
            realized_pnl=Decimal("5000"),
            win_rate=60.0,
        )

        assert qualify_insider(metrics) is True

    def test_not_enough_trades(self):
        """Test wallet with insufficient trades."""
        from polymarket.copytrade_profiler import TraderPerformanceMetrics

        metrics = TraderPerformanceMetrics(
            address="0x123",
            total_trades=5,
            winning_trades=3,
            losing_trades=2,
            realized_pnl=Decimal("2000"),
            win_rate=60.0,
        )

        assert qualify_insider(metrics) is False

    def test_low_win_rate(self):
        """Test wallet with low win rate."""
        from polymarket.copytrade_profiler import TraderPerformanceMetrics

        metrics = TraderPerformanceMetrics(
            address="0x123",
            total_trades=20,
            winning_trades=8,
            losing_trades=12,
            realized_pnl=Decimal("5000"),
            win_rate=40.0,
        )

        assert qualify_insider(metrics) is False

    def test_low_pnl(self):
        """Test wallet with low PnL."""
        from polymarket.copytrade_profiler import TraderPerformanceMetrics

        metrics = TraderPerformanceMetrics(
            address="0x123",
            total_trades=20,
            winning_trades=12,
            losing_trades=8,
            realized_pnl=Decimal("500"),
            win_rate=60.0,
        )

        assert qualify_insider(metrics) is False

    def test_none_metrics(self):
        """Test with None metrics."""
        assert qualify_insider(None) is False


class TestDiscountedArbitrageSignal:
    """Tests for signal generation."""

    def test_has_insider_confirmation(self):
        """Test signal with insider confirmation."""
        market = DiscountedMarket(
            market_id="test",
            token_id_yes="yes",
            token_id_no="no",
            question="Test?",
            vertical="crypto",
            yes_price=Decimal("0.30"),
            no_price=Decimal("0.70"),
            discounted_side="YES",
            discounted_price=Decimal("0.30"),
            end_date=None,
            volume_usd=None,
        )

        insider = InsiderPosition(
            wallet_address="0x123",
            token_id="yes",
            market_id="test",
            side="YES",
            position_size_usd=Decimal("1500"),
            entry_price=Decimal("0.28"),
            entry_time=None,
            wallet_pnl=Decimal("5000"),
            wallet_win_rate=65.0,
            wallet_total_trades=20,
        )

        signal = DiscountedArbitrageSignal(
            timestamp=datetime.now(UTC),
            market=market,
            insider_confirmations=[insider],
            confirmation_count=1,
            total_insider_position_usd=Decimal("1500"),
            avg_insider_win_rate=65.0,
        )

        assert signal.has_insider_confirmation is True
        assert signal.confidence_score > 0

    def test_no_insider_confirmation(self):
        """Test signal without insider confirmation."""
        market = DiscountedMarket(
            market_id="test",
            token_id_yes="yes",
            token_id_no="no",
            question="Test?",
            vertical="crypto",
            yes_price=Decimal("0.30"),
            no_price=Decimal("0.70"),
            discounted_side="YES",
            discounted_price=Decimal("0.30"),
            end_date=None,
            volume_usd=None,
        )

        signal = DiscountedArbitrageSignal(
            timestamp=datetime.now(UTC),
            market=market,
            insider_confirmations=[],
            confirmation_count=0,
            total_insider_position_usd=Decimal("0"),
            avg_insider_win_rate=0.0,
        )

        assert signal.has_insider_confirmation is False
        assert signal.confidence_score == 0.0


class TestStrategyPerformance:
    """Tests for performance tracking."""

    def test_overall_win_rate(self):
        """Test overall win rate calculation."""
        perf = StrategyPerformance(
            resolved_trades=10,
            winning_trades=6,
            losing_trades=4,
        )

        assert perf.overall_win_rate == 60.0

    def test_no_resolved_trades(self):
        """Test win rate with no resolved trades."""
        perf = StrategyPerformance()

        assert perf.overall_win_rate == 0.0

    def test_insider_win_rate(self):
        """Test insider win rate calculation."""
        perf = StrategyPerformance(
            insider_confirmed_trades=10,
            insider_winning=7,
            insider_losing=3,
        )

        assert perf.insider_win_rate == 70.0


class TestDiscountedArbitrageTracker:
    """Tests for the tracker."""

    def test_load_save_trades(self, tmp_path: Path):
        """Test loading and saving trades."""
        tracker = DiscountedArbitrageTracker(data_dir=tmp_path)

        # Create a trade
        trade = PaperTrade(
            trade_id="test-1",
            timestamp=datetime.now(UTC),
            signal=DiscountedArbitrageSignal(
                timestamp=datetime.now(UTC),
                market=DiscountedMarket(
                    market_id="m1",
                    token_id_yes="y1",
                    token_id_no="n1",
                    question="Test?",
                    vertical="crypto",
                    yes_price=Decimal("0.30"),
                    no_price=Decimal("0.70"),
                    discounted_side="YES",
                    discounted_price=Decimal("0.30"),
                    end_date=None,
                    volume_usd=Decimal("10000"),
                ),
                insider_confirmations=[],
                confirmation_count=0,
                total_insider_position_usd=Decimal("0"),
                avg_insider_win_rate=0.0,
            ),
            side="buy_yes",
            position_size=Decimal("33.33"),
            entry_price=Decimal("0.30"),
            expected_value=Decimal("6.67"),
        )

        tracker.record_trade(trade)

        loaded = tracker.load_trades()
        assert len(loaded) == 1
        assert loaded[0].trade_id == "test-1"

    def test_get_performance_empty(self, tmp_path: Path):
        """Test performance with no trades."""
        tracker = DiscountedArbitrageTracker(data_dir=tmp_path)

        perf = tracker.get_performance()

        assert perf.total_trades == 0
        assert perf.overall_win_rate == 0.0

    def test_get_trades_by_vertical(self, tmp_path: Path):
        """Test grouping trades by vertical."""
        tracker = DiscountedArbitrageTracker(data_dir=tmp_path)

        # Create trades in different verticals
        for vertical in ["crypto", "politics", "crypto"]:
            trade = PaperTrade(
                trade_id=f"test-{vertical}",
                timestamp=datetime.now(UTC),
                signal=DiscountedArbitrageSignal(
                    timestamp=datetime.now(UTC),
                    market=DiscountedMarket(
                        market_id="m1",
                        token_id_yes="y1",
                        token_id_no="n1",
                        question="Test?",
                        vertical=vertical,
                        yes_price=Decimal("0.30"),
                        no_price=Decimal("0.70"),
                        discounted_side="YES",
                        discounted_price=Decimal("0.30"),
                        end_date=None,
                        volume_usd=None,
                    ),
                    insider_confirmations=[],
                    confirmation_count=0,
                    total_insider_position_usd=Decimal("0"),
                    avg_insider_win_rate=0.0,
                ),
                side="buy_yes",
                position_size=Decimal("33.33"),
                entry_price=Decimal("0.30"),
                expected_value=Decimal("6.67"),
            )
            tracker.record_trade(trade)

        by_vertical = tracker.get_trades_by_vertical()

        assert len(by_vertical.get("crypto", [])) == 2
        assert len(by_vertical.get("politics", [])) == 1


class TestRunDiscountedArbitrageScan:
    """Integration tests for the scan function."""

    def test_scan_no_snapshots(self, tmp_path: Path):
        """Test scan with no snapshots available."""
        result = run_discounted_arbitrage_scan(
            snapshots_dir=tmp_path,
            data_dir=tmp_path / "data",
            dry_run=True,
        )

        assert "error" in result
        assert "No snapshots found" in result["error"]

    def test_scan_with_snapshot(self, tmp_path: Path):
        """Test scan with available snapshot."""
        # Create a snapshot
        snapshot_dir = tmp_path / "snapshots"
        snapshot_dir.mkdir()

        snapshot = {
            "generated_at": datetime.now(UTC).isoformat(),
            "markets": [
                {
                    "market_id": "test-1",
                    "question": "Will BTC go up?",
                    "clobTokenIds": ["yes1", "no1"],
                    "end_date": (datetime.now(UTC) + timedelta(days=1)).isoformat(),
                    "books": {
                        "yes": {
                            "bids": [{"price": "0.30", "size": "100"}],
                            "asks": [{"price": "0.32", "size": "100"}],
                        },
                        "no": {
                            "bids": [{"price": "0.68", "size": "100"}],
                            "asks": [{"price": "0.70", "size": "100"}],
                        },
                    },
                }
            ],
        }

        snapshot_path = snapshot_dir / "snapshot_test_20260216T000000Z.json"
        snapshot_path.write_text(json.dumps(snapshot))

        result = run_discounted_arbitrage_scan(
            snapshots_dir=snapshot_dir,
            data_dir=tmp_path / "data",
            dry_run=True,
        )

        assert "error" not in result
        assert result["markets_discounted"] == 1
        assert "crypto" in result.get("by_vertical", {})
