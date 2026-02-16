"""Tests for copytrade profiler module."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

import json

from polymarket.copytrade_profiler import (
    CopytradeProfiler,
    TradeRoundTrip,
    TraderPerformanceMetrics,
    TraderRanking,
)
from polymarket.trader_fills import TraderFill


class TestTradeRoundTrip:
    """Tests for TradeRoundTrip dataclass."""

    def test_creation(self) -> None:
        """Test creating a TradeRoundTrip."""
        now = datetime.now(UTC)
        rt = TradeRoundTrip(
            token_id="token123",
            market_slug="test-market",
            entry_time=now - timedelta(hours=2),
            exit_time=now,
            entry_price=Decimal("0.5"),
            exit_price=Decimal("0.6"),
            size=Decimal("100"),
            fees=Decimal("0.5"),
            realized_pnl=Decimal("9.5"),
            realized_pnl_pct=Decimal("19.0"),
        )

        assert rt.token_id == "token123"
        assert rt.is_win is True
        assert rt.hold_time.total_seconds() == pytest.approx(7200, abs=1)

    def test_is_win_false(self) -> None:
        """Test is_win property for losing trade."""
        now = datetime.now(UTC)
        rt = TradeRoundTrip(
            token_id="token123",
            market_slug=None,
            entry_time=now - timedelta(hours=1),
            exit_time=now,
            entry_price=Decimal("0.5"),
            exit_price=Decimal("0.4"),
            size=Decimal("100"),
            fees=Decimal("0.5"),
            realized_pnl=Decimal("-10.5"),
            realized_pnl_pct=Decimal("-21.0"),
        )

        assert rt.is_win is False


class TestTraderPerformanceMetrics:
    """Tests for TraderPerformanceMetrics dataclass."""

    def test_creation(self) -> None:
        """Test creating TraderPerformanceMetrics."""
        metrics = TraderPerformanceMetrics(
            address="0x1234",
            total_trades=10,
            winning_trades=6,
            losing_trades=4,
            realized_pnl=Decimal("1000"),
            win_rate=60.0,
        )

        assert metrics.address == "0x1234"
        assert metrics.total_trades == 10
        assert metrics.win_rate == 60.0

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        metrics = TraderPerformanceMetrics(
            address="0x1234",
            realized_pnl=Decimal("1000.50"),
            win_rate=60.0,
        )

        data = metrics.to_dict()

        assert data["address"] == "0x1234"
        assert data["realized_pnl"] == 1000.50
        assert data["win_rate"] == 60.0

    def test_from_dict(self) -> None:
        """Test creation from dictionary."""
        data = {
            "address": "0x1234",
            "computed_at": "2024-01-01T00:00:00+00:00",
            "total_trades": 10,
            "winning_trades": 6,
            "losing_trades": 4,
            "realized_pnl": 1000.0,
            "total_fees": 50.0,
            "gross_pnl": 1050.0,
            "avg_trade_pnl": 100.0,
            "best_trade": 500.0,
            "worst_trade": -100.0,
            "win_rate": 60.0,
            "avg_hold_time_hours": 2.5,
            "median_hold_time_hours": 2.0,
            "min_hold_time_hours": 0.5,
            "max_hold_time_hours": 5.0,
            "pnl_volatility": 150.0,
            "sharpe_like_ratio": 0.67,
            "max_consecutive_wins": 3,
            "max_consecutive_losses": 2,
            "total_volume": 10000.0,
            "unique_markets": 5,
            "first_trade_at": "2024-01-01T00:00:00+00:00",
            "last_trade_at": "2024-01-10T00:00:00+00:00",
            "trading_days": 10,
        }

        metrics = TraderPerformanceMetrics.from_dict(data)

        assert metrics.address == "0x1234"
        assert metrics.total_trades == 10
        assert metrics.realized_pnl == Decimal("1000")
        assert metrics.win_rate == 60.0


class TestTraderRanking:
    """Tests for TraderRanking dataclass."""

    def test_creation(self) -> None:
        """Test creating TraderRanking."""
        ranking = TraderRanking(
            address="0x1234",
            rank=1,
            overall_score=85.5,
            pnl_score=90.0,
            win_rate_score=80.0,
        )

        assert ranking.address == "0x1234"
        assert ranking.rank == 1
        assert ranking.overall_score == 85.5

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        ranking = TraderRanking(
            address="0x1234",
            rank=1,
            overall_score=85.5,
            realized_pnl=Decimal("1000"),
            win_rate=60.0,
        )

        data = ranking.to_dict()

        assert data["address"] == "0x1234"
        assert data["rank"] == 1
        assert data["overall_score"] == 85.5
        assert data["realized_pnl"] == 1000.0


class TestCopytradeProfiler:
    """Tests for CopytradeProfiler class."""

    def test_init_creates_directories(self, tmp_path: Path) -> None:
        """Test initialization creates data directories."""
        data_dir = tmp_path / "test_copytrade"

        _profiler = CopytradeProfiler(data_dir=data_dir)

        assert data_dir.exists()
        assert (data_dir / "metrics").exists()

    def test_compute_metrics_empty_fills(self, tmp_path: Path) -> None:
        """Test compute_metrics returns empty metrics when no fills."""
        profiler = CopytradeProfiler(data_dir=tmp_path)

        metrics = profiler.compute_metrics("0x1234")

        assert metrics.address == "0x1234"
        assert metrics.total_trades == 0
        assert metrics.realized_pnl == Decimal("0")

    def test_compute_round_trips_fifo_matching(self, tmp_path: Path) -> None:
        """Test FIFO matching for round trips."""
        profiler = CopytradeProfiler(data_dir=tmp_path)

        # Create fills: buy 100 @ 0.5, sell 100 @ 0.6
        now = datetime.now(UTC)
        fills = [
            TraderFill(
                token_id="token123",
                side="buy",
                size=Decimal("100"),
                price=Decimal("0.5"),
                fee=Decimal("0.5"),
                timestamp=(now - timedelta(hours=2)).isoformat(),
                transaction_hash="tx1",
                market_slug="test-market",
            ),
            TraderFill(
                token_id="token123",
                side="sell",
                size=Decimal("100"),
                price=Decimal("0.6"),
                fee=Decimal("0.5"),
                timestamp=now.isoformat(),
                transaction_hash="tx2",
                market_slug="test-market",
            ),
        ]

        round_trips = profiler._compute_round_trips(fills)

        assert len(round_trips) == 1
        assert round_trips[0].size == Decimal("100")
        assert round_trips[0].is_win is True
        # PnL = 100 * (0.6 - 0.5) - 1.0 fees = 9.0
        assert round_trips[0].realized_pnl > 0

    def test_compute_round_trips_partial_sell(self, tmp_path: Path) -> None:
        """Test partial sell matching."""
        profiler = CopytradeProfiler(data_dir=tmp_path)

        now = datetime.now(UTC)
        fills = [
            TraderFill(
                token_id="token123",
                side="buy",
                size=Decimal("100"),
                price=Decimal("0.5"),
                fee=Decimal("0.5"),
                timestamp=(now - timedelta(hours=2)).isoformat(),
            ),
            TraderFill(
                token_id="token123",
                side="sell",
                size=Decimal("50"),  # Partial sell
                price=Decimal("0.6"),
                fee=Decimal("0.3"),
                timestamp=now.isoformat(),
            ),
        ]

        round_trips = profiler._compute_round_trips(fills)

        assert len(round_trips) == 1
        assert round_trips[0].size == Decimal("50")
        assert round_trips[0].is_win is True

    def test_compute_metrics_with_round_trips(self, tmp_path: Path) -> None:
        """Test compute_metrics with actual round trips."""
        profiler = CopytradeProfiler(data_dir=tmp_path)

        now = datetime.now(UTC)
        address = "0x1234"

        # Create fills file
        fills_dir = tmp_path / "fills"
        fills_dir.mkdir(parents=True, exist_ok=True)
        fills_file = fills_dir / f"{address}.jsonl"

        fills_data = [
            {
                "token_id": "token123",
                "side": "buy",
                "size": "100",
                "price": "0.5",
                "fee": "0.5",
                "timestamp": (now - timedelta(hours=2)).isoformat(),
                "transaction_hash": "tx1",
                "market_slug": "test-market",
            },
            {
                "token_id": "token123",
                "side": "sell",
                "size": "100",
                "price": "0.6",
                "fee": "0.5",
                "timestamp": now.isoformat(),
                "transaction_hash": "tx2",
                "market_slug": "test-market",
            },
        ]

        with open(fills_file, "w") as f:
            for fill in fills_data:
                f.write(json.dumps(fill) + "\n")

        metrics = profiler.compute_metrics(address)

        assert metrics.total_trades == 1
        assert metrics.winning_trades == 1
        assert metrics.losing_trades == 0
        assert metrics.win_rate == 100.0
        assert metrics.realized_pnl > 0

    def test_save_and_load_metrics(self, tmp_path: Path) -> None:
        """Test saving and loading metrics."""
        profiler = CopytradeProfiler(data_dir=tmp_path)

        metrics = TraderPerformanceMetrics(
            address="0x1234",
            total_trades=10,
            realized_pnl=Decimal("1000"),
            win_rate=60.0,
        )

        saved_path = profiler.save_metrics("0x1234", metrics)

        assert saved_path.exists()

        loaded = profiler.load_metrics("0x1234")

        assert loaded is not None
        assert loaded.address == "0x1234"
        assert loaded.total_trades == 10
        assert loaded.realized_pnl == Decimal("1000")

    def test_rank_traders(self, tmp_path: Path) -> None:
        """Test ranking traders."""
        profiler = CopytradeProfiler(data_dir=tmp_path)

        # Create metrics for multiple traders
        addresses = ["0x1111", "0x2222", "0x3333"]

        for addr in addresses:
            profiler._metrics_cache[addr] = TraderPerformanceMetrics(
                address=addr,
                total_trades=10,
                winning_trades=6 if addr == "0x1111" else 5,
                losing_trades=4 if addr == "0x1111" else 5,
                realized_pnl=Decimal("1000") if addr == "0x1111" else Decimal("500"),
                win_rate=60.0 if addr == "0x1111" else 50.0,
                total_volume=Decimal("10000"),
                trading_days=10,
            )

        rankings = profiler.rank_traders(addresses, min_trades=5)

        assert len(rankings) == 3
        # Best trader should be ranked #1
        assert rankings[0].address == "0x1111"
        assert rankings[0].rank == 1

    def test_rank_traders_filters_min_trades(self, tmp_path: Path) -> None:
        """Test that rank_traders filters by min_trades."""
        profiler = CopytradeProfiler(data_dir=tmp_path)

        addresses = ["0x1111", "0x2222"]

        profiler._metrics_cache["0x1111"] = TraderPerformanceMetrics(
            address="0x1111",
            total_trades=10,
            realized_pnl=Decimal("1000"),
        )
        profiler._metrics_cache["0x2222"] = TraderPerformanceMetrics(
            address="0x2222",
            total_trades=2,  # Below threshold
            realized_pnl=Decimal("5000"),
        )

        rankings = profiler.rank_traders(addresses, min_trades=5)

        assert len(rankings) == 1
        assert rankings[0].address == "0x1111"

    def test_get_top_traders(self, tmp_path: Path) -> None:
        """Test getting top traders."""
        profiler = CopytradeProfiler(data_dir=tmp_path)

        addresses = [f"0x{i:04d}" for i in range(10)]

        for i, addr in enumerate(addresses):
            profiler._metrics_cache[addr] = TraderPerformanceMetrics(
                address=addr,
                total_trades=10,
                realized_pnl=Decimal(str(1000 - i * 100)),  # Decreasing PnL
                win_rate=50.0 + i * 2,
                total_volume=Decimal("10000"),
                trading_days=10,
            )

        top = profiler.get_top_traders(addresses, k=3, min_trades=5)

        assert len(top) == 3
        # Should be sorted by score descending
        assert top[0].address == "0x0000"  # Highest PnL

    def test_get_trader_summary(self, tmp_path: Path) -> None:
        """Test getting trader summary."""
        profiler = CopytradeProfiler(data_dir=tmp_path)

        profiler._metrics_cache["0x1234"] = TraderPerformanceMetrics(
            address="0x1234",
            total_trades=10,
            winning_trades=6,
            losing_trades=4,
            realized_pnl=Decimal("1000"),
            win_rate=60.0,
            avg_hold_time_hours=2.5,
            best_trade=Decimal("500"),
            worst_trade=Decimal("-100"),
            total_volume=Decimal("10000"),
            unique_markets=5,
            trading_days=10,
        )

        summary = profiler.get_trader_summary("0x1234")

        assert summary["address"] == "0x1234"
        assert summary["performance"]["realized_pnl_usd"] == 1000.0
        assert summary["performance"]["win_rate_pct"] == 60.0
        assert summary["hold_times"]["avg_hours"] == 2.5

    def test_export_rankings(self, tmp_path: Path) -> None:
        """Test exporting rankings to file."""
        profiler = CopytradeProfiler(data_dir=tmp_path)

        addresses = ["0x1111", "0x2222"]

        for addr in addresses:
            profiler._metrics_cache[addr] = TraderPerformanceMetrics(
                address=addr,
                total_trades=10,
                realized_pnl=Decimal("1000"),
                win_rate=50.0,
                total_volume=Decimal("10000"),
                trading_days=10,
            )

        output_path = tmp_path / "rankings_test.json"
        result_path = profiler.export_rankings(addresses, output_path)

        assert result_path.exists()

        # Verify content
        data = json.loads(result_path.read_text())
        assert "exported_at" in data
        assert "rankings" in data
        assert len(data["rankings"]) == 2
