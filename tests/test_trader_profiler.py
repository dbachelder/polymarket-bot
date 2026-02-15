"""Tests for trader profiling module."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

from polymarket.trader_profiler import TraderProfile, TraderProfiler, TraderScore


class TestTraderProfile:
    """Tests for TraderProfile dataclass."""

    def test_creation(self) -> None:
        """Test creating a TraderProfile."""
        profile = TraderProfile(
            address="0x1234567890abcdef",
            username="test_trader",
            pnl_lifetime=Decimal("10000"),
            pnl_30d=Decimal("5000"),
            pnl_7d=Decimal("1000"),
            volume_lifetime=Decimal("100000"),
            markets_traded=10,
            rank=1,
            source="leaderboard",
        )

        assert profile.address == "0x1234567890abcdef"
        assert profile.username == "test_trader"
        assert profile.pnl_lifetime == Decimal("10000")
        assert profile.rank == 1

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        profile = TraderProfile(
            address="0x1234",
            pnl_lifetime=Decimal("1000"),
            volume_lifetime=Decimal("5000"),
        )

        data = profile.to_dict()

        assert data["address"] == "0x1234"
        assert data["pnl_lifetime"] == 1000.0
        assert data["volume_lifetime"] == 5000.0

    def test_from_dict(self) -> None:
        """Test creation from dictionary."""
        data = {
            "address": "0x1234",
            "username": "test",
            "pnl_lifetime": 1000.0,
            "pnl_30d": 500.0,
            "pnl_7d": 100.0,
            "volume_lifetime": 5000.0,
            "markets_traded": 5,
            "rank": 1,
            "source": "leaderboard",
            "discovered_at": "2024-01-01T00:00:00+00:00",
            "last_updated": "2024-01-01T00:00:00+00:00",
            "tags": ["top_performer"],
        }

        profile = TraderProfile.from_dict(data)

        assert profile.address == "0x1234"
        assert profile.username == "test"
        assert profile.pnl_lifetime == Decimal("1000")

    def test_pnl_30d_roi(self) -> None:
        """Test 30d ROI calculation."""
        profile = TraderProfile(
            address="0x1234",
            pnl_30d=Decimal("1000"),
            volume_lifetime=Decimal("120000"),  # ~10k monthly
        )

        roi = profile.pnl_30d_roi

        # Expected: 1000 / 10000 = 0.1 = 10%
        assert roi > Decimal("0.09")
        assert roi < Decimal("0.11")

    def test_consistency_score_both_positive(self) -> None:
        """Test consistency score when both 7d and 30d are positive."""
        profile = TraderProfile(
            address="0x1234",
            pnl_30d=Decimal("4000"),
            pnl_7d=Decimal("1000"),  # Exactly 1/4 of 30d
        )

        score = profile.consistency_score

        assert score >= 90.0  # Should be high consistency

    def test_consistency_score_negative(self) -> None:
        """Test consistency score when periods are negative."""
        profile = TraderProfile(
            address="0x1234",
            pnl_30d=Decimal("-1000"),
            pnl_7d=Decimal("-500"),
        )

        score = profile.consistency_score

        assert score < 50.0  # Should be low


class TestTraderScore:
    """Tests for TraderScore dataclass."""

    def test_creation(self) -> None:
        """Test creating a TraderScore."""
        score = TraderScore(
            address="0x1234",
            total_score=75.0,
            pnl_score=80.0,
            consistency_score=70.0,
        )

        assert score.address == "0x1234"
        assert score.total_score == 75.0

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        score = TraderScore(address="0x1234", total_score=75.0)

        data = score.to_dict()

        assert data["address"] == "0x1234"
        assert data["total_score"] == 75.0

    def test_from_dict(self) -> None:
        """Test creation from dictionary."""
        data = {
            "address": "0x1234",
            "total_score": 75.0,
            "pnl_score": 80.0,
            "consistency_score": 70.0,
            "volume_score": 60.0,
            "diversity_score": 50.0,
            "computed_at": "2024-01-01T00:00:00+00:00",
        }

        score = TraderScore.from_dict(data)

        assert score.address == "0x1234"
        assert score.total_score == 75.0


class TestTraderProfiler:
    """Tests for TraderProfiler class."""

    def test_init_creates_directories(self, tmp_path: Path) -> None:
        """Test initialization creates data directories."""
        data_dir = tmp_path / "test_profiles"

        profiler = TraderProfiler(data_dir=data_dir)

        assert data_dir.exists()
        assert (data_dir / "fills").exists()
        assert (data_dir / "nav").exists()

    def test_add_trader(self, tmp_path: Path) -> None:
        """Test adding a trader."""
        profiler = TraderProfiler(data_dir=tmp_path)

        profile = TraderProfile(
            address="0x1234",
            username="test",
            pnl_30d=Decimal("1000"),
        )

        profiler.add_or_update_trader(profile)

        assert "0x1234" in profiler.traders
        assert profiler.traders["0x1234"].username == "test"

    def test_add_trader_merges(self, tmp_path: Path) -> None:
        """Test that adding trader merges with existing."""
        profiler = TraderProfiler(data_dir=tmp_path)

        profile1 = TraderProfile(
            address="0x1234",
            username="original",
            pnl_30d=Decimal("1000"),
        )
        profiler.add_or_update_trader(profile1)

        profile2 = TraderProfile(
            address="0x1234",
            username="updated",
            pnl_7d=Decimal("500"),
        )
        profiler.add_or_update_trader(profile2)

        trader = profiler.traders["0x1234"]
        assert trader.username == "original"  # Preserved
        assert trader.pnl_30d == Decimal("1000")  # Preserved

    def test_save_and_load(self, tmp_path: Path) -> None:
        """Test saving and loading traders."""
        profiler1 = TraderProfiler(data_dir=tmp_path)

        profile = TraderProfile(
            address="0x1234",
            username="test",
            pnl_30d=Decimal("1000"),
        )
        profiler1.add_or_update_trader(profile)
        profiler1.save_traders()

        # Create new profiler to load
        profiler2 = TraderProfiler(data_dir=tmp_path)

        assert "0x1234" in profiler2.traders
        assert profiler2.traders["0x1234"].username == "test"

    def test_compute_scores(self, tmp_path: Path) -> None:
        """Test computing trader scores."""
        profiler = TraderProfiler(data_dir=tmp_path)

        # Add traders with different performance
        profiler.add_or_update_trader(TraderProfile(
            address="0x1",
            pnl_30d=Decimal("10000"),
            volume_lifetime=Decimal("1000000"),
            markets_traded=10,
            source="leaderboard",
        ))
        profiler.add_or_update_trader(TraderProfile(
            address="0x2",
            pnl_30d=Decimal("1000"),
            volume_lifetime=Decimal("100000"),
            markets_traded=5,
            source="leaderboard",
        ))

        scores = profiler.compute_scores(min_volume=Decimal("1000"), min_markets=2)

        assert len(scores) == 2
        # Higher PnL trader should have higher score
        assert scores[0].address == "0x1"
        assert scores[0].total_score > scores[1].total_score

    def test_get_top_traders(self, tmp_path: Path) -> None:
        """Test getting top traders."""
        profiler = TraderProfiler(data_dir=tmp_path)

        # Add traders and compute scores
        for i in range(10):
            profiler.add_or_update_trader(TraderProfile(
                address=f"0x{i}",
                pnl_30d=Decimal(str(10000 - i * 1000)),
                volume_lifetime=Decimal("100000"),
                markets_traded=5,
                source="leaderboard",
            ))

        profiler.compute_scores()
        top = profiler.get_top_traders(k=3)

        assert len(top) == 3
        # Should be sorted by score
        scores = [s.total_score for _, s in top]
        assert scores == sorted(scores, reverse=True)


class TestTraderProfilerFiltering:
    """Tests for trader filtering logic."""

    def test_compute_scores_filters_volume(self, tmp_path: Path) -> None:
        """Test that compute_scores filters by volume."""
        profiler = TraderProfiler(data_dir=tmp_path)

        profiler.add_or_update_trader(TraderProfile(
            address="0x1",
            pnl_30d=Decimal("10000"),
            volume_lifetime=Decimal("100"),  # Below threshold
            markets_traded=10,
            source="leaderboard",
        ))

        scores = profiler.compute_scores(min_volume=Decimal("1000"))

        assert len(scores) == 0  # Filtered out

    def test_compute_scores_filters_markets(self, tmp_path: Path) -> None:
        """Test that compute_scores filters by markets traded."""
        profiler = TraderProfiler(data_dir=tmp_path)

        profiler.add_or_update_trader(TraderProfile(
            address="0x1",
            pnl_30d=Decimal("10000"),
            volume_lifetime=Decimal("100000"),
            markets_traded=1,  # Below threshold
            source="leaderboard",
        ))

        scores = profiler.compute_scores(min_markets=3)

        assert len(scores) == 0  # Filtered out

    def test_get_top_traders_min_score(self, tmp_path: Path) -> None:
        """Test filtering top traders by minimum score."""
        profiler = TraderProfiler(data_dir=tmp_path)

        # Add traders with varying scores
        profiler.add_or_update_trader(TraderProfile(
            address="0x1",
            pnl_30d=Decimal("100000"),
            volume_lifetime=Decimal("1000000"),
            markets_traded=20,
            source="leaderboard",
        ))
        profiler.add_or_update_trader(TraderProfile(
            address="0x2",
            pnl_30d=Decimal("100"),
            volume_lifetime=Decimal("100000"),
            markets_traded=3,
            source="leaderboard",
        ))

        profiler.compute_scores()
        top = profiler.get_top_traders(k=10, min_score=50.0)

        # Should only include high-scoring traders
        assert all(s.total_score >= 50.0 for _, s in top)
