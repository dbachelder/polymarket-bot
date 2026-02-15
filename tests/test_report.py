"""Tests for polymarket.report module.

Tests cover:
- Collector health analysis (snapshot parsing, gap detection, freshness, backoff)
- BTC microstructure stats (book parsing, spread/depth/imbalance)
- Momentum signal computation
- End-to-end hourly digest generation
- Edge cases (empty data, missing files, malformed JSON)
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest import mock

import pytest

from polymarket.report import (
    CollectorHealth,
    HourlyDigest,
    MicrostructureStats,
    MomentumSignal,
    _parse_snapshot_timestamp,
    analyze_btc_microstructure,
    analyze_collector_health,
    compute_momentum_signal,
    generate_hourly_digest,
)


class TestParseSnapshotTimestamp:
    """Tests for _parse_snapshot_timestamp function."""

    def test_valid_filename(self) -> None:
        """Parse valid snapshot filename."""
        result = _parse_snapshot_timestamp("snapshot_15m_20260215T053045Z.json")
        assert result is not None
        assert result.year == 2026
        assert result.month == 2
        assert result.day == 15
        assert result.hour == 5
        assert result.minute == 30
        assert result.second == 45
        assert result.tzinfo == UTC

    def test_invalid_prefix_returns_none(self) -> None:
        """Non-snapshot filenames return None."""
        assert _parse_snapshot_timestamp("other_file.json") is None
        assert _parse_snapshot_timestamp("snapshot_1h_20260215T053045Z.json") is None

    def test_invalid_timestamp_format_returns_none(self) -> None:
        """Malformed timestamps return None."""
        assert _parse_snapshot_timestamp("snapshot_15m_invalid.json") is None
        assert _parse_snapshot_timestamp("snapshot_15m_2026-02-15T05:30:45Z.json") is None

    def test_empty_string_returns_none(self) -> None:
        """Empty string returns None."""
        assert _parse_snapshot_timestamp("") is None


class TestCollectorHealth:
    """Tests for analyze_collector_health function."""

    @pytest.fixture
    def temp_data_dir(self, tmp_path: Path) -> Path:
        """Create a temporary data directory."""
        return tmp_path / "data"

    def test_empty_directory(self, temp_data_dir: Path) -> None:
        """Empty directory returns zeroed health metrics."""
        temp_data_dir.mkdir()
        now = datetime(2026, 2, 15, 12, 0, 0, tzinfo=UTC)

        result = analyze_collector_health(temp_data_dir, now=now)

        assert result.latest_snapshot_at is None
        assert result.freshness_seconds is None
        assert result.snapshots_last_hour == 0
        assert result.expected_snapshots == 720  # 3600 / 5
        assert result.capture_rate_pct == 0.0
        assert result.backoff_evidence is False
        assert result.snapshot_dir == str(temp_data_dir.absolute())

    def test_single_snapshot(self, temp_data_dir: Path) -> None:
        """Single snapshot within last hour."""
        temp_data_dir.mkdir()
        now = datetime(2026, 2, 15, 12, 0, 0, tzinfo=UTC)
        snapshot_time = now - timedelta(minutes=30)
        filename = snapshot_time.strftime("snapshot_15m_%Y%m%dT%H%M%SZ.json")
        (temp_data_dir / filename).write_text("{}")

        result = analyze_collector_health(temp_data_dir, now=now)

        assert result.latest_snapshot_at == snapshot_time.isoformat()
        assert result.freshness_seconds == 1800.0  # 30 minutes
        assert result.snapshots_last_hour == 1
        assert result.capture_rate_pct == pytest.approx(0.139, rel=0.01)  # 1/720

    def test_multiple_snapshots_last_hour(self, temp_data_dir: Path) -> None:
        """Multiple snapshots within last hour."""
        temp_data_dir.mkdir()
        now = datetime(2026, 2, 15, 12, 0, 0, tzinfo=UTC)

        # Create 10 snapshots spaced 5 minutes apart
        for i in range(10):
            snapshot_time = now - timedelta(minutes=i * 5)
            filename = snapshot_time.strftime("snapshot_15m_%Y%m%dT%H%M%SZ.json")
            (temp_data_dir / filename).write_text("{}")

        result = analyze_collector_health(temp_data_dir, now=now)

        assert result.snapshots_last_hour == 10
        assert result.freshness_seconds == 0.0  # Most recent

    def test_old_snapshots_not_counted(self, temp_data_dir: Path) -> None:
        """Snapshots older than 1 hour not counted in last hour."""
        temp_data_dir.mkdir()
        now = datetime(2026, 2, 15, 12, 0, 0, tzinfo=UTC)

        # Create old snapshot (2 hours ago)
        old_time = now - timedelta(hours=2)
        old_filename = old_time.strftime("snapshot_15m_%Y%m%dT%H%M%SZ.json")
        (temp_data_dir / old_filename).write_text("{}")

        # Create recent snapshot
        recent_time = now - timedelta(minutes=30)
        recent_filename = recent_time.strftime("snapshot_15m_%Y%m%dT%H%M%SZ.json")
        (temp_data_dir / recent_filename).write_text("{}")

        result = analyze_collector_health(temp_data_dir, now=now)

        assert result.snapshots_last_hour == 1
        assert result.latest_snapshot_at == recent_time.isoformat()

    def test_backoff_evidence_detected(self, temp_data_dir: Path) -> None:
        """Detect gaps suggesting backoff (gap > 3x expected interval)."""
        temp_data_dir.mkdir()
        now = datetime(2026, 2, 15, 12, 0, 0, tzinfo=UTC)

        # Create snapshots with a large gap (30 seconds vs expected 5s)
        times = [
            now - timedelta(minutes=10),
            now - timedelta(minutes=9, seconds=30),  # 30s gap (expected 5s)
            now - timedelta(minutes=5),
            now,
        ]

        for t in times:
            filename = t.strftime("snapshot_15m_%Y%m%dT%H%M%SZ.json")
            (temp_data_dir / filename).write_text("{}")

        result = analyze_collector_health(temp_data_dir, interval_seconds=5.0, now=now)

        assert result.backoff_evidence is True

    def test_no_backoff_with_normal_gaps(self, temp_data_dir: Path) -> None:
        """Normal gaps don't trigger backoff evidence."""
        temp_data_dir.mkdir()
        now = datetime(2026, 2, 15, 12, 0, 0, tzinfo=UTC)

        # Create snapshots with normal gaps (5-10 seconds)
        for i in range(10):
            snapshot_time = now - timedelta(seconds=i * 10)
            filename = snapshot_time.strftime("snapshot_15m_%Y%m%dT%H%M%SZ.json")
            (temp_data_dir / filename).write_text("{}")

        result = analyze_collector_health(temp_data_dir, interval_seconds=5.0, now=now)

        assert result.backoff_evidence is False

    def test_latest_pointer_used(self, temp_data_dir: Path) -> None:
        """latest_15m.json pointer is used if available."""
        temp_data_dir.mkdir()
        now = datetime(2026, 2, 15, 12, 0, 0, tzinfo=UTC)

        # Create a snapshot
        snapshot_time = now - timedelta(minutes=5)
        filename = snapshot_time.strftime("snapshot_15m_%Y%m%dT%H%M%SZ.json")
        snapshot_path = temp_data_dir / filename
        snapshot_path.write_text("{}")

        # Create pointer to that snapshot
        pointer = temp_data_dir / "latest_15m.json"
        pointer.write_text(json.dumps({"path": str(snapshot_path)}))

        result = analyze_collector_health(temp_data_dir, now=now)

        assert result.latest_snapshot_at == snapshot_time.isoformat()

    def test_latest_pointer_invalid_json(self, temp_data_dir: Path) -> None:
        """Invalid JSON in pointer falls back to scanning."""
        temp_data_dir.mkdir()
        now = datetime(2026, 2, 15, 12, 0, 0, tzinfo=UTC)

        # Create a snapshot
        snapshot_time = now - timedelta(minutes=5)
        filename = snapshot_time.strftime("snapshot_15m_%Y%m%dT%H%M%SZ.json")
        (temp_data_dir / filename).write_text("{}")

        # Create invalid pointer
        pointer = temp_data_dir / "latest_15m.json"
        pointer.write_text("invalid json")

        result = analyze_collector_health(temp_data_dir, now=now)

        assert result.latest_snapshot_at == snapshot_time.isoformat()

    def test_latest_pointer_missing_file(self, temp_data_dir: Path) -> None:
        """Pointer to non-existent file falls back to scanning."""
        temp_data_dir.mkdir()
        now = datetime(2026, 2, 15, 12, 0, 0, tzinfo=UTC)

        # Create a snapshot
        snapshot_time = now - timedelta(minutes=5)
        filename = snapshot_time.strftime("snapshot_15m_%Y%m%dT%H%M%SZ.json")
        (temp_data_dir / filename).write_text("{}")

        # Create pointer to non-existent file
        pointer = temp_data_dir / "latest_15m.json"
        pointer.write_text(json.dumps({"path": str(temp_data_dir / "nonexistent.json")}))

        result = analyze_collector_health(temp_data_dir, now=now)

        assert result.latest_snapshot_at == snapshot_time.isoformat()

    def test_custom_interval(self, temp_data_dir: Path) -> None:
        """Custom interval_seconds changes expected snapshots."""
        temp_data_dir.mkdir()
        now = datetime(2026, 2, 15, 12, 0, 0, tzinfo=UTC)

        result = analyze_collector_health(temp_data_dir, interval_seconds=10.0, now=now)

        assert result.expected_snapshots == 360  # 3600 / 10

    def test_capture_rate_capped_at_100(self, temp_data_dir: Path) -> None:
        """Capture rate doesn't exceed 100%."""
        temp_data_dir.mkdir()
        now = datetime(2026, 2, 15, 12, 0, 0, tzinfo=UTC)

        # Create way more snapshots than expected (edge case)
        for i in range(1000):
            snapshot_time = now - timedelta(seconds=i)
            filename = snapshot_time.strftime("snapshot_15m_%Y%m%dT%H%M%SZ.json")
            (temp_data_dir / filename).write_text("{}")

        result = analyze_collector_health(temp_data_dir, interval_seconds=5.0, now=now)

        assert result.capture_rate_pct == 100.0


class TestBtcMicrostructure:
    """Tests for analyze_btc_microstructure function."""

    @pytest.fixture
    def btc_snapshot_data(self) -> dict:
        """Create valid BTC snapshot data.

        Bids sorted ascending (worst to best), asks sorted descending (worst to best).
        """
        return {
            "markets": [
                {
                    "title": "Bitcoin to exceed $100k",
                    "question": "Will Bitcoin exceed $100,000?",
                    "books": {
                        "yes": {
                            "bids": [
                                # Sorted ascending: worst bid first, best bid last
                                {"price": "0.44", "size": "200"},
                                {"price": "0.45", "size": "100"},
                            ],
                            "asks": [
                                # Sorted descending: worst ask first, best ask last
                                {"price": "0.48", "size": "300"},
                                {"price": "0.47", "size": "150"},
                            ],
                        }
                    },
                }
            ]
        }

    def test_valid_btc_snapshot(self, tmp_path: Path, btc_snapshot_data: dict) -> None:
        """Parse valid BTC snapshot correctly."""
        snapshot_path = tmp_path / "snapshot.json"
        snapshot_path.write_text(json.dumps(btc_snapshot_data))

        result = analyze_btc_microstructure(snapshot_path)

        assert result.market == "BTC"
        assert result.best_bid == 0.45
        assert result.best_ask == 0.47
        assert result.spread == pytest.approx(0.02)
        assert result.spread_bps == pytest.approx(434.78, rel=0.01)  # (0.02/0.46)*10000
        assert result.best_bid_depth == 300.0  # Sum of both bids: 200+100
        assert result.best_ask_depth == 450.0  # Sum of both asks: 300+150
        assert result.depth_imbalance == pytest.approx(-0.2)  # (300-450)/(300+450)

    def test_btc_in_question_field(self, tmp_path: Path) -> None:
        """Find BTC market via question field (bitcoin match)."""
        data = {
            "markets": [
                {
                    "title": "Some other market",
                    "question": "Will Bitcoin price go up?",
                    "books": {
                        "yes": {
                            "bids": [{"price": "0.5", "size": "100"}],
                            "asks": [{"price": "0.52", "size": "100"}],
                        }
                    },
                }
            ]
        }
        snapshot_path = tmp_path / "snapshot.json"
        snapshot_path.write_text(json.dumps(data))

        result = analyze_btc_microstructure(snapshot_path)

        assert result.best_bid == 0.5

    def test_no_btc_market(self, tmp_path: Path) -> None:
        """Return defaults when no BTC market found."""
        data = {"markets": [{"title": "Ethereum market", "books": {}}]}
        snapshot_path = tmp_path / "snapshot.json"
        snapshot_path.write_text(json.dumps(data))

        result = analyze_btc_microstructure(snapshot_path)

        assert result.best_bid is None
        assert result.best_ask is None
        assert result.spread is None
        assert result.best_bid_depth == 0.0

    def test_empty_books(self, tmp_path: Path) -> None:
        """Handle empty order books."""
        data = {
            "markets": [
                {
                    "title": "Bitcoin market",
                    "books": {"yes": {"bids": [], "asks": []}},
                }
            ]
        }
        snapshot_path = tmp_path / "snapshot.json"
        snapshot_path.write_text(json.dumps(data))

        result = analyze_btc_microstructure(snapshot_path)

        assert result.best_bid is None
        assert result.best_ask is None
        assert result.spread is None
        assert result.best_bid_depth == 0.0
        assert result.best_ask_depth == 0.0

    def test_missing_books_key(self, tmp_path: Path) -> None:
        """Handle missing books key."""
        data = {"markets": [{"title": "Bitcoin market"}]}
        snapshot_path = tmp_path / "snapshot.json"
        snapshot_path.write_text(json.dumps(data))

        result = analyze_btc_microstructure(snapshot_path)

        assert result.best_bid is None

    def test_missing_yes_book(self, tmp_path: Path) -> None:
        """Handle missing yes book."""
        data = {
            "markets": [
                {"title": "Bitcoin market", "books": {"no": {"bids": [], "asks": []}}}
            ]
        }
        snapshot_path = tmp_path / "snapshot.json"
        snapshot_path.write_text(json.dumps(data))

        result = analyze_btc_microstructure(snapshot_path)

        assert result.best_bid is None

    def test_malformed_json(self, tmp_path: Path) -> None:
        """Return defaults on malformed JSON."""
        snapshot_path = tmp_path / "snapshot.json"
        snapshot_path.write_text("not valid json")

        result = analyze_btc_microstructure(snapshot_path)

        assert result.best_bid is None
        assert result.market == "BTC"

    def test_file_not_exists(self, tmp_path: Path) -> None:
        """Return defaults when file doesn't exist."""
        result = analyze_btc_microstructure(tmp_path / "nonexistent.json")

        assert result.best_bid is None
        assert result.market == "BTC"

    def test_depth_aggregation_top_5(self, tmp_path: Path) -> None:
        """Aggregate depth from top 5 levels.

        Bids sorted ascending (worst to best), asks sorted descending (worst to best).
        Last 5 entries are the best 5.
        """
        data = {
            "markets": [
                {
                    "title": "Bitcoin",
                    "books": {
                        "yes": {
                            "bids": [
                                # Sorted ascending: worst to best
                                {"price": "0.45", "size": "999"},  # Worst, excluded from top 5
                                {"price": "0.46", "size": "50"},   # Included
                                {"price": "0.47", "size": "40"},   # Included
                                {"price": "0.48", "size": "30"},   # Included
                                {"price": "0.49", "size": "20"},   # Included
                                {"price": "0.50", "size": "10"},   # Best, included
                            ],
                            "asks": [
                                # Sorted descending: worst to best
                                {"price": "0.52", "size": "25"},   # Worst
                                {"price": "0.51", "size": "15"},   # Best
                            ],
                        }
                    },
                }
            ]
        }
        snapshot_path = tmp_path / "snapshot.json"
        snapshot_path.write_text(json.dumps(data))

        result = analyze_btc_microstructure(snapshot_path)

        # Best bid is 0.50, depth sums last 5: 50+40+30+20+10 = 150
        assert result.best_bid_depth == 150.0
        # Best ask is 0.51, depth sums last 5: 25+15 = 40
        assert result.best_ask_depth == 40.0

    def test_zero_ask_price_no_spread(self, tmp_path: Path) -> None:
        """No spread when ask <= bid."""
        data = {
            "markets": [
                {
                    "title": "Bitcoin",
                    "books": {
                        "yes": {
                            "bids": [{"price": "0.50", "size": "100"}],
                            "asks": [{"price": "0.50", "size": "100"}],  # Same as bid
                        }
                    },
                }
            ]
        }
        snapshot_path = tmp_path / "snapshot.json"
        snapshot_path.write_text(json.dumps(data))

        result = analyze_btc_microstructure(snapshot_path)

        assert result.best_bid == 0.50
        assert result.best_ask == 0.50
        assert result.spread is None  # ask not > bid

    def test_btc_lowercase_in_title(self, tmp_path: Path) -> None:
        """Match 'btc' in lowercase title."""
        data = {
            "markets": [
                {
                    "title": "btc price prediction",
                    "books": {
                        "yes": {
                            "bids": [{"price": "0.60", "size": "100"}],
                            "asks": [{"price": "0.62", "size": "100"}],
                        }
                    },
                }
            ]
        }
        snapshot_path = tmp_path / "snapshot.json"
        snapshot_path.write_text(json.dumps(data))

        result = analyze_btc_microstructure(snapshot_path)

        assert result.best_bid == 0.60


class TestMomentumSignal:
    """Tests for compute_momentum_signal function."""

    @pytest.fixture
    def temp_data_dir(self, tmp_path: Path) -> Path:
        """Create temporary data directory with helper."""
        return tmp_path / "data"

    def _create_snapshot(
        self,
        data_dir: Path,
        timestamp: datetime,
        bid: float,
        ask: float,
    ) -> None:
        """Helper to create a snapshot file."""
        data_dir.mkdir(parents=True, exist_ok=True)
        filename = timestamp.strftime("snapshot_15m_%Y%m%dT%H%M%SZ.json")
        data = {
            "markets": [
                {
                    "title": "Bitcoin Price",
                    "books": {
                        "yes": {
                            "bids": [{"price": str(bid), "size": "100"}],
                            "asks": [{"price": str(ask), "size": "100"}],
                        }
                    },
                }
            ]
        }
        (data_dir / filename).write_text(json.dumps(data))

    def test_insufficient_data(self, temp_data_dir: Path) -> None:
        """Neutral signal with no confidence when insufficient data."""
        temp_data_dir.mkdir()

        result = compute_momentum_signal(temp_data_dir)

        assert result.signal == "neutral"
        assert result.confidence == 0.0
        assert result.mid_price_change_1h is None
        assert "Insufficient data" in result.reasoning

    def test_single_snapshot_only(self, temp_data_dir: Path) -> None:
        """Neutral signal when only one snapshot available."""
        now = datetime.now(UTC)
        self._create_snapshot(temp_data_dir, now, 0.50, 0.52)

        result = compute_momentum_signal(temp_data_dir)

        assert result.signal == "neutral"
        assert result.confidence == 0.0

    def test_long_signal_upward_trend(self, temp_data_dir: Path) -> None:
        """Long signal when price increases > 1%."""
        now = datetime.now(UTC)
        # Price increased from 0.50 to 0.55 (10% increase)
        self._create_snapshot(temp_data_dir, now - timedelta(minutes=30), 0.50, 0.52)
        self._create_snapshot(temp_data_dir, now, 0.54, 0.56)

        result = compute_momentum_signal(temp_data_dir)

        assert result.signal == "long"
        assert result.confidence > 0
        assert result.mid_price_change_1h == pytest.approx(7.55, rel=0.1)

    def test_short_signal_downward_trend(self, temp_data_dir: Path) -> None:
        """Short signal when price decreases > 1%."""
        now = datetime.now(UTC)
        # Price decreased from 0.55 to 0.50 (~9% decrease)
        self._create_snapshot(temp_data_dir, now - timedelta(minutes=30), 0.54, 0.56)
        self._create_snapshot(temp_data_dir, now, 0.49, 0.51)

        result = compute_momentum_signal(temp_data_dir)

        assert result.signal == "short"
        assert result.confidence > 0

    def test_neutral_signal_small_change(self, temp_data_dir: Path) -> None:
        """Neutral signal when price change < 1%."""
        now = datetime.now(UTC)
        # Price changed only 0.5%
        self._create_snapshot(temp_data_dir, now - timedelta(minutes=30), 0.50, 0.52)
        self._create_snapshot(temp_data_dir, now, 0.502, 0.522)

        result = compute_momentum_signal(temp_data_dir)

        assert result.signal == "neutral"
        assert result.confidence == 0.0

    def test_confidence_capped_at_1(self, temp_data_dir: Path) -> None:
        """Confidence capped at 1.0 for large moves."""
        now = datetime.now(UTC)
        # Price increased 20% (way above 5% threshold)
        self._create_snapshot(temp_data_dir, now - timedelta(minutes=30), 0.50, 0.52)
        self._create_snapshot(temp_data_dir, now, 0.60, 0.62)

        result = compute_momentum_signal(temp_data_dir)

        assert result.signal == "long"
        assert result.confidence == 1.0

    def test_lookback_hours_filter(self, temp_data_dir: Path) -> None:
        """Respect lookback_hours parameter."""
        now = datetime.now(UTC)
        # Old snapshot (outside 1h window)
        self._create_snapshot(temp_data_dir, now - timedelta(hours=2), 0.40, 0.42)
        # Recent snapshot
        self._create_snapshot(temp_data_dir, now - timedelta(minutes=30), 0.50, 0.52)
        self._create_snapshot(temp_data_dir, now, 0.55, 0.57)

        result = compute_momentum_signal(temp_data_dir, lookback_hours=1)

        # Should only use recent snapshots, not the 2h old one
        assert result.signal == "long"
        # Change from 0.51 to 0.56 mid
        assert result.mid_price_change_1h == pytest.approx(9.8, rel=0.1)

    def test_malformed_json_ignored(self, temp_data_dir: Path) -> None:
        """Malformed JSON files are ignored gracefully."""
        now = datetime.now(UTC)
        temp_data_dir.mkdir(parents=True, exist_ok=True)

        # Valid snapshot
        self._create_snapshot(temp_data_dir, now - timedelta(minutes=30), 0.50, 0.52)

        # Invalid snapshot
        invalid_filename = now.strftime("snapshot_15m_%Y%m%dT%H%M%SZ.json")
        (temp_data_dir / invalid_filename).write_text("not json")

        result = compute_momentum_signal(temp_data_dir)

        # Should still work with just one valid snapshot (neutral due to insufficient data)
        assert result.signal == "neutral"

    def test_missing_btc_market_ignored(self, temp_data_dir: Path) -> None:
        """Snapshots without BTC market are ignored."""
        now = datetime.now(UTC)
        temp_data_dir.mkdir(parents=True, exist_ok=True)

        # Snapshot with BTC
        self._create_snapshot(temp_data_dir, now - timedelta(minutes=30), 0.50, 0.52)

        # Snapshot without BTC
        no_btc_filename = now.strftime("snapshot_15m_%Y%m%dT%H%M%SZ.json")
        no_btc_data = {"markets": [{"title": "Ethereum market"}]}
        (temp_data_dir / no_btc_filename).write_text(json.dumps(no_btc_data))

        result = compute_momentum_signal(temp_data_dir)

        # Only one valid BTC snapshot
        assert result.signal == "neutral"

    def test_empty_books_skipped(self, temp_data_dir: Path) -> None:
        """Snapshots with empty books are skipped."""
        now = datetime.now(UTC)
        temp_data_dir.mkdir(parents=True, exist_ok=True)

        # Valid snapshot
        self._create_snapshot(temp_data_dir, now - timedelta(minutes=30), 0.50, 0.52)

        # Snapshot with empty books
        empty_filename = now.strftime("snapshot_15m_%Y%m%dT%H%M%SZ.json")
        empty_data = {
            "markets": [
                {
                    "title": "Bitcoin",
                    "books": {"yes": {"bids": [], "asks": []}},
                }
            ]
        }
        (temp_data_dir / empty_filename).write_text(json.dumps(empty_data))

        result = compute_momentum_signal(temp_data_dir)

        assert result.signal == "neutral"

    def test_reasoning_includes_change_pct(self, temp_data_dir: Path) -> None:
        """Reasoning includes percentage change."""
        now = datetime.now(UTC)
        self._create_snapshot(temp_data_dir, now - timedelta(minutes=30), 0.50, 0.52)
        self._create_snapshot(temp_data_dir, now, 0.55, 0.57)

        result = compute_momentum_signal(temp_data_dir)

        assert "BTC up" in result.reasoning
        assert "%" in result.reasoning


class TestGenerateHourlyDigest:
    """Tests for generate_hourly_digest function."""

    @pytest.fixture
    def temp_data_dir(self, tmp_path: Path) -> Path:
        """Create temporary data directory."""
        return tmp_path / "data"

    def _create_btc_snapshot(
        self,
        data_dir: Path,
        timestamp: datetime,
        bid: float,
        ask: float,
    ) -> None:
        """Helper to create a BTC snapshot file."""
        data_dir.mkdir(parents=True, exist_ok=True)
        filename = timestamp.strftime("snapshot_15m_%Y%m%dT%H%M%SZ.json")
        data = {
            "markets": [
                {
                    "title": "Bitcoin Price",
                    "books": {
                        "yes": {
                            "bids": [{"price": str(bid), "size": "100"}],
                            "asks": [{"price": str(ask), "size": "100"}],
                        }
                    },
                }
            ]
        }
        (data_dir / filename).write_text(json.dumps(data))

    def test_end_to_end_empty_directory(self, temp_data_dir: Path) -> None:
        """Generate digest for empty directory."""
        temp_data_dir.mkdir()

        result = generate_hourly_digest(temp_data_dir)

        assert isinstance(result, HourlyDigest)
        assert result.generated_at is not None
        assert result.collector_health.snapshots_last_hour == 0
        assert result.btc_microstructure.best_bid is None
        assert result.paper_strategy.signal == "neutral"

    def test_end_to_end_with_data(self, temp_data_dir: Path) -> None:
        """Generate digest with actual snapshot data."""
        now = datetime.now(UTC)

        # Create multiple snapshots
        for i in range(5):
            ts = now - timedelta(minutes=i * 10)
            self._create_btc_snapshot(temp_data_dir, ts, 0.50 + i * 0.01, 0.52 + i * 0.01)

        result = generate_hourly_digest(temp_data_dir)

        assert isinstance(result, HourlyDigest)
        assert result.collector_health.snapshots_last_hour == 5
        assert result.btc_microstructure.best_bid is not None
        assert result.btc_microstructure.market == "BTC"

    def test_uses_latest_pointer_when_available(self, temp_data_dir: Path) -> None:
        """Uses latest_15m.json pointer for microstructure."""
        now = datetime.now(UTC)

        # Create two snapshots
        old_time = now - timedelta(minutes=30)
        new_time = now - timedelta(minutes=5)

        self._create_btc_snapshot(temp_data_dir, old_time, 0.45, 0.47)
        self._create_btc_snapshot(temp_data_dir, new_time, 0.55, 0.57)

        # Create pointer to newer snapshot
        new_filename = new_time.strftime("snapshot_15m_%Y%m%dT%H%M%SZ.json")
        new_path = temp_data_dir / new_filename
        pointer = temp_data_dir / "latest_15m.json"
        pointer.write_text(json.dumps({"path": str(new_path)}))

        result = generate_hourly_digest(temp_data_dir)

        # Should use the newer snapshot via pointer
        assert result.btc_microstructure.best_bid == 0.55

    def test_digest_to_dict(self, temp_data_dir: Path) -> None:
        """HourlyDigest can be converted to dict."""
        temp_data_dir.mkdir()

        result = generate_hourly_digest(temp_data_dir)
        result_dict = result.to_dict()

        assert "generated_at" in result_dict
        assert "collector_health" in result_dict
        assert "btc_microstructure" in result_dict
        assert "paper_strategy" in result_dict

    def test_custom_interval(self, temp_data_dir: Path) -> None:
        """Custom interval_seconds changes expected snapshot count."""
        temp_data_dir.mkdir()

        result = generate_hourly_digest(temp_data_dir, interval_seconds=10.0)

        assert result.collector_health.expected_snapshots == 360


class TestDataclasses:
    """Tests for dataclass behavior."""

    def test_collector_health_immutable(self) -> None:
        """CollectorHealth is frozen/immutable."""
        health = CollectorHealth(
            latest_snapshot_at="2026-02-15T12:00:00+00:00",
            freshness_seconds=0.0,
            snapshots_last_hour=100,
            expected_snapshots=720,
            capture_rate_pct=13.9,
            backoff_evidence=False,
            snapshot_dir="/tmp",
        )

        with pytest.raises(AttributeError):
            health.snapshots_last_hour = 200  # type: ignore[misc]

    def test_microstructure_stats_immutable(self) -> None:
        """MicrostructureStats is frozen/immutable."""
        stats = MicrostructureStats(
            market="BTC",
            best_bid=0.5,
            best_ask=0.52,
            spread=0.02,
            spread_bps=40.0,
            best_bid_depth=100.0,
            best_ask_depth=100.0,
            depth_imbalance=0.0,
        )

        with pytest.raises(AttributeError):
            stats.best_bid = 0.6  # type: ignore[misc]

    def test_momentum_signal_immutable(self) -> None:
        """MomentumSignal is frozen/immutable."""
        signal = MomentumSignal(
            signal="long",
            confidence=0.8,
            mid_price_change_1h=5.0,
            volume_surge=None,
            reasoning="Test",
        )

        with pytest.raises(AttributeError):
            signal.signal = "short"  # type: ignore[misc]

    def test_hourly_digest_immutable(self) -> None:
        """HourlyDigest is frozen/immutable."""
        health = CollectorHealth(
            latest_snapshot_at=None,
            freshness_seconds=None,
            snapshots_last_hour=0,
            expected_snapshots=720,
            capture_rate_pct=0.0,
            backoff_evidence=False,
            snapshot_dir="/tmp",
        )
        stats = MicrostructureStats(
            market="BTC",
            best_bid=None,
            best_ask=None,
            spread=None,
            spread_bps=None,
            best_bid_depth=0.0,
            best_ask_depth=0.0,
            depth_imbalance=None,
        )
        signal = MomentumSignal(
            signal="neutral",
            confidence=0.0,
            mid_price_change_1h=None,
            volume_surge=None,
            reasoning="Test",
        )

        digest = HourlyDigest(
            generated_at="2026-02-15T12:00:00+00:00",
            collector_health=health,
            btc_microstructure=stats,
            paper_strategy=signal,
        )

        with pytest.raises(AttributeError):
            digest.generated_at = "other"  # type: ignore[misc]
