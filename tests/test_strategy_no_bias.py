"""Tests for NO bias exploit strategy module."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from polymarket.strategy_no_bias import (
    BASE_RATE_DATABASE,
    MarketVertical,
    NoBiasPosition,
    NoBiasScanResult,
    NoBiasSignal,
    NoBiasTracker,
    calculate_edge,
    calculate_position_size,
    classify_vertical,
    generate_no_bias_signal,
    get_no_bias_performance,
    match_market_to_base_rate,
    scan_markets_for_no_bias,
)


class TestMatchMarketToBaseRate:
    """Test base rate matching."""

    def test_match_kanye_presidential_run(self) -> None:
        """Test matching Kanye presidential run markets."""
        question = "Will Kanye West run for president in 2028?"
        result = match_market_to_base_rate(question)
        assert result is not None
        assert result.vertical == MarketVertical.POLITICS
        assert result.base_rate == 0.02

    def test_match_trump_resign(self) -> None:
        """Test matching Trump resignation markets."""
        question = "Will Trump resign before the election?"
        result = match_market_to_base_rate(question)
        assert result is not None
        assert result.vertical == MarketVertical.POLITICS
        assert result.base_rate == 0.05

    def test_match_mars_colony(self) -> None:
        """Test matching Mars colony markets."""
        question = "Will humans establish a Mars colony by 2030?"
        result = match_market_to_base_rate(question)
        assert result is not None
        assert result.vertical == MarketVertical.SPACE
        assert result.base_rate == 0.01

    def test_match_alien_contact(self) -> None:
        """Test matching alien contact markets."""
        question = "Will we make contact with aliens in 2025?"
        result = match_market_to_base_rate(question)
        assert result is not None
        assert result.vertical == MarketVertical.SPACE
        assert result.base_rate == 0.001

    def test_match_gta_delay(self) -> None:
        """Test matching GTA delay markets."""
        question = "Will GTA 6 be delayed past 2025?"
        result = match_market_to_base_rate(question)
        assert result is not None
        assert result.vertical == MarketVertical.POP_CULTURE
        assert result.base_rate == 0.30

    def test_no_match(self) -> None:
        """Test question that doesn't match any pattern."""
        question = "Will Bitcoin price be above $100K by end of year?"
        result = match_market_to_base_rate(question)
        assert result is None


class TestClassifyVertical:
    """Test market vertical classification."""

    def test_classify_politics(self) -> None:
        """Test politics classification."""
        question = "Will Trump win the election?"
        assert classify_vertical(question) == MarketVertical.POLITICS

    def test_classify_space(self) -> None:
        """Test space classification."""
        question = "Will SpaceX reach Mars this year?"
        assert classify_vertical(question) == MarketVertical.SPACE

    def test_classify_tech(self) -> None:
        """Test tech classification."""
        question = "Will Bitcoin ETF be approved?"
        assert classify_vertical(question) == MarketVertical.TECH

    def test_classify_pop_culture(self) -> None:
        """Test pop culture classification."""
        question = "Will Taylor Swift win Album of the Year?"
        assert classify_vertical(question) == MarketVertical.POP_CULTURE

    def test_classify_unknown(self) -> None:
        """Test unknown classification."""
        question = "Will something happen?"
        assert classify_vertical(question) == MarketVertical.UNKNOWN


class TestCalculateEdge:
    """Test edge calculation."""

    def test_basic_edge_calculation(self) -> None:
        """Test basic edge calculation."""
        # YES at 15%, base rate 5% -> NO at 85%, fair NO at 95%
        # Edge = (0.95 - 0.85) / 0.85 = ~11.8%
        edge = calculate_edge(yes_ask=0.15, base_rate=0.05)
        assert edge > 0.10
        assert edge < 0.13

    def test_no_edge_when_fairly_priced(self) -> None:
        """Test zero edge when fairly priced."""
        edge = calculate_edge(yes_ask=0.10, base_rate=0.10)
        assert abs(edge) < 0.001

    def test_negative_edge_when_underpriced(self) -> None:
        """Test negative edge when YES underpriced."""
        # YES at 5%, base rate 10% -> overpaying for NO
        edge = calculate_edge(yes_ask=0.05, base_rate=0.10)
        assert edge < 0

    def test_zero_edge_when_no_market_invalid(self) -> None:
        """Test zero edge when NO market price invalid."""
        edge = calculate_edge(yes_ask=1.0, base_rate=0.5)
        assert edge == 0.0


class TestGenerateNoBiasSignal:
    """Test signal generation."""

    def test_generates_signal_for_mispriced_market(self) -> None:
        """Test signal generation for mispriced market."""
        market = {
            "market_id": "test-123",
            "question": "Will Kanye West run for president in 2028?",
            "books": {
                "yes": {
                    "bids": [{"price": "0.08", "size": "100"}],
                    "asks": [{"price": "0.10", "size": "100"}],
                }
            },
            "volume": 50000,
            "clob_token_ids": ["yes-token", "no-token"],
        }

        signal = generate_no_bias_signal(
            market,
            min_mispricing_ratio=3.0,
            min_volume_usd=10000,
            max_yes_price=0.30,
        )

        assert signal is not None
        assert signal.market_id == "test-123"
        assert signal.yes_ask == 0.10
        assert signal.base_rate == 0.02  # Kanye pattern
        assert signal.mispricing_ratio == 5.0  # 0.10 / 0.02
        assert signal.edge > 0

    def test_no_signal_when_not_mispriced(self) -> None:
        """Test no signal when not mispriced."""
        market = {
            "market_id": "test-123",
            "question": "Will Kanye West run for president in 2028?",
            "books": {
                "yes": {
                    "bids": [{"price": "0.02", "size": "100"}],
                    "asks": [{"price": "0.025", "size": "100"}],
                }
            },
            "volume": 50000,
            "clob_token_ids": ["yes-token", "no-token"],
        }

        signal = generate_no_bias_signal(
            market,
            min_mispricing_ratio=3.0,
        )

        # 0.025 / 0.02 = 1.25x, below 3x threshold
        assert signal is None

    def test_no_signal_when_no_base_rate_match(self) -> None:
        """Test no signal when no base rate match."""
        market = {
            "market_id": "test-123",
            "question": "Will Bitcoin reach $100K?",
            "books": {
                "yes": {
                    "bids": [{"price": "0.50", "size": "100"}],
                    "asks": [{"price": "0.55", "size": "100"}],
                }
            },
            "volume": 50000,
            "clob_token_ids": ["yes-token", "no-token"],
        }

        signal = generate_no_bias_signal(market)
        assert signal is None

    def test_no_signal_when_yes_price_too_high(self) -> None:
        """Test no signal when YES price above threshold."""
        market = {
            "market_id": "test-123",
            "question": "Will Kanye West run for president?",
            "books": {
                "yes": {
                    "bids": [{"price": "0.40", "size": "100"}],
                    "asks": [{"price": "0.45", "size": "100"}],
                }
            },
            "volume": 50000,
            "clob_token_ids": ["yes-token", "no-token"],
        }

        signal = generate_no_bias_signal(
            market,
            max_yes_price=0.30,
        )

        assert signal is None

    def test_no_signal_when_volume_too_low(self) -> None:
        """Test no signal when volume below threshold."""
        market = {
            "market_id": "test-123",
            "question": "Will Kanye West run for president?",
            "books": {
                "yes": {
                    "bids": [{"price": "0.08", "size": "100"}],
                    "asks": [{"price": "0.10", "size": "100"}],
                }
            },
            "volume": 1000,  # Below 10K threshold
            "clob_token_ids": ["yes-token", "no-token"],
        }

        signal = generate_no_bias_signal(
            market,
            min_volume_usd=10000,
        )

        assert signal is None


class TestScanMarketsForNoBias:
    """Test market scanning."""

    def test_scan_finds_signals(self) -> None:
        """Test scanning finds signals."""
        markets = [
            {
                "market_id": "kanye-123",
                "question": "Will Kanye West run for president in 2028?",
                "books": {
                    "yes": {
                        "bids": [{"price": "0.08", "size": "100"}],
                        "asks": [{"price": "0.10", "size": "100"}],
                    }
                },
                "volume": 50000,
                "clob_token_ids": ["yes-token", "no-token"],
            },
            {
                "market_id": "alien-456",
                "question": "Will we make alien contact this year?",
                "books": {
                    "yes": {
                        "bids": [{"price": "0.005", "size": "100"}],
                        "asks": [{"price": "0.008", "size": "100"}],
                    }
                },
                "volume": 20000,
                "clob_token_ids": ["yes-token", "no-token"],
            },
        ]

        result = scan_markets_for_no_bias(markets)

        assert result.markets_analyzed == 2
        assert result.signals_generated > 0
        # Kanye: 0.10 / 0.02 = 5x mispricing, should generate signal
        # Alien: 0.008 / 0.001 = 8x mispricing, should generate signal

    def test_scan_sorts_by_quality(self) -> None:
        """Test that signals are sorted by edge * confidence."""
        markets = [
            {
                "market_id": "alien-123",
                "question": "Will we make alien contact this year?",
                "books": {
                    "yes": {
                        "bids": [{"price": "0.005", "size": "100"}],
                        "asks": [{"price": "0.008", "size": "100"}],
                    }
                },
                "volume": 20000,
                "clob_token_ids": ["yes-token", "no-token"],
            },
            {
                "market_id": "kanye-456",
                "question": "Will Kanye West run for president?",
                "books": {
                    "yes": {
                        "bids": [{"price": "0.08", "size": "100"}],
                        "asks": [{"price": "0.10", "size": "100"}],
                    }
                },
                "volume": 100000,  # Higher volume = higher confidence
                "clob_token_ids": ["yes-token", "no-token"],
            },
        ]

        result = scan_markets_for_no_bias(markets)

        if len(result.signals) >= 2:
            # Higher volume signal should rank higher
            assert result.signals[0].volume_usd >= result.signals[1].volume_usd


class TestCalculatePositionSize:
    """Test position sizing."""

    def test_base_position_size(self) -> None:
        """Test base position sizing."""
        signal = NoBiasSignal(
            timestamp=datetime.now(UTC),
            market_id="test",
            token_id_yes="yes",
            token_id_no="no",
            market_question="Test?",
            vertical=MarketVertical.POLITICS,
            yes_ask=0.10,
            no_bid=0.90,
            base_rate=0.02,
            mispricing_ratio=5.0,
            edge=0.10,
            confidence=0.7,
            volume_usd=50000,
            time_to_resolution=None,
            reasoning="Test",
        )

        size = calculate_position_size(signal, bankroll=10000, base_pct=2.0, max_pct=5.0)

        # Should be between base and max
        assert size >= 10000 * 0.02 * 0.5  # Some minimum
        assert size <= 10000 * 0.05  # Max cap

    def test_position_size_scales_with_confidence(self) -> None:
        """Test that position size scales with confidence."""
        base_signal = NoBiasSignal(
            timestamp=datetime.now(UTC),
            market_id="test",
            token_id_yes="yes",
            token_id_no="no",
            market_question="Test?",
            vertical=MarketVertical.POLITICS,
            yes_ask=0.10,
            no_bid=0.90,
            base_rate=0.02,
            mispricing_ratio=5.0,
            edge=0.10,
            confidence=0.5,
            volume_usd=50000,
            time_to_resolution=None,
            reasoning="Test",
        )

        high_conf_signal = NoBiasSignal(
            timestamp=datetime.now(UTC),
            market_id="test",
            token_id_yes="yes",
            token_id_no="no",
            market_question="Test?",
            vertical=MarketVertical.POLITICS,
            yes_ask=0.10,
            no_bid=0.90,
            base_rate=0.02,
            mispricing_ratio=5.0,
            edge=0.10,
            confidence=0.9,
            volume_usd=50000,
            time_to_resolution=None,
            reasoning="Test",
        )

        low_size = calculate_position_size(base_signal, bankroll=10000)
        high_size = calculate_position_size(high_conf_signal, bankroll=10000)

        assert high_size > low_size


class TestNoBiasPosition:
    """Test NoBiasPosition dataclass."""

    def test_position_creation(self) -> None:
        """Test creating a position."""
        position = NoBiasPosition(
            position_id="test-123",
            timestamp=datetime.now(UTC),
            market_id="market-456",
            token_id="no-token",
            market_question="Will Kanye run?",
            vertical=MarketVertical.POLITICS,
            entry_no_price=0.90,
            position_size_usd=500,
            expected_edge=0.10,
        )

        assert position.is_open
        assert position.position_id == "test-123"

    def test_position_close(self) -> None:
        """Test closing a position."""
        position = NoBiasPosition(
            position_id="test-123",
            timestamp=datetime.now(UTC),
            market_id="market-456",
            token_id="no-token",
            market_question="Will Kanye run?",
            vertical=MarketVertical.POLITICS,
            entry_no_price=0.90,
            position_size_usd=500,
            expected_edge=0.10,
        )

        position.close(exit_price=0.95, reason="edge_gone")

        assert not position.is_open
        assert position.exit_price == 0.95
        assert position.exit_reason == "edge_gone"
        assert position.pnl is not None
        # Bought at 0.90, sold at 0.95, 500/0.90 = 555.56 contracts
        # PnL = (0.95 - 0.90) * 555.56 = ~27.78
        assert position.pnl > 0

    def test_position_settle_no_wins(self) -> None:
        """Test settling when NO wins."""
        position = NoBiasPosition(
            position_id="test-123",
            timestamp=datetime.now(UTC),
            market_id="market-456",
            token_id="no-token",
            market_question="Will Kanye run?",
            vertical=MarketVertical.POLITICS,
            entry_no_price=0.90,
            position_size_usd=500,
            expected_edge=0.10,
        )

        position.settle(no_wins=True)

        assert not position.is_open
        assert position.settled
        assert position.exit_price == 1.0
        assert position.pnl is not None
        assert position.pnl > 0

    def test_position_settle_yes_wins(self) -> None:
        """Test settling when YES wins."""
        position = NoBiasPosition(
            position_id="test-123",
            timestamp=datetime.now(UTC),
            market_id="market-456",
            token_id="no-token",
            market_question="Will Kanye run?",
            vertical=MarketVertical.POLITICS,
            entry_no_price=0.90,
            position_size_usd=500,
            expected_edge=0.10,
        )

        position.settle(no_wins=False)

        assert not position.is_open
        assert position.settled
        assert position.exit_price == 0.0
        assert position.pnl is not None
        assert position.pnl < 0

    def test_position_to_dict(self) -> None:
        """Test serializing open position to dict."""
        position = NoBiasPosition(
            position_id="test-123",
            timestamp=datetime(2026, 2, 15, 12, 0, 0, tzinfo=UTC),
            market_id="market-456",
            token_id="no-token",
            market_question="Will Kanye run?",
            vertical=MarketVertical.POLITICS,
            entry_no_price=0.90,
            position_size_usd=500,
            expected_edge=0.10,
        )

        data = position.to_dict()

        assert data["position_id"] == "test-123"
        assert data["market_id"] == "market-456"
        assert data["token_id"] == "no-token"
        assert data["market_question"] == "Will Kanye run?"
        assert data["vertical"] == "politics"
        assert data["entry_no_price"] == 0.90
        assert data["position_size_usd"] == 500
        assert data["expected_edge"] == 0.10
        assert data["exit_price"] is None
        assert data["exit_timestamp"] is None
        assert data["exit_reason"] is None
        assert data["pnl"] is None
        assert data["settled"] is False
        assert data["timestamp"] == "2026-02-15T12:00:00+00:00"

    def test_position_to_dict_closed(self) -> None:
        """Test serializing closed position to dict."""
        position = NoBiasPosition(
            position_id="test-123",
            timestamp=datetime(2026, 2, 15, 12, 0, 0, tzinfo=UTC),
            market_id="market-456",
            token_id="no-token",
            market_question="Will Kanye run?",
            vertical=MarketVertical.POLITICS,
            entry_no_price=0.90,
            position_size_usd=500,
            expected_edge=0.10,
        )
        position.close(exit_price=0.95, reason="edge_gone", timestamp=datetime(2026, 2, 16, 12, 0, 0, tzinfo=UTC))

        data = position.to_dict()

        assert data["exit_price"] == 0.95
        assert data["exit_timestamp"] == "2026-02-16T12:00:00+00:00"
        assert data["exit_reason"] == "edge_gone"
        assert data["pnl"] is not None
        assert data["pnl"] > 0

    def test_position_from_dict_open(self) -> None:
        """Test deserializing open position from dict."""
        data = {
            "position_id": "test-123",
            "timestamp": "2026-02-15T12:00:00+00:00",
            "market_id": "market-456",
            "token_id": "no-token",
            "market_question": "Will Kanye run?",
            "vertical": "politics",
            "entry_no_price": 0.90,
            "position_size_usd": 500,
            "expected_edge": 0.10,
            "exit_price": None,
            "exit_timestamp": None,
            "exit_reason": None,
            "pnl": None,
            "settled": False,
        }

        position = NoBiasPosition.from_dict(data)

        assert position.position_id == "test-123"
        assert position.market_id == "market-456"
        assert position.token_id == "no-token"
        assert position.market_question == "Will Kanye run?"
        assert position.vertical == MarketVertical.POLITICS
        assert position.entry_no_price == 0.90
        assert position.position_size_usd == 500
        assert position.expected_edge == 0.10
        assert position.is_open
        assert position.exit_price is None
        assert position.exit_timestamp is None
        assert position.exit_reason is None
        assert position.pnl is None
        assert not position.settled
        assert position.timestamp == datetime(2026, 2, 15, 12, 0, 0, tzinfo=UTC)

    def test_position_from_dict_closed(self) -> None:
        """Test deserializing closed position from dict."""
        data = {
            "position_id": "test-123",
            "timestamp": "2026-02-15T12:00:00+00:00",
            "market_id": "market-456",
            "token_id": "no-token",
            "market_question": "Will Kanye run?",
            "vertical": "politics",
            "entry_no_price": 0.90,
            "position_size_usd": 500,
            "expected_edge": 0.10,
            "exit_price": 0.95,
            "exit_timestamp": "2026-02-16T12:00:00+00:00",
            "exit_reason": "edge_gone",
            "pnl": 27.78,
            "settled": False,
        }

        position = NoBiasPosition.from_dict(data)

        assert not position.is_open
        assert position.exit_price == 0.95
        assert position.exit_timestamp == datetime(2026, 2, 16, 12, 0, 0, tzinfo=UTC)
        assert position.exit_reason == "edge_gone"
        assert position.pnl == 27.78
        assert not position.settled

    def test_position_from_dict_settled(self) -> None:
        """Test deserializing settled position from dict."""
        data = {
            "position_id": "test-123",
            "timestamp": "2026-02-15T12:00:00+00:00",
            "market_id": "market-456",
            "token_id": "no-token",
            "market_question": "Will Kanye run?",
            "vertical": "politics",
            "entry_no_price": 0.90,
            "position_size_usd": 500,
            "expected_edge": 0.10,
            "exit_price": 1.0,
            "exit_timestamp": "2026-02-20T12:00:00+00:00",
            "exit_reason": "settlement_no_wins",
            "pnl": 55.56,
            "settled": True,
        }

        position = NoBiasPosition.from_dict(data)

        assert not position.is_open
        assert position.settled
        assert position.exit_price == 1.0
        assert position.exit_reason == "settlement_no_wins"
        assert position.pnl == 55.56

    def test_position_roundtrip(self) -> None:
        """Test full serialization roundtrip."""
        original = NoBiasPosition(
            position_id="test-123",
            timestamp=datetime(2026, 2, 15, 12, 0, 0, tzinfo=UTC),
            market_id="market-456",
            token_id="no-token",
            market_question="Will Kanye run?",
            vertical=MarketVertical.POLITICS,
            entry_no_price=0.90,
            position_size_usd=500,
            expected_edge=0.10,
        )
        original.close(exit_price=0.95, reason="test_exit", timestamp=datetime(2026, 2, 16, 12, 0, 0, tzinfo=UTC))

        # Serialize and deserialize
        data = original.to_dict()
        restored = NoBiasPosition.from_dict(data)

        # Verify all fields match
        assert restored.position_id == original.position_id
        assert restored.timestamp == original.timestamp
        assert restored.market_id == original.market_id
        assert restored.token_id == original.token_id
        assert restored.market_question == original.market_question
        assert restored.vertical == original.vertical
        assert restored.entry_no_price == original.entry_no_price
        assert restored.position_size_usd == original.position_size_usd
        assert restored.expected_edge == original.expected_edge
        assert restored.exit_price == original.exit_price
        assert restored.exit_timestamp == original.exit_timestamp
        assert restored.exit_reason == original.exit_reason
        assert restored.pnl == original.pnl
        assert restored.settled == original.settled


class TestNoBiasTracker:
    """Test NoBiasTracker."""

    def test_tracker_can_open_position(self, tmp_path: Path) -> None:
        """Test tracker can open position."""
        tracker = NoBiasTracker(data_dir=tmp_path, max_positions_per_vertical=2)

        signal = NoBiasSignal(
            timestamp=datetime.now(UTC),
            market_id="test",
            token_id_yes="yes",
            token_id_no="no",
            market_question="Will Kanye run?",
            vertical=MarketVertical.POLITICS,
            yes_ask=0.10,
            no_bid=0.90,
            base_rate=0.02,
            mispricing_ratio=5.0,
            edge=0.10,
            confidence=0.7,
            volume_usd=50000,
            time_to_resolution=None,
            reasoning="Test",
        )

        position = tracker.open_position(signal, bankroll=10000, dry_run=True)

        assert position is not None
        assert position.market_id == "test"
        assert len(tracker.get_open_positions()) == 1

    def test_tracker_respects_vertical_limits(self, tmp_path: Path) -> None:
        """Test tracker respects position limits per vertical."""
        tracker = NoBiasTracker(data_dir=tmp_path, max_positions_per_vertical=1)

        signal = NoBiasSignal(
            timestamp=datetime.now(UTC),
            market_id="test1",
            token_id_yes="yes",
            token_id_no="no",
            market_question="Will Kanye run?",
            vertical=MarketVertical.POLITICS,
            yes_ask=0.10,
            no_bid=0.90,
            base_rate=0.02,
            mispricing_ratio=5.0,
            edge=0.10,
            confidence=0.7,
            volume_usd=50000,
            time_to_resolution=None,
            reasoning="Test",
        )

        tracker.open_position(signal, bankroll=10000, dry_run=True)

        # Second position in same vertical should fail
        signal2 = NoBiasSignal(
            timestamp=datetime.now(UTC),
            market_id="test2",
            token_id_yes="yes",
            token_id_no="no",
            market_question="Will Trump resign?",
            vertical=MarketVertical.POLITICS,
            yes_ask=0.08,
            no_bid=0.92,
            base_rate=0.05,
            mispricing_ratio=3.0,
            edge=0.08,
            confidence=0.6,
            volume_usd=50000,
            time_to_resolution=None,
            reasoning="Test",
        )

        position2 = tracker.open_position(signal2, bankroll=10000, dry_run=True)

        assert position2 is None  # Limit reached

    def test_tracker_performance_summary(self, tmp_path: Path) -> None:
        """Test performance summary."""
        tracker = NoBiasTracker(data_dir=tmp_path)

        # Add some closed positions
        pos1 = NoBiasPosition(
            position_id="win-1",
            timestamp=datetime.now(UTC),
            market_id="m1",
            token_id="no",
            market_question="Q1?",
            vertical=MarketVertical.POLITICS,
            entry_no_price=0.90,
            position_size_usd=500,
            expected_edge=0.10,
        )
        pos1.settle(no_wins=True)
        tracker.positions["win-1"] = pos1

        pos2 = NoBiasPosition(
            position_id="loss-1",
            timestamp=datetime.now(UTC),
            market_id="m2",
            token_id="no",
            market_question="Q2?",
            vertical=MarketVertical.SPACE,
            entry_no_price=0.85,
            position_size_usd=500,
            expected_edge=0.10,
        )
        pos2.settle(no_wins=False)
        tracker.positions["loss-1"] = pos2

        summary = tracker.get_performance_summary()

        assert summary["total_trades"] == 2
        assert summary["win_rate"] == 0.5
        assert "by_vertical" in summary

    def test_tracker_persistence_roundtrip(self, tmp_path: Path) -> None:
        """Test that positions survive save/load roundtrip."""
        data_dir = tmp_path / "no_bias"

        # Create tracker and add positions
        tracker1 = NoBiasTracker(data_dir=data_dir, max_positions_per_vertical=3)

        # Add open position
        signal1 = NoBiasSignal(
            timestamp=datetime(2026, 2, 15, 12, 0, 0, tzinfo=UTC),
            market_id="open-market",
            token_id_yes="yes1",
            token_id_no="no1",
            market_question="Will Kanye run?",
            vertical=MarketVertical.POLITICS,
            yes_ask=0.10,
            no_bid=0.90,
            base_rate=0.02,
            mispricing_ratio=5.0,
            edge=0.10,
            confidence=0.7,
            volume_usd=50000,
            time_to_resolution=None,
            reasoning="Test",
        )
        pos1 = tracker1.open_position(signal1, bankroll=10000, dry_run=True)
        assert pos1 is not None

        # Add closed position
        signal2 = NoBiasSignal(
            timestamp=datetime(2026, 2, 14, 12, 0, 0, tzinfo=UTC),
            market_id="closed-market",
            token_id_yes="yes2",
            token_id_no="no2",
            market_question="Will aliens visit?",
            vertical=MarketVertical.SPACE,
            yes_ask=0.05,
            no_bid=0.95,
            base_rate=0.001,
            mispricing_ratio=50.0,
            edge=0.15,
            confidence=0.8,
            volume_usd=30000,
            time_to_resolution=None,
            reasoning="Test",
        )
        pos2 = tracker1.open_position(signal2, bankroll=10000, dry_run=True)
        assert pos2 is not None
        pos2.close(exit_price=0.98, reason="test_exit", timestamp=datetime(2026, 2, 15, 12, 0, 0, tzinfo=UTC))

        # Force save
        tracker1._save_positions()

        # Verify file was created
        positions_file = data_dir / "positions.json"
        assert positions_file.exists()

        # Create new tracker instance pointing to same directory
        tracker2 = NoBiasTracker(data_dir=data_dir, max_positions_per_vertical=3)

        # Verify positions were restored
        assert len(tracker2.positions) == 2

        # Verify open position
        restored_open = tracker2.positions.get(pos1.position_id)
        assert restored_open is not None
        assert restored_open.is_open
        assert restored_open.market_id == "open-market"
        assert restored_open.vertical == MarketVertical.POLITICS
        assert restored_open.entry_no_price == 0.90
        assert restored_open.position_size_usd == pos1.position_size_usd
        # Timestamp is set to now() when position is created, not signal timestamp
        assert restored_open.timestamp == pos1.timestamp

        # Verify closed position
        restored_closed = tracker2.positions.get(pos2.position_id)
        assert restored_closed is not None
        assert not restored_closed.is_open
        assert restored_closed.market_id == "closed-market"
        assert restored_closed.vertical == MarketVertical.SPACE
        assert restored_closed.exit_price == 0.98
        assert restored_closed.exit_reason == "test_exit"
        assert restored_closed.exit_timestamp == datetime(2026, 2, 15, 12, 0, 0, tzinfo=UTC)
        assert restored_closed.pnl is not None

        # Verify open positions list
        open_positions = tracker2.get_open_positions()
        assert len(open_positions) == 1
        assert open_positions[0].position_id == pos1.position_id

    def test_tracker_persistence_with_settled_position(self, tmp_path: Path) -> None:
        """Test that settled positions are correctly persisted and restored."""
        data_dir = tmp_path / "no_bias"

        # Create tracker with settled position
        tracker1 = NoBiasTracker(data_dir=data_dir, max_positions_per_vertical=3)

        signal = NoBiasSignal(
            timestamp=datetime(2026, 2, 15, 12, 0, 0, tzinfo=UTC),
            market_id="settled-market",
            token_id_yes="yes1",
            token_id_no="no1",
            market_question="Will Kanye run?",
            vertical=MarketVertical.POLITICS,
            yes_ask=0.10,
            no_bid=0.90,
            base_rate=0.02,
            mispricing_ratio=5.0,
            edge=0.10,
            confidence=0.7,
            volume_usd=50000,
            time_to_resolution=None,
            reasoning="Test",
        )
        pos = tracker1.open_position(signal, bankroll=10000, dry_run=True)
        assert pos is not None
        pos.settle(no_wins=True, timestamp=datetime(2026, 2, 20, 12, 0, 0, tzinfo=UTC))

        tracker1._save_positions()

        # Create new tracker and verify
        tracker2 = NoBiasTracker(data_dir=data_dir, max_positions_per_vertical=3)

        restored = tracker2.positions.get(pos.position_id)
        assert restored is not None
        assert not restored.is_open
        assert restored.settled
        assert restored.exit_price == 1.0
        assert restored.exit_reason == "settlement_no_wins"
        assert restored.pnl is not None and restored.pnl > 0


class TestGetNoBiasPerformance:
    """Test get_no_bias_performance function."""

    def test_get_performance(self, tmp_path: Path) -> None:
        """Test getting performance."""
        tracker = NoBiasTracker(data_dir=tmp_path)

        # Add open position
        signal = NoBiasSignal(
            timestamp=datetime.now(UTC),
            market_id="test",
            token_id_yes="yes",
            token_id_no="no",
            market_question="Will Kanye run?",
            vertical=MarketVertical.POLITICS,
            yes_ask=0.10,
            no_bid=0.90,
            base_rate=0.02,
            mispricing_ratio=5.0,
            edge=0.10,
            confidence=0.7,
            volume_usd=50000,
            time_to_resolution=None,
            reasoning="Test",
        )
        tracker.open_position(signal, bankroll=10000, dry_run=True)

        result = get_no_bias_performance(tracker)

        assert result["open_count"] == 1
        assert len(result["open_positions"]) == 1
        assert "summary" in result
        assert "timestamp" in result


class TestNoBiasSignalDict:
    """Test signal serialization."""

    def test_signal_to_dict(self) -> None:
        """Test converting signal to dict."""
        signal = NoBiasSignal(
            timestamp=datetime(2026, 2, 15, 12, 0, 0, tzinfo=UTC),
            market_id="test-123",
            token_id_yes="yes-token",
            token_id_no="no-token",
            market_question="Will Kanye run?",
            vertical=MarketVertical.POLITICS,
            yes_ask=0.10,
            no_bid=0.90,
            base_rate=0.02,
            mispricing_ratio=5.0,
            edge=0.10,
            confidence=0.7,
            volume_usd=50000,
            time_to_resolution=timedelta(days=30),
            reasoning="Test reasoning",
        )

        d = signal.to_dict()

        assert d["market_id"] == "test-123"
        assert d["yes_ask"] == 0.10
        assert d["mispricing_ratio"] == 5.0
        assert d["vertical"] == "politics"
        assert d["time_to_resolution_hours"] == 720.0  # 30 days


class TestNoBiasScanResultDict:
    """Test scan result serialization."""

    def test_result_to_dict(self) -> None:
        """Test converting result to dict."""
        signal = NoBiasSignal(
            timestamp=datetime(2026, 2, 15, 12, 0, 0, tzinfo=UTC),
            market_id="test",
            token_id_yes="yes",
            token_id_no="no",
            market_question="Will Kanye run?",
            vertical=MarketVertical.POLITICS,
            yes_ask=0.10,
            no_bid=0.90,
            base_rate=0.02,
            mispricing_ratio=5.0,
            edge=0.10,
            confidence=0.7,
            volume_usd=50000,
            time_to_resolution=None,
            reasoning="Test",
        )

        result = NoBiasScanResult(
            timestamp=datetime(2026, 2, 15, 12, 0, 0, tzinfo=UTC),
            markets_analyzed=100,
            signals_generated=5,
            actionable_signals=3,
            signals=[signal],
            positions_taken=2,
            total_capital_deployed=1000.0,
        )

        d = result.to_dict()

        assert d["markets_analyzed"] == 100
        assert d["signals_generated"] == 5
        assert len(d["signals"]) == 1


class TestBaseRateDatabase:
    """Test base rate database contents."""

    def test_base_rates_are_reasonable(self) -> None:
        """Test that all base rates are reasonable probabilities."""
        for estimate in BASE_RATE_DATABASE:
            assert 0.0 < estimate.base_rate <= 1.0, f"Invalid base rate for {estimate.pattern}"

    def test_patterns_are_valid_regex(self) -> None:
        """Test that all patterns are valid regex."""
        import re

        for estimate in BASE_RATE_DATABASE:
            try:
                re.compile(estimate.pattern)
            except re.error as e:
                pytest.fail(f"Invalid regex pattern {estimate.pattern}: {e}")

    def test_all_patterns_have_sources(self) -> None:
        """Test that all estimates have sources."""
        for estimate in BASE_RATE_DATABASE:
            assert estimate.source, f"Missing source for {estimate.pattern}"
            assert estimate.reasoning, f"Missing reasoning for {estimate.pattern}"
