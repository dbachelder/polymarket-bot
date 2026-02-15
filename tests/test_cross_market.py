"""Tests for cross-market arbitrage strategy.
"""

from datetime import UTC, datetime

import pytest

from polymarket.cross_market import (
    KALSHI_FEE_SCHEDULE,
    POLYMARKET_FEE_SCHEDULE,
    ArbitrageOpportunity,
    CrossMarketEvent,
    VenueMarket,
)
from polymarket.cross_market.calculator import SpreadCalculator, quick_spread_check
from polymarket.cross_market.matcher import EventMatcher, EventNormalizer
from polymarket.cross_market.tracker import PaperTradeTracker


class TestFeeSchedule:
    def test_default_polymarket_fees(self):
        assert POLYMARKET_FEE_SCHEDULE.taker_fee == 0.0
        assert POLYMARKET_FEE_SCHEDULE.maker_fee == 0.0
        assert POLYMARKET_FEE_SCHEDULE.withdrawal_fee == 0.02

    def test_default_kalshi_fees(self):
        assert KALSHI_FEE_SCHEDULE.taker_fee == 0.0
        assert KALSHI_FEE_SCHEDULE.maker_fee == 0.0
        assert KALSHI_FEE_SCHEDULE.withdrawal_fee == 0.0
        assert KALSHI_FEE_SCHEDULE.max_fee == 0.05


class TestEventNormalizer:
    def test_normalize_title(self):
        normalizer = EventNormalizer()

        # Test basic normalization
        title = "Will Bitcoin exceed $100,000 by end of 2025?"
        normalized = normalizer.normalize_title(title)

        assert "bitcoin" in normalized
        assert "exceed" in normalized
        assert "100000" in normalized or "$" not in normalized

    def test_detect_category(self):
        normalizer = EventNormalizer()

        assert normalizer.detect_category("Will Bitcoin hit 100k?") == "crypto"
        assert normalizer.detect_category("Will Trump win 2024?") == "politics"
        assert normalizer.detect_category("Super Bowl winner?") == "sports"
        assert normalizer.detect_category("Random question?") == "other"

    def test_extract_key_entities(self):
        normalizer = EventNormalizer()

        title = "Will BTC trade above $50,000 by March 2025?"
        entities = normalizer.extract_key_entities(title)

        assert "btc" in entities
        assert "above" in entities or "50000" in entities
        assert "2025" in entities


class TestEventMatcher:
    def test_calculate_similarity_identical(self):
        matcher = EventMatcher()

        title1 = "Will Bitcoin exceed 100k by 2025"
        title2 = "Will Bitcoin exceed 100k by 2025"

        similarity = matcher.calculate_similarity(title1, title2)
        assert similarity == 1.0

    def test_calculate_similarity_different(self):
        matcher = EventMatcher()

        title1 = "Will Bitcoin exceed 100k by 2025"
        title2 = "Will Ethereum reach 10k this year"

        similarity = matcher.calculate_similarity(title1, title2)
        assert 0.0 <= similarity < 0.5  # Low similarity expected


class TestSpreadCalculator:
    def test_calculate_spread_profitable(self):
        """Test spread calculation when arbitrage exists."""
        calculator = SpreadCalculator(min_gross_spread=0.01, min_net_spread=0.005)

        now = datetime.now(UTC)

        # Create markets where YES + NO < 1.0
        pm_market = VenueMarket(
            venue="polymarket",
            market_id="pm-1",
            token_id_yes="yes-1",
            token_id_no="no-1",
            yes_ask=0.45,
            no_ask=0.52,
            fees=POLYMARKET_FEE_SCHEDULE,
            last_updated=now,
        )

        kalshi_market = VenueMarket(
            venue="kalshi",
            market_id="ks-1",
            token_id_yes="ks-yes-1",
            token_id_no="ks-no-1",
            yes_ask=0.46,
            no_ask=0.50,
            fees=KALSHI_FEE_SCHEDULE,
            last_updated=now,
        )

        event = CrossMarketEvent(
            event_id="test-1",
            title="Will BTC hit 100k",
            description="Test event",
            category="crypto",
            resolution_date=None,
            resolution_source="",
            venues={
                "polymarket": pm_market,
                "kalshi": kalshi_market,
            },
        )

        opp = calculator.calculate_spread(event)

        assert opp is not None
        assert opp.gross_spread > 0
        assert opp.net_spread > 0

    def test_calculate_spread_no_arbitrage(self):
        """Test spread calculation when no arbitrage exists."""
        calculator = SpreadCalculator(min_gross_spread=0.01, min_net_spread=0.005)

        now = datetime.now(UTC)

        # Create markets where YES + NO > 1.0 (no arbitrage)
        pm_market = VenueMarket(
            venue="polymarket",
            market_id="pm-1",
            token_id_yes="yes-1",
            token_id_no="no-1",
            yes_ask=0.60,
            no_ask=0.50,
            fees=POLYMARKET_FEE_SCHEDULE,
            last_updated=now,
        )

        kalshi_market = VenueMarket(
            venue="kalshi",
            market_id="ks-1",
            token_id_yes="ks-yes-1",
            token_id_no="ks-no-1",
            yes_ask=0.55,
            no_ask=0.48,
            fees=KALSHI_FEE_SCHEDULE,
            last_updated=now,
        )

        event = CrossMarketEvent(
            event_id="test-1",
            title="Will BTC hit 100k",
            description="Test event",
            category="crypto",
            resolution_date=None,
            resolution_source="",
            venues={
                "polymarket": pm_market,
                "kalshi": kalshi_market,
            },
        )

        opp = calculator.calculate_spread(event)

        # No arbitrage opportunity should be found
        assert opp is None


class TestQuickSpreadCheck:
    def test_profitable_spread(self):
        result = quick_spread_check(0.45, 0.52)

        assert result["sum_cost"] == pytest.approx(0.97)
        assert result["gross_spread"] == pytest.approx(0.03)
        assert result["is_profitable"] is True

    def test_unprofitable_spread(self):
        result = quick_spread_check(0.60, 0.50)

        assert result["sum_cost"] == pytest.approx(1.10)
        assert result["gross_spread"] == pytest.approx(-0.10)
        assert result["is_profitable"] is False


class TestPaperTradeTracker:
    def test_enter_position(self, tmp_path):
        tracker = PaperTradeTracker(data_dir=tmp_path)

        now = datetime.now(UTC)

        opp = ArbitrageOpportunity(
            event=CrossMarketEvent(
                event_id="test-1",
                title="Test Event",
                description="Test",
                category="test",
                resolution_date=None,
                resolution_source="",
                venues={},
            ),
            venue_yes="polymarket",
            venue_no="kalshi",
            yes_price=0.45,
            no_price=0.52,
            gross_spread=0.03,
            net_spread=0.025,
            fees_yes=0.002,
            fees_no=0.001,
            total_fees=0.003,
            confidence=0.9,
            timestamp=now,
        )

        trade = tracker.enter_position(opp, position_size=1.0)

        assert trade.trade_id is not None
        assert trade.status == "open"
        assert trade.position_size == 1.0
        assert len(tracker.get_open_positions()) == 1

    def test_close_position(self, tmp_path):
        tracker = PaperTradeTracker(data_dir=tmp_path)

        now = datetime.now(UTC)

        opp = ArbitrageOpportunity(
            event=CrossMarketEvent(
                event_id="test-1",
                title="Test Event",
                description="Test",
                category="test",
                resolution_date=None,
                resolution_source="",
                venues={},
            ),
            venue_yes="polymarket",
            venue_no="kalshi",
            yes_price=0.45,
            no_price=0.52,
            gross_spread=0.03,
            net_spread=0.025,
            fees_yes=0.002,
            fees_no=0.001,
            total_fees=0.003,
            confidence=0.9,
            timestamp=now,
        )

        trade = tracker.enter_position(opp, position_size=1.0)

        # Close the position
        closed = tracker.close_position(trade.trade_id)

        assert closed is not None
        assert closed.status == "held_to_resolution"
        assert closed.realized_pnl is not None
        assert len(tracker.get_open_positions()) == 0
        assert len(tracker.get_closed_positions()) == 1

    def test_performance_summary(self, tmp_path):
        tracker = PaperTradeTracker(data_dir=tmp_path)

        summary = tracker.get_performance_summary()

        assert summary["total_trades"] == 0
        assert summary["open_positions"] == 0
        assert summary["closed_positions"] == 0
        assert summary["total_realized_pnl"] == 0.0


class TestVenueMarket:
    def test_venue_market_creation(self):
        now = datetime.now(UTC)

        market = VenueMarket(
            venue="polymarket",
            market_id="test-123",
            token_id_yes="yes-123",
            token_id_no="no-123",
            yes_price=0.55,
            no_price=0.45,
            volume_24h=10000.0,
            last_updated=now,
        )

        assert market.venue == "polymarket"
        assert market.yes_price == 0.55
        assert market.no_price == 0.45


class TestCrossMarketEvent:
    def test_event_creation(self):
        now = datetime.now(UTC)

        pm_market = VenueMarket(
            venue="polymarket",
            market_id="pm-1",
            token_id_yes="yes-1",
            token_id_no="no-1",
            last_updated=now,
        )

        kalshi_market = VenueMarket(
            venue="kalshi",
            market_id="ks-1",
            token_id_yes="ks-yes-1",
            token_id_no="ks-no-1",
            last_updated=now,
        )

        event = CrossMarketEvent(
            event_id="test-1",
            title="Will BTC hit 100k",
            description="Test description",
            category="crypto",
            resolution_date=None,
            resolution_source="official",
            venues={
                "polymarket": pm_market,
                "kalshi": kalshi_market,
            },
        )

        assert event.event_id == "test-1"
        assert len(event.venues) == 2
        assert "polymarket" in event.venues
        assert "kalshi" in event.venues
