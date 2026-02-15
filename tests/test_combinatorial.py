"""Tests for combinatorial arbitrage strategy."""

from datetime import UTC, datetime

import pytest

from polymarket.combinatorial import (
    BasketOutcome,
    DutchBookBasket,
    CombinatorialScanResult,
    format_scan_report,
)


class TestBasketOutcome:
    """Test BasketOutcome dataclass."""

    def test_to_dict(self):
        outcome = BasketOutcome(
            market_id="123",
            market_slug="test-market",
            market_question="Will X happen?",
            token_id_yes="token123",
            best_ask_yes=0.35,
            best_bid_yes=0.33,
            liquidity=1000.0,
            outcome_index=0,
        )
        
        d = outcome.to_dict()
        assert d["market_id"] == "123"
        assert d["best_ask_yes"] == 0.35
        assert d["liquidity"] == 1000.0


class TestDutchBookBasket:
    """Test DutchBookBasket calculations."""

    @pytest.fixture
    def sample_outcomes(self):
        """Create sample outcomes for testing."""
        return [
            BasketOutcome(
                market_id="1",
                market_slug="candidate-a",
                market_question="Will A win?",
                token_id_yes="token1",
                best_ask_yes=0.30,
                best_bid_yes=0.28,
                liquidity=500.0,
                outcome_index=0,
            ),
            BasketOutcome(
                market_id="2",
                market_slug="candidate-b",
                market_question="Will B win?",
                token_id_yes="token2",
                best_ask_yes=0.25,
                best_bid_yes=0.23,
                liquidity=600.0,
                outcome_index=1,
            ),
            BasketOutcome(
                market_id="3",
                market_slug="candidate-c",
                market_question="Will C win?",
                token_id_yes="token3",
                best_ask_yes=0.20,
                best_bid_yes=0.18,
                liquidity=700.0,
                outcome_index=2,
            ),
        ]

    @pytest.fixture
    def profitable_basket(self, sample_outcomes):
        """Create a profitable Dutch book basket."""
        return DutchBookBasket(
            basket_id="test_123",
            event_id="event123",
            event_title="Test Election",
            relationship_type="winner_take_all",
            outcomes=sample_outcomes,
            sum_best_ask=0.75,  # Sum < 1.0, should be profitable
            fee_rate=0.0315,
            min_edge_after_fees=0.015,
            timestamp=datetime.now(UTC),
            notes="Test basket",
        )

    @pytest.fixture
    def unprofitable_basket(self, sample_outcomes):
        """Create an unprofitable Dutch book basket."""
        # Modify outcomes to create unprofitable basket
        expensive_outcomes = [
            BasketOutcome(
                market_id="1",
                market_slug="candidate-a",
                market_question="Will A win?",
                token_id_yes="token1",
                best_ask_yes=0.50,
                best_bid_yes=0.48,
                liquidity=500.0,
                outcome_index=0,
            ),
            BasketOutcome(
                market_id="2",
                market_slug="candidate-b",
                market_question="Will B win?",
                token_id_yes="token2",
                best_ask_yes=0.40,
                best_bid_yes=0.38,
                liquidity=600.0,
                outcome_index=1,
            ),
            BasketOutcome(
                market_id="3",
                market_slug="candidate-c",
                market_question="Will C win?",
                token_id_yes="token3",
                best_ask_yes=0.35,
                best_bid_yes=0.33,
                liquidity=700.0,
                outcome_index=2,
            ),
        ]
        
        return DutchBookBasket(
            basket_id="test_456",
            event_id="event456",
            event_title="Test Election 2",
            relationship_type="winner_take_all",
            outcomes=expensive_outcomes,
            sum_best_ask=1.25,  # Sum > 1.0, not profitable
            fee_rate=0.0315,
            min_edge_after_fees=0.015,
            timestamp=datetime.now(UTC),
            notes="Test basket",
        )

    def test_gross_profit_calculation(self, profitable_basket):
        """Test gross profit calculation."""
        # Gross profit = 1.0 - sum_best_ask = 1.0 - 0.75 = 0.25
        assert profitable_basket.gross_profit == 0.25

    def test_settlement_fees_calculation(self, profitable_basket):
        """Test settlement fees calculation."""
        # Settlement fees = fee_rate * 1.0 = 0.0315
        assert profitable_basket.settlement_fees == 0.0315

    def test_net_profit_calculation(self, profitable_basket):
        """Test net profit calculation."""
        # Net profit = gross_profit - settlement_fees = 0.25 - 0.0315 = 0.2185
        expected = 0.25 - 0.0315
        assert abs(profitable_basket.net_profit - expected) < 0.0001

    def test_net_edge_percent_calculation(self, profitable_basket):
        """Test net edge percentage calculation."""
        # Net edge % = (net_profit / sum_best_ask) * 100
        # = (0.2185 / 0.75) * 100 = 29.13%
        expected = (0.2185 / 0.75) * 100
        assert abs(profitable_basket.net_edge_percent - expected) < 0.01

    def test_is_profitable_true(self, profitable_basket):
        """Test is_profitable returns True for profitable basket."""
        assert profitable_basket.is_profitable is True

    def test_is_profitable_false(self, unprofitable_basket):
        """Test is_profitable returns False for unprofitable basket."""
        assert unprofitable_basket.is_profitable is False

    def test_min_liquidity_calculation(self, profitable_basket):
        """Test minimum liquidity calculation."""
        # Min of [500, 600, 700] = 500
        assert profitable_basket.min_liquidity == 500.0

    def test_outcome_count(self, profitable_basket):
        """Test outcome count property."""
        assert profitable_basket.outcome_count == 3

    def test_to_dict(self, profitable_basket):
        """Test serialization to dict."""
        d = profitable_basket.to_dict()
        assert d["basket_id"] == "test_123"
        assert d["event_title"] == "Test Election"
        assert d["sum_best_ask"] == 0.75
        assert d["gross_profit"] == 0.25
        assert d["is_profitable"] is True
        assert len(d["outcomes"]) == 3


class TestCombinatorialScanResult:
    """Test CombinatorialScanResult."""

    def test_to_dict(self):
        """Test serialization."""
        timestamp = datetime.now(UTC)
        result = CombinatorialScanResult(
            timestamp=timestamp,
            events_scanned=100,
            baskets_constructed=10,
            opportunities_found=2,
            baskets=[],
            profitable_baskets=[],
            parameters={"fee_rate": 0.0315},
        )
        
        d = result.to_dict()
        assert d["events_scanned"] == 100
        assert d["baskets_constructed"] == 10
        assert d["opportunities_found"] == 2
        assert d["parameters"]["fee_rate"] == 0.0315


class TestFormatScanReport:
    """Test report formatting."""

    def test_format_includes_header(self):
        """Test report includes header."""
        timestamp = datetime.now(UTC)
        result = CombinatorialScanResult(
            timestamp=timestamp,
            events_scanned=100,
            baskets_constructed=5,
            opportunities_found=0,
            baskets=[],
            profitable_baskets=[],
            parameters={},
        )
        
        report = format_scan_report(result)
        assert "COMBINATORIAL ARBITRAGE SCAN REPORT" in report
        assert "Events scanned: 100" in report

    def test_format_with_profitable_baskets(self):
        """Test report includes profitable baskets."""
        timestamp = datetime.now(UTC)
        
        outcome = BasketOutcome(
            market_id="1",
            market_slug="test",
            market_question="Test?",
            token_id_yes="token1",
            best_ask_yes=0.30,
            best_bid_yes=0.28,
            liquidity=500.0,
            outcome_index=0,
        )
        
        basket = DutchBookBasket(
            basket_id="test",
            event_id="event1",
            event_title="Test Event",
            relationship_type="winner_take_all",
            outcomes=[outcome],
            sum_best_ask=0.90,
            fee_rate=0.0315,
            min_edge_after_fees=0.015,
            timestamp=timestamp,
        )
        
        result = CombinatorialScanResult(
            timestamp=timestamp,
            events_scanned=100,
            baskets_constructed=1,
            opportunities_found=1,
            baskets=[basket],
            profitable_baskets=[basket],
            parameters={},
        )
        
        report = format_scan_report(result)
        assert "PROFITABLE OPPORTUNITIES" in report
        assert "Test Event" in report

    def test_format_with_detailed_output(self):
        """Test detailed report includes outcome details."""
        timestamp = datetime.now(UTC)
        
        outcome = BasketOutcome(
            market_id="1",
            market_slug="test",
            market_question="Will candidate win?",
            token_id_yes="token1",
            best_ask_yes=0.30,
            best_bid_yes=0.28,
            liquidity=500.0,
            outcome_index=0,
        )
        
        basket = DutchBookBasket(
            basket_id="test",
            event_id="event1",
            event_title="Test Event",
            relationship_type="winner_take_all",
            outcomes=[outcome],
            sum_best_ask=0.90,
            fee_rate=0.0315,
            min_edge_after_fees=0.015,
            timestamp=timestamp,
        )
        
        result = CombinatorialScanResult(
            timestamp=timestamp,
            events_scanned=100,
            baskets_constructed=1,
            opportunities_found=1,
            baskets=[basket],
            profitable_baskets=[basket],
            parameters={},
        )
        
        report = format_scan_report(result, detailed=True)
        assert "Outcomes:" in report
        assert "Will candidate win?" in report


class TestManualBasketDefinitions:
    """Test manual basket definitions."""

    def test_definitions_exist(self):
        """Test that manual basket definitions exist."""
        from polymarket.combinatorial import MANUAL_BASKET_DEFINITIONS
        
        assert len(MANUAL_BASKET_DEFINITIONS) > 0
        
        # Check required fields
        for definition in MANUAL_BASKET_DEFINITIONS:
            assert "event_slug_starts_with" in definition
            assert "relationship_type" in definition
            assert "description" in definition
            assert "max_outcomes" in definition

    def test_definitions_have_valid_relationship_types(self):
        """Test that relationship types are valid."""
        from polymarket.combinatorial import MANUAL_BASKET_DEFINITIONS
        
        valid_types = {
            "winner_take_all",
            "nomination_winner",
            "election_winner",
            "primary_winner",
        }
        
        for definition in MANUAL_BASKET_DEFINITIONS:
            assert definition["relationship_type"] in valid_types


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_basket_properties(self):
        """Test basket with no outcomes."""
        timestamp = datetime.now(UTC)
        
        basket = DutchBookBasket(
            basket_id="empty",
            event_id="event1",
            event_title="Empty Event",
            relationship_type="winner_take_all",
            outcomes=[],
            sum_best_ask=0.0,
            fee_rate=0.0315,
            min_edge_after_fees=0.015,
            timestamp=timestamp,
        )
        
        assert basket.outcome_count == 0
        assert basket.min_liquidity == 0.0
        assert basket.gross_profit == 1.0  # 1.0 - 0.0
        # Net edge with 0 sum should be 0
        assert basket.net_edge_percent == 0.0

    def test_single_outcome_basket(self):
        """Test basket with single outcome."""
        timestamp = datetime.now(UTC)
        
        outcome = BasketOutcome(
            market_id="1",
            market_slug="test",
            market_question="Test?",
            token_id_yes="token1",
            best_ask_yes=0.90,
            best_bid_yes=0.88,
            liquidity=500.0,
            outcome_index=0,
        )
        
        basket = DutchBookBasket(
            basket_id="single",
            event_id="event1",
            event_title="Single Outcome Event",
            relationship_type="winner_take_all",
            outcomes=[outcome],
            sum_best_ask=0.90,
            fee_rate=0.0315,
            min_edge_after_fees=0.015,
            timestamp=timestamp,
        )
        
        assert basket.outcome_count == 1
        assert abs(basket.gross_profit - 0.10) < 0.0001  # 1.0 - 0.90
        assert basket.is_profitable is True  # 0.10 - 0.0315 = 0.0685 > 0.015
