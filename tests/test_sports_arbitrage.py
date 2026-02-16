"""Tests for sports_arbitrage module."""

from decimal import Decimal


from polymarket.sports_arbitrage import (
    ArbitrageOpportunity,
    PaperTrade,
    calculate_edge,
    calculate_edge_after_fees,
    calculate_kelly_size,
    execute_paper_trade,
    match_markets,
)
from polymarket.odds_api import SportsMarket
from polymarket.sports_markets import PolymarketSportMarket


class TestCalculateEdge:
    """Tests for calculate_edge function."""

    def test_edge_yes_positive(self) -> None:
        """Test positive edge on YES side."""
        # PM: 45%, Sharp: 50% -> 5% edge
        edge = calculate_edge(Decimal("0.45"), Decimal("0.50"), "yes")
        assert edge == Decimal("0.05")

    def test_edge_yes_zero(self) -> None:
        """Test zero edge on YES side."""
        edge = calculate_edge(Decimal("0.50"), Decimal("0.50"), "yes")
        assert edge == Decimal("0")

    def test_edge_no_positive(self) -> None:
        """Test positive edge on NO side."""
        # PM YES 55% -> PM NO 45%
        # Sharp YES 60% -> Sharp NO 40%
        # Edge buying NO = 45% - 40% = 5%
        edge = calculate_edge(Decimal("0.55"), Decimal("0.60"), "no")
        assert edge == Decimal("0.05")


class TestCalculateEdgeAfterFees:
    """Tests for calculate_edge_after_fees function."""

    def test_edge_after_fees(self) -> None:
        """Test edge calculation after fees."""
        edge = Decimal("0.05")  # 5% edge
        after_fees = calculate_edge_after_fees(edge)
        # 5% - 2% withdrawal fee = 3%
        assert after_fees == Decimal("0.03")

    def test_edge_negative_after_fees(self) -> None:
        """Test that small edges become negative after fees."""
        edge = Decimal("0.015")  # 1.5% edge
        after_fees = calculate_edge_after_fees(edge)
        # 1.5% - 2% = -0.5%
        assert after_fees == Decimal("-0.005")


class TestCalculateKellySize:
    """Tests for calculate_kelly_size function."""

    def test_kelly_calculation(self) -> None:
        """Test Kelly criterion calculation."""
        bankroll = Decimal("10000")
        edge = Decimal("0.05")  # 5% edge
        implied_prob = Decimal("0.50")  # 50% win probability

        size = calculate_kelly_size(bankroll, edge, implied_prob)

        # Kelly = edge / odds = 0.05 / 1.0 = 0.05
        # Kelly/4 = 0.0125
        # Size = 10000 * 0.0125 = 125
        assert size > Decimal("0")
        assert size <= bankroll * Decimal("0.05")  # Max 5% bankroll

    def test_kelly_zero_edge(self) -> None:
        """Test Kelly with zero edge."""
        size = calculate_kelly_size(Decimal("10000"), Decimal("0"), Decimal("0.50"))
        assert size == Decimal("0")

    def test_kelly_negative_edge(self) -> None:
        """Test Kelly with negative edge."""
        size = calculate_kelly_size(Decimal("10000"), Decimal("-0.05"), Decimal("0.50"))
        assert size == Decimal("0")

    def test_kelly_respects_max_bankroll(self) -> None:
        """Test that position size respects max bankroll percentage."""
        bankroll = Decimal("10000")
        edge = Decimal("0.50")  # Huge 50% edge
        implied_prob = Decimal("0.90")  # 90% win probability

        size = calculate_kelly_size(bankroll, edge, implied_prob)

        # Should be capped at 5% of bankroll
        assert size <= Decimal("500")


class TestMatchMarkets:
    """Tests for match_markets function."""

    def test_match_by_sport_and_teams(self) -> None:
        """Test matching markets by sport and team names."""
        pm_markets = [
            PolymarketSportMarket(
                market_id="pm1",
                slug="chiefs-eagles",
                question="Will Chiefs beat Eagles?",
                description=None,
                sport_key="americanfootball_nfl",
                sport_title="NFL",
                event_name="Chiefs vs Eagles",
                outcome_yes="Yes",
                outcome_no="No",
                volume=Decimal("50000"),
                yes_price=Decimal("0.55"),
                no_price=Decimal("0.45"),
                yes_token_id="yes1",
                no_token_id="no1",
                end_date=None,
                tags=[],
            )
        ]

        sharp_markets = [
            SportsMarket(
                id="sharp1",
                sport_key="americanfootball_nfl",
                sport_title="NFL",
                home_team="Chiefs",
                away_team="Eagles",
                commence_time="2024-01-01T20:00:00Z",
                market_key="h2h",
                outcomes=[],
            )
        ]

        matches = match_markets(pm_markets, sharp_markets)
        assert len(matches) == 1

    def test_no_match_different_sport(self) -> None:
        """Test no match when sports differ."""
        pm_markets = [
            PolymarketSportMarket(
                market_id="pm1",
                slug="chiefs-eagles",
                question="Will Chiefs beat Eagles?",
                description=None,
                sport_key="americanfootball_nfl",
                sport_title="NFL",
                event_name="Chiefs vs Eagles",
                outcome_yes="Yes",
                outcome_no="No",
                volume=Decimal("50000"),
                yes_price=Decimal("0.55"),
                no_price=Decimal("0.45"),
                yes_token_id="yes1",
                no_token_id="no1",
                end_date=None,
                tags=[],
            )
        ]

        sharp_markets = [
            SportsMarket(
                id="sharp1",
                sport_key="basketball_nba",  # Different sport
                sport_title="NBA",
                home_team="Lakers",
                away_team="Celtics",
                commence_time="2024-01-01T20:00:00Z",
                market_key="h2h",
                outcomes=[],
            )
        ]

        matches = match_markets(pm_markets, sharp_markets)
        assert len(matches) == 0

    def test_no_match_different_teams(self) -> None:
        """Test no match when teams differ."""
        pm_markets = [
            PolymarketSportMarket(
                market_id="pm1",
                slug="chiefs-eagles",
                question="Will Chiefs beat Eagles?",
                description=None,
                sport_key="americanfootball_nfl",
                sport_title="NFL",
                event_name="Chiefs vs Eagles",
                outcome_yes="Yes",
                outcome_no="No",
                volume=Decimal("50000"),
                yes_price=Decimal("0.55"),
                no_price=Decimal("0.45"),
                yes_token_id="yes1",
                no_token_id="no1",
                end_date=None,
                tags=[],
            )
        ]

        sharp_markets = [
            SportsMarket(
                id="sharp1",
                sport_key="americanfootball_nfl",
                sport_title="NFL",
                home_team="Cowboys",  # Different team
                away_team="Eagles",
                commence_time="2024-01-01T20:00:00Z",
                market_key="h2h",
                outcomes=[],
            )
        ]

        matches = match_markets(pm_markets, sharp_markets)
        # Should NOT match - only one team name in common
        assert len(matches) == 0


class TestArbitrageOpportunity:
    """Tests for ArbitrageOpportunity dataclass."""

    def test_is_valid_true(self) -> None:
        """Test valid opportunity."""
        opp = ArbitrageOpportunity(
            pm_market={"market_id": "1"},
            sharp_market={"id": "2"},
            side="yes",
            pm_implied=Decimal("0.45"),
            sharp_implied=Decimal("0.50"),
            edge=Decimal("0.05"),
            edge_after_fees=Decimal("0.03"),
            confidence=Decimal("0.8"),
            timestamp="2024-01-01T00:00:00Z",
        )
        assert opp.is_valid is True

    def test_is_valid_false_edge(self) -> None:
        """Test invalid due to edge after fees."""
        opp = ArbitrageOpportunity(
            pm_market={"market_id": "1"},
            sharp_market={"id": "2"},
            side="yes",
            pm_implied=Decimal("0.49"),
            sharp_implied=Decimal("0.50"),
            edge=Decimal("0.01"),
            edge_after_fees=Decimal("-0.01"),  # Negative after fees
            confidence=Decimal("0.8"),
            timestamp="2024-01-01T00:00:00Z",
        )
        assert opp.is_valid is False

    def test_is_valid_false_confidence(self) -> None:
        """Test invalid due to low confidence."""
        opp = ArbitrageOpportunity(
            pm_market={"market_id": "1"},
            sharp_market={"id": "2"},
            side="yes",
            pm_implied=Decimal("0.45"),
            sharp_implied=Decimal("0.50"),
            edge=Decimal("0.05"),
            edge_after_fees=Decimal("0.03"),
            confidence=Decimal("0.3"),  # Below 0.5 threshold
            timestamp="2024-01-01T00:00:00Z",
        )
        assert opp.is_valid is False


class TestExecutePaperTrade:
    """Tests for execute_paper_trade function."""

    def test_trade_creation(self) -> None:
        """Test paper trade creation."""
        opp = ArbitrageOpportunity(
            pm_market={
                "market_id": "pm123",
                "question": "Will Chiefs win?",
                "yes_token_id": "yes_token_123",
            },
            sharp_market={"id": "sharp456"},
            side="yes",
            pm_implied=Decimal("0.45"),
            sharp_implied=Decimal("0.50"),
            edge=Decimal("0.05"),
            edge_after_fees=Decimal("0.03"),
            confidence=Decimal("0.8"),
            timestamp="2024-01-01T00:00:00Z",
        )

        bankroll = Decimal("10000")
        trade = execute_paper_trade(opp, bankroll, trade_id="test_trade_1")

        assert trade.trade_id == "test_trade_1"
        assert trade.pm_market_id == "pm123"
        assert trade.side == "yes"
        assert trade.entry_price == Decimal("0.45")
        assert trade.edge_at_entry == Decimal("0.03")
        assert trade.status == "open"

    def test_trade_auto_id(self) -> None:
        """Test auto-generated trade ID."""
        opp = ArbitrageOpportunity(
            pm_market={"market_id": "pm123", "yes_token_id": "yes_token"},
            sharp_market={},
            side="yes",
            pm_implied=Decimal("0.45"),
            sharp_implied=Decimal("0.50"),
            edge=Decimal("0.05"),
            edge_after_fees=Decimal("0.03"),
            confidence=Decimal("0.8"),
            timestamp="2024-01-01T00:00:00Z",
        )

        trade = execute_paper_trade(opp, Decimal("10000"))
        assert trade.trade_id.startswith("arb_")


class TestPaperTrade:
    """Tests for PaperTrade dataclass."""

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        trade = PaperTrade(
            trade_id="test_1",
            timestamp="2024-01-01T00:00:00Z",
            pm_market_id="pm123",
            pm_token_id="token_yes",
            side="yes",
            size=Decimal("100"),
            entry_price=Decimal("0.55"),
            sharp_implied_at_entry=Decimal("0.60"),
            edge_at_entry=Decimal("0.03"),
            status="open",
        )

        d = trade.to_dict()
        assert d["trade_id"] == "test_1"
        assert d["pm_market_id"] == "pm123"
        assert d["side"] == "yes"
        assert d["size"] == "100"
        assert d["status"] == "open"
        assert d["exit_price"] is None
