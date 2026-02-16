"""Tests for odds_api module."""

from decimal import Decimal


from polymarket.odds_api import (
    BookOdds,
    MarketOutcome,
    SportsMarket,
    get_sharp_implied_prob,
)


class TestBookOdds:
    """Tests for BookOdds dataclass."""

    def test_american_to_decimal_positive(self) -> None:
        """Test conversion of positive American odds."""
        odds = BookOdds.from_api("test", "Test", 150, "2024-01-01")
        assert odds.decimal_odds == Decimal("2.5")

    def test_american_to_decimal_negative(self) -> None:
        """Test conversion of negative American odds."""
        odds = BookOdds.from_api("test", "Test", -200, "2024-01-01")
        assert odds.decimal_odds == Decimal("1.5")

    def test_implied_probability_positive(self) -> None:
        """Test implied probability from positive odds."""
        odds = BookOdds.from_api("test", "Test", 100, "2024-01-01")
        # +100 = 2.0 decimal = 0.5 implied
        assert odds.implied_prob == Decimal("0.5")

    def test_implied_probability_negative(self) -> None:
        """Test implied probability from negative odds."""
        odds = BookOdds.from_api("test", "Test", -200, "2024-01-01")
        # -200 = 1.5 decimal = 0.666... implied
        assert odds.implied_prob == Decimal("2") / Decimal("3")


class TestMarketOutcome:
    """Tests for MarketOutcome dataclass."""

    def test_best_odds_empty(self) -> None:
        """Test best_odds with no odds."""
        outcome = MarketOutcome(name="Test", odds=[])
        assert outcome.best_odds is None

    def test_best_odds_multiple(self) -> None:
        """Test best_odds selects highest decimal."""
        odds1 = BookOdds.from_api("book1", "Book 1", -200, "2024-01-01")  # 1.5
        odds2 = BookOdds.from_api("book2", "Book 2", 150, "2024-01-01")  # 2.5
        outcome = MarketOutcome(name="Test", odds=[odds1, odds2])
        assert outcome.best_odds == odds2  # Higher decimal

    def test_best_sharp_prob_prioritizes_sharp(self) -> None:
        """Test that sharp books are prioritized."""
        pinnacle = BookOdds.from_api("pinnacle", "Pinnacle", -150, "2024-01-01")
        random_book = BookOdds.from_api("random", "Random", 200, "2024-01-01")
        outcome = MarketOutcome(name="Test", odds=[random_book, pinnacle])
        # Should return pinnacle's prob even though random has higher odds
        assert outcome.best_sharp_prob == pinnacle.implied_prob

    def test_best_sharp_prob_fallback(self) -> None:
        """Test fallback to best overall when no sharp books."""
        book1 = BookOdds.from_api("book1", "Book 1", -200, "2024-01-01")
        book2 = BookOdds.from_api("book2", "Book 2", 150, "2024-01-01")
        outcome = MarketOutcome(name="Test", odds=[book1, book2])
        assert outcome.best_sharp_prob == book2.implied_prob  # Best overall


class TestSportsMarket:
    """Tests for SportsMarket dataclass."""

    def test_is_sports_true(self) -> None:
        """Test is_sports for actual sports."""
        market = SportsMarket(
            id="1",
            sport_key="americanfootball_nfl",
            sport_title="NFL",
            home_team="Chiefs",
            away_team="Eagles",
            commence_time="2024-01-01T20:00:00Z",
            market_key="h2h",
            outcomes=[],
        )
        assert market.is_sports is True

    def test_is_sports_false_elections(self) -> None:
        """Test is_sports excludes elections."""
        market = SportsMarket(
            id="1",
            sport_key="elections",
            sport_title="Elections",
            home_team="Candidate A",
            away_team="Candidate B",
            commence_time="2024-01-01T20:00:00Z",
            market_key="h2h",
            outcomes=[],
        )
        assert market.is_sports is False

    def test_is_sports_false_crypto(self) -> None:
        """Test is_sports excludes crypto."""
        market = SportsMarket(
            id="1",
            sport_key="crypto_bitcoin",
            sport_title="Crypto",
            home_team="",
            away_team="",
            commence_time="2024-01-01T20:00:00Z",
            market_key="h2h",
            outcomes=[],
        )
        assert market.is_sports is False

    def test_get_outcome_found(self) -> None:
        """Test getting existing outcome."""
        outcome = MarketOutcome(name="Chiefs", odds=[])
        market = SportsMarket(
            id="1",
            sport_key="nfl",
            sport_title="NFL",
            home_team="Chiefs",
            away_team="Eagles",
            commence_time="2024-01-01T20:00:00Z",
            market_key="h2h",
            outcomes=[outcome],
        )
        assert market.get_outcome("Chiefs") == outcome

    def test_get_outcome_not_found(self) -> None:
        """Test getting non-existent outcome."""
        market = SportsMarket(
            id="1",
            sport_key="nfl",
            sport_title="NFL",
            home_team="Chiefs",
            away_team="Eagles",
            commence_time="2024-01-01T20:00:00Z",
            market_key="h2h",
            outcomes=[],
        )
        assert market.get_outcome("Cowboys") is None

    def test_get_outcome_case_insensitive(self) -> None:
        """Test case-insensitive outcome lookup."""
        outcome = MarketOutcome(name="Chiefs", odds=[])
        market = SportsMarket(
            id="1",
            sport_key="nfl",
            sport_title="NFL",
            home_team="Chiefs",
            away_team="Eagles",
            commence_time="2024-01-01T20:00:00Z",
            market_key="h2h",
            outcomes=[outcome],
        )
        assert market.get_outcome("CHIEFS") == outcome


class TestGetSharpImpliedProb:
    """Tests for get_sharp_implied_prob function."""

    def test_found_outcome(self) -> None:
        """Test getting prob for existing outcome."""
        outcome = MarketOutcome(
            name="Chiefs",
            odds=[BookOdds.from_api("pinnacle", "Pinnacle", -150, "2024-01-01")],
        )
        market = SportsMarket(
            id="1",
            sport_key="nfl",
            sport_title="NFL",
            home_team="Chiefs",
            away_team="Eagles",
            commence_time="2024-01-01T20:00:00Z",
            market_key="h2h",
            outcomes=[outcome],
        )
        prob = get_sharp_implied_prob(market, "Chiefs")
        assert prob is not None
        assert prob > Decimal("0")

    def test_not_found(self) -> None:
        """Test getting prob for non-existent outcome."""
        market = SportsMarket(
            id="1",
            sport_key="nfl",
            sport_title="NFL",
            home_team="Chiefs",
            away_team="Eagles",
            commence_time="2024-01-01T20:00:00Z",
            market_key="h2h",
            outcomes=[],
        )
        prob = get_sharp_implied_prob(market, "Cowboys")
        assert prob is None
