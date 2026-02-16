"""Tests for weather strategy module."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

from polymarket.strategy_weather import (
    WeatherMarket,
    WeatherSignal,
    _extract_city_from_question,
    _extract_date_from_question,
    _extract_temperature_threshold,
    _is_weather_market,
    generate_signals,
)
from polymarket.weather import ModelConsensus, TemperatureForecast


class TestExtractCityFromQuestion:
    """Test city extraction from market questions."""

    def test_extract_nyc(self) -> None:
        """Test extracting NYC from various formats."""
        questions = [
            "Will NYC have a high temp above 70°F tomorrow?",
            "New York City temperature forecast",
            "Daily high for New York, NY",
        ]
        for q in questions:
            city = _extract_city_from_question(q)
            assert city in ["nyc", "new_york"], f"Failed for: {q}"

    def test_extract_london(self) -> None:
        """Test extracting London."""
        questions = [
            "Will London have rain tomorrow?",
            "London, UK high temperature",
        ]
        for q in questions:
            city = _extract_city_from_question(q)
            assert city == "london", f"Failed for: {q}"

    def test_extract_chicago(self) -> None:
        """Test extracting Chicago."""
        city = _extract_city_from_question("Will Chicago be hot?")
        assert city == "chicago"

    def test_no_city(self) -> None:
        """Test question with no recognizable city."""
        city = _extract_city_from_question("Will it rain somewhere?")
        assert city is None


class TestExtractDateFromQuestion:
    """Test date extraction from market questions."""

    def test_extract_tomorrow(self) -> None:
        """Test extracting 'tomorrow' date."""
        result = _extract_date_from_question("Will NYC be hot tomorrow?")
        expected = date.today() + timedelta(days=1)
        assert result == expected

    def test_extract_specific_date(self) -> None:
        """Test extracting specific dates."""
        result = _extract_date_from_question("High temp on February 15, 2026?")
        assert result == date(2026, 2, 15)

    def test_extract_abbreviated_date(self) -> None:
        """Test extracting abbreviated month names."""
        result = _extract_date_from_question("Temp on Feb 15?")
        assert result.month == 2
        assert result.day == 15

    def test_extract_with_ordinal(self) -> None:
        """Test extracting dates with ordinal suffixes."""
        result = _extract_date_from_question("High on February 15th?")
        assert result.month == 2
        assert result.day == 15


class TestExtractTemperatureThreshold:
    """Test temperature threshold extraction."""

    def test_extract_above_threshold(self) -> None:
        """Test 'above X' patterns."""
        threshold, condition = _extract_temperature_threshold("Will temp be above 70 degrees?")
        assert threshold == 70.0
        assert condition == "above"

    def test_extract_below_threshold(self) -> None:
        """Test 'below X' patterns."""
        threshold, condition = _extract_temperature_threshold("Will temp be below 50°F?")
        assert threshold == 50.0
        assert condition == "below"

    def test_extract_greater_than(self) -> None:
        """Test 'greater than X' pattern."""
        threshold, condition = _extract_temperature_threshold("Will temp be greater than 65?")
        assert threshold == 65.0
        assert condition == "above"

    def test_extract_under(self) -> None:
        """Test 'under X' pattern."""
        threshold, condition = _extract_temperature_threshold("Will it be under 40 degrees?")
        assert threshold == 40.0
        assert condition == "below"

    def test_extract_exceed(self) -> None:
        """Test 'exceed X' pattern."""
        threshold, condition = _extract_temperature_threshold("Will temp exceed 80?")
        assert threshold == 80.0
        assert condition == "above"

    def test_no_threshold(self) -> None:
        """Test question with no temperature."""
        threshold, condition = _extract_temperature_threshold("Will it rain tomorrow?")
        assert threshold is None
        assert condition is None


class TestIsWeatherMarket:
    """Test weather market detection."""

    def test_temperature_market(self) -> None:
        """Test detecting temperature markets."""
        assert _is_weather_market("Will NYC high temp exceed 70°F?")
        assert _is_weather_market("Daily temperature forecast")

    def test_rain_market(self) -> None:
        """Test detecting rain markets."""
        assert _is_weather_market("Will it rain in London?")

    def test_non_weather_market(self) -> None:
        """Test non-weather markets are not detected."""
        assert not _is_weather_market("Will Bitcoin go up?")
        assert not _is_weather_market("Trump approval rating")


class TestWeatherMarket:
    """Test WeatherMarket dataclass."""

    def test_creation(self) -> None:
        """Test creating a WeatherMarket."""
        market = WeatherMarket(
            market_id="123",
            token_id_yes="yes_token",
            token_id_no="no_token",
            question="Will NYC be above 70°F?",
            city="nyc",
            forecast_date=date(2026, 2, 16),
            market_end_date=datetime(2026, 2, 16, 23, 59, tzinfo=UTC),
            threshold_temp=70.0,
            condition="above",
            current_yes_price=0.65,
            current_no_price=0.35,
        )

        assert market.market_id == "123"
        assert market.city == "nyc"
        assert market.threshold_temp == 70.0
        assert market.condition == "above"
        assert market.implied_probability == 0.65

    def test_implied_probability_none(self) -> None:
        """Test implied probability when no price available."""
        market = WeatherMarket(
            market_id="123",
            token_id_yes="yes_token",
            token_id_no="no_token",
            question="Test",
            city=None,
            forecast_date=None,
            market_end_date=datetime.now(UTC),
            threshold_temp=None,
            condition=None,
            current_yes_price=None,
        )

        assert market.implied_probability is None


class TestGenerateSignals:
    """Test signal generation."""

    def create_consensus(self, high_temp: float) -> ModelConsensus:
        """Helper to create a ModelConsensus."""
        forecasts = [
            TemperatureForecast(
                city="nyc",
                forecast_date=date(2026, 2, 16),
                forecast_made_at=datetime.now(UTC),
                source="test",
                model="gfs",
                temp_high=high_temp,
                temp_low=high_temp - 15,
            ),
            TemperatureForecast(
                city="nyc",
                forecast_date=date(2026, 2, 16),
                forecast_made_at=datetime.now(UTC),
                source="test",
                model="ecmwf",
                temp_high=high_temp + 1,
                temp_low=high_temp - 14,
            ),
        ]

        return ModelConsensus(
            city="nyc",
            forecast_date=date(2026, 2, 16),
            consensus_high=high_temp + 0.5,
            consensus_low=high_temp - 14.5,
            model_count=2,
            models=forecasts,
            agreement_score=0.9,
        )

    def test_buy_yes_signal(self) -> None:
        """Test generating buy_yes signal when conditions are met."""
        market = WeatherMarket(
            market_id="123",
            token_id_yes="yes_token",
            token_id_no="no_token",
            question="Will NYC be above 70°F?",
            city="nyc",
            forecast_date=date(2026, 2, 16),
            market_end_date=datetime(2026, 2, 16, 23, 59, tzinfo=UTC),
            threshold_temp=70.0,
            condition="above",
            current_yes_price=0.10,  # Low market price
            current_no_price=0.90,
        )

        # Consensus says 80°F - high probability of being above 70°F
        consensus = self.create_consensus(high_temp=80.0)

        signals = generate_signals([market], {"nyc": consensus})

        assert len(signals) == 1
        signal = signals[0]
        assert signal.side == "buy_yes"
        assert signal.expected_value > 0

    def test_buy_no_signal(self) -> None:
        """Test generating buy_no signal when conditions are met."""
        market = WeatherMarket(
            market_id="123",
            token_id_yes="yes_token",
            token_id_no="no_token",
            question="Will NYC be above 80°F?",
            city="nyc",
            forecast_date=date(2026, 2, 16),
            market_end_date=datetime(2026, 2, 16, 23, 59, tzinfo=UTC),
            threshold_temp=80.0,
            condition="above",
            current_yes_price=0.50,  # Market split
            current_no_price=0.50,  # NO at 0.50 > 0.45 threshold
        )

        # Consensus says 65°F - low probability (~10%) of being above 80°F
        # So NO probability should be ~90%, which is > 0.55 threshold
        consensus = self.create_consensus(high_temp=65.0)

        signals = generate_signals([market], {"nyc": consensus})

        assert len(signals) == 1
        signal = signals[0]
        # Should recommend buying NO since models show low YES probability
        # meaning high NO probability
        assert signal.side == "buy_no"

    def test_no_trade_when_no_edge(self) -> None:
        """Test no trade when market price matches model."""
        market = WeatherMarket(
            market_id="123",
            token_id_yes="yes_token",
            token_id_no="no_token",
            question="Will NYC be above 70°F?",
            city="nyc",
            forecast_date=date(2026, 2, 16),
            market_end_date=datetime(2026, 2, 16, 23, 59, tzinfo=UTC),
            threshold_temp=70.0,
            condition="above",
            current_yes_price=0.75,  # Market agrees with model
            current_no_price=0.25,
        )

        # Consensus says 75°F - model agrees with market
        consensus = self.create_consensus(high_temp=75.0)

        signals = generate_signals([market], {"nyc": consensus})

        assert len(signals) == 1
        signal = signals[0]
        assert signal.side == "no_trade"

    def test_generates_signals_for_cheap_sides_without_city(self) -> None:
        """Test that cheap sides generate signals even without city identification."""
        market = WeatherMarket(
            market_id="123",
            token_id_yes="yes_token",
            token_id_no="no_token",
            question="Will somewhere be hot?",
            city=None,  # No city identified
            forecast_date=date(2026, 2, 16),
            market_end_date=datetime.now(UTC),
            threshold_temp=70.0,
            condition="above",
            current_yes_price=0.10,  # Cheap side
        )

        consensus = self.create_consensus(high_temp=80.0)

        signals = generate_signals([market], {"nyc": consensus})
        # With loosened thresholds, cheap sides generate signals even without city/consensus
        assert len(signals) == 1
        # When YES is cheap (< 0.15), the NO side is expensive (> 0.85), so we buy NO
        assert signals[0].side == "buy_no"

    def test_generates_signals_for_cheap_sides_without_consensus(self) -> None:
        """Test that cheap sides generate signals even without consensus data."""
        market = WeatherMarket(
            market_id="123",
            token_id_yes="yes_token",
            token_id_no="no_token",
            question="Will Chicago be above 70°F?",
            city="chicago",
            forecast_date=date(2026, 2, 16),
            market_end_date=datetime.now(UTC),
            threshold_temp=70.0,
            condition="above",
            current_yes_price=0.10,  # Cheap side
        )

        # Only have consensus for NYC, not Chicago
        consensus = self.create_consensus(high_temp=80.0)

        signals = generate_signals([market], {"nyc": consensus})
        # With loosened thresholds, cheap sides generate signals even without consensus
        assert len(signals) == 1
        # When YES is cheap (< 0.15), the NO side is expensive (> 0.85), so we buy NO
        assert signals[0].side == "buy_no"


class TestWeatherSignal:
    """Test WeatherSignal dataclass."""

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        market = WeatherMarket(
            market_id="123",
            token_id_yes="yes_token",
            token_id_no="no_token",
            question="Will NYC be above 70°F?",
            city="nyc",
            forecast_date=date(2026, 2, 16),
            market_end_date=datetime.now(UTC),
            threshold_temp=70.0,
            condition="above",
            current_yes_price=0.10,
        )

        forecasts = [
            TemperatureForecast(
                city="nyc",
                forecast_date=date(2026, 2, 16),
                forecast_made_at=datetime.now(UTC),
                source="test",
                model="gfs",
                temp_high=80.0,
                temp_low=65.0,
            ),
        ]

        consensus = ModelConsensus(
            city="nyc",
            forecast_date=date(2026, 2, 16),
            consensus_high=80.0,
            consensus_low=65.0,
            model_count=1,
            models=forecasts,
            agreement_score=0.9,
        )

        signal = WeatherSignal(
            timestamp=datetime.now(UTC),
            market=market,
            consensus=consensus,
            side="buy_yes",
            market_prob=0.10,
            model_prob=0.85,
            edge=0.75,
            confidence=0.8,
            expected_value=0.15,
        )

        d = signal.to_dict()
        assert d["side"] == "buy_yes"
        assert d["market_prob"] == 0.10
        assert d["model_prob"] == 0.85
        assert d["edge"] == 0.75
        assert d["expected_value"] == 0.15
