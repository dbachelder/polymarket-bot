"""Tests for weather data module."""

from __future__ import annotations

from datetime import UTC, date, datetime

from polymarket.weather import (
    CITY_COORDINATES,
    NOAA_STATIONS,
    ModelConsensus,
    TemperatureForecast,
    compute_model_consensus,
    fetch_all_forecasts,
    get_consensus_for_cities,
)


class TestTemperatureForecast:
    """Test TemperatureForecast dataclass."""

    def test_creation(self) -> None:
        """Test creating a TemperatureForecast."""
        forecast = TemperatureForecast(
            city="nyc",
            forecast_date=date(2026, 2, 16),
            forecast_made_at=datetime(2026, 2, 15, 12, 0, 0, tzinfo=UTC),
            source="open-meteo",
            model="gfs",
            temp_high=45.0,
            temp_low=32.0,
        )

        assert forecast.city == "nyc"
        assert forecast.forecast_date == date(2026, 2, 16)
        assert forecast.source == "open-meteo"
        assert forecast.model == "gfs"
        assert forecast.temp_high == 45.0
        assert forecast.temp_low == 32.0
        assert forecast.confidence is None

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        forecast = TemperatureForecast(
            city="nyc",
            forecast_date=date(2026, 2, 16),
            forecast_made_at=datetime(2026, 2, 15, 12, 0, 0, tzinfo=UTC),
            source="open-meteo",
            model="gfs",
            temp_high=45.0,
            temp_low=32.0,
            confidence=0.85,
        )

        d = forecast.to_dict()
        assert d["city"] == "nyc"
        assert d["forecast_date"] == "2026-02-16"
        assert d["temp_high"] == 45.0
        assert d["confidence"] == 0.85


class TestModelConsensus:
    """Test ModelConsensus dataclass and methods."""

    def test_creation(self) -> None:
        """Test creating a ModelConsensus."""
        forecasts = [
            TemperatureForecast(
                city="nyc",
                forecast_date=date(2026, 2, 16),
                forecast_made_at=datetime.now(UTC),
                source="open-meteo",
                model="gfs",
                temp_high=45.0,
                temp_low=32.0,
            ),
            TemperatureForecast(
                city="nyc",
                forecast_date=date(2026, 2, 16),
                forecast_made_at=datetime.now(UTC),
                source="open-meteo",
                model="ecmwf",
                temp_high=46.0,
                temp_low=33.0,
            ),
        ]

        consensus = ModelConsensus(
            city="nyc",
            forecast_date=date(2026, 2, 16),
            consensus_high=45.5,
            consensus_low=32.5,
            model_count=2,
            models=forecasts,
            agreement_score=0.95,
        )

        assert consensus.city == "nyc"
        assert consensus.consensus_high == 45.5
        assert consensus.temp_range_midpoint == 39.0

    def test_probability_above_threshold(self) -> None:
        """Test probability calculation."""
        forecasts = [
            TemperatureForecast(
                city="nyc",
                forecast_date=date(2026, 2, 16),
                forecast_made_at=datetime.now(UTC),
                source="open-meteo",
                model="gfs",
                temp_high=75.0,
                temp_low=60.0,
            ),
            TemperatureForecast(
                city="nyc",
                forecast_date=date(2026, 2, 16),
                forecast_made_at=datetime.now(UTC),
                source="open-meteo",
                model="ecmwf",
                temp_high=76.0,
                temp_low=61.0,
            ),
        ]

        consensus = ModelConsensus(
            city="nyc",
            forecast_date=date(2026, 2, 16),
            consensus_high=75.5,
            consensus_low=60.5,
            model_count=2,
            models=forecasts,
            agreement_score=0.95,
        )

        # High consensus temp (75.5) vs threshold 70 should give high probability
        prob = consensus.probability_above_threshold(70.0)
        assert prob > 0.8
        assert prob < 1.0

        # High consensus temp vs threshold 80 should give lower probability
        prob = consensus.probability_above_threshold(80.0)
        assert prob < 0.5

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        forecasts = [
            TemperatureForecast(
                city="nyc",
                forecast_date=date(2026, 2, 16),
                forecast_made_at=datetime.now(UTC),
                source="open-meteo",
                model="gfs",
                temp_high=45.0,
                temp_low=32.0,
            ),
        ]

        consensus = ModelConsensus(
            city="nyc",
            forecast_date=date(2026, 2, 16),
            consensus_high=45.0,
            consensus_low=32.0,
            model_count=1,
            models=forecasts,
            agreement_score=1.0,
        )

        d = consensus.to_dict()
        assert d["city"] == "nyc"
        assert d["consensus_high"] == 45.0
        assert d["model_count"] == 1
        assert len(d["models"]) == 1


class TestComputeModelConsensus:
    """Test compute_model_consensus function."""

    def test_insufficient_models(self) -> None:
        """Test that None is returned when insufficient models."""
        forecasts = [
            TemperatureForecast(
                city="nyc",
                forecast_date=date(2026, 2, 16),
                forecast_made_at=datetime.now(UTC),
                source="open-meteo",
                model="gfs",
                temp_high=45.0,
                temp_low=32.0,
            ),
        ]

        result = compute_model_consensus(forecasts, "nyc", min_models=2)
        assert result is None

    def test_sufficient_models(self) -> None:
        """Test consensus calculation with sufficient models."""
        target_date = date(2026, 2, 16)
        forecasts = [
            TemperatureForecast(
                city="nyc",
                forecast_date=target_date,
                forecast_made_at=datetime.now(UTC),
                source="open-meteo",
                model="gfs",
                temp_high=45.0,
                temp_low=32.0,
            ),
            TemperatureForecast(
                city="nyc",
                forecast_date=target_date,
                forecast_made_at=datetime.now(UTC),
                source="open-meteo",
                model="ecmwf",
                temp_high=47.0,
                temp_low=34.0,
            ),
        ]

        result = compute_model_consensus(forecasts, "nyc", target_date=target_date, min_models=2)
        assert result is not None
        assert result.consensus_high == 46.0  # Average of 45 and 47
        assert result.consensus_low == 33.0   # Average of 32 and 34
        assert result.model_count == 2

    def test_filters_by_date(self) -> None:
        """Test that forecasts are filtered by target date."""
        target_date = date(2026, 2, 16)
        other_date = date(2026, 2, 17)

        forecasts = [
            TemperatureForecast(
                city="nyc",
                forecast_date=target_date,
                forecast_made_at=datetime.now(UTC),
                source="open-meteo",
                model="gfs",
                temp_high=45.0,
                temp_low=32.0,
            ),
            TemperatureForecast(
                city="nyc",
                forecast_date=target_date,
                forecast_made_at=datetime.now(UTC),
                source="open-meteo",
                model="ecmwf",
                temp_high=47.0,
                temp_low=34.0,
            ),
            TemperatureForecast(
                city="nyc",
                forecast_date=other_date,  # Different date
                forecast_made_at=datetime.now(UTC),
                source="open-meteo",
                model="icon",
                temp_high=50.0,
                temp_low=35.0,
            ),
        ]

        result = compute_model_consensus(forecasts, "nyc", target_date=target_date, min_models=2)
        assert result is not None
        assert result.model_count == 2  # Only 2 forecasts for target date
        assert result.consensus_high == 46.0


class TestCityCoordinates:
    """Test city coordinate constants."""

    def test_all_cities_have_coordinates(self) -> None:
        """Test that all cities in NOAA_STATIONS have coordinates."""
        for city in NOAA_STATIONS:
            assert city in CITY_COORDINATES, f"City {city} missing coordinates"

    def test_coordinates_are_valid(self) -> None:
        """Test that coordinates are valid lat/lon values."""
        for city, (lat, lon) in CITY_COORDINATES.items():
            assert -90 <= lat <= 90, f"Invalid latitude for {city}: {lat}"
            assert -180 <= lon <= 180, f"Invalid longitude for {city}: {lon}"


class TestFetchAllForecasts:
    """Integration tests for forecast fetching (requires network)."""

    def test_fetch_all_forecasts_returns_data(self) -> None:
        """Test that we can fetch forecasts for supported cities."""
        cities = ["nyc", "london"]
        result = fetch_all_forecasts(cities)

        # Should return data for at least one city
        assert len(result) > 0

        # Each city should have forecasts
        for city in result:
            assert len(result[city]) > 0
            forecast = result[city][0]
            assert isinstance(forecast, TemperatureForecast)
            assert forecast.city == city

    def test_get_consensus_for_cities(self) -> None:
        """Test getting consensus for multiple cities."""
        cities = ["nyc"]
        consensus = get_consensus_for_cities(cities, min_models=1)

        # Should have consensus for at least one city
        if consensus:
            for city, cons in consensus.items():
                assert cons.city == city
                assert cons.model_count >= 1
                assert cons.consensus_high > cons.consensus_low
