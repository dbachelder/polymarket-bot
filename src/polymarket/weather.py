"""Weather data collection from NOAA and other free sources.

This module provides access to weather forecast data from:
- NOAA NWS API (free, no API key required)
- Open-Meteo (free, no API key required)

These sources provide GFS/ECMWF/ICON model data for temperature forecasts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# City coordinates for weather lookups
CITY_COORDINATES: dict[str, tuple[float, float]] = {
    "nyc": (40.7128, -74.0060),
    "new_york": (40.7128, -74.0060),
    "london": (51.5074, -0.1278),
    "chicago": (41.8781, -87.6298),
    "seattle": (47.6062, -122.3321),
    "atlanta": (33.7490, -84.3880),
    "dallas": (32.7767, -96.7970),
    "miami": (25.7617, -80.1918),
}

# Weather station IDs for NOAA GHCN (for historical verification)
NOAA_STATIONS: dict[str, str] = {
    "nyc": "USW00014732",  # Central Park
    "new_york": "USW00014732",
    "london": "UK000003377",  # Heathrow
    "chicago": "USW00014819",  # O'Hare
    "seattle": "USW00024233",  # Sea-Tac
    "atlanta": "USW00013874",  # Hartsfield-Jackson
    "dallas": "USW00003927",  # DFW
    "miami": "USW00012839",  # Miami International
}


@dataclass(frozen=True)
class TemperatureForecast:
    """Temperature forecast for a specific city and date.

    Attributes:
        city: City identifier (nyc, london, etc.)
        forecast_date: The date being forecasted
        forecast_made_at: When this forecast was generated (model run time)
        source: Data source (noaa, open-meteo)
        model: Weather model used (gfs, ecmwf, icon, blend)
        temp_high: Predicted high temperature (Fahrenheit)
        temp_low: Predicted low temperature (Fahrenheit)
        confidence: Model confidence if available (0-1)
    """

    city: str
    forecast_date: date
    forecast_made_at: datetime
    source: str
    model: str
    temp_high: float
    temp_low: float
    confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "city": self.city,
            "forecast_date": self.forecast_date.isoformat(),
            "forecast_made_at": self.forecast_made_at.isoformat(),
            "source": self.source,
            "model": self.model,
            "temp_high": self.temp_high,
            "temp_low": self.temp_low,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class ModelConsensus:
    """Consensus forecast across multiple weather models.

    Attributes:
        city: City identifier
        forecast_date: The date being forecasted
        consensus_high: Average high temperature across models
        consensus_low: Average low temperature across models
        model_count: Number of models agreeing
        models: List of individual model forecasts
        agreement_score: How well models agree (0-1, 1 = perfect agreement)
    """

    city: str
    forecast_date: date
    consensus_high: float
    consensus_low: float
    model_count: int
    models: list[TemperatureForecast]
    agreement_score: float

    @property
    def temp_range_midpoint(self) -> float:
        """Midpoint of the consensus temperature range."""
        return (self.consensus_high + self.consensus_low) / 2

    def probability_above_threshold(self, threshold: float) -> float:
        """Calculate probability that high temp exceeds threshold.

        Uses simple model: assumes normal distribution around consensus
        with std dev proportional to model spread.
        """
        if self.model_count < 2:
            # Single model: assume 85% confidence at ±3°F
            spread = 3.0
        else:
            # Multiple models: use actual spread
            highs = [m.temp_high for m in self.models]
            spread = max(2.0, (max(highs) - min(highs)) / 2)

        # Z-score for threshold
        z = (self.consensus_high - threshold) / spread
        # Convert to probability using error function approximation
        # P(X > threshold) = 0.5 * (1 + erf(z / sqrt(2)))
        import math

        prob = 0.5 * (1 + math.erf(z / math.sqrt(2)))
        return min(0.99, max(0.01, prob))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "city": self.city,
            "forecast_date": self.forecast_date.isoformat(),
            "consensus_high": self.consensus_high,
            "consensus_low": self.consensus_low,
            "model_count": self.model_count,
            "agreement_score": self.agreement_score,
            "temp_range_midpoint": self.temp_range_midpoint,
            "models": [m.to_dict() for m in self.models],
        }


def _fahrenheit(celsius: float) -> float:
    """Convert Celsius to Fahrenheit."""
    return (celsius * 9 / 5) + 32


def fetch_noaa_forecast(city: str) -> list[TemperatureForecast]:
    """Fetch forecast from NOAA NWS API.

    NOAA provides forecasts based on blended model output.
    Free, no API key required.

    Args:
        city: City identifier (must be in CITY_COORDINATES)

    Returns:
        List of forecasts for upcoming days
    """
    if city not in CITY_COORDINATES:
        raise ValueError(f"Unknown city: {city}")

    lat, lon = CITY_COORDINATES[city]

    try:
        # First get the grid endpoint for the location
        points_url = f"https://api.weather.gov/points/{lat},{lon}"

        with httpx.Client(timeout=30.0) as client:
            points_resp = client.get(points_url, headers={"User-Agent": "polymarket-bot/0.1"})
            points_resp.raise_for_status()
            points_data = points_resp.json()

            # Get the forecast URL from the points response
            forecast_url = points_data["properties"]["forecast"]

            forecast_resp = client.get(forecast_url, headers={"User-Agent": "polymarket-bot/0.1"})
            forecast_resp.raise_for_status()
            forecast_data = forecast_resp.json()

        forecasts: list[TemperatureForecast] = []
        now = datetime.now(UTC)

        for period in forecast_data["properties"]["periods"]:
            # Parse the forecast date
            start_time = period.get("startTime", "")
            try:
                forecast_datetime = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
                forecast_date = forecast_datetime.date()
            except (ValueError, AttributeError):
                continue

            # Extract temperature
            temp = period.get("temperature")
            if temp is None:
                continue

            temp_val = float(temp)

            # Determine high/low based on period name
            is_daytime = period.get("isDaytime", True)

            if is_daytime:
                # Daytime period = high temp
                forecasts.append(
                    TemperatureForecast(
                        city=city,
                        forecast_date=forecast_date,
                        forecast_made_at=now,
                        source="noaa",
                        model="blend",
                        temp_high=temp_val,
                        temp_low=temp_val - 15,  # Estimate low
                    )
                )

        return forecasts

    except httpx.HTTPStatusError as e:
        logger.exception("NOAA API error for %s: %s", city, e)
        return []
    except Exception as e:
        logger.exception("Error fetching NOAA forecast for %s: %s", city, e)
        return []


def fetch_openmeteo_forecast(city: str) -> list[TemperatureForecast]:
    """Fetch forecast from Open-Meteo API.

    Open-Meteo provides access to GFS, ECMWF, and ICON models.
    Free for non-commercial use, no API key required.

    Args:
        city: City identifier (must be in CITY_COORDINATES)

    Returns:
        List of forecasts for upcoming days across multiple models
    """
    if city not in CITY_COORDINATES:
        raise ValueError(f"Unknown city: {city}")

    lat, lon = CITY_COORDINATES[city]
    forecasts: list[TemperatureForecast] = []
    now = datetime.now(UTC)

    # Models to query: GFS, ECMWF, ICON
    models = {
        "gfs": "gfs_seamless",
        "ecmwf": "ecmwf_ifs04",
        "icon": "icon_seamless",
    }

    with httpx.Client(timeout=30.0) as client:
        for model_name, model_param in models.items():
            try:
                url = (
                    f"https://api.open-meteo.com/v1/forecast?"
                    f"latitude={lat}&longitude={lon}&"
                    f"daily=temperature_2m_max,temperature_2m_min&"
                    f"models={model_param}&"
                    f"temperature_unit=fahrenheit&"
                    f"timezone=auto&"
                    f"forecast_days=7"
                )

                resp = client.get(url)
                resp.raise_for_status()
                data = resp.json()

                daily = data.get("daily", {})
                dates = daily.get("time", [])
                max_temps = daily.get("temperature_2m_max", [])
                min_temps = daily.get("temperature_2m_min", [])

                for i, date_str in enumerate(dates):
                    try:
                        forecast_date = date.fromisoformat(date_str)
                    except (ValueError, TypeError):
                        continue

                    if i < len(max_temps) and i < len(min_temps):
                        forecasts.append(
                            TemperatureForecast(
                                city=city,
                                forecast_date=forecast_date,
                                forecast_made_at=now,
                                source="open-meteo",
                                model=model_name,
                                temp_high=float(max_temps[i]),
                                temp_low=float(min_temps[i]),
                            )
                        )

            except httpx.HTTPStatusError as e:
                logger.warning("Open-Meteo %s API error for %s: %s", model_name, city, e)
            except Exception as e:
                logger.warning("Error fetching Open-Meteo %s for %s: %s", model_name, city, e)

    return forecasts


def compute_model_consensus(
    forecasts: list[TemperatureForecast],
    city: str,
    target_date: date | None = None,
    min_models: int = 2,
) -> ModelConsensus | None:
    """Compute consensus forecast from multiple model outputs.

    Args:
        forecasts: List of forecasts from different models
        city: City identifier
        target_date: Specific date to compute consensus for (default: tomorrow)
        min_models: Minimum number of models required for consensus

    Returns:
        ModelConsensus or None if insufficient data
    """
    if target_date is None:
        target_date = date.today() + timedelta(days=1)

    # Filter forecasts for target date
    date_forecasts = [f for f in forecasts if f.forecast_date == target_date]

    if len(date_forecasts) < min_models:
        return None

    # Group by model to avoid duplicates
    seen_models: set[str] = set()
    unique_forecasts: list[TemperatureForecast] = []
    for f in date_forecasts:
        key = f"{f.source}-{f.model}"
        if key not in seen_models:
            seen_models.add(key)
            unique_forecasts.append(f)

    if len(unique_forecasts) < min_models:
        return None

    # Calculate consensus
    highs = [f.temp_high for f in unique_forecasts]
    lows = [f.temp_low for f in unique_forecasts]

    consensus_high = sum(highs) / len(highs)
    consensus_low = sum(lows) / len(lows)

    # Agreement score: 1 - (coefficient of variation)
    import statistics

    if len(highs) > 1:
        high_std = statistics.stdev(highs)
        high_cv = high_std / consensus_high if consensus_high != 0 else 0
        agreement_score = max(0, 1 - high_cv)
    else:
        agreement_score = 1.0

    return ModelConsensus(
        city=city,
        forecast_date=target_date,
        consensus_high=consensus_high,
        consensus_low=consensus_low,
        model_count=len(unique_forecasts),
        models=unique_forecasts,
        agreement_score=agreement_score,
    )


def fetch_all_forecasts(cities: list[str] | None = None) -> dict[str, list[TemperatureForecast]]:
    """Fetch forecasts for all supported cities from all sources.

    Args:
        cities: List of city identifiers (default: all supported cities)

    Returns:
        Dictionary mapping city -> list of forecasts
    """
    if cities is None:
        cities = list(CITY_COORDINATES.keys())

    results: dict[str, list[TemperatureForecast]] = {}

    for city in cities:
        forecasts: list[TemperatureForecast] = []

        # Try NOAA first (US cities only)
        if city in NOAA_STATIONS or city.replace("_", " ") in ["new york"]:
            try:
                noaa_forecasts = fetch_noaa_forecast(city)
                forecasts.extend(noaa_forecasts)
            except Exception as e:
                logger.warning("NOAA fetch failed for %s: %s", city, e)

        # Always try Open-Meteo (works globally)
        try:
            om_forecasts = fetch_openmeteo_forecast(city)
            forecasts.extend(om_forecasts)
        except Exception as e:
            logger.warning("Open-Meteo fetch failed for %s: %s", city, e)

        if forecasts:
            results[city] = forecasts

    return results


def get_consensus_for_cities(
    cities: list[str] | None = None,
    target_date: date | None = None,
    min_models: int = 2,
) -> dict[str, ModelConsensus]:
    """Get model consensus forecasts for multiple cities.

    Args:
        cities: List of city identifiers (default: all)
        target_date: Specific date (default: tomorrow)
        min_models: Minimum models for consensus

    Returns:
        Dictionary mapping city -> ModelConsensus
    """
    all_forecasts = fetch_all_forecasts(cities)

    consensus: dict[str, ModelConsensus] = {}
    for city, forecasts in all_forecasts.items():
        cons = compute_model_consensus(forecasts, city, target_date, min_models)
        if cons:
            consensus[city] = cons

    return consensus
