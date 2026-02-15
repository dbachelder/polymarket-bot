"""Kalshi client for cross-market arbitrage.

Fetches market data from Kalshi API.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

import httpx

from . import KALSHI_FEE_SCHEDULE, FeeSchedule, VenueMarket

logger = logging.getLogger(__name__)

# Kalshi API endpoints
KALSHI_API_BASE = "https://trading-api.kalshi.com/v1"


class KalshiClient:
    """Client for fetching Kalshi market data.

    Note: Kalshi requires authentication for most endpoints.
    This implementation uses public endpoints where available
    and supports authenticated access when credentials are provided.
    """

    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
        fee_schedule: FeeSchedule | None = None,
    ) -> None:
        """Initialize client.

        Args:
            api_key: Kalshi API key (optional for public endpoints)
            api_secret: Kalshi API secret (optional for public endpoints)
            fee_schedule: Fee structure (uses default if not provided)
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.fee_schedule = fee_schedule or KALSHI_FEE_SCHEDULE
        self._http = httpx.Client(timeout=30.0, headers={"User-Agent": "polymarket-bot/0.1"})

    def _get_auth_headers(self) -> dict[str, str]:
        """Get authentication headers if credentials are available."""
        if not self.api_key or not self.api_secret:
            return {}

        # Kalshi uses simple API key auth
        return {
            "Authorization": f"Bearer {self.api_key}",
        }

    def fetch_active_markets(
        self,
        limit: int = 100,
        categories: list[str] | None = None,
        status: str = "open",
    ) -> list[VenueMarket]:
        """Fetch active markets from Kalshi.

        Args:
            limit: Maximum markets to fetch
            categories: Filter by event categories
            status: Market status (open, closed, all)

        Returns:
            List of VenueMarket objects
        """
        markets: list[VenueMarket] = []

        try:
            # Kalshi markets endpoint
            params: dict[str, Any] = {
                "limit": min(limit, 1000),  # Kalshi max is 1000
                "status": status,
            }

            if categories:
                # Kalshi uses event_ticker prefix for categories
                # e.g., "KX" for crypto, "KP" for politics
                pass  # Would need to map categories to tickers

            resp = self._http.get(
                f"{KALSHI_API_BASE}/markets",
                params=params,
                headers=self._get_auth_headers(),
            )
            resp.raise_for_status()
            data = resp.json()

            for item in data.get("markets", []):
                try:
                    market = self._parse_market(item)
                    if market:
                        markets.append(market)
                except Exception as e:
                    logger.debug("Error parsing Kalshi market %s: %s", item.get("ticker"), e)

        except httpx.HTTPStatusError as e:
            logger.exception("HTTP error fetching Kalshi markets: %s", e)
        except Exception as e:
            logger.exception("Error fetching Kalshi markets: %s", e)

        logger.info("Fetched %d markets from Kalshi", len(markets))
        return markets

    def fetch_markets_by_series(self, series_ticker: str) -> list[VenueMarket]:
        """Fetch markets for a specific series (event group).

        Args:
            series_ticker: Series ticker (e.g., "KXBTC" for BTC markets)

        Returns:
            List of VenueMarket objects
        """
        markets: list[VenueMarket] = []

        try:
            # Kalshi series markets endpoint
            resp = self._http.get(
                f"{KALSHI_API_BASE}/series/{series_ticker}/markets",
                headers=self._get_auth_headers(),
            )
            resp.raise_for_status()
            data = resp.json()

            for item in data.get("markets", []):
                try:
                    market = self._parse_market(item)
                    if market:
                        markets.append(market)
                except Exception as e:
                    logger.debug("Error parsing Kalshi market %s: %s", item.get("ticker"), e)

        except httpx.HTTPStatusError as e:
            logger.exception("HTTP error fetching Kalshi series %s: %s", series_ticker, e)
        except Exception as e:
            logger.exception("Error fetching Kalshi series %s: %s", series_ticker, e)

        return markets

    def _parse_market(self, data: dict[str, Any]) -> VenueMarket | None:
        """Parse a market from Kalshi API response."""
        ticker = data.get("ticker", "")
        if not ticker:
            return None

        # Kalshi markets have YES/NO outcomes
        # The market itself represents the YES contract
        # Price is in cents (0-100)

        # Check if this is a binary (YES/NO) market
        # yes_subtitle and no_subtitle indicate binary markets
        _ = data.get("yes_sub_title", "")
        _ = data.get("no_sub_title", "")

        # Kalshi prices are in cents, convert to 0-1 range
        yes_price = data.get("last_price")
        if yes_price is not None:
            yes_price = float(yes_price) / 100

        # Calculate NO price (complement)
        no_price = 1.0 - yes_price if yes_price is not None else None

        # Get bid/ask if available
        yes_bid = data.get("bid")
        yes_ask = data.get("ask")
        if yes_bid is not None:
            yes_bid = float(yes_bid) / 100
        if yes_ask is not None:
            yes_ask = float(yes_ask) / 100

        no_bid = 1.0 - yes_ask if yes_ask is not None else None
        no_ask = 1.0 - yes_bid if yes_bid is not None else None

        # Volume
        volume = data.get("volume", 0)
        if isinstance(volume, str):
            volume = float(volume)

        # Liquidity estimate from open interest
        open_interest = data.get("open_interest", 0)
        if isinstance(open_interest, str):
            open_interest = float(open_interest)

        return VenueMarket(
            venue="kalshi",
            market_id=ticker,
            token_id_yes=ticker,  # Kalshi uses ticker as ID
            token_id_no=f"{ticker}-NO",  # Synthetic NO identifier
            yes_price=yes_price,
            no_price=no_price,
            yes_ask=yes_ask,
            no_ask=no_ask,
            yes_bid=yes_bid,
            no_bid=no_bid,
            volume_24h=float(volume or 0),
            liquidity=float(open_interest or 0),
            fees=self.fee_schedule,
            last_updated=datetime.now(UTC),
        )

    def get_market(self, ticker: str) -> VenueMarket | None:
        """Fetch a specific market by ticker.

        Args:
            ticker: Kalshi market ticker

        Returns:
            VenueMarket or None if not found
        """
        try:
            resp = self._http.get(
                f"{KALSHI_API_BASE}/markets/{ticker}",
                headers=self._get_auth_headers(),
            )
            resp.raise_for_status()
            data = resp.json()
            market_data = data.get("market", data)  # Handle both wrapped and unwrapped
            return self._parse_market(market_data)
        except Exception as e:
            logger.exception("Error fetching Kalshi market %s: %s", ticker, e)
            return None

    def get_exchange_status(self) -> dict[str, Any]:
        """Get Kalshi exchange status.

        Returns:
            Dictionary with exchange status info
        """
        try:
            resp = self._http.get(
                f"{KALSHI_API_BASE}/exchange/status",
                headers=self._get_auth_headers(),
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.exception("Error fetching Kalshi exchange status: %s", e)
            return {"error": str(e)}

    def close(self) -> None:
        """Close the HTTP client."""
        self._http.close()

    def __enter__(self) -> KalshiClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
