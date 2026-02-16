"""Polymarket client for cross-market arbitrage.

Fetches market data from Polymarket CLOB and site API.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

import httpx

from ..clob import get_book
from ..site import fetch_predictions_page
from . import POLYMARKET_FEE_SCHEDULE, FeeSchedule, VenueMarket

logger = logging.getLogger(__name__)

# Polymarket API endpoints
GAMMA_API_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"


class PolymarketClient:
    """Client for fetching Polymarket data."""

    def __init__(self, fee_schedule: FeeSchedule | None = None) -> None:
        """Initialize client.

        Args:
            fee_schedule: Fee structure (uses default if not provided)
        """
        self.fee_schedule = fee_schedule or POLYMARKET_FEE_SCHEDULE
        self._http = httpx.Client(timeout=30.0, headers={"User-Agent": "polymarket-bot/0.1"})

    def fetch_active_markets(
        self,
        limit: int = 100,
        categories: list[str] | None = None,
    ) -> list[VenueMarket]:
        """Fetch active markets from Polymarket.

        Args:
            limit: Maximum markets to fetch
            categories: Filter by categories (politics, crypto, sports, etc.)

        Returns:
            List of VenueMarket objects
        """
        markets: list[VenueMarket] = []

        try:
            # Use Gamma API for market discovery
            # Note: This is a simplified endpoint - actual implementation
            # may need pagination and more sophisticated filtering
            params: dict[str, Any] = {
                "active": True,
                "closed": False,
                "limit": limit,
            }

            if categories:
                params["category"] = ",".join(categories)

            resp = self._http.get(f"{GAMMA_API_BASE}/markets", params=params)
            resp.raise_for_status()
            data = resp.json()

            for item in data:
                try:
                    market = self._parse_market(item)
                    if market:
                        markets.append(market)
                except Exception as e:
                    logger.debug("Error parsing market %s: %s", item.get("id"), e)

        except httpx.HTTPStatusError as e:
            logger.exception("HTTP error fetching Polymarket markets: %s", e)
        except Exception as e:
            logger.exception("Error fetching Polymarket markets: %s", e)

        logger.info("Fetched %d markets from Polymarket", len(markets))
        return markets

    def fetch_markets_by_page(self, page_slug: str = "politics") -> list[VenueMarket]:
        """Fetch markets from a specific tag (e.g., politics, crypto, sports).

        Args:
            page_slug: Tag slug (politics, crypto, sports, etc.)

        Returns:
            List of VenueMarket objects
        """
        markets: list[VenueMarket] = []

        try:
            # Use Gamma API for tag-based discovery
            events = fetch_predictions_page(page_slug)

            for event_data in events:
                markets_data = event_data.get("markets", [])
                for m in markets_data:
                    try:
                        market = self._parse_market_from_page(m)
                        if market:
                            markets.append(market)
                    except Exception as e:
                        logger.debug("Error parsing page market: %s", e)

        except Exception as e:
            logger.exception("Error fetching page %s: %s", page_slug, e)

        return markets

    def _parse_market(self, data: dict[str, Any]) -> VenueMarket | None:
        """Parse a market from Gamma API response."""
        token_ids = data.get("clobTokenIds", [])
        if len(token_ids) != 2:
            return None

        market_id = str(data.get("id", ""))
        if not market_id:
            return None

        # Get current prices from CLOB
        yes_id = str(token_ids[0])
        no_id = str(token_ids[1])

        try:
            yes_book = get_book(yes_id)
            no_book = get_book(no_id)

            yes_bids = yes_book.get("bids", [])
            yes_asks = yes_book.get("asks", [])
            no_bids = no_book.get("bids", [])
            no_asks = no_book.get("asks", [])

            # Calculate mid prices
            yes_bid = float(yes_bids[0]["price"]) if yes_bids else None
            yes_ask = float(yes_asks[0]["price"]) if yes_asks else None
            no_bid = float(no_bids[0]["price"]) if no_bids else None
            no_ask = float(no_asks[0]["price"]) if no_asks else None

            yes_price = (yes_bid + yes_ask) / 2 if yes_bid and yes_ask else yes_bid or yes_ask
            no_price = (no_bid + no_ask) / 2 if no_bid and no_ask else no_bid or no_ask

            # Calculate liquidity at best prices
            yes_liquidity = sum(float(b["size"]) for b in yes_bids[:3]) if yes_bids else 0
            no_liquidity = sum(float(b["size"]) for b in no_bids[:3]) if no_bids else 0

            return VenueMarket(
                venue="polymarket",
                market_id=market_id,
                token_id_yes=yes_id,
                token_id_no=no_id,
                yes_price=yes_price,
                no_price=no_price,
                yes_ask=yes_ask,
                no_ask=no_ask,
                yes_bid=yes_bid,
                no_bid=no_bid,
                volume_24h=float(data.get("volume24h", 0) or data.get("volume24Hour", 0) or 0),
                liquidity=min(yes_liquidity, no_liquidity),
                fees=self.fee_schedule,
                last_updated=datetime.now(UTC),
            )

        except Exception as e:
            logger.debug("Error fetching CLOB data for %s: %s", market_id, e)
            # Return without price data - we'll update later
            return VenueMarket(
                venue="polymarket",
                market_id=market_id,
                token_id_yes=yes_id,
                token_id_no=no_id,
                fees=self.fee_schedule,
                last_updated=datetime.now(UTC),
            )

    def _parse_market_from_page(self, data: dict[str, Any]) -> VenueMarket | None:
        """Parse a market from page data (site scraping)."""
        token_ids = data.get("clobTokenIds", [])
        if len(token_ids) != 2:
            return None

        market_id = str(data.get("id", ""))
        if not market_id:
            return None

        yes_id = str(token_ids[0])
        no_id = str(token_ids[1])

        # Try to get orderbook data if available in the page data
        books = data.get("books", {})
        yes_book = books.get("yes", {})
        no_book = books.get("no", {})

        yes_bids = yes_book.get("bids", [])
        yes_asks = yes_book.get("asks", [])
        no_bids = no_book.get("bids", [])
        no_asks = no_book.get("asks", [])

        yes_bid = float(yes_bids[0]["price"]) if yes_bids else None
        yes_ask = float(yes_asks[0]["price"]) if yes_asks else None
        no_bid = float(no_bids[0]["price"]) if no_bids else None
        no_ask = float(no_asks[0]["price"]) if no_asks else None

        yes_price = (yes_bid + yes_ask) / 2 if yes_bid and yes_ask else yes_bid or yes_ask
        no_price = (no_bid + no_ask) / 2 if no_bid and no_ask else no_bid or no_ask

        return VenueMarket(
            venue="polymarket",
            market_id=market_id,
            token_id_yes=yes_id,
            token_id_no=no_id,
            yes_price=yes_price,
            no_price=no_price,
            yes_ask=yes_ask,
            no_ask=no_ask,
            yes_bid=yes_bid,
            no_bid=no_bid,
            volume_24h=float(data.get("volume24h", 0) or 0),
            liquidity=0.0,  # Would need to calculate from book
            fees=self.fee_schedule,
            last_updated=datetime.now(UTC),
        )

    def get_market(self, market_id: str) -> VenueMarket | None:
        """Fetch a specific market by ID.

        Args:
            market_id: Polymarket market ID

        Returns:
            VenueMarket or None if not found
        """
        try:
            resp = self._http.get(f"{GAMMA_API_BASE}/markets/{market_id}")
            resp.raise_for_status()
            data = resp.json()
            return self._parse_market(data)
        except Exception as e:
            logger.exception("Error fetching market %s: %s", market_id, e)
            return None

    def close(self) -> None:
        """Close the HTTP client."""
        self._http.close()

    def __enter__(self) -> PolymarketClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
