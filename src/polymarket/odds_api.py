"""Odds API client for fetching sharp sportsbook lines.

Supports Pinnacle, Betfair Exchange, and aggregated odds via OddsAPI.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

import httpx

# Free tier API: https://the-odds-api.com
ODDS_API_BASE = "https://api.the-odds-api.com/v4"


@dataclass(frozen=True)
class BookOdds:
    """Odds from a single book for a specific outcome.

    Attributes:
        book_key: Book identifier (pinnacle, betfair_exchange, etc.)
        book_title: Human-readable book name
        price: American odds (+150, -200, etc.)
        decimal_odds: Decimal odds (2.5, 1.5, etc.)
        implied_prob: Implied probability (0.0-1.0)
        last_update: Timestamp of last update
    """

    book_key: str
    book_title: str
    price: int
    decimal_odds: Decimal
    implied_prob: Decimal
    last_update: str

    @classmethod
    def from_api(cls, book_key: str, book_title: str, price: int, last_update: str) -> BookOdds:
        """Create BookOdds from API response data."""
        decimal = cls._american_to_decimal(price)
        implied = cls._decimal_to_implied(decimal)
        return cls(
            book_key=book_key,
            book_title=book_title,
            price=price,
            decimal_odds=decimal,
            implied_prob=implied,
            last_update=last_update,
        )

    @staticmethod
    def _american_to_decimal(american: int) -> Decimal:
        """Convert American odds to decimal odds."""
        if american > 0:
            return Decimal(str(american)) / Decimal("100") + Decimal("1")
        return Decimal("100") / Decimal(str(abs(american))) + Decimal("1")

    @staticmethod
    def _decimal_to_implied(decimal: Decimal) -> Decimal:
        """Convert decimal odds to implied probability."""
        return Decimal("1") / decimal


@dataclass(frozen=True)
class MarketOutcome:
    """A single outcome in a betting market.

    Attributes:
        name: Outcome name (e.g., "Yes", "Over", team name)
        odds: List of odds from different books
        best_odds: Best available decimal odds
        best_implied: Best implied probability
    """

    name: str
    odds: list[BookOdds]

    @property
    def best_odds(self) -> BookOdds | None:
        """Return the best (highest) odds available."""
        if not self.odds:
            return None
        return max(self.odds, key=lambda o: o.decimal_odds)

    @property
    def best_sharp_prob(self) -> Decimal | None:
        """Get best implied probability from sharp books only.

        Prioritizes Pinnacle and Betfair Exchange as sharpest sources.
        """
        sharp_keys = {"pinnacle", "betfair_exchange", "betfair", "circa"}
        sharp_odds = [o for o in self.odds if o.book_key in sharp_keys]
        if sharp_odds:
            best = max(sharp_odds, key=lambda o: o.decimal_odds)
            return best.implied_prob
        # Fallback to best overall if no sharp book
        return self.best_odds.implied_prob if self.best_odds else None


@dataclass(frozen=True)
class SportsMarket:
    """A sports betting market from odds API.

    Attributes:
        id: Market ID from API
        sport_key: Sport identifier
        sport_title: Human-readable sport name
        home_team: Home team name
        away_team: Away team name
        commence_time: Event start time (ISO format)
        market_key: Market type (h2h, spreads, totals)
        outcomes: List of possible outcomes
    """

    id: str
    sport_key: str
    sport_title: str
    home_team: str
    away_team: str
    commence_time: str
    market_key: str
    outcomes: list[MarketOutcome]

    @property
    def is_sports(self) -> bool:
        """Check if this is a sports market (not elections/weather/crypto)."""
        non_sports = {"elections", "politics", "weather", "crypto", "bitcoin", "ethereum"}
        return self.sport_key not in non_sports and not any(
            x in self.sport_key for x in non_sports
        )

    def get_outcome(self, name: str) -> MarketOutcome | None:
        """Get outcome by name (case-insensitive)."""
        name_lower = name.lower()
        for outcome in self.outcomes:
            if outcome.name.lower() == name_lower:
                return outcome
        return None


class OddsApiClient:
    """Client for The Odds API (aggregates Pinnacle, Betfair, etc.).

    Free tier: 500 requests/month
    """

    def __init__(self, api_key: str | None = None) -> None:
        """Initialize client.

        Args:
            api_key: API key from https://the-odds-api.com
                     Loads from ODDS_API_KEY env var if not provided.
        """
        self.api_key = api_key or self._load_key_from_env()

    @staticmethod
    def _load_key_from_env() -> str:
        """Load API key from environment."""
        import os

        key = os.getenv("ODDS_API_KEY")
        if not key:
            msg = "ODDS_API_KEY not set. Get free key at https://the-odds-api.com"
            raise ValueError(msg)
        return key

    def _client(self, timeout: float = 30.0) -> httpx.Client:
        """Create HTTP client."""
        return httpx.Client(timeout=timeout, headers={"User-Agent": "polymarket-bot/0.1"})

    def get_sports(self) -> list[dict]:
        """Get list of available sports.

        Returns:
            List of sport dictionaries with keys, titles, etc.
        """
        with self._client() as c:
            r = c.get(
                f"{ODDS_API_BASE}/sports",
                params={"apiKey": self.api_key},
            )
            r.raise_for_status()
            return r.json()

    def get_odds(
        self,
        sport: str,
        regions: str = "us,eu",
        markets: str = "h2h",
        odds_format: str = "american",
    ) -> list[SportsMarket]:
        """Fetch odds for a sport.

        Args:
            sport: Sport key (e.g., "americanfootball_nfl", "basketball_nba")
            regions: Comma-separated regions (us, eu, uk, au)
            markets: Market types (h2h, spreads, totals)
            odds_format: american, decimal, or fractional

        Returns:
            List of SportsMarket objects.
        """
        with self._client() as c:
            r = c.get(
                f"{ODDS_API_BASE}/sports/{sport}/odds",
                params={
                    "apiKey": self.api_key,
                    "regions": regions,
                    "markets": markets,
                    "oddsFormat": odds_format,
                },
            )
            r.raise_for_status()
            data = r.json()

        return [self._parse_market(m) for m in data]

    def _parse_market(self, data: dict) -> SportsMarket:
        """Parse API response into SportsMarket."""
        outcomes: list[MarketOutcome] = []

        for book_data in data.get("bookmakers", []):
            book_key = book_data.get("key", "")
            book_title = book_data.get("title", "")

            for market in book_data.get("markets", []):
                for outcome_data in market.get("outcomes", []):
                    outcome_name = outcome_data.get("name", "")
                    price = outcome_data.get("price", 0)
                    last_update = book_data.get("last_update", "")

                    # Find or create outcome
                    existing = next(
                        (o for o in outcomes if o.name == outcome_name),
                        None,
                    )
                    if existing is None:
                        existing = MarketOutcome(name=outcome_name, odds=[])
                        outcomes.append(existing)

                    # Add odds to outcome (we'll rebuild properly below)
                    book_odds = BookOdds.from_api(
                        book_key=book_key,
                        book_title=book_title,
                        price=price,
                        last_update=last_update,
                    )
                    existing.odds.append(book_odds)

        return SportsMarket(
            id=data.get("id", ""),
            sport_key=data.get("sport_key", ""),
            sport_title=data.get("sport_title", ""),
            home_team=data.get("home_team", ""),
            away_team=data.get("away_team", ""),
            commence_time=data.get("commence_time", ""),
            market_key="h2h",  # Default, could parse from data
            outcomes=outcomes,
        )

    def get_pinnacle_odds(self, sport: str) -> list[SportsMarket]:
        """Get odds filtered to Pinnacle only (sharpest US-allowed book).

        Args:
            sport: Sport key

        Returns:
            Markets with only Pinnacle odds.
        """
        markets = self.get_odds(sport, regions="us")
        filtered: list[SportsMarket] = []

        for market in markets:
            filtered_outcomes: list[MarketOutcome] = []
            for outcome in market.outcomes:
                pinnacle_odds = [o for o in outcome.odds if o.book_key == "pinnacle"]
                if pinnacle_odds:
                    filtered_outcomes.append(
                        MarketOutcome(name=outcome.name, odds=pinnacle_odds)
                    )

            if filtered_outcomes:
                filtered.append(
                    SportsMarket(
                        id=market.id,
                        sport_key=market.sport_key,
                        sport_title=market.sport_title,
                        home_team=market.home_team,
                        away_team=market.away_team,
                        commence_time=market.commence_time,
                        market_key=market.market_key,
                        outcomes=filtered_outcomes,
                    )
                )

        return filtered


def get_sharp_implied_prob(market: SportsMarket, outcome_name: str) -> Decimal | None:
    """Get the sharp book implied probability for an outcome.

    Prioritizes Pinnacle, then Betfair Exchange, then best overall.

    Args:
        market: Sports market from odds API
        outcome_name: Name of outcome (team name, "Yes", etc.)

    Returns:
        Implied probability as Decimal, or None if not found.
    """
    outcome = market.get_outcome(outcome_name)
    if not outcome:
        return None
    return outcome.best_sharp_prob
