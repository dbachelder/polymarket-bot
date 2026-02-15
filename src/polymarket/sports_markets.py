"""Polymarket sports market identification and filtering.

Identifies sports markets from Gamma API responses and maps them to
standardized sport/outcome identifiers for cross-platform matching.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal

# Sport keywords for classification
SPORT_KEYWORDS = {
    "americanfootball_nfl": [
        "nfl", "super bowl", "american football", "football game",
        "touchdown", "quarterback", "patriots", "cowboys", "chiefs"
    ],
    "basketball_nba": [
        "nba", "basketball", "lakers", "celtics", "warriors", "bucks",
        "nuggets", "heat", "knicks", "76ers"
    ],
    "basketball_ncaa": [
        "ncaa basketball", "college basketball", "march madness"
    ],
    "baseball_mlb": [
        "mlb", "baseball", "yankees", "dodgers", "red sox", "cubs",
        "astros", "braves", "home run", "pitcher"
    ],
    "soccer_usa_mls": [
        "mls", "inter miami", "la galaxy", "messi"
    ],
    "soccer_epl": [
        "premier league", "epl", "manchester united", "liverpool",
        "arsenal", "chelsea", "manchester city", "tottenham"
    ],
    "soccer_uefa": [
        "champions league", "europa league", "uefa", "real madrid",
        "barcelona", "bayern", "psg"
    ],
    "soccer_fifa": [
        "world cup", "fifa", "international soccer"
    ],
    "icehockey_nhl": [
        "nhl", "hockey", "rangers", "bruins", "maple leafs", "oilers",
        "avalanche", "lightning"
    ],
    "tennis": [
        "tennis", "wimbledon", "us open", "french open", "australian open",
        "djokovic", "nadal", "alcaraz", "sinner"
    ],
    "golf": [
        "golf", "masters", "pga", "the open", "augusta", "pga championship"
    ],
    "mma": [
        "ufc", "mma", "boxing", "title fight", "championship fight"
    ],
}

# Non-sports categories to exclude
NON_SPORTS_KEYWORDS = [
    "election", "president", "senate", "house", "congress",
    "crypto", "bitcoin", "ethereum", "btc", "eth",
    "weather", "temperature", "rain", "snow",
    "oscar", "emmy", "grammy", "award",
    "inflation", "gdp", "unemployment", "fed", "federal reserve",
    "trump", "biden", "harris", "democrat", "republican",
]


@dataclass(frozen=True)
class PolymarketSportMarket:
    """A sports market from Polymarket Gamma API.

    Attributes:
        market_id: Polymarket market ID
        slug: Market slug/URL identifier
        question: Market question text
        description: Market description
        sport_key: Mapped sport key (e.g., "americanfootball_nfl")
        sport_title: Human-readable sport name
        event_name: Parsed event name (e.g., "Chiefs vs Eagles")
        outcome_yes: What "Yes" means in this market
        outcome_no: What "No" means in this market
        volume: Total market volume
        yes_price: Current YES token price
        no_price: Current NO token price
        yes_token_id: CLOB token ID for YES
        no_token_id: CLOB token ID for NO
        end_date: Market resolution date
        tags: List of market tags
    """

    market_id: str
    slug: str
    question: str
    description: str | None
    sport_key: str | None
    sport_title: str | None
    event_name: str | None
    outcome_yes: str
    outcome_no: str
    volume: Decimal
    yes_price: Decimal
    no_price: Decimal
    yes_token_id: str
    no_token_id: str
    end_date: str | None
    tags: list[str]

    @property
    def implied_yes_prob(self) -> Decimal:
        """Calculate implied probability from YES price."""
        return self.yes_price

    @property
    def implied_no_prob(self) -> Decimal:
        """Calculate implied probability from NO price."""
        return self.no_price

    @property
    def is_valid_sports(self) -> bool:
        """Check if this is a valid sports market (not elections/crypto/weather)."""
        if self.sport_key is None:
            return False

        question_lower = self.question.lower()
        for keyword in NON_SPORTS_KEYWORDS:
            if keyword in question_lower:
                return False

        return True

    @property
    def has_liquidity(self) -> bool:
        """Check if market has sufficient liquidity (> $10k)."""
        return self.volume >= Decimal("10000")


def classify_sport(question: str, description: str | None = None, tags: list[str] | None = None) -> tuple[str | None, str | None]:
    """Classify a market question into a sport category.

    Args:
        question: Market question text
        description: Market description
        tags: Market tags

    Returns:
        Tuple of (sport_key, sport_title) or (None, None) if not sports.
    """
    text = question.lower()
    if description:
        text += " " + description.lower()
    if tags:
        text += " " + " ".join(t.lower() for t in tags)

    # Check for non-sports first
    for keyword in NON_SPORTS_KEYWORDS:
        if keyword in text:
            return None, None

    # Check sport keywords (most specific first)
    for sport_key, keywords in SPORT_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text:
                # Map key to title
                title_map = {
                    "americanfootball_nfl": "NFL",
                    "basketball_nba": "NBA",
                    "basketball_ncaa": "NCAA Basketball",
                    "baseball_mlb": "MLB",
                    "soccer_usa_mls": "MLS",
                    "soccer_epl": "Premier League",
                    "soccer_uefa": "UEFA Champions League",
                    "soccer_fifa": "FIFA World Cup",
                    "icehockey_nhl": "NHL",
                    "tennis": "Tennis",
                    "golf": "Golf",
                    "mma": "MMA/UFC",
                }
                return sport_key, title_map.get(sport_key, sport_key)

    return None, None


def parse_event_name(question: str) -> str | None:
    """Try to extract event name (Team A vs Team B) from question.

    Args:
        question: Market question

    Returns:
        Extracted event name or None.
    """
    # Common patterns
    patterns = [
        # "Will Team A beat Team B?"
        r"will\s+([\w\s]+?)\s+(?:beat|defeat|win against|cover against)\s+([\w\s]+?)\?",
        # "Team A vs Team B"
        r"([\w\s]+?)\s+vs\.?\s+([\w\s]+?)(?:\?|$)",
        # "Team A @ Team B"
        r"([\w\s]+?)\s+@\s+([\w\s]+?)(?:\?|$)",
    ]

    for pattern in patterns:
        match = re.search(pattern, question, re.IGNORECASE)
        if match:
            team1 = match.group(1).strip()
            team2 = match.group(2).strip()
            return f"{team1} vs {team2}"

    return None


def extract_outcomes(question: str) -> tuple[str, str]:
    """Extract what YES and NO mean from a binary sports question.

    Args:
        question: Binary sports question

    Returns:
        Tuple of (yes_outcome, no_outcome) descriptions.
    """
    question_lower = question.lower()

    # Pattern: "Will X happen?"
    if question_lower.startswith("will "):
        event = question[5:].rstrip("?").strip()
        return f"Yes - {event}", f"No - {event} does not happen"

    # Pattern: "Team A vs Team B"
    event_name = parse_event_name(question)
    if event_name:
        return f"Yes - {event_name}", f"No - {event_name} does not occur"

    # Default
    return "Yes", "No"


def from_gamma_market(data: dict) -> PolymarketSportMarket | None:
    """Parse a Gamma API market into PolymarketSportMarket.

    Args:
        data: Raw market data from Gamma API

    Returns:
        PolymarketSportMarket or None if not a sports market.
    """
    question = data.get("question", "")
    description = data.get("description")
    tags = data.get("tags", [])

    # Classify sport
    sport_key, sport_title = classify_sport(question, description, tags)

    # Parse outcomes
    yes_outcome, no_outcome = extract_outcomes(question)

    # Get token IDs from outcome prices
    outcome_prices = data.get("outcomePrices", [])
    clob_token_ids = data.get("clobTokenIds", [])

    yes_token_id = clob_token_ids[0] if len(clob_token_ids) > 0 else ""
    no_token_id = clob_token_ids[1] if len(clob_token_ids) > 1 else ""

    # Parse prices
    try:
        yes_price = Decimal(str(outcome_prices[0])) if len(outcome_prices) > 0 else Decimal("0.5")
        no_price = Decimal(str(outcome_prices[1])) if len(outcome_prices) > 1 else Decimal("0.5")
    except (ValueError, TypeError):
        yes_price = Decimal("0.5")
        no_price = Decimal("0.5")

    # Parse volume
    try:
        volume = Decimal(str(data.get("volume", "0")))
    except (ValueError, TypeError):
        volume = Decimal("0")

    return PolymarketSportMarket(
        market_id=str(data.get("id", "")),
        slug=data.get("slug", ""),
        question=question,
        description=description,
        sport_key=sport_key,
        sport_title=sport_title,
        event_name=parse_event_name(question),
        outcome_yes=yes_outcome,
        outcome_no=no_outcome,
        volume=volume,
        yes_price=yes_price,
        no_price=no_price,
        yes_token_id=yes_token_id,
        no_token_id=no_token_id,
        end_date=data.get("endDate"),
        tags=tags,
    )


def filter_sports_markets(markets: list[dict]) -> list[PolymarketSportMarket]:
    """Filter Gamma API markets to valid sports markets only.

    Args:
        markets: Raw markets from Gamma API

    Returns:
        List of valid sports markets with liquidity.
    """
    results: list[PolymarketSportMarket] = []

    for data in markets:
        try:
            market = from_gamma_market(data)
            if market and market.is_valid_sports and market.has_liquidity:
                results.append(market)
        except Exception:
            # Skip malformed markets
            continue

    return results


def get_sports_markets_from_gamma(
    active: bool = True,
    limit: int = 100,
) -> list[PolymarketSportMarket]:
    """Fetch and filter sports markets from Gamma API.

    Args:
        active: Only fetch active markets
        limit: Maximum markets to fetch

    Returns:
        List of valid sports markets.
    """
    from .gamma import get_markets

    all_markets = get_markets(active=active, limit=limit)
    return filter_sports_markets(all_markets)
