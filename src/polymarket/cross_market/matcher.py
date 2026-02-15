"""Event matcher for cross-market arbitrage.

Matches equivalent events across prediction market venues using
title normalization and fuzzy string matching.
"""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime
from typing import Any

from . import CrossMarketEvent, VenueMarket

logger = logging.getLogger(__name__)


class EventNormalizer:
    """Normalizes event titles and metadata for cross-venue matching."""

    # Common words to remove during normalization
    STOP_WORDS = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "as", "is", "was", "will", "be", "this",
        "that", "these", "those", "it", "its", "market", "prediction", "bet",
    }

    # Category patterns
    CATEGORY_PATTERNS = {
        "politics": [
            "election", "president", "senate", "house", "congress", "vote",
            "ballot", "trump", "biden", "democrat", "republican", "gop",
            "midterm", "primary", "nomination", "candidate", "poll",
        ],
        "crypto": [
            "bitcoin", "btc", "ethereum", "eth", "crypto", "cryptocurrency",
            "blockchain", "solana", "sol", "cardano", "ada", "binance", "coinbase",
        ],
        "sports": [
            "super bowl", "world cup", "nba", "nfl", "mlb", "nhl", "soccer",
            "football", "basketball", "baseball", "hockey", "tennis", "golf",
            "olympics", "championship", "tournament", "playoff", "final",
        ],
        "finance": [
            "fed", "federal reserve", "interest rate", "cpi", "inflation",
            "recession", "gdp", "unemployment", "jobs report", "nasdaq",
            "sp500", "s&p", "stock market", "etf",
        ],
        "weather": [
            "temperature", "rain", "snow", "hurricane", "storm", "weather",
            "forecast", "degrees", "fahrenheit", "celsius",
        ],
    }

    def normalize_title(self, title: str) -> str:
        """Normalize an event title for matching.

        Steps:
        1. Lowercase
        2. Remove punctuation
        3. Remove extra whitespace
        4. Remove stop words
        5. Sort remaining words for consistency

        Args:
            title: Raw event title

        Returns:
            Normalized title
        """
        # Lowercase
        text = title.lower()

        # Remove punctuation except hyphens
        text = re.sub(r"[^\w\s-]", " ", text)

        # Replace multiple spaces with single space
        text = re.sub(r"\s+", " ", text)

        # Split into words
        words = text.split()

        # Remove stop words and short words
        words = [w for w in words if w not in self.STOP_WORDS and len(w) > 2]

        # Sort for consistency
        words.sort()

        return " ".join(words)

    def extract_key_entities(self, title: str) -> set[str]:
        """Extract key entities from title for matching.

        Args:
            title: Raw event title

        Returns:
            Set of key entity strings
        """
        normalized = self.normalize_title(title)
        words = normalized.split()

        # Extract multi-word entities (names, tickers, etc.)
        entities: set[str] = set()

        # Look for known patterns
        patterns = [
            r"\b(btc|eth|sol|ada|xrp|doge)\b",  # Crypto tickers
            r"\b(trump|biden|musk)\b",  # People
            r"\b(\d{4})\b",  # Years
            r"\b(above|below)\s+(\d+)\b",  # Thresholds
        ]

        for pattern in patterns:
            matches = re.findall(pattern, title.lower())
            for match in matches:
                if isinstance(match, tuple):
                    entities.add(" ".join(match))
                else:
                    entities.add(match)

        # Add significant words (longer words are more distinctive)
        for word in words:
            if len(word) > 4:
                entities.add(word)

        return entities

    def detect_category(self, title: str) -> str:
        """Detect event category from title.

        Args:
            title: Event title

        Returns:
            Category string
        """
        title_lower = title.lower()

        scores: dict[str, int] = {}
        for category, patterns in self.CATEGORY_PATTERNS.items():
            score = sum(1 for p in patterns if p in title_lower)
            if score > 0:
                scores[category] = score

        if not scores:
            return "other"

        return max(scores, key=scores.get)

    def generate_event_id(self, normalized_title: str, resolution_date: datetime | None) -> str:
        """Generate a stable event ID from normalized title and date.

        Args:
            normalized_title: Normalized event title
            resolution_date: Event resolution date

        Returns:
            Stable event ID (first 16 chars of MD5 hash)
        """
        # Combine title and date for uniqueness
        date_str = resolution_date.strftime("%Y%m%d") if resolution_date else ""
        combined = f"{normalized_title}|{date_str}"

        # Generate hash
        hash_bytes = hashlib.md5(combined.encode()).digest()  # noqa: S324
        return hash_bytes.hex()[:16]


class EventMatcher:
    """Matches equivalent events across prediction market venues."""

    def __init__(self, similarity_threshold: float = 0.7) -> None:
        """Initialize matcher.

        Args:
            similarity_threshold: Minimum similarity score for match (0-1)
        """
        self.normalizer = EventNormalizer()
        self.similarity_threshold = similarity_threshold

    def calculate_similarity(self, title1: str, title2: str) -> float:
        """Calculate similarity between two event titles.

        Uses Jaccard similarity on key entities.

        Args:
            title1: First title
            title2: Second title

        Returns:
            Similarity score (0-1)
        """
        entities1 = self.normalizer.extract_key_entities(title1)
        entities2 = self.normalizer.extract_key_entities(title2)

        if not entities1 or not entities2:
            return 0.0

        # Jaccard similarity
        intersection = len(entities1 & entities2)
        union = len(entities1 | entities2)

        if union == 0:
            return 0.0

        return intersection / union

    def match_events(
        self,
        venues_data: dict[str, list[VenueMarket]],
    ) -> list[CrossMarketEvent]:
        """Match equivalent events across venues.

        Args:
            venues_data: Dict mapping venue name -> list of VenueMarket

        Returns:
            List of CrossMarketEvent with matched venues
        """
        # Build candidate matches
        candidates: dict[str, dict[str, Any]] = {}

        for venue, markets in venues_data.items():
            for market in markets:
                # We need the market title - this would come from additional metadata
                # For now, we'll create a simple key from market_id
                # In production, this would use actual market titles/descriptions

                # Normalize
                normalized = self.normalizer.normalize_title(market.market_id)
                category = "unknown"  # Would extract from actual title

                # Generate event ID
                event_id = self.normalizer.generate_event_id(normalized, None)

                if event_id not in candidates:
                    candidates[event_id] = {
                        "normalized_title": normalized,
                        "category": category,
                        "venues": {},
                    }

                candidates[event_id]["venues"][venue] = market

        # Filter to events with multiple venues (potential arbitrage)
        matched: list[CrossMarketEvent] = []

        for event_id, data in candidates.items():
            if len(data["venues"]) >= 2:
                # Create CrossMarketEvent
                event = CrossMarketEvent(
                    event_id=event_id,
                    title=data["normalized_title"],
                    description="",  # Would populate from source
                    category=data["category"],
                    resolution_date=None,
                    resolution_source="",
                    venues=data["venues"],
                )
                matched.append(event)

        logger.info("Matched %d cross-venue events from %d candidates", len(matched), len(candidates))
        return matched

    def find_similar_events(
        self,
        reference_title: str,
        candidate_markets: list[VenueMarket],
        min_similarity: float | None = None,
    ) -> list[tuple[VenueMarket, float]]:
        """Find events similar to a reference title.

        Args:
            reference_title: Title to match against
            candidate_markets: Markets to search
            min_similarity: Minimum similarity threshold

        Returns:
            List of (market, similarity) tuples sorted by similarity
        """
        threshold = min_similarity or self.similarity_threshold

        matches: list[tuple[VenueMarket, float]] = []

        for market in candidate_markets:
            # We'd need the actual title here
            # For now, use market_id as proxy
            similarity = self.calculate_similarity(reference_title, market.market_id)

            if similarity >= threshold:
                matches.append((market, similarity))

        # Sort by similarity descending
        matches.sort(key=lambda x: x[1], reverse=True)

        return matches


def match_events_by_normalized_title(
    polymarket_markets: list[VenueMarket],
    kalshi_markets: list[VenueMarket],
    title_map: dict[str, str] | None = None,
) -> list[CrossMarketEvent]:
    """Match events using explicit title mapping.

    This is a simplified matcher that uses a pre-built title mapping
    for known equivalent markets.

    Args:
        polymarket_markets: Markets from Polymarket
        kalshi_markets: Markets from Kalshi
        title_map: Optional dict mapping Polymarket titles to Kalshi titles

    Returns:
        List of matched CrossMarketEvent
    """
    matcher = EventMatcher()
    events: list[CrossMarketEvent] = []

    # Build lookup by normalized title
    kalshi_by_title: dict[str, VenueMarket] = {}
    for m in kalshi_markets:
        normalized = matcher.normalizer.normalize_title(m.market_id)
        kalshi_by_title[normalized] = m

    # Match Polymarket markets
    for pm_market in polymarket_markets:
        pm_normalized = matcher.normalizer.normalize_title(pm_market.market_id)

        # Check for exact match
        if pm_normalized in kalshi_by_title:
            kalshi_market = kalshi_by_title[pm_normalized]

            event_id = matcher.normalizer.generate_event_id(pm_normalized, None)

            event = CrossMarketEvent(
                event_id=event_id,
                title=pm_normalized,
                description="",
                category="unknown",
                resolution_date=None,
                resolution_source="",
                venues={
                    "polymarket": pm_market,
                    "kalshi": kalshi_market,
                },
            )
            events.append(event)

    logger.info("Matched %d events by normalized title", len(events))
    return events
