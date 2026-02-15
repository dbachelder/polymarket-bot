"""News-Driven Momentum Trading Strategy.

Hypothesis: Breaking news creates 2-15 minute information windows where prediction
markets under-react before retail catches up. By monitoring primary news sources
and rapidly interpreting impact, we can enter positions at favorable odds and exit
into the subsequent momentum wave—capturing 5-20% price moves without holding to
settlement.

Inspiration:
- Car (@CarOnPolymarket) — $850K profit using news-driven discretionary trading
- Pattern: Buy on breaking news when market hasn't priced impact → ride viral spread
  → exit at peak sentiment before event resolves
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class NewsCategory(Enum):
    """Categories of news that can impact prediction markets."""

    POLITICS = "politics"
    CRYPTO = "crypto"
    SPORTS = "sports"
    POP_CULTURE = "pop_culture"
    MACRO = "macro"
    REGULATORY = "regulatory"
    UNKNOWN = "unknown"


class SourceReliability(Enum):
    """Reliability tiers for news sources."""

    VERIFIED = 1.0  # Official accounts, government sources
    MAJOR_OUTLET = 0.8  # Reuters, AP, Bloomberg, WSJ, etc.
    ESTABLISHED = 0.6  # Major crypto news, sports outlets
    AGGREGATOR = 0.4  # News aggregators, newsletters
    RUMOR = 0.2  # Unverified sources


class ImpactDirection(Enum):
    """Directional impact of news on a market outcome."""

    POSITIVE = "positive"  # News increases likelihood of YES
    NEGATIVE = "negative"  # News increases likelihood of NO
    NEUTRAL = "neutral"  # No clear directional impact
    UNCLEAR = "unclear"  # Need more information


# Keywords for categorizing news
CATEGORY_KEYWORDS: dict[NewsCategory, list[str]] = {
    NewsCategory.POLITICS: [
        "election",
        "vote",
        "ballot",
        "poll",
        "campaign",
        "candidate",
        "president",
        "congress",
        "senate",
        "house",
        "legislation",
        "policy",
        "bill",
        "law",
        "regulation",
        "executive order",
        "debate",
        "primary",
        "caucus",
        "impeachment",
        "resign",
        "trump",
        "biden",
        "harris",
        "musk",
        "fed",
        "federal reserve",
        "white house",
        "capitol",
        "supreme court",
        "scotus",
    ],
    NewsCategory.CRYPTO: [
        "bitcoin",
        "btc",
        "ethereum",
        "eth",
        "solana",
        "sol",
        "crypto",
        "cryptocurrency",
        "blockchain",
        "defi",
        "nft",
        "exchange",
        "binance",
        "coinbase",
        "listing",
        "delisting",
        "sec",
        "regulation",
        "etf",
        "halving",
        "mining",
        "wallet",
        "hack",
        "exploit",
        "bridge",
        "protocol",
        "token",
    ],
    NewsCategory.SPORTS: [
        "game",
        "match",
        "score",
        "win",
        "loss",
        "victory",
        "championship",
        "playoff",
        "final",
        "quarterfinal",
        "semifinal",
        "injury",
        "injured",
        "out",
        "returning",
        "lineup",
        "starter",
        "quarterback",
        "pitcher",
        "goal",
        "touchdown",
        "home run",
        "nba",
        "nfl",
        "mlb",
        "nhl",
        "fifa",
        "world cup",
        "olympics",
    ],
    NewsCategory.POP_CULTURE: [
        "album",
        "single",
        "release",
        "drop",
        "stream",
        "movie",
        "film",
        "premiere",
        "box office",
        "trailer",
        "celebrity",
        "divorce",
        "marriage",
        "baby",
        "pregnant",
        "award",
        "grammy",
        "oscar",
        "emmy",
        "nominated",
        "gta",
        "grand theft auto",
        "video game",
        "taylor swift",
        "kanye",
        "album",
        "concert",
        "tour",
        "ticket",
    ],
    NewsCategory.MACRO: [
        "gdp",
        "inflation",
        "cpi",
        "ppi",
        "employment",
        "jobs",
        "unemployment",
        "nfp",
        "nonfarm payrolls",
        "recession",
        "interest rate",
        "rate hike",
        "rate cut",
        "fomc",
        "fed",
        "treasury",
        "yield",
        "bond",
        "dollar",
        "eurusd",
    ],
    NewsCategory.REGULATORY: [
        "sec",
        "cftc",
        "fda",
        "approval",
        "denied",
        "license",
        "regulation",
        "regulated",
        "compliance",
        "fine",
        "settlement",
        "lawsuit",
        "sued",
        "legal",
        "court",
        "ruling",
        "appeal",
    ],
}

# Market patterns to match news against
MARKET_PATTERNS: dict[NewsCategory, list[tuple[str, ImpactDirection]]] = {
    NewsCategory.POLITICS: [
        (r"trump.*win", ImpactDirection.POSITIVE),
        (r"biden.*win", ImpactDirection.POSITIVE),
        (r"harris.*win", ImpactDirection.POSITIVE),
        (r"election.*delay", ImpactDirection.NEGATIVE),
        (r"congress.*pass", ImpactDirection.POSITIVE),
        (r"veto", ImpactDirection.NEGATIVE),
        (r"resign", ImpactDirection.POSITIVE),
        (r"impeach", ImpactDirection.POSITIVE),
    ],
    NewsCategory.CRYPTO: [
        (r"bitcoin.*etf.*approve", ImpactDirection.POSITIVE),
        (r"ethereum.*etf.*approve", ImpactDirection.POSITIVE),
        (r"sec.*approve", ImpactDirection.POSITIVE),
        (r"sec.*deny", ImpactDirection.NEGATIVE),
        (r"exchange.*hack", ImpactDirection.NEGATIVE),
        (r"protocol.*exploit", ImpactDirection.NEGATIVE),
        (r"listing", ImpactDirection.POSITIVE),
        (r"delisting", ImpactDirection.NEGATIVE),
    ],
    NewsCategory.SPORTS: [
        (r"injury", ImpactDirection.NEGATIVE),
        (r"injured", ImpactDirection.NEGATIVE),
        (r"out.*game", ImpactDirection.NEGATIVE),
        (r"returning", ImpactDirection.POSITIVE),
        (r"win", ImpactDirection.POSITIVE),
        (r"lose", ImpactDirection.NEGATIVE),
    ],
    NewsCategory.POP_CULTURE: [
        (r"release.*date", ImpactDirection.POSITIVE),
        (r"delayed", ImpactDirection.NEGATIVE),
        (r"cancel", ImpactDirection.NEGATIVE),
        (r"announce", ImpactDirection.POSITIVE),
    ],
}


@dataclass(frozen=True)
class NewsItem:
    """A news item from any source."""

    timestamp: datetime
    source: str
    source_reliability: SourceReliability
    headline: str
    url: str | None = None
    category: NewsCategory = NewsCategory.UNKNOWN
    content: str | None = None

    @property
    def full_text(self) -> str:
        """Full text for analysis (headline + content)."""
        if self.content:
            return f"{self.headline} {self.content}"
        return self.headline


@dataclass(frozen=True)
class NewsImpact:
    """Analyzed impact of news on a specific market."""

    news_item: NewsItem
    market_id: str
    market_question: str
    direction: ImpactDirection
    confidence: float  # 0-1, how confident in the direction
    price_impact_estimate: float  # Estimated fair value change (0-1)
    source_quality_score: float  # 0-1, combined reliability + freshness
    reasoning: str


@dataclass
class NewsDrivenPosition:
    """A position taken based on news impact."""

    position_id: str
    timestamp: datetime
    market_id: str
    token_id: str
    market_question: str
    side: str  # "buy_yes" or "buy_no"
    entry_price: float
    position_size: float
    news_impact: NewsImpact

    # Exit tracking
    exit_price: float | None = None
    exit_timestamp: datetime | None = None
    exit_reason: str | None = None
    pnl: float | None = None

    # Momentum tracking
    price_history: list[tuple[datetime, float]] = field(default_factory=list)
    peak_price: float | None = None

    @property
    def is_open(self) -> bool:
        return self.exit_price is None

    @property
    def holding_duration(self) -> timedelta | None:
        if self.exit_timestamp:
            return self.exit_timestamp - self.timestamp
        return None

    def record_price(self, timestamp: datetime, price: float) -> None:
        """Record a price observation for momentum tracking."""
        self.price_history.append((timestamp, price))
        if self.peak_price is None or price > self.peak_price:
            self.peak_price = price

    def check_momentum_exit(self, current_price: float) -> bool:
        """Check if momentum has turned negative (second derivative)."""
        if len(self.price_history) < 3:
            return False

        # Get last 3 prices
        recent = self.price_history[-3:]
        prices = [p for _, p in recent]

        # Calculate first differences
        d1 = prices[1] - prices[0]
        d2 = prices[2] - prices[1]

        # Momentum exit: second derivative turns negative
        return d2 < 0 and d1 > 0


@dataclass(frozen=True)
class NewsDrivenSignal:
    """Trading signal from news analysis."""

    timestamp: datetime
    market_id: str
    token_id_yes: str
    token_id_no: str
    market_question: str
    side: str  # "buy_yes", "buy_no", or "no_trade"
    current_price: float
    target_price: float  # Model-implied fair value
    edge: float  # Target - current
    confidence: float  # 0-1
    time_since_news_seconds: float
    news_source: str
    reasoning: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "market_id": self.market_id,
            "market_question": self.market_question,
            "side": self.side,
            "current_price": self.current_price,
            "target_price": self.target_price,
            "edge": self.edge,
            "confidence": self.confidence,
            "time_since_news_seconds": self.time_since_news_seconds,
            "news_source": self.news_source,
            "reasoning": self.reasoning,
        }


@dataclass(frozen=True)
class NewsDrivenTrade:
    """Executed trade based on news signal."""

    timestamp: datetime
    signal: NewsDrivenSignal
    order_result: Any
    position_size: float
    entry_price: float
    position_id: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "position_id": self.position_id,
            "signal": self.signal.to_dict(),
            "order_result": {
                "success": self.order_result.success,
                "dry_run": self.order_result.dry_run,
                "message": self.order_result.message,
                "order_id": self.order_result.order_id,
            }
            if hasattr(self.order_result, "success")
            else str(self.order_result),
            "position_size": self.position_size,
            "entry_price": self.entry_price,
        }


# Strategy configuration
DEFAULT_CONFIG = {
    # Entry thresholds
    "max_time_since_news_seconds": 120,  # Must enter within 2 minutes
    "min_edge_for_entry": 0.05,  # 5% edge minimum
    "min_confidence": 0.6,  # 60% confidence minimum
    "max_price_movement": 0.20,  # Don't enter if price already moved 20%
    # Position sizing
    "base_position_size": 2.0,  # 2% of capital
    "scaled_position_size": 10.0,  # 10% when high confidence
    "max_position_size": 15.0,  # 15% hard cap
    "min_liquidity_usd": 50000,  # $50K minimum liquidity
    # Exit rules
    "max_hold_hours": 24,  # Maximum hold time
    "stop_loss_pct": 0.15,  # 15% stop loss
    "momentum_exit_enabled": True,
    "profit_target_pct": 0.20,  # Take 20% profit
    # Source filtering
    "min_source_reliability": SourceReliability.AGGREGATOR,
}


def categorize_news(headline: str, content: str | None = None) -> NewsCategory:
    """Categorize news based on keywords.

    Args:
        headline: News headline
        content: Optional news content

    Returns:
        Best matching news category
    """
    text = f"{headline} {content or ''}".lower()

    scores: dict[NewsCategory, int] = {}
    for category, keywords in CATEGORY_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text)
        if score > 0:
            scores[category] = score

    if not scores:
        return NewsCategory.UNKNOWN

    # Return category with highest score
    return max(scores, key=scores.get)


def analyze_news_impact(
    news_item: NewsItem,
    market_question: str,
    market_id: str,
) -> NewsImpact | None:
    """Analyze how a news item impacts a specific market.

    Args:
        news_item: The news item to analyze
        market_question: The market question text
        market_id: Market identifier

    Returns:
        NewsImpact if analysis successful, None otherwise
    """
    category = categorize_news(news_item.headline, news_item.content)

    if category == NewsCategory.UNKNOWN:
        return None

    # Check if news category matches market
    market_lower = market_question.lower()
    news_text = news_item.full_text.lower()

    # Extract key entities from market question
    direction = ImpactDirection.UNCLEAR
    confidence = 0.0
    price_impact = 0.0
    reasoning = "No clear impact pattern matched"

    # Check for direct entity matches
    if category == NewsCategory.POLITICS:
        if "trump" in market_lower and "trump" in news_text:
            if "win" in news_text or "victory" in news_text or "elected" in news_text:
                direction = ImpactDirection.POSITIVE
                confidence = 0.8
                price_impact = 0.15
                reasoning = "Trump victory news directly impacts Trump win markets"
            elif "lose" in news_text or "defeat" in news_text:
                direction = ImpactDirection.NEGATIVE
                confidence = 0.8
                price_impact = -0.15
                reasoning = "Trump defeat news directly impacts Trump win markets"
        elif "biden" in market_lower and "biden" in news_text:
            if "win" in news_text or "victory" in news_text:
                direction = ImpactDirection.POSITIVE
                confidence = 0.8
                price_impact = 0.15
                reasoning = "Biden victory news directly impacts Biden markets"
            elif "resign" in news_text or "drop out" in news_text:
                direction = ImpactDirection.NEGATIVE
                confidence = 0.9
                price_impact = -0.20
                reasoning = "Biden resignation directly impacts Biden-related markets"

    elif category == NewsCategory.CRYPTO:
        if "bitcoin" in market_lower or "btc" in market_lower:
            if "etf" in news_text and "approve" in news_text:
                direction = ImpactDirection.POSITIVE
                confidence = 0.9
                price_impact = 0.20
                reasoning = "Bitcoin ETF approval is major bullish catalyst"
            elif "hack" in news_text or "exploit" in news_text:
                direction = ImpactDirection.NEGATIVE
                confidence = 0.7
                price_impact = -0.10
                reasoning = "Security incident may impact crypto sentiment"

    elif category == NewsCategory.POP_CULTURE:
        if "gta" in market_lower or "grand theft auto" in market_lower:
            if "delay" in news_text or "delayed" in news_text:
                direction = ImpactDirection.NEGATIVE
                confidence = 0.85
                price_impact = -0.25
                reasoning = "GTA delay directly impacts release date markets"
            elif "fire" in news_text or "cancel" in news_text:
                direction = ImpactDirection.NEGATIVE
                confidence = 0.8
                price_impact = -0.20
                reasoning = "Development issues suggest potential delay"

    # Check pattern matches
    if direction == ImpactDirection.UNCLEAR:
        patterns = MARKET_PATTERNS.get(category, [])
        for pattern, pattern_direction in patterns:
            if re.search(pattern, news_text):
                direction = pattern_direction
                confidence = 0.6
                price_impact = 0.10 if direction == ImpactDirection.POSITIVE else -0.10
                reasoning = f"Pattern match: {pattern}"
                break

    if direction == ImpactDirection.UNCLEAR or direction == ImpactDirection.NEUTRAL:
        return None

    # Adjust confidence based on source reliability
    source_score = news_item.source_reliability.value * confidence

    return NewsImpact(
        news_item=news_item,
        market_id=market_id,
        market_question=market_question,
        direction=direction,
        confidence=confidence,
        price_impact_estimate=abs(price_impact),
        source_quality_score=source_score,
        reasoning=reasoning,
    )


def find_impacted_markets(
    news_item: NewsItem,
    markets: list[dict[str, Any]],
) -> list[NewsImpact]:
    """Find all markets potentially impacted by a news item.

    Args:
        news_item: The news item to analyze
        markets: List of market dicts from snapshot

    Returns:
        List of NewsImpact for markets that may be affected
    """
    impacts: list[NewsImpact] = []

    for market in markets:
        question = market.get("question", market.get("title", ""))
        market_id = str(market.get("market_id", market.get("id", "")))

        impact = analyze_news_impact(news_item, question, market_id)
        if impact and impact.confidence >= 0.5:
            impacts.append(impact)

    # Sort by source quality score * confidence
    impacts.sort(key=lambda x: x.source_quality_score * x.confidence, reverse=True)

    return impacts


def generate_news_signal(
    impact: NewsImpact,
    market_data: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> NewsDrivenSignal | None:
    """Generate a trading signal from news impact analysis.

    Args:
        impact: Analyzed news impact
        market_data: Market snapshot data
        config: Optional strategy configuration (merged with DEFAULT_CONFIG)

    Returns:
        NewsDrivenSignal if actionable, None otherwise
    """
    # Merge config with defaults
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    now = datetime.now(UTC)

    # Check time window
    time_since_news = (now - impact.news_item.timestamp).total_seconds()
    if time_since_news > cfg["max_time_since_news_seconds"]:
        return None

    # Check source reliability
    if impact.news_item.source_reliability.value < cfg["min_source_reliability"].value:
        return None

    # Check confidence threshold
    if impact.confidence < cfg["min_confidence"]:
        return None

    # Get market prices from book
    books = market_data.get("books", {})
    yes_book = books.get("yes", {})
    yes_bids = yes_book.get("bids", [])
    yes_asks = yes_book.get("asks", [])

    if not yes_bids or not yes_asks:
        return None

    try:
        best_bid = max(float(b["price"]) for b in yes_bids)
        best_ask = min(float(a["price"]) for a in yes_asks)
        current_yes_price = (best_bid + best_ask) / 2
    except (ValueError, KeyError):
        return None

    # Calculate target price based on impact
    if impact.direction == ImpactDirection.POSITIVE:
        target_price = min(0.95, current_yes_price + impact.price_impact_estimate)
        side = "buy_yes"
        edge = target_price - current_yes_price
    else:  # NEGATIVE
        target_price = max(0.05, current_yes_price - impact.price_impact_estimate)
        side = "buy_no"
        # Edge for NO position is the move down
        edge = current_yes_price - target_price

    # Check minimum edge
    if edge < cfg["min_edge_for_entry"]:
        return None

    # Check if price already moved too much
    price_change = abs(target_price - current_yes_price)
    if price_change > cfg["max_price_movement"]:
        return None

    # Get token IDs
    token_ids = market_data.get("clob_token_ids", market_data.get("clobTokenIds", []))
    if len(token_ids) != 2:
        return None

    token_id_yes = str(token_ids[0])
    token_id_no = str(token_ids[1])

    return NewsDrivenSignal(
        timestamp=now,
        market_id=impact.market_id,
        token_id_yes=token_id_yes,
        token_id_no=token_id_no,
        market_question=impact.market_question,
        side=side,
        current_price=current_yes_price,
        target_price=target_price,
        edge=edge,
        confidence=impact.confidence * impact.source_quality_score,
        time_since_news_seconds=time_since_news,
        news_source=impact.news_item.source,
        reasoning=impact.reasoning,
    )


def calculate_position_size(
    signal: NewsDrivenSignal,
    capital: float = 10000,
    config: dict[str, Any] | None = None,
) -> float:
    """Calculate position size based on signal quality.

    Args:
        signal: Trading signal
        capital: Available capital
        config: Strategy configuration

    Returns:
        Position size in USD
    """
    cfg = config or DEFAULT_CONFIG

    # Base position size
    base_size = cfg["base_position_size"] / 100 * capital

    # Scale based on confidence and edge
    confidence_multiplier = signal.confidence
    edge_multiplier = min(2.0, signal.edge * 10)  # Cap at 2x

    # High confidence + high edge = scaled size
    if signal.confidence >= 0.8 and signal.edge >= 0.10:
        scaled_size = cfg["scaled_position_size"] / 100 * capital
        position_size = scaled_size * confidence_multiplier
    else:
        position_size = base_size * confidence_multiplier * edge_multiplier

    # Apply hard cap
    max_size = cfg["max_position_size"] / 100 * capital
    position_size = min(position_size, max_size)

    return round(position_size, 2)


def execute_news_trade(
    signal: NewsDrivenSignal,
    capital: float = 10000,
    dry_run: bool = True,
    config: dict[str, Any] | None = None,
) -> NewsDrivenTrade | None:
    """Execute a trade based on a news-driven signal.

    Args:
        signal: News-driven trading signal
        capital: Available capital
        dry_run: If True, don't actually submit orders
        config: Strategy configuration

    Returns:
        NewsDrivenTrade if executed, None otherwise
    """
    if signal.side == "no_trade":
        return None

    now = datetime.now(UTC)
    position_id = f"news_{now.strftime('%Y%m%d%H%M%S')}_{signal.market_id[:8]}"

    # Calculate position size
    position_usd = calculate_position_size(signal, capital, config)

    # Determine order parameters
    if signal.side == "buy_yes":
        token_id = signal.token_id_yes
        side = "buy"
        # Entry at ask for aggressive fill
        entry_price = min(0.99, signal.current_price * 1.01)
    elif signal.side == "buy_no":
        token_id = signal.token_id_no
        side = "buy"
        # NO price is inverse of YES
        no_price = 1.0 - signal.current_price
        entry_price = min(0.99, no_price * 1.01)
    else:
        return None

    # Calculate position size in contracts
    position_contracts = position_usd / entry_price
    position_contracts = round(position_contracts, 2)

    try:
        from .trading import Order, submit_order

        order = Order(
            token_id=token_id,
            side=side,
            size=position_contracts,
            price=entry_price,
        )

        result = submit_order(order)

        return NewsDrivenTrade(
            timestamp=now,
            signal=signal,
            order_result=result,
            position_size=position_contracts,
            entry_price=entry_price,
            position_id=position_id,
        )

    except Exception as e:
        logger.exception("Error executing news-driven trade: %s", e)
        return None


class NewsMomentumTracker:
    """Tracks news-driven positions and manages exits."""

    def __init__(
        self,
        data_dir: Path | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the tracker.

        Args:
            data_dir: Directory for storing position data
            config: Strategy configuration
        """
        self.data_dir = data_dir or Path("data/news_momentum")
        self.config = config or DEFAULT_CONFIG
        self.positions: dict[str, NewsDrivenPosition] = {}

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._load_positions()

    def _load_positions(self) -> None:
        """Load existing positions from disk."""
        positions_file = self.data_dir / "positions.json"
        if positions_file.exists():
            try:
                data = json.loads(positions_file.read_text())
                # Reconstruct positions from dict
                # (simplified - full implementation would deserialize properly)
                logger.info("Loaded %d positions", len(data))
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning("Failed to load positions: %s", e)

    def _save_positions(self) -> None:
        """Save positions to disk."""
        positions_file = self.data_dir / "positions.json"
        data = {
            "saved_at": datetime.now(UTC).isoformat(),
            "positions": {
                pid: {
                    "market_id": pos.market_id,
                    "side": pos.side,
                    "entry_price": pos.entry_price,
                    "is_open": pos.is_open,
                    "pnl": pos.pnl,
                }
                for pid, pos in self.positions.items()
            },
        }
        positions_file.write_text(json.dumps(data, indent=2))

    def add_position(self, trade: NewsDrivenTrade) -> NewsDrivenPosition:
        """Add a new position from an executed trade.

        Args:
            trade: Executed trade

        Returns:
            Created position
        """
        token_id = (
            trade.signal.token_id_yes
            if trade.signal.side == "buy_yes"
            else trade.signal.token_id_no
        )

        position = NewsDrivenPosition(
            position_id=trade.position_id,
            timestamp=trade.timestamp,
            market_id=trade.signal.market_id,
            token_id=token_id,
            market_question=trade.signal.market_question,
            side=trade.signal.side,
            entry_price=trade.entry_price,
            position_size=trade.position_size,
            news_impact=NewsImpact(
                news_item=NewsItem(
                    timestamp=trade.timestamp,
                    source=trade.signal.news_source,
                    source_reliability=SourceReliability.MAJOR_OUTLET,
                    headline=trade.signal.reasoning,
                ),
                market_id=trade.signal.market_id,
                market_question=trade.signal.market_question,
                direction=ImpactDirection.POSITIVE,
                confidence=trade.signal.confidence,
                price_impact_estimate=trade.signal.edge,
                source_quality_score=trade.signal.confidence,
                reasoning=trade.signal.reasoning,
            ),
        )

        self.positions[trade.position_id] = position
        self._save_positions()

        return position

    def check_exit_signals(
        self,
        position: NewsDrivenPosition,
        current_price: float,
        current_timestamp: datetime | None = None,
    ) -> tuple[bool, str]:
        """Check if position should be exited.

        Args:
            position: Position to check
            current_price: Current market price
            current_timestamp: Optional timestamp override

        Returns:
            Tuple of (should_exit, reason)
        """
        now = current_timestamp or datetime.now(UTC)

        # Calculate PnL
        if position.side == "buy_yes":
            pnl = (current_price - position.entry_price) / position.entry_price
        else:  # buy_no
            no_entry = 1.0 - position.entry_price
            no_current = 1.0 - current_price
            pnl = (no_current - no_entry) / no_entry if no_entry > 0 else 0

        # Check stop loss
        if pnl < -self.config["stop_loss_pct"]:
            return True, f"stop_loss_{pnl:.2%}"

        # Check profit target
        if pnl >= self.config["profit_target_pct"]:
            return True, f"profit_target_{pnl:.2%}"

        # Check max hold time
        hold_time = now - position.timestamp
        if hold_time > timedelta(hours=self.config["max_hold_hours"]):
            return True, f"max_hold_time_{hold_time.total_seconds() / 3600:.1f}h"

        # Check momentum exit
        if self.config["momentum_exit_enabled"]:
            position.record_price(now, current_price)
            if position.check_momentum_exit(current_price):
                return True, "momentum_turned"

        # Check if price reached target (fair value)
        if (
            position.side == "buy_yes"
            and current_price >= position.news_impact.price_impact_estimate
        ):
            return True, "reached_fair_value"
        if (
            position.side == "buy_no"
            and (1.0 - current_price) >= position.news_impact.price_impact_estimate
        ):
            return True, "reached_fair_value"

        return False, ""

    def close_position(
        self,
        position_id: str,
        exit_price: float,
        reason: str,
        timestamp: datetime | None = None,
    ) -> NewsDrivenPosition | None:
        """Close a position.

        Args:
            position_id: Position to close
            exit_price: Exit price
            reason: Exit reason
            timestamp: Optional timestamp override

        Returns:
            Closed position or None if not found
        """
        position = self.positions.get(position_id)
        if not position or not position.is_open:
            return None

        now = timestamp or datetime.now(UTC)

        # Calculate final PnL
        if position.side == "buy_yes":
            pnl = (exit_price - position.entry_price) * position.position_size
        else:  # buy_no
            no_entry = 1.0 - position.entry_price
            no_exit = 1.0 - exit_price
            pnl = (no_exit - no_entry) * position.position_size

        # Update position
        object.__setattr__(position, "exit_price", exit_price)
        object.__setattr__(position, "exit_timestamp", now)
        object.__setattr__(position, "exit_reason", reason)
        object.__setattr__(position, "pnl", pnl)

        self._save_positions()

        return position

    def get_open_positions(self) -> list[NewsDrivenPosition]:
        """Get all open positions."""
        return [p for p in self.positions.values() if p.is_open]

    def get_performance_summary(self) -> dict[str, Any]:
        """Get performance summary of all trades."""
        closed = [p for p in self.positions.values() if not p.is_open]

        if not closed:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "avg_pnl": 0.0,
                "total_pnl": 0.0,
                "avg_hold_time_hours": 0.0,
            }

        pnls = [p.pnl or 0 for p in closed]
        wins = sum(1 for pnl in pnls if pnl > 0)

        hold_times = [(p.holding_duration or timedelta()).total_seconds() / 3600 for p in closed]

        return {
            "total_trades": len(closed),
            "win_rate": wins / len(closed),
            "avg_pnl": np.mean(pnls),
            "total_pnl": sum(pnls),
            "avg_hold_time_hours": np.mean(hold_times) if hold_times else 0,
        }


def run_news_momentum_scan(
    news_items: list[NewsItem] | None = None,
    snapshots_dir: Path | None = None,
    capital: float = 10000,
    dry_run: bool = True,
    max_positions: int = 5,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run a complete news-driven momentum scan.

    Args:
        news_items: List of news items to analyze (if None, uses mock/example data)
        snapshots_dir: Directory with market snapshots
        capital: Available capital
        dry_run: If True, don't execute trades
        max_positions: Maximum positions to take
        config: Strategy configuration

    Returns:
        Dictionary with scan results
    """
    now = datetime.now(UTC)
    cfg = config or DEFAULT_CONFIG

    logger.info("Starting news momentum scan at %s", now.isoformat())

    # Use example news if none provided
    if news_items is None:
        news_items = [
            NewsItem(
                timestamp=now - timedelta(seconds=30),
                source="@realDonaldTrump",
                source_reliability=SourceReliability.VERIFIED,
                headline="Just won the election by a landslide! Thank you America!",
                category=NewsCategory.POLITICS,
            ),
        ]

    # Find markets from snapshot
    if snapshots_dir is None:
        snapshots_dir = Path("data")

    markets: list[dict[str, Any]] = []
    if snapshots_dir.exists():
        snapshot_files = sorted(snapshots_dir.glob("snapshot_*m_*.json"))
        if snapshot_files:
            try:
                data = json.loads(snapshot_files[-1].read_text())
                markets = data.get("markets", [])
            except (json.JSONDecodeError, FileNotFoundError):
                pass

    # Analyze news impact on markets
    all_signals: list[NewsDrivenSignal] = []
    for news_item in news_items:
        impacts = find_impacted_markets(news_item, markets)

        for impact in impacts:
            # Find market data
            market_data = None
            for m in markets:
                if str(m.get("market_id", m.get("id", ""))) == impact.market_id:
                    market_data = m
                    break

            if market_data:
                signal = generate_news_signal(impact, market_data, cfg)
                if signal:
                    all_signals.append(signal)

    # Sort by confidence * edge
    all_signals.sort(key=lambda s: s.confidence * s.edge, reverse=True)

    # Filter to actionable signals
    actionable = [s for s in all_signals if s.side != "no_trade"]

    logger.info("Generated %d signals, %d actionable", len(all_signals), len(actionable))

    # Execute trades
    tracker = NewsMomentumTracker(config=cfg)
    trades: list[NewsDrivenTrade] = []

    for signal in actionable[:max_positions]:
        trade = execute_news_trade(signal, capital, dry_run, cfg)
        if trade:
            trades.append(trade)
            tracker.add_position(trade)
            logger.info(
                "Executed %s trade: %s @ %.3f (edge: %.2f)",
                signal.side,
                signal.market_question[:50],
                trade.entry_price,
                signal.edge,
            )

    # Summary statistics
    buy_yes_signals = [s for s in all_signals if s.side == "buy_yes"]
    buy_no_signals = [s for s in all_signals if s.side == "buy_no"]

    return {
        "timestamp": now.isoformat(),
        "news_items_analyzed": len(news_items),
        "markets_available": len(markets),
        "signals_generated": len(all_signals),
        "actionable_signals": len(actionable),
        "trades_executed": len(trades),
        "dry_run": dry_run,
        "open_positions": len(tracker.get_open_positions()),
        "summary": {
            "buy_yes_count": len(buy_yes_signals),
            "buy_no_count": len(buy_no_signals),
            "avg_edge": np.mean([s.edge for s in actionable]) if actionable else 0,
            "avg_confidence": np.mean([s.confidence for s in actionable]) if actionable else 0,
        },
        "signals": [s.to_dict() for s in all_signals[:20]],  # Limit output
        "trades": [t.to_dict() for t in trades],
    }


def check_position_exits(
    tracker: NewsMomentumTracker,
    snapshot_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Check all open positions for exit signals.

    Args:
        tracker: Position tracker
        snapshot_path: Optional path to snapshot for current prices

    Returns:
        List of exit actions taken
    """
    from .pnl import load_orderbooks_from_snapshot

    exits: list[dict[str, Any]] = []

    # Load orderbooks
    orderbooks = None
    if snapshot_path and snapshot_path.exists():
        orderbooks = load_orderbooks_from_snapshot(snapshot_path)

    for position in tracker.get_open_positions():
        if orderbooks and position.token_id in orderbooks:
            book = orderbooks[position.token_id]
            current_price = book.mid_price

            if current_price is not None:
                should_exit, reason = tracker.check_exit_signals(position, current_price)

                if should_exit:
                    closed = tracker.close_position(position.position_id, current_price, reason)
                    if closed:
                        exits.append(
                            {
                                "position_id": position.position_id,
                                "market": position.market_question[:50],
                                "exit_price": current_price,
                                "reason": reason,
                                "pnl": closed.pnl,
                            }
                        )

    return exits
