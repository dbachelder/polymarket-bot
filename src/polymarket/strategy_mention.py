"""Default-to-NO strategy for mention markets.

Hypothesis: Markets about whether a person/entity will be "mentioned" exhibit
structural YES overpricing due to behavioral biases:
- Availability bias: traders overweight salient recent mentions
- FOMO: fear of missing out on "obvious" YES outcomes
- Asymmetric attention: monitoring for mentions is costly; NO is the "lazy" bet

Strategy: Default to NO positions unless there's strong evidence for YES.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Entry thresholds for NO positions
# These are conservative - we want high confidence before taking YES
DEFAULT_NO_ENTRY_MIN_PRICE = 0.35  # Buy NO when YES price > 0.65 (NO < 0.35)
DEFAULT_NO_ENTRY_MAX_PRICE = 0.50  # Don't buy NO if it's already expensive
DEFAULT_YES_ENTRY_MAX_PRICE = 0.30  # Only buy YES when it's cheap (high conviction)

# Position sizing
MAX_POSITION_SIZE = 5.0  # contracts per trade
MAX_POSITIONS_PER_SCAN = 10

# Confidence thresholds
MIN_EDGE_FOR_TRADE = 0.10  # 10% edge required


@dataclass(frozen=True)
class MentionMarket:
    """A mention-related market on Polymarket.

    Attributes:
        market_id: Polymarket market ID
        token_id_yes: YES token ID
        token_id_no: NO token ID
        question: Market question text
        mention_target: What's being monitored for mentions (person, entity, etc.)
        mention_context: Where the mention would occur (tweet, speech, etc.)
        current_yes_price: Current YES price (mid)
        current_no_price: Current NO price (mid)
        end_date: When the market resolves
    """

    market_id: str
    token_id_yes: str
    token_id_no: str
    question: str
    mention_target: str | None = None
    mention_context: str | None = None
    current_yes_price: float | None = None
    current_no_price: float | None = None
    end_date: datetime | None = None

    @property
    def implied_probability(self) -> float | None:
        """Market-implied probability of YES outcome."""
        return self.current_yes_price

    @property
    def is_expired(self) -> bool:
        """Check if market has expired."""
        if self.end_date is None:
            return False
        return datetime.now(UTC) > self.end_date


# Keywords that indicate mention markets
MENTION_KEYWORDS = [
    "mention",
    "mentions",
    "mentioned",
    "name-dropped",
    "name dropped",
    "shoutout",
    "shout out",
    "shout-out",
    "reference",
    "references",
    "referenced",
]

# Context patterns - where mentions occur
CONTEXT_PATTERNS = [
    ("tweet", ["tweet", "twitter", "x post", "post on x"]),
    ("speech", ["speech", "address", "remarks", "statement"]),
    ("interview", ["interview", "interviewed"]),
    ("press", ["press conference", "briefing", "presser"]),
    ("debate", ["debate", "debates"]),
    ("media", ["cnn", "fox", "msnbc", "news", "article", "coverage"]),
    ("congress", ["congress", "senate", "house floor", "hearing"]),
]


def _is_mention_market(question: str) -> bool:
    """Check if a market question is about mentions.

    Args:
        question: The market question text

    Returns:
        True if this is a mention market
    """
    question_lower = question.lower()
    return any(kw in question_lower for kw in MENTION_KEYWORDS)


def _extract_mention_target(question: str) -> str | None:
    """Extract what/who is being monitored for mentions.

    Uses simple heuristics to identify the target entity.

    Args:
        question: The market question text

    Returns:
        The target entity name or None if can't extract
    """
    question_lower = question.lower()

    # Common patterns:
    # "Will [X] mention [Y]?" -> Y is the target
    # "Will [X] be mentioned?" -> X is the target
    # "Will [X] mention [Y] in [Z]?" -> Y is the target

    # Pattern: "Will X mention Y" or "Will X mention Y in Z"
    # Stop at prepositions: in, during, at, on, by, before, after, etc.
    mention_pattern = r"will\s+\w+(?:\s+\w+){0,3}\s+mention\s+([^\s]+(?:\s+[^\s]+){0,3})(?:\s+(?:in|during|at|on|by|before|after|for|of|to|about|regarding|concerning|$))"
    match = re.search(mention_pattern, question_lower)
    if match:
        # Return the object of "mention" (who/what is being mentioned)
        target = match.group(1).strip().rstrip("?.")
        # Clean up common stop words
        target = re.sub(r"^(the|a|an)\s+", "", target)
        return target.title() if target else None

    # Pattern: "Will X be mentioned"
    be_mentioned_pattern = r"will\s+([^\s]+(?:\s+[^\s]+){0,3})\s+be\s+mentioned"
    match = re.search(be_mentioned_pattern, question_lower)
    if match:
        target = match.group(1).strip().rstrip("?.")
        target = re.sub(r"^(the|a|an)\s+", "", target)
        return target.title() if target else None

    # Pattern: mention of [X]
    of_pattern = r"mention\s+of\s+([^\s]+(?:\s+[^\s]+){0,3})(?:\s+(?:in|during|at|on|by|before|after|for|$))"
    match = re.search(of_pattern, question_lower)
    if match:
        target = match.group(1).strip().rstrip("?.")
        # Remove leading "the" or "of" if present
        target = re.sub(r"^(the|a|an|of)\s+", "", target)
        return target.title() if target else None

    # Fallback: extract capitalized phrases (likely proper nouns)
    capitalized = re.findall(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*", question)
    if capitalized:
        # Return the last capitalized phrase (often the entity)
        return capitalized[-1]

    return None


def _extract_mention_context(question: str) -> str | None:
    """Extract the context where the mention would occur.

    Args:
        question: The market question text

    Returns:
        The context category or None
    """
    question_lower = question.lower()

    for context, patterns in CONTEXT_PATTERNS:
        if any(p in question_lower for p in patterns):
            return context

    return None


def _compute_theoretical_yes_probability(
    market: MentionMarket,
    base_rate: float = 0.15,
) -> float:
    """Compute theoretical probability of YES based on structural factors.

    The default-to-NO hypothesis: base rate for mentions is low,
    but markets systematically overprice YES due to biases.

    Args:
        market: The mention market
        base_rate: Historical base rate for mentions (default 15%)

    Returns:
        Theoretical probability (0-1)
    """
    # Start with base rate
    prob = base_rate

    # Adjust based on mention context
    if market.mention_context:
        # Different contexts have different base rates
        context_adjustments = {
            "tweet": 0.10,  # Tweets are common but specific mentions are rare
            "speech": 0.20,  # Speeches often mention various topics
            "interview": 0.15,
            "press": 0.18,
            "debate": 0.25,  # Debates often involve many mentions
            "media": 0.12,
            "congress": 0.15,
        }
        adj = context_adjustments.get(market.mention_context, 0.0)
        prob = adj  # Use context-specific base rate

    # Adjust for time until resolution (closer = more predictable)
    if market.end_date:
        now = datetime.now(UTC)
        if market.end_date > now:
            hours_remaining = (market.end_date - now).total_seconds() / 3600
            if hours_remaining < 1:
                # Very close to resolution - if no mention yet, less likely
                prob *= 0.5
            elif hours_remaining < 6:
                prob *= 0.8

    return min(0.95, max(0.05, prob))


def find_mention_markets(snapshots_dir: Path | None = None) -> list[MentionMarket]:
    """Find mention-related markets from snapshot data.

    Args:
        snapshots_dir: Directory with snapshot files (optional)

    Returns:
        List of MentionMarket objects
    """
    markets: list[MentionMarket] = []

    if snapshots_dir is None:
        snapshots_dir = Path("data")

    if not snapshots_dir.exists():
        logger.warning("Snapshots directory not found: %s", snapshots_dir)
        return markets

    # Find the most recent 5m or 15m snapshot
    snapshot_files = sorted(snapshots_dir.glob("snapshot_*m_*.json"))
    if not snapshot_files:
        logger.warning("No snapshot files found in %s", snapshots_dir)
        return markets

    latest = snapshot_files[-1]
    logger.debug("Scanning for mention markets in: %s", latest)

    try:
        data = json.loads(latest.read_text())
        for m in data.get("markets", []):
            question = m.get("question", m.get("title", ""))

            if not _is_mention_market(question):
                continue

            token_ids = m.get("clob_token_ids", m.get("clobTokenIds", []))
            if len(token_ids) != 2:
                continue

            end_date_str = m.get("end_date", m.get("endDate", ""))
            end_date = None
            if end_date_str:
                try:
                    end_date = datetime.fromisoformat(str(end_date_str).replace("Z", "+00:00"))
                except (ValueError, AttributeError):
                    pass

            mention_target = _extract_mention_target(question)
            mention_context = _extract_mention_context(question)

            # Get current prices from book
            books = m.get("books", {})
            yes_book = books.get("yes", {})
            yes_bids = yes_book.get("bids", [])
            yes_asks = yes_book.get("asks", [])

            current_yes_price = None
            if yes_bids and yes_asks:
                try:
                    best_bid = max(float(b["price"]) for b in yes_bids)
                    best_ask = min(float(a["price"]) for a in yes_asks)
                    current_yes_price = (best_bid + best_ask) / 2
                except (ValueError, KeyError):
                    pass

            market = MentionMarket(
                market_id=str(m.get("market_id", m.get("id", ""))),
                token_id_yes=str(token_ids[0]),
                token_id_no=str(token_ids[1]),
                question=question,
                mention_target=mention_target,
                mention_context=mention_context,
                current_yes_price=current_yes_price,
                current_no_price=1.0 - current_yes_price if current_yes_price else None,
                end_date=end_date,
            )

            markets.append(market)

    except Exception as e:
        logger.exception("Error scanning snapshots for mention markets: %s", e)

    logger.info("Found %d mention markets", len(markets))
    return markets


@dataclass(frozen=True)
class MentionSignal:
    """Trading signal from mention market analysis.

    Attributes:
        timestamp: When signal was generated
        market: MentionMarket this signal is for
        side: "buy_yes", "buy_no", or "no_trade"
        market_prob: Current market-implied probability
        theoretical_prob: Our estimated true probability
        edge: Difference between theoretical and market probability
        confidence: Signal confidence (0-1)
        expected_value: Expected value of trade
        reasoning: Human-readable explanation
    """

    timestamp: datetime
    market: MentionMarket
    side: str  # "buy_yes", "buy_no", "no_trade"
    market_prob: float
    theoretical_prob: float
    edge: float
    confidence: float
    expected_value: float
    reasoning: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "market": {
                "market_id": self.market.market_id,
                "question": self.market.question,
                "mention_target": self.market.mention_target,
                "mention_context": self.market.mention_context,
                "current_yes_price": self.market.current_yes_price,
                "current_no_price": self.market.current_no_price,
            },
            "side": self.side,
            "market_prob": self.market_prob,
            "theoretical_prob": self.theoretical_prob,
            "edge": self.edge,
            "confidence": self.confidence,
            "expected_value": self.expected_value,
            "reasoning": self.reasoning,
        }


def generate_signals(
    markets: list[MentionMarket],
    base_rate: float = 0.15,
    no_entry_min_price: float = DEFAULT_NO_ENTRY_MIN_PRICE,
    no_entry_max_price: float = DEFAULT_NO_ENTRY_MAX_PRICE,
    yes_entry_max_price: float = DEFAULT_YES_ENTRY_MAX_PRICE,
    min_edge: float = MIN_EDGE_FOR_TRADE,
) -> list[MentionSignal]:
    """Generate trading signals using default-to-NO logic.

    Args:
        markets: List of mention markets
        base_rate: Historical base rate for mentions
        no_entry_min_price: Minimum NO price to consider entry
        no_entry_max_price: Maximum NO price to consider entry
        yes_entry_max_price: Maximum YES price for YES entry
        min_edge: Minimum edge required for any trade

    Returns:
        List of trading signals
    """
    signals: list[MentionSignal] = []
    now = datetime.now(UTC)

    for market in markets:
        # Skip if no price data or expired
        if market.current_yes_price is None or market.is_expired:
            continue

        market_prob = market.current_yes_price
        theoretical_prob = _compute_theoretical_yes_probability(market, base_rate)

        # Edge is the difference between our estimate and market's estimate
        edge = theoretical_prob - market_prob

        side = "no_trade"
        confidence = 0.0
        expected_value = 0.0
        reasoning = ""

        # DEFAULT-TO-NO LOGIC:
        # We only go YES if there's a strong signal (very cheap YES)
        # Otherwise, we look for overpriced YES (cheap NO)

        if market_prob <= yes_entry_max_price and edge > min_edge:
            # YES is cheap and we have positive edge - buy YES
            side = "buy_yes"
            confidence = min(1.0, abs(edge) * 2)  # Scale edge to confidence
            # EV = (win_prob * win_amount) - (lose_prob * lose_amount)
            win_amount = 1.0 - market_prob
            lose_amount = market_prob
            expected_value = (theoretical_prob * win_amount) - ((1 - theoretical_prob) * lose_amount)
            reasoning = (
                f"YES underpriced: market={market_prob:.2%}, "
                f"theoretical={theoretical_prob:.2%}, edge={edge:+.1%}"
            )

        else:
            # Check for buy_no opportunity
            # NO price = 1 - YES price
            no_price = market.current_no_price or (1.0 - market_prob)
            if no_price > 0:
                # For NO position: we win if YES doesn't happen
                theoretical_no_prob = 1.0 - theoretical_prob
                market_no_prob = 1.0 - market_prob
                no_edge = theoretical_no_prob - market_no_prob

                # Buy NO when NO is cheap enough (YES is expensive)
                # and we have sufficient edge
                if (no_entry_min_price <= no_price <= no_entry_max_price and
                        no_edge > min_edge):
                    side = "buy_no"
                    confidence = min(1.0, abs(no_edge) * 2)
                    # EV for NO side
                    win_amount = 1.0 - no_price
                    lose_amount = no_price
                    expected_value = (theoretical_no_prob * win_amount) - ((1 - theoretical_no_prob) * lose_amount)
                    reasoning = (
                        f"YES overpriced: market={market_prob:.2%}, "
                        f"theoretical={theoretical_prob:.2%}, "
                        f"NO_edge={no_edge:+.1%}"
                    )

        if side == "no_trade":
            reasoning = (
                f"No edge: market={market_prob:.2%}, "
                f"theoretical={theoretical_prob:.2%}, edge={edge:+.1%}"
            )

        signals.append(
            MentionSignal(
                timestamp=now,
                market=market,
                side=side,
                market_prob=market_prob,
                theoretical_prob=theoretical_prob,
                edge=edge,
                confidence=confidence,
                expected_value=expected_value,
                reasoning=reasoning,
            )
        )

    # Sort by expected value descending
    signals.sort(key=lambda s: s.expected_value, reverse=True)

    return signals


@dataclass(frozen=True)
class MentionTrade:
    """An executed mention market trade.

    Attributes:
        timestamp: When trade was executed
        signal: The signal that triggered the trade
        order_result: Result from order submission
        position_size: Number of contracts
        entry_price: Price paid per contract
    """

    timestamp: datetime
    signal: MentionSignal
    order_result: Any
    position_size: float
    entry_price: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "signal": self.signal.to_dict(),
            "order_result": {
                "success": self.order_result.success,
                "dry_run": self.order_result.dry_run,
                "message": self.order_result.message,
                "order_id": self.order_result.order_id,
            } if hasattr(self.order_result, "success") else str(self.order_result),
            "position_size": self.position_size,
            "entry_price": self.entry_price,
        }


def execute_trade(
    signal: MentionSignal,
    dry_run: bool = True,
    max_position_size: float = MAX_POSITION_SIZE,
) -> MentionTrade | None:
    """Execute a trade based on a mention market signal.

    Args:
        signal: MentionSignal with side and market info
        dry_run: If True, don't actually submit orders
        max_position_size: Maximum position size

    Returns:
        MentionTrade if executed, None otherwise
    """
    if signal.side == "no_trade":
        return None

    now = datetime.now(UTC)

    # Determine order parameters
    if signal.side == "buy_yes":
        token_id = signal.market.token_id_yes
        side = "buy"
        price = signal.market.current_yes_price or 0.10
    elif signal.side == "buy_no":
        token_id = signal.market.token_id_no
        side = "buy"
        price = signal.market.current_no_price or 0.10
    else:
        return None

    # Limit price: be slightly aggressive to get filled
    limit_price = min(0.99, price * 1.02)  # Pay up to 2% over mid

    # Position sizing based on confidence and EV
    base_size = 1.0
    confidence_multiplier = min(5.0, signal.confidence * 5)
    position_size = min(max_position_size, base_size * confidence_multiplier)
    position_size = round(position_size, 2)

    try:
        from .trading import Order, submit_order

        order = Order(
            token_id=token_id,
            side=side,
            size=position_size,
            price=limit_price,
        )

        result = submit_order(order)

        return MentionTrade(
            timestamp=now,
            signal=signal,
            order_result=result,
            position_size=position_size,
            entry_price=limit_price,
        )

    except Exception as e:
        logger.exception("Error executing mention trade: %s", e)
        return None


def run_mention_scan(
    snapshots_dir: Path | None = None,
    base_rate: float = 0.15,
    dry_run: bool = True,
    max_positions: int = MAX_POSITIONS_PER_SCAN,
) -> dict[str, Any]:
    """Run a complete mention market scan with default-to-NO strategy.

    Args:
        snapshots_dir: Directory with market snapshots
        base_rate: Historical base rate for mentions
        dry_run: If True, don't execute trades
        max_positions: Maximum positions to take

    Returns:
        Dictionary with scan results
    """
    now = datetime.now(UTC)

    logger.info("Starting mention market scan at %s", now.isoformat())

    # Step 1: Find mention markets
    markets = find_mention_markets(snapshots_dir)
    logger.info("Found %d mention markets", len(markets))

    # Step 2: Generate signals using default-to-NO logic
    signals = generate_signals(markets, base_rate=base_rate)
    logger.info("Generated %d signals", len(signals))

    # Step 3: Filter to actionable signals
    actionable = [s for s in signals if s.side != "no_trade" and s.expected_value > 0.05]
    logger.info("%d actionable signals", len(actionable))

    # Step 4: Execute trades (respecting limits)
    trades: list[MentionTrade] = []
    for signal in actionable[:max_positions]:
        trade = execute_trade(signal, dry_run=dry_run)
        if trade:
            trades.append(trade)
            logger.info(
                "Executed %s trade: %s @ %.3f (EV: %.3f)",
                signal.side,
                signal.market.question[:50],
                trade.entry_price,
                signal.expected_value,
            )

    # Compute summary statistics
    buy_yes_signals = [s for s in signals if s.side == "buy_yes"]
    buy_no_signals = [s for s in signals if s.side == "buy_no"]
    no_trade_signals = [s for s in signals if s.side == "no_trade"]

    avg_edge_no = np.mean([s.edge for s in buy_no_signals]) if buy_no_signals else 0.0
    avg_edge_yes = np.mean([s.edge for s in buy_yes_signals]) if buy_yes_signals else 0.0

    return {
        "timestamp": now.isoformat(),
        "markets_scanned": len(markets),
        "signals_generated": len(signals),
        "actionable_signals": len(actionable),
        "trades_executed": len(trades),
        "dry_run": dry_run,
        "summary": {
            "buy_yes_count": len(buy_yes_signals),
            "buy_no_count": len(buy_no_signals),
            "no_trade_count": len(no_trade_signals),
            "avg_edge_buy_no": avg_edge_no,
            "avg_edge_buy_yes": avg_edge_yes,
        },
        "markets": [
            {
                "market_id": m.market_id,
                "question": m.question,
                "target": m.mention_target,
                "context": m.mention_context,
                "yes_price": m.current_yes_price,
            }
            for m in markets
        ],
        "signals": [s.to_dict() for s in signals],
        "trades": [t.to_dict() for t in trades],
    }
