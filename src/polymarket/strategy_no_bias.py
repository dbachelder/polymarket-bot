"""NO Bias Exploit Strategy for Phrase-Based Markets.

Hypothesis: Retail traders on Polymarket systematically overprice YES shares in
low-probability, phrase-based markets due to optimism bias and lottery-ticket
mentality. By identifying markets where YES is priced >3x the base rate
probability and fading with NO, we capture positive EV from structural
mispricing without requiring prediction skill.

Rationale:
- Research shows consistent "NO bias" opportunity: retail prefers buying YES
  (hope/optimism), creating upward pressure on YES prices
- Especially pronounced in:
  - Long-shot political outcomes (e.g., "Will Kanye run?")
  - Viral/phrase-based markets where narrative drives attention
  - Markets with charismatic/viral figures (Elon, Trump, etc.)
  - Low-probability events with catchy headlines (<10% base rate)
- Typical mispricing: YES trades at 8-15% when true probability is 2-5%
- NO at 85-92% when fair is 95-98% = ~5-10% edge per trade
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .config import PolymarketConfig

logger = logging.getLogger(__name__)


class MarketVertical(Enum):
    """Market verticals for NO bias targeting."""

    POLITICS = "politics"
    SPACE = "space"
    TECH = "tech"
    POP_CULTURE = "pop_culture"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class BaseRateEstimate:
    """Base rate estimate for a market type."""

    vertical: MarketVertical
    pattern: str  # Regex pattern that matches this market type
    base_rate: float  # Historical probability (0-1)
    source: str  # Source of the estimate
    reasoning: str  # Why this base rate


# Base rate database for phrase-based markets
# These are estimates based on historical frequency of similar events
BASE_RATE_DATABASE: list[BaseRateEstimate] = [
    # Politics - long shot outcomes
    BaseRateEstimate(
        vertical=MarketVertical.POLITICS,
        pattern=r"(?i)(kanye|west).*?(run|president|election)",
        base_rate=0.02,
        source="historical_third_party",
        reasoning="Celebrity independent runs almost never materialize (<2%)",
    ),
    BaseRateEstimate(
        vertical=MarketVertical.POLITICS,
        pattern=r"(?i)(trump|biden).*?(resign|impeach|remove).*?before",
        base_rate=0.05,
        source="historical_presidential",
        reasoning="Presidential resignations/removals are extremely rare in modern era",
    ),
    BaseRateEstimate(
        vertical=MarketVertical.POLITICS,
        pattern=r"(?i)(third.party|independent).*?win",
        base_rate=0.01,
        source="historical_elections",
        reasoning="No third party has won presidency since 1860",
    ),
    BaseRateEstimate(
        vertical=MarketVertical.POLITICS,
        pattern=r"(?i)(war|nuclear|attack).*?(us|usa|america|nato)",
        base_rate=0.03,
        source="historical_conflicts",
        reasoning="Direct attacks on US homeland are rare, wars declared rarely",
    ),
    BaseRateEstimate(
        vertical=MarketVertical.POLITICS,
        pattern=r"(?i)(martial.law|suspend.*election|cancel.*election)",
        base_rate=0.01,
        source="historical_democracy",
        reasoning="US has never suspended elections or declared martial law nationwide",
    ),
    # Space - extremely low probability events
    BaseRateEstimate(
        vertical=MarketVertical.SPACE,
        pattern=r"(?i)(mars|venus).*?(colony|settlement|city)",
        base_rate=0.01,
        source="space_timeline",
        reasoning="Mars colonies remain decades away technically",
    ),
    BaseRateEstimate(
        vertical=MarketVertical.SPACE,
        pattern=r"(?i)(alien|extraterrestrial|ufo).*?(contact|contacted|landing|discovered|discover)",
        base_rate=0.001,
        source="fermi_paradox",
        reasoning="No confirmed alien contact in recorded history",
    ),
    BaseRateEstimate(
        vertical=MarketVertical.SPACE,
        pattern=r"(?i)(contact|contacted).*?(alien|extraterrestrial|ufo|martian)",
        base_rate=0.001,
        source="fermi_paradox",
        reasoning="No confirmed alien contact in recorded history",
    ),
    BaseRateEstimate(
        vertical=MarketVertical.SPACE,
        pattern=r"(?i)(asteroid|meteor).*?(earth|impact|destroy)",
        base_rate=0.0001,
        source="nasa_neo",
        reasoning=" civilization-ending asteroid strikes ~1 per 100M years",
    ),
    BaseRateEstimate(
        vertical=MarketVertical.SPACE,
        pattern=r"(?i)(moon.*base|lunar.*colony).*?(202|203)",
        base_rate=0.10,
        source="artemis_program",
        reasoning="Artemis has delays, 2028+ more realistic for permanent base",
    ),
    # Tech - viral/low probability
    BaseRateEstimate(
        vertical=MarketVertical.TECH,
        pattern=r"(?i)(elon|musk).*?(twitter|x.com).*?(shutdown|delete|quit)",
        base_rate=0.05,
        source="behavioral_pattern",
        reasoning="Threats rarely followed through on, but higher than baseline",
    ),
    BaseRateEstimate(
        vertical=MarketVertical.TECH,
        pattern=r"(?i)(apple|google|microsoft).*?(acquire|buy).*?(twitter|x|tesla)",
        base_rate=0.03,
        source="m&a_history",
        reasoning="Mega-mergers of this scale are rare and face regulatory hurdles",
    ),
    BaseRateEstimate(
        vertical=MarketVertical.TECH,
        pattern=r"(?i)(bitcoin|btc|crypto).*?(ban|illegal|outlaw).*?(us|usa)",
        base_rate=0.08,
        source="regulatory_trends",
        reasoning="US has moved toward regulation not bans, but non-zero risk",
    ),
    BaseRateEstimate(
        vertical=MarketVertical.TECH,
        pattern=r"(?i)(gpt|ai).*?(sentient|conscious|self-aware)",
        base_rate=0.05,
        source="technical_assessment",
        reasoning="Current AI architectures are not on path to consciousness",
    ),
    # Pop Culture - celebrity/prediction markets
    BaseRateEstimate(
        vertical=MarketVertical.POP_CULTURE,
        pattern=r"(?i)(taylor.swift|kanye|elon).*?(president|election|run)",
        base_rate=0.02,
        source="celebrity_politics",
        reasoning="Celebrity political runs rarely succeed, even when attempted",
    ),
    BaseRateEstimate(
        vertical=MarketVertical.POP_CULTURE,
        pattern=r"(?i)(oscar|grammy|emmy).*?(surprise|upset|unexpected)",
        base_rate=0.15,
        source="award_history",
        reasoning="Award upsets happen but are less common than 15%",
    ),
    BaseRateEstimate(
        vertical=MarketVertical.POP_CULTURE,
        pattern=r"(?i)(gta.6|grand.theft.auto).*?(delay|postpone|push)",
        base_rate=0.30,
        source="game_development",
        reasoning="AAA game delays are common (~30% get delayed)",
    ),
    BaseRateEstimate(
        vertical=MarketVertical.POP_CULTURE,
        pattern=r"(?i)(kardashian|jenner).*?(president|congress|senate)",
        base_rate=0.01,
        source="celebrity_politics",
        reasoning="Reality stars rarely successfully transition to major political office",
    ),
]


@dataclass(frozen=True)
class NoBiasSignal:
    """Trading signal from NO bias analysis."""

    timestamp: datetime
    market_id: str
    token_id_yes: str
    token_id_no: str
    market_question: str
    vertical: MarketVertical

    # Price data
    yes_ask: float  # Current YES ask price
    no_bid: float  # Current NO bid price

    # Analysis
    base_rate: float  # Estimated true probability
    mispricing_ratio: float  # YES price / base rate
    edge: float  # Expected return from buying NO
    confidence: float  # 0-1 confidence in the signal

    # Metadata
    volume_usd: float
    time_to_resolution: timedelta | None
    reasoning: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "market_id": self.market_id,
            "market_question": self.market_question,
            "vertical": self.vertical.value,
            "yes_ask": self.yes_ask,
            "no_bid": self.no_bid,
            "base_rate": self.base_rate,
            "mispricing_ratio": self.mispricing_ratio,
            "edge": self.edge,
            "confidence": self.confidence,
            "volume_usd": self.volume_usd,
            "time_to_resolution_hours": (
                self.time_to_resolution.total_seconds() / 3600
                if self.time_to_resolution
                else None
            ),
            "reasoning": self.reasoning,
        }


@dataclass
class NoBiasPosition:
    """Position taken based on NO bias signal."""

    position_id: str
    timestamp: datetime
    market_id: str
    token_id: str
    market_question: str
    vertical: MarketVertical

    # Entry
    entry_no_price: float
    position_size_usd: float
    expected_edge: float

    # Order tracking (for live trading)
    order_id: str | None = None
    order_status: str | None = None  # 'pending', 'filled', 'failed', 'dry_run'
    fill_price: float | None = None
    contracts: float | None = None

    # Exit tracking
    exit_price: float | None = None
    exit_timestamp: datetime | None = None
    exit_reason: str | None = None
    pnl: float | None = None
    settled: bool = False

    @property
    def is_open(self) -> bool:
        return self.exit_price is None and not self.settled

    def close(
        self,
        exit_price: float,
        reason: str,
        timestamp: datetime | None = None,
    ) -> None:
        """Close the position."""
        ts = timestamp or datetime.now(UTC)

        # Calculate PnL for NO position
        # NO pays $1 if event doesn't happen, $0 if it does
        # Entry: buy NO at entry_no_price (e.g., $0.90)
        # Exit: sell NO at exit_price (e.g., $0.95)
        # Or hold to settlement: $1.00 if NO wins
        pnl_per_contract = exit_price - self.entry_no_price
        contracts = self.position_size_usd / self.entry_no_price
        total_pnl = pnl_per_contract * contracts

        object.__setattr__(self, "exit_price", exit_price)
        object.__setattr__(self, "exit_timestamp", ts)
        object.__setattr__(self, "exit_reason", reason)
        object.__setattr__(self, "pnl", total_pnl)

    def settle(self, no_wins: bool, timestamp: datetime | None = None) -> None:
        """Settle position at resolution."""
        ts = timestamp or datetime.now(UTC)
        settlement_price = 1.0 if no_wins else 0.0
        reason = "settlement_no_wins" if no_wins else "settlement_yes_wins"

        self.close(settlement_price, reason, ts)
        object.__setattr__(self, "settled", True)


@dataclass
class NoBiasScanResult:
    """Result of NO bias scan."""

    timestamp: datetime
    markets_analyzed: int
    signals_generated: int
    actionable_signals: int

    # Signals by category
    signals: list[NoBiasSignal]

    # Portfolio summary
    positions_taken: int
    total_capital_deployed: float

    # Metadata
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "markets_analyzed": self.markets_analyzed,
            "signals_generated": self.signals_generated,
            "actionable_signals": self.actionable_signals,
            "signals": [s.to_dict() for s in self.signals],
            "positions_taken": self.positions_taken,
            "total_capital_deployed": self.total_capital_deployed,
            "config": self.config,
        }


def match_market_to_base_rate(market_question: str) -> BaseRateEstimate | None:
    """Match a market question to its base rate estimate.

    Args:
        market_question: The market question text

    Returns:
        BaseRateEstimate if matched, None otherwise
    """
    for estimate in BASE_RATE_DATABASE:
        if re.search(estimate.pattern, market_question):
            return estimate
    return None


def classify_vertical(market_question: str) -> MarketVertical:
    """Classify market into vertical based on keywords.

    Args:
        market_question: The market question text

    Returns:
        MarketVertical classification
    """
    text = market_question.lower()

    # Check politics first
    politics_keywords = [
        "president",
        "election",
        "vote",
        "congress",
        "senate",
        "legislation",
        "trump",
        "biden",
        "harris",
        "republican",
        "democrat",
        "impeach",
        "resign",
        "war",
        "nuclear",
    ]
    if any(kw in text for kw in politics_keywords):
        return MarketVertical.POLITICS

    # Space keywords
    space_keywords = [
        "mars",
        "moon",
        "alien",
        "asteroid",
        "space",
        "nasa",
        "spacex",
        "ufo",
        "extraterrestrial",
    ]
    if any(kw in text for kw in space_keywords):
        return MarketVertical.SPACE

    # Tech keywords
    tech_keywords = [
        "bitcoin",
        "crypto",
        "ai",
        "gpt",
        "tesla",
        "twitter",
        "apple",
        "google",
        "meta",
        "tech",
    ]
    if any(kw in text for kw in tech_keywords):
        return MarketVertical.TECH

    # Pop culture keywords
    pop_keywords = [
        "taylor swift",
        "kanye",
        "kardashian",
        "oscar",
        "grammy",
        "emmy",
        "gta",
        "album",
        "movie",
        "celebrity",
    ]
    if any(kw in text for kw in pop_keywords):
        return MarketVertical.POP_CULTURE

    return MarketVertical.UNKNOWN


def calculate_edge(yes_ask: float, base_rate: float) -> float:
    """Calculate expected edge from buying NO.

    Args:
        yes_ask: Current YES ask price
        base_rate: Estimated true probability

    Returns:
        Expected return as percentage
    """
    # If YES is overpriced, NO is underpriced
    # NO price = 1 - YES price
    no_fair = 1.0 - base_rate
    no_market = 1.0 - yes_ask

    if no_market <= 0:
        return 0.0

    # Edge is (fair - market) / market
    return (no_fair - no_market) / no_market


def generate_no_bias_signal(
    market: dict[str, Any],
    min_mispricing_ratio: float = 3.0,
    min_volume_usd: float = 10000.0,
    max_yes_price: float = 0.30,
) -> NoBiasSignal | None:
    """Generate NO bias signal for a single market.

    Args:
        market: Market data dict
        min_mispricing_ratio: Minimum YES_price/base_rate ratio
        min_volume_usd: Minimum market volume
        max_yes_price: Maximum YES price to consider (avoid high-prob markets)

    Returns:
        NoBiasSignal if criteria met, None otherwise
    """
    question = market.get("question", market.get("title", ""))
    market_id = str(market.get("market_id", market.get("id", "")))

    if not question or not market_id:
        return None

    # Match to base rate
    base_rate_estimate = match_market_to_base_rate(question)
    if base_rate_estimate is None:
        return None

    base_rate = base_rate_estimate.base_rate

    # Get orderbook data
    books = market.get("books", {})
    yes_book = books.get("yes", {})
    yes_asks = yes_book.get("asks", [])
    yes_bids = yes_book.get("bids", [])

    if not yes_asks or not yes_bids:
        return None

    try:
        yes_ask = min(float(a["price"]) for a in yes_asks)
    except (ValueError, KeyError):
        return None

    # Skip if YES price too high (not a long-shot market)
    if yes_ask > max_yes_price:
        return None

    # Calculate NO bid (inverse of YES)
    no_bid = 1.0 - yes_ask
    if no_bid <= 0.05:  # Need some liquidity on NO side
        return None

    # Check volume
    volume = market.get("volume", market.get("volume24h", 0))
    if isinstance(volume, str):
        try:
            volume = float(volume)
        except ValueError:
            volume = 0.0
    if volume < min_volume_usd:
        return None

    # Calculate mispricing
    mispricing_ratio = yes_ask / base_rate if base_rate > 0 else 0

    # Check mispricing threshold
    if mispricing_ratio < min_mispricing_ratio:
        return None

    # Calculate edge
    edge = calculate_edge(yes_ask, base_rate)

    # Confidence based on:
    # - How extreme the mispricing is (higher = more confident)
    # - Base rate certainty (lower base rates are more certain)
    # - Volume (higher = more confident it's real)
    confidence_factors = [
        min(1.0, (mispricing_ratio - 1) / 5),  # Ratio component
        1.0 - base_rate,  # Lower base rate = higher confidence
        min(1.0, volume / 100000),  # Volume component
    ]
    confidence = np.mean(confidence_factors)

    # Get token IDs
    token_ids = market.get("clob_token_ids", market.get("clobTokenIds", []))
    if len(token_ids) != 2:
        return None

    # Parse resolution time if available
    end_date = market.get("end_date", market.get("resolution_date"))
    time_to_resolution = None
    if end_date:
        try:
            resolution_dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
            time_to_resolution = resolution_dt - datetime.now(UTC)
        except (ValueError, TypeError):
            pass

    reasoning = (
        f"YES trading at {yes_ask:.1%} vs base rate {base_rate:.1%} "
        f"({mispricing_ratio:.1f}x overpricing). "
        f"Expected edge: {edge:.1%}. {base_rate_estimate.reasoning}"
    )

    return NoBiasSignal(
        timestamp=datetime.now(UTC),
        market_id=market_id,
        token_id_yes=str(token_ids[0]),
        token_id_no=str(token_ids[1]),
        market_question=question,
        vertical=base_rate_estimate.vertical,
        yes_ask=yes_ask,
        no_bid=no_bid,
        base_rate=base_rate,
        mispricing_ratio=mispricing_ratio,
        edge=edge,
        confidence=confidence,
        volume_usd=float(volume),
        time_to_resolution=time_to_resolution,
        reasoning=reasoning,
    )


def scan_markets_for_no_bias(
    markets: list[dict[str, Any]],
    min_mispricing_ratio: float = 3.0,
    min_volume_usd: float = 10000.0,
    max_yes_price: float = 0.30,
    min_edge: float = 0.05,
) -> NoBiasScanResult:
    """Scan markets for NO bias opportunities.

    Args:
        markets: List of market dicts
        min_mispricing_ratio: Minimum YES_price/base_rate ratio
        min_volume_usd: Minimum market volume
        max_yes_price: Maximum YES price to consider
        min_edge: Minimum expected edge

    Returns:
        NoBiasScanResult with signals
    """
    now = datetime.now(UTC)
    signals: list[NoBiasSignal] = []

    for market in markets:
        signal = generate_no_bias_signal(
            market,
            min_mispricing_ratio=min_mispricing_ratio,
            min_volume_usd=min_volume_usd,
            max_yes_price=max_yes_price,
        )
        if signal and signal.edge >= min_edge:
            signals.append(signal)

    # Sort by edge * confidence
    signals.sort(key=lambda s: s.edge * s.confidence, reverse=True)

    return NoBiasScanResult(
        timestamp=now,
        markets_analyzed=len(markets),
        signals_generated=len(signals),
        actionable_signals=len([s for s in signals if s.edge >= min_edge]),
        signals=signals,
        positions_taken=0,
        total_capital_deployed=0.0,
        config={
            "min_mispricing_ratio": min_mispricing_ratio,
            "min_volume_usd": min_volume_usd,
            "max_yes_price": max_yes_price,
            "min_edge": min_edge,
        },
    )


def calculate_position_size(
    signal: NoBiasSignal,
    bankroll: float,
    base_pct: float = 2.0,
    max_pct: float = 5.0,
) -> float:
    """Calculate position size based on signal quality.

    Args:
        signal: NO bias signal
        bankroll: Total available capital
        base_pct: Base position size as % of bankroll
        max_pct: Maximum position size as % of bankroll

    Returns:
        Position size in USD
    """
    # Scale by confidence and edge
    edge_factor = min(2.0, signal.edge * 10)  # Cap at 2x
    confidence_factor = signal.confidence

    # Higher conviction = larger size
    position_pct = base_pct * edge_factor * confidence_factor
    position_pct = min(position_pct, max_pct)  # Hard cap

    return bankroll * (position_pct / 100)


class NoBiasTracker:
    """Track NO bias positions and performance."""

    def __init__(
        self,
        data_dir: Path | None = None,
        max_positions_per_vertical: int = 3,
    ) -> None:
        """Initialize tracker.

        Args:
            data_dir: Directory for storing position data
            max_positions_per_vertical: Max concurrent positions per vertical
        """
        self.data_dir = data_dir or Path("data/no_bias")
        self.max_positions_per_vertical = max_positions_per_vertical
        self.positions: dict[str, NoBiasPosition] = {}

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._load_positions()

    def _load_positions(self) -> None:
        """Load positions from disk."""
        positions_file = self.data_dir / "positions.json"
        if positions_file.exists():
            try:
                data = json.loads(positions_file.read_text())
                # Reconstruct positions (simplified)
                logger.info("Loaded %d positions from %s", len(data), positions_file)
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
                    "vertical": pos.vertical.value,
                    "entry_price": pos.entry_no_price,
                    "position_size": pos.position_size_usd,
                    "is_open": pos.is_open,
                    "pnl": pos.pnl,
                    "order_id": pos.order_id,
                    "order_status": pos.order_status,
                    "fill_price": pos.fill_price,
                    "contracts": pos.contracts,
                }
                for pid, pos in self.positions.items()
            },
        }
        positions_file.write_text(json.dumps(data, indent=2))

    def can_open_position(self, vertical: MarketVertical) -> bool:
        """Check if we can open a new position in this vertical.

        Args:
            vertical: Market vertical

        Returns:
            True if under position limit
        """
        open_in_vertical = sum(
            1
            for p in self.positions.values()
            if p.is_open and p.vertical == vertical
        )
        return open_in_vertical < self.max_positions_per_vertical

    def open_position(
        self,
        signal: NoBiasSignal,
        bankroll: float,
        dry_run: bool = True,
        config: PolymarketConfig | None = None,
    ) -> NoBiasPosition | None:
        """Open a new position from a signal.

        Args:
            signal: NO bias signal
            bankroll: Available capital
            dry_run: If True, don't actually trade
            config: Optional PolymarketConfig for live trading

        Returns:
            NoBiasPosition if opened, None otherwise
        """
        if not self.can_open_position(signal.vertical):
            logger.info("Position limit reached for %s", signal.vertical.value)
            return None

        position_size = calculate_position_size(signal, bankroll)

        position_id = f"no_bias_{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}_{signal.market_id[:8]}"

        position = NoBiasPosition(
            position_id=position_id,
            timestamp=datetime.now(UTC),
            market_id=signal.market_id,
            token_id=signal.token_id_no,
            market_question=signal.market_question,
            vertical=signal.vertical,
            entry_no_price=signal.no_bid,
            position_size_usd=position_size,
            expected_edge=signal.edge,
            contracts=position_size / signal.no_bid if signal.no_bid > 0 else 0,
        )

        if dry_run:
            position.order_status = "dry_run"
            logger.info(
                "[DRY-RUN] Would buy NO: %s at %.3f, size=$%.2f (%.2f contracts)",
                signal.market_question[:50],
                signal.no_bid,
                position_size,
                position.contracts or 0,
            )
        else:
            # Live trading: submit order via trading module
            position = self._submit_live_order(position, signal, config)
            if position.order_status == "failed":
                # Don't save failed positions
                logger.error(
                    "Failed to open position for %s: %s",
                    signal.market_question[:50],
                    position.order_id,  # Stores error message on failure
                )
                return None

        self.positions[position_id] = position
        self._save_positions()

        return position

    def _submit_live_order(
        self,
        position: NoBiasPosition,
        signal: NoBiasSignal,
        config: PolymarketConfig | None = None,
    ) -> NoBiasPosition:
        """Submit live order to Polymarket CLOB.

        Args:
            position: The position to open
            signal: The signal that generated this position
            config: Optional PolymarketConfig

        Returns:
            Updated position with order details
        """
        from .trading import Order, submit_order

        if config is None:
            from .config import load_config

            config = load_config()

        # Calculate order size in contracts
        # NO price = signal.no_bid, position size in USD
        # contracts = USD / price per contract
        contracts = position.position_size_usd / signal.no_bid

        # Polymarket has minimum order size constraints
        # Ensure we meet minimums (typically around $1-5 worth)
        min_contracts = 1.0  # Minimum 1 contract
        if contracts < min_contracts:
            logger.warning(
                "Order size too small: %.4f contracts, adjusting to %.1f",
                contracts,
                min_contracts,
            )
            contracts = min_contracts

        # Round to reasonable precision (Polymarket uses 2 decimals for size)
        contracts = round(contracts, 2)

        # Price must be between 0.01 and 0.99
        # We buy NO at the current bid (which is 1 - YES ask)
        order_price = round(signal.no_bid, 2)
        order_price = max(0.01, min(0.99, order_price))

        try:
            order = Order(
                token_id=signal.token_id_no,
                side="buy",
                size=Decimal(str(contracts)),
                price=Decimal(str(order_price)),
            )
        except ValueError as e:
            logger.exception("Invalid order parameters: %s", e)
            position.order_status = "failed"
            position.order_id = f"validation_error: {e}"
            return position

        logger.info(
            "Submitting live order: buy %s NO at %.2f, size=%.2f contracts ($%.2f)",
            signal.market_question[:50],
            order_price,
            contracts,
            position.position_size_usd,
        )

        try:
            result = submit_order(order, config)
        except Exception as e:
            logger.exception("Order submission failed: %s", e)
            position.order_status = "failed"
            position.order_id = f"exception: {e}"
            return position

        if result.success:
            position.order_id = result.order_id
            position.order_status = "filled" if not result.dry_run else "dry_run"
            position.fill_price = float(order_price)

            logger.info(
                "Order submitted successfully: %s (order_id: %s)",
                result.message,
                result.order_id,
            )
        else:
            position.order_status = "failed"
            position.order_id = result.message[:255]  # Truncate long error messages

            logger.error(
                "Order submission failed: %s",
                result.message,
            )

        return position

    def get_open_positions(self) -> list[NoBiasPosition]:
        """Get all open positions."""
        return [p for p in self.positions.values() if p.is_open]

    def get_performance_summary(self) -> dict[str, Any]:
        """Get performance summary."""
        closed = [p for p in self.positions.values() if not p.is_open]

        if not closed:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "avg_pnl": 0.0,
                "total_pnl": 0.0,
                "by_vertical": {},
            }

        pnls = [p.pnl or 0 for p in closed]
        wins = sum(1 for pnl in pnls if pnl > 0)

        # Breakdown by vertical
        by_vertical: dict[str, dict[str, Any]] = {}
        for vertical in MarketVertical:
            vertical_positions = [p for p in closed if p.vertical == vertical]
            if vertical_positions:
                v_pnls = [p.pnl or 0 for p in vertical_positions]
                by_vertical[vertical.value] = {
                    "trades": len(vertical_positions),
                    "win_rate": sum(1 for p in v_pnls if p > 0) / len(v_pnls),
                    "total_pnl": sum(v_pnls),
                    "avg_pnl": np.mean(v_pnls),
                }

        return {
            "total_trades": len(closed),
            "win_rate": wins / len(closed),
            "avg_pnl": np.mean(pnls),
            "total_pnl": sum(pnls),
            "by_vertical": by_vertical,
        }


def run_no_bias_scan(
    snapshots_dir: Path | None = None,
    bankroll: float = 10000.0,
    dry_run: bool = True,
    max_positions: int = 10,
    min_mispricing_ratio: float = 3.0,
    min_volume_usd: float = 10000.0,
    max_yes_price: float = 0.30,
    min_edge: float = 0.05,
    config: PolymarketConfig | None = None,
) -> dict[str, Any]:
    """Run complete NO bias scan and execute trades.

    Args:
        snapshots_dir: Directory with market snapshots
        bankroll: Available capital
        dry_run: If True, don't execute trades
        max_positions: Maximum positions to take
        min_mispricing_ratio: Minimum YES_price/base_rate ratio
        min_volume_usd: Minimum market volume
        max_yes_price: Maximum YES price to consider
        min_edge: Minimum expected edge
        config: Optional PolymarketConfig for live trading

    Returns:
        Dictionary with scan results and trades
    """
    now = datetime.now(UTC)
    logger.info("Starting NO bias scan at %s", now.isoformat())

    # Load markets from snapshot
    if snapshots_dir is None:
        snapshots_dir = Path("data")

    markets: list[dict[str, Any]] = []
    if snapshots_dir.exists():
        snapshot_files = sorted(snapshots_dir.glob("snapshot_*.json"))
        if snapshot_files:
            try:
                data = json.loads(snapshot_files[-1].read_text())
                markets = data.get("markets", [])
                logger.info("Loaded %d markets from %s", len(markets), snapshot_files[-1])
            except (json.JSONDecodeError, FileNotFoundError) as e:
                logger.warning("Failed to load snapshot: %s", e)

    # Scan for signals
    result = scan_markets_for_no_bias(
        markets,
        min_mispricing_ratio=min_mispricing_ratio,
        min_volume_usd=min_volume_usd,
        max_yes_price=max_yes_price,
        min_edge=min_edge,
    )

    # Execute trades
    tracker = NoBiasTracker()
    positions: list[NoBiasPosition] = []

    for signal in result.signals[:max_positions]:
        position = tracker.open_position(signal, bankroll, dry_run=dry_run, config=config)
        if position:
            positions.append(position)
            logger.info(
                "Opened NO position: %s at %.3f, edge=%.1%%, conf=%.1f%%",
                signal.market_question[:50],
                signal.no_bid,
                signal.edge,
                signal.confidence,
            )

    # Update result with positions
    result.positions_taken = len(positions)
    result.total_capital_deployed = sum(p.position_size_usd for p in positions)

    return result.to_dict()


def get_no_bias_performance(tracker: NoBiasTracker | None = None) -> dict[str, Any]:
    """Get NO bias strategy performance summary.

    Args:
        tracker: Optional tracker instance

    Returns:
        Performance summary dict
    """
    if tracker is None:
        tracker = NoBiasTracker()

    summary = tracker.get_performance_summary()
    open_positions = tracker.get_open_positions()

    return {
        "summary": summary,
        "open_positions": [
            {
                "position_id": p.position_id,
                "market": p.market_question,
                "vertical": p.vertical.value,
                "entry_price": p.entry_no_price,
                "position_size": p.position_size_usd,
                "expected_edge": p.expected_edge,
                "order_id": p.order_id,
                "order_status": p.order_status,
                "fill_price": p.fill_price,
                "contracts": p.contracts,
            }
            for p in open_positions
        ],
        "open_count": len(open_positions),
        "timestamp": datetime.now(UTC).isoformat(),
    }
