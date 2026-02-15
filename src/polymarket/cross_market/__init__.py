"""Cross-market arbitrage for prediction markets.

Monitors identical events across Polymarket, Kalshi, and other venues
to identify arbitrage opportunities where YES+NO prices sum to <100c.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class CrossMarketEvent:
    """A normalized event across prediction market venues.

    Attributes:
        event_id: Normalized event identifier
        title: Normalized event title
        description: Event description
        category: Event category (politics, crypto, sports, etc.)
        resolution_date: When the event resolves
        resolution_source: How the event resolves (e.g., "CNN call", "official data")
        venues: Dict of venue -> VenueMarket
    """

    event_id: str
    title: str
    description: str
    category: str
    resolution_date: datetime | None
    resolution_source: str
    venues: dict[str, "VenueMarket"]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "event_id": self.event_id,
            "title": self.title,
            "description": self.description,
            "category": self.category,
            "resolution_date": self.resolution_date.isoformat() if self.resolution_date else None,
            "resolution_source": self.resolution_source,
            "venues": {k: v.to_dict() for k, v in self.venues.items()},
        }


@dataclass(frozen=True)
class VenueMarket:
    """Market data from a specific venue.

    Attributes:
        venue: Venue name (polymarket, kalshi, etc.)
        market_id: Venue-specific market ID
        token_id_yes: YES token/contract ID
        token_id_no: NO token/contract ID
        yes_price: Current YES price (0-1)
        no_price: Current NO price (0-1)
        yes_ask: Best ask for YES (what you'd pay to buy)
        no_ask: Best ask for NO (what you'd pay to buy)
        yes_bid: Best bid for YES (what you'd get if selling)
        no_bid: Best bid for NO (what you'd get if selling)
        volume_24h: 24h trading volume
        liquidity: Available liquidity at best prices
        fees: Fee structure for this venue
        last_updated: Timestamp of price data
    """

    venue: str
    market_id: str
    token_id_yes: str
    token_id_no: str
    yes_price: float | None = None
    no_price: float | None = None
    yes_ask: float | None = None
    no_ask: float | None = None
    yes_bid: float | None = None
    no_bid: float | None = None
    volume_24h: float = 0.0
    liquidity: float = 0.0
    fees: "FeeSchedule" | None = None
    last_updated: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "venue": self.venue,
            "market_id": self.market_id,
            "token_id_yes": self.token_id_yes,
            "token_id_no": self.token_id_no,
            "yes_price": self.yes_price,
            "no_price": self.no_price,
            "yes_ask": self.yes_ask,
            "no_ask": self.no_ask,
            "yes_bid": self.yes_bid,
            "no_bid": self.no_bid,
            "volume_24h": self.volume_24h,
            "liquidity": self.liquidity,
            "fees": self.fees.to_dict() if self.fees else None,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
        }


@dataclass(frozen=True)
class FeeSchedule:
    """Fee structure for a prediction market venue.

    Attributes:
        taker_fee: Fee on taking liquidity (e.g., 0.02 for 2%)
        maker_fee: Fee on providing liquidity
        withdrawal_fee: Fixed withdrawal fee
        deposit_fee: Fixed deposit fee
        max_fee: Maximum fee cap per trade
    """

    taker_fee: float = 0.0
    maker_fee: float = 0.0
    withdrawal_fee: float = 0.0
    deposit_fee: float = 0.0
    max_fee: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "taker_fee": self.taker_fee,
            "maker_fee": self.maker_fee,
            "withdrawal_fee": self.withdrawal_fee,
            "deposit_fee": self.deposit_fee,
            "max_fee": self.max_fee,
        }


@dataclass(frozen=True)
class ArbitrageOpportunity:
    """An identified arbitrage opportunity.

    Attributes:
        event: The underlying event
        venue_yes: Venue to buy YES
        venue_no: Venue to buy NO
        yes_price: Price to buy YES
        no_price: Price to buy NO
        gross_spread: Gross profit (1 - sum of prices)
        net_spread: Profit after fees
        fees_yes: Fees for YES position
        fees_no: Fees for NO position
        total_fees: Combined fees
        confidence: Match confidence (0-1)
        timestamp: When opportunity was identified
    """

    event: CrossMarketEvent
    venue_yes: str
    venue_no: str
    yes_price: float
    no_price: float
    gross_spread: float
    net_spread: float
    fees_yes: float
    fees_no: float
    total_fees: float
    confidence: float
    timestamp: datetime

    @property
    def roi(self) -> float:
        """Return on investment (net spread / capital deployed)."""
        capital = self.yes_price + self.no_price
        if capital <= 0:
            return 0.0
        return self.net_spread / capital

    @property
    def annualized_return(self) -> float | None:
        """Estimate annualized return if held to resolution."""
        if self.event.resolution_date is None:
            return None
        days_to_resolution = (self.event.resolution_date - self.timestamp).days
        if days_to_resolution <= 0:
            return None
        return (1 + self.roi) ** (365 / days_to_resolution) - 1

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "event": self.event.to_dict(),
            "venue_yes": self.venue_yes,
            "venue_no": self.venue_no,
            "yes_price": self.yes_price,
            "no_price": self.no_price,
            "gross_spread": self.gross_spread,
            "net_spread": self.net_spread,
            "fees_yes": self.fees_yes,
            "fees_no": self.fees_no,
            "total_fees": self.total_fees,
            "roi": self.roi,
            "annualized_return": self.annualized_return,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass(frozen=True)
class PaperTrade:
    """A paper trade for tracking arbitrage performance.

    Attributes:
        trade_id: Unique trade identifier
        opportunity: The opportunity that triggered this trade
        position_size: Number of contracts per side
        entry_yes_price: Entry price for YES
        entry_no_price: Entry price for NO
        entry_time: When trade was entered
        exit_yes_price: Exit price for YES (if closed)
        exit_no_price: Exit price for NO (if closed)
        exit_time: When trade was closed
        status: open, closed, or held_to_resolution
        realized_pnl: Realized PnL (if closed)
        theoretical_pnl: Theoretical PnL at current prices
    """

    trade_id: str
    opportunity: ArbitrageOpportunity
    position_size: float
    entry_yes_price: float
    entry_no_price: float
    entry_time: datetime
    exit_yes_price: float | None = None
    exit_no_price: float | None = None
    exit_time: datetime | None = None
    status: str = "open"  # open, closed, held_to_resolution
    realized_pnl: float | None = None
    theoretical_pnl: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "trade_id": self.trade_id,
            "opportunity": self.opportunity.to_dict(),
            "position_size": self.position_size,
            "entry_yes_price": self.entry_yes_price,
            "entry_no_price": self.entry_no_price,
            "entry_time": self.entry_time.isoformat(),
            "exit_yes_price": self.exit_yes_price,
            "exit_no_price": self.exit_no_price,
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "status": self.status,
            "realized_pnl": self.realized_pnl,
            "theoretical_pnl": self.theoretical_pnl,
        }


# Default fee schedules per venue
KALSHI_FEE_SCHEDULE = FeeSchedule(
    taker_fee=0.0,  # No taker fee on Kalshi
    maker_fee=0.0,  # No maker fee
    withdrawal_fee=0.0,  # Free ACH withdrawal
    deposit_fee=0.0,
    max_fee=0.05,  # 5% cap per market
)

POLYMARKET_FEE_SCHEDULE = FeeSchedule(
    taker_fee=0.0,  # No trading fees currently
    maker_fee=0.0,
    withdrawal_fee=0.02,  # 2% withdrawal fee
    deposit_fee=0.0,
    max_fee=None,
)
