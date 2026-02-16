"""Both-sides mispricing arbitrage strategy for BTC 5m/15m markets.

Detects arbitrage opportunities when both UP and DOWN sides of BTC interval
markets trade at prices summing to < $1.00, creating risk-free profit potential.

Strategy based on account88888's successful approach:
- Monitor BTC 5m/15m markets continuously
- Buy BOTH up AND down when sum < 1.0 - fee_buffer
- Hold until settlement (one side pays $1, other $0)
- Optional: timeframe alignment filter (5m signals aligned with 15m trend)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .site import CryptoIntervalEvent


# Strategy constants
TAKER_FEE_RATE = Decimal("0.02")  # 2% taker fee per side
TOTAL_FEE_BUFFER = Decimal("0.04")  # ~4% total (2% per side)
MIN_SPREAD_AFTER_FEES = Decimal("0.02")  # Minimum 2% profit after fees
MIN_MARKET_VOLUME = Decimal("5000")  # $5k minimum volume
MAX_HOLD_MINUTES_5M = 5  # Max hold time for 5m markets
MAX_HOLD_MINUTES_15M = 15  # Max hold time for 15m markets
MAX_POSITION_SIZE = Decimal("500")  # Max $500 per side
DEFAULT_POSITION_SIZE = Decimal("100")  # Default $100 per side


@dataclass(frozen=True)
class BothSidesOpportunity:
    """A detected both-sides mispricing opportunity.

    Attributes:
        market_id: Polymarket market ID
        event_id: Polymarket event ID
        interval: "5m" or "15m"
        up_token_id: Token ID for UP side
        down_token_id: Token ID for DOWN side
        up_price: Current price of UP side (ask)
        down_price: Current price of DOWN side (ask)
        price_sum: Sum of both prices
        fee_buffer: Fee buffer applied (typically 4%)
        spread: Risk-free spread (1.0 - price_sum - fees)
        spread_after_fees: Profit after accounting for fees
        aligned_15m: Whether 5m signal aligns with 15m trend
        confidence: Confidence score (0-1)
        timestamp: Detection timestamp
        market_metadata: Additional market info
    """

    market_id: str
    event_id: str
    interval: str
    up_token_id: str
    down_token_id: str
    up_price: Decimal
    down_price: Decimal
    price_sum: Decimal
    fee_buffer: Decimal
    spread: Decimal
    spread_after_fees: Decimal
    aligned_15m: bool | None
    confidence: Decimal
    timestamp: str
    market_metadata: dict

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "market_id": self.market_id,
            "event_id": self.event_id,
            "interval": self.interval,
            "up_token_id": self.up_token_id,
            "down_token_id": self.down_token_id,
            "up_price": str(self.up_price),
            "down_price": str(self.down_price),
            "price_sum": str(self.price_sum),
            "fee_buffer": str(self.fee_buffer),
            "spread": str(self.spread),
            "spread_after_fees": str(self.spread_after_fees),
            "aligned_15m": self.aligned_15m,
            "confidence": str(self.confidence),
            "timestamp": self.timestamp,
            "market_metadata": self.market_metadata,
        }

    @property
    def is_valid(self) -> bool:
        """Check if opportunity meets all criteria."""
        return (
            self.spread_after_fees >= MIN_SPREAD_AFTER_FEES
            and self.confidence >= Decimal("0.5")
            and self.price_sum < Decimal("1.0") - self.fee_buffer
        )

    @property
    def total_position_cost(self) -> Decimal:
        """Calculate total cost to enter both sides."""
        return self.up_price + self.down_price

    @property
    def guaranteed_payout(self) -> Decimal:
        """Guaranteed payout (always $1 per $1 position)."""
        return Decimal("1.0")

    @property
    def gross_profit(self) -> Decimal:
        """Gross profit before fees."""
        return self.guaranteed_payout - self.total_position_cost

    @property
    def fees(self) -> Decimal:
        """Total fees (2% per side)."""
        return self.total_position_cost * TOTAL_FEE_BUFFER

    @property
    def net_profit(self) -> Decimal:
        """Net profit after fees."""
        return self.gross_profit - self.fees


@dataclass(frozen=True)
class BothSidesTrade:
    """A paper trade record for both-sides arbitrage.

    Attributes:
        trade_id: Unique trade ID
        timestamp: Entry timestamp
        market_id: Polymarket market ID
        event_id: Polymarket event ID
        interval: "5m" or "15m"
        up_token_id: Token ID for UP side
        down_token_id: Token ID for DOWN side
        up_entry_price: Entry price for UP side
        down_entry_price: Entry price for DOWN side
        position_size: Size per side in contracts
        total_cost: Total cost to enter
        spread_at_entry: Spread captured at entry
        aligned_15m: Whether 5m signal aligned with 15m
        status: "open", "closed", or "expired"
        exit_timestamp: Exit timestamp
        winning_side: "up", "down", or None (if not resolved)
        payout: Actual payout received
        net_pnl: Net profit/loss
        close_reason: "resolution", "manual", "timeout"
    """

    trade_id: str
    timestamp: str
    market_id: str
    event_id: str
    interval: str
    up_token_id: str
    down_token_id: str
    up_entry_price: Decimal
    down_entry_price: Decimal
    position_size: Decimal
    total_cost: Decimal
    spread_at_entry: Decimal
    aligned_15m: bool | None
    status: str = "open"
    exit_timestamp: str | None = None
    winning_side: str | None = None
    payout: Decimal | None = None
    net_pnl: Decimal | None = None
    close_reason: str | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "trade_id": self.trade_id,
            "timestamp": self.timestamp,
            "market_id": self.market_id,
            "event_id": self.event_id,
            "interval": self.interval,
            "up_token_id": self.up_token_id,
            "down_token_id": self.down_token_id,
            "up_entry_price": str(self.up_entry_price),
            "down_entry_price": str(self.down_entry_price),
            "position_size": str(self.position_size),
            "total_cost": str(self.total_cost),
            "spread_at_entry": str(self.spread_at_entry),
            "aligned_15m": self.aligned_15m,
            "status": self.status,
            "exit_timestamp": self.exit_timestamp,
            "winning_side": self.winning_side,
            "payout": str(self.payout) if self.payout else None,
            "net_pnl": str(self.net_pnl) if self.net_pnl else None,
            "close_reason": self.close_reason,
        }


def calculate_spread(up_price: Decimal, down_price: Decimal) -> Decimal:
    """Calculate the risk-free spread.

    Args:
        up_price: Price of UP side
        down_price: Price of DOWN side

    Returns:
        Spread as decimal (1.0 - up_price - down_price).
    """
    return Decimal("1.0") - up_price - down_price


def calculate_spread_after_fees(up_price: Decimal, down_price: Decimal) -> Decimal:
    """Calculate spread after accounting for fees.

    Args:
        up_price: Price of UP side
        down_price: Price of DOWN side

    Returns:
        Net profit after fees.
    """
    total_cost = up_price + down_price
    gross_profit = Decimal("1.0") - total_cost
    fees = total_cost * TOTAL_FEE_BUFFER
    return gross_profit - fees


def check_mispricing(
    up_price: Decimal,
    down_price: Decimal,
    fee_buffer: Decimal = TOTAL_FEE_BUFFER,
    min_spread: Decimal = MIN_SPREAD_AFTER_FEES,
) -> tuple[bool, Decimal]:
    """Check if both sides are mispriced.

    Args:
        up_price: Price of UP side
        down_price: Price of DOWN side
        fee_buffer: Fee buffer to account for
        min_spread: Minimum spread required after fees

    Returns:
        Tuple of (is_mispriced, spread_after_fees).
    """
    price_sum = up_price + down_price
    spread_after_fees = calculate_spread_after_fees(up_price, down_price)

    is_mispriced = (
        price_sum < Decimal("1.0") - fee_buffer
        and spread_after_fees >= min_spread
    )

    return is_mispriced, spread_after_fees


def calculate_confidence(
    up_price: Decimal,
    down_price: Decimal,
    volume: Decimal | None = None,
    aligned_15m: bool | None = None,
) -> Decimal:
    """Calculate confidence score for opportunity.

    Factors:
    - Price sanity check (prices should be reasonable)
    - Volume on market
    - Timeframe alignment (if applicable)

    Args:
        up_price: Price of UP side
        down_price: Price of DOWN side
        volume: Market volume
        aligned_15m: Whether 5m signal aligns with 15m

    Returns:
        Confidence score 0-1.
    """
    confidence = Decimal("0.5")

    # Price sanity check - prices should be between 0.01 and 0.99
    if Decimal("0.01") <= up_price <= Decimal("0.99") and Decimal("0.01") <= down_price <= Decimal("0.99"):
        confidence += Decimal("0.1")

    # Neither side should be extremely cheap (indicates resolution)
    if up_price > Decimal("0.05") and down_price > Decimal("0.05"):
        confidence += Decimal("0.1")

    # Volume factor
    if volume is not None:
        if volume >= Decimal("100000"):
            confidence += Decimal("0.15")
        elif volume >= Decimal("50000"):
            confidence += Decimal("0.1")
        elif volume >= Decimal("10000"):
            confidence += Decimal("0.05")

    # Timeframe alignment bonus
    if aligned_15m is True:
        confidence += Decimal("0.15")

    # Cap at 1.0
    return min(confidence, Decimal("1.0"))


def determine_alignment_15m(
    market_5m: CryptoIntervalEvent,
    markets_15m: list[CryptoIntervalEvent],
) -> bool | None:
    """Determine if a 5m market aligns with the 15m trend direction.

    Args:
        market_5m: The 5m market to check
        markets_15m: List of active 15m markets

    Returns:
        True if aligned, False if contra, None if can't determine.
    """
    # Find matching 15m market by extracting asset (BTC) from title
    # 5m titles: "Bitcoin (BTC) up or down in next 5 min?"
    # 15m titles: "Bitcoin (BTC) up or down in next 15 min?"

    if "bitcoin" not in market_5m.title.lower():
        return None

    # Find BTC 15m market
    btc_15m = None
    for m15 in markets_15m:
        if "bitcoin" in m15.title.lower():
            btc_15m = m15
            break

    if btc_15m is None:
        return None

    # Get 15m price data to determine trend
    # For now, use mid price as trend indicator
    # A more sophisticated approach would use orderbook imbalance
    try:
        from . import clob

        up_book = clob.get_book(btc_15m.clob_token_ids[0])
        down_book = clob.get_book(btc_15m.clob_token_ids[1])

        up_mid = _calculate_mid(up_book)
        down_mid = _calculate_mid(down_book)

        if up_mid is None or down_mid is None:
            return None

        # If UP > DOWN in 15m, trend is up
        trend_up = up_mid > down_mid

        # For 5m alignment: if 15m trend is up, 5m UP should be favored
        # We consider it "aligned" if we're buying the trending side
        # For both-sides arb, we buy both, so alignment affects confidence
        return trend_up

    except Exception:
        return None


def _calculate_mid(book: dict) -> Decimal | None:
    """Calculate mid price from orderbook."""
    bids = book.get("bids", [])
    asks = book.get("asks", [])

    if not bids or not asks:
        return None

    best_bid = Decimal(str(bids[0]["price"])) if bids else None
    best_ask = Decimal(str(asks[0]["price"])) if asks else None

    if best_bid is None or best_ask is None:
        return None

    return (best_bid + best_ask) / Decimal("2")


def get_best_ask(book: dict) -> Decimal | None:
    """Get best ask price from orderbook."""
    asks = book.get("asks", [])
    if not asks:
        return None
    return Decimal(str(asks[0]["price"]))


def find_both_sides_opportunities(
    markets: list[CryptoIntervalEvent],
    interval: str = "5m",
    check_alignment: bool = False,
    markets_15m: list[CryptoIntervalEvent] | None = None,
    min_spread: Decimal = MIN_SPREAD_AFTER_FEES,
) -> list[BothSidesOpportunity]:
    """Find both-sides mispricing opportunities.

    Args:
        markets: List of interval markets to analyze
        interval: "5m" or "15m"
        check_alignment: Whether to check 15m alignment for 5m markets
        markets_15m: List of 15m markets (required if check_alignment=True)
        min_spread: Minimum spread required

    Returns:
        List of valid opportunities.
    """
    opportunities: list[BothSidesOpportunity] = []

    for market in markets:
        # Skip non-BTC markets for now
        if "bitcoin" not in market.title.lower() and "btc" not in market.title.lower():
            continue

        try:
            from . import clob

            # Fetch orderbooks for both sides
            up_book = clob.get_book(market.clob_token_ids[0])
            down_book = clob.get_book(market.clob_token_ids[1])

            # Get best ask prices (what we'd pay to buy)
            up_ask = get_best_ask(up_book)
            down_ask = get_best_ask(down_book)

            if up_ask is None or down_ask is None:
                continue

            # Check for mispricing
            is_mispriced, spread_after_fees = check_mispricing(
                up_ask, down_ask, min_spread=min_spread
            )

            if not is_mispriced:
                continue

            # Check alignment if requested
            aligned_15m = None
            if check_alignment and interval == "5m" and markets_15m:
                aligned_15m = determine_alignment_15m(market, markets_15m)

            # Calculate confidence
            confidence = calculate_confidence(
                up_ask, down_ask, volume=None, aligned_15m=aligned_15m
            )

            price_sum = up_ask + down_ask
            spread = Decimal("1.0") - price_sum

            opportunities.append(BothSidesOpportunity(
                market_id=market.market_id,
                event_id=market.event_id,
                interval=interval,
                up_token_id=market.clob_token_ids[0],
                down_token_id=market.clob_token_ids[1],
                up_price=up_ask,
                down_price=down_ask,
                price_sum=price_sum,
                fee_buffer=TOTAL_FEE_BUFFER,
                spread=spread,
                spread_after_fees=spread_after_fees,
                aligned_15m=aligned_15m,
                confidence=confidence,
                timestamp=datetime.now(UTC).isoformat(),
                market_metadata={
                    "title": market.title,
                    "question": market.question,
                    "end_date": market.end_date,
                },
            ))

        except Exception:
            # Skip markets that fail to fetch
            continue

    # Sort by spread (highest first)
    opportunities.sort(key=lambda o: o.spread_after_fees, reverse=True)
    return opportunities


def execute_paper_trade(
    opportunity: BothSidesOpportunity,
    position_size: Decimal = DEFAULT_POSITION_SIZE,
    trade_id: str | None = None,
) -> BothSidesTrade:
    """Execute a paper trade for an opportunity.

    Args:
        opportunity: Both-sides opportunity
        position_size: Position size per side
        trade_id: Optional trade ID

    Returns:
        Paper trade record.
    """
    if trade_id is None:
        trade_id = (
            f"bsa_{opportunity.interval}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_"
            f"{opportunity.market_id[:8]}"
        )

    total_cost = opportunity.up_price + opportunity.down_price

    return BothSidesTrade(
        trade_id=trade_id,
        timestamp=opportunity.timestamp,
        market_id=opportunity.market_id,
        event_id=opportunity.event_id,
        interval=opportunity.interval,
        up_token_id=opportunity.up_token_id,
        down_token_id=opportunity.down_token_id,
        up_entry_price=opportunity.up_price,
        down_entry_price=opportunity.down_price,
        position_size=position_size,
        total_cost=total_cost,
        spread_at_entry=opportunity.spread_after_fees,
        aligned_15m=opportunity.aligned_15m,
        status="open",
    )


def close_trade(
    trade: BothSidesTrade,
    winning_side: str,
    close_reason: str = "resolution",
) -> BothSidesTrade:
    """Close a paper trade at settlement.

    Args:
        trade: The trade to close
        winning_side: "up" or "down" - which side won
        close_reason: Reason for closing

    Returns:
        Updated trade record.
    """
    payout = Decimal("1.0") * trade.position_size
    net_pnl = payout - trade.total_cost - (trade.total_cost * TOTAL_FEE_BUFFER)

    return BothSidesTrade(
        trade_id=trade.trade_id,
        timestamp=trade.timestamp,
        market_id=trade.market_id,
        event_id=trade.event_id,
        interval=trade.interval,
        up_token_id=trade.up_token_id,
        down_token_id=trade.down_token_id,
        up_entry_price=trade.up_entry_price,
        down_entry_price=trade.down_entry_price,
        position_size=trade.position_size,
        total_cost=trade.total_cost,
        spread_at_entry=trade.spread_at_entry,
        aligned_15m=trade.aligned_15m,
        status="closed",
        exit_timestamp=datetime.now(UTC).isoformat(),
        winning_side=winning_side,
        payout=payout,
        net_pnl=net_pnl,
        close_reason=close_reason,
    )


class BothSidesArbitrageStrategy:
    """Main strategy class for both-sides mispricing arbitrage."""

    def __init__(
        self,
        position_size: Decimal = DEFAULT_POSITION_SIZE,
        check_alignment: bool = False,
        data_dir: Path | None = None,
    ) -> None:
        """Initialize strategy.

        Args:
            position_size: Position size per side
            check_alignment: Whether to check 15m alignment for 5m
            data_dir: Directory for trade logs
        """
        self.position_size = position_size
        self.check_alignment = check_alignment
        self.data_dir = data_dir or Path("data/both_sides_arb")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.opportunities_file = self.data_dir / "opportunities.jsonl"
        self.trades_file = self.data_dir / "trades.jsonl"

    def scan(
        self,
        interval: str = "5m",
        min_spread: Decimal = MIN_SPREAD_AFTER_FEES,
    ) -> list[BothSidesOpportunity]:
        """Scan for both-sides opportunities.

        Args:
            interval: "5m" or "15m"
            min_spread: Minimum spread required

        Returns:
            List of opportunities found.
        """
        from .site import extract_crypto_interval_events, fetch_crypto_interval_page, parse_next_data

        # Fetch markets for the specified interval
        html = fetch_crypto_interval_page(interval.upper())
        data = parse_next_data(html)
        markets = extract_crypto_interval_events(data, interval_slug=interval.upper())

        # Fetch 15m markets for alignment check if needed
        markets_15m = None
        if self.check_alignment and interval == "5m":
            try:
                html_15m = fetch_crypto_interval_page("15M")
                data_15m = parse_next_data(html_15m)
                markets_15m = extract_crypto_interval_events(data_15m, interval_slug="15M")
            except Exception:
                markets_15m = None

        # Find opportunities
        opportunities = find_both_sides_opportunities(
            markets=markets,
            interval=interval,
            check_alignment=self.check_alignment,
            markets_15m=markets_15m,
            min_spread=min_spread,
        )

        # Log opportunities
        self._log_opportunities(opportunities)

        return opportunities

    def paper_trade(self, opportunity: BothSidesOpportunity) -> BothSidesTrade:
        """Execute a paper trade.

        Args:
            opportunity: Both-sides opportunity

        Returns:
            Paper trade record.
        """
        trade = execute_paper_trade(opportunity, self.position_size)
        self._log_trade(trade)
        return trade

    def _log_opportunities(self, opportunities: list[BothSidesOpportunity]) -> None:
        """Append opportunities to log file."""
        with open(self.opportunities_file, "a") as f:
            for opp in opportunities:
                f.write(json.dumps(opp.to_dict()) + "\n")

    def _log_trade(self, trade: BothSidesTrade) -> None:
        """Append trade to log file."""
        with open(self.trades_file, "a") as f:
            f.write(json.dumps(trade.to_dict()) + "\n")

    def get_stats(self) -> dict:
        """Get strategy statistics."""
        stats = {
            "total_opportunities": 0,
            "total_trades": 0,
            "open_trades": 0,
            "closed_trades": 0,
            "total_pnl": Decimal("0"),
            "avg_spread": Decimal("0"),
            "aligned_trades": 0,
            "non_aligned_trades": 0,
        }

        # Count opportunities
        if self.opportunities_file.exists():
            with open(self.opportunities_file) as f:
                stats["total_opportunities"] = sum(1 for _ in f)

        # Calculate trade stats
        if self.trades_file.exists():
            trades: list[BothSidesTrade] = []
            with open(self.trades_file) as f:
                for line in f:
                    data = json.loads(line)
                    trade = BothSidesTrade(
                        trade_id=data["trade_id"],
                        timestamp=data["timestamp"],
                        market_id=data["market_id"],
                        event_id=data["event_id"],
                        interval=data["interval"],
                        up_token_id=data["up_token_id"],
                        down_token_id=data["down_token_id"],
                        up_entry_price=Decimal(data["up_entry_price"]),
                        down_entry_price=Decimal(data["down_entry_price"]),
                        position_size=Decimal(data["position_size"]),
                        total_cost=Decimal(data["total_cost"]),
                        spread_at_entry=Decimal(data["spread_at_entry"]),
                        aligned_15m=data.get("aligned_15m"),
                        status=data["status"],
                        exit_timestamp=data.get("exit_timestamp"),
                        winning_side=data.get("winning_side"),
                        payout=Decimal(data["payout"]) if data.get("payout") else None,
                        net_pnl=Decimal(data["net_pnl"]) if data.get("net_pnl") else None,
                        close_reason=data.get("close_reason"),
                    )
                    trades.append(trade)

            stats["total_trades"] = len(trades)
            stats["open_trades"] = sum(1 for t in trades if t.status == "open")
            stats["closed_trades"] = sum(1 for t in trades if t.status == "closed")
            stats["aligned_trades"] = sum(
                1 for t in trades if t.aligned_15m is True
            )
            stats["non_aligned_trades"] = sum(
                1 for t in trades if t.aligned_15m is False
            )

            closed_pnls = [t.net_pnl for t in trades if t.net_pnl is not None]
            if closed_pnls:
                stats["total_pnl"] = sum(closed_pnls, Decimal("0"))

            spreads = [t.spread_at_entry for t in trades]
            if spreads:
                stats["avg_spread"] = sum(spreads, Decimal("0")) / len(spreads)

        return stats
