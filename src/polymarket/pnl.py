"""PnL verification engine for Polymarket.

Computes true realized PnL and realizable liquidation value from fills data.
Handles buy/sell of YES/NO shares with proper cost basis tracking.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from decimal import Decimal
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class Fill:
    """Represents a single fill (executed trade).

    Attributes:
        token_id: The token ID traded (YES or NO token)
        side: 'buy' or 'sell'
        size: Number of shares/contracts
        price: Execution price (0.0 to 1.0)
        fee: Trading fee paid (in USDC)
        timestamp: ISO8601 timestamp of the fill
        transaction_hash: Optional blockchain transaction hash
    """

    token_id: str
    side: str
    size: Decimal
    price: Decimal
    fee: Decimal
    timestamp: str
    transaction_hash: str | None = None

    @classmethod
    def from_dict(cls, data: dict) -> Fill:
        """Create a Fill from a dictionary (API response or JSON)."""
        # Handle various field naming conventions from Polymarket API
        token_id = data.get("token_id") or data.get("asset_id") or data.get("tokenID") or ""
        side = (data.get("side") or data.get("trade_side") or "buy").lower()

        # Size might be called 'size', 'amount', 'takerAmount', etc.
        size_val = data.get("size") or data.get("amount") or data.get("takerAmount") or "0"

        # Price might be called 'price', 'execution_price', etc.
        price_val = (
            data.get("price") or data.get("execution_price") or data.get("priceInNative") or "0"
        )

        # Fee might be called 'fee', 'trade_fee', etc.
        fee_val = data.get("fee") or data.get("trade_fee") or data.get("gas_fee") or "0"

        timestamp = (
            data.get("timestamp") or data.get("created_at") or data.get("transaction_time") or ""
        )
        tx_hash = data.get("transaction_hash") or data.get("tx_hash") or data.get("transactionHash")

        return cls(
            token_id=str(token_id),
            side=side,
            size=Decimal(str(size_val)),
            price=Decimal(str(price_val)),
            fee=Decimal(str(fee_val)),
            timestamp=str(timestamp),
            transaction_hash=tx_hash,
        )


@dataclass
class Position:
    """Tracks a position in a specific token.

    Uses average cost basis for realized PnL calculations.
    """

    token_id: str
    net_size: Decimal = field(default_factory=lambda: Decimal("0"))
    total_cost: Decimal = field(default_factory=lambda: Decimal("0"))
    total_fees: Decimal = field(default_factory=lambda: Decimal("0"))
    realized_pnl: Decimal = field(default_factory=lambda: Decimal("0"))
    buy_count: int = 0
    sell_count: int = 0

    @property
    def avg_cost_basis(self) -> Decimal:
        """Average cost per share for current position."""
        if self.net_size <= 0:
            return Decimal("0")
        return self.total_cost / self.net_size

    def add_buy(self, size: Decimal, price: Decimal, fee: Decimal) -> None:
        """Record a buy fill."""
        cost = size * price
        self.net_size += size
        self.total_cost += cost
        self.total_fees += fee
        self.buy_count += 1

    def add_sell(self, size: Decimal, price: Decimal, fee: Decimal) -> Decimal:
        """Record a sell fill and return the realized PnL.

        Uses average cost basis for the shares being sold.
        """
        if self.net_size <= 0:
            # Short selling - track separately
            proceeds = size * price
            self.net_size -= size
            self.total_fees += fee
            self.sell_count += 1
            # For shorts, realized PnL is calculated on close
            return Decimal("0")

        # Calculate cost basis for shares being sold
        sell_size = min(size, self.net_size)
        cost_basis = sell_size * self.avg_cost_basis
        proceeds = sell_size * price

        # Realized PnL = proceeds - cost basis - fee
        realized = proceeds - cost_basis - fee
        self.realized_pnl += realized

        # Update position
        self.net_size -= sell_size
        self.total_cost -= cost_basis
        self.total_fees += fee
        self.sell_count += 1

        return realized


@dataclass
class BookLevel:
    """Single level in an orderbook."""

    price: Decimal
    size: Decimal


@dataclass
class OrderBook:
    """Orderbook snapshot for liquidation value calculation."""

    token_id: str
    bids: list[BookLevel] = field(default_factory=list)  # Sorted high to low
    asks: list[BookLevel] = field(default_factory=list)  # Sorted low to high

    @classmethod
    def from_dict(cls, token_id: str, data: dict) -> OrderBook:
        """Create OrderBook from CLOB book response."""
        bids = [
            BookLevel(price=Decimal(str(b["price"])), size=Decimal(str(b["size"])))
            for b in data.get("bids", [])
        ]
        asks = [
            BookLevel(price=Decimal(str(a["price"])), size=Decimal(str(a["size"])))
            for a in data.get("asks", [])
        ]
        # Sort bids descending (highest first), asks ascending (lowest first)
        bids.sort(key=lambda x: x.price, reverse=True)
        asks.sort(key=lambda x: x.price)
        return cls(token_id=token_id, bids=bids, asks=asks)

    def get_walk_liquidation_value(self, position_size: Decimal, is_yes: bool = True) -> Decimal:
        """Calculate liquidation value by walking the book.

        For YES positions: sell into bids (highest to lowest)
        For NO positions: effectively sell YES at (1 - bid_price),
                        but we model as buying YES at ask

        Args:
            position_size: Number of shares (positive for long)
            is_yes: Whether this is a YES token

        Returns:
            Total USDC value from liquidating the position
        """
        if position_size <= 0:
            return Decimal("0")

        remaining = position_size
        total_value = Decimal("0")

        if is_yes:
            # Selling YES: walk bids from highest to lowest
            for level in self.bids:
                if remaining <= 0:
                    break
                fill_size = min(remaining, level.size)
                total_value += fill_size * level.price
                remaining -= fill_size
        else:
            # Selling NO: effectively buying YES at ask prices
            # NO position profits when YES goes down
            # To close: buy YES at ask, which costs money
            for level in self.asks:
                if remaining <= 0:
                    break
                fill_size = min(remaining, level.size)
                # Closing NO means buying YES
                total_value -= fill_size * level.price
                remaining -= fill_size

        # If we couldn't fully liquidate, assume remainder at 0 (worst case)
        # or 0.5 (fair value) depending on conservatism preference
        # Using 0 for conservative estimate
        return max(total_value, Decimal("0"))


@dataclass
class PnLReport:
    """Complete PnL verification report."""

    # Summary
    total_fills: int = 0
    unique_tokens: int = 0

    # PnL Breakdown
    realized_pnl: Decimal = field(default_factory=lambda: Decimal("0"))
    unrealized_pnl: Decimal = field(default_factory=lambda: Decimal("0"))
    total_fees: Decimal = field(default_factory=lambda: Decimal("0"))
    net_pnl: Decimal = field(default_factory=lambda: Decimal("0"))

    # Liquidation Analysis
    mark_to_market: Decimal = field(default_factory=lambda: Decimal("0"))
    liquidation_value: Decimal = field(default_factory=lambda: Decimal("0"))
    liquidation_discount: Decimal = field(default_factory=lambda: Decimal("0"))

    # Position Details
    positions: list[dict] = field(default_factory=list)

    # Warnings
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "summary": {
                "total_fills": self.total_fills,
                "unique_tokens": self.unique_tokens,
            },
            "pnl": {
                "realized_pnl": float(self.realized_pnl),
                "unrealized_pnl": float(self.unrealized_pnl),
                "total_fees": float(self.total_fees),
                "net_pnl": float(self.net_pnl),
            },
            "liquidation": {
                "mark_to_market": float(self.mark_to_market),
                "liquidation_value": float(self.liquidation_value),
                "liquidation_discount": float(self.liquidation_discount),
                "discount_pct": (
                    float(self.liquidation_discount / self.mark_to_market * 100)
                    if self.mark_to_market > 0
                    else 0.0
                ),
            },
            "positions": self.positions,
            "warnings": self.warnings,
        }


def compute_pnl(
    fills: list[Fill],
    current_prices: dict[str, Decimal] | None = None,
    orderbooks: dict[str, OrderBook] | None = None,
) -> PnLReport:
    """Compute PnL report from a list of fills.

    Args:
        fills: List of Fill objects representing all trades
        current_prices: Optional dict of token_id -> current price for mark-to-market
        orderbooks: Optional dict of token_id -> OrderBook for liquidation value

    Returns:
        PnLReport with full analysis
    """
    report = PnLReport()
    report.total_fills = len(fills)

    # Group fills by token and build positions
    positions: dict[str, Position] = {}

    # Sort fills by timestamp to process chronologically
    sorted_fills = sorted(fills, key=lambda f: f.timestamp)

    for fill in sorted_fills:
        if fill.token_id not in positions:
            positions[fill.token_id] = Position(token_id=fill.token_id)

        pos = positions[fill.token_id]

        if fill.side == "buy":
            pos.add_buy(fill.size, fill.price, fill.fee)
        else:  # sell
            pos.add_sell(fill.size, fill.price, fill.fee)

    report.unique_tokens = len(positions)
    report.total_fees = sum(p.total_fees for p in positions.values())
    report.realized_pnl = sum(p.realized_pnl for p in positions.values())

    # Build position details and compute unrealized PnL
    position_details = []
    total_mtm = Decimal("0")
    total_liquidation = Decimal("0")

    for token_id, pos in positions.items():
        if pos.net_size == 0:
            continue  # Skip closed positions

        # Get current price for mark-to-market
        current_price = Decimal("0.5")  # Default to 0.5 if unknown
        if current_prices and token_id in current_prices:
            current_price = current_prices[token_id]
        elif orderbooks and token_id in orderbooks:
            # Use best bid as conservative price estimate
            book = orderbooks[token_id]
            if book.bids:
                current_price = book.bids[0].price

        # Calculate unrealized PnL
        unrealized = (current_price - pos.avg_cost_basis) * pos.net_size

        # Mark to market = position value at current price
        mtm = current_price * pos.net_size

        # Liquidation value
        liquidation = Decimal("0")
        if orderbooks and token_id in orderbooks:
            # Assume YES token for now - need to determine from context
            liquidation = orderbooks[token_id].get_walk_liquidation_value(pos.net_size, is_yes=True)
        else:
            # Without orderbook, assume 10% discount for estimation
            liquidation = mtm * Decimal("0.9")

        total_mtm += mtm
        total_liquidation += liquidation

        position_details.append(
            {
                "token_id": token_id,
                "net_size": float(pos.net_size),
                "avg_cost_basis": float(pos.avg_cost_basis),
                "current_price": float(current_price),
                "unrealized_pnl": float(unrealized),
                "mark_to_market": float(mtm),
                "liquidation_value": float(liquidation),
                "buy_count": pos.buy_count,
                "sell_count": pos.sell_count,
            }
        )

    report.positions = position_details
    report.unrealized_pnl = Decimal(str(sum(p["unrealized_pnl"] for p in position_details)))
    report.net_pnl = report.realized_pnl + report.unrealized_pnl - report.total_fees
    report.mark_to_market = total_mtm
    report.liquidation_value = total_liquidation
    report.liquidation_discount = total_mtm - total_liquidation

    # Add warnings for edge cases
    if any(p.net_size < 0 for p in positions.values()):
        report.warnings.append("Short positions detected - PnL may be incomplete")

    if total_mtm > 0 and total_liquidation / total_mtm < Decimal("0.5"):
        report.warnings.append("Large liquidation discount - position may be illiquid")

    return report


def load_fills_from_file(path: Path) -> list[Fill]:
    """Load fills from a JSON file.

    Supports both:
    - Array of fill objects: [{...}, {...}]
    - Object with 'fills' key: {"fills": [...]}
    - Object with 'data' key: {"data": [...]}
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    # Handle different JSON structures
    if isinstance(data, list):
        fill_list = data
    elif isinstance(data, dict):
        if "fills" in data:
            fill_list = data["fills"]
        elif "data" in data:
            fill_list = data["data"]
        elif "orders" in data:
            fill_list = data["orders"]
        else:
            # Single fill object
            fill_list = [data]
    else:
        fill_list = []

    return [Fill.from_dict(f) for f in fill_list]


def load_orderbooks_from_file(path: Path) -> dict[str, OrderBook]:
    """Load orderbooks from a JSON file.

    Expected format: {"token_id": {"bids": [...], "asks": [...]}, ...}
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    books = {}
    for token_id, book_data in data.items():
        books[token_id] = OrderBook.from_dict(token_id, book_data)

    return books
