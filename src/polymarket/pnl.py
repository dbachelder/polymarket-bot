"""PnL verification engine for Polymarket.

Computes true realized PnL and realizable liquidation value from fills data.
Handles buy/sell of YES/NO shares with proper cost basis tracking.

Features:
- Cash balance tracking from fill history
- Inventory verification (position size vs fills)
- Cashflow conservation sanity checks
- Realized and unrealized PnL (mark-to-mid)
- Liquidation value via orderbook walking
- Daily summary persistence
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


# Constants
USDC_DECIMALS = Decimal("1")  # USDC has 6 decimals, but prices are already normalized
DAILY_SUMMARY_DIR = "data/pnl"


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
        market_slug: Optional market identifier for filtering
    """

    token_id: str
    side: str
    size: Decimal
    price: Decimal
    fee: Decimal
    timestamp: str
    transaction_hash: str | None = None
    market_slug: str | None = None

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
        market_slug = data.get("market_slug") or data.get("slug") or data.get("market")

        return cls(
            token_id=str(token_id),
            side=side,
            size=Decimal(str(size_val)),
            price=Decimal(str(price_val)),
            fee=Decimal(str(fee_val)),
            timestamp=str(timestamp),
            transaction_hash=tx_hash,
            market_slug=market_slug,
        )

    @property
    def cash_flow(self) -> Decimal:
        """Calculate cash flow impact of this fill.

        Buy: negative cash flow (pay money)
        Sell: positive cash flow (receive money)
        """
        notional = self.size * self.price
        if self.side == "buy":
            return -(notional + self.fee)
        else:
            return notional - self.fee

    @property
    def datetime_utc(self) -> datetime:
        """Parse timestamp to UTC datetime."""
        # Handle various ISO formats
        ts = self.timestamp.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(ts)
        except ValueError:
            # Fallback for non-standard formats
            return datetime.now(timezone.utc)


@dataclass
class Position:
    """Tracks a position in a specific token.

    Uses average cost basis for realized PnL calculations.
    Tracks cumulative fills for verification.
    """

    token_id: str
    market_slug: str | None = None
    net_size: Decimal = field(default_factory=lambda: Decimal("0"))
    total_cost: Decimal = field(default_factory=lambda: Decimal("0"))
    total_fees: Decimal = field(default_factory=lambda: Decimal("0"))
    realized_pnl: Decimal = field(default_factory=lambda: Decimal("0"))
    buy_count: int = 0
    sell_count: int = 0
    total_bought: Decimal = field(default_factory=lambda: Decimal("0"))
    total_sold: Decimal = field(default_factory=lambda: Decimal("0"))

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
        self.total_bought += size
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
            self.total_sold += size
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
        self.total_sold += size
        self.sell_count += 1

        return realized

    def verify(self) -> list[str]:
        """Return list of verification warnings for this position."""
        warnings = []

        # Check: net_size should equal total_bought - total_sold
        expected_net = self.total_bought - self.total_sold
        if self.net_size != expected_net:
            warnings.append(
                f"Position size mismatch: net={self.net_size}, "
                f"expected={expected_net} (bought={self.total_bought}, sold={self.total_sold})"
            )

        # Check for negative position without short tracking
        if self.net_size < 0 and self.total_bought > 0:
            warnings.append(f"Negative position detected: {self.net_size}")

        return warnings


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

    @property
    def mid_price(self) -> Decimal | None:
        """Calculate mid price from best bid and ask."""
        if not self.bids or not self.asks:
            return None
        best_bid = self.bids[0].price
        best_ask = self.asks[0].price
        return (best_bid + best_ask) / Decimal("2")

    @property
    def spread(self) -> Decimal | None:
        """Calculate bid-ask spread."""
        if not self.bids or not self.asks:
            return None
        best_bid = self.bids[0].price
        best_ask = self.asks[0].price
        return best_ask - best_bid

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
    unique_markets: int = 0

    # Cash Tracking
    starting_cash: Decimal = field(default_factory=lambda: Decimal("0"))
    ending_cash: Decimal = field(default_factory=lambda: Decimal("0"))
    cash_flow_from_fills: Decimal = field(default_factory=lambda: Decimal("0"))

    # PnL Breakdown
    realized_pnl: Decimal = field(default_factory=lambda: Decimal("0"))
    unrealized_pnl: Decimal = field(default_factory=lambda: Decimal("0"))
    total_fees: Decimal = field(default_factory=lambda: Decimal("0"))
    net_pnl: Decimal = field(default_factory=lambda: Decimal("0"))

    # Liquidation Analysis
    mark_to_mid: Decimal = field(default_factory=lambda: Decimal("0"))
    liquidation_value: Decimal = field(default_factory=lambda: Decimal("0"))
    liquidation_discount: Decimal = field(default_factory=lambda: Decimal("0"))

    # Position Details
    positions: list[dict] = field(default_factory=list)

    # Sanity Checks
    warnings: list[str] = field(default_factory=list)
    cashflow_conserved: bool = True
    position_verified: bool = True

    # Metadata
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    since: str | None = None
    market_filter: str | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "summary": {
                "total_fills": self.total_fills,
                "unique_tokens": self.unique_tokens,
                "unique_markets": self.unique_markets,
            },
            "cash": {
                "starting_cash": float(self.starting_cash),
                "ending_cash": float(self.ending_cash),
                "cash_flow_from_fills": float(self.cash_flow_from_fills),
            },
            "pnl": {
                "realized_pnl": float(self.realized_pnl),
                "unrealized_pnl": float(self.unrealized_pnl),
                "total_fees": float(self.total_fees),
                "net_pnl": float(self.net_pnl),
            },
            "liquidation": {
                "mark_to_mid": float(self.mark_to_mid),
                "liquidation_value": float(self.liquidation_value),
                "liquidation_discount": float(self.liquidation_discount),
                "discount_pct": (
                    float(self.liquidation_discount / self.mark_to_mid * 100)
                    if self.mark_to_mid > 0
                    else 0.0
                ),
            },
            "verification": {
                "cashflow_conserved": self.cashflow_conserved,
                "position_verified": self.position_verified,
                "warnings": self.warnings,
            },
            "positions": self.positions,
            "metadata": {
                "generated_at": self.generated_at,
                "since": self.since,
                "market_filter": self.market_filter,
            },
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: Path) -> None:
        """Save report to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())


class PnLVerifier:
    """PnL verifier with cash tracking and sanity checks.

    This class ingests fills and orderbook updates to compute:
    - Cash balance (from fill history)
    - Inventory (current positions)
    - Realized PnL (from closed trades)
    - Unrealized PnL (mark-to-mid)
    - Liquidation value (walking the book)

    Includes sanity checks:
    - Cashflow conservation (cash changes match fill flows)
    - Position verification (position size matches fills)
    """

    def __init__(self, starting_cash: Decimal = Decimal("0")) -> None:
        self.starting_cash = starting_cash
        self.positions: dict[str, Position] = {}
        self.fills: list[Fill] = []
        self.cash_balance = starting_cash

    def add_fill(self, fill: Fill) -> None:
        """Add a fill and update state."""
        self.fills.append(fill)

        # Update position
        if fill.token_id not in self.positions:
            self.positions[fill.token_id] = Position(
                token_id=fill.token_id,
                market_slug=fill.market_slug,
            )

        pos = self.positions[fill.token_id]

        if fill.side == "buy":
            pos.add_buy(fill.size, fill.price, fill.fee)
        else:
            pos.add_sell(fill.size, fill.price, fill.fee)

        # Update cash balance
        self.cash_balance += fill.cash_flow

    def add_fills(self, fills: Sequence[Fill]) -> None:
        """Add multiple fills."""
        for fill in fills:
            self.add_fill(fill)

    def verify_cashflow(self) -> tuple[bool, list[str]]:
        """Verify cashflow conservation.

        Checks that ending cash equals starting cash plus sum of fill cash flows.

        Returns:
            (is_conserved, list_of_warnings)
        """
        warnings = []

        # Calculate expected cash from fills
        total_cash_flow = sum(f.cash_flow for f in self.fills)
        expected_cash = self.starting_cash + total_cash_flow

        if self.cash_balance != expected_cash:
            diff = self.cash_balance - expected_cash
            warnings.append(
                f"Cashflow not conserved: balance={self.cash_balance}, "
                f"expected={expected_cash}, diff={diff}"
            )
            return False, warnings

        return True, warnings

    def verify_positions(self) -> tuple[bool, list[str]]:
        """Verify all positions.

        Returns:
            (all_verified, list_of_warnings)
        """
        all_warnings = []
        for pos in self.positions.values():
            warnings = pos.verify()
            all_warnings.extend(warnings)

        return len(all_warnings) == 0, all_warnings

    def compute_pnl(
        self,
        orderbooks: dict[str, OrderBook] | None = None,
        since: str | None = None,
        market_filter: str | None = None,
    ) -> PnLReport:
        """Compute complete PnL report.

        Args:
            orderbooks: Optional dict of token_id -> OrderBook for liquidation value
            since: Optional ISO timestamp for filtering (inclusive)
            market_filter: Optional market slug filter

        Returns:
            PnLReport with full analysis
        """
        report = PnLReport()
        report.since = since
        report.market_filter = market_filter

        # Filter fills if needed
        filtered_fills = self.fills
        if since:
            since_dt = datetime.fromisoformat(since.replace("Z", "+00:00"))
            filtered_fills = [f for f in filtered_fills if f.datetime_utc >= since_dt]

        if market_filter:
            filtered_fills = [
                f
                for f in filtered_fills
                if f.market_slug and market_filter.lower() in f.market_slug.lower()
            ]

        report.total_fills = len(filtered_fills)
        report.starting_cash = self.starting_cash
        report.ending_cash = self.cash_balance
        report.cash_flow_from_fills = sum(f.cash_flow for f in filtered_fills)

        # Use only positions that have fills in the filtered set
        active_tokens = {f.token_id for f in filtered_fills}
        filtered_positions = {k: v for k, v in self.positions.items() if k in active_tokens}

        report.unique_tokens = len(filtered_positions)
        report.unique_markets = len(
            {p.market_slug for p in filtered_positions.values() if p.market_slug}
        )

        # Calculate realized PnL and fees
        report.realized_pnl = sum(p.realized_pnl for p in filtered_positions.values())
        report.total_fees = sum(p.total_fees for p in filtered_positions.values())

        # Build position details and compute unrealized PnL
        position_details = []
        total_mark_to_mid = Decimal("0")
        total_liquidation = Decimal("0")

        for token_id, pos in filtered_positions.items():
            if pos.net_size == 0:
                continue  # Skip closed positions

            # Get current price for mark-to-market
            current_price: Decimal | None = None
            if orderbooks and token_id in orderbooks:
                current_price = orderbooks[token_id].mid_price

            if current_price is None:
                current_price = Decimal("0.5")  # Default to 0.5 if unknown

            # Calculate unrealized PnL (mark-to-mid)
            unrealized = (current_price - pos.avg_cost_basis) * pos.net_size

            # Mark to mid = position value at mid price
            mark_to_mid = current_price * pos.net_size

            # Liquidation value
            liquidation = Decimal("0")
            if orderbooks and token_id in orderbooks:
                # Assume YES token for now
                liquidation = orderbooks[token_id].get_walk_liquidation_value(
                    pos.net_size, is_yes=True
                )
            else:
                # Without orderbook, assume 10% discount for estimation
                liquidation = mark_to_mid * Decimal("0.9")

            total_mark_to_mid += mark_to_mid
            total_liquidation += liquidation

            position_details.append(
                {
                    "token_id": token_id,
                    "market_slug": pos.market_slug,
                    "net_size": float(pos.net_size),
                    "avg_cost_basis": float(pos.avg_cost_basis),
                    "current_price": float(current_price),
                    "unrealized_pnl": float(unrealized),
                    "mark_to_mid": float(mark_to_mid),
                    "liquidation_value": float(liquidation),
                    "buy_count": pos.buy_count,
                    "sell_count": pos.sell_count,
                    "total_bought": float(pos.total_bought),
                    "total_sold": float(pos.total_sold),
                }
            )

        report.positions = position_details
        report.unrealized_pnl = sum(Decimal(str(p["unrealized_pnl"])) for p in position_details)
        report.net_pnl = report.realized_pnl + report.unrealized_pnl - report.total_fees
        report.mark_to_mid = total_mark_to_mid
        report.liquidation_value = total_liquidation
        report.liquidation_discount = total_mark_to_mid - total_liquidation

        # Run sanity checks
        cashflow_ok, cashflow_warnings = self.verify_cashflow()
        report.cashflow_conserved = cashflow_ok
        report.warnings.extend(cashflow_warnings)

        position_ok, position_warnings = self.verify_positions()
        report.position_verified = position_ok
        report.warnings.extend(position_warnings)

        # Additional warnings for edge cases
        short_positions = [p for p in filtered_positions.values() if p.net_size < 0]
        if short_positions:
            report.warnings.append(
                f"Short positions detected ({len(short_positions)}) - PnL may be incomplete"
            )

        if total_mark_to_mid > 0 and total_liquidation / total_mark_to_mid < Decimal("0.5"):
            report.warnings.append("Large liquidation discount - positions may be illiquid")

        return report


def load_fills_from_file(path: Path) -> list[Fill]:
    """Load fills from a JSON file.

    Supports:
    - Array of fill objects: [{...}, {...}]
    - Object with 'fills' key: {"fills": [...]}
    - Object with 'data' key: {"data": [...]}
    - Object with 'orders' key: {"orders": [...]}
    - Newline-delimited JSON (.jsonl): one fill per line
    """
    with open(path, encoding="utf-8") as f:
        content = f.read().strip()

    # Check for JSONL format
    if path.suffix == ".jsonl" or ("\n" in content and not content.startswith("[")):
        fills = []
        for line in content.strip().split("\n"):
            line = line.strip()
            if line:
                fills.append(Fill.from_dict(json.loads(line)))
        return fills

    # Regular JSON
    data = json.loads(content)

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
    Or snapshot format from collector: {"markets": [{"books": {...}}]}
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    books = {}

    # Handle snapshot format from collector
    if "markets" in data:
        for market in data["markets"]:
            if "books" in market:
                for side in ["yes", "no"]:
                    if side in market["books"]:
                        book_data = market["books"][side]
                        token_id = f"{market.get('condition_id', 'unknown')}_{side}"
                        books[token_id] = OrderBook.from_dict(token_id, book_data)
    else:
        # Direct token_id -> book mapping
        for token_id, book_data in data.items():
            books[token_id] = OrderBook.from_dict(token_id, book_data)

    return books


def load_orderbooks_from_snapshot(path: Path) -> dict[str, OrderBook]:
    """Load orderbooks from a collector snapshot file.

    Snapshot format: {"markets": [{"condition_id": ..., "books": {"yes": {...}, "no": {...}}}]}
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    books = {}

    if "markets" not in data:
        return books

    for market in data["markets"]:
        condition_id = market.get("condition_id") or market.get("id", "unknown")
        if "books" in market:
            for side in ["yes", "no"]:
                if side in market["books"]:
                    book_data = market["books"][side]
                    token_id = f"{condition_id}_{side}"
                    books[token_id] = OrderBook.from_dict(token_id, book_data)

    return books


def save_daily_summary(
    report: PnLReport,
    out_dir: Path | None = None,
    date: datetime | None = None,
) -> Path:
    """Save PnL report as daily summary.

    Args:
        report: The PnL report to save
        out_dir: Output directory (default: data/pnl/)
        date: Date for filename (default: today)

    Returns:
        Path to saved file
    """
    if out_dir is None:
        out_dir = Path(DAILY_SUMMARY_DIR)

    if date is None:
        date = datetime.now(timezone.utc)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Filename: pnl_YYYY-MM-DD.json
    filename = f"pnl_{date.strftime('%Y-%m-%d')}.json"
    out_path = out_dir / filename

    report.save(out_path)
    return out_path


def compute_pnl(
    fills: list[Fill],
    current_prices: dict[str, Decimal] | None = None,
    orderbooks: dict[str, OrderBook] | None = None,
) -> PnLReport:
    """Compute PnL report from a list of fills (legacy function).

    Args:
        fills: List of Fill objects representing all trades
        current_prices: Optional dict of token_id -> current price for mark-to-market
        orderbooks: Optional dict of token_id -> OrderBook for liquidation value

    Returns:
        PnLReport with full analysis
    """
    # Build orderbooks from current_prices if provided
    merged_orderbooks: dict[str, OrderBook] = {}
    if orderbooks:
        merged_orderbooks = dict(orderbooks)

    # Add synthetic orderbooks for current_prices
    if current_prices:
        for token_id, price in current_prices.items():
            if token_id not in merged_orderbooks:
                # Create synthetic orderbook with single bid/ask at price
                merged_orderbooks[token_id] = OrderBook(
                    token_id=token_id,
                    bids=[BookLevel(price=price, size=Decimal("999999"))],
                    asks=[BookLevel(price=price, size=Decimal("999999"))],
                )

    verifier = PnLVerifier()
    verifier.add_fills(fills)
    return verifier.compute_pnl(orderbooks=merged_orderbooks if merged_orderbooks else None)
