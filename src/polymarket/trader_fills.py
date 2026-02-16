"""Trader fill tracking and NAV accounting per trader.

Fetches and persists fills/events for tracked traders,
computes realized/unrealized PnL and NAV over time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING

import httpx

from .endpoints import DATA_BASE
from .pnl import Fill as PnlFill

if TYPE_CHECKING:
    from collections.abc import Sequence


# Constants
DEFAULT_DATA_DIR = Path("data/trader_profiles")
TRADER_FILLS_DIR = "fills"
TRADER_NAV_DIR = "nav"


def _data_client(timeout: float = 30.0) -> httpx.Client:
    """Create HTTP client for Data API."""
    return httpx.Client(
        base_url=DATA_BASE, timeout=timeout, headers={"User-Agent": "polymarket-bot/0.1"}
    )


@dataclass(frozen=True)
class TraderFill:
    """A fill/trade executed by a tracked trader.

    Similar to pnl.Fill but with additional trader-specific metadata.
    """

    token_id: str
    side: str  # 'buy' or 'sell'
    size: Decimal
    price: Decimal
    fee: Decimal
    timestamp: str  # ISO8601
    transaction_hash: str | None = None
    market_slug: str | None = None
    market_question: str | None = None
    condition_id: str | None = None
    # Additional metadata
    block_number: int | None = None
    log_index: int | None = None

    @classmethod
    def from_api(cls, data: dict) -> TraderFill:
        """Create from Polymarket API response."""
        return cls(
            token_id=data.get("asset_id", data.get("token_id", "")),
            side=(data.get("side") or data.get("trade_side", "buy")).lower(),
            size=Decimal(str(data.get("size", data.get("takerAmount", "0")))),
            price=Decimal(str(data.get("price", data.get("execution_price", "0")))),
            fee=Decimal(str(data.get("fee", data.get("trade_fee", "0")))),
            timestamp=data.get("timestamp", data.get("created_at", datetime.now(UTC).isoformat())),
            transaction_hash=data.get("transaction_hash") or data.get("tx_hash"),
            market_slug=data.get("market_slug") or data.get("slug"),
            market_question=data.get("market_question") or data.get("question"),
            condition_id=data.get("condition_id"),
            block_number=data.get("block_number"),
            log_index=data.get("log_index"),
        )

    @classmethod
    def from_dict(cls, data: dict) -> TraderFill:
        """Create from dictionary."""
        return cls(
            token_id=data["token_id"],
            side=data["side"],
            size=Decimal(str(data["size"])),
            price=Decimal(str(data["price"])),
            fee=Decimal(str(data.get("fee", "0"))),
            timestamp=data["timestamp"],
            transaction_hash=data.get("transaction_hash"),
            market_slug=data.get("market_slug"),
            market_question=data.get("market_question"),
            condition_id=data.get("condition_id"),
            block_number=data.get("block_number"),
            log_index=data.get("log_index"),
        )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "token_id": self.token_id,
            "side": self.side,
            "size": str(self.size),
            "price": str(self.price),
            "fee": str(self.fee),
            "timestamp": self.timestamp,
            "transaction_hash": self.transaction_hash,
            "market_slug": self.market_slug,
            "market_question": self.market_question,
            "condition_id": self.condition_id,
            "block_number": self.block_number,
            "log_index": self.log_index,
        }

    def to_pnl_fill(self) -> PnlFill:
        """Convert to PnL Fill for use with PnLVerifier."""
        return PnlFill(
            token_id=self.token_id,
            side=self.side,
            size=self.size,
            price=self.price,
            fee=self.fee,
            timestamp=self.timestamp,
            transaction_hash=self.transaction_hash,
            market_slug=self.market_slug,
        )

    @property
    def cash_flow(self) -> Decimal:
        """Calculate cash flow impact of this fill."""
        notional = self.size * self.price
        if self.side == "buy":
            return -(notional + self.fee)
        else:
            return notional - self.fee


@dataclass
class TraderPosition:
    """Position state for a trader in a specific token.

    Similar to pnl.Position but with additional tracking.
    """

    token_id: str
    trader_address: str
    market_slug: str | None = None
    net_size: Decimal = field(default_factory=lambda: Decimal("0"))
    total_cost: Decimal = field(default_factory=lambda: Decimal("0"))
    realized_pnl: Decimal = field(default_factory=lambda: Decimal("0"))
    total_fees: Decimal = field(default_factory=lambda: Decimal("0"))
    buy_count: int = 0
    sell_count: int = 0
    total_bought: Decimal = field(default_factory=lambda: Decimal("0"))
    total_sold: Decimal = field(default_factory=lambda: Decimal("0"))
    last_fill_at: str | None = None
    first_fill_at: str | None = None

    def add_buy(self, size: Decimal, price: Decimal, fee: Decimal, timestamp: str) -> None:
        """Record a buy fill."""
        cost = size * price
        self.net_size += size
        self.total_cost += cost
        self.total_fees += fee
        self.total_bought += size
        self.buy_count += 1
        self.last_fill_at = timestamp
        if self.first_fill_at is None:
            self.first_fill_at = timestamp

    def add_sell(self, size: Decimal, price: Decimal, fee: Decimal, timestamp: str) -> Decimal:
        """Record a sell fill and return realized PnL."""
        if self.net_size <= 0:
            # Short selling
            self.net_size -= size
            self.total_fees += fee
            self.total_sold += size
            self.sell_count += 1
            self.last_fill_at = timestamp
            return Decimal("0")

        # Calculate cost basis for shares being sold
        sell_size = min(size, self.net_size)
        current_avg_cost = self._compute_avg_cost_basis()
        cost_basis = sell_size * current_avg_cost
        proceeds = sell_size * price

        # Realized PnL
        realized = proceeds - cost_basis - fee
        self.realized_pnl += realized

        # Update position
        self.net_size -= sell_size
        self.total_cost -= cost_basis
        self.total_fees += fee
        self.total_sold += size
        self.sell_count += 1
        self.last_fill_at = timestamp

        return realized

    def _compute_avg_cost_basis(self) -> Decimal:
        """Compute average cost per share for current position."""
        if self.net_size <= 0:
            return Decimal("0")
        return self.total_cost / self.net_size

    @property
    def avg_cost_basis(self) -> Decimal:
        """Average cost per share for current position."""
        return self._compute_avg_cost_basis()

    def unrealized_pnl(self, current_price: Decimal = Decimal("0.5")) -> Decimal:
        """Calculate unrealized PnL at given price."""
        if self.net_size == 0:
            return Decimal("0")
        return (current_price - self.avg_cost_basis) * self.net_size

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "token_id": self.token_id,
            "trader_address": self.trader_address,
            "market_slug": self.market_slug,
            "net_size": str(self.net_size),
            "avg_cost_basis": str(self.avg_cost_basis),
            "total_cost": str(self.total_cost),
            "realized_pnl": str(self.realized_pnl),
            "total_fees": str(self.total_fees),
            "buy_count": self.buy_count,
            "sell_count": self.sell_count,
            "total_bought": str(self.total_bought),
            "total_sold": str(self.total_sold),
            "last_fill_at": self.last_fill_at,
            "first_fill_at": self.first_fill_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> TraderPosition:
        """Create from dictionary."""
        # Note: avg_cost_basis is computed, not stored
        return cls(
            token_id=data["token_id"],
            trader_address=data["trader_address"],
            market_slug=data.get("market_slug"),
            net_size=Decimal(str(data.get("net_size", "0"))),
            total_cost=Decimal(str(data.get("total_cost", "0"))),
            realized_pnl=Decimal(str(data.get("realized_pnl", "0"))),
            total_fees=Decimal(str(data.get("total_fees", "0"))),
            buy_count=data.get("buy_count", 0),
            sell_count=data.get("sell_count", 0),
            total_bought=Decimal(str(data.get("total_bought", "0"))),
            total_sold=Decimal(str(data.get("total_sold", "0"))),
            last_fill_at=data.get("last_fill_at"),
            first_fill_at=data.get("first_fill_at"),
        )


@dataclass
class TraderNAVSnapshot:
    """NAV snapshot for a trader at a point in time.

    Similar to paper_trading.EquitySnapshot but for tracking
    per-trader performance.
    """

    trader_address: str
    timestamp: str  # ISO8601
    cash_balance: Decimal  # Estimated from fills
    positions_value: Decimal  # Mark-to-market value
    realized_pnl: Decimal
    unrealized_pnl: Decimal
    total_fees: Decimal
    nav: Decimal  # Total NAV = cash + positions
    position_count: int
    open_position_count: int

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "trader_address": self.trader_address,
            "timestamp": self.timestamp,
            "cash_balance": str(self.cash_balance),
            "positions_value": str(self.positions_value),
            "realized_pnl": str(self.realized_pnl),
            "unrealized_pnl": str(self.unrealized_pnl),
            "total_fees": str(self.total_fees),
            "nav": str(self.nav),
            "position_count": self.position_count,
            "open_position_count": self.open_position_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> TraderNAVSnapshot:
        """Create from dictionary."""
        return cls(
            trader_address=data["trader_address"],
            timestamp=data["timestamp"],
            cash_balance=Decimal(str(data.get("cash_balance", "0"))),
            positions_value=Decimal(str(data.get("positions_value", "0"))),
            realized_pnl=Decimal(str(data.get("realized_pnl", "0"))),
            unrealized_pnl=Decimal(str(data.get("unrealized_pnl", "0"))),
            total_fees=Decimal(str(data.get("total_fees", "0"))),
            nav=Decimal(str(data.get("nav", "0"))),
            position_count=data.get("position_count", 0),
            open_position_count=data.get("open_position_count", 0),
        )


@dataclass
class TraderAccounting:
    """Complete accounting state for a single trader.

    Tracks fills, positions, and NAV history for one trader.
    """

    trader_address: str
    fills: list[TraderFill] = field(default_factory=list)
    positions: dict[str, TraderPosition] = field(default_factory=dict)
    cash_balance: Decimal = field(default_factory=lambda: Decimal("0"))
    total_realized_pnl: Decimal = field(default_factory=lambda: Decimal("0"))
    total_fees: Decimal = field(default_factory=lambda: Decimal("0"))
    last_fill_timestamp: str | None = None

    def add_fill(self, fill: TraderFill) -> None:
        """Add a fill and update position state."""
        self.fills.append(fill)

        # Update cash balance
        self.cash_balance += fill.cash_flow

        # Update or create position
        if fill.token_id not in self.positions:
            self.positions[fill.token_id] = TraderPosition(
                token_id=fill.token_id,
                trader_address=self.trader_address,
                market_slug=fill.market_slug,
            )

        pos = self.positions[fill.token_id]

        if fill.side == "buy":
            pos.add_buy(fill.size, fill.price, fill.fee, fill.timestamp)
        else:
            realized = pos.add_sell(fill.size, fill.price, fill.fee, fill.timestamp)
            self.total_realized_pnl += realized

        self.total_fees += fill.fee
        self.last_fill_timestamp = fill.timestamp

    def compute_nav(
        self,
        current_prices: dict[str, Decimal] | None = None,
        timestamp: str | None = None,
    ) -> TraderNAVSnapshot:
        """Compute current NAV snapshot.

        Args:
            current_prices: Optional dict of token_id -> price for mark-to-market
            timestamp: Optional timestamp (defaults to now)

        Returns:
            TraderNAVSnapshot with current state
        """
        if timestamp is None:
            timestamp = datetime.now(UTC).isoformat()

        # Calculate positions value
        positions_value = Decimal("0")
        unrealized_pnl = Decimal("0")
        open_positions = 0

        for token_id, pos in self.positions.items():
            if pos.net_size == 0:
                continue

            # Get current price
            price = Decimal("0.5")  # Default to 0.5 if unknown
            if current_prices and token_id in current_prices:
                price = current_prices[token_id]

            pos_value = pos.net_size * price
            positions_value += pos_value
            unrealized_pnl += (price - pos.avg_cost_basis) * pos.net_size
            open_positions += 1

        nav = self.cash_balance + positions_value

        return TraderNAVSnapshot(
            trader_address=self.trader_address,
            timestamp=timestamp,
            cash_balance=self.cash_balance,
            positions_value=positions_value,
            realized_pnl=self.total_realized_pnl,
            unrealized_pnl=unrealized_pnl,
            total_fees=self.total_fees,
            nav=nav,
            position_count=len(self.positions),
            open_position_count=open_positions,
        )


class TraderFillTracker:
    """Tracks fills and computes PnL/NAV for multiple traders.

    Features:
    - Fetch fills from Polymarket API for tracked traders
    - Persist fills per trader (append-only JSONL)
    - Compute realized/unrealized PnL
    - Track NAV over time
    """

    data_dir: Path
    traders: dict[str, TraderAccounting]

    def __init__(self, data_dir: Path | str | None = None) -> None:
        """Initialize fill tracker.

        Args:
            data_dir: Directory for trader data storage
        """
        self.data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / TRADER_FILLS_DIR).mkdir(exist_ok=True)
        (self.data_dir / TRADER_NAV_DIR).mkdir(exist_ok=True)

        self.traders: dict[str, TraderAccounting] = {}

    def get_fills_path(self, address: str) -> Path:
        """Get path to fills file for a trader."""
        return self.data_dir / TRADER_FILLS_DIR / f"{address.lower()}.jsonl"

    def get_nav_path(self, address: str) -> Path:
        """Get path to NAV history file for a trader."""
        return self.data_dir / TRADER_NAV_DIR / f"{address.lower()}.jsonl"

    def load_fills(self, address: str) -> list[TraderFill]:
        """Load fills for a trader from disk.

        Args:
            address: Trader wallet address

        Returns:
            List of TraderFill objects
        """
        fills_path = self.get_fills_path(address)
        if not fills_path.exists():
            return []

        fills = []
        with open(fills_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    fills.append(TraderFill.from_dict(data))
                except (json.JSONDecodeError, ValueError):
                    continue

        return fills

    def save_fill(self, address: str, fill: TraderFill) -> None:
        """Append a fill to a trader's fill journal.

        Args:
            address: Trader wallet address
            fill: TraderFill to save
        """
        fills_path = self.get_fills_path(address)
        with open(fills_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(fill.to_dict(), sort_keys=True) + "\n")

    def load_accounting(self, address: str) -> TraderAccounting:
        """Load or create accounting state for a trader.

        Reconstructs state from persisted fills.

        Args:
            address: Trader wallet address

        Returns:
            TraderAccounting with reconstructed state
        """
        address = address.lower()

        if address in self.traders:
            return self.traders[address]

        # Create new accounting and replay fills
        accounting = TraderAccounting(trader_address=address)
        fills = self.load_fills(address)

        for fill in sorted(fills, key=lambda f: f.timestamp):
            accounting.add_fill(fill)

        self.traders[address] = accounting
        return accounting

    def fetch_fills_from_api(
        self,
        address: str,
        since: str | None = None,
        limit: int = 1000,
    ) -> list[TraderFill]:
        """Fetch fills from Polymarket Data API.

        Args:
            address: Trader wallet address
            since: Optional ISO timestamp to fetch from
            limit: Maximum fills to fetch

        Returns:
            List of TraderFill objects from API
        """
        fills = []

        try:
            with _data_client() as client:
                params: dict = {
                    "address": address,
                    "limit": limit,
                }
                if since:
                    params["since"] = since

                response = client.get("/fills", params=params)
                response.raise_for_status()
                data = response.json()

                for fill_data in data.get("data", []):
                    try:
                        fill = TraderFill.from_api(fill_data)
                        fills.append(fill)
                    except (ValueError, KeyError):
                        continue

        except httpx.HTTPError as e:
            print(f"Error fetching fills for {address}: {e}")

        return fills

    def sync_trader_fills(
        self,
        address: str,
        fetch_limit: int = 1000,
        window_hours: float = 24.0,
        overlap_minutes: float = 5.0,
    ) -> tuple[int, int]:
        """Sync fills for a trader from API to local storage.

        Args:
            address: Trader wallet address
            fetch_limit: Maximum fills to fetch

        Returns:
            Tuple of (new_fills_count, total_fills_count)
        """
        address = address.lower()

        # Load existing fills to build a resilient fetch window.
        #
        # IMPORTANT: relying purely on "since = latest_timestamp" can create long gaps if the
        # API returns out-of-order data, has inclusive/exclusive edge behavior, or we miss a page.
        # Instead, re-scan a sliding recent window and de-dupe locally.
        existing_fills = self.load_fills(address)

        now = datetime.now(UTC)
        window_start = now - timedelta(hours=window_hours)
        # Add a small overlap to avoid fencepost issues when timestamps are truncated/rounded.
        window_start = window_start - timedelta(minutes=overlap_minutes)
        since = window_start.isoformat()

        # Fetch fills from the recent window
        new_fills = self.fetch_fills_from_api(address, since=since, limit=fetch_limit)

        # Filter out duplicates (by transaction_hash or timestamp+token_id+side)
        existing_keys = set()
        for f in existing_fills:
            key = f.transaction_hash or f"{f.timestamp}:{f.token_id}:{f.side}"
            existing_keys.add(key)

        added = 0
        for fill in new_fills:
            key = fill.transaction_hash or f"{fill.timestamp}:{fill.token_id}:{fill.side}"
            if key not in existing_keys:
                self.save_fill(address, fill)
                added += 1

        return added, len(existing_fills) + added

    def sync_all_traders(
        self,
        addresses: Sequence[str],
        fetch_limit: int = 1000,
    ) -> dict[str, tuple[int, int]]:
        """Sync fills for multiple traders.

        Args:
            addresses: List of trader addresses
            fetch_limit: Maximum fills per trader

        Returns:
            Dict of address -> (new_count, total_count)
        """
        results = {}
        for address in addresses:
            new_count, total_count = self.sync_trader_fills(address, fetch_limit)
            results[address] = (new_count, total_count)
        return results

    def compute_trader_nav(
        self,
        address: str,
        current_prices: dict[str, Decimal] | None = None,
    ) -> TraderNAVSnapshot:
        """Compute current NAV for a trader.

        Args:
            address: Trader wallet address
            current_prices: Optional token prices for mark-to-market

        Returns:
            TraderNAVSnapshot with current state
        """
        accounting = self.load_accounting(address)
        return accounting.compute_nav(current_prices)

    def record_nav_snapshot(
        self,
        address: str,
        current_prices: dict[str, Decimal] | None = None,
    ) -> TraderNAVSnapshot:
        """Compute and record NAV snapshot for a trader.

        Args:
            address: Trader wallet address
            current_prices: Optional token prices

        Returns:
            Recorded TraderNAVSnapshot
        """
        snapshot = self.compute_trader_nav(address, current_prices)

        nav_path = self.get_nav_path(address)
        with open(nav_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(snapshot.to_dict(), sort_keys=True) + "\n")

        return snapshot

    def get_nav_history(self, address: str) -> list[TraderNAVSnapshot]:
        """Get NAV history for a trader.

        Args:
            address: Trader wallet address

        Returns:
            List of TraderNAVSnapshot objects
        """
        nav_path = self.get_nav_path(address)
        if not nav_path.exists():
            return []

        snapshots = []
        with open(nav_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    snapshots.append(TraderNAVSnapshot.from_dict(data))
                except (json.JSONDecodeError, ValueError):
                    continue

        return snapshots

    def get_trader_summary(self, address: str) -> dict:
        """Get summary statistics for a trader.

        Args:
            address: Trader wallet address

        Returns:
            Dict with summary statistics
        """
        accounting = self.load_accounting(address)
        nav = accounting.compute_nav()
        history = self.get_nav_history(address)

        # Calculate return metrics if we have history
        total_return = Decimal("0")
        total_return_pct = Decimal("0")
        max_drawdown = Decimal("0")

        if len(history) >= 2:
            start_nav = history[0].nav
            current_nav = nav.nav
            if start_nav > 0:
                total_return = current_nav - start_nav
                total_return_pct = (total_return / start_nav) * 100

            # Calculate max drawdown
            peak = history[0].nav
            for snap in history:
                if snap.nav > peak:
                    peak = snap.nav
                drawdown = peak - snap.nav
                if drawdown > max_drawdown:
                    max_drawdown = drawdown

        return {
            "address": address,
            "total_fills": len(accounting.fills),
            "total_positions": len(accounting.positions),
            "open_positions": nav.open_position_count,
            "cash_balance": float(nav.cash_balance),
            "realized_pnl": float(nav.realized_pnl),
            "unrealized_pnl": float(nav.unrealized_pnl),
            "total_fees": float(nav.total_fees),
            "current_nav": float(nav.nav),
            "nav_data_points": len(history),
            "total_return": float(total_return),
            "total_return_pct": float(total_return_pct),
            "max_drawdown": float(max_drawdown),
            "last_fill_at": accounting.last_fill_timestamp,
        }
