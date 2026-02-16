"""SQLite-based PnL/NAV accounting for paper copytrade simulation.

This module provides:
- Schema for positions + fills + cash ledger
- Realized PnL, unrealized PnL, NAV, liquidation value at snapshot times
- Account summary queries (last 7d + current exposures)
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

# Constants
DEFAULT_DB_PATH = Path("data/accounting.db")
USDC_DECIMALS = Decimal("1")


@dataclass(frozen=True)
class AccountingFill:
    """A fill recorded in the accounting system.

    This represents a single trade execution (buy or sell) that affects
    the cash ledger and position inventory.
    """

    fill_id: str
    timestamp: datetime
    token_id: str
    side: str  # 'buy' or 'sell'
    size: Decimal
    price: Decimal
    fee: Decimal
    cash_flow: Decimal  # Pre-calculated cash impact
    market_slug: str | None = None
    market_question: str | None = None
    trader_source: str | None = None  # For copy trading: original trader address
    original_tx_hash: str | None = None  # For copy trading: original tx reference

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "fill_id": self.fill_id,
            "timestamp": self.timestamp.isoformat(),
            "token_id": self.token_id,
            "side": self.side,
            "size": str(self.size),
            "price": str(self.price),
            "fee": str(self.fee),
            "cash_flow": str(self.cash_flow),
            "market_slug": self.market_slug,
            "market_question": self.market_question,
            "trader_source": self.trader_source,
            "original_tx_hash": self.original_tx_hash,
        }

    @classmethod
    def from_dict(cls, data: dict) -> AccountingFill:
        """Create from dictionary."""
        return cls(
            fill_id=data["fill_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            token_id=data["token_id"],
            side=data["side"],
            size=Decimal(str(data["size"])),
            price=Decimal(str(data["price"])),
            fee=Decimal(str(data["fee"])),
            cash_flow=Decimal(str(data["cash_flow"])),
            market_slug=data.get("market_slug"),
            market_question=data.get("market_question"),
            trader_source=data.get("trader_source"),
            original_tx_hash=data.get("original_tx_hash"),
        )


@dataclass
class AccountingPosition:
    """Position state tracked in accounting system.

    Represents the current state of a position in a specific token,
    including cost basis and realized PnL from partial closes.
    """

    position_id: int | None = None
    token_id: str = ""
    market_slug: str | None = None
    net_size: Decimal = field(default_factory=lambda: Decimal("0"))
    total_cost: Decimal = field(default_factory=lambda: Decimal("0"))
    realized_pnl: Decimal = field(default_factory=lambda: Decimal("0"))
    total_fees: Decimal = field(default_factory=lambda: Decimal("0"))
    fill_count: int = 0
    last_fill_at: datetime | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @property
    def avg_cost_basis(self) -> Decimal:
        """Average cost per share for current position."""
        if self.net_size == 0:
            return Decimal("0")
        return self.total_cost / self.net_size

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "position_id": self.position_id,
            "token_id": self.token_id,
            "market_slug": self.market_slug,
            "net_size": str(self.net_size),
            "avg_cost_basis": str(self.avg_cost_basis),
            "total_cost": str(self.total_cost),
            "realized_pnl": str(self.realized_pnl),
            "total_fees": str(self.total_fees),
            "fill_count": self.fill_count,
            "last_fill_at": self.last_fill_at.isoformat() if self.last_fill_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


@dataclass
class CashLedgerEntry:
    """Single entry in the cash ledger.

    Every cash movement is recorded for auditability.
    """

    entry_id: int | None = None
    timestamp: datetime | None = None
    entry_type: str = ""  # 'fill', 'deposit', 'withdrawal', 'adjustment'
    amount: Decimal = field(default_factory=lambda: Decimal("0"))
    balance_after: Decimal = field(default_factory=lambda: Decimal("0"))
    reference_id: str | None = None  # fill_id, etc.
    description: str | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "entry_id": self.entry_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "entry_type": self.entry_type,
            "amount": str(self.amount),
            "balance_after": str(self.balance_after),
            "reference_id": self.reference_id,
            "description": self.description,
        }


@dataclass
class SnapshotValuation:
    """Valuation of a position at a specific snapshot time.

    Captures mark-to-market and liquidation value at a point in time.
    """

    snapshot_id: int | None = None
    position_id: int | None = None
    token_id: str = ""
    snapshot_time: datetime | None = None
    position_size: Decimal = field(default_factory=lambda: Decimal("0"))
    avg_cost_basis: Decimal = field(default_factory=lambda: Decimal("0"))
    mark_price: Decimal = field(default_factory=lambda: Decimal("0"))
    mark_to_market: Decimal = field(default_factory=lambda: Decimal("0"))
    unrealized_pnl: Decimal = field(default_factory=lambda: Decimal("0"))
    liquidation_value: Decimal = field(default_factory=lambda: Decimal("0"))

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "snapshot_id": self.snapshot_id,
            "position_id": self.position_id,
            "token_id": self.token_id,
            "snapshot_time": self.snapshot_time.isoformat() if self.snapshot_time else None,
            "position_size": str(self.position_size),
            "avg_cost_basis": str(self.avg_cost_basis),
            "mark_price": str(self.mark_price),
            "mark_to_market": str(self.mark_to_market),
            "unrealized_pnl": str(self.unrealized_pnl),
            "liquidation_value": str(self.liquidation_value),
        }


@dataclass
class PortfolioSnapshot:
    """Complete portfolio snapshot at a point in time.

    Aggregates all positions, cash, and PnL metrics at snapshot time.
    """

    snapshot_id: int | None = None
    snapshot_time: datetime | None = None
    cash_balance: Decimal = field(default_factory=lambda: Decimal("0"))
    positions_value: Decimal = field(default_factory=lambda: Decimal("0"))
    liquidation_value: Decimal = field(default_factory=lambda: Decimal("0"))
    realized_pnl: Decimal = field(default_factory=lambda: Decimal("0"))
    unrealized_pnl: Decimal = field(default_factory=lambda: Decimal("0"))
    total_fees: Decimal = field(default_factory=lambda: Decimal("0"))
    nav: Decimal = field(default_factory=lambda: Decimal("0"))
    open_position_count: int = 0
    total_position_count: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "snapshot_id": self.snapshot_id,
            "snapshot_time": self.snapshot_time.isoformat() if self.snapshot_time else None,
            "cash_balance": str(self.cash_balance),
            "positions_value": str(self.positions_value),
            "liquidation_value": str(self.liquidation_value),
            "realized_pnl": str(self.realized_pnl),
            "unrealized_pnl": str(self.unrealized_pnl),
            "total_fees": str(self.total_fees),
            "nav": str(self.nav),
            "open_position_count": self.open_position_count,
            "total_position_count": self.total_position_count,
        }


class AccountingDB:
    """SQLite database for PnL/NAV accounting.

    Manages:
    - fills: All trade executions
    - positions: Current position state per token
    - cash_ledger: Audit trail of all cash movements
    - portfolio_snapshots: NAV snapshots over time
    - snapshot_valuations: Position-level valuations at snapshot times
    """

    def __init__(self, db_path: Path | str | None = None) -> None:
        """Initialize database connection.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
            # Enable foreign keys
            self._conn.execute("PRAGMA foreign_keys = ON")
        return self._conn

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> AccountingDB:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
        """Context manager exit."""
        self.close()

    def init_schema(self) -> None:
        """Create database schema if not exists."""
        conn = self._get_connection()

        # Fills table - records all trade executions
        conn.execute("""
            CREATE TABLE IF NOT EXISTS fills (
                fill_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                token_id TEXT NOT NULL,
                side TEXT NOT NULL CHECK(side IN ('buy', 'sell')),
                size TEXT NOT NULL,
                price TEXT NOT NULL,
                fee TEXT NOT NULL,
                cash_flow TEXT NOT NULL,
                market_slug TEXT,
                market_question TEXT,
                trader_source TEXT,
                original_tx_hash TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Positions table - current state per token
        conn.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                position_id INTEGER PRIMARY KEY AUTOINCREMENT,
                token_id TEXT UNIQUE NOT NULL,
                market_slug TEXT,
                net_size TEXT NOT NULL DEFAULT '0',
                total_cost TEXT NOT NULL DEFAULT '0',
                realized_pnl TEXT NOT NULL DEFAULT '0',
                total_fees TEXT NOT NULL DEFAULT '0',
                fill_count INTEGER NOT NULL DEFAULT 0,
                last_fill_at TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Cash ledger - audit trail of all cash movements
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cash_ledger (
                entry_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                entry_type TEXT NOT NULL CHECK(entry_type IN ('fill', 'deposit', 'withdrawal', 'adjustment', 'initial')),
                amount TEXT NOT NULL,
                balance_after TEXT NOT NULL,
                reference_id TEXT,
                description TEXT
            )
        """)

        # Portfolio snapshots - NAV at points in time
        conn.execute("""
            CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
                snapshot_time TEXT UNIQUE NOT NULL,
                cash_balance TEXT NOT NULL DEFAULT '0',
                positions_value TEXT NOT NULL DEFAULT '0',
                liquidation_value TEXT NOT NULL DEFAULT '0',
                realized_pnl TEXT NOT NULL DEFAULT '0',
                unrealized_pnl TEXT NOT NULL DEFAULT '0',
                total_fees TEXT NOT NULL DEFAULT '0',
                nav TEXT NOT NULL DEFAULT '0',
                open_position_count INTEGER NOT NULL DEFAULT 0,
                total_position_count INTEGER NOT NULL DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Snapshot valuations - position-level detail at snapshot time
        conn.execute("""
            CREATE TABLE IF NOT EXISTS snapshot_valuations (
                valuation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                snapshot_id INTEGER NOT NULL,
                position_id INTEGER,
                token_id TEXT NOT NULL,
                position_size TEXT NOT NULL DEFAULT '0',
                avg_cost_basis TEXT NOT NULL DEFAULT '0',
                mark_price TEXT NOT NULL DEFAULT '0',
                mark_to_market TEXT NOT NULL DEFAULT '0',
                unrealized_pnl TEXT NOT NULL DEFAULT '0',
                liquidation_value TEXT NOT NULL DEFAULT '0',
                FOREIGN KEY (snapshot_id) REFERENCES portfolio_snapshots(snapshot_id)
            )
        """)

        # Indexes for common queries
        conn.execute("CREATE INDEX IF NOT EXISTS idx_fills_token ON fills(token_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_fills_timestamp ON fills(timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_fills_trader ON fills(trader_source)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ledger_timestamp ON cash_ledger(timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_snapshots_time ON portfolio_snapshots(snapshot_time)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_valuations_snapshot ON snapshot_valuations(snapshot_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_valuations_token ON snapshot_valuations(token_id)")

        conn.commit()

    def record_fill(self, fill: AccountingFill) -> None:
        """Record a fill and update position state.

        Args:
            fill: The fill to record
        """
        conn = self._get_connection()

        # Insert fill
        conn.execute(
            """
            INSERT INTO fills (
                fill_id, timestamp, token_id, side, size, price, fee, cash_flow,
                market_slug, market_question, trader_source, original_tx_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                fill.fill_id,
                fill.timestamp.isoformat(),
                fill.token_id,
                fill.side,
                str(fill.size),
                str(fill.price),
                str(fill.fee),
                str(fill.cash_flow),
                fill.market_slug,
                fill.market_question,
                fill.trader_source,
                fill.original_tx_hash,
            ),
        )

        # Update or create position
        self._update_position_from_fill(conn, fill)

        # Record cash ledger entry
        self._record_cash_entry(conn, fill)

        conn.commit()

    def _update_position_from_fill(self, conn: sqlite3.Connection, fill: AccountingFill) -> None:
        """Update position state based on fill.

        Args:
            conn: Database connection
            fill: The fill to apply
        """
        # Get existing position
        row = conn.execute(
            "SELECT * FROM positions WHERE token_id = ?",
            (fill.token_id,),
        ).fetchone()

        if row is None:
            # Create new position
            if fill.side == "buy":
                net_size = fill.size
                total_cost = fill.size * fill.price
                realized_pnl = Decimal("0")
            else:
                # Selling without position = short
                net_size = -fill.size
                total_cost = Decimal("0")
                realized_pnl = Decimal("0")

            conn.execute(
                """
                INSERT INTO positions (
                    token_id, market_slug, net_size, total_cost, realized_pnl,
                    total_fees, fill_count, last_fill_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    fill.token_id,
                    fill.market_slug,
                    str(net_size),
                    str(total_cost),
                    str(realized_pnl),
                    str(fill.fee),
                    1,
                    fill.timestamp.isoformat(),
                ),
            )
        else:
            # Update existing position
            net_size = Decimal(str(row["net_size"]))
            total_cost = Decimal(str(row["total_cost"]))
            realized_pnl = Decimal(str(row["realized_pnl"]))
            total_fees = Decimal(str(row["total_fees"]))
            fill_count = row["fill_count"]

            if fill.side == "buy":
                # Adding to position
                total_cost += fill.size * fill.price
                net_size += fill.size
                total_fees += fill.fee
            else:
                # Selling - calculate realized PnL for closed portion
                sell_size = fill.size
                if net_size > 0:
                    # Closing long position
                    close_size = min(sell_size, net_size)
                    # Calculate avg cost per share (excluding fees from cost basis)
                    avg_entry_price = total_cost / net_size if net_size > 0 else Decimal("0")
                    cost_basis = close_size * avg_entry_price
                    proceeds = close_size * fill.price
                    # Calculate gross PnL on this portion
                    gross_pnl = proceeds - cost_basis
                    # Realized PnL = gross_pnl - sell_fee only
                    # (buy fees are tracked in total_fees but not deducted from realized_pnl)
                    realized_pnl += gross_pnl - fill.fee
                    total_cost -= cost_basis
                    net_size -= close_size
                    sell_size -= close_size

                if sell_size > 0:
                    # Remaining is short
                    net_size -= sell_size

                total_fees += fill.fee

            conn.execute(
                """
                UPDATE positions SET
                    net_size = ?,
                    total_cost = ?,
                    realized_pnl = ?,
                    total_fees = ?,
                    fill_count = ?,
                    last_fill_at = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE token_id = ?
                """,
                (
                    str(net_size),
                    str(total_cost),
                    str(realized_pnl),
                    str(total_fees),
                    fill_count + 1,
                    fill.timestamp.isoformat(),
                    fill.token_id,
                ),
            )

    def _record_cash_entry(self, conn: sqlite3.Connection, fill: AccountingFill) -> None:
        """Record cash ledger entry for a fill.

        Args:
            conn: Database connection
            fill: The fill to record
        """
        # Get current balance
        row = conn.execute(
            "SELECT balance_after FROM cash_ledger ORDER BY entry_id DESC LIMIT 1"
        ).fetchone()

        current_balance = Decimal(str(row["balance_after"])) if row else Decimal("0")
        new_balance = current_balance + fill.cash_flow

        conn.execute(
            """
            INSERT INTO cash_ledger (timestamp, entry_type, amount, balance_after, reference_id, description)
            VALUES (?, 'fill', ?, ?, ?, ?)
            """,
            (
                fill.timestamp.isoformat(),
                str(fill.cash_flow),
                str(new_balance),
                fill.fill_id,
                f"{fill.side.upper()} {fill.size} @ {fill.price}",
            ),
        )

    def set_initial_cash(self, amount: Decimal, timestamp: datetime | None = None) -> None:
        """Set initial cash balance.

        Args:
            amount: Initial cash amount
            timestamp: Optional timestamp (defaults to now)
        """
        conn = self._get_connection()
        ts = timestamp or datetime.now(UTC)

        # Check if there's already an initial entry
        row = conn.execute(
            "SELECT entry_id FROM cash_ledger WHERE entry_type = 'initial'"
        ).fetchone()

        if row:
            # Update existing
            conn.execute(
                "UPDATE cash_ledger SET amount = ?, balance_after = ?, timestamp = ? WHERE entry_id = ?",
                (str(amount), str(amount), ts.isoformat(), row["entry_id"]),
            )
        else:
            # Insert new
            conn.execute(
                """
                INSERT INTO cash_ledger (timestamp, entry_type, amount, balance_after, reference_id, description)
                VALUES (?, 'initial', ?, ?, NULL, 'Initial cash deposit')
                """,
                (ts.isoformat(), str(amount), str(amount)),
            )

        conn.commit()

    def get_cash_balance(self) -> Decimal:
        """Get current cash balance."""
        conn = self._get_connection()
        row = conn.execute(
            "SELECT balance_after FROM cash_ledger ORDER BY entry_id DESC LIMIT 1"
        ).fetchone()
        return Decimal(str(row["balance_after"])) if row else Decimal("0")

    def get_positions(self, open_only: bool = True) -> list[AccountingPosition]:
        """Get all positions.

        Args:
            open_only: If True, only return positions with non-zero size

        Returns:
            List of positions
        """
        conn = self._get_connection()

        query = "SELECT * FROM positions"
        if open_only:
            query += " WHERE net_size != '0'"

        rows = conn.execute(query).fetchall()

        positions = []
        for row in rows:
            positions.append(AccountingPosition(
                position_id=row["position_id"],
                token_id=row["token_id"],
                market_slug=row["market_slug"],
                net_size=Decimal(str(row["net_size"])),
                total_cost=Decimal(str(row["total_cost"])),
                realized_pnl=Decimal(str(row["realized_pnl"])),
                total_fees=Decimal(str(row["total_fees"])),
                fill_count=row["fill_count"],
                last_fill_at=datetime.fromisoformat(row["last_fill_at"]) if row["last_fill_at"] else None,
                created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else None,
                updated_at=datetime.fromisoformat(row["updated_at"]) if row["updated_at"] else None,
            ))

        return positions

    def get_position(self, token_id: str) -> AccountingPosition | None:
        """Get a specific position by token ID.

        Args:
            token_id: Token ID to look up

        Returns:
            Position if found, None otherwise
        """
        conn = self._get_connection()
        row = conn.execute(
            "SELECT * FROM positions WHERE token_id = ?",
            (token_id,),
        ).fetchone()

        if row is None:
            return None

        return AccountingPosition(
            position_id=row["position_id"],
            token_id=row["token_id"],
            market_slug=row["market_slug"],
            net_size=Decimal(str(row["net_size"])),
            total_cost=Decimal(str(row["total_cost"])),
            realized_pnl=Decimal(str(row["realized_pnl"])),
            total_fees=Decimal(str(row["total_fees"])),
            fill_count=row["fill_count"],
            last_fill_at=datetime.fromisoformat(row["last_fill_at"]) if row["last_fill_at"] else None,
            created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else None,
            updated_at=datetime.fromisoformat(row["updated_at"]) if row["updated_at"] else None,
        )

    def get_fills(
        self,
        since: datetime | None = None,
        token_id: str | None = None,
        trader_source: str | None = None,
        limit: int | None = None,
    ) -> list[AccountingFill]:
        """Get fills with optional filtering.

        Args:
            since: Only return fills since this time
            token_id: Filter by token ID
            trader_source: Filter by trader source
            limit: Maximum number of fills to return

        Returns:
            List of fills
        """
        conn = self._get_connection()

        conditions = []
        params = []

        if since:
            conditions.append("timestamp >= ?")
            params.append(since.isoformat())
        if token_id:
            conditions.append("token_id = ?")
            params.append(token_id)
        if trader_source:
            conditions.append("trader_source = ?")
            params.append(trader_source)

        query = "SELECT * FROM fills"
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY timestamp DESC"
        if limit:
            query += f" LIMIT {limit}"

        rows = conn.execute(query, params).fetchall()

        fills = []
        for row in rows:
            fills.append(AccountingFill(
                fill_id=row["fill_id"],
                timestamp=datetime.fromisoformat(row["timestamp"]),
                token_id=row["token_id"],
                side=row["side"],
                size=Decimal(str(row["size"])),
                price=Decimal(str(row["price"])),
                fee=Decimal(str(row["fee"])),
                cash_flow=Decimal(str(row["cash_flow"])),
                market_slug=row["market_slug"],
                market_question=row["market_question"],
                trader_source=row["trader_source"],
                original_tx_hash=row["original_tx_hash"],
            ))

        return fills

    def record_portfolio_snapshot(
        self,
        prices: dict[str, Decimal],
        orderbooks: dict | None = None,
        timestamp: datetime | None = None,
    ) -> PortfolioSnapshot:
        """Record a portfolio snapshot with current valuations.

        Args:
            prices: Dict of token_id -> current price (mid)
            orderbooks: Optional dict of token_id -> orderbook for liquidation calc
            timestamp: Optional timestamp (defaults to now)

        Returns:
            The recorded portfolio snapshot
        """
        conn = self._get_connection()
        snapshot_time = timestamp or datetime.now(UTC)

        # Get cash balance
        cash_balance = self.get_cash_balance()

        # Get all open positions
        positions = self.get_positions(open_only=True)

        # Calculate valuations
        positions_value = Decimal("0")
        liquidation_value = Decimal("0")
        unrealized_pnl = Decimal("0")
        realized_pnl = Decimal("0")
        total_fees = Decimal("0")

        # Insert snapshot header
        conn.execute(
            """
            INSERT INTO portfolio_snapshots (
                snapshot_time, cash_balance, positions_value, liquidation_value,
                realized_pnl, unrealized_pnl, total_fees, nav,
                open_position_count, total_position_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                snapshot_time.isoformat(),
                str(cash_balance),
                "0",  # Will update after calculations
                "0",
                "0",
                "0",
                "0",
                "0",
                len(positions),
                len(self.get_positions(open_only=False)),
            ),
        )

        snapshot_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        # Calculate and record position valuations
        for pos in positions:
            price = prices.get(pos.token_id, Decimal("0.5"))
            mark_to_market = pos.net_size * price
            pos_unrealized = (price - pos.avg_cost_basis) * pos.net_size

            # Calculate liquidation value
            pos_liquidation = mark_to_market  # Default to mark
            if orderbooks and pos.token_id in orderbooks:
                # Use orderbook walking for more accurate liquidation
                from .pnl import OrderBook
                book = orderbooks[pos.token_id]
                if isinstance(book, dict):
                    book = OrderBook.from_dict(pos.token_id, book)
                pos_liquidation = book.get_walk_liquidation_value(pos.net_size, is_yes=True)

            positions_value += mark_to_market
            liquidation_value += pos_liquidation
            unrealized_pnl += pos_unrealized
            realized_pnl += pos.realized_pnl
            total_fees += pos.total_fees

            # Record valuation
            conn.execute(
                """
                INSERT INTO snapshot_valuations (
                    snapshot_id, position_id, token_id, position_size,
                    avg_cost_basis, mark_price, mark_to_market, unrealized_pnl, liquidation_value
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot_id,
                    pos.position_id,
                    pos.token_id,
                    str(pos.net_size),
                    str(pos.avg_cost_basis),
                    str(price),
                    str(mark_to_market),
                    str(pos_unrealized),
                    str(pos_liquidation),
                ),
            )

        # Calculate NAV
        nav = cash_balance + positions_value

        # Update snapshot with calculated values
        conn.execute(
            """
            UPDATE portfolio_snapshots SET
                positions_value = ?,
                liquidation_value = ?,
                realized_pnl = ?,
                unrealized_pnl = ?,
                total_fees = ?,
                nav = ?
            WHERE snapshot_id = ?
            """,
            (
                str(positions_value),
                str(liquidation_value),
                str(realized_pnl),
                str(unrealized_pnl),
                str(total_fees),
                str(nav),
                snapshot_id,
            ),
        )

        conn.commit()

        return PortfolioSnapshot(
            snapshot_id=snapshot_id,
            snapshot_time=snapshot_time,
            cash_balance=cash_balance,
            positions_value=positions_value,
            liquidation_value=liquidation_value,
            realized_pnl=realized_pnl,
            unrealized_pnl=unrealized_pnl,
            total_fees=total_fees,
            nav=nav,
            open_position_count=len(positions),
            total_position_count=len(self.get_positions(open_only=False)),
        )

    def get_latest_snapshot(self) -> PortfolioSnapshot | None:
        """Get the most recent portfolio snapshot.

        Returns:
            Latest snapshot or None if no snapshots exist
        """
        conn = self._get_connection()
        row = conn.execute(
            "SELECT * FROM portfolio_snapshots ORDER BY snapshot_time DESC LIMIT 1"
        ).fetchone()

        if row is None:
            return None

        return PortfolioSnapshot(
            snapshot_id=row["snapshot_id"],
            snapshot_time=datetime.fromisoformat(row["snapshot_time"]),
            cash_balance=Decimal(str(row["cash_balance"])),
            positions_value=Decimal(str(row["positions_value"])),
            liquidation_value=Decimal(str(row["liquidation_value"])),
            realized_pnl=Decimal(str(row["realized_pnl"])),
            unrealized_pnl=Decimal(str(row["unrealized_pnl"])),
            total_fees=Decimal(str(row["total_fees"])),
            nav=Decimal(str(row["nav"])),
            open_position_count=row["open_position_count"],
            total_position_count=row["total_position_count"],
        )

    def get_snapshots(
        self,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int | None = None,
    ) -> list[PortfolioSnapshot]:
        """Get portfolio snapshots with optional filtering.

        Args:
            since: Only return snapshots since this time
            until: Only return snapshots until this time
            limit: Maximum number of snapshots

        Returns:
            List of snapshots
        """
        conn = self._get_connection()

        conditions = []
        params = []

        if since:
            conditions.append("snapshot_time >= ?")
            params.append(since.isoformat())
        if until:
            conditions.append("snapshot_time <= ?")
            params.append(until.isoformat())

        query = "SELECT * FROM portfolio_snapshots"
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY snapshot_time DESC"
        if limit:
            query += f" LIMIT {limit}"

        rows = conn.execute(query, params).fetchall()

        snapshots = []
        for row in rows:
            snapshots.append(PortfolioSnapshot(
                snapshot_id=row["snapshot_id"],
                snapshot_time=datetime.fromisoformat(row["snapshot_time"]),
                cash_balance=Decimal(str(row["cash_balance"])),
                positions_value=Decimal(str(row["positions_value"])),
                liquidation_value=Decimal(str(row["liquidation_value"])),
                realized_pnl=Decimal(str(row["realized_pnl"])),
                unrealized_pnl=Decimal(str(row["unrealized_pnl"])),
                total_fees=Decimal(str(row["total_fees"])),
                nav=Decimal(str(row["nav"])),
                open_position_count=row["open_position_count"],
                total_position_count=row["total_position_count"],
            ))

        return snapshots

    def get_account_summary(self, days: int = 7) -> dict:
        """Get account summary for the last N days.

        Args:
            days: Number of days to include

        Returns:
            Dict with summary statistics
        """
        conn = self._get_connection()
        since = datetime.now(UTC) - timedelta(days=days)

        # Get starting snapshot (closest to since)
        start_row = conn.execute(
            "SELECT * FROM portfolio_snapshots WHERE snapshot_time >= ? ORDER BY snapshot_time ASC LIMIT 1",
            (since.isoformat(),),
        ).fetchone()

        # Get latest snapshot
        latest = self.get_latest_snapshot()

        # Get fills in period
        fills = self.get_fills(since=since)

        # Calculate statistics
        start_nav = Decimal(str(start_row["nav"])) if start_row else Decimal("0")
        current_nav = latest.nav if latest else Decimal("0")
        nav_change = current_nav - start_nav
        nav_change_pct = (nav_change / start_nav * 100) if start_nav > 0 else Decimal("0")

        # Get all snapshots in period for high/low
        snapshots = self.get_snapshots(since=since)
        navs = [s.nav for s in snapshots] if snapshots else [current_nav]
        max_nav = max(navs) if navs else Decimal("0")
        min_nav = min(navs) if navs else Decimal("0")

        # Calculate drawdown
        peak = Decimal("0")
        max_drawdown = Decimal("0")
        max_drawdown_pct = Decimal("0")
        for nav in navs:
            if nav > peak:
                peak = nav
            drawdown = peak - nav
            if drawdown > max_drawdown:
                max_drawdown = drawdown
                if peak > 0:
                    max_drawdown_pct = (drawdown / peak) * 100

        return {
            "period_days": days,
            "period_start": since.isoformat(),
            "starting_nav": float(start_nav),
            "current_nav": float(current_nav),
            "nav_change": float(nav_change),
            "nav_change_pct": float(nav_change_pct),
            "max_nav": float(max_nav),
            "min_nav": float(min_nav),
            "max_drawdown": float(max_drawdown),
            "max_drawdown_pct": float(max_drawdown_pct),
            "total_fills": len(fills),
            "open_positions": len(self.get_positions(open_only=True)),
            "realized_pnl": float(latest.realized_pnl) if latest else float(
                sum(p.realized_pnl for p in self.get_positions(open_only=False))
            ),
            "unrealized_pnl": float(latest.unrealized_pnl) if latest else 0.0,
            "total_fees": float(latest.total_fees) if latest else float(
                sum(p.total_fees for p in self.get_positions(open_only=False))
            ),
            "cash_balance": float(latest.cash_balance) if latest else float(self.get_cash_balance()),
            "liquidation_value": float(latest.liquidation_value) if latest else 0.0,
        }

    def get_exposures(self) -> list[dict]:
        """Get current position exposures.

        Returns:
            List of exposure dicts with position details
        """
        positions = self.get_positions(open_only=True)
        latest = self.get_latest_snapshot()
        total_nav = latest.nav if latest else Decimal("0")

        exposures = []
        for pos in positions:
            # Get latest valuation for this position
            conn = self._get_connection()
            row = conn.execute(
                """
                SELECT sv.* FROM snapshot_valuations sv
                JOIN portfolio_snapshots ps ON sv.snapshot_id = ps.snapshot_id
                WHERE sv.token_id = ?
                ORDER BY ps.snapshot_time DESC
                LIMIT 1
                """,
                (pos.token_id,),
            ).fetchone()

            if row:
                mark_price = Decimal(str(row["mark_price"]))
                mark_to_market = Decimal(str(row["mark_to_market"]))
                liquidation_value = Decimal(str(row["liquidation_value"]))
                unrealized_pnl = Decimal(str(row["unrealized_pnl"]))
            else:
                # Fallback to cost basis
                mark_price = pos.avg_cost_basis
                mark_to_market = pos.net_size * mark_price
                liquidation_value = mark_to_market
                unrealized_pnl = Decimal("0")

            nav_pct = (mark_to_market / total_nav * 100) if total_nav > 0 else Decimal("0")

            exposures.append({
                "token_id": pos.token_id,
                "market_slug": pos.market_slug,
                "net_size": float(pos.net_size),
                "avg_cost_basis": float(pos.avg_cost_basis),
                "mark_price": float(mark_price),
                "mark_to_market": float(mark_to_market),
                "liquidation_value": float(liquidation_value),
                "unrealized_pnl": float(unrealized_pnl),
                "realized_pnl": float(pos.realized_pnl),
                "total_fees": float(pos.total_fees),
                "nav_pct": float(nav_pct),
            })

        # Sort by NAV percentage (descending)
        exposures.sort(key=lambda x: x["nav_pct"], reverse=True)
        return exposures

    def get_cash_ledger(
        self,
        since: datetime | None = None,
        limit: int | None = None,
    ) -> list[CashLedgerEntry]:
        """Get cash ledger entries.

        Args:
            since: Only return entries since this time
            limit: Maximum number of entries

        Returns:
            List of cash ledger entries
        """
        conn = self._get_connection()

        query = "SELECT * FROM cash_ledger"
        params = []

        if since:
            query += " WHERE timestamp >= ?"
            params.append(since.isoformat())

        query += " ORDER BY timestamp DESC"
        if limit:
            query += f" LIMIT {limit}"

        rows = conn.execute(query, params).fetchall()

        entries = []
        for row in rows:
            entries.append(CashLedgerEntry(
                entry_id=row["entry_id"],
                timestamp=datetime.fromisoformat(row["timestamp"]) if row["timestamp"] else None,
                entry_type=row["entry_type"],
                amount=Decimal(str(row["amount"])),
                balance_after=Decimal(str(row["balance_after"])),
                reference_id=row["reference_id"],
                description=row["description"],
            ))

        return entries


def init_accounting_db(db_path: Path | str | None = None) -> AccountingDB:
    """Initialize accounting database with schema.

    Args:
        db_path: Path to database file

    Returns:
        Initialized AccountingDB instance
    """
    db = AccountingDB(db_path)
    db.init_schema()
    return db
