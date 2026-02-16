"""Paper-copy trading for top-performing Polymarket traders.

Copies trades from top-K traders with configurable slippage model,
tracks copy performance vs original traders.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from .pnl import Fill as PnlFill
from .trader_fills import TraderFill, TraderFillTracker

# Constants
DEFAULT_DATA_DIR = Path("data/copy_trading")
COPY_FILLS_FILE = "fills.jsonl"
COPY_POSITIONS_FILE = "positions.json"
COPY_EQUITY_FILE = "equity.jsonl"
COPY_CONFIG_FILE = "config.json"

# Slippage model defaults (in basis points)
DEFAULT_SLIPPAGE_BPS = 10  # 0.1% base slippage
DEFAULT_SPREAD_IMPACT_BPS = 50  # 0.5% spread impact
DEFAULT_SIZE_IMPACT_FACTOR = Decimal("0.0001")  # 1bp per $1000 of size


@dataclass
class SlippageModel:
    """Slippage model for paper-copy trading.

    Models the price impact of copying a trade based on:
    - Base slippage (fixed cost)
    - Spread impact (market liquidity)
    - Size impact (trade size relative to market)

    Attributes:
        base_slippage_bps: Fixed slippage in basis points
        spread_impact_bps: Spread-based slippage in basis points
        size_impact_factor: Multiplier for size-based impact
        max_slippage_bps: Maximum slippage cap
    """

    base_slippage_bps: int = DEFAULT_SLIPPAGE_BPS
    spread_impact_bps: int = DEFAULT_SPREAD_IMPACT_BPS
    size_impact_factor: Decimal = DEFAULT_SIZE_IMPACT_FACTOR
    max_slippage_bps: int = 200  # Max 2% slippage

    def calculate_slippage(
        self,
        original_price: Decimal,
        size: Decimal,
        spread: Decimal | None = None,
    ) -> Decimal:
        """Calculate slippage-adjusted price.

        Args:
            original_price: The original trader's fill price
            size: Size of the trade being copied
            spread: Optional bid-ask spread for liquidity adjustment

        Returns:
            Slippage amount (positive = worse price for buyer)
        """
        # Base slippage
        slippage_bps = Decimal(str(self.base_slippage_bps))

        # Spread impact
        if spread:
            spread_bps = (spread / original_price) * 10000
            slippage_bps += min(spread_bps * Decimal(str(self.spread_impact_bps)) / 100, 100)

        # Size impact (larger trades = more slippage)
        size_usd = size * original_price
        size_impact = size_usd * self.size_impact_factor
        slippage_bps += size_impact

        # Cap at maximum
        slippage_bps = min(slippage_bps, Decimal(str(self.max_slippage_bps)))

        # Convert to price adjustment
        slippage_amount = original_price * (slippage_bps / 10000)
        return slippage_amount

    def adjust_price_for_copy(
        self,
        side: str,
        original_price: Decimal,
        size: Decimal,
        spread: Decimal | None = None,
    ) -> Decimal:
        """Get the execution price for a copy trade.

        For buys: price is higher (worse)
        For sells: price is lower (worse)

        Args:
            side: 'buy' or 'sell'
            original_price: Original trader's fill price
            size: Trade size
            spread: Optional spread

        Returns:
            Adjusted execution price
        """
        slippage = self.calculate_slippage(original_price, size, spread)

        if side == "buy":
            # Buyer pays more
            return min(original_price + slippage, Decimal("0.99"))
        else:
            # Seller receives less
            return max(original_price - slippage, Decimal("0.01"))

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "base_slippage_bps": self.base_slippage_bps,
            "spread_impact_bps": self.spread_impact_bps,
            "size_impact_factor": str(self.size_impact_factor),
            "max_slippage_bps": self.max_slippage_bps,
        }

    @classmethod
    def from_dict(cls, data: dict) -> SlippageModel:
        """Create from dictionary."""
        return cls(
            base_slippage_bps=data.get("base_slippage_bps", DEFAULT_SLIPPAGE_BPS),
            spread_impact_bps=data.get("spread_impact_bps", DEFAULT_SPREAD_IMPACT_BPS),
            size_impact_factor=Decimal(str(data.get("size_impact_factor", DEFAULT_SIZE_IMPACT_FACTOR))),
            max_slippage_bps=data.get("max_slippage_bps", 200),
        )


@dataclass(frozen=True)
class CopyFill:
    """A copied trade fill.

    Tracks the relationship between original and copy trade.
    """

    # Copy trade details
    copy_fill_id: str
    timestamp: str
    token_id: str
    side: str
    size: Decimal
    copy_price: Decimal
    fee: Decimal

    # Original trade reference
    original_trader: str
    original_tx_hash: str | None
    original_price: Decimal
    original_timestamp: str

    # Market info
    market_slug: str | None = None
    market_question: str | None = None

    # Slippage tracking
    slippage_amount: Decimal = Decimal("0")
    slippage_bps: Decimal = Decimal("0")

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "copy_fill_id": self.copy_fill_id,
            "timestamp": self.timestamp,
            "token_id": self.token_id,
            "side": self.side,
            "size": str(self.size),
            "copy_price": str(self.copy_price),
            "fee": str(self.fee),
            "original_trader": self.original_trader,
            "original_tx_hash": self.original_tx_hash,
            "original_price": str(self.original_price),
            "original_timestamp": self.original_timestamp,
            "market_slug": self.market_slug,
            "market_question": self.market_question,
            "slippage_amount": str(self.slippage_amount),
            "slippage_bps": str(self.slippage_bps),
        }

    @classmethod
    def from_dict(cls, data: dict) -> CopyFill:
        """Create from dictionary."""
        return cls(
            copy_fill_id=data["copy_fill_id"],
            timestamp=data["timestamp"],
            token_id=data["token_id"],
            side=data["side"],
            size=Decimal(str(data["size"])),
            copy_price=Decimal(str(data["copy_price"])),
            fee=Decimal(str(data.get("fee", "0"))),
            original_trader=data["original_trader"],
            original_tx_hash=data.get("original_tx_hash"),
            original_price=Decimal(str(data["original_price"])),
            original_timestamp=data["original_timestamp"],
            market_slug=data.get("market_slug"),
            market_question=data.get("market_question"),
            slippage_amount=Decimal(str(data.get("slippage_amount", "0"))),
            slippage_bps=Decimal(str(data.get("slippage_bps", "0"))),
        )

    def to_pnl_fill(self) -> PnlFill:
        """Convert to PnL Fill."""
        return PnlFill(
            token_id=self.token_id,
            side=self.side,
            size=self.size,
            price=self.copy_price,
            fee=self.fee,
            timestamp=self.timestamp,
            transaction_hash=self.copy_fill_id,
            market_slug=self.market_slug,
        )

    @property
    def cash_flow(self) -> Decimal:
        """Cash flow impact of this fill."""
        notional = self.size * self.copy_price
        if self.side == "buy":
            return -(notional + self.fee)
        else:
            return notional - self.fee


@dataclass
class CopyPosition:
    """Position state for copy trading."""

    token_id: str
    net_size: Decimal = field(default_factory=lambda: Decimal("0"))
    total_cost: Decimal = field(default_factory=lambda: Decimal("0"))
    realized_pnl: Decimal = field(default_factory=lambda: Decimal("0"))
    total_fees: Decimal = field(default_factory=lambda: Decimal("0"))
    copy_count: int = 0

    def add_fill(self, fill: CopyFill) -> None:
        """Add a copy fill to this position."""
        if fill.side == "buy":
            cost = fill.size * fill.copy_price
            self.net_size += fill.size
            self.total_cost += cost
            self.total_fees += fill.fee
        else:
            # Calculate realized PnL for the portion that closes existing position
            if self.net_size > 0:
                sell_size = min(fill.size, self.net_size)
                cost_basis = sell_size * self.avg_cost_basis
                proceeds = sell_size * fill.copy_price
                realized = proceeds - cost_basis - fill.fee
                self.realized_pnl += realized
                self.net_size -= sell_size
                self.total_cost -= cost_basis

                # Handle remaining as short (if selling more than we have)
                remaining = fill.size - sell_size
                if remaining > 0:
                    self.net_size -= remaining
            else:
                # Short selling - no position to close
                self.net_size -= fill.size

            self.total_fees += fill.fee

        self.copy_count += 1

    @property
    def avg_cost_basis(self) -> Decimal:
        """Average cost per share."""
        if self.net_size <= 0:
            return Decimal("0")
        return self.total_cost / self.net_size

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "token_id": self.token_id,
            "net_size": str(self.net_size),
            "avg_cost_basis": str(self.avg_cost_basis),
            "total_cost": str(self.total_cost),
            "realized_pnl": str(self.realized_pnl),
            "total_fees": str(self.total_fees),
            "copy_count": self.copy_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> CopyPosition:
        """Create from dictionary."""
        # Note: avg_cost_basis is computed, not stored
        return cls(
            token_id=data["token_id"],
            net_size=Decimal(str(data.get("net_size", "0"))),
            total_cost=Decimal(str(data.get("total_cost", "0"))),
            realized_pnl=Decimal(str(data.get("realized_pnl", "0"))),
            total_fees=Decimal(str(data.get("total_fees", "0"))),
            copy_count=data.get("copy_count", 0),
        )


@dataclass
class CopyEquitySnapshot:
    """Equity snapshot for copy trading portfolio."""

    timestamp: str
    cash_balance: Decimal
    positions_value: Decimal
    realized_pnl: Decimal
    unrealized_pnl: Decimal
    total_fees: Decimal
    net_equity: Decimal
    position_count: int
    open_position_count: int
    copied_traders: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "cash_balance": str(self.cash_balance),
            "positions_value": str(self.positions_value),
            "realized_pnl": str(self.realized_pnl),
            "unrealized_pnl": str(self.unrealized_pnl),
            "total_fees": str(self.total_fees),
            "net_equity": str(self.net_equity),
            "position_count": self.position_count,
            "open_position_count": self.open_position_count,
            "copied_traders": self.copied_traders,
        }

    @classmethod
    def from_dict(cls, data: dict) -> CopyEquitySnapshot:
        """Create from dictionary."""
        return cls(
            timestamp=data["timestamp"],
            cash_balance=Decimal(str(data.get("cash_balance", "0"))),
            positions_value=Decimal(str(data.get("positions_value", "0"))),
            realized_pnl=Decimal(str(data.get("realized_pnl", "0"))),
            unrealized_pnl=Decimal(str(data.get("unrealized_pnl", "0"))),
            total_fees=Decimal(str(data.get("total_fees", "0"))),
            net_equity=Decimal(str(data.get("net_equity", "0"))),
            position_count=data.get("position_count", 0),
            open_position_count=data.get("open_position_count", 0),
            copied_traders=data.get("copied_traders", []),
        )


@dataclass
class CopyTradeConfig:
    """Configuration for copy trading."""

    # Trader selection
    top_k: int = 5  # Number of top traders to copy
    min_trader_score: float = 30.0  # Minimum score to qualify

    # Position sizing
    position_size_usd: Decimal = field(default_factory=lambda: Decimal("100"))
    max_position_per_market: Decimal = field(default_factory=lambda: Decimal("500"))
    max_total_exposure: Decimal = field(default_factory=lambda: Decimal("2000"))

    # Risk limits
    max_open_positions: int = 20
    max_positions_per_trader: int = 5

    # Slippage model
    slippage: SlippageModel = field(default_factory=SlippageModel)

    # Starting capital
    starting_cash: Decimal = field(default_factory=lambda: Decimal("10000"))

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "top_k": self.top_k,
            "min_trader_score": self.min_trader_score,
            "position_size_usd": str(self.position_size_usd),
            "max_position_per_market": str(self.max_position_per_market),
            "max_total_exposure": str(self.max_total_exposure),
            "max_open_positions": self.max_open_positions,
            "max_positions_per_trader": self.max_positions_per_trader,
            "slippage": self.slippage.to_dict(),
            "starting_cash": str(self.starting_cash),
        }

    @classmethod
    def from_dict(cls, data: dict) -> CopyTradeConfig:
        """Create from dictionary."""
        return cls(
            top_k=data.get("top_k", 5),
            min_trader_score=data.get("min_trader_score", 30.0),
            position_size_usd=Decimal(str(data.get("position_size_usd", "100"))),
            max_position_per_market=Decimal(str(data.get("max_position_per_market", "500"))),
            max_total_exposure=Decimal(str(data.get("max_total_exposure", "2000"))),
            max_open_positions=data.get("max_open_positions", 20),
            max_positions_per_trader=data.get("max_positions_per_trader", 5),
            slippage=SlippageModel.from_dict(data.get("slippage", {})),
            starting_cash=Decimal(str(data.get("starting_cash", "10000"))),
        )


class PaperCopyEngine:
    """Engine for paper-copy trading top Polymarket traders.

    Features:
    - Copy trades from top-K traders with slippage model
    - Track copy performance vs original traders
    - NAV accounting for copy portfolio
    - Configurable position sizing and risk limits
    """

    data_dir: Path
    config: CopyTradeConfig
    fills: list[CopyFill]
    positions: dict[str, CopyPosition]
    cash_balance: Decimal
    copied_traders: set[str]

    def __init__(
        self,
        data_dir: Path | str | None = None,
        config: CopyTradeConfig | None = None,
    ) -> None:
        """Initialize paper-copy engine.

        Args:
            data_dir: Directory for copy trading data
            config: Copy trading configuration
        """
        self.data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.config = config or self._load_or_create_config()
        self.fills = []
        self.positions = {}
        self.cash_balance = self.config.starting_cash
        self.copied_traders = set()

        self._load_state()

    def _load_or_create_config(self) -> CopyTradeConfig:
        """Load existing config or create default."""
        config_path = self.data_dir / COPY_CONFIG_FILE
        if config_path.exists():
            try:
                data = json.loads(config_path.read_text())
                return CopyTradeConfig.from_dict(data)
            except (json.JSONDecodeError, KeyError):
                pass

        config = CopyTradeConfig()
        self._save_config(config)
        return config

    def _save_config(self, config: CopyTradeConfig) -> None:
        """Save config to disk."""
        config_path = self.data_dir / COPY_CONFIG_FILE
        config_path.write_text(json.dumps(config.to_dict(), indent=2))

    @property
    def fills_path(self) -> Path:
        """Path to fills journal."""
        return self.data_dir / COPY_FILLS_FILE

    @property
    def positions_path(self) -> Path:
        """Path to positions file."""
        return self.data_dir / COPY_POSITIONS_FILE

    @property
    def equity_path(self) -> Path:
        """Path to equity curve file."""
        return self.data_dir / COPY_EQUITY_FILE

    def _load_state(self) -> None:
        """Load state from disk."""
        # Load fills
        if self.fills_path.exists():
            with open(self.fills_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        fill = CopyFill.from_dict(json.loads(line))
                        self.fills.append(fill)
                        self.copied_traders.add(fill.original_trader)
                    except (json.JSONDecodeError, ValueError):
                        continue

        # Replay fills to build positions
        self._rebuild_state()

    def _rebuild_state(self) -> None:
        """Rebuild positions and cash from fill history."""
        self.positions = {}
        self.cash_balance = self.config.starting_cash

        for fill in sorted(self.fills, key=lambda f: f.timestamp):
            self._apply_fill(fill, record=False)

    def _apply_fill(self, fill: CopyFill, record: bool = True) -> None:
        """Apply a fill to state.

        Args:
            fill: CopyFill to apply
            record: Whether to record to journal
        """
        # Update cash
        self.cash_balance += fill.cash_flow

        # Update position
        if fill.token_id not in self.positions:
            self.positions[fill.token_id] = CopyPosition(token_id=fill.token_id)

        self.positions[fill.token_id].add_fill(fill)
        self.copied_traders.add(fill.original_trader)

        if record:
            self.fills.append(fill)
            with open(self.fills_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(fill.to_dict(), sort_keys=True) + "\n")

    def copy_trade(
        self,
        original_fill: TraderFill,
        size: Decimal | None = None,
        spread: Decimal | None = None,
    ) -> CopyFill | None:
        """Create a copy of a trader's fill.

        Args:
            original_fill: The original trader's fill
            size: Optional override size (defaults to config position_size)
            spread: Optional spread for slippage calculation

        Returns:
            CopyFill if trade was executed, None if skipped
        """
        # Check risk limits
        if not self._can_copy_trade(original_fill):
            return None

        # Determine size
        if size is None:
            size = self.config.position_size_usd / original_fill.price

        # Apply slippage model
        copy_price = self.config.slippage.adjust_price_for_copy(
            original_fill.side,
            original_fill.price,
            size,
            spread,
        )

        # Calculate fee (assume 0.2% taker fee)
        fee_rate = Decimal("0.002")
        fee = size * copy_price * fee_rate

        # Generate fill ID
        fill_id = f"copy_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{len(self.fills)}"

        # Calculate slippage metrics
        slippage_amount = abs(copy_price - original_fill.price)
        slippage_bps = (slippage_amount / original_fill.price) * 10000 if original_fill.price > 0 else Decimal("0")

        copy_fill = CopyFill(
            copy_fill_id=fill_id,
            timestamp=datetime.now(UTC).isoformat(),
            token_id=original_fill.token_id,
            side=original_fill.side,
            size=size,
            copy_price=copy_price,
            fee=fee,
            original_trader=original_fill.transaction_hash or "unknown" if hasattr(original_fill, 'transaction_hash') else "unknown",
            original_tx_hash=original_fill.transaction_hash,
            original_price=original_fill.price,
            original_timestamp=original_fill.timestamp,
            market_slug=original_fill.market_slug,
            market_question=original_fill.market_question,
            slippage_amount=slippage_amount,
            slippage_bps=slippage_bps,
        )

        # Extract trader address from the fill data if possible
        # Note: original_fill doesn't have trader address, caller should provide it

        self._apply_fill(copy_fill)
        return copy_fill

    def _can_copy_trade(self, original_fill: TraderFill) -> bool:
        """Check if a trade can be copied given risk limits.

        Args:
            original_fill: The original trader's fill

        Returns:
            True if trade can be copied
        """
        # Check max open positions
        open_count = len([p for p in self.positions.values() if p.net_size != 0])
        if open_count >= self.config.max_open_positions:
            return False

        # Check if we already have position in this market
        if original_fill.token_id in self.positions:
            pos = self.positions[original_fill.token_id]
            pos_value = pos.net_size * original_fill.price
            if pos_value >= self.config.max_position_per_market:
                return False

        return True

    def sync_from_trader(
        self,
        trader_address: str,
        fill_tracker: TraderFillTracker,
        max_copies: int = 10,
    ) -> list[CopyFill]:
        """Sync and copy new fills from a tracked trader.

        Args:
            trader_address: Trader to copy from
            fill_tracker: TraderFillTracker with trader data
            max_copies: Maximum new trades to copy in this sync

        Returns:
            List of new CopyFills created
        """
        # Load trader's fills
        fills = fill_tracker.load_fills(trader_address)

        # Find fills we haven't copied yet
        existing_original_ids = {
            f.original_tx_hash for f in self.fills if f.original_tx_hash
        }

        new_copies = []
        for fill in fills:
            if fill.transaction_hash and fill.transaction_hash in existing_original_ids:
                continue

            # Mark the original as "tracked" even if we don't copy
            existing_original_ids.add(fill.transaction_hash)

            copy_fill = self.copy_trade(fill)
            if copy_fill:
                # Set the correct original trader address
                copy_fill = CopyFill(
                    copy_fill_id=copy_fill.copy_fill_id,
                    timestamp=copy_fill.timestamp,
                    token_id=copy_fill.token_id,
                    side=copy_fill.side,
                    size=copy_fill.size,
                    copy_price=copy_fill.copy_price,
                    fee=copy_fill.fee,
                    original_trader=trader_address,
                    original_tx_hash=fill.transaction_hash,
                    original_price=copy_fill.original_price,
                    original_timestamp=copy_fill.original_timestamp,
                    market_slug=copy_fill.market_slug,
                    market_question=copy_fill.market_question,
                    slippage_amount=copy_fill.slippage_amount,
                    slippage_bps=copy_fill.slippage_bps,
                )
                new_copies.append(copy_fill)

                if len(new_copies) >= max_copies:
                    break

        return new_copies

    def compute_equity(
        self,
        current_prices: dict[str, Decimal] | None = None,
    ) -> CopyEquitySnapshot:
        """Compute current equity snapshot.

        Args:
            current_prices: Optional token prices for mark-to-market

        Returns:
            CopyEquitySnapshot with current state
        """
        positions_value = Decimal("0")
        unrealized_pnl = Decimal("0")
        open_count = 0

        for token_id, pos in self.positions.items():
            if pos.net_size == 0:
                continue

            price = Decimal("0.5")
            if current_prices and token_id in current_prices:
                price = current_prices[token_id]

            pos_value = pos.net_size * price
            positions_value += pos_value
            unrealized_pnl += (price - pos.avg_cost_basis) * pos.net_size
            open_count += 1

        realized_pnl = sum(p.realized_pnl for p in self.positions.values())
        total_fees = sum(p.total_fees for p in self.positions.values())

        return CopyEquitySnapshot(
            timestamp=datetime.now(UTC).isoformat(),
            cash_balance=self.cash_balance,
            positions_value=positions_value,
            realized_pnl=realized_pnl,
            unrealized_pnl=unrealized_pnl,
            total_fees=total_fees,
            net_equity=self.cash_balance + positions_value,
            position_count=len(self.positions),
            open_position_count=open_count,
            copied_traders=list(self.copied_traders),
        )

    def record_equity(
        self,
        current_prices: dict[str, Decimal] | None = None,
    ) -> CopyEquitySnapshot:
        """Compute and record equity snapshot.

        Args:
            current_prices: Optional token prices

        Returns:
            Recorded CopyEquitySnapshot
        """
        snapshot = self.compute_equity(current_prices)

        with open(self.equity_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(snapshot.to_dict(), sort_keys=True) + "\n")

        return snapshot

    def get_equity_curve(self) -> list[CopyEquitySnapshot]:
        """Get full equity curve history."""
        if not self.equity_path.exists():
            return []

        snapshots = []
        with open(self.equity_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    snapshots.append(CopyEquitySnapshot.from_dict(json.loads(line)))
                except (json.JSONDecodeError, ValueError):
                    continue

        return snapshots

    def get_performance_summary(self) -> dict:
        """Get performance summary for copy trading."""
        equity = self.compute_equity()
        curve = self.get_equity_curve()

        total_return = Decimal("0")
        total_return_pct = Decimal("0")
        max_drawdown = Decimal("0")

        if len(curve) >= 2:
            start = curve[0].net_equity
            current = equity.net_equity
            if start > 0:
                total_return = current - start
                total_return_pct = (total_return / start) * 100

            peak = curve[0].net_equity
            for snap in curve:
                if snap.net_equity > peak:
                    peak = snap.net_equity
                drawdown = peak - snap.net_equity
                if drawdown > max_drawdown:
                    max_drawdown = drawdown

        return {
            "starting_cash": float(self.config.starting_cash),
            "current_cash": float(self.cash_balance),
            "current_equity": float(equity.net_equity),
            "total_return": float(total_return),
            "total_return_pct": float(total_return_pct),
            "max_drawdown": float(max_drawdown),
            "realized_pnl": float(equity.realized_pnl),
            "unrealized_pnl": float(equity.unrealized_pnl),
            "total_fees": float(equity.total_fees),
            "open_positions": equity.open_position_count,
            "total_positions": equity.position_count,
            "copied_traders": len(self.copied_traders),
            "total_copy_trades": len(self.fills),
        }
