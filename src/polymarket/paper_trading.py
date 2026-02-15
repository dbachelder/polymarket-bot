"""Paper trading engine for live-readiness PnL tracking.

Provides end-to-end paper trading simulation:
- Fills storage and replay
- Position tracking from fill history
- Equity curve calculation over time
- Reconciliation against collector snapshots
- PnL attribution by market and time period

Features:
- Persistent fill journal (append-only)
- Position state reconstruction from fills
- Mark-to-market equity using snapshot prices
- Drift detection between calculated and expected PnL
- Time-series equity reporting
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from .pnl import Fill, OrderBook, PnLReport, PnLVerifier, load_orderbooks_from_snapshot


# Constants
DEFAULT_DATA_DIR = Path("data/paper_trading")
FILLS_FILE = "fills.jsonl"
POSITIONS_FILE = "positions.json"
EQUITY_FILE = "equity.jsonl"


@dataclass
class PositionState:
    """Current state of a position (reconstructed from fills)."""

    token_id: str
    market_slug: str | None = None
    market_question: str | None = None
    net_size: Decimal = field(default_factory=lambda: Decimal("0"))
    avg_cost_basis: Decimal = field(default_factory=lambda: Decimal("0"))
    total_cost: Decimal = field(default_factory=lambda: Decimal("0"))
    realized_pnl: Decimal = field(default_factory=lambda: Decimal("0"))
    total_fees: Decimal = field(default_factory=lambda: Decimal("0"))
    last_fill_at: str | None = None
    fill_count: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "token_id": self.token_id,
            "market_slug": self.market_slug,
            "market_question": self.market_question,
            "net_size": float(self.net_size),
            "avg_cost_basis": float(self.avg_cost_basis),
            "total_cost": float(self.total_cost),
            "realized_pnl": float(self.realized_pnl),
            "total_fees": float(self.total_fees),
            "last_fill_at": self.last_fill_at,
            "fill_count": self.fill_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> PositionState:
        """Create from dictionary."""
        return cls(
            token_id=data["token_id"],
            market_slug=data.get("market_slug"),
            market_question=data.get("market_question"),
            net_size=Decimal(str(data.get("net_size", 0))),
            avg_cost_basis=Decimal(str(data.get("avg_cost_basis", 0))),
            total_cost=Decimal(str(data.get("total_cost", 0))),
            realized_pnl=Decimal(str(data.get("realized_pnl", 0))),
            total_fees=Decimal(str(data.get("total_fees", 0))),
            last_fill_at=data.get("last_fill_at"),
            fill_count=data.get("fill_count", 0),
        )


@dataclass
class EquitySnapshot:
    """Equity state at a point in time."""

    timestamp: str  # ISO8601
    cash_balance: Decimal
    mark_to_market: Decimal  # Value of positions at mid prices
    liquidation_value: Decimal  # Value walking the book
    realized_pnl: Decimal
    unrealized_pnl: Decimal
    total_fees: Decimal
    net_equity: Decimal  # cash + mark_to_market
    position_count: int
    open_position_count: int

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp,
            "cash_balance": float(self.cash_balance),
            "mark_to_market": float(self.mark_to_market),
            "liquidation_value": float(self.liquidation_value),
            "realized_pnl": float(self.realized_pnl),
            "unrealized_pnl": float(self.unrealized_pnl),
            "total_fees": float(self.total_fees),
            "net_equity": float(self.net_equity),
            "position_count": self.position_count,
            "open_position_count": self.open_position_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> EquitySnapshot:
        """Create from dictionary."""
        return cls(
            timestamp=data["timestamp"],
            cash_balance=Decimal(str(data["cash_balance"])),
            mark_to_market=Decimal(str(data["mark_to_market"])),
            liquidation_value=Decimal(str(data["liquidation_value"])),
            realized_pnl=Decimal(str(data["realized_pnl"])),
            unrealized_pnl=Decimal(str(data["unrealized_pnl"])),
            total_fees=Decimal(str(data["total_fees"])),
            net_equity=Decimal(str(data["net_equity"])),
            position_count=data["position_count"],
            open_position_count=data["open_position_count"],
        )


@dataclass
class ReconciliationResult:
    """Result of reconciling positions against a snapshot."""

    snapshot_timestamp: str
    positions_reconciled: int
    positions_with_drift: int
    total_drift_usd: Decimal
    max_drift_pct: Decimal
    warnings: list[str] = field(default_factory=list)
    position_drifts: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "snapshot_timestamp": self.snapshot_timestamp,
            "positions_reconciled": self.positions_reconciled,
            "positions_with_drift": self.positions_with_drift,
            "total_drift_usd": float(self.total_drift_usd),
            "max_drift_pct": float(self.max_drift_pct),
            "warnings": self.warnings,
            "position_drifts": self.position_drifts,
        }


@dataclass
class PaperTradingEngine:
    """Paper trading engine with persistent fill journal.

    Manages:
    - Append-only fill journal (fills.jsonl)
    - Position state reconstruction from fills
    - Mark-to-market equity calculations
    - Snapshot reconciliation for drift detection
    """

    data_dir: Path = field(default_factory=lambda: Path(DEFAULT_DATA_DIR))
    starting_cash: Decimal = field(default_factory=lambda: Decimal("10000"))
    _verifier: PnLVerifier | None = field(default=None, repr=False)
    _fills_loaded: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize data directory and verifier."""
        self.data_dir = Path(self.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        if self._verifier is None:
            self._verifier = PnLVerifier(starting_cash=self.starting_cash)

    @property
    def fills_path(self) -> Path:
        """Path to fills journal file."""
        return self.data_dir / FILLS_FILE

    @property
    def positions_path(self) -> Path:
        """Path to positions state file."""
        return self.data_dir / POSITIONS_FILE

    @property
    def equity_path(self) -> Path:
        """Path to equity curve file."""
        return self.data_dir / EQUITY_FILE

    def record_fill(
        self,
        token_id: str,
        side: str,
        size: Decimal,
        price: Decimal,
        fee: Decimal = Decimal("0"),
        timestamp: str | None = None,
        market_slug: str | None = None,
        market_question: str | None = None,
        transaction_hash: str | None = None,
    ) -> Fill:
        """Record a fill to the journal and update state.

        Args:
            token_id: Token ID (YES or NO token)
            side: 'buy' or 'sell'
            size: Number of shares
            price: Execution price (0.0 to 1.0)
            fee: Trading fee paid
            timestamp: ISO8601 timestamp (defaults to now)
            market_slug: Market identifier
            market_question: Market question text
            transaction_hash: Optional blockchain tx hash

        Returns:
            The recorded Fill object
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc).isoformat()

        fill = Fill(
            token_id=token_id,
            side=side.lower(),
            size=size,
            price=price,
            fee=fee,
            timestamp=timestamp,
            market_slug=market_slug,
            transaction_hash=transaction_hash,
        )

        # Append to journal
        self._append_fill_to_journal(fill, market_question)

        # Update verifier state
        self._verifier.add_fill(fill)

        return fill

    def _append_fill_to_journal(self, fill: Fill, market_question: str | None = None) -> None:
        """Append a fill to the journal file."""
        record = {
            "token_id": fill.token_id,
            "side": fill.side,
            "size": str(fill.size),
            "price": str(fill.price),
            "fee": str(fill.fee),
            "timestamp": fill.timestamp,
            "market_slug": fill.market_slug,
            "market_question": market_question,
            "transaction_hash": fill.transaction_hash,
        }

        # Append as newline-delimited JSON
        with open(self.fills_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")

    def load_fills(self, since: str | None = None) -> list[Fill]:
        """Load fills from journal.

        Args:
            since: Optional ISO timestamp to filter from

        Returns:
            List of Fill objects
        """
        if not self.fills_path.exists():
            return []

        fills = []
        since_dt = None
        if since:
            since_dt = datetime.fromisoformat(since.replace("Z", "+00:00"))

        with open(self.fills_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    fill = Fill.from_dict(data)
                    if since_dt is None or fill.datetime_utc >= since_dt:
                        fills.append(fill)
                except (json.JSONDecodeError, ValueError):
                    continue

        return fills

    def rebuild_state_from_fills(self, since: str | None = None) -> None:
        """Rebuild verifier state from fill journal.

        Args:
            since: Optional ISO timestamp to rebuild from
        """
        self._verifier = PnLVerifier(starting_cash=self.starting_cash)
        fills = self.load_fills(since=since)
        self._verifier.add_fills(fills)
        self._fills_loaded = True

    def get_positions(self) -> dict[str, PositionState]:
        """Get current positions reconstructed from fills.

        Returns:
            Dict of token_id -> PositionState
        """
        if not self._fills_loaded:
            self.rebuild_state_from_fills()

        positions: dict[str, PositionState] = {}

        for token_id, pos in self._verifier.positions.items():
            positions[token_id] = PositionState(
                token_id=token_id,
                market_slug=pos.market_slug,
                net_size=pos.net_size,
                avg_cost_basis=pos.avg_cost_basis,
                total_cost=pos.total_cost,
                realized_pnl=pos.realized_pnl,
                total_fees=pos.total_fees,
                fill_count=pos.buy_count + pos.sell_count,
            )

        return positions

    def save_positions(self) -> Path:
        """Save current positions to state file.

        Returns:
            Path to saved file
        """
        positions = self.get_positions()
        data = {
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "positions": {k: v.to_dict() for k, v in positions.items()},
        }
        self.positions_path.write_text(json.dumps(data, indent=2))
        return self.positions_path

    def compute_equity(
        self,
        orderbooks: dict[str, OrderBook] | None = None,
        snapshot_path: Path | None = None,
    ) -> EquitySnapshot:
        """Compute current equity snapshot.

        Args:
            orderbooks: Optional dict of token_id -> OrderBook
            snapshot_path: Optional path to collector snapshot for prices

        Returns:
            EquitySnapshot with current equity state
        """
        if not self._fills_loaded:
            self.rebuild_state_from_fills()

        # Load orderbooks from snapshot if provided
        if snapshot_path and snapshot_path.exists():
            orderbooks = load_orderbooks_from_snapshot(snapshot_path)

        # Get PnL report
        pnl_report = self._verifier.compute_pnl(orderbooks=orderbooks)

        # Count positions
        positions = self.get_positions()
        open_positions = [p for p in positions.values() if p.net_size != 0]

        return EquitySnapshot(
            timestamp=datetime.now(timezone.utc).isoformat(),
            cash_balance=self._verifier.cash_balance,
            mark_to_market=pnl_report.mark_to_mid,
            liquidation_value=pnl_report.liquidation_value,
            realized_pnl=pnl_report.realized_pnl,
            unrealized_pnl=pnl_report.unrealized_pnl,
            total_fees=pnl_report.total_fees,
            net_equity=self._verifier.cash_balance + pnl_report.mark_to_mid,
            position_count=len(positions),
            open_position_count=len(open_positions),
        )

    def record_equity(
        self,
        orderbooks: dict[str, OrderBook] | None = None,
        snapshot_path: Path | None = None,
    ) -> EquitySnapshot:
        """Compute and record equity to equity curve file.

        Args:
            orderbooks: Optional dict of token_id -> OrderBook
            snapshot_path: Optional path to collector snapshot

        Returns:
            Recorded EquitySnapshot
        """
        equity = self.compute_equity(orderbooks=orderbooks, snapshot_path=snapshot_path)

        # Append to equity curve
        with open(self.equity_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(equity.to_dict(), sort_keys=True) + "\n")

        return equity

    def load_equity_curve(self, since: str | None = None) -> list[EquitySnapshot]:
        """Load equity curve history.

        Args:
            since: Optional ISO timestamp to filter from

        Returns:
            List of EquitySnapshot objects
        """
        if not self.equity_path.exists():
            return []

        snapshots = []
        since_dt = None
        if since:
            since_dt = datetime.fromisoformat(since.replace("Z", "+00:00"))

        with open(self.equity_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    snap = EquitySnapshot.from_dict(data)
                    snap_dt = datetime.fromisoformat(snap.timestamp.replace("Z", "+00:00"))
                    if since_dt is None or snap_dt >= since_dt:
                        snapshots.append(snap)
                except (json.JSONDecodeError, ValueError):
                    continue

        return snapshots

    def reconcile_against_snapshot(
        self,
        snapshot_path: Path,
        drift_threshold_usd: Decimal = Decimal("0.01"),
        drift_threshold_pct: Decimal = Decimal("0.01"),
    ) -> ReconciliationResult:
        """Reconcile positions against a collector snapshot.

        Compares calculated position values against snapshot prices to detect drift.

        Args:
            snapshot_path: Path to collector snapshot
            drift_threshold_usd: Minimum USD drift to flag
            drift_threshold_pct: Minimum percentage drift to flag

        Returns:
            ReconciliationResult with drift analysis
        """
        if not self._fills_loaded:
            self.rebuild_state_from_fills()

        # Load snapshot orderbooks
        orderbooks = load_orderbooks_from_snapshot(snapshot_path)

        # Get snapshot timestamp from filename or file content
        try:
            snapshot_data = json.loads(snapshot_path.read_text())
            snapshot_ts = snapshot_data.get("generated_at", datetime.now(timezone.utc).isoformat())
        except (json.JSONDecodeError, FileNotFoundError):
            snapshot_ts = datetime.now(timezone.utc).isoformat()

        positions = self.get_positions()
        drifts = []
        warnings = []
        total_drift = Decimal("0")
        max_drift_pct = Decimal("0")

        for token_id, pos in positions.items():
            if pos.net_size == 0:
                continue

            if token_id not in orderbooks:
                warnings.append(f"No orderbook found for {token_id}")
                continue

            book = orderbooks[token_id]
            mid_price = book.mid_price

            if mid_price is None:
                warnings.append(f"No mid price for {token_id}")
                continue

            # Calculate expected value at snapshot price
            expected_value = pos.net_size * mid_price
            calculated_value = pos.net_size * pos.avg_cost_basis

            # Drift = difference between expected and calculated
            drift_usd = expected_value - calculated_value
            drift_pct = (
                (drift_usd / calculated_value * 100) if calculated_value > 0 else Decimal("0")
            )

            if abs(drift_usd) >= drift_threshold_usd or abs(drift_pct) >= drift_threshold_pct:
                drifts.append(
                    {
                        "token_id": token_id,
                        "market_slug": pos.market_slug,
                        "net_size": float(pos.net_size),
                        "avg_cost_basis": float(pos.avg_cost_basis),
                        "snapshot_mid": float(mid_price),
                        "expected_value": float(expected_value),
                        "calculated_value": float(calculated_value),
                        "drift_usd": float(drift_usd),
                        "drift_pct": float(drift_pct),
                    }
                )
                total_drift += abs(drift_usd)
                max_drift_pct = max(max_drift_pct, abs(drift_pct))

        return ReconciliationResult(
            snapshot_timestamp=snapshot_ts,
            positions_reconciled=len([p for p in positions.values() if p.net_size != 0]),
            positions_with_drift=len(drifts),
            total_drift_usd=total_drift,
            max_drift_pct=max_drift_pct,
            warnings=warnings,
            position_drifts=drifts,
        )

    def generate_pnl_report(
        self,
        since: str | None = None,
        orderbooks: dict[str, OrderBook] | None = None,
        snapshot_path: Path | None = None,
    ) -> PnLReport:
        """Generate comprehensive PnL report.

        Args:
            since: Optional ISO timestamp to filter from
            orderbooks: Optional dict of token_id -> OrderBook
            snapshot_path: Optional path to collector snapshot

        Returns:
            PnLReport with full analysis
        """
        if not self._fills_loaded:
            self.rebuild_state_from_fills()

        # Load orderbooks from snapshot if provided
        if snapshot_path and snapshot_path.exists():
            orderbooks = load_orderbooks_from_snapshot(snapshot_path)

        return self._verifier.compute_pnl(orderbooks=orderbooks, since=since)

    def get_equity_curve_summary(self, since: str | None = None) -> dict:
        """Get summary statistics for equity curve.

        Args:
            since: Optional ISO timestamp to filter from

        Returns:
            Dict with equity curve summary
        """
        curve = self.load_equity_curve(since=since)

        if not curve:
            return {
                "data_points": 0,
                "starting_equity": float(self.starting_cash),
                "current_equity": float(self.starting_cash),
                "total_return": 0.0,
                "total_return_pct": 0.0,
            }

        starting = curve[0].net_equity
        current = curve[-1].net_equity
        total_return = current - starting
        total_return_pct = (total_return / starting * 100) if starting > 0 else Decimal("0")

        # Calculate drawdown
        peak = curve[0].net_equity
        max_drawdown = Decimal("0")
        max_drawdown_pct = Decimal("0")

        for snap in curve:
            if snap.net_equity > peak:
                peak = snap.net_equity
            drawdown = peak - snap.net_equity
            drawdown_pct = (drawdown / peak * 100) if peak > 0 else Decimal("0")
            if drawdown > max_drawdown:
                max_drawdown = drawdown
                max_drawdown_pct = drawdown_pct

        return {
            "data_points": len(curve),
            "first_recorded_at": curve[0].timestamp,
            "last_recorded_at": curve[-1].timestamp,
            "starting_equity": float(starting),
            "current_equity": float(current),
            "total_return": float(total_return),
            "total_return_pct": float(total_return_pct),
            "max_drawdown": float(max_drawdown),
            "max_drawdown_pct": float(max_drawdown_pct),
            "realized_pnl": float(curve[-1].realized_pnl),
            "unrealized_pnl": float(curve[-1].unrealized_pnl),
            "total_fees": float(curve[-1].total_fees),
        }


def run_equity_calculation_against_snapshots(
    data_dir: Path,
    snapshot_dir: Path,
    output_file: Path | None = None,
) -> dict:
    """Run equity calculation against all 15m collector snapshots.

    Processes each snapshot in chronological order, computing equity at each
    point in time using snapshot prices for mark-to-market.

    Args:
        data_dir: Directory with paper trading data (fills.jsonl)
        snapshot_dir: Directory with collector snapshots
        output_file: Optional path to write equity curve JSON

    Returns:
        Dict with summary statistics
    """
    engine = PaperTradingEngine(data_dir=data_dir)

    # Find all 15m snapshots
    snapshot_files = sorted(snapshot_dir.glob("snapshot_15m_*.json"))

    if not snapshot_files:
        return {"error": f"No 15m snapshots found in {snapshot_dir}"}

    # Process each snapshot
    equity_curve = []
    for snap_file in snapshot_files:
        equity = engine.record_equity(snapshot_path=snap_file)
        equity_curve.append(equity.to_dict())

    # Write output if requested
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(json.dumps(equity_curve, indent=2))

    # Return summary
    summary = engine.get_equity_curve_summary()
    summary["snapshots_processed"] = len(snapshot_files)
    summary["snapshot_dir"] = str(snapshot_dir)

    return summary


def generate_pnl_attribution_report(
    engine: PaperTradingEngine,
    since: str | None = None,
) -> dict:
    """Generate PnL attribution by market.

    Args:
        engine: PaperTradingEngine instance
        since: Optional ISO timestamp to filter from

    Returns:
        Dict with PnL attribution by market
    """
    if not engine._fills_loaded:
        engine.rebuild_state_from_fills(since=since)

    positions = engine.get_positions()

    attribution = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "since": since,
        "markets": [],
        "summary": {
            "total_markets": 0,
            "winning_markets": 0,
            "losing_markets": 0,
            "total_realized_pnl": 0.0,
            "total_fees": 0.0,
        },
    }

    total_realized = Decimal("0")
    total_fees = Decimal("0")
    winning = 0
    losing = 0

    for token_id, pos in positions.items():
        if pos.realized_pnl == 0 and pos.net_size == 0:
            continue

        market_data = {
            "token_id": token_id,
            "market_slug": pos.market_slug,
            "realized_pnl": float(pos.realized_pnl),
            "total_fees": float(pos.total_fees),
            "net_pnl": float(pos.realized_pnl - pos.total_fees),
            "fill_count": pos.fill_count,
            "net_size": float(pos.net_size),
        }

        attribution["markets"].append(market_data)

        total_realized += pos.realized_pnl
        total_fees += pos.total_fees

        net_pnl = pos.realized_pnl - pos.total_fees
        if net_pnl > 0:
            winning += 1
        elif net_pnl < 0:
            losing += 1

    attribution["summary"]["total_markets"] = len(attribution["markets"])
    attribution["summary"]["winning_markets"] = winning
    attribution["summary"]["losing_markets"] = losing
    attribution["summary"]["total_realized_pnl"] = float(total_realized)
    attribution["summary"]["total_fees"] = float(total_fees)
    attribution["summary"]["total_net_pnl"] = float(total_realized - total_fees)

    # Sort by net PnL descending
    attribution["markets"].sort(key=lambda x: x["net_pnl"], reverse=True)

    return attribution
