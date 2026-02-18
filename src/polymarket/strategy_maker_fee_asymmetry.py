"""Maker Fee Asymmetry Strategy — Passive Liquidity Provision.

Hypothesis: Polymarket's fee structure creates a structural edge for passive
liquidity providers. Makers pay 0% fees; takers pay 2%. By posting limit orders
at mispriced levels (rather than taking liquidity), we capture the spread AND
avoid the 2% taker fee — turning a marginal trade into +EV.

Core Mechanic:
- Monitor orderbooks for >3% implied edge vs fair probability estimate
- Post passive limit orders (maker) at prices better than current market
- Capture 2% fee savings + spread vs takers
- Requires patience — fills only when price moves to your level

Data Dependencies:
- Real-time orderbook snapshots (existing 15m collector)
- Fair probability estimate: can use cross-platform consensus (Kalshi/Betfair) or proprietary model
- Order book imbalance signal for likely price direction

Entry/Exit Rules:
- Entry: Post limit order when |market_implied_prob - fair_prob| > 3% + spread buffer
- Sizing: 2-5% of bankroll per level, ladder 3 levels deep
- Exit: Let fills come passively; hedge on other platforms if needed
- Stop: Cancel and re-post if fair_prob moves against position by >2%

Success Criteria (4-week paper test):
- Hit rate on passive fills >40%
- Average captured edge per fill >1.5% (after accounting for adverse selection)
- Total fees paid <0.5% of volume (vs 2% for taker strategies)
- Net EV per trade >0.5%
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from .paper_trading import PaperTradingEngine

logger = logging.getLogger(__name__)

# Constants
DEFAULT_DATA_DIR = Path("data/maker_fee_asymmetry")
DEFAULT_EDGE_THRESHOLD = Decimal("0.03")  # 3% edge threshold
DEFAULT_SPREAD_BUFFER = Decimal("0.005")  # 0.5% spread buffer
DEFAULT_FAIR_PROB = Decimal("0.5")  # Default fair probability
DEFAULT_BANKROLL_PCT = Decimal("0.03")  # 3% of bankroll per level
DEFAULT_MAX_LEVELS = 3  # Ladder 3 levels deep
DEFAULT_POSITION_SIZE = Decimal("10")  # $10 per position for paper trading
MAKER_FEE_BPS = 0  # 0% maker fee
TAKER_FEE_BPS = 200  # 2% taker fee (for comparison)


@dataclass(frozen=True)
class OrderBookLevel:
    """Single level of the orderbook."""

    price: Decimal
    size: Decimal


@dataclass(frozen=True)
class OrderBookSide:
    """One side of the orderbook (bids or asks)."""

    levels: list[OrderBookLevel]

    def best_level(self) -> OrderBookLevel | None:
        """Get the best (top of book) level."""
        return self.levels[0] if self.levels else None

    def depth_at_price(self, price: Decimal) -> Decimal:
        """Get total size at or better than price."""
        return sum(level.size for level in self.levels if level.price >= price)


@dataclass(frozen=True)
class MarketOrderBook:
    """Full orderbook for a binary market."""

    market_id: str
    token_id_yes: str
    token_id_no: str
    question: str

    yes_bids: OrderBookSide  # Bids to buy YES
    yes_asks: OrderBookSide  # Asks to sell YES

    @property
    def best_bid_yes(self) -> Decimal | None:
        """Best YES bid price."""
        level = self.yes_bids.best_level()
        return level.price if level else None

    @property
    def best_ask_yes(self) -> Decimal | None:
        """Best YES ask price."""
        level = self.yes_asks.best_level()
        return level.price if level else None

    @property
    def mid_price(self) -> Decimal | None:
        """Mid price (average of best bid and ask)."""
        bid = self.best_bid_yes
        ask = self.best_ask_yes
        if bid is None or ask is None:
            return None
        return (bid + ask) / 2

    @property
    def spread(self) -> Decimal | None:
        """Bid-ask spread."""
        bid = self.best_bid_yes
        ask = self.best_ask_yes
        if bid is None or ask is None:
            return None
        return ask - bid

    def get_implied_probability(self) -> Decimal | None:
        """Market-implied probability from mid price."""
        mid = self.mid_price
        return mid if mid is not None else None


@dataclass(frozen=True)
class FairProbabilityEstimate:
    """Fair probability estimate for a market."""

    market_id: str
    fair_prob: Decimal  # 0-1
    source: str  # e.g., "cross_platform", "model", "default"
    confidence: float  # 0-1
    timestamp: datetime


@dataclass(frozen=True)
class MakerSignal:
    """Trading signal for maker fee asymmetry strategy."""

    timestamp: datetime
    market_id: str
    question: str

    # Fair vs market
    fair_prob: Decimal
    market_implied_prob: Decimal
    edge: Decimal  # |fair - market|

    # Trade direction
    direction: str  # 'BUY_YES' or 'BUY_NO'
    target_price: Decimal  # Limit order price to post

    # Market context
    current_best_bid: Decimal
    current_best_ask: Decimal
    spread: Decimal

    # Sizing
    position_size: Decimal

    # Fee comparison
    maker_fee_savings: Decimal  # 2% saved vs taker

    @property
    def has_edge(self) -> bool:
        """True if edge exceeds threshold."""
        return abs(self.edge) > DEFAULT_EDGE_THRESHOLD

    @property
    def is_passive(self) -> bool:
        """True if target price improves the market."""
        if self.direction == "BUY_YES":
            return self.target_price > self.current_best_bid
        return self.target_price < (Decimal("1") - self.current_best_ask)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "market_id": self.market_id,
            "question": self.question,
            "fair_prob": float(self.fair_prob),
            "market_implied_prob": float(self.market_implied_prob),
            "edge": float(self.edge),
            "direction": self.direction,
            "target_price": float(self.target_price),
            "current_best_bid": float(self.current_best_bid),
            "current_best_ask": float(self.current_best_ask),
            "spread": float(self.spread),
            "position_size": float(self.position_size),
            "maker_fee_savings": float(self.maker_fee_savings),
            "has_edge": self.has_edge,
            "is_passive": self.is_passive,
        }


@dataclass(frozen=True)
class PassiveOrder:
    """A passive limit order posted by the strategy."""

    order_id: str
    timestamp: datetime
    signal: MakerSignal

    # Order details
    side: str  # 'buy_yes' or 'buy_no'
    price: Decimal
    size: Decimal

    # Status
    status: str = "open"  # 'open', 'filled', 'cancelled', 'expired'
    filled_at: datetime | None = None
    fill_price: Decimal | None = None

    # PnL tracking
    realized_pnl: Decimal | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "order_id": self.order_id,
            "timestamp": self.timestamp.isoformat(),
            "market_id": self.signal.market_id,
            "question": self.signal.question,
            "side": self.side,
            "price": float(self.price),
            "size": float(self.size),
            "status": self.status,
            "filled_at": self.filled_at.isoformat() if self.filled_at else None,
            "fill_price": float(self.fill_price) if self.fill_price else None,
            "realized_pnl": float(self.realized_pnl) if self.realized_pnl else None,
            "edge_at_entry": float(self.signal.edge),
            "maker_fee_savings": float(self.signal.maker_fee_savings),
        }


@dataclass
class StrategyPerformance:
    """Performance metrics for the maker fee asymmetry strategy."""

    # Order stats
    total_orders_posted: int = 0
    orders_filled: int = 0
    orders_cancelled: int = 0
    orders_expired: int = 0

    # Fill stats
    total_fills: int = 0
    fills_with_edge: int = 0

    # PnL
    total_realized_pnl: Decimal = field(default_factory=lambda: Decimal("0"))
    total_fees_paid: Decimal = field(default_factory=lambda: Decimal("0"))

    # Volume
    total_volume: Decimal = field(default_factory=lambda: Decimal("0"))

    @property
    def fill_rate(self) -> float:
        """Hit rate on passive fills."""
        if self.total_orders_posted == 0:
            return 0.0
        return (self.orders_filled / self.total_orders_posted) * 100

    @property
    def avg_pnl_per_fill(self) -> Decimal:
        """Average PnL per filled order."""
        if self.orders_filled == 0:
            return Decimal("0")
        return self.total_realized_pnl / self.orders_filled

    @property
    def fee_rate(self) -> float:
        """Total fees as % of volume."""
        if self.total_volume == 0:
            return 0.0
        return float(self.total_fees_paid / self.total_volume) * 100

    @property
    def ev_per_trade(self) -> Decimal:
        """Expected value per trade."""
        if self.total_fills == 0:
            return Decimal("0")
        return self.total_realized_pnl / self.total_fills

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_orders_posted": self.total_orders_posted,
            "orders_filled": self.orders_filled,
            "orders_cancelled": self.orders_cancelled,
            "orders_expired": self.orders_expired,
            "total_fills": self.total_fills,
            "fills_with_edge": self.fills_with_edge,
            "fill_rate": self.fill_rate,
            "total_realized_pnl": float(self.total_realized_pnl),
            "total_fees_paid": float(self.total_fees_paid),
            "total_volume": float(self.total_volume),
            "avg_pnl_per_fill": float(self.avg_pnl_per_fill),
            "fee_rate": self.fee_rate,
            "ev_per_trade": float(self.ev_per_trade),
        }


def _to_decimal(val: str | float | int) -> Decimal:
    """Convert a value to Decimal."""
    return Decimal(str(val))


def parse_orderbook_from_snapshot(
    market_data: dict[str, Any],
    timestamp: datetime | None = None,
) -> MarketOrderBook | None:
    """Parse orderbook from a market snapshot.

    Args:
        market_data: Market dict with 'books' key containing 'yes' and 'no' books
        timestamp: Override timestamp

    Returns:
        MarketOrderBook or None if data is insufficient
    """
    market_id = str(market_data.get("market_id", market_data.get("id", "")))
    question = market_data.get("question", market_data.get("title", ""))

    token_ids = market_data.get("clob_token_ids", market_data.get("clobTokenIds", []))
    if len(token_ids) != 2:
        return None

    books = market_data.get("books", {})
    yes_book = books.get("yes", {})

    yes_bids_raw = yes_book.get("bids", [])
    yes_asks_raw = yes_book.get("asks", [])

    if not yes_bids_raw or not yes_asks_raw:
        return None

    # Parse levels
    yes_bids = OrderBookSide(
        levels=[
            OrderBookLevel(price=_to_decimal(b["price"]), size=_to_decimal(b["size"]))
            for b in yes_bids_raw
        ]
    )
    yes_asks = OrderBookSide(
        levels=[
            OrderBookLevel(price=_to_decimal(a["price"]), size=_to_decimal(a["size"]))
            for a in yes_asks_raw
        ]
    )

    return MarketOrderBook(
        market_id=market_id,
        token_id_yes=str(token_ids[0]),
        token_id_no=str(token_ids[1]),
        question=question,
        yes_bids=yes_bids,
        yes_asks=yes_asks,
    )


def compute_fair_probability(
    orderbook: MarketOrderBook,
    external_estimate: Decimal | None = None,
) -> FairProbabilityEstimate:
    """Compute fair probability estimate for a market.

    For now, uses simple midpoint. Can be extended to:
    - Cross-platform consensus (Kalshi/Betfair)
    - Proprietary model
    - Orderbook imbalance adjustment

    Args:
        orderbook: Market orderbook
        external_estimate: Optional external probability estimate

    Returns:
        FairProbabilityEstimate
    """
    now = datetime.now(UTC)

    if external_estimate is not None:
        return FairProbabilityEstimate(
            market_id=orderbook.market_id,
            fair_prob=external_estimate,
            source="external",
            confidence=0.8,
            timestamp=now,
        )

    # Default: use mid price as fair probability
    # This assumes market is roughly efficient on average
    mid = orderbook.mid_price
    if mid is not None:
        return FairProbabilityEstimate(
            market_id=orderbook.market_id,
            fair_prob=mid,
            source="midpoint",
            confidence=0.5,
            timestamp=now,
        )

    # Fallback to 0.5
    return FairProbabilityEstimate(
        market_id=orderbook.market_id,
        fair_prob=DEFAULT_FAIR_PROB,
        source="default",
        confidence=0.3,
        timestamp=now,
    )


def generate_maker_signal(
    orderbook: MarketOrderBook,
    fair_estimate: FairProbabilityEstimate,
    edge_threshold: Decimal = DEFAULT_EDGE_THRESHOLD,
    spread_buffer: Decimal = DEFAULT_SPREAD_BUFFER,
    position_size: Decimal = DEFAULT_POSITION_SIZE,
) -> MakerSignal | None:
    """Generate a maker signal if edge exceeds threshold.

    Args:
        orderbook: Market orderbook
        fair_estimate: Fair probability estimate
        edge_threshold: Minimum edge to trigger (absolute)
        spread_buffer: Additional buffer beyond spread
        position_size: Position size in USD

    Returns:
        MakerSignal or None if no edge
    """
    market_prob = orderbook.get_implied_probability()
    if market_prob is None:
        return None

    spread = orderbook.spread
    if spread is None:
        return None

    fair_prob = fair_estimate.fair_prob
    edge = fair_prob - market_prob

    # Total required edge: threshold + spread buffer
    required_edge = edge_threshold + spread_buffer

    now = datetime.now(UTC)
    best_bid = orderbook.best_bid_yes or Decimal("0")
    best_ask = orderbook.best_ask_yes or Decimal("1")

    # Determine direction and target price
    direction = None
    target_price = None

    if edge > required_edge:
        # Fair > Market: Buy YES (underpriced)
        direction = "BUY_YES"
        # Post at market best bid + small improvement
        target_price = min(best_bid + Decimal("0.001"), fair_prob - spread_buffer)
    elif edge < -required_edge:
        # Fair < Market: Buy NO (overpriced YES)
        direction = "BUY_NO"
        # NO price = 1 - YES price
        no_fair = Decimal("1") - fair_prob
        no_best_ask = Decimal("1") - best_bid  # Ask to buy NO is bid for YES
        target_price = min(no_best_ask + Decimal("0.001"), no_fair - spread_buffer)

    if direction is None:
        return None

    return MakerSignal(
        timestamp=now,
        market_id=orderbook.market_id,
        question=orderbook.question,
        fair_prob=fair_prob,
        market_implied_prob=market_prob,
        edge=edge,
        direction=direction,
        target_price=target_price,
        current_best_bid=best_bid,
        current_best_ask=best_ask,
        spread=spread,
        position_size=position_size,
        maker_fee_savings=Decimal(str(TAKER_FEE_BPS)) / 10000,  # 2% savings
    )


def post_passive_order(
    signal: MakerSignal,
    orderbook: MarketOrderBook,
    engine: PaperTradingEngine,
) -> PassiveOrder | None:
    """Post a passive limit order based on a signal.

    Args:
        signal: MakerSignal with trade details
        orderbook: Market orderbook for token IDs
        engine: Paper trading engine

    Returns:
        PassiveOrder if posted, None otherwise
    """
    now = datetime.now(UTC)

    # Determine token and side
    if signal.direction == "BUY_YES":
        token_id = orderbook.token_id_yes
        side = "buy_yes"
        price = signal.target_price
    else:
        token_id = orderbook.token_id_no
        side = "buy_no"
        price = signal.target_price

    # Calculate number of shares
    num_shares = signal.position_size / price

    # Record as a "limit order" via fill at target price
    # In paper trading, we record the intent; actual fill happens later
    engine.record_fill(
        token_id=token_id,
        side="buy",
        size=num_shares,
        price=price,
        fee=Decimal("0"),  # Maker pays 0 fees
        timestamp=now.isoformat(),
        market_slug=signal.market_id,
        market_question=signal.question,
    )

    order_id = f"maker_{now.strftime('%Y%m%d%H%M%S')}_{signal.market_id[:8]}"

    return PassiveOrder(
        order_id=order_id,
        timestamp=now,
        signal=signal,
        side=side,
        price=price,
        size=num_shares,
        status="open",
    )


class MakerFeeAsymmetryTracker:
    """Tracks passive orders and performance for the strategy."""

    data_dir: Path
    paper_engine: PaperTradingEngine

    def __init__(self, data_dir: Path | str | None = None) -> None:
        """Initialize tracker.

        Args:
            data_dir: Directory for storing data
        """
        self.data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.paper_engine = PaperTradingEngine(
            data_dir=self.data_dir / "paper_trading",
            starting_cash=Decimal("10000"),
        )

    def get_orders_file(self) -> Path:
        """Get path to orders file."""
        return self.data_dir / "orders.jsonl"

    def get_signals_file(self) -> Path:
        """Get path to signals file."""
        return self.data_dir / "signals.jsonl"

    def load_orders(self) -> list[PassiveOrder]:
        """Load all recorded orders."""
        orders: list[PassiveOrder] = []
        orders_file = self.get_orders_file()

        if not orders_file.exists():
            return orders

        with open(orders_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    order = self._order_from_dict(data)
                    if order:
                        orders.append(order)
                except (json.JSONDecodeError, KeyError):
                    continue

        return orders

    def _order_from_dict(self, data: dict[str, Any]) -> PassiveOrder | None:
        """Reconstruct PassiveOrder from dict."""
        try:
            return PassiveOrder(
                order_id=data["order_id"],
                timestamp=datetime.fromisoformat(data["timestamp"]),
                signal=MakerSignal(  # Simplified reconstruction
                    timestamp=datetime.fromisoformat(data["timestamp"]),
                    market_id=data["market_id"],
                    question=data["question"],
                    fair_prob=Decimal("0.5"),
                    market_implied_prob=Decimal("0.5"),
                    edge=Decimal(str(data.get("edge_at_entry", 0))),
                    direction="BUY_YES" if data["side"] == "buy_yes" else "BUY_NO",
                    target_price=Decimal(str(data["price"])),
                    current_best_bid=Decimal("0"),
                    current_best_ask=Decimal("1"),
                    spread=Decimal("0.01"),
                    position_size=Decimal(str(data["size"])),
                    maker_fee_savings=Decimal(str(data.get("maker_fee_savings", 0.02))),
                ),
                side=data["side"],
                price=Decimal(str(data["price"])),
                size=Decimal(str(data["size"])),
                status=data.get("status", "open"),
                filled_at=datetime.fromisoformat(data["filled_at"])
                if data.get("filled_at")
                else None,
                fill_price=Decimal(str(data["fill_price"])) if data.get("fill_price") else None,
                realized_pnl=Decimal(str(data["realized_pnl"]))
                if data.get("realized_pnl")
                else None,
            )
        except (KeyError, ValueError):
            return None

    def record_order(self, order: PassiveOrder) -> None:
        """Record an order to file."""
        orders_file = self.get_orders_file()
        with open(orders_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(order.to_dict(), sort_keys=True) + "\n")

    def record_signal(self, signal: MakerSignal) -> None:
        """Record a signal to file."""
        signals_file = self.get_signals_file()
        with open(signals_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(signal.to_dict(), sort_keys=True) + "\n")

    def get_performance(self) -> StrategyPerformance:
        """Calculate strategy performance metrics."""
        orders = self.load_orders()
        perf = StrategyPerformance()

        for order in orders:
            perf.total_orders_posted += 1

            if order.status == "filled":
                perf.orders_filled += 1
                perf.total_fills += 1
                if order.signal.has_edge:
                    perf.fills_with_edge += 1
                if order.realized_pnl is not None:
                    perf.total_realized_pnl += order.realized_pnl
            elif order.status == "cancelled":
                perf.orders_cancelled += 1
            elif order.status == "expired":
                perf.orders_expired += 1

            perf.total_volume += order.price * order.size
            # Maker pays 0 fees

        return perf

    def get_open_orders(self) -> list[PassiveOrder]:
        """Get all open orders."""
        return [o for o in self.load_orders() if o.status == "open"]


def scan_for_maker_opportunities(
    snapshot_path: Path,
    target_market_substring: str = "bitcoin",
    edge_threshold: Decimal = DEFAULT_EDGE_THRESHOLD,
    spread_buffer: Decimal = DEFAULT_SPREAD_BUFFER,
    position_size: Decimal = DEFAULT_POSITION_SIZE,
) -> list[MakerSignal]:
    """Scan a snapshot for maker fee asymmetry opportunities.

    Args:
        snapshot_path: Path to market snapshot
        target_market_substring: Filter markets by this substring
        edge_threshold: Minimum edge to trigger
        spread_buffer: Additional buffer beyond spread
        position_size: Position size in USD

    Returns:
        List of MakerSignal opportunities
    """
    signals: list[MakerSignal] = []

    try:
        data = json.loads(snapshot_path.read_text())
    except (json.JSONDecodeError, FileNotFoundError) as e:
        logger.error("Error loading snapshot: %s", e)
        return signals

    for market_data in data.get("markets", []):
        question = market_data.get("question", market_data.get("title", ""))
        if target_market_substring.lower() not in question.lower():
            continue

        orderbook = parse_orderbook_from_snapshot(market_data)
        if orderbook is None:
            continue

        fair_estimate = compute_fair_probability(orderbook)
        signal = generate_maker_signal(
            orderbook,
            fair_estimate,
            edge_threshold=edge_threshold,
            spread_buffer=spread_buffer,
            position_size=position_size,
        )

        if signal:
            signals.append(signal)

    # Sort by edge magnitude descending
    signals.sort(key=lambda s: abs(s.edge), reverse=True)

    return signals


def run_maker_fee_asymmetry_scan(
    snapshots_dir: Path,
    data_dir: Path | None = None,
    dry_run: bool = True,
    max_positions: int = 10,
    target_market_substring: str = "bitcoin",
    edge_threshold: Decimal = DEFAULT_EDGE_THRESHOLD,
) -> dict[str, Any]:
    """Run a complete maker fee asymmetry scan.

    Args:
        snapshots_dir: Directory with market snapshots
        data_dir: Directory for storing data
        dry_run: If True, don't execute trades
        max_positions: Maximum number of positions to take
        target_market_substring: Filter for target markets
        edge_threshold: Minimum edge to trigger

    Returns:
        Dictionary with scan results
    """
    now = datetime.now(UTC)

    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR

    tracker = MakerFeeAsymmetryTracker(data_dir=data_dir)

    logger.info("Starting maker fee asymmetry scan at %s", now.isoformat())

    # Find latest snapshot
    snapshot_files = sorted(snapshots_dir.glob("snapshot_*.json"))
    if not snapshot_files:
        return {
            "timestamp": now.isoformat(),
            "error": f"No snapshots found in {snapshots_dir}",
        }

    latest_snapshot = snapshot_files[-1]
    logger.info("Using snapshot: %s", latest_snapshot)

    # Scan for opportunities
    signals = scan_for_maker_opportunities(
        latest_snapshot,
        target_market_substring=target_market_substring,
        edge_threshold=edge_threshold,
    )

    logger.info("Found %d signals with edge > %s", len(signals), edge_threshold)

    # Execute paper trades
    orders: list[PassiveOrder] = []

    if not dry_run:
        for signal in signals[:max_positions]:
            # Parse orderbook again for token IDs
            data = json.loads(latest_snapshot.read_text())
            for market_data in data.get("markets", []):
                if market_data.get("market_id") == signal.market_id:
                    orderbook = parse_orderbook_from_snapshot(market_data)
                    if orderbook:
                        order = post_passive_order(signal, orderbook, tracker.paper_engine)
                        if order:
                            tracker.record_order(order)
                            tracker.record_signal(signal)
                            orders.append(order)
                            logger.info(
                                "Posted %s order: %s @ %.3f (edge: %.2f%%)",
                                order.side,
                                signal.question[:50],
                                order.price,
                                float(signal.edge) * 100,
                            )
                    break

    # Get performance summary
    performance = tracker.get_performance()

    return {
        "timestamp": now.isoformat(),
        "snapshot": str(latest_snapshot),
        "signals_found": len(signals),
        "orders_posted": len(orders),
        "dry_run": dry_run,
        "edge_threshold": float(edge_threshold),
        "performance": performance.to_dict(),
        "top_signals": [s.to_dict() for s in signals[:10]],
    }


def load_snapshots_for_backtest(
    data_dir: Path,
    interval: str = "15m",
    start_time: datetime | None = None,
    end_time: datetime | None = None,
) -> list[Path]:
    """Load snapshot paths for backtesting, filtered by time range.

    Args:
        data_dir: Directory containing snapshot files
        interval: '5m' or '15m'
        start_time: Optional start filter
        end_time: Optional end filter

    Returns:
        List of snapshot paths sorted by time
    """
    pattern = f"snapshot_{interval}_*.json"
    snapshots = sorted(data_dir.glob(pattern))

    filtered = []
    for snap in snapshots:
        # Parse timestamp from filename
        try:
            # Format: snapshot_15m_20260215T040615Z.json
            ts_str = snap.stem.split("_")[2]
            snap_ts = datetime.strptime(ts_str, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)

            if start_time and snap_ts < start_time:
                continue
            if end_time and snap_ts > end_time:
                continue

            filtered.append(snap)
        except (IndexError, ValueError):
            # Can't parse timestamp, include anyway
            filtered.append(snap)

    return filtered


def run_backtest(
    snapshots: list[Path],
    edge_threshold: Decimal = DEFAULT_EDGE_THRESHOLD,
    spread_buffer: Decimal = DEFAULT_SPREAD_BUFFER,
    position_size: Decimal = DEFAULT_POSITION_SIZE,
    target_market_substring: str = "bitcoin",
    hold_horizon: int = 4,  # Number of snapshots to hold
) -> dict[str, Any]:
    """Run backtest on historical snapshots.

    Args:
        snapshots: List of snapshot file paths (sorted by time)
        edge_threshold: Minimum edge to trigger
        spread_buffer: Additional buffer beyond spread
        position_size: Position size in USD
        target_market_substring: Filter for target markets
        hold_horizon: Number of snapshots to simulate hold period

    Returns:
        Dictionary with backtest results
    """
    trades: list[dict[str, Any]] = []

    for i, snap_path in enumerate(snapshots[:-hold_horizon]):
        signals = scan_for_maker_opportunities(
            snap_path,
            target_market_substring=target_market_substring,
            edge_threshold=edge_threshold,
            spread_buffer=spread_buffer,
            position_size=position_size,
        )

        for signal in signals[:5]:  # Top 5 signals per snapshot
            # Find exit snapshot
            exit_snap = snapshots[i + hold_horizon]

            # Get exit price
            exit_price = None
            try:
                data = json.loads(exit_snap.read_text())
                for market_data in data.get("markets", []):
                    if market_data.get("market_id") == signal.market_id:
                        orderbook = parse_orderbook_from_snapshot(market_data)
                        if orderbook:
                            exit_price = orderbook.mid_price
                        break
            except (json.JSONDecodeError, FileNotFoundError):
                pass

            if exit_price is not None:
                if signal.direction == "BUY_YES":
                    pnl = exit_price - signal.target_price
                else:
                    pnl = signal.target_price - (Decimal("1") - exit_price)

                # Add maker fee savings
                pnl += Decimal("0.02")  # 2% fee savings

                trades.append({
                    "entry_time": signal.timestamp.isoformat(),
                    "market_id": signal.market_id,
                    "direction": signal.direction,
                    "entry_price": float(signal.target_price),
                    "exit_price": float(exit_price),
                    "edge_at_entry": float(signal.edge),
                    "pnl": float(pnl),
                    "pnl_pct": float(pnl / signal.target_price * 100),
                })

    # Compute metrics
    if trades:
        total_pnl = sum(t["pnl"] for t in trades)
        avg_pnl = total_pnl / len(trades)
        winning_trades = [t for t in trades if t["pnl"] > 0]
        win_rate = len(winning_trades) / len(trades) * 100
        avg_edge = sum(t["edge_at_entry"] for t in trades) / len(trades)
    else:
        total_pnl = avg_pnl = win_rate = avg_edge = 0

    return {
        "total_trades": len(trades),
        "winning_trades": len(winning_trades) if trades else 0,
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "avg_pnl_per_trade": avg_pnl,
        "avg_edge_at_entry": avg_edge,
        "trades": trades,
        "params": {
            "edge_threshold": float(edge_threshold),
            "spread_buffer": float(spread_buffer),
            "position_size": float(position_size),
            "hold_horizon": hold_horizon,
        },
    }
