"""Discounted Outcome Arbitrage with Insider Confirmation strategy.

Hypothesis: Markets that price outcomes below 35¢ (extreme discount) combined
with informed wallet positioning yield positive EV across all verticals.

Mechanics:
1. Scan for markets where either YES or NO < $0.35 (65%+ implied discount)
2. Cross-reference with insider wallet positioning: wallets that historically
   enter early, size consistently, exit before peak
3. Only enter when both conditions align: deep discount + informed flow confirmation
4. Hold to resolution (no mid-market exit)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

from .copytrade_profiler import CopytradeProfiler, TraderPerformanceMetrics
from .paper_trading import PaperTradingEngine
from .trader_fills import TraderFillTracker

logger = logging.getLogger(__name__)

# Constants
DEFAULT_DATA_DIR = Path("data/discounted_arbitrage")
DISCOUNT_THRESHOLD = Decimal("0.35")  # $0.35 = 35 cents
INSIDER_MIN_PNL = Decimal("1000")  # Minimum $1000 realized PnL
INSIDER_MIN_WIN_RATE = 55.0  # Minimum 55% win rate
INSIDER_MIN_TRADES = 10  # Minimum trades for statistical significance
POSITION_SIZE = Decimal("10")  # $10 per position (micro-betting)
MIN_WALLET_POSITION_USD = 1000  # Minimum $1000 position to consider confirmation

# Vertical categories
VERTICALS = {
    "crypto": ["bitcoin", "btc", "ethereum", "eth", "solana", "sol", "crypto"],
    "politics": ["election", "trump", "biden", "vote", "poll", "senate", "house", "governor"],
    "sports": ["game", "match", "score", "win", "championship", "nba", "nfl", "mlb", "nhl"],
    "weather": ["temperature", "degrees", "rain", "snow", "forecast", "high temp", "low temp"],
}


@dataclass(frozen=True)
class DiscountedMarket:
    """A market with deeply discounted outcome pricing."""

    market_id: str
    token_id_yes: str
    token_id_no: str
    question: str
    vertical: str

    # Pricing
    yes_price: Decimal
    no_price: Decimal
    discounted_side: str  # 'YES' or 'NO'
    discounted_price: Decimal

    # Market metadata
    end_date: datetime | None
    volume_usd: Decimal | None

    @property
    def implied_probability(self) -> Decimal:
        """Market-implied probability of discounted outcome."""
        return self.discounted_price

    @property
    def discount_to_fair(self) -> Decimal:
        """Discount to fair value (assuming fair = 0.5 for binary)."""
        return Decimal("0.5") - self.discounted_price


@dataclass(frozen=True)
class InsiderPosition:
    """Position held by an insider wallet."""

    wallet_address: str
    token_id: str
    market_id: str
    side: str  # 'YES' or 'NO'
    position_size_usd: Decimal
    entry_price: Decimal | None
    entry_time: datetime | None

    # Insider metrics
    wallet_pnl: Decimal
    wallet_win_rate: float
    wallet_total_trades: int


@dataclass(frozen=True)
class DiscountedArbitrageSignal:
    """Trading signal combining discount + insider confirmation."""

    timestamp: datetime
    market: DiscountedMarket
    insider_confirmations: list[InsiderPosition]

    # Signal strength
    confirmation_count: int
    total_insider_position_usd: Decimal
    avg_insider_win_rate: float

    @property
    def has_insider_confirmation(self) -> bool:
        """True if at least one insider confirms the trade."""
        return len(self.insider_confirmations) > 0

    @property
    def confidence_score(self) -> float:
        """Confidence score 0-1 based on insider quality and position size."""
        if not self.insider_confirmations:
            return 0.0

        # Base confidence from number of insiders
        base_conf = min(0.5, len(self.insider_confirmations) * 0.15)

        # Bonus from average win rate
        wr_bonus = (self.avg_insider_win_rate - 50) / 100 if self.avg_insider_win_rate > 50 else 0

        # Bonus from position size (capped)
        size_usd = float(self.total_insider_position_usd)
        size_bonus = min(0.2, size_usd / 10000)  # $10k = max bonus

        return min(1.0, base_conf + wr_bonus + size_bonus)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "market": {
                "market_id": self.market.market_id,
                "question": self.market.question,
                "vertical": self.market.vertical,
                "discounted_side": self.market.discounted_side,
                "discounted_price": float(self.market.discounted_price),
                "implied_probability": float(self.market.implied_probability),
                "discount_to_fair": float(self.market.discount_to_fair),
            },
            "insider_confirmations": [
                {
                    "wallet": c.wallet_address,
                    "side": c.side,
                    "position_size_usd": float(c.position_size_usd),
                    "wallet_pnl": float(c.wallet_pnl),
                    "wallet_win_rate": c.wallet_win_rate,
                }
                for c in self.insider_confirmations
            ],
            "confirmation_count": self.confirmation_count,
            "total_insider_position_usd": float(self.total_insider_position_usd),
            "avg_insider_win_rate": self.avg_insider_win_rate,
            "confidence_score": self.confidence_score,
            "has_insider_confirmation": self.has_insider_confirmation,
        }


@dataclass(frozen=True)
class PaperTrade:
    """A paper trade executed by the strategy."""

    trade_id: str
    timestamp: datetime
    signal: DiscountedArbitrageSignal
    side: str  # 'buy_yes' or 'buy_no'
    position_size: Decimal
    entry_price: Decimal
    expected_value: Decimal

    # Resolution tracking
    resolved: bool = False
    resolved_at: datetime | None = None
    winning_outcome: bool | None = None  # True if discounted side won
    pnl: Decimal | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "trade_id": self.trade_id,
            "timestamp": self.timestamp.isoformat(),
            "market_id": self.signal.market.market_id,
            "question": self.signal.market.question,
            "vertical": self.signal.market.vertical,
            "side": self.side,
            "discounted_side": self.signal.market.discounted_side,
            "entry_price": float(self.entry_price),
            "position_size": float(self.position_size),
            "expected_value": float(self.expected_value),
            "confidence_score": self.signal.confidence_score,
            "insider_confirmation": self.signal.has_insider_confirmation,
            "resolved": self.resolved,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "winning_outcome": self.winning_outcome,
            "pnl": float(self.pnl) if self.pnl else None,
        }


@dataclass
class StrategyPerformance:
    """Performance metrics for the strategy."""

    total_trades: int = 0
    insider_confirmed_trades: int = 0
    non_insider_trades: int = 0

    winning_trades: int = 0
    losing_trades: int = 0

    # With insider confirmation
    insider_winning: int = 0
    insider_losing: int = 0

    # Without insider confirmation
    non_insider_winning: int = 0
    non_insider_losing: int = 0

    total_pnl: Decimal = field(default_factory=lambda: Decimal("0"))
    insider_pnl: Decimal = field(default_factory=lambda: Decimal("0"))
    non_insider_pnl: Decimal = field(default_factory=lambda: Decimal("0"))

    total_hold_time_hours: float = 0.0
    resolved_trades: int = 0

    @property
    def overall_win_rate(self) -> float:
        """Overall win rate across all trades."""
        if self.resolved_trades == 0:
            return 0.0
        return (self.winning_trades / self.resolved_trades) * 100

    @property
    def insider_win_rate(self) -> float:
        """Win rate when insider confirmation applied."""
        total = self.insider_winning + self.insider_losing
        if total == 0:
            return 0.0
        return (self.insider_winning / total) * 100

    @property
    def non_insider_win_rate(self) -> float:
        """Win rate without insider confirmation."""
        total = self.non_insider_winning + self.non_insider_losing
        if total == 0:
            return 0.0
        return (self.non_insider_winning / total) * 100

    @property
    def avg_hold_time_hours(self) -> float:
        """Average hold time for resolved trades."""
        if self.resolved_trades == 0:
            return 0.0
        return self.total_hold_time_hours / self.resolved_trades

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_trades": self.total_trades,
            "insider_confirmed_trades": self.insider_confirmed_trades,
            "non_insider_trades": self.non_insider_trades,
            "resolved_trades": self.resolved_trades,
            "overall_win_rate": self.overall_win_rate,
            "insider_win_rate": self.insider_win_rate,
            "non_insider_win_rate": self.non_insider_win_rate,
            "total_pnl": float(self.total_pnl),
            "insider_pnl": float(self.insider_pnl),
            "non_insider_pnl": float(self.non_insider_pnl),
            "avg_hold_time_hours": self.avg_hold_time_hours,
        }


def detect_vertical(question: str) -> str:
    """Detect market vertical from question text."""
    question_lower = question.lower()

    for vertical, keywords in VERTICALS.items():
        for keyword in keywords:
            if keyword in question_lower:
                return vertical

    return "other"


def find_discounted_markets(
    snapshot_path: Path,
    discount_threshold: Decimal = DISCOUNT_THRESHOLD,
) -> list[DiscountedMarket]:
    """Find markets with deeply discounted outcomes.

    Args:
        snapshot_path: Path to market snapshot
        discount_threshold: Price threshold for discount (default $0.35)

    Returns:
        List of DiscountedMarket objects
    """
    markets: list[DiscountedMarket] = []

    try:
        data = json.loads(snapshot_path.read_text())
    except (json.JSONDecodeError, FileNotFoundError) as e:
        logger.error("Error loading snapshot: %s", e)
        return markets

    for m in data.get("markets", []):
        question = m.get("question", m.get("title", ""))
        token_ids = m.get("clob_token_ids", m.get("clobTokenIds", []))

        if len(token_ids) != 2:
            continue

        books = m.get("books", {})
        yes_book = books.get("yes", {})
        no_book = books.get("no", {})

        yes_bids = yes_book.get("bids", [])
        yes_asks = yes_book.get("asks", [])
        no_bids = no_book.get("bids", [])
        no_asks = no_book.get("asks", [])

        if not yes_bids or not yes_asks or not no_bids or not no_asks:
            continue

        # Calculate mid prices
        yes_bid = max(float(b["price"]) for b in yes_bids)
        yes_ask = min(float(a["price"]) for a in yes_asks)
        no_bid = max(float(b["price"]) for b in no_bids)
        no_ask = min(float(a["price"]) for a in no_asks)

        yes_mid = Decimal(str((yes_bid + yes_ask) / 2))
        no_mid = Decimal(str((no_bid + no_ask) / 2))

        # Check for discount on either side
        discounted_side = None
        discounted_price = None

        if yes_mid < discount_threshold:
            discounted_side = "YES"
            discounted_price = yes_mid
        elif no_mid < discount_threshold:
            discounted_side = "NO"
            discounted_price = no_mid

        if discounted_side is None:
            continue

        # Parse end date
        end_date = None
        end_date_str = m.get("end_date", m.get("endDate", ""))
        if end_date_str:
            try:
                end_date = datetime.fromisoformat(str(end_date_str).replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass

        # Parse volume
        volume = None
        volume_usd = m.get("volume_usd", m.get("volume", m.get("volumeNum", 0)))
        if volume_usd:
            try:
                volume = Decimal(str(volume_usd))
            except (ValueError, TypeError):
                pass

        market = DiscountedMarket(
            market_id=str(m.get("market_id", m.get("id", ""))),
            token_id_yes=str(token_ids[0]),
            token_id_no=str(token_ids[1]),
            question=question,
            vertical=detect_vertical(question),
            yes_price=yes_mid,
            no_price=no_mid,
            discounted_side=discounted_side,
            discounted_price=discounted_price,
            end_date=end_date,
            volume_usd=volume,
        )

        markets.append(market)

    logger.info("Found %d discounted markets in snapshot", len(markets))
    return markets


def load_insider_wallets(data_dir: Path) -> list[str]:
    """Load list of insider wallet addresses to track.

    Args:
        data_dir: Directory with wallet data

    Returns:
        List of wallet addresses
    """
    # Try to load from profiler rankings
    profiler_dir = data_dir / "copytrade_profiles"
    rankings_file = profiler_dir / "rankings.json"

    if rankings_file.exists():
        try:
            data = json.loads(rankings_file.read_text())
            rankings = data.get("rankings", [])
            # Return top 20 wallets
            return [r["address"] for r in rankings[:20]]
        except (json.JSONDecodeError, KeyError):
            pass

    return []


def qualify_insider(
    metrics: TraderPerformanceMetrics | None,
    min_pnl: Decimal = INSIDER_MIN_PNL,
    min_win_rate: float = INSIDER_MIN_WIN_RATE,
    min_trades: int = INSIDER_MIN_TRADES,
) -> bool:
    """Check if a wallet qualifies as an insider.

    Args:
        metrics: Trader performance metrics
        min_pnl: Minimum realized PnL
        min_win_rate: Minimum win rate percentage
        min_trades: Minimum number of trades

    Returns:
        True if wallet qualifies as insider
    """
    if metrics is None:
        return False

    if metrics.total_trades < min_trades:
        return False

    if metrics.win_rate < min_win_rate:
        return False

    if metrics.realized_pnl < min_pnl:
        return False

    return True


def get_wallet_positions(
    wallet_address: str,
    fill_tracker: TraderFillTracker,
) -> dict[str, Decimal]:
    """Get current positions for a wallet.

    Args:
        wallet_address: Wallet address
        fill_tracker: Fill tracker instance

    Returns:
        Dict of token_id -> position size
    """
    fills = fill_tracker.load_fills(wallet_address)

    # Group by token and calculate net position
    positions: dict[str, Decimal] = {}

    for fill in fills:
        token_id = fill.token_id
        if token_id not in positions:
            positions[token_id] = Decimal("0")

        if fill.side == "buy":
            positions[token_id] += fill.size
        else:
            positions[token_id] -= fill.size

    return positions


def check_insider_confirmations(
    market: DiscountedMarket,
    insider_wallets: list[str],
    profiler: CopytradeProfiler,
    fill_tracker: TraderFillTracker,
    min_position_usd: int = MIN_WALLET_POSITION_USD,
) -> list[InsiderPosition]:
    """Check which insiders have positions in the discounted direction.

    Args:
        market: Discounted market to check
        insider_wallets: List of insider wallet addresses
        profiler: Copytrade profiler for metrics
        fill_tracker: Fill tracker for positions
        min_position_usd: Minimum position size to consider

    Returns:
        List of InsiderPosition confirmations
    """
    confirmations: list[InsiderPosition] = []

    for wallet in insider_wallets:
        metrics = profiler.compute_metrics(wallet)

        if not qualify_insider(metrics):
            continue

        positions = get_wallet_positions(wallet, fill_tracker)

        # Check position in the discounted token
        token_id = (
            market.token_id_yes
            if market.discounted_side == "YES"
            else market.token_id_no
        )

        position_size = positions.get(token_id, Decimal("0"))

        if position_size <= 0:
            continue

        # Calculate position value
        entry_price = market.discounted_price
        position_value = position_size * entry_price

        if position_value < min_position_usd:
            continue

        confirmation = InsiderPosition(
            wallet_address=wallet,
            token_id=token_id,
            market_id=market.market_id,
            side=market.discounted_side,
            position_size_usd=position_value,
            entry_price=entry_price,
            entry_time=None,  # Could parse from fills
            wallet_pnl=metrics.realized_pnl,
            wallet_win_rate=metrics.win_rate,
            wallet_total_trades=metrics.total_trades,
        )

        confirmations.append(confirmation)

    return confirmations


def generate_signals(
    markets: list[DiscountedMarket],
    insider_wallets: list[str],
    profiler: CopytradeProfiler,
    fill_tracker: TraderFillTracker,
) -> list[DiscountedArbitrageSignal]:
    """Generate trading signals with insider confirmation.

    Args:
        markets: List of discounted markets
        insider_wallets: List of insider wallet addresses
        profiler: Copytrade profiler
        fill_tracker: Fill tracker

    Returns:
        List of trading signals
    """
    signals: list[DiscountedArbitrageSignal] = []
    now = datetime.now(UTC)

    for market in markets:
        confirmations = check_insider_confirmations(
            market, insider_wallets, profiler, fill_tracker
        )

        total_position = sum(c.position_size_usd for c in confirmations)
        avg_win_rate = (
            sum(c.wallet_win_rate for c in confirmations) / len(confirmations)
            if confirmations
            else 0.0
        )

        signal = DiscountedArbitrageSignal(
            timestamp=now,
            market=market,
            insider_confirmations=confirmations,
            confirmation_count=len(confirmations),
            total_insider_position_usd=total_position,
            avg_insider_win_rate=avg_win_rate,
        )

        signals.append(signal)

    # Sort by confidence score descending
    signals.sort(key=lambda s: s.confidence_score, reverse=True)

    return signals


def execute_paper_trade(
    signal: DiscountedArbitrageSignal,
    engine: PaperTradingEngine,
    position_size: Decimal = POSITION_SIZE,
) -> PaperTrade | None:
    """Execute a paper trade based on a signal.

    Args:
        signal: DiscountedArbitrageSignal with trade details
        engine: Paper trading engine
        position_size: Position size in USD

    Returns:
        PaperTrade if executed, None otherwise
    """
    now = datetime.now(UTC)

    # Determine which side to buy
    if signal.market.discounted_side == "YES":
        side = "buy_yes"
        token_id = signal.market.token_id_yes
        entry_price = signal.market.yes_price
    else:
        side = "buy_no"
        token_id = signal.market.token_id_no
        entry_price = signal.market.no_price

    # Calculate number of shares
    num_shares = position_size / entry_price

    # Expected value calculation
    # Assume fair value is 0.5 for binary outcomes
    # EV = (win_prob * payout) - (lose_prob * cost)
    # Simplified: EV = position_size * (0.5 - entry_price) / entry_price
    expected_value = position_size * (Decimal("0.5") - entry_price) / entry_price

    # Record fill
    engine.record_fill(
        token_id=token_id,
        side="buy",
        size=num_shares,
        price=entry_price,
        fee=Decimal("0"),
        timestamp=now.isoformat(),
        market_slug=signal.market.market_id,
        market_question=signal.market.question,
    )

    trade_id = f"disc_{now.strftime('%Y%m%d%H%M%S')}_{signal.market.market_id[:8]}"

    return PaperTrade(
        trade_id=trade_id,
        timestamp=now,
        signal=signal,
        side=side,
        position_size=num_shares,
        entry_price=entry_price,
        expected_value=expected_value,
    )


class DiscountedArbitrageTracker:
    """Tracks paper trades and performance for the strategy."""

    data_dir: Path
    paper_engine: PaperTradingEngine
    profiler: CopytradeProfiler
    fill_tracker: TraderFillTracker

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
        self.profiler = CopytradeProfiler(data_dir=self.data_dir / "copytrade_profiles")
        self.fill_tracker = TraderFillTracker(data_dir=self.data_dir / "copytrade_profiles")

    def get_trades_file(self) -> Path:
        """Get path to trades file."""
        return self.data_dir / "trades.jsonl"

    def load_trades(self) -> list[PaperTrade]:
        """Load all recorded trades."""
        trades: list[PaperTrade] = []
        trades_file = self.get_trades_file()

        if not trades_file.exists():
            return trades

        with open(trades_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    trade = PaperTrade(
                        trade_id=data["trade_id"],
                        timestamp=datetime.fromisoformat(data["timestamp"]),
                        signal=DiscountedArbitrageSignal(  # Simplified reconstruction
                            timestamp=datetime.fromisoformat(data["timestamp"]),
                            market=DiscountedMarket(
                                market_id=data["market_id"],
                                token_id_yes="",
                                token_id_no="",
                                question=data["question"],
                                vertical=data["vertical"],
                                yes_price=Decimal("0"),
                                no_price=Decimal("0"),
                                discounted_side=data["discounted_side"],
                                discounted_price=Decimal(str(data["entry_price"])),
                                end_date=None,
                                volume_usd=None,
                            ),
                            insider_confirmations=[],
                            confirmation_count=0,
                            total_insider_position_usd=Decimal("0"),
                            avg_insider_win_rate=0.0,
                        ),
                        side=data["side"],
                        position_size=Decimal(str(data["position_size"])),
                        entry_price=Decimal(str(data["entry_price"])),
                        expected_value=Decimal(str(data["expected_value"])),
                        resolved=data.get("resolved", False),
                        resolved_at=datetime.fromisoformat(data["resolved_at"])
                        if data.get("resolved_at")
                        else None,
                        winning_outcome=data.get("winning_outcome"),
                        pnl=Decimal(str(data["pnl"])) if data.get("pnl") else None,
                    )
                    trades.append(trade)
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue

        return trades

    def record_trade(self, trade: PaperTrade) -> None:
        """Record a trade to file."""
        trades_file = self.get_trades_file()
        with open(trades_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(trade.to_dict(), sort_keys=True) + "\n")

    def resolve_trade(
        self,
        trade_id: str,
        winning_outcome: bool,
        resolution_price: Decimal = Decimal("1.0"),
    ) -> PaperTrade | None:
        """Mark a trade as resolved with outcome."""
        trades = self.load_trades()

        for i, trade in enumerate(trades):
            if trade.trade_id == trade_id:
                now = datetime.now(UTC)

                # Calculate PnL
                if winning_outcome:
                    pnl = trade.position_size * (resolution_price - trade.entry_price)
                else:
                    pnl = -trade.position_size * trade.entry_price  # Full loss

                resolved_trade = PaperTrade(
                    trade_id=trade.trade_id,
                    timestamp=trade.timestamp,
                    signal=trade.signal,
                    side=trade.side,
                    position_size=trade.position_size,
                    entry_price=trade.entry_price,
                    expected_value=trade.expected_value,
                    resolved=True,
                    resolved_at=now,
                    winning_outcome=winning_outcome,
                    pnl=pnl,
                )

                # Update file
                trades[i] = resolved_trade
                self._save_trades(trades)

                return resolved_trade

        return None

    def _save_trades(self, trades: list[PaperTrade]) -> None:
        """Save all trades to file."""
        trades_file = self.get_trades_file()
        with open(trades_file, "w", encoding="utf-8") as f:
            for trade in trades:
                f.write(json.dumps(trade.to_dict(), sort_keys=True) + "\n")

    def get_performance(self) -> StrategyPerformance:
        """Calculate strategy performance metrics."""
        trades = self.load_trades()
        perf = StrategyPerformance()

        for trade in trades:
            perf.total_trades += 1

            if trade.signal.has_insider_confirmation:
                perf.insider_confirmed_trades += 1
            else:
                perf.non_insider_trades += 1

            if trade.resolved and trade.pnl is not None:
                perf.resolved_trades += 1

                if trade.pnl > 0:
                    perf.winning_trades += 1
                    if trade.signal.has_insider_confirmation:
                        perf.insider_winning += 1
                    else:
                        perf.non_insider_winning += 1
                else:
                    perf.losing_trades += 1
                    if trade.signal.has_insider_confirmation:
                        perf.insider_losing += 1
                    else:
                        perf.non_insider_losing += 1

                perf.total_pnl += trade.pnl

                if trade.signal.has_insider_confirmation:
                    perf.insider_pnl += trade.pnl
                else:
                    perf.non_insider_pnl += trade.pnl

                if trade.resolved_at and trade.timestamp:
                    hold_hours = (trade.resolved_at - trade.timestamp).total_seconds() / 3600
                    perf.total_hold_time_hours += hold_hours

        return perf

    def get_trades_by_vertical(self) -> dict[str, list[PaperTrade]]:
        """Group trades by vertical."""
        trades = self.load_trades()
        result: dict[str, list[PaperTrade]] = {}

        for trade in trades:
            vertical = trade.signal.market.vertical
            if vertical not in result:
                result[vertical] = []
            result[vertical].append(trade)

        return result


def run_discounted_arbitrage_scan(
    snapshots_dir: Path,
    data_dir: Path | None = None,
    dry_run: bool = True,
    max_positions: int = 20,
) -> dict[str, Any]:
    """Run a complete discounted outcome arbitrage scan.

    Args:
        snapshots_dir: Directory with market snapshots
        data_dir: Directory for storing data
        dry_run: If True, don't execute trades
        max_positions: Maximum number of positions to take

    Returns:
        Dictionary with scan results
    """
    now = datetime.now(UTC)

    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR

    tracker = DiscountedArbitrageTracker(data_dir=data_dir)

    logger.info("Starting discounted outcome arbitrage scan at %s", now.isoformat())

    # Find latest snapshot
    snapshot_files = sorted(snapshots_dir.glob("snapshot_*.json"))
    if not snapshot_files:
        return {
            "timestamp": now.isoformat(),
            "error": f"No snapshots found in {snapshots_dir}",
        }

    latest_snapshot = snapshot_files[-1]
    logger.info("Using snapshot: %s", latest_snapshot)

    # Step 1: Find discounted markets
    discounted_markets = find_discounted_markets(latest_snapshot)
    logger.info("Found %d discounted markets", len(discounted_markets))

    # Group by vertical
    by_vertical: dict[str, list[DiscountedMarket]] = {}
    for m in discounted_markets:
        if m.vertical not in by_vertical:
            by_vertical[m.vertical] = []
        by_vertical[m.vertical].append(m)

    for vertical, markets in by_vertical.items():
        logger.info("  %s: %d markets", vertical, len(markets))

    # Step 2: Load insider wallets
    insider_wallets = load_insider_wallets(data_dir)
    logger.info("Loaded %d insider wallets", len(insider_wallets))

    # Step 3: Generate signals with insider confirmation
    signals = generate_signals(
        discounted_markets,
        insider_wallets,
        tracker.profiler,
        tracker.fill_tracker,
    )

    # Split into confirmed and non-confirmed
    confirmed_signals = [s for s in signals if s.has_insider_confirmation]
    non_confirmed_signals = [s for s in signals if not s.has_insider_confirmation]

    logger.info("Generated %d confirmed signals, %d non-confirmed", 
                len(confirmed_signals), len(non_confirmed_signals))

    # Step 4: Execute paper trades (prioritize confirmed signals)
    trades: list[PaperTrade] = []

    if not dry_run:
        # Take confirmed signals first
        for signal in confirmed_signals[:max_positions]:
            trade = execute_paper_trade(signal, tracker.paper_engine)
            if trade:
                tracker.record_trade(trade)
                trades.append(trade)
                logger.info(
                    "Executed confirmed trade: %s @ %.3f (confidence: %.2f)",
                    trade.signal.market.question[:50],
                    trade.entry_price,
                    signal.confidence_score,
                )

        # Fill remaining slots with non-confirmed
        remaining = max_positions - len(trades)
        for signal in non_confirmed_signals[:remaining]:
            trade = execute_paper_trade(signal, tracker.paper_engine)
            if trade:
                tracker.record_trade(trade)
                trades.append(trade)
                logger.info(
                    "Executed non-confirmed trade: %s @ %.3f",
                    trade.signal.market.question[:50],
                    trade.entry_price,
                )

    # Get performance summary
    performance = tracker.get_performance()

    return {
        "timestamp": now.isoformat(),
        "snapshot": str(latest_snapshot),
        "markets_discounted": len(discounted_markets),
        "by_vertical": {v: len(m) for v, m in by_vertical.items()},
        "signals_confirmed": len(confirmed_signals),
        "signals_non_confirmed": len(non_confirmed_signals),
        "trades_executed": len(trades),
        "dry_run": dry_run,
        "performance": performance.to_dict(),
        "top_confirmed_signals": [s.to_dict() for s in confirmed_signals[:5]],
        "top_non_confirmed_signals": [s.to_dict() for s in non_confirmed_signals[:5]],
    }
