"""Copytrade profiling: rank traders from fills (realized PnL, win rate, hold time).

Analyzes trader performance from actual fill data rather than leaderboard data,
providing metrics like:
- Realized PnL from completed round-trips
- Win rate (% of profitable trades)
- Average hold time
- Sharpe-like risk-adjusted metrics
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

from .trader_fills import TraderFill, TraderFillTracker


# Constants
DEFAULT_DATA_DIR = Path("data/copytrade_profiles")
PROFILES_FILE = "profiles.json"
TRADER_METRICS_DIR = "metrics"


@dataclass
class TradeRoundTrip:
    """A completed buy-sell round trip for PnL calculation.

    Represents a single profitable or loss-making trade where
    shares were bought and later sold.
    """

    token_id: str
    market_slug: str | None
    entry_time: datetime
    exit_time: datetime
    entry_price: Decimal
    exit_price: Decimal
    size: Decimal
    fees: Decimal
    realized_pnl: Decimal
    realized_pnl_pct: Decimal  # Percentage return

    @property
    def hold_time(self) -> timedelta:
        """Time between entry and exit."""
        return self.exit_time - self.entry_time

    @property
    def is_win(self) -> bool:
        """True if this trade was profitable."""
        return self.realized_pnl > 0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "token_id": self.token_id,
            "market_slug": self.market_slug,
            "entry_time": self.entry_time.isoformat(),
            "exit_time": self.exit_time.isoformat(),
            "entry_price": float(self.entry_price),
            "exit_price": float(self.exit_price),
            "size": float(self.size),
            "fees": float(self.fees),
            "realized_pnl": float(self.realized_pnl),
            "realized_pnl_pct": float(self.realized_pnl_pct),
            "hold_time_seconds": self.hold_time.total_seconds(),
            "is_win": self.is_win,
        }


@dataclass
class TraderPerformanceMetrics:
    """Performance metrics for a trader computed from fills.

    All metrics are calculated from actual fill data, not leaderboard data.
    """

    address: str
    computed_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    # Trade counts
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0

    # PnL metrics
    realized_pnl: Decimal = field(default_factory=lambda: Decimal("0"))
    total_fees: Decimal = field(default_factory=lambda: Decimal("0"))
    gross_pnl: Decimal = field(default_factory=lambda: Decimal("0"))  # Before fees
    avg_trade_pnl: Decimal = field(default_factory=lambda: Decimal("0"))
    best_trade: Decimal = field(default_factory=lambda: Decimal("0"))
    worst_trade: Decimal = field(default_factory=lambda: Decimal("0"))

    # Win rate
    win_rate: float = 0.0  # 0-100

    # Hold time metrics (in hours)
    avg_hold_time_hours: float = 0.0
    median_hold_time_hours: float = 0.0
    min_hold_time_hours: float = 0.0
    max_hold_time_hours: float = 0.0

    # Risk metrics
    pnl_volatility: float = 0.0  # Std dev of trade PnLs
    sharpe_like_ratio: float = 0.0  # Avg PnL / volatility
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0

    # Volume and activity
    total_volume: Decimal = field(default_factory=lambda: Decimal("0"))
    unique_markets: int = 0
    first_trade_at: str | None = None
    last_trade_at: str | None = None
    trading_days: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "address": self.address,
            "computed_at": self.computed_at,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "realized_pnl": float(self.realized_pnl),
            "total_fees": float(self.total_fees),
            "gross_pnl": float(self.gross_pnl),
            "avg_trade_pnl": float(self.avg_trade_pnl),
            "best_trade": float(self.best_trade),
            "worst_trade": float(self.worst_trade),
            "win_rate": self.win_rate,
            "avg_hold_time_hours": self.avg_hold_time_hours,
            "median_hold_time_hours": self.median_hold_time_hours,
            "min_hold_time_hours": self.min_hold_time_hours,
            "max_hold_time_hours": self.max_hold_time_hours,
            "pnl_volatility": self.pnl_volatility,
            "sharpe_like_ratio": self.sharpe_like_ratio,
            "max_consecutive_wins": self.max_consecutive_wins,
            "max_consecutive_losses": self.max_consecutive_losses,
            "total_volume": float(self.total_volume),
            "unique_markets": self.unique_markets,
            "first_trade_at": self.first_trade_at,
            "last_trade_at": self.last_trade_at,
            "trading_days": self.trading_days,
        }

    @classmethod
    def from_dict(cls, data: dict) -> TraderPerformanceMetrics:
        """Create from dictionary."""
        return cls(
            address=data["address"],
            computed_at=data.get("computed_at", datetime.now(UTC).isoformat()),
            total_trades=data.get("total_trades", 0),
            winning_trades=data.get("winning_trades", 0),
            losing_trades=data.get("losing_trades", 0),
            realized_pnl=Decimal(str(data.get("realized_pnl", 0))),
            total_fees=Decimal(str(data.get("total_fees", 0))),
            gross_pnl=Decimal(str(data.get("gross_pnl", 0))),
            avg_trade_pnl=Decimal(str(data.get("avg_trade_pnl", 0))),
            best_trade=Decimal(str(data.get("best_trade", 0))),
            worst_trade=Decimal(str(data.get("worst_trade", 0))),
            win_rate=data.get("win_rate", 0.0),
            avg_hold_time_hours=data.get("avg_hold_time_hours", 0.0),
            median_hold_time_hours=data.get("median_hold_time_hours", 0.0),
            min_hold_time_hours=data.get("min_hold_time_hours", 0.0),
            max_hold_time_hours=data.get("max_hold_time_hours", 0.0),
            pnl_volatility=data.get("pnl_volatility", 0.0),
            sharpe_like_ratio=data.get("sharpe_like_ratio", 0.0),
            max_consecutive_wins=data.get("max_consecutive_wins", 0),
            max_consecutive_losses=data.get("max_consecutive_losses", 0),
            total_volume=Decimal(str(data.get("total_volume", 0))),
            unique_markets=data.get("unique_markets", 0),
            first_trade_at=data.get("first_trade_at"),
            last_trade_at=data.get("last_trade_at"),
            trading_days=data.get("trading_days", 0),
        )


@dataclass
class TraderRanking:
    """Ranking score for a trader based on multiple factors."""

    address: str
    rank: int = 0
    overall_score: float = 0.0

    # Component scores (0-100)
    pnl_score: float = 0.0
    win_rate_score: float = 0.0
    consistency_score: float = 0.0
    experience_score: float = 0.0

    # Raw metrics snapshot
    realized_pnl: Decimal = field(default_factory=lambda: Decimal("0"))
    win_rate: float = 0.0
    avg_hold_time_hours: float = 0.0
    total_trades: int = 0

    computed_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "address": self.address,
            "rank": self.rank,
            "overall_score": self.overall_score,
            "pnl_score": self.pnl_score,
            "win_rate_score": self.win_rate_score,
            "consistency_score": self.consistency_score,
            "experience_score": self.experience_score,
            "realized_pnl": float(self.realized_pnl),
            "win_rate": self.win_rate,
            "avg_hold_time_hours": self.avg_hold_time_hours,
            "total_trades": self.total_trades,
            "computed_at": self.computed_at,
        }


class CopytradeProfiler:
    """Profiles traders from fills for copytrade ranking.

    Analyzes trader performance from actual fill data to identify
    top performers based on realized PnL, win rate, and hold time.

    Usage:
        profiler = CopytradeProfiler()
        metrics = profiler.compute_metrics("0x1234...")
        rankings = profiler.rank_traders(["0x1234...", "0x5678..."])
    """

    data_dir: Path
    fill_tracker: TraderFillTracker
    _metrics_cache: dict[str, TraderPerformanceMetrics]

    def __init__(self, data_dir: Path | str | None = None) -> None:
        """Initialize profiler.

        Args:
            data_dir: Directory for storing computed metrics
        """
        self.data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / TRADER_METRICS_DIR).mkdir(exist_ok=True)

        self.fill_tracker = TraderFillTracker(data_dir=self.data_dir)
        self._metrics_cache = {}

    def _parse_timestamp(self, ts: str) -> datetime:
        """Parse ISO timestamp to datetime."""
        ts = ts.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(ts)
        except ValueError:
            return datetime.now(UTC)

    def _compute_round_trips(self, fills: list[TraderFill]) -> list[TradeRoundTrip]:
        """Compute completed round trips from fills.

        Matches buys with subsequent sells to calculate realized PnL
        using FIFO (first in, first out) matching.

        Args:
            fills: List of trader fills

        Returns:
            List of completed trade round trips
        """
        round_trips = []

        # Group fills by token_id
        fills_by_token: dict[str, list[TraderFill]] = {}
        for fill in fills:
            if fill.token_id not in fills_by_token:
                fills_by_token[fill.token_id] = []
            fills_by_token[fill.token_id].append(fill)

        # Process each token separately
        for token_id, token_fills in fills_by_token.items():
            # Sort by timestamp
            sorted_fills = sorted(token_fills, key=lambda f: f.timestamp)

            # Track buy queue (FIFO)
            buy_queue: list[
                tuple[datetime, Decimal, Decimal, Decimal]
            ] = []  # (time, size, price, fee)

            for fill in sorted_fills:
                fill_time = self._parse_timestamp(fill.timestamp)

                if fill.side == "buy":
                    # Add to buy queue
                    buy_queue.append((fill_time, fill.size, fill.price, fill.fee))
                else:
                    # Sell: match against buy queue (FIFO)
                    remaining = fill.size
                    total_cost = Decimal("0")
                    total_entry_fees = Decimal("0")
                    entry_time: datetime | None = None

                    while remaining > 0 and buy_queue:
                        buy_time, buy_size, buy_price, buy_fee = buy_queue[0]
                        match_size = min(remaining, buy_size)

                        total_cost += match_size * buy_price
                        total_entry_fees += buy_fee * (match_size / buy_size)
                        if entry_time is None:
                            entry_time = buy_time

                        remaining -= match_size
                        buy_queue[0] = (buy_time, buy_size - match_size, buy_price, buy_fee)

                        if buy_queue[0][1] <= 0:
                            buy_queue.pop(0)

                    if entry_time and fill.size > remaining:
                        # Calculate PnL for completed round trip
                        matched_size = fill.size - remaining
                        proceeds = matched_size * fill.price
                        fees = total_entry_fees + fill.fee
                        realized_pnl = proceeds - total_cost - fees

                        # Calculate percentage return
                        if total_cost > 0:
                            realized_pnl_pct = (realized_pnl / total_cost) * 100
                        else:
                            realized_pnl_pct = Decimal("0")

                        round_trips.append(
                            TradeRoundTrip(
                                token_id=token_id,
                                market_slug=fill.market_slug,
                                entry_time=entry_time,
                                exit_time=fill_time,
                                entry_price=total_cost / matched_size
                                if matched_size > 0
                                else Decimal("0"),
                                exit_price=fill.price,
                                size=matched_size,
                                fees=fees,
                                realized_pnl=realized_pnl,
                                realized_pnl_pct=realized_pnl_pct,
                            )
                        )

        return round_trips

    def compute_metrics(self, address: str) -> TraderPerformanceMetrics:
        """Compute performance metrics from fills.

        Args:
            address: Trader wallet address

        Returns:
            TraderPerformanceMetrics with all computed metrics
        """
        address = address.lower()

        # Check cache
        if address in self._metrics_cache:
            return self._metrics_cache[address]

        # Load fills
        fills = self.fill_tracker.load_fills(address)

        if not fills:
            return TraderPerformanceMetrics(address=address)

        # Compute round trips
        round_trips = self._compute_round_trips(fills)

        # Basic metrics
        metrics = TraderPerformanceMetrics(address=address)
        metrics.total_trades = len(round_trips)

        if not round_trips:
            return metrics

        # PnL metrics
        pnls = [rt.realized_pnl for rt in round_trips]
        metrics.realized_pnl = sum(pnls)
        metrics.gross_pnl = metrics.realized_pnl  # Already includes fees in calculation
        metrics.total_fees = sum(rt.fees for rt in round_trips)
        metrics.avg_trade_pnl = metrics.realized_pnl / len(round_trips)
        metrics.best_trade = max(pnls)
        metrics.worst_trade = min(pnls)

        # Win rate
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        metrics.winning_trades = len(wins)
        metrics.losing_trades = len(losses)
        metrics.win_rate = (len(wins) / len(pnls) * 100) if pnls else 0.0

        # Hold time metrics
        hold_times = [rt.hold_time.total_seconds() / 3600 for rt in round_trips]
        if hold_times:
            metrics.avg_hold_time_hours = sum(hold_times) / len(hold_times)
            metrics.min_hold_time_hours = min(hold_times)
            metrics.max_hold_time_hours = max(hold_times)
            # Median
            sorted_times = sorted(hold_times)
            mid = len(sorted_times) // 2
            if len(sorted_times) % 2 == 0:
                metrics.median_hold_time_hours = (sorted_times[mid - 1] + sorted_times[mid]) / 2
            else:
                metrics.median_hold_time_hours = sorted_times[mid]

        # Risk metrics
        if len(pnls) > 1:
            mean_pnl = float(metrics.avg_trade_pnl)
            variance = sum((float(p) - mean_pnl) ** 2 for p in pnls) / (len(pnls) - 1)
            metrics.pnl_volatility = variance**0.5
            if metrics.pnl_volatility > 0:
                metrics.sharpe_like_ratio = mean_pnl / metrics.pnl_volatility

        # Consecutive wins/losses
        current_streak = 0
        max_wins = 0
        max_losses = 0
        for rt in round_trips:
            if rt.is_win:
                if current_streak > 0:
                    current_streak += 1
                else:
                    current_streak = 1
                max_wins = max(max_wins, current_streak)
            else:
                if current_streak < 0:
                    current_streak -= 1
                else:
                    current_streak = -1
                max_losses = max(max_losses, abs(current_streak))
        metrics.max_consecutive_wins = max_wins
        metrics.max_consecutive_losses = max_losses

        # Volume and activity
        metrics.total_volume = sum(f.size * f.price for f in fills)
        metrics.unique_markets = len(set(f.market_slug for f in fills if f.market_slug))

        # Time range
        timestamps = [self._parse_timestamp(f.timestamp) for f in fills]
        if timestamps:
            first = min(timestamps)
            last = max(timestamps)
            metrics.first_trade_at = first.isoformat()
            metrics.last_trade_at = last.isoformat()
            metrics.trading_days = (last - first).days + 1

        # Cache result
        self._metrics_cache[address] = metrics

        return metrics

    def save_metrics(self, address: str, metrics: TraderPerformanceMetrics | None = None) -> Path:
        """Save computed metrics to disk.

        Args:
            address: Trader address
            metrics: Metrics to save (computes if not provided)

        Returns:
            Path to saved file
        """
        if metrics is None:
            metrics = self.compute_metrics(address)

        metrics_path = self.data_dir / TRADER_METRICS_DIR / f"{address.lower()}.json"
        metrics_path.write_text(json.dumps(metrics.to_dict(), indent=2))
        return metrics_path

    def load_metrics(self, address: str) -> TraderPerformanceMetrics | None:
        """Load previously computed metrics from disk.

        Args:
            address: Trader address

        Returns:
            TraderPerformanceMetrics or None if not found
        """
        metrics_path = self.data_dir / TRADER_METRICS_DIR / f"{address.lower()}.json"
        if not metrics_path.exists():
            return None

        try:
            data = json.loads(metrics_path.read_text())
            return TraderPerformanceMetrics.from_dict(data)
        except (json.JSONDecodeError, KeyError, ValueError):
            return None

    def rank_traders(
        self,
        addresses: Sequence[str],
        min_trades: int = 5,
        min_win_rate: float = 0.0,
    ) -> list[TraderRanking]:
        """Rank traders based on performance metrics.

        Scoring:
        - PnL score (40%): Based on total realized PnL relative to peers
        - Win rate score (30%): Based on win percentage
        - Consistency score (20%): Based on Sharpe-like ratio and hold times
        - Experience score (10%): Based on number of trades and trading days

        Args:
            addresses: List of trader addresses to rank
            min_trades: Minimum trades required to be ranked
            min_win_rate: Minimum win rate to be ranked

        Returns:
            List of TraderRanking sorted by overall score
        """
        # Compute metrics for all traders
        all_metrics: dict[str, TraderPerformanceMetrics] = {}
        for addr in addresses:
            metrics = self.compute_metrics(addr)
            if metrics.total_trades >= min_trades and metrics.win_rate >= min_win_rate:
                all_metrics[addr.lower()] = metrics

        if not all_metrics:
            return []

        # Calculate normalization factors
        max_pnl = max(m.realized_pnl for m in all_metrics.values())
        min_pnl = min(m.realized_pnl for m in all_metrics.values())
        pnl_range = float(max_pnl - min_pnl) if max_pnl != min_pnl else 1.0

        max_trades = max(m.total_trades for m in all_metrics.values())
        max_days = max(m.trading_days for m in all_metrics.values())

        # Score each trader
        rankings: list[TraderRanking] = []
        for addr, metrics in all_metrics.items():
            ranking = TraderRanking(address=addr)

            # Raw metrics
            ranking.realized_pnl = metrics.realized_pnl
            ranking.win_rate = metrics.win_rate
            ranking.avg_hold_time_hours = metrics.avg_hold_time_hours
            ranking.total_trades = metrics.total_trades

            # PnL score (0-100): normalized relative to peers
            if pnl_range > 0:
                pnl_normalized = float(metrics.realized_pnl - min_pnl) / pnl_range
                ranking.pnl_score = max(0, min(100, pnl_normalized * 100))
            else:
                ranking.pnl_score = 50.0 if metrics.realized_pnl >= 0 else 0.0

            # Win rate score (0-100): direct mapping
            ranking.win_rate_score = metrics.win_rate

            # Consistency score (0-100): based on Sharpe and hold time consistency
            sharpe_score = min(max(metrics.sharpe_like_ratio * 20, 0), 50)
            # Penalty for very short hold times (potential wash trading)
            hold_time_score = 50.0
            if metrics.avg_hold_time_hours < 1:
                hold_time_score = 25.0  # Penalty for sub-1-hour average
            elif metrics.avg_hold_time_hours > 24:
                hold_time_score = 50.0  # Full score for day+ holds
            else:
                hold_time_score = 25.0 + (metrics.avg_hold_time_hours / 24) * 25.0

            ranking.consistency_score = sharpe_score + hold_time_score

            # Experience score (0-100): based on trades and days
            trade_score = min(metrics.total_trades / max_trades * 50, 50) if max_trades > 0 else 0
            days_score = min(metrics.trading_days / max_days * 50, 50) if max_days > 0 else 0
            ranking.experience_score = trade_score + days_score

            # Overall score: weighted composite
            ranking.overall_score = (
                ranking.pnl_score * 0.40
                + ranking.win_rate_score * 0.30
                + ranking.consistency_score * 0.20
                + ranking.experience_score * 0.10
            )

            rankings.append(ranking)

        # Sort by overall score descending
        rankings.sort(key=lambda r: r.overall_score, reverse=True)

        # Assign ranks
        for i, ranking in enumerate(rankings, 1):
            ranking.rank = i

        return rankings

    def get_top_traders(
        self,
        addresses: Sequence[str],
        k: int = 10,
        min_trades: int = 5,
        min_score: float = 0.0,
    ) -> list[TraderRanking]:
        """Get top-K traders by ranking score.

        Args:
            addresses: List of trader addresses to consider
            k: Number of top traders to return
            min_trades: Minimum trades required
            min_score: Minimum overall score required

        Returns:
            List of top TraderRanking objects
        """
        rankings = self.rank_traders(addresses, min_trades=min_trades)
        return [r for r in rankings[:k] if r.overall_score >= min_score]

    def get_trader_summary(self, address: str) -> dict:
        """Get human-readable summary for a trader.

        Args:
            address: Trader address

        Returns:
            Dict with summary statistics
        """
        metrics = self.compute_metrics(address)

        return {
            "address": address,
            "performance": {
                "realized_pnl_usd": float(metrics.realized_pnl),
                "win_rate_pct": round(metrics.win_rate, 1),
                "total_trades": metrics.total_trades,
                "winning_trades": metrics.winning_trades,
                "losing_trades": metrics.losing_trades,
            },
            "hold_times": {
                "avg_hours": round(metrics.avg_hold_time_hours, 1),
                "median_hours": round(metrics.median_hold_time_hours, 1),
                "min_hours": round(metrics.min_hold_time_hours, 1),
                "max_hours": round(metrics.max_hold_time_hours, 1),
            },
            "risk": {
                "best_trade_usd": float(metrics.best_trade),
                "worst_trade_usd": float(metrics.worst_trade),
                "pnl_volatility": round(metrics.pnl_volatility, 2),
                "sharpe_like_ratio": round(metrics.sharpe_like_ratio, 2),
                "max_consecutive_wins": metrics.max_consecutive_wins,
                "max_consecutive_losses": metrics.max_consecutive_losses,
            },
            "activity": {
                "total_volume_usd": float(metrics.total_volume),
                "unique_markets": metrics.unique_markets,
                "trading_days": metrics.trading_days,
                "first_trade": metrics.first_trade_at,
                "last_trade": metrics.last_trade_at,
            },
        }

    def export_rankings(
        self,
        addresses: Sequence[str],
        output_path: Path | str | None = None,
    ) -> Path:
        """Export rankings to JSON file.

        Args:
            addresses: List of trader addresses
            output_path: Output file path (default: data/copytrade_profiles/rankings_YYYYMMDD.json)

        Returns:
            Path to exported file
        """
        rankings = self.rank_traders(addresses)

        data = {
            "exported_at": datetime.now(UTC).isoformat(),
            "trader_count": len(addresses),
            "ranked_count": len(rankings),
            "rankings": [r.to_dict() for r in rankings],
        }

        if output_path is None:
            timestamp = datetime.now(UTC).strftime("%Y%m%d")
            output_path = self.data_dir / f"rankings_{timestamp}.json"
        else:
            output_path = Path(output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(data, indent=2))
        return output_path
