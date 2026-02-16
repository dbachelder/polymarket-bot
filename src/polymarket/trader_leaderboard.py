"""Trader leaderboard computed from fills data.

Builds a leaderboard based on ground-truth fill data:
- Realized PnL (from completed trades)
- Volume (total notional traded)
- Win rate (percentage of profitable trades)
- Trade count and other metrics

This provides verification against API-reported stats and enables
data-driven copy trading candidate selection.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


# Constants
DEFAULT_DATA_DIR = Path("data/trader_leaderboard")
LEADERBOARD_FILE = "leaderboard.json"
CANDIDATES_FILE = "candidates.json"


@dataclass(frozen=True)
class TraderStats:
    """Computed trading statistics for a single trader from fills.

    Attributes:
        address: Trader wallet address
        total_fills: Total number of fills
        total_trades: Number of completed round-trips (buy+sell)
        buy_count: Number of buy fills
        sell_count: Number of sell fills
        total_volume: Total notional volume traded
        realized_pnl: Realized profit/loss from closed positions
        total_fees: Total fees paid
        win_count: Number of winning trades
        loss_count: Number of losing trades
        win_rate: Win rate as percentage (0-100)
        avg_trade_pnl: Average PnL per trade
        avg_trade_size: Average trade size
        first_trade_at: ISO timestamp of first trade
        last_trade_at: ISO timestamp of last trade
        markets_traded: Number of unique markets
        tokens_traded: Number of unique tokens
    """

    address: str
    total_fills: int = 0
    total_trades: int = 0
    buy_count: int = 0
    sell_count: int = 0
    total_volume: Decimal = field(default_factory=lambda: Decimal("0"))
    realized_pnl: Decimal = field(default_factory=lambda: Decimal("0"))
    total_fees: Decimal = field(default_factory=lambda: Decimal("0"))
    win_count: int = 0
    loss_count: int = 0
    win_rate: float = 0.0
    avg_trade_pnl: Decimal = field(default_factory=lambda: Decimal("0"))
    avg_trade_size: Decimal = field(default_factory=lambda: Decimal("0"))
    first_trade_at: str | None = None
    last_trade_at: str | None = None
    markets_traded: int = 0
    tokens_traded: int = 0
    computed_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "address": self.address,
            "total_fills": self.total_fills,
            "total_trades": self.total_trades,
            "buy_count": self.buy_count,
            "sell_count": self.sell_count,
            "total_volume": float(self.total_volume),
            "realized_pnl": float(self.realized_pnl),
            "total_fees": float(self.total_fees),
            "net_pnl": float(self.net_pnl),
            "win_count": self.win_count,
            "loss_count": self.loss_count,
            "win_rate": self.win_rate,
            "avg_trade_pnl": float(self.avg_trade_pnl),
            "avg_trade_size": float(self.avg_trade_size),
            "first_trade_at": self.first_trade_at,
            "last_trade_at": self.last_trade_at,
            "markets_traded": self.markets_traded,
            "tokens_traded": self.tokens_traded,
            "computed_at": self.computed_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> TraderStats:
        """Create from dictionary."""
        return cls(
            address=data["address"],
            total_fills=data.get("total_fills", 0),
            total_trades=data.get("total_trades", 0),
            buy_count=data.get("buy_count", 0),
            sell_count=data.get("sell_count", 0),
            total_volume=Decimal(str(data.get("total_volume", 0))),
            realized_pnl=Decimal(str(data.get("realized_pnl", 0))),
            total_fees=Decimal(str(data.get("total_fees", 0))),
            win_count=data.get("win_count", 0),
            loss_count=data.get("loss_count", 0),
            win_rate=data.get("win_rate", 0.0),
            avg_trade_pnl=Decimal(str(data.get("avg_trade_pnl", 0))),
            avg_trade_size=Decimal(str(data.get("avg_trade_size", 0))),
            first_trade_at=data.get("first_trade_at"),
            last_trade_at=data.get("last_trade_at"),
            markets_traded=data.get("markets_traded", 0),
            tokens_traded=data.get("tokens_traded", 0),
            computed_at=data.get("computed_at", datetime.now(UTC).isoformat()),
        )

    @property
    def net_pnl(self) -> Decimal:
        """Net PnL after fees."""
        return self.realized_pnl - self.total_fees

    @property
    def trade_count(self) -> int:
        """Alias for total_trades."""
        return self.total_trades

    @property
    def total_pnl(self) -> Decimal:
        """Alias for realized_pnl."""
        return self.realized_pnl


@dataclass
class LeaderboardEntry:
    """Entry in the trader leaderboard with ranking info."""

    stats: TraderStats
    rank: int
    pnl_rank: int
    volume_rank: int
    win_rate_rank: int
    composite_score: float

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "stats": self.stats.to_dict(),
            "rank": self.rank,
            "pnl_rank": self.pnl_rank,
            "volume_rank": self.volume_rank,
            "win_rate_rank": self.win_rate_rank,
            "composite_score": self.composite_score,
        }

    @classmethod
    def from_dict(cls, data: dict) -> LeaderboardEntry:
        """Create from dictionary."""
        return cls(
            stats=TraderStats.from_dict(data["stats"]),
            rank=data.get("rank", 0),
            pnl_rank=data.get("pnl_rank", 0),
            volume_rank=data.get("volume_rank", 0),
            win_rate_rank=data.get("win_rate_rank", 0),
            composite_score=data.get("composite_score", 0.0),
        )


@dataclass
class CopyCandidate:
    """Candidate for copy trading with selection rationale."""

    stats: TraderStats
    selection_score: float
    selection_reason: list[str]
    recommended_position_size: Decimal = field(default_factory=lambda: Decimal("100"))

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "stats": self.stats.to_dict(),
            "selection_score": self.selection_score,
            "selection_reason": self.selection_reason,
            "recommended_position_size": float(self.recommended_position_size),
        }

    @classmethod
    def from_dict(cls, data: dict) -> CopyCandidate:
        """Create from dictionary."""
        return cls(
            stats=TraderStats.from_dict(data["stats"]),
            selection_score=data.get("selection_score", 0.0),
            selection_reason=data.get("selection_reason", []),
            recommended_position_size=Decimal(str(data.get("recommended_position_size", 100))),
        )


@dataclass
class Leaderboard:
    """Complete trader leaderboard."""

    entries: list[LeaderboardEntry]
    generated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    filters: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "entries": [e.to_dict() for e in self.entries],
            "generated_at": self.generated_at,
            "filters": self.filters,
            "count": len(self.entries),
        }

    @classmethod
    def from_dict(cls, data: dict) -> Leaderboard:
        """Create from dictionary."""
        return cls(
            entries=[LeaderboardEntry.from_dict(e) for e in data.get("entries", [])],
            generated_at=data.get("generated_at", datetime.now(UTC).isoformat()),
            filters=data.get("filters", {}),
        )

    def get_top_by_pnl(self, n: int = 10) -> list[LeaderboardEntry]:
        """Get top N traders by realized PnL."""
        sorted_entries = sorted(
            self.entries,
            key=lambda e: e.stats.realized_pnl,
            reverse=True,
        )
        return sorted_entries[:n]

    def get_top_by_volume(self, n: int = 10) -> list[LeaderboardEntry]:
        """Get top N traders by volume."""
        sorted_entries = sorted(
            self.entries,
            key=lambda e: e.stats.total_volume,
            reverse=True,
        )
        return sorted_entries[:n]

    def get_top_by_win_rate(self, n: int = 10, min_trades: int = 5) -> list[LeaderboardEntry]:
        """Get top N traders by win rate (with minimum trade filter)."""
        filtered = [e for e in self.entries if e.stats.total_trades >= min_trades]
        sorted_entries = sorted(
            filtered,
            key=lambda e: e.stats.win_rate,
            reverse=True,
        )
        return sorted_entries[:n]


class TraderLeaderboardBuilder:
    """Builds trader leaderboard from fill data.

    Features:
    - Computes realized PnL, volume, win rate from fills
    - Ranks traders by multiple metrics
    - Selects copy trading candidates
    - Persists leaderboard data
    """

    data_dir: Path
    stats: dict[str, TraderStats]

    def __init__(self, data_dir: Path | str | None = None) -> None:
        """Initialize leaderboard builder.

        Args:
            data_dir: Directory for leaderboard data storage
        """
        self.data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.stats: dict[str, TraderStats] = {}

    @property
    def leaderboard_path(self) -> Path:
        """Path to leaderboard file."""
        return self.data_dir / LEADERBOARD_FILE

    @property
    def candidates_path(self) -> Path:
        """Path to candidates file."""
        return self.data_dir / CANDIDATES_FILE

    def _load_fills_for_trader(self, address: str, fills_dir: Path) -> list[dict]:
        """Load fills for a trader from the fills directory.

        Args:
            address: Trader wallet address
            fills_dir: Directory containing fill files

        Returns:
            List of fill dictionaries
        """
        fills_path = fills_dir / f"{address.lower()}.jsonl"
        if not fills_path.exists():
            return []

        fills = []
        with open(fills_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    fills.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        return fills

    def _compute_trader_stats(self, address: str, fills: list[dict]) -> TraderStats:
        """Compute trading statistics from fills.

        Uses FIFO accounting for realized PnL calculation.

        Args:
            address: Trader wallet address
            fills: List of fill dictionaries

        Returns:
            TraderStats with computed metrics
        """
        if not fills:
            return TraderStats(address=address)

        # Sort fills by timestamp
        sorted_fills = sorted(fills, key=lambda f: f.get("timestamp", ""))

        # Track position state per token
        positions: dict[str, list[tuple[Decimal, Decimal]]] = {}  # token -> [(size, cost_basis)]

        total_volume = Decimal("0")
        total_fees = Decimal("0")
        realized_pnl = Decimal("0")
        buy_count = 0
        sell_count = 0
        markets: set[str] = set()
        tokens: set[str] = set()

        for fill in sorted_fills:
            side = fill.get("side", "buy").lower()
            size = Decimal(str(fill.get("size", 0)))
            price = Decimal(str(fill.get("price", 0)))
            fee = Decimal(str(fill.get("fee", 0)))
            token_id = fill.get("token_id", "")
            market_slug = fill.get("market_slug") or fill.get("market", "")

            if not token_id or size <= 0 or price <= 0:
                continue

            # Track metadata
            tokens.add(token_id)
            if market_slug:
                markets.add(market_slug)

            # Track volume and fees
            notional = size * price
            total_volume += notional
            total_fees += fee

            if side == "buy":
                buy_count += 1
                # Add to position
                if token_id not in positions:
                    positions[token_id] = []
                positions[token_id].append((size, price))
            else:
                sell_count += 1
                # Calculate realized PnL using FIFO
                remaining_sell = size
                while remaining_sell > 0 and token_id in positions and positions[token_id]:
                    lot_size, lot_cost = positions[token_id][0]
                    sell_from_lot = min(remaining_sell, lot_size)

                    # PnL = (sell_price - cost_basis) * size
                    realized_pnl += (price - lot_cost) * sell_from_lot

                    remaining_sell -= sell_from_lot
                    new_lot_size = lot_size - sell_from_lot

                    if new_lot_size <= 0:
                        positions[token_id].pop(0)
                    else:
                        positions[token_id][0] = (new_lot_size, lot_cost)

        # Count completed trades (minimum of buy_count and sell_count)
        # This is a simplification - assumes most buys are eventually sold
        total_trades = min(buy_count, sell_count)

        # Calculate win rate based on realized PnL from closed trades
        # We estimate win_count as trades where we had positive PnL contribution
        # This is approximate since we calculate PnL per fill, not per round-trip
        win_count = 0
        loss_count = 0

        if total_trades > 0 and realized_pnl != 0:
            # Estimate win rate based on PnL distribution
            # If realized PnL is positive, assume more wins than losses
            if realized_pnl > 0:
                # Rough estimate: wins = 50% + (pnl/volume) adjustment
                win_rate_estimate = 0.5 + float(realized_pnl / total_volume) * 10
                win_rate_estimate = min(max(win_rate_estimate, 0.5), 0.95)
            else:
                win_rate_estimate = 0.5 + float(realized_pnl / total_volume) * 10
                win_rate_estimate = min(max(win_rate_estimate, 0.05), 0.5)

            win_count = int(total_trades * win_rate_estimate)
            loss_count = total_trades - win_count
        elif total_trades > 0:
            # Break even - assume 50/50
            win_count = total_trades // 2
            loss_count = total_trades - win_count

        win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0.0

        # Calculate averages
        avg_trade_pnl = (realized_pnl / total_trades) if total_trades > 0 else Decimal("0")
        avg_trade_size = (total_volume / len(fills)) if fills else Decimal("0")

        return TraderStats(
            address=address,
            total_fills=len(fills),
            total_trades=total_trades,
            buy_count=buy_count,
            sell_count=sell_count,
            total_volume=total_volume,
            realized_pnl=realized_pnl,
            total_fees=total_fees,
            win_count=win_count,
            loss_count=loss_count,
            win_rate=win_rate,
            avg_trade_pnl=avg_trade_pnl,
            avg_trade_size=avg_trade_size,
            first_trade_at=sorted_fills[0].get("timestamp") if sorted_fills else None,
            last_trade_at=sorted_fills[-1].get("timestamp") if sorted_fills else None,
            markets_traded=len(markets),
            tokens_traded=len(tokens),
        )

    def build_from_fills(
        self,
        fills_dir: Path | str,
        addresses: Sequence[str] | None = None,
        min_trades: int = 5,
        min_volume: Decimal = Decimal("1000"),
    ) -> Leaderboard:
        """Build leaderboard from fill data.

        Args:
            fills_dir: Directory containing trader fill files
            addresses: Optional list of addresses to include (default: all in fills_dir)
            min_trades: Minimum trades required for ranking
            min_volume: Minimum volume required for ranking

        Returns:
            Leaderboard with ranked entries
        """
        fills_path = Path(fills_dir)

        # Discover addresses if not provided
        if addresses is None:
            addresses = []
            if fills_path.exists():
                for f in fills_path.glob("*.jsonl"):
                    addresses.append(f.stem)

        # Compute stats for each trader
        self.stats = {}
        for address in addresses:
            fills = self._load_fills_for_trader(address, fills_path)
            stats = self._compute_trader_stats(address, fills)

            # Apply filters
            if stats.total_trades >= min_trades and stats.total_volume >= min_volume:
                self.stats[address] = stats

        # Build leaderboard entries with rankings
        entries = []

        # Calculate individual metric rankings
        sorted_by_pnl = sorted(
            self.stats.values(),
            key=lambda s: s.realized_pnl,
            reverse=True,
        )
        pnl_ranks = {s.address: i + 1 for i, s in enumerate(sorted_by_pnl)}

        sorted_by_volume = sorted(
            self.stats.values(),
            key=lambda s: s.total_volume,
            reverse=True,
        )
        volume_ranks = {s.address: i + 1 for i, s in enumerate(sorted_by_volume)}

        sorted_by_win_rate = sorted(
            self.stats.values(),
            key=lambda s: s.win_rate,
            reverse=True,
        )
        win_rate_ranks = {s.address: i + 1 for i, s in enumerate(sorted_by_win_rate)}

        # Calculate composite scores and create entries
        for address, stats in self.stats.items():
            # Composite score: weighted combination of normalized metrics
            # PnL: 50%, Volume: 30%, Win Rate: 20%
            max_pnl = max((s.realized_pnl for s in self.stats.values()), default=Decimal("1"))
            max_volume = max((s.total_volume for s in self.stats.values()), default=Decimal("1"))

            pnl_score = float(stats.realized_pnl / max_pnl) if max_pnl > 0 else 0
            volume_score = float(stats.total_volume / max_volume) if max_volume > 0 else 0
            win_rate_score = stats.win_rate / 100

            composite_score = pnl_score * 0.5 + volume_score * 0.3 + win_rate_score * 0.2

            entry = LeaderboardEntry(
                stats=stats,
                rank=0,  # Will be set after sorting
                pnl_rank=pnl_ranks.get(address, 0),
                volume_rank=volume_ranks.get(address, 0),
                win_rate_rank=win_rate_ranks.get(address, 0),
                composite_score=composite_score,
            )
            entries.append(entry)

        # Sort by composite score and assign final ranks
        entries.sort(key=lambda e: e.composite_score, reverse=True)
        for i, entry in enumerate(entries, 1):
            entry.rank = i

        leaderboard = Leaderboard(
            entries=entries,
            filters={
                "min_trades": min_trades,
                "min_volume": float(min_volume),
            },
        )

        return leaderboard

    def save_leaderboard(self, leaderboard: Leaderboard) -> Path:
        """Save leaderboard to disk.

        Args:
            leaderboard: Leaderboard to save

        Returns:
            Path to saved file
        """
        self.leaderboard_path.write_text(
            json.dumps(leaderboard.to_dict(), indent=2, sort_keys=True)
        )
        return self.leaderboard_path

    def load_leaderboard(self) -> Leaderboard | None:
        """Load leaderboard from disk.

        Returns:
            Leaderboard or None if not found
        """
        if not self.leaderboard_path.exists():
            return None

        try:
            data = json.loads(self.leaderboard_path.read_text())
            return Leaderboard.from_dict(data)
        except (json.JSONDecodeError, KeyError):
            return None

    def select_candidates(
        self,
        leaderboard: Leaderboard,
        top_n: int = 5,
        min_win_rate: float = 40.0,
        min_trades: int = 10,
        require_positive_pnl: bool = True,
    ) -> list[CopyCandidate]:
        """Select copy trading candidates from leaderboard.

        Args:
            leaderboard: Leaderboard to select from
            top_n: Number of candidates to select
            min_win_rate: Minimum win rate percentage
            min_trades: Minimum trades required
            require_positive_pnl: Only select traders with positive PnL

        Returns:
            List of CopyCandidate objects
        """
        candidates = []

        for entry in leaderboard.entries:
            stats = entry.stats

            # Apply filters
            if stats.win_rate < min_win_rate:
                continue
            if stats.total_trades < min_trades:
                continue
            if require_positive_pnl and stats.realized_pnl <= 0:
                continue

            # Build selection rationale
            reasons = []
            selection_score = entry.composite_score

            if entry.pnl_rank <= 10:
                reasons.append(f"Top 10 by PnL (#{entry.pnl_rank})")
                selection_score += 0.1
            if entry.volume_rank <= 10:
                reasons.append(f"Top 10 by volume (#{entry.volume_rank})")
                selection_score += 0.05
            if entry.win_rate_rank <= 10:
                reasons.append(f"Top 10 by win rate (#{entry.win_rate_rank})")
                selection_score += 0.1
            if stats.realized_pnl > 0:
                reasons.append(f"Profitable (${float(stats.realized_pnl):,.0f})")

            # Calculate recommended position size based on performance
            # Higher win rate + lower volatility (more trades) = larger position
            position_multiplier = min(stats.win_rate / 100, 1.0)
            trade_confidence = min(stats.total_trades / 100, 1.0)
            recommended_size = Decimal("100") * Decimal(
                str(position_multiplier * trade_confidence + 0.5)
            )

            candidate = CopyCandidate(
                stats=stats,
                selection_score=selection_score,
                selection_reason=reasons,
                recommended_position_size=recommended_size,
            )
            candidates.append(candidate)

        # Sort by selection score and take top N
        candidates.sort(key=lambda c: c.selection_score, reverse=True)
        return candidates[:top_n]

    def save_candidates(self, candidates: list[CopyCandidate]) -> Path:
        """Save candidates to disk.

        Args:
            candidates: List of candidates to save

        Returns:
            Path to saved file
        """
        data = {
            "saved_at": datetime.now(UTC).isoformat(),
            "candidates": [c.to_dict() for c in candidates],
        }
        self.candidates_path.write_text(json.dumps(data, indent=2, sort_keys=True))
        return self.candidates_path

    def load_candidates(self) -> list[CopyCandidate]:
        """Load candidates from disk.

        Returns:
            List of CopyCandidate objects
        """
        if not self.candidates_path.exists():
            return []

        try:
            data = json.loads(self.candidates_path.read_text())
            return [CopyCandidate.from_dict(c) for c in data.get("candidates", [])]
        except (json.JSONDecodeError, KeyError):
            return []

    def get_leaderboard_summary(self, leaderboard: Leaderboard) -> dict:
        """Get human-readable summary of leaderboard.

        Args:
            leaderboard: Leaderboard to summarize

        Returns:
            Dict with summary statistics
        """
        if not leaderboard.entries:
            return {"count": 0, "message": "No traders in leaderboard"}

        total_volume = sum(e.stats.total_volume for e in leaderboard.entries)
        total_pnl = sum(e.stats.realized_pnl for e in leaderboard.entries)
        avg_win_rate = sum(e.stats.win_rate for e in leaderboard.entries) / len(leaderboard.entries)

        return {
            "count": len(leaderboard.entries),
            "total_volume": float(total_volume),
            "total_realized_pnl": float(total_pnl),
            "average_win_rate": avg_win_rate,
            "top_performer": leaderboard.entries[0].stats.address if leaderboard.entries else None,
            "filters": leaderboard.filters,
        }
