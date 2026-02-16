"""Trader profiling and leaderboard integration for Polymarket.

Fetches top traders from Polymarket APIs, tracks their performance,
and enables paper-copy trading of top performers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING

import httpx

from .endpoints import DATA_BASE, GAMMA_BASE

if TYPE_CHECKING:
    from collections.abc import Sequence


# Constants
DEFAULT_DATA_DIR = Path("data/trader_profiles")
TRADERS_FILE = "traders.json"
TRADER_FILLS_DIR = "fills"
TRADER_NAV_DIR = "nav"
DEFAULT_TOP_N = 50  # Number of candidate traders to track
DEFAULT_COPY_K = 5  # Number of top traders to paper-copy


def _data_client(timeout: float = 30.0) -> httpx.Client:
    """Create HTTP client for Data API."""
    return httpx.Client(
        base_url=DATA_BASE, timeout=timeout, headers={"User-Agent": "polymarket-bot/0.1"}
    )


def _gamma_client(timeout: float = 30.0) -> httpx.Client:
    """Create HTTP client for Gamma API."""
    return httpx.Client(
        base_url=GAMMA_BASE, timeout=timeout, headers={"User-Agent": "polymarket-bot/0.1"}
    )


@dataclass(frozen=True)
class TraderProfile:
    """Profile of a Polymarket trader.

    Attributes:
        address: Wallet address (0x...)
        username: Display name (if available)
        pnl_lifetime: Lifetime PnL in USD
        pnl_30d: 30-day PnL in USD
        pnl_7d: 7-day PnL in USD
        volume_lifetime: Lifetime volume in USD
        markets_traded: Number of markets traded
        rank: Leaderboard rank (if from leaderboard)
        source: Where this trader was discovered ('leaderboard', 'manual', etc.)
        discovered_at: ISO timestamp when first discovered
        last_updated: ISO timestamp of last update
        tags: List of tags (e.g., 'top_performer', 'consistent', 'high_volume')
    """

    address: str
    username: str | None = None
    pnl_lifetime: Decimal = field(default_factory=lambda: Decimal("0"))
    pnl_30d: Decimal = field(default_factory=lambda: Decimal("0"))
    pnl_7d: Decimal = field(default_factory=lambda: Decimal("0"))
    volume_lifetime: Decimal = field(default_factory=lambda: Decimal("0"))
    markets_traded: int = 0
    rank: int | None = None
    source: str = "unknown"
    discovered_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_updated: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "address": self.address,
            "username": self.username,
            "pnl_lifetime": float(self.pnl_lifetime),
            "pnl_30d": float(self.pnl_30d),
            "pnl_7d": float(self.pnl_7d),
            "volume_lifetime": float(self.volume_lifetime),
            "markets_traded": self.markets_traded,
            "rank": self.rank,
            "source": self.source,
            "discovered_at": self.discovered_at,
            "last_updated": self.last_updated,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: dict) -> TraderProfile:
        """Create from dictionary."""
        return cls(
            address=data["address"],
            username=data.get("username"),
            pnl_lifetime=Decimal(str(data.get("pnl_lifetime", 0))),
            pnl_30d=Decimal(str(data.get("pnl_30d", 0))),
            pnl_7d=Decimal(str(data.get("pnl_7d", 0))),
            volume_lifetime=Decimal(str(data.get("volume_lifetime", 0))),
            markets_traded=data.get("markets_traded", 0),
            rank=data.get("rank"),
            source=data.get("source", "unknown"),
            discovered_at=data.get("discovered_at", datetime.now(timezone.utc).isoformat()),
            last_updated=data.get("last_updated", datetime.now(timezone.utc).isoformat()),
            tags=data.get("tags", []),
        )

    @property
    def pnl_30d_roi(self) -> Decimal:
        """Calculate 30-day ROI based on volume (approximation)."""
        if self.volume_lifetime == 0:
            return Decimal("0")
        # Rough approximation: assume 30d volume is ~1/12 of lifetime
        estimated_30d_volume = self.volume_lifetime / 12
        if estimated_30d_volume == 0:
            return Decimal("0")
        return self.pnl_30d / estimated_30d_volume

    @property
    def consistency_score(self) -> float:
        """Score based on consistency of returns (0-100).

        Higher score if 7d and 30d PnL are both positive and proportional.
        """
        score = 0.0

        # Both 7d and 30d positive
        if self.pnl_7d > 0 and self.pnl_30d > 0:
            score += 50.0

            # 7d is roughly 1/4 of 30d (consistent weekly performance)
            if self.pnl_30d > 0:
                ratio = float(self.pnl_7d * 4 / self.pnl_30d)
                if 0.5 <= ratio <= 2.0:  # Within 2x of expected
                    score += 50.0

        return min(score, 100.0)


@dataclass
class TraderScore:
    """Computed score for a trader based on multiple factors.

    Attributes:
        address: Trader address
        total_score: Composite score (0-100)
        pnl_score: Score based on PnL performance (0-100)
        consistency_score: Score based on consistent returns (0-100)
        volume_score: Score based on trading volume (0-100)
        diversity_score: Score based on market diversity (0-100)
        computed_at: ISO timestamp
    """

    address: str
    total_score: float = 0.0
    pnl_score: float = 0.0
    consistency_score: float = 0.0
    volume_score: float = 0.0
    diversity_score: float = 0.0
    computed_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "address": self.address,
            "total_score": self.total_score,
            "pnl_score": self.pnl_score,
            "consistency_score": self.consistency_score,
            "volume_score": self.volume_score,
            "diversity_score": self.diversity_score,
            "computed_at": self.computed_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> TraderScore:
        """Create from dictionary."""
        return cls(
            address=data["address"],
            total_score=data.get("total_score", 0.0),
            pnl_score=data.get("pnl_score", 0.0),
            consistency_score=data.get("consistency_score", 0.0),
            volume_score=data.get("volume_score", 0.0),
            diversity_score=data.get("diversity_score", 0.0),
            computed_at=data.get("computed_at", datetime.now(timezone.utc).isoformat()),
        )


class TraderProfiler:
    """Manages trader discovery, scoring, and persistence.

    Features:
    - Fetch top traders from Polymarket leaderboard
    - Score traders based on multiple factors
    - Persist trader profiles and history
    - Select top-K traders for paper-copy trading
    """

    data_dir: Path
    traders: dict[str, TraderProfile]
    scores: dict[str, TraderScore]

    def __init__(self, data_dir: Path | str | None = None) -> None:
        """Initialize trader profiler.

        Args:
            data_dir: Directory for trader data storage
        """
        self.data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / TRADER_FILLS_DIR).mkdir(exist_ok=True)
        (self.data_dir / TRADER_NAV_DIR).mkdir(exist_ok=True)

        self.traders: dict[str, TraderProfile] = {}
        self.scores: dict[str, TraderScore] = {}
        self._load_traders()

    @property
    def traders_path(self) -> Path:
        """Path to traders database file."""
        return self.data_dir / TRADERS_FILE

    def _load_traders(self) -> None:
        """Load traders from disk."""
        if not self.traders_path.exists():
            return

        try:
            data = json.loads(self.traders_path.read_text())
            for addr, trader_data in data.get("traders", {}).items():
                self.traders[addr] = TraderProfile.from_dict(trader_data)
            for addr, score_data in data.get("scores", {}).items():
                self.scores[addr] = TraderScore.from_dict(score_data)
        except (json.JSONDecodeError, KeyError, ValueError):
            pass

    def save_traders(self) -> Path:
        """Save traders to disk.

        Returns:
            Path to saved file
        """
        data = {
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "traders": {addr: t.to_dict() for addr, t in self.traders.items()},
            "scores": {addr: s.to_dict() for addr, s in self.scores.items()},
        }
        self.traders_path.write_text(json.dumps(data, indent=2, sort_keys=True))
        return self.traders_path

    def fetch_leaderboard(
        self,
        timeframe: str = "30d",
        limit: int = DEFAULT_TOP_N,
    ) -> list[TraderProfile]:
        """Fetch top traders from Polymarket leaderboard.

        Args:
            timeframe: '7d', '30d', or 'lifetime'
            limit: Number of traders to fetch

        Returns:
            List of TraderProfile objects
        """
        profiles = []

        try:
            with _data_client() as client:
                # Fetch leaderboard data
                params = {
                    "timeframe": timeframe,
                    "limit": limit,
                }
                response = client.get("/leaderboard", params=params)
                response.raise_for_status()
                data = response.json()

                for idx, entry in enumerate(data.get("data", []), 1):
                    profile = TraderProfile(
                        address=entry.get("address", ""),
                        username=entry.get("username"),
                        pnl_lifetime=Decimal(str(entry.get("profit_lifetime", 0))),
                        pnl_30d=Decimal(str(entry.get("profit_30d", 0))),
                        pnl_7d=Decimal(str(entry.get("profit_7d", 0))),
                        volume_lifetime=Decimal(str(entry.get("volume_lifetime", 0))),
                        markets_traded=entry.get("markets_traded", 0),
                        rank=idx,
                        source=f"leaderboard_{timeframe}",
                    )
                    if profile.address:
                        profiles.append(profile)

        except httpx.HTTPError as e:
            print(f"Error fetching leaderboard: {e}")

        return profiles

    def fetch_user_stats(self, address: str) -> TraderProfile | None:
        """Fetch detailed stats for a specific user.

        Args:
            address: Wallet address

        Returns:
            TraderProfile or None if not found
        """
        try:
            with _data_client() as client:
                response = client.get(f"/user/{address}")
                if response.status_code == 404:
                    return None
                response.raise_for_status()
                data = response.json()

                return TraderProfile(
                    address=address,
                    username=data.get("username"),
                    pnl_lifetime=Decimal(str(data.get("profit_lifetime", 0))),
                    pnl_30d=Decimal(str(data.get("profit_30d", 0))),
                    pnl_7d=Decimal(str(data.get("profit_7d", 0))),
                    volume_lifetime=Decimal(str(data.get("volume_lifetime", 0))),
                    markets_traded=data.get("markets_traded", 0),
                    source="api_lookup",
                )

        except httpx.HTTPError:
            return None

    def add_or_update_trader(self, profile: TraderProfile) -> None:
        """Add or update a trader profile.

        Args:
            profile: TraderProfile to add/update
        """
        address = profile.address.lower()

        if address in self.traders:
            # Merge with existing, preserving discovered_at and preferring existing non-null values
            existing = self.traders[address]
            profile = TraderProfile(
                address=address,
                username=existing.username or profile.username,
                pnl_lifetime=profile.pnl_lifetime
                if profile.pnl_lifetime != Decimal("0")
                else existing.pnl_lifetime,
                pnl_30d=profile.pnl_30d if profile.pnl_30d != Decimal("0") else existing.pnl_30d,
                pnl_7d=profile.pnl_7d if profile.pnl_7d != Decimal("0") else existing.pnl_7d,
                volume_lifetime=profile.volume_lifetime
                if profile.volume_lifetime != Decimal("0")
                else existing.volume_lifetime,
                markets_traded=max(profile.markets_traded, existing.markets_traded),
                rank=existing.rank or profile.rank,
                source=existing.source if existing.source != "unknown" else profile.source,
                discovered_at=existing.discovered_at,
                last_updated=datetime.now(timezone.utc).isoformat(),
                tags=list(set(existing.tags + profile.tags)),
            )

        self.traders[address] = profile

    def discover_traders(
        self,
        top_n: int = DEFAULT_TOP_N,
        include_leaderboard: bool = True,
        manual_addresses: Sequence[str] | None = None,
    ) -> list[TraderProfile]:
        """Discover and add candidate traders.

        Args:
            top_n: Number of traders to fetch from leaderboard
            include_leaderboard: Whether to fetch from leaderboard API
            manual_addresses: Optional list of addresses to add manually

        Returns:
            List of newly discovered/updated traders
        """
        discovered = []

        if include_leaderboard:
            # Fetch from multiple timeframes for better coverage
            for timeframe in ["30d", "7d", "lifetime"]:
                profiles = self.fetch_leaderboard(timeframe=timeframe, limit=top_n)
                for profile in profiles:
                    self.add_or_update_trader(profile)
                    discovered.append(profile)

        if manual_addresses:
            for address in manual_addresses:
                address = address.lower()
                if address not in self.traders:
                    profile = self.fetch_user_stats(address)
                    if profile:
                        self.add_or_update_trader(profile)
                        discovered.append(profile)

        self.save_traders()
        return discovered

    def compute_scores(
        self,
        min_volume: Decimal = Decimal("1000"),
        min_markets: int = 3,
    ) -> list[TraderScore]:
        """Compute scores for all tracked traders.

        Scoring factors:
        - PnL performance (30d and 7d)
        - Consistency between timeframes
        - Volume (filters out low-activity)
        - Market diversity

        Args:
            min_volume: Minimum lifetime volume to be considered
            min_markets: Minimum markets traded to be considered

        Returns:
            List of TraderScore objects sorted by total_score desc
        """
        scores = []

        for address, profile in self.traders.items():
            # Skip traders below minimum thresholds
            if profile.volume_lifetime < min_volume:
                continue
            if profile.markets_traded < min_markets:
                continue

            score = TraderScore(address=address)

            # PnL score (0-100): based on 30d PnL relative to volume
            if profile.volume_lifetime > 0:
                pnl_ratio = float(profile.pnl_30d / (profile.volume_lifetime / 12))
                # Scale: 10% monthly ROI = 100 points
                score.pnl_score = min(max(pnl_ratio * 1000, 0), 100)

            # Consistency score (0-100)
            score.consistency_score = profile.consistency_score

            # Volume score (0-100): log scale
            if profile.volume_lifetime > 0:
                import math

                volume_millions = float(profile.volume_lifetime) / 1_000_000
                score.volume_score = min(max(math.log10(volume_millions + 0.1) * 25, 0), 100)

            # Diversity score (0-100): more markets = higher score
            score.diversity_score = min(profile.markets_traded * 2, 100)

            # Total score: weighted composite
            score.total_score = (
                score.pnl_score * 0.4
                + score.consistency_score * 0.3
                + score.volume_score * 0.2
                + score.diversity_score * 0.1
            )

            self.scores[address] = score
            scores.append(score)

        # Sort by total score descending
        scores.sort(key=lambda s: s.total_score, reverse=True)
        self.save_traders()

        return scores

    def get_top_traders(
        self,
        k: int = DEFAULT_COPY_K,
        min_score: float = 0.0,
    ) -> list[tuple[TraderProfile, TraderScore]]:
        """Get top-K traders for paper-copy trading.

        Args:
            k: Number of top traders to return
            min_score: Minimum total score to be included

        Returns:
            List of (TraderProfile, TraderScore) tuples
        """
        # Recompute scores if needed
        if not self.scores:
            self.compute_scores()

        # Sort by score and filter
        sorted_scores = sorted(
            self.scores.values(),
            key=lambda s: s.total_score,
            reverse=True,
        )

        result = []
        for score in sorted_scores[:k]:
            if score.total_score >= min_score:
                profile = self.traders.get(score.address)
                if profile:
                    result.append((profile, score))

        return result

    def get_trader_fills_path(self, address: str) -> Path:
        """Get path to fills file for a specific trader."""
        return self.data_dir / TRADER_FILLS_DIR / f"{address.lower()}.jsonl"

    def get_trader_nav_path(self, address: str) -> Path:
        """Get path to NAV history file for a specific trader."""
        return self.data_dir / TRADER_NAV_DIR / f"{address.lower()}.jsonl"

    def get_all_traders(self) -> list[TraderProfile]:
        """Get all tracked traders."""
        return list(self.traders.values())

    def get_trader(self, address: str) -> TraderProfile | None:
        """Get a specific trader by address."""
        return self.traders.get(address.lower())
