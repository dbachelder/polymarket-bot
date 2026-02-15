"""Hourly digest report for Polymarket 15m crypto snapshots.

Reports on:
- Collector health (freshness, snapshots/hour, backoff evidence)
- BTC 15m microstructure (spread, depth, imbalance)
- Paper strategy metric (simple momentum signal)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CollectorHealth:
    """Collector health metrics."""

    latest_snapshot_at: str | None
    freshness_seconds: float | None
    snapshots_last_hour: int
    expected_snapshots: int
    capture_rate_pct: float
    backoff_evidence: bool
    snapshot_dir: str


@dataclass(frozen=True)
class MicrostructureStats:
    """Order book microstructure stats."""

    market: str
    best_bid: float | None
    best_ask: float | None
    spread: float | None
    spread_bps: float | None
    best_bid_depth: float
    best_ask_depth: float
    depth_imbalance: float | None


@dataclass(frozen=True)
class MomentumSignal:
    """Simple momentum-based paper strategy signal."""

    signal: str  # "long", "short", "neutral"
    confidence: float  # 0-1 scale
    mid_price_change_1h: float | None
    volume_surge: float | None
    reasoning: str


@dataclass(frozen=True)
class HourlyDigest:
    """Complete hourly digest report."""

    generated_at: str
    collector_health: CollectorHealth
    btc_microstructure: MicrostructureStats
    paper_strategy: MomentumSignal

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _parse_snapshot_timestamp(filename: str) -> datetime | None:
    """Parse timestamp from snapshot filename."""
    try:
        # Format: snapshot_15m_20260215T053045Z.json
        if not filename.startswith("snapshot_15m_"):
            return None
        ts_str = filename.replace("snapshot_15m_", "").replace(".json", "")
        return datetime.strptime(ts_str, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
    except (ValueError, IndexError):
        return None


def analyze_collector_health(
    data_dir: Path,
    interval_seconds: float = 5.0,
    now: datetime | None = None,
) -> CollectorHealth:
    """Analyze collector health from snapshot files.

    Args:
        data_dir: Directory containing snapshot files
        interval_seconds: Expected collection interval (default 5s)
        now: Reference time (default: UTC now)

    Returns:
        CollectorHealth metrics
    """
    if now is None:
        now = datetime.now(UTC)

    cutoff = now - timedelta(hours=1)
    snapshots_last_hour = 0
    latest_ts: datetime | None = None
    gap_evidence = False

    # Check for latest_15m.json pointer first
    latest_pointer = data_dir / "latest_15m.json"
    if latest_pointer.exists():
        try:
            ptr_data = json.loads(latest_pointer.read_text())
            ptr_path = Path(ptr_data.get("path", ""))
            if ptr_path.exists():
                ptr_ts = _parse_snapshot_timestamp(ptr_path.name)
                if ptr_ts:
                    latest_ts = ptr_ts
        except (json.JSONDecodeError, OSError):
            pass

    # Scan all 15m snapshots
    timestamps: list[datetime] = []
    for p in data_dir.glob("snapshot_15m_*.json"):
        ts = _parse_snapshot_timestamp(p.name)
        if ts:
            timestamps.append(ts)
            if latest_ts is None or ts > latest_ts:
                latest_ts = ts
            if ts >= cutoff:
                snapshots_last_hour += 1

    timestamps.sort()

    # Detect gaps that suggest backoff
    if len(timestamps) >= 2:
        expected_gap = interval_seconds
        max_gap = expected_gap * 3  # 3x expected = likely backoff
        for i in range(1, len(timestamps)):
            gap = (timestamps[i] - timestamps[i - 1]).total_seconds()
            if gap > max_gap:
                gap_evidence = True
                break

    freshness_seconds = None
    if latest_ts:
        freshness_seconds = (now - latest_ts).total_seconds()

    # Expected snapshots in an hour at given interval
    expected_snapshots = int(3600 / interval_seconds)
    capture_rate_pct = min(100.0, (snapshots_last_hour / expected_snapshots) * 100) if expected_snapshots > 0 else 0.0

    return CollectorHealth(
        latest_snapshot_at=latest_ts.isoformat() if latest_ts else None,
        freshness_seconds=freshness_seconds,
        snapshots_last_hour=snapshots_last_hour,
        expected_snapshots=expected_snapshots,
        capture_rate_pct=capture_rate_pct,
        backoff_evidence=gap_evidence,
        snapshot_dir=str(data_dir.absolute()),
    )


def analyze_btc_microstructure(snapshot_path: Path) -> MicrostructureStats:
    """Analyze BTC 15m microstructure from a snapshot.

    Args:
        snapshot_path: Path to snapshot JSON file

    Returns:
        MicrostructureStats for BTC market
    """
    default = MicrostructureStats(
        market="BTC",
        best_bid=None,
        best_ask=None,
        spread=None,
        spread_bps=None,
        best_bid_depth=0.0,
        best_ask_depth=0.0,
        depth_imbalance=None,
    )

    if not snapshot_path.exists():
        return default

    try:
        data = json.loads(snapshot_path.read_text())
    except (json.JSONDecodeError, OSError):
        return default

    # Find BTC market
    btc_market = None
    for market in data.get("markets", []):
        title = market.get("title", "").lower()
        question = market.get("question", "").lower()
        if "bitcoin" in title or "bitcoin" in question or "btc" in title:
            btc_market = market
            break

    if not btc_market:
        return default

    books = btc_market.get("books", {})
    yes_book = books.get("yes", {})

    # Get best bid from bids array (bids sorted ascending, so best is last)
    bids = yes_book.get("bids", [])
    yes_bid_price = float(bids[-1].get("price", 0)) if bids else None
    yes_bid_depth = sum(float(e.get("size", 0)) for e in bids[-5:]) if bids else 0.0

    # Get best ask from asks array (asks sorted descending, so best is last)
    asks = yes_book.get("asks", [])
    yes_ask_price = float(asks[-1].get("price", 0)) if asks else None
    yes_ask_depth = sum(float(e.get("size", 0)) for e in asks[-5:]) if asks else 0.0

    # Calculate spread
    spread = None
    spread_bps = None
    if yes_bid_price is not None and yes_ask_price is not None and yes_ask_price > yes_bid_price:
        spread = yes_ask_price - yes_bid_price
        mid = (yes_bid_price + yes_ask_price) / 2
        if mid > 0:
            spread_bps = (spread / mid) * 10000

    # Depth imbalance
    depth_imbalance = None
    total_depth = yes_bid_depth + yes_ask_depth
    if total_depth > 0:
        depth_imbalance = (yes_bid_depth - yes_ask_depth) / total_depth

    return MicrostructureStats(
        market="BTC",
        best_bid=yes_bid_price,
        best_ask=yes_ask_price,
        spread=spread,
        spread_bps=spread_bps,
        best_bid_depth=yes_bid_depth,
        best_ask_depth=yes_ask_depth,
        depth_imbalance=depth_imbalance,
    )


def compute_momentum_signal(
    data_dir: Path,
    lookback_hours: int = 1,
) -> MomentumSignal:
    """Compute simple momentum-based paper strategy signal.

    Uses price change over lookback period as primary signal,
    with volume surge as confirmation.

    Args:
        data_dir: Directory containing snapshot files
        lookback_hours: Hours to look back for momentum (default 1)

    Returns:
        MomentumSignal with signal and confidence
    """
    now = datetime.now(UTC)
    cutoff = now - timedelta(hours=lookback_hours)

    # Collect snapshots with BTC data
    snapshots: list[tuple[datetime, float]] = []
    for p in data_dir.glob("snapshot_15m_*.json"):
        ts = _parse_snapshot_timestamp(p.name)
        if not ts or ts < cutoff:
            continue

        try:
            data = json.loads(p.read_text())
            for market in data.get("markets", []):
                title = market.get("title", "").lower()
                if "bitcoin" in title or "btc" in title:
                    books = market.get("books", {})
                    yes_book = books.get("yes", {})
                    bids = yes_book.get("bids", [])
                    asks = yes_book.get("asks", [])
                    if bids and asks:
                        # Bids sorted ascending (best is last), asks sorted descending (best is last)
                        bid = float(bids[-1].get("price", 0))
                        ask = float(asks[-1].get("price", 0))
                        if bid > 0 and ask > 0:
                            mid = (bid + ask) / 2
                            snapshots.append((ts, mid))
                            break
        except (json.JSONDecodeError, OSError, ValueError):
            continue

    if len(snapshots) < 2:
        return MomentumSignal(
            signal="neutral",
            confidence=0.0,
            mid_price_change_1h=None,
            volume_surge=None,
            reasoning="Insufficient data for momentum calculation",
        )

    snapshots.sort(key=lambda x: x[0])
    first_price = snapshots[0][1]
    last_price = snapshots[-1][1]
    price_change = ((last_price - first_price) / first_price) * 100 if first_price > 0 else 0.0

    # Simple momentum: positive change = long signal, negative = short
    if price_change > 1.0:
        signal = "long"
        confidence = min(1.0, abs(price_change) / 5.0)  # Cap at 5% = full confidence
        reasoning = f"BTC up {price_change:.2f}% over {lookback_hours}h (momentum)"
    elif price_change < -1.0:
        signal = "short"
        confidence = min(1.0, abs(price_change) / 5.0)
        reasoning = f"BTC down {price_change:.2f}% over {lookback_hours}h (momentum)"
    else:
        signal = "neutral"
        confidence = 0.0
        reasoning = f"BTC flat ({price_change:.2f}% over {lookback_hours}h)"

    return MomentumSignal(
        signal=signal,
        confidence=round(confidence, 2),
        mid_price_change_1h=round(price_change, 4),
        volume_surge=None,  # TODO: implement volume tracking
        reasoning=reasoning,
    )


def generate_hourly_digest(
    data_dir: Path,
    interval_seconds: float = 5.0,
) -> HourlyDigest:
    """Generate complete hourly digest report.

    Args:
        data_dir: Directory containing snapshot files
        interval_seconds: Expected collection interval (default 5s)

    Returns:
        HourlyDigest with all report components
    """
    now = datetime.now(UTC)

    # 1. Collector health
    health = analyze_collector_health(data_dir, interval_seconds, now)

    # 2. Find latest snapshot for microstructure
    latest_snapshot: Path | None = None
    latest_ts: datetime | None = None

    # Check pointer first
    latest_pointer = data_dir / "latest_15m.json"
    if latest_pointer.exists():
        try:
            ptr_data = json.loads(latest_pointer.read_text())
            ptr_path = Path(ptr_data.get("path", ""))
            if ptr_path.exists():
                latest_snapshot = ptr_path
        except (json.JSONDecodeError, OSError):
            pass

    # Fallback to newest file
    if latest_snapshot is None:
        for p in data_dir.glob("snapshot_15m_*.json"):
            ts = _parse_snapshot_timestamp(p.name)
            if ts and (latest_ts is None or ts > latest_ts):
                latest_ts = ts
                latest_snapshot = p

    # 3. BTC microstructure
    btc_stats = analyze_btc_microstructure(latest_snapshot or Path("/nonexistent"))

    # 4. Momentum signal
    momentum = compute_momentum_signal(data_dir)

    return HourlyDigest(
        generated_at=now.isoformat(),
        collector_health=health,
        btc_microstructure=btc_stats,
        paper_strategy=momentum,
    )
