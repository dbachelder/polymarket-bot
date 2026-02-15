"""BTC interval orderbook-imbalance strategy.

Hypothesis: On BTC Up/Down interval markets (5m + 15m), short-horizon CLOB
microstructure contains predictive info beyond the displayed mid price.
Specifically, sustained YES-side depth imbalance (bid depth >> ask depth)
indicates informed buying pressure and increases P(UP) vs market-implied probability.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ImbalanceFeatures:
    """Features extracted from orderbook at a specific timestamp."""

    timestamp: datetime
    market_id: str
    market_title: str
    interval_minutes: int  # 5 or 15

    # Top-of-book prices
    best_bid_yes: float
    best_ask_yes: float
    spread: float
    mid_yes: float

    # Depth imbalance at k levels (YES side)
    # imbalance_k = sum(bid_sz[1..k]) / (sum(bid_sz[1..k]) + sum(ask_sz[1..k]))
    imbalance_1: float
    imbalance_3: float
    imbalance_5: float

    # Optional: short-term change in mid
    mid_delta: float | None = None  # Change from prior snapshot

    # Market outcome (for labeling)
    outcome_up: bool | None = None  # True if UP (end >= start)


def _to_float(val: str | float | int) -> float:
    """Convert a value to float."""
    return float(val) if not isinstance(val, float) else val


def _compute_imbalance_at_levels(
    bids: list[dict],
    asks: list[dict],
    levels: int,
) -> float | None:
    """Compute YES-side depth imbalance at top N levels.

    imbalance = sum(bid_sz[1..k]) / (sum(bid_sz[1..k]) + sum(ask_sz[1..k]))
    Range: 0 (all ask) to 1 (all bid), 0.5 = balanced

    Args:
        bids: List of bid orders with 'price' and 'size'
        asks: List of ask orders with 'price' and 'size'
        levels: Number of book levels to include

    Returns:
        Imbalance ratio or None if no depth
    """
    # Sort bids descending (highest first), asks ascending (lowest first)
    sorted_bids = sorted(bids, key=lambda x: _to_float(x["price"]), reverse=True)
    sorted_asks = sorted(asks, key=lambda x: _to_float(x["price"]))

    top_bids = sorted_bids[:levels]
    top_asks = sorted_asks[:levels]

    bid_depth = sum(_to_float(b["size"]) for b in top_bids)
    ask_depth = sum(_to_float(a["size"]) for a in top_asks)

    total_depth = bid_depth + ask_depth
    if total_depth == 0:
        return None

    return bid_depth / total_depth


def extract_features_from_market(
    market_data: dict[str, Any],
    timestamp: datetime | None = None,
    prior_mid: float | None = None,
) -> ImbalanceFeatures | None:
    """Extract imbalance features from a single market's book data.

    Args:
        market_data: Market dict with 'books' key containing 'yes' and 'no' books
        timestamp: Override timestamp (uses market_data timestamp if None)
        prior_mid: Previous mid price for computing delta

    Returns:
        ImbalanceFeatures or None if data is insufficient
    """
    books = market_data.get("books", {})
    yes_book = books.get("yes", {})

    yes_bids = yes_book.get("bids", [])
    yes_asks = yes_book.get("asks", [])

    if not yes_bids or not yes_asks:
        return None

    # Top-of-book prices
    sorted_bids = sorted(yes_bids, key=lambda x: _to_float(x["price"]), reverse=True)
    sorted_asks = sorted(yes_asks, key=lambda x: _to_float(x["price"]))

    best_bid_yes = _to_float(sorted_bids[0]["price"]) if sorted_bids else None
    best_ask_yes = _to_float(sorted_asks[0]["price"]) if sorted_asks else None

    if best_bid_yes is None or best_ask_yes is None:
        return None

    spread = best_ask_yes - best_bid_yes
    mid_yes = (best_bid_yes + best_ask_yes) / 2

    # Compute imbalances at different levels
    imbalance_1 = _compute_imbalance_at_levels(yes_bids, yes_asks, 1)
    imbalance_3 = _compute_imbalance_at_levels(yes_bids, yes_asks, 3)
    imbalance_5 = _compute_imbalance_at_levels(yes_bids, yes_asks, 5)

    if imbalance_1 is None or imbalance_3 is None or imbalance_5 is None:
        return None

    # Compute mid delta if prior mid provided
    mid_delta = None
    if prior_mid is not None:
        mid_delta = mid_yes - prior_mid

    # Determine interval from title
    title = market_data.get("title", market_data.get("question", "")).lower()
    interval_minutes = 15  # Default
    # Check 15m first to avoid matching "5m" inside "15m"
    if "15m" in title or "15 minute" in title:
        interval_minutes = 15
    elif "5m" in title or "5 minute" in title:
        interval_minutes = 5

    # Parse timestamp
    ts = timestamp
    if ts is None:
        ts_str = market_data.get("timestamp") or market_data.get("generated_at")
        if ts_str:
            try:
                ts = datetime.fromisoformat(ts_str)
            except (ValueError, TypeError):
                ts = datetime.now(UTC)
        else:
            ts = datetime.now(UTC)

    return ImbalanceFeatures(
        timestamp=ts,
        market_id=market_data.get("market_id", ""),
        market_title=market_data.get("title", market_data.get("question", "Unknown")),
        interval_minutes=interval_minutes,
        best_bid_yes=best_bid_yes,
        best_ask_yes=best_ask_yes,
        spread=spread,
        mid_yes=mid_yes,
        imbalance_1=imbalance_1,
        imbalance_3=imbalance_3,
        imbalance_5=imbalance_5,
        mid_delta=mid_delta,
    )


def extract_features_from_snapshot(
    snapshot_path: Path,
    target_market_substring: str = "bitcoin",
) -> list[ImbalanceFeatures]:
    """Extract imbalance features from all matching markets in a snapshot.

    Args:
        snapshot_path: Path to snapshot JSON file
        target_market_substring: Filter markets by this substring

    Returns:
        List of ImbalanceFeatures
    """
    data = json.loads(snapshot_path.read_text())
    markets = data.get("markets", [])

    # Get timestamp from snapshot
    ts_str = data.get("generated_at")
    snapshot_ts = None
    if ts_str:
        try:
            snapshot_ts = datetime.fromisoformat(ts_str)
        except (ValueError, TypeError):
            pass

    features = []
    for market in markets:
        title = market.get("title", market.get("question", ""))
        if target_market_substring.lower() not in title.lower():
            continue

        feat = extract_features_from_market(market, timestamp=snapshot_ts)
        if feat:
            features.append(feat)

    return features


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


@dataclass(frozen=True)
class TradeDecision:
    """A trading decision from the strategy."""

    timestamp: datetime
    market_id: str
    market_title: str
    decision: str  # 'UP', 'DOWN', or 'NO_TRADE'
    imbalance_k: int  # Which k was used
    imbalance_value: float
    mid_yes: float
    entry_price: float  # Pessimistic fill price
    confidence: float  # How extreme the imbalance is


@dataclass(frozen=True)
class BacktestResult:
    """Results from a backtest run."""

    trades: list[TradeDecision]
    metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "trades": [
                {
                    "timestamp": t.timestamp.isoformat(),
                    "market_id": t.market_id,
                    "market_title": t.market_title,
                    "decision": t.decision,
                    "imbalance_k": t.imbalance_k,
                    "imbalance_value": t.imbalance_value,
                    "mid_yes": t.mid_yes,
                    "entry_price": t.entry_price,
                    "confidence": t.confidence,
                }
                for t in self.trades
            ],
            "metrics": self.metrics,
        }


def run_backtest(
    snapshots: list[Path],
    k: int = 3,
    theta: float = 0.70,
    p_max: float = 0.65,
    target_market_substring: str = "bitcoin",
) -> BacktestResult:
    """Run backtest on historical snapshots.

    Decision rule (one-parameter family):
    - go UP if imbalance_k > theta and midYes < pMax (avoid paying 0.70+)
    - go DOWN if imbalance_k < 1-theta and midYes > 1-pMax (symmetry)
    - otherwise no-trade

    Entry price assumption: pessimistic (buy at ask for chosen side)
    Exit: hold to expiry (outcome determined by market resolution)

    Args:
        snapshots: List of snapshot file paths
        k: Number of depth levels for imbalance (1, 3, or 5)
        theta: Imbalance threshold (0.5 to 1.0)
        p_max: Max price to pay for position (avoid expensive skewed contracts)
        target_market_substring: Filter for BTC markets

    Returns:
        BacktestResult with trades and metrics
    """
    trades: list[TradeDecision] = []

    for snap_path in snapshots:
        features_list = extract_features_from_snapshot(snap_path, target_market_substring)

        for feat in features_list:
            # Select imbalance based on k
            if k == 1:
                imbalance = feat.imbalance_1
            elif k == 3:
                imbalance = feat.imbalance_3
            elif k == 5:
                imbalance = feat.imbalance_5
            else:
                imbalance = feat.imbalance_3  # Default

            # Decision logic
            decision = "NO_TRADE"
            entry_price = feat.mid_yes
            confidence = abs(imbalance - 0.5) * 2  # Scale to 0-1

            # Go UP if imbalance > theta and mid < p_max
            if imbalance > theta and feat.mid_yes < p_max:
                decision = "UP"
                entry_price = feat.best_ask_yes  # Pessimistic: pay the ask

            # Go DOWN if imbalance < 1-theta and mid > 1-p_max
            elif imbalance < (1 - theta) and feat.mid_yes > (1 - p_max):
                decision = "DOWN"
                entry_price = 1.0 - feat.best_ask_yes  # NO price

            if decision != "NO_TRADE":
                trades.append(
                    TradeDecision(
                        timestamp=feat.timestamp,
                        market_id=feat.market_id,
                        market_title=feat.market_title,
                        decision=decision,
                        imbalance_k=k,
                        imbalance_value=imbalance,
                        mid_yes=feat.mid_yes,
                        entry_price=entry_price,
                        confidence=confidence,
                    )
                )

    # Compute metrics
    up_trades = [t for t in trades if t.decision == "UP"]
    down_trades = [t for t in trades if t.decision == "DOWN"]

    metrics = {
        "total_trades": len(trades),
        "up_trades": len(up_trades),
        "down_trades": len(down_trades),
        "avg_confidence": np.mean([t.confidence for t in trades]) if trades else 0,
        "avg_entry_price": np.mean([t.entry_price for t in trades]) if trades else 0,
        "params": {
            "k": k,
            "theta": theta,
            "p_max": p_max,
        },
    }

    return BacktestResult(trades=trades, metrics=metrics)


def parameter_sweep(
    snapshots: list[Path],
    k_values: list[int] | None = None,
    theta_values: list[float] | None = None,
    p_max_values: list[float] | None = None,
    target_market_substring: str = "bitcoin",
) -> list[dict[str, Any]]:
    """Sweep parameter space and return results for each combination.

    Args:
        snapshots: List of snapshot file paths
        k_values: List of k values to test (default: [1, 3, 5])
        theta_values: List of theta values (default: [0.60, 0.65, 0.70, 0.75])
        p_max_values: List of p_max values (default: [0.60, 0.65])
        target_market_substring: Filter for BTC markets

    Returns:
        List of result dicts sorted by trade count (descending)
    """
    if k_values is None:
        k_values = [1, 3, 5]
    if theta_values is None:
        theta_values = [0.60, 0.65, 0.70, 0.75]
    if p_max_values is None:
        p_max_values = [0.60, 0.65]

    results = []

    for k in k_values:
        for theta in theta_values:
            for p_max in p_max_values:
                result = run_backtest(
                    snapshots=snapshots,
                    k=k,
                    theta=theta,
                    p_max=p_max,
                    target_market_substring=target_market_substring,
                )

                results.append({
                    "params": {"k": k, "theta": theta, "p_max": p_max},
                    "metrics": result.metrics,
                })

    # Sort by trade count descending
    results.sort(key=lambda x: x["metrics"]["total_trades"], reverse=True)

    return results
