"""BTC interval orderbook-imbalance strategy.

Hypothesis: On BTC Up/Down interval markets (5m + 15m), short-horizon CLOB
microstructure contains predictive info beyond the displayed mid price.
Specifically, sustained YES-side depth imbalance (bid depth >> ask depth)
indicates informed buying pressure and increases P(UP) vs market-implied probability.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
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


def _parse_snapshot_timestamp(snap_path: Path) -> datetime | None:
    """Parse timestamp from snapshot filename.

    Format: snapshot_15m_20260215T040615Z.json
    """
    try:
        ts_str = snap_path.stem.split("_")[2]
        return datetime.strptime(ts_str, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
    except (IndexError, ValueError):
        return None


def _find_future_snapshot(
    current_path: Path,
    snapshots: list[Path],
    horizon: int,
) -> Path | None:
    """Find the snapshot N steps ahead (horizon) from current.

    Args:
        current_path: Current snapshot path
        snapshots: Sorted list of all snapshot paths
        horizon: Number of snapshots to look ahead

    Returns:
        Future snapshot path or None if not found
    """
    try:
        current_idx = snapshots.index(current_path)
        future_idx = current_idx + horizon
        if future_idx < len(snapshots):
            return snapshots[future_idx]
    except ValueError:
        pass
    return None


def _get_mid_price_from_snapshot(
    snapshot_path: Path,
    market_id: str,
    target_market_substring: str,
) -> float | None:
    """Extract mid price for a specific market from a snapshot.

    Args:
        snapshot_path: Path to snapshot file
        market_id: Market ID to look for
        target_market_substring: Fallback filter if market_id not found

    Returns:
        Mid price or None if not found
    """
    try:
        data = json.loads(snapshot_path.read_text())
        markets = data.get("markets", [])

        for market in markets:
            if market.get("market_id") == market_id:
                books = market.get("books", {})
                yes_book = books.get("yes", {})
                bids = yes_book.get("bids", [])
                asks = yes_book.get("asks", [])

                if bids and asks:
                    best_bid = _to_float(
                        sorted(bids, key=lambda x: _to_float(x["price"]), reverse=True)[0]["price"]
                    )
                    best_ask = _to_float(
                        sorted(asks, key=lambda x: _to_float(x["price"]))[0]["price"]
                    )
                    return (best_bid + best_ask) / 2
                return None

        # Fallback: try by substring match
        for market in markets:
            title = market.get("title", market.get("question", ""))
            if target_market_substring.lower() in title.lower():
                books = market.get("books", {})
                yes_book = books.get("yes", {})
                bids = yes_book.get("bids", [])
                asks = yes_book.get("asks", [])

                if bids and asks:
                    best_bid = _to_float(
                        sorted(bids, key=lambda x: _to_float(x["price"]), reverse=True)[0]["price"]
                    )
                    best_ask = _to_float(
                        sorted(asks, key=lambda x: _to_float(x["price"]))[0]["price"]
                    )
                    return (best_bid + best_ask) / 2
                return None

    except (json.JSONDecodeError, FileNotFoundError, KeyError, IndexError):
        pass

    return None


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
    confidence: float  # How extreme the imbalance is (0-1)
    prob_up: float  # Model probability of UP (derived from imbalance)

    # Outcome tracking (filled in after backtest)
    outcome_up: bool | None = None  # True if market went UP
    pnl: float | None = None  # PnL for this trade
    horizon_return: float | None = None  # Return over horizon


@dataclass
class BacktestResult:
    """Results from a backtest run."""

    trades: list[TradeDecision]
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_trades: bool = True) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Args:
            include_trades: Whether to include full trade list
        """
        result = {"metrics": self.metrics}

        if include_trades:
            result["trades"] = [
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
                    "prob_up": t.prob_up,
                    "outcome_up": t.outcome_up,
                    "pnl": t.pnl,
                    "horizon_return": t.horizon_return,
                }
                for t in self.trades
            ]

        return result


def _compute_prob_up(imbalance: float, theta: float) -> float:
    """Convert imbalance to probability of UP.

    Maps imbalance [0, 1] to probability [0, 1] where:
    - imbalance = 0.5 -> prob = 0.5 (neutral)
    - imbalance > 0.5 -> prob > 0.5 (UP bias)
    - imbalance < 0.5 -> prob < 0.5 (DOWN bias)

    Uses a scaled logistic-like mapping centered at 0.5.
    """
    # Scale imbalance to [-1, 1] range centered at 0.5
    scaled = (imbalance - 0.5) * 2
    # Map to probability using sigmoid-like function
    # This gives prob=0.5 at imbalance=0.5, and stretches toward extremes
    prob = 0.5 + scaled * 0.5  # Linear mapping for simplicity
    return max(0.01, min(0.99, prob))  # Clamp to avoid 0/1


def _compute_pnl(
    decision: str,
    entry_price: float,
    exit_price: float,
    fee_bps: float = 50.0,
    slippage_bps: float = 10.0,
) -> float:
    """Compute PnL for a trade including fees and slippage.

    Args:
        decision: 'UP' or 'DOWN'
        entry_price: Entry price (0-1)
        exit_price: Exit price (0-1)
        fee_bps: Taker fee in basis points (default 50 = 0.5%)
        slippage_bps: Slippage in basis points (default 10 = 0.1%)

    Returns:
        PnL as decimal (e.g., 0.05 = 5% profit)
    """
    # Convert bps to decimal
    fee = fee_bps / 10000.0
    slippage = slippage_bps / 10000.0

    # Total cost to enter and exit
    total_cost = 2 * fee + 2 * slippage

    if decision == "UP":
        # Buy YES at entry, sell at exit
        gross_return = exit_price - entry_price
    elif decision == "DOWN":
        # Buy NO at (1-entry), sell at (1-exit)
        # Equivalent to: entry - exit (since YES price went down)
        gross_return = entry_price - exit_price
    else:
        return 0.0

    net_pnl = gross_return - total_cost
    return net_pnl


def _compute_brier_score(prob_up: float, outcome_up: bool) -> float:
    """Compute Brier score for a probability forecast.

    Brier score = (prob - outcome)^2 where outcome is 1 or 0.
    Lower is better (0 = perfect, 1 = worst).
    """
    outcome = 1.0 if outcome_up else 0.0
    return (prob_up - outcome) ** 2


def _compute_log_loss(prob_up: float, outcome_up: bool) -> float:
    """Compute log loss for a probability forecast.

    Log loss = -[outcome * log(prob) + (1-outcome) * log(1-prob)]
    Lower is better.
    """
    import math

    # Clamp probabilities to avoid log(0)
    prob = max(0.0001, min(0.9999, prob_up))
    outcome = 1.0 if outcome_up else 0.0

    return -(outcome * math.log(prob) + (1 - outcome) * math.log(1 - prob))


def run_backtest(
    snapshots: list[Path],
    k: int = 3,
    theta: float = 0.70,
    p_max: float = 0.65,
    target_market_substring: str = "bitcoin",
    horizon: int = 1,
    fee_bps: float = 50.0,
    slippage_bps: float = 10.0,
) -> BacktestResult:
    """Run backtest on historical snapshots with outcome tracking.

    Decision rule (one-parameter family):
    - go UP if imbalance_k > theta and midYes < pMax (avoid paying 0.70+)
    - go DOWN if imbalance_k < 1-theta and midYes > 1-pMax (symmetry)
    - otherwise no-trade

    Entry price assumption: pessimistic (buy at ask for chosen side)
    Exit: evaluated at horizon snapshots ahead

    Args:
        snapshots: List of snapshot file paths (must be sorted by time)
        k: Number of depth levels for imbalance (1, 3, or 5)
        theta: Imbalance threshold (0.5 to 1.0)
        p_max: Max price to pay for position (avoid expensive skewed contracts)
        target_market_substring: Filter for BTC markets
        horizon: Number of snapshots ahead to evaluate outcome (default 1)
        fee_bps: Taker fee in basis points (default 50 = 0.5%)
        slippage_bps: Slippage in basis points (default 10 = 0.1%)

    Returns:
        BacktestResult with trades and performance metrics
    """
    trades: list[TradeDecision] = []

    # Pre-sort snapshots by timestamp
    sorted_snapshots = sorted(
        snapshots, key=lambda p: _parse_snapshot_timestamp(p) or datetime.min.replace(tzinfo=UTC)
    )

    for snap_path in sorted_snapshots:
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
            prob_up = _compute_prob_up(imbalance, theta)

            # Go UP if imbalance > theta and mid < p_max
            if imbalance > theta and feat.mid_yes < p_max:
                decision = "UP"
                entry_price = feat.best_ask_yes  # Pessimistic: pay the ask

            # Go DOWN if imbalance < 1-theta and mid > 1-p_max
            elif imbalance < (1 - theta) and feat.mid_yes > (1 - p_max):
                decision = "DOWN"
                entry_price = 1.0 - feat.best_ask_yes  # NO price

            if decision != "NO_TRADE":
                # Find outcome at horizon
                future_snap = _find_future_snapshot(snap_path, sorted_snapshots, horizon)
                outcome_up = None
                horizon_return = None
                pnl = None

                if future_snap:
                    exit_price = _get_mid_price_from_snapshot(
                        future_snap, feat.market_id, target_market_substring
                    )
                    if exit_price is not None:
                        outcome_up = exit_price > feat.mid_yes
                        horizon_return = (
                            exit_price - feat.mid_yes
                            if decision == "UP"
                            else feat.mid_yes - exit_price
                        )
                        pnl = _compute_pnl(decision, entry_price, exit_price, fee_bps, slippage_bps)

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
                        prob_up=prob_up,
                        outcome_up=outcome_up,
                        pnl=pnl,
                        horizon_return=horizon_return,
                    )
                )

    # Compute metrics
    return _compute_metrics(trades, k, theta, p_max, horizon, fee_bps, slippage_bps)


def _compute_metrics(
    trades: list[TradeDecision],
    k: int,
    theta: float,
    p_max: float,
    horizon: int,
    fee_bps: float,
    slippage_bps: float,
) -> BacktestResult:
    """Compute comprehensive metrics from trades."""
    up_trades = [t for t in trades if t.decision == "UP"]
    down_trades = [t for t in trades if t.decision == "DOWN"]

    # Hit rate (accuracy) - only for trades with known outcomes
    trades_with_outcome = [t for t in trades if t.outcome_up is not None]
    correct_trades = [
        t
        for t in trades_with_outcome
        if (t.decision == "UP" and t.outcome_up) or (t.decision == "DOWN" and not t.outcome_up)
    ]
    hit_rate = len(correct_trades) / len(trades_with_outcome) if trades_with_outcome else 0.0

    # Direction-specific hit rates
    up_correct = [t for t in up_trades if t.outcome_up is True]
    down_correct = [t for t in down_trades if t.outcome_up is False]
    up_with_outcome = [t for t in up_trades if t.outcome_up is not None]
    down_with_outcome = [t for t in down_trades if t.outcome_up is not None]
    up_hit_rate = len(up_correct) / len(up_with_outcome) if up_with_outcome else 0.0
    down_hit_rate = len(down_correct) / len(down_with_outcome) if down_with_outcome else 0.0

    # Brier score (probability calibration)
    brier_scores = []
    for t in trades_with_outcome:
        if t.outcome_up is not None:
            brier_scores.append(_compute_brier_score(t.prob_up, t.outcome_up))
    avg_brier = np.mean(brier_scores) if brier_scores else 0.0

    # Log loss
    log_losses = []
    for t in trades_with_outcome:
        if t.outcome_up is not None:
            log_losses.append(_compute_log_loss(t.prob_up, t.outcome_up))
    avg_log_loss = np.mean(log_losses) if log_losses else 0.0

    # PnL metrics
    trades_with_pnl = [t for t in trades if t.pnl is not None]
    pnls = [t.pnl for t in trades_with_pnl]
    total_pnl = sum(pnls) if pnls else 0.0
    avg_pnl = np.mean(pnls) if pnls else 0.0
    sharpe = np.mean(pnls) / np.std(pnls) if pnls and np.std(pnls) > 0 else 0.0

    # Win rate (positive PnL)
    winning_trades = [t for t in trades_with_pnl if t.pnl and t.pnl > 0]
    win_rate = len(winning_trades) / len(trades_with_pnl) if trades_with_pnl else 0.0

    # Expected value per trade
    ev_per_trade = avg_pnl

    metrics = {
        # Trade counts
        "total_trades": len(trades),
        "up_trades": len(up_trades),
        "down_trades": len(down_trades),
        "trades_with_outcome": len(trades_with_outcome),
        "trades_with_pnl": len(trades_with_pnl),
        # Accuracy metrics
        "hit_rate": hit_rate,
        "up_hit_rate": up_hit_rate,
        "down_hit_rate": down_hit_rate,
        # Calibration metrics
        "avg_brier_score": avg_brier,
        "avg_log_loss": avg_log_loss,
        # PnL metrics
        "total_pnl": total_pnl,
        "avg_pnl": avg_pnl,
        "win_rate": win_rate,
        "ev_per_trade": ev_per_trade,
        "sharpe_ratio": sharpe,
        # Legacy metrics
        "avg_confidence": np.mean([t.confidence for t in trades]) if trades else 0,
        "avg_entry_price": np.mean([t.entry_price for t in trades]) if trades else 0,
        # Parameters
        "params": {
            "k": k,
            "theta": theta,
            "p_max": p_max,
            "horizon": horizon,
            "fee_bps": fee_bps,
            "slippage_bps": slippage_bps,
        },
    }

    return BacktestResult(trades=trades, metrics=metrics)


def parameter_sweep(
    snapshots: list[Path],
    k_values: list[int] | None = None,
    theta_values: list[float] | None = None,
    p_max_values: list[float] | None = None,
    target_market_substring: str = "bitcoin",
    horizon: int = 1,
    fee_bps: float = 50.0,
    slippage_bps: float = 10.0,
) -> list[dict[str, Any]]:
    """Sweep parameter space and return results for each combination.

    Args:
        snapshots: List of snapshot file paths
        k_values: List of k values to test (default: [1, 3, 5])
        theta_values: List of theta values (default: [0.60, 0.65, 0.70, 0.75])
        p_max_values: List of p_max values (default: [0.60, 0.65])
        target_market_substring: Filter for BTC markets
        horizon: Number of snapshots ahead to evaluate outcome
        fee_bps: Taker fee in basis points
        slippage_bps: Slippage in basis points

    Returns:
        List of result dicts sorted by total PnL (descending)
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
                    horizon=horizon,
                    fee_bps=fee_bps,
                    slippage_bps=slippage_bps,
                )

                results.append(
                    {
                        "params": result.metrics["params"],
                        "metrics": result.metrics,
                    }
                )

    # Sort by total PnL descending (better than trade count for evaluating performance)
    results.sort(key=lambda x: x["metrics"]["total_pnl"], reverse=True)

    return results
