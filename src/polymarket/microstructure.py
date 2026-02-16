from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Alert thresholds
DEFAULT_SPREAD_ALERT_THRESHOLD = 0.50  # Alert if spread > 50 cents
DEFAULT_EXTREME_PIN_THRESHOLD = 0.05  # Alert if best bid <= 0.05 or best ask >= 0.95
DEFAULT_DEPTH_LEVELS = 10  # Top N levels for depth calculation
DEFAULT_TIGHT_SPREAD_THRESHOLD = 0.05  # Spread <= 5 cents is considered tight
DEFAULT_CONSISTENCY_THRESHOLD = 0.10  # Consistency diff <= 0.10 is considered OK


def _to_float(val: str | float | int) -> float:
    """Convert a value to float."""
    return float(val) if not isinstance(val, float) else val


def _compute_book_metrics(
    bids: list[dict],
    asks: list[dict],
    depth_levels: int = DEFAULT_DEPTH_LEVELS,
) -> dict[str, Any]:
    """Compute microstructure metrics for a single side (YES or NO) book.

    Returns:
        Dict with best_bid, best_ask, spread, bid_depth, ask_depth, imbalance
    """
    # Sort bids descending (highest first), asks ascending (lowest first)
    sorted_bids = sorted(bids, key=lambda x: _to_float(x["price"]), reverse=True)
    sorted_asks = sorted(asks, key=lambda x: _to_float(x["price"]))

    best_bid = _to_float(sorted_bids[0]["price"]) if sorted_bids else None
    best_ask = _to_float(sorted_asks[0]["price"]) if sorted_asks else None

    spread = None
    if best_bid is not None and best_ask is not None:
        spread = best_ask - best_bid

    # Calculate depth at top N levels
    top_bids = sorted_bids[:depth_levels]
    top_asks = sorted_asks[:depth_levels]

    bid_depth = sum(_to_float(b["size"]) for b in top_bids) if top_bids else None
    ask_depth = sum(_to_float(a["size"]) for a in top_asks) if top_asks else None

    # Imbalance: (bid_depth - ask_depth) / (bid_depth + ask_depth)
    # Range: -1 (all ask) to +1 (all bid), 0 = balanced
    imbalance = None
    if bid_depth is not None and ask_depth is not None:
        total_depth = bid_depth + ask_depth
        if total_depth > 0:
            imbalance = (bid_depth - ask_depth) / total_depth

    return {
        "best_bid": best_bid,
        "best_ask": best_ask,
        "spread": spread,
        "bid_depth": bid_depth,
        "ask_depth": ask_depth,
        "imbalance": imbalance,
    }


def _compute_implied_probabilities(
    yes_metrics: dict[str, Any],
    no_metrics: dict[str, Any],
) -> dict[str, Any] | None:
    """Compute implied probabilities from YES/NO books.

    In a complete market, YES_price + NO_price should equal ~1.0 (minus fees).
    If there's no mid-price liquidity (e.g., YES bid=0.01/ask=0.99, NO bid=0.01/ask=0.99),
    the implied probability is unreliable.
    """
    yes_best_bid = yes_metrics.get("best_bid")
    yes_best_ask = yes_metrics.get("best_ask")
    no_best_bid = no_metrics.get("best_bid")
    no_best_ask = no_metrics.get("best_ask")

    if None in (yes_best_bid, yes_best_ask, no_best_bid, no_best_ask):
        return None

    # Mid-price estimates
    yes_mid = (yes_best_bid + yes_best_ask) / 2
    no_mid = (no_best_bid + no_best_ask) / 2

    # Sum of mid prices (should be ~1.0 in a complete market with liquidity)
    mid_sum = yes_mid + no_mid

    # Implied probability of YES (from YES book)
    yes_implied = yes_mid

    # Implied probability of YES (from NO book: 1 - NO_mid)
    yes_implied_from_no = 1.0 - no_mid

    # Consistency check: how close are the two estimates?
    consistency_diff = abs(yes_implied - yes_implied_from_no)

    return {
        "yes_mid": yes_mid,
        "no_mid": no_mid,
        "mid_sum": mid_sum,
        "yes_implied": yes_implied,
        "yes_implied_from_no": yes_implied_from_no,
        "consistency_diff": consistency_diff,
    }


def _is_tight_spread(
    spread: float | None, threshold: float = DEFAULT_TIGHT_SPREAD_THRESHOLD
) -> bool:
    """Check if spread is tight (<= threshold)."""
    return spread is not None and spread <= threshold


def _is_consistent_probabilities(
    implied: dict[str, Any] | None,
    threshold: float = DEFAULT_CONSISTENCY_THRESHOLD,
) -> bool:
    """Check if implied probabilities are consistent (sum ~1.0, low consistency_diff)."""
    if implied is None:
        return False
    mid_sum = implied.get("mid_sum")
    consistency_diff = implied.get("consistency_diff")
    if mid_sum is None or consistency_diff is None:
        return False
    # Market is consistent if sum is close to 1.0 and diff between estimates is low
    return abs(mid_sum - 1.0) <= threshold and consistency_diff <= threshold


def check_alerts(
    metrics: dict[str, Any],
    spread_threshold: float = DEFAULT_SPREAD_ALERT_THRESHOLD,
    extreme_pin_threshold: float = DEFAULT_EXTREME_PIN_THRESHOLD,
    tight_spread_threshold: float = DEFAULT_TIGHT_SPREAD_THRESHOLD,
    consistency_threshold: float = DEFAULT_CONSISTENCY_THRESHOLD,
    btc_context: dict[str, Any] | None = None,
) -> list[str]:
    """Check for alert conditions in market metrics.

    Args:
        metrics: Market microstructure metrics
        spread_threshold: Alert if spread exceeds this value
        extreme_pin_threshold: Alert if price is at extremes (<= this or >= 1-this)
        tight_spread_threshold: Consider spread "tight" if <= this value
        consistency_threshold: Consider probs "consistent" if sum diff and consistency_diff <= this
        btc_context: Optional dict with BTC movement context (e.g., {"return_5m": 0.02, "return_1h": -0.05})

    Returns:
        List of alert messages.
    """
    alerts = []

    yes_metrics = metrics.get("yes", {})
    no_metrics = metrics.get("no", {})
    market_title = metrics.get("market_title", "Unknown")
    implied_probs = metrics.get("implied_probabilities")

    # Check YES book spread
    yes_spread = yes_metrics.get("spread")
    if yes_spread is not None and yes_spread > spread_threshold:
        alerts.append(
            f"[{market_title}] YES spread alert: {yes_spread:.2f} "
            f"(threshold: {spread_threshold:.2f})"
        )

    # Check NO book spread
    no_spread = no_metrics.get("spread")
    if no_spread is not None and no_spread > spread_threshold:
        alerts.append(
            f"[{market_title}] NO spread alert: {no_spread:.2f} (threshold: {spread_threshold:.2f})"
        )

    # Check for extreme pinning (best bid at floor or best ask at ceiling)
    yes_best_bid = yes_metrics.get("best_bid")
    yes_best_ask = yes_metrics.get("best_ask")
    no_best_bid = no_metrics.get("best_bid")
    no_best_ask = no_metrics.get("best_ask")

    # Determine if market is healthy (tight spread + consistent probs)
    yes_tight = _is_tight_spread(yes_spread, tight_spread_threshold)
    no_tight = _is_tight_spread(no_spread, tight_spread_threshold)
    probs_consistent = _is_consistent_probabilities(implied_probs, consistency_threshold)
    market_healthy = (yes_tight or no_tight) and probs_consistent

    # Build BTC context string if provided
    btc_context_str = ""
    if btc_context:
        parts = []
        if "return_5m" in btc_context:
            parts.append(f"5m={btc_context['return_5m']:+.2%}")
        if "return_1h" in btc_context:
            parts.append(f"1h={btc_context['return_1h']:+.2%}")
        if "return_24h" in btc_context:
            parts.append(f"24h={btc_context['return_24h']:+.2%}")
        if parts:
            btc_context_str = f" [BTC: {', '.join(parts)}]"

    # YES best bid at extreme
    if yes_best_bid is not None and yes_best_bid <= extreme_pin_threshold:
        if market_healthy:
            # Suppressed: tight spread + consistent probs = likely genuine extreme
            alerts.append(
                f"[{market_title}] YES best bid at extreme: {yes_best_bid:.2f} "
                f"(SUPPRESSED: spread tight={yes_spread:.2f}, probs consistent)"
            )
        else:
            msg = (
                f"[{market_title}] YES best bid pinned at extreme: {yes_best_bid:.2f} "
                f"(threshold: <= {extreme_pin_threshold:.2f})"
            )
            if yes_spread is not None and yes_spread > tight_spread_threshold:
                msg += f" [spread={yes_spread:.2f} > {tight_spread_threshold:.2f}]"
            if implied_probs and not probs_consistent:
                consistency_diff = implied_probs.get("consistency_diff", "N/A")
                msg += f" [inconsistent probs: diff={consistency_diff:.3f}]"
            if btc_context_str:
                msg += btc_context_str
            alerts.append(msg)

    # YES best ask at extreme
    if yes_best_ask is not None and yes_best_ask >= (1.0 - extreme_pin_threshold):
        if market_healthy:
            alerts.append(
                f"[{market_title}] YES best ask at extreme: {yes_best_ask:.2f} "
                f"(SUPPRESSED: spread tight={yes_spread:.2f}, probs consistent)"
            )
        else:
            msg = (
                f"[{market_title}] YES best ask pinned at extreme: {yes_best_ask:.2f} "
                f"(threshold: >= {1.0 - extreme_pin_threshold:.2f})"
            )
            if yes_spread is not None and yes_spread > tight_spread_threshold:
                msg += f" [spread={yes_spread:.2f} > {tight_spread_threshold:.2f}]"
            if implied_probs and not probs_consistent:
                consistency_diff = implied_probs.get("consistency_diff", "N/A")
                msg += f" [inconsistent probs: diff={consistency_diff:.3f}]"
            if btc_context_str:
                msg += btc_context_str
            alerts.append(msg)

    # NO best bid at extreme
    if no_best_bid is not None and no_best_bid <= extreme_pin_threshold:
        if market_healthy:
            alerts.append(
                f"[{market_title}] NO best bid at extreme: {no_best_bid:.2f} "
                f"(SUPPRESSED: spread tight={no_spread:.2f}, probs consistent)"
            )
        else:
            msg = (
                f"[{market_title}] NO best bid pinned at extreme: {no_best_bid:.2f} "
                f"(threshold: <= {extreme_pin_threshold:.2f})"
            )
            if no_spread is not None and no_spread > tight_spread_threshold:
                msg += f" [spread={no_spread:.2f} > {tight_spread_threshold:.2f}]"
            if implied_probs and not probs_consistent:
                consistency_diff = implied_probs.get("consistency_diff", "N/A")
                msg += f" [inconsistent probs: diff={consistency_diff:.3f}]"
            if btc_context_str:
                msg += btc_context_str
            alerts.append(msg)

    # NO best ask at extreme
    if no_best_ask is not None and no_best_ask >= (1.0 - extreme_pin_threshold):
        if market_healthy:
            alerts.append(
                f"[{market_title}] NO best ask at extreme: {no_best_ask:.2f} "
                f"(SUPPRESSED: spread tight={no_spread:.2f}, probs consistent)"
            )
        else:
            msg = (
                f"[{market_title}] NO best ask pinned at extreme: {no_best_ask:.2f} "
                f"(threshold: >= {1.0 - extreme_pin_threshold:.2f})"
            )
            if no_spread is not None and no_spread > tight_spread_threshold:
                msg += f" [spread={no_spread:.2f} > {tight_spread_threshold:.2f}]"
            if implied_probs and not probs_consistent:
                consistency_diff = implied_probs.get("consistency_diff", "N/A")
                msg += f" [inconsistent probs: diff={consistency_diff:.3f}]"
            if btc_context_str:
                msg += btc_context_str
            alerts.append(msg)

    return alerts


def analyze_market_microstructure(
    market_data: dict[str, Any],
    depth_levels: int = DEFAULT_DEPTH_LEVELS,
) -> dict[str, Any]:
    """Analyze microstructure for a single market.

    Args:
        market_data: Market dict with 'books' key containing 'yes' and 'no' books
        depth_levels: Number of book levels to include in depth calculation

    Returns:
        Dict with microstructure analysis including spread, depth, imbalance, and alerts
    """
    books = market_data.get("books", {})
    yes_book = books.get("yes", {})
    no_book = books.get("no", {})

    yes_bids = yes_book.get("bids", [])
    yes_asks = yes_book.get("asks", [])
    no_bids = no_book.get("bids", [])
    no_asks = no_book.get("asks", [])

    # Compute metrics for each side
    yes_metrics = _compute_book_metrics(yes_bids, yes_asks, depth_levels)
    no_metrics = _compute_book_metrics(no_bids, no_asks, depth_levels)

    # Compute implied probabilities
    implied_probs = _compute_implied_probabilities(yes_metrics, no_metrics)

    result = {
        "market_title": market_data.get("title", market_data.get("question", "Unknown")),
        "market_id": market_data.get("market_id"),
        "event_id": market_data.get("event_id"),
        "timestamp": datetime.now(UTC).isoformat(),
        "yes": yes_metrics,
        "no": no_metrics,
        "implied_probabilities": implied_probs,
    }

    return result


def analyze_snapshot_microstructure(
    snapshot_path: Path,
    target_market_substring: str | None = "bitcoin",
    depth_levels: int = DEFAULT_DEPTH_LEVELS,
) -> list[dict[str, Any]]:
    """Analyze microstructure for markets in a snapshot file.

    Args:
        snapshot_path: Path to the snapshot JSON file
        target_market_substring: Optional substring to filter markets (e.g., 'bitcoin')
        depth_levels: Number of book levels to include in depth calculation

    Returns:
        List of microstructure analysis results
    """
    data = json.loads(snapshot_path.read_text())
    markets = data.get("markets", [])

    results = []
    for market in markets:
        title = market.get("title", market.get("question", ""))
        if target_market_substring and target_market_substring.lower() not in title.lower():
            continue

        analysis = analyze_market_microstructure(market, depth_levels)
        results.append(analysis)

    return results


def generate_microstructure_summary(
    snapshot_path: Path,
    target_market_substring: str | None = "bitcoin",
    spread_threshold: float = DEFAULT_SPREAD_ALERT_THRESHOLD,
    extreme_pin_threshold: float = DEFAULT_EXTREME_PIN_THRESHOLD,
    depth_levels: int = DEFAULT_DEPTH_LEVELS,
    tight_spread_threshold: float = DEFAULT_TIGHT_SPREAD_THRESHOLD,
    consistency_threshold: float = DEFAULT_CONSISTENCY_THRESHOLD,
    btc_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate a summary report of market microstructure with alerts.

    Args:
        snapshot_path: Path to the snapshot JSON file
        target_market_substring: Optional substring to filter markets (e.g., 'bitcoin')
        spread_threshold: Alert threshold for spread
        extreme_pin_threshold: Alert threshold for extreme price pinning
        depth_levels: Number of book levels to include in depth calculation
        tight_spread_threshold: Consider spread "tight" if <= this value
        consistency_threshold: Consider probs "consistent" if sum diff and consistency_diff <= this
        btc_context: Optional dict with BTC movement context for alert enrichment

    Returns:
        Dict with summary statistics and alerts
    """
    analyses = analyze_snapshot_microstructure(snapshot_path, target_market_substring, depth_levels)

    all_alerts = []
    market_summaries = []

    for analysis in analyses:
        market_summary = {
            "market_title": analysis["market_title"],
            "market_id": analysis["market_id"],
            "timestamp": analysis["timestamp"],
        }

        # Add YES metrics
        yes = analysis.get("yes", {})
        market_summary["yes_spread"] = yes.get("spread")
        market_summary["yes_bid_depth"] = yes.get("bid_depth")
        market_summary["yes_ask_depth"] = yes.get("ask_depth")
        market_summary["yes_imbalance"] = yes.get("imbalance")
        market_summary["yes_best_bid"] = yes.get("best_bid")
        market_summary["yes_best_ask"] = yes.get("best_ask")

        # Add NO metrics
        no = analysis.get("no", {})
        market_summary["no_spread"] = no.get("spread")
        market_summary["no_bid_depth"] = no.get("bid_depth")
        market_summary["no_ask_depth"] = no.get("ask_depth")
        market_summary["no_imbalance"] = no.get("imbalance")
        market_summary["no_best_bid"] = no.get("best_bid")
        market_summary["no_best_ask"] = no.get("best_ask")

        # Add implied probabilities
        implied = analysis.get("implied_probabilities")
        if implied:
            market_summary["implied_yes_mid"] = implied.get("yes_mid")
            market_summary["implied_no_mid"] = implied.get("no_mid")
            market_summary["implied_sum"] = implied.get("mid_sum")
            market_summary["consistency_diff"] = implied.get("consistency_diff")

        market_summaries.append(market_summary)

        # Check for alerts with new thresholds and BTC context
        alerts = check_alerts(
            analysis,
            spread_threshold,
            extreme_pin_threshold,
            tight_spread_threshold,
            consistency_threshold,
            btc_context,
        )
        all_alerts.extend(alerts)

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "snapshot_path": str(snapshot_path),
        "markets_analyzed": len(analyses),
        "spread_threshold": spread_threshold,
        "extreme_pin_threshold": extreme_pin_threshold,
        "tight_spread_threshold": tight_spread_threshold,
        "consistency_threshold": consistency_threshold,
        "depth_levels": depth_levels,
        "market_summaries": market_summaries,
        "alerts": all_alerts,
        "alert_count": len(all_alerts),
    }


def write_microstructure_report(
    snapshot_path: Path,
    out_path: Path,
    target_market_substring: str | None = "bitcoin",
    spread_threshold: float = DEFAULT_SPREAD_ALERT_THRESHOLD,
    extreme_pin_threshold: float = DEFAULT_EXTREME_PIN_THRESHOLD,
    depth_levels: int = DEFAULT_DEPTH_LEVELS,
    tight_spread_threshold: float = DEFAULT_TIGHT_SPREAD_THRESHOLD,
    consistency_threshold: float = DEFAULT_CONSISTENCY_THRESHOLD,
    btc_context: dict[str, Any] | None = None,
) -> Path:
    """Generate and write a microstructure report to disk.

    Returns the output path.
    """
    summary = generate_microstructure_summary(
        snapshot_path=snapshot_path,
        target_market_substring=target_market_substring,
        spread_threshold=spread_threshold,
        extreme_pin_threshold=extreme_pin_threshold,
        depth_levels=depth_levels,
        tight_spread_threshold=tight_spread_threshold,
        consistency_threshold=consistency_threshold,
        btc_context=btc_context,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    return out_path


def log_microstructure_alerts(summary: dict[str, Any]) -> None:
    """Log alerts from a microstructure summary."""
    alerts = summary.get("alerts", [])
    if not alerts:
        logger.info("No microstructure alerts. Markets appear healthy.")
        return

    logger.warning("Microstructure alerts detected (%d):", len(alerts))
    for alert in alerts:
        logger.warning("  - %s", alert)
