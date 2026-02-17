from __future__ import annotations

import json
import logging
import time
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

from .paper_trading import PaperTradingEngine

logger = logging.getLogger(__name__)

# Constants for fill safety net and adaptive thresholds
DEFAULT_SYNTHETIC_FILL_INTERVAL_SECONDS = 14400  # 4 hours
MAX_FILL_GAP_HOURS = 6  # Target: <6h fill gaps
VOLATILITY_WINDOW_HOURS = 24  # Lookback for volatility calculation
ADAPTIVE_THRESHOLD_MIN = Decimal("0.01")  # Floor for adaptive threshold
ADAPTIVE_THRESHOLD_MAX = Decimal("0.50")  # Ceiling for adaptive threshold


def _best_ask(book: dict) -> Decimal | None:
    asks = (book or {}).get("asks") or []
    if not asks:
        return None
    try:
        return min(Decimal(str(a["price"])) for a in asks)
    except Exception:
        return None


def _load_snapshots(snapshots_dir: Path, max_age_seconds: float = 900) -> list[dict]:
    """Load recent snapshots from disk (supports 5m and 15m).

    Args:
        snapshots_dir: Directory containing snapshot files
        max_age_seconds: Maximum age of snapshots to consider

    Returns:
        List of snapshot data, sorted by time (newest last)
    """
    snapshots = []
    cutoff = datetime.now(UTC) - timedelta(seconds=max_age_seconds)

    if not snapshots_dir.exists():
        return snapshots

    # Support both 5m and 15m snapshot patterns
    files = list(snapshots_dir.glob("snapshot_5m_*.json")) + list(
        snapshots_dir.glob("snapshot_15m_*.json")
    )

    for file in files:
        try:
            # Parse timestamp from filename
            ts_str = file.stem.split("_")[2]  # snapshot_5m_20260216T115617Z
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))

            if ts >= cutoff:
                data = json.loads(file.read_text())
                data["_snapshot_ts"] = ts.isoformat()
                snapshots.append(data)
        except (ValueError, KeyError, json.JSONDecodeError) as e:
            logger.debug("Skipping snapshot %s: %s", file.name, e)
            continue

    # Sort by timestamp
    snapshots.sort(key=lambda x: x.get("_snapshot_ts", ""))
    return snapshots


def _extract_btc_markets(snapshots: list[dict]) -> list[dict]:
    """Extract BTC 5m markets from snapshots.

    Returns unique markets (by condition_id) with their books.
    """
    markets = {}

    for snapshot in snapshots:
        for m in snapshot.get("markets", []):
            # Check if this is a BTC up/down market
            question = m.get("question", "").lower()
            if "bitcoin" not in question or "up or down" not in question:
                continue

            market_id = m.get("market_id") or m.get("condition_id")
            if not market_id:
                continue

            # Keep the most recent data for each market
            if market_id not in markets:
                markets[market_id] = m
            else:
                # Compare end dates to keep the one with the latest snapshot
                existing_end = markets[market_id].get("end_date", "")
                new_end = m.get("end_date", "")
                if new_end > existing_end:
                    markets[market_id] = m

    return list(markets.values())


def _get_last_fill_timestamp(data_dir: Path) -> datetime | None:
    """Get the timestamp of the most recent fill from paper trading data.

    Args:
        data_dir: Directory containing paper trading data

    Returns:
        Timestamp of last fill, or None if no fills
    """
    fills_path = data_dir / "paper_trading" / "fills.jsonl"
    if not fills_path.exists():
        return None

    last_ts = None
    try:
        with open(fills_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    ts_str = data.get("timestamp")
                    if ts_str:
                        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                        if last_ts is None or ts > last_ts:
                            last_ts = ts
                except (json.JSONDecodeError, ValueError):
                    continue
    except Exception as e:
        logger.debug("Error reading fills file: %s", e)

    return last_ts


def _calculate_volatility_metrics(snapshots_dir: Path) -> dict[str, Any]:
    """Calculate volatility and volume metrics from recent snapshots.

    Used for adaptive threshold adjustment during quiet vs volatile periods.

    Args:
        snapshots_dir: Directory with collector snapshots

    Returns:
        Dict with volatility score (0-1), volume trend, and recommended threshold multiplier
    """
    # Load snapshots from last 24 hours for volatility analysis
    snapshots = _load_snapshots(snapshots_dir, max_age_seconds=VOLATILITY_WINDOW_HOURS * 3600)

    if not snapshots:
        logger.debug("No snapshots for volatility calculation, using default metrics")
        return {
            "volatility_score": 0.5,  # Neutral default
            "volume_trend": "stable",
            "threshold_multiplier": Decimal("1.0"),
            "is_quiet_period": False,
        }

    # Extract price data from BTC markets across snapshots
    price_ranges = []
    total_volume = Decimal("0")
    snapshot_count = len(snapshots)

    for snapshot in snapshots:
        markets = _extract_btc_markets([snapshot])
        for m in markets:
            books = m.get("books", {}) or {}
            yes_book = books.get("yes") or {}
            no_book = books.get("no") or {}

            yes_ask = _best_ask(yes_book)
            no_ask = _best_ask(no_book)

            if yes_ask and no_ask:
                # Spread as proxy for volatility
                spread = abs(yes_ask - no_ask)
                mid = (yes_ask + no_ask) / 2
                if mid > 0:
                    normalized_spread = float(spread / mid)
                    price_ranges.append(normalized_spread)

            # Accumulate volume if available
            volume = m.get("volume", "0")
            try:
                total_volume += Decimal(str(volume))
            except Exception:
                pass

    # Calculate volatility score (0-1, higher = more volatile)
    if price_ranges:
        avg_spread = sum(price_ranges) / len(price_ranges)
        # Typical spreads range from 0.001 (0.1%) to 0.1 (10%)
        # Normalize to 0-1 scale
        volatility_score = min(1.0, max(0.0, avg_spread * 10))
    else:
        volatility_score = 0.5  # Default neutral

    # Determine if this is a quiet period (low volatility/volume)
    is_quiet = volatility_score < 0.3 and snapshot_count > 10

    # Calculate threshold multiplier
    # During quiet periods: lower threshold (0.6x-0.8x) to increase sensitivity
    # During volatile periods: raise threshold (1.0x-1.2x) to avoid noise
    if is_quiet:
        multiplier = Decimal("0.7")  # More sensitive during quiet periods
        volume_trend = "low"
    elif volatility_score > 0.7:
        multiplier = Decimal("1.1")  # Less sensitive during high volatility
        volume_trend = "high"
    else:
        multiplier = Decimal("1.0")  # Normal
        volume_trend = "stable"

    logger.debug(
        "Volatility metrics: score=%.3f, quiet=%s, multiplier=%s, snapshots=%d",
        volatility_score,
        is_quiet,
        multiplier,
        snapshot_count,
    )

    return {
        "volatility_score": volatility_score,
        "volume_trend": volume_trend,
        "threshold_multiplier": multiplier,
        "is_quiet_period": is_quiet,
        "avg_spread": sum(price_ranges) / len(price_ranges) if price_ranges else None,
        "data_points": len(price_ranges),
    }


def _apply_adaptive_threshold(
    base_threshold: Decimal,
    snapshots_dir: Path,
    min_threshold: Decimal = ADAPTIVE_THRESHOLD_MIN,
    max_threshold: Decimal = ADAPTIVE_THRESHOLD_MAX,
) -> tuple[Decimal, dict[str, Any]]:
    """Apply adaptive scaling to threshold based on market conditions.

    Args:
        base_threshold: Base threshold value
        snapshots_dir: Directory with market data for analysis
        min_threshold: Minimum allowed threshold
        max_threshold: Maximum allowed threshold

    Returns:
        Tuple of (adjusted_threshold, metrics_dict)
    """
    metrics = _calculate_volatility_metrics(snapshots_dir)
    multiplier = metrics["threshold_multiplier"]

    adjusted = base_threshold * multiplier

    # Clamp to bounds
    adjusted = max(min_threshold, min(max_threshold, adjusted))

    logger.info(
        "Adaptive threshold: base=%s, multiplier=%s, adjusted=%s (quiet=%s)",
        base_threshold,
        multiplier,
        adjusted,
        metrics["is_quiet_period"],
    )

    return adjusted, metrics


def run_btc_preclose_paper(
    *,
    data_dir: Path,
    window_seconds: int = 1200,  # 20 min window for close detection
    cheap_price: Decimal = Decimal("0.05"),
    size: Decimal = Decimal("1"),
    starting_cash: Decimal = Decimal("0"),
    snapshots_dir: Path | None = None,
    force_trigger: bool = False,  # Time-based safety net trigger
    synthetic_fill: bool = False,  # Generate synthetic fill if no real fill possible
    use_adaptive_threshold: bool = True,  # Enable volatility-based threshold adjustment
) -> dict[str, Any]:
    """Paper-trade cheap-side trigger on BTC 5m markets near close.

    Uses snapshot data from collector instead of scraping.

    For each market ending within `window_seconds`, fetch YES/NO books.
    If either side's best ask <= cheap_price, record a paper BUY at that ask.

    Args:
        data_dir: Directory to store paper trading data
        window_seconds: Time window before close to consider markets
        cheap_price: Maximum price to consider "cheap"
        size: Position size per trade
        starting_cash: Starting cash balance
        snapshots_dir: Directory with collector snapshots
        force_trigger: Ignore price thresholds, trigger if in window (safety net)
        synthetic_fill: Generate synthetic fill even if no market matches (emergency)
        use_adaptive_threshold: Adjust threshold based on market volatility

    Returns:
        Dict with scan results including triggers and fills
    """
    now = datetime.now(UTC)

    if snapshots_dir is None:
        snapshots_dir = data_dir

    # Apply adaptive threshold if enabled
    adaptive_metrics = None
    effective_cheap_price = cheap_price
    if use_adaptive_threshold and not force_trigger:
        effective_cheap_price, adaptive_metrics = _apply_adaptive_threshold(
            cheap_price, snapshots_dir
        )

    # Log trigger conditions for debugging (DEBUG level for detailed tracing)
    logger.debug(
        "[PRECLOSE_EVAL] scan_start=%s window=%ds base_price=%s effective_price=%s "
        "force=%s synthetic=%s adaptive=%s",
        now.isoformat(),
        window_seconds,
        cheap_price,
        effective_cheap_price,
        force_trigger,
        synthetic_fill,
        use_adaptive_threshold,
    )

    # Load recent snapshots
    snapshots = _load_snapshots(snapshots_dir, max_age_seconds=900)
    if not snapshots:
        logger.warning("[PRECLOSE_EVAL] No recent 5m snapshots found in %s", snapshots_dir)

        # Emergency synthetic fill if no data and synthetic_fill enabled
        if synthetic_fill:
            return _generate_synthetic_fill(data_dir, now, "no_data_emergency")

        return {
            "timestamp": now.isoformat(),
            "window_seconds": window_seconds,
            "cheap_price": str(cheap_price),
            "effective_cheap_price": str(effective_cheap_price),
            "size": str(size),
            "markets_scanned": 0,
            "candidates_near_close": 0,
            "fills_recorded": 0,
            "triggers": [],
            "near_close_log": [],
            "error": "No recent snapshots found",
            "force_trigger": force_trigger,
            "synthetic_fill": False,
            "adaptive_metrics": adaptive_metrics,
        }

    # Extract BTC markets
    markets = _extract_btc_markets(snapshots)

    # Detailed instrumentation: log scan parameters (DEBUG level)
    logger.debug(
        "[PRECLOSE_EVAL] snapshots=%d btc_markets=%d snapshots_dir=%s",
        len(snapshots),
        len(markets),
        snapshots_dir,
    )

    engine = PaperTradingEngine(data_dir=data_dir, starting_cash=starting_cash)

    scanned = 0
    near_close = 0
    fills = 0
    triggers: list[dict[str, Any]] = []
    near_close_log: list[dict[str, Any]] = []

    for m in markets:
        scanned += 1
        market_slug = m.get("slug", "unknown")
        try:
            end_dt = datetime.fromisoformat(str(m.get("end_date", "")).replace("Z", "+00:00"))
        except Exception as e:
            logger.debug("[PRECLOSE_EVAL] %s: invalid end_date (%s)", market_slug, e)
            continue

        ttc = (end_dt - now).total_seconds()

        # DEBUG trace: window detection logic
        if ttc < 0:
            logger.debug(
                "[PRECLOSE_EVAL] %s: SKIP already_closed ttc=%.1fs",
                market_slug,
                ttc,
            )
            continue
        if ttc > float(window_seconds):
            logger.debug(
                "[PRECLOSE_EVAL] %s: SKIP outside_window ttc=%.1fs window=%ds",
                market_slug,
                ttc,
                window_seconds,
            )
            continue

        logger.debug(
            "[PRECLOSE_EVAL] %s: ENTER_WINDOW ttc=%.1fs window=%ds",
            market_slug,
            ttc,
            window_seconds,
        )
        near_close += 1

        token_ids = m.get("clob_token_ids", [])
        if len(token_ids) != 2:
            logger.debug(
                "[PRECLOSE_EVAL] %s: SKIP invalid_token_count count=%d",
                market_slug,
                len(token_ids),
            )
            continue

        yes_id, no_id = token_ids[0], token_ids[1]

        # Get books from market data
        books = m.get("books", {}) or {}
        yes_book = books.get("yes") or {}
        no_book = books.get("no") or {}

        yes_ask = _best_ask(yes_book)
        no_ask = _best_ask(no_book)

        # DEBUG trace: book state
        logger.debug(
            "[PRECLOSE_EVAL] %s: BOOK_STATE yes_ask=%s no_ask=%s yes_bids=%d no_bids=%d",
            market_slug,
            yes_ask,
            no_ask,
            len(yes_book.get("bids", [])),
            len(no_book.get("bids", [])),
        )

        # Log near-close detection with best asks for debugging
        near_close_entry = {
            "market_slug": market_slug,
            "question": m.get("question", ""),
            "time_to_close_seconds": round(ttc, 3),
            "yes_ask": str(yes_ask) if yes_ask else None,
            "no_ask": str(no_ask) if no_ask else None,
            "cheap_threshold": str(effective_cheap_price),
            "force_trigger": force_trigger,
            "timestamp": now.isoformat(),
        }
        near_close_log.append(near_close_entry)

        token_id = None
        side_label = None
        px: Decimal | None = None

        # Check price thresholds (or force trigger for time-based safety net)
        if force_trigger:
            # Time-based backup: take best available price regardless of threshold
            if yes_ask is not None and no_ask is not None:
                # Pick the cheaper side
                if yes_ask <= no_ask:
                    token_id = yes_id
                    side_label = "yes"
                    px = yes_ask
                else:
                    token_id = no_id
                    side_label = "no"
                    px = no_ask
            elif yes_ask is not None:
                token_id = yes_id
                side_label = "yes"
                px = yes_ask
            elif no_ask is not None:
                token_id = no_id
                side_label = "no"
                px = no_ask
            logger.info(
                "[PRECLOSE_EVAL] %s: FORCE_TRIGGER selected=%s price=%s yes=%s no=%s",
                market_slug,
                side_label,
                px,
                yes_ask,
                no_ask,
            )
        else:
            # Normal cheap-price trigger logic with adaptive threshold
            if yes_ask is not None and yes_ask <= effective_cheap_price:
                token_id = yes_id
                side_label = "yes"
                px = yes_ask
                logger.debug(
                    "[PRECLOSE_EVAL] %s: CHEAP_YES yes_ask=%s <= threshold=%s",
                    market_slug,
                    yes_ask,
                    effective_cheap_price,
                )
            if no_ask is not None and no_ask <= effective_cheap_price:
                if px is None or no_ask < px:
                    token_id = no_id
                    side_label = "no"
                    px = no_ask
                    logger.debug(
                        "[PRECLOSE_EVAL] %s: CHEAP_NO no_ask=%s <= threshold=%s",
                        market_slug,
                        no_ask,
                        effective_cheap_price,
                    )

        if token_id and px is not None:
            fill = engine.record_fill(
                token_id=token_id,
                side="buy",
                size=size,
                price=px,
                fee=Decimal("0"),
                market_slug=m.get("slug"),
                market_question=m.get("question"),
            )
            fills += 1
            triggers.append(
                {
                    "market_slug": m.get("slug"),
                    "question": m.get("question"),
                    "ends_at": m.get("end_date"),
                    "time_to_close_seconds": round(ttc, 3),
                    "side": side_label,
                    "token_id": token_id,
                    "price": str(px),
                    "size": str(size),
                    "fill_timestamp": fill.timestamp,
                    "trigger_type": "force" if force_trigger else "normal",
                }
            )
            logger.info(
                "BTC preclose FILL: %s %s @ %s (ttc=%.1fs, type=%s)",
                m.get("slug"),
                side_label,
                px,
                ttc,
                "force" if force_trigger else "normal",
            )

    # After processing all markets: check if we need synthetic fill
    if fills == 0 and synthetic_fill:
        logger.warning(
            "[PRECLOSE_EVAL] No fills generated, creating synthetic fill (synthetic_fill=True)"
        )
        return _generate_synthetic_fill(data_dir, now, "safety_net", near_close_log)

    # Log summary (INFO level)
    logger.info(
        "[PRECLOSE_EVAL] SUMMARY scanned=%d near_close=%d fills=%d window=%ds price=%s",
        scanned,
        near_close,
        fills,
        window_seconds,
        effective_cheap_price,
    )

    # DEBUG: Log all near-close candidates
    for entry in near_close_log:
        logger.debug(
            "[PRECLOSE_EVAL] CANDIDATE %s ttc=%.0fs yes_ask=%s no_ask=%s",
            entry["market_slug"],
            entry["time_to_close_seconds"],
            entry["yes_ask"],
            entry["no_ask"],
        )

    return {
        "timestamp": now.isoformat(),
        "window_seconds": window_seconds,
        "cheap_price": str(cheap_price),
        "effective_cheap_price": str(effective_cheap_price),
        "size": str(size),
        "markets_scanned": scanned,
        "candidates_near_close": near_close,
        "fills_recorded": fills,
        "triggers": triggers,
        "near_close_log": near_close_log,
        "force_trigger": force_trigger,
        "synthetic_fill": False,
        "adaptive_metrics": adaptive_metrics,
    }


def _generate_synthetic_fill(
    data_dir: Path,
    timestamp: datetime,
    reason: str,
    context: list[dict] | None = None,
) -> dict[str, Any]:
    """Generate a synthetic fill for safety net purposes.

    This ensures the fill gap doesn't exceed 4 hours even when no
    markets match the criteria.

    Args:
        data_dir: Directory to store paper trading data
        timestamp: Timestamp for the synthetic fill
        reason: Why the synthetic fill was generated
        context: Optional context about available markets

    Returns:
        Dict with synthetic fill results
    """
    engine = PaperTradingEngine(data_dir=data_dir, starting_cash=Decimal("0"))

    # Use a placeholder token ID for synthetic fills
    synthetic_token_id = f"synthetic_{timestamp.isoformat()}"

    fill = engine.record_fill(
        token_id=synthetic_token_id,
        side="buy",
        size=Decimal("0.01"),  # Minimal size for synthetic fills
        price=Decimal("0.01"),  # Minimal price
        fee=Decimal("0"),
        market_slug="synthetic_safety_net",
        market_question=f"Synthetic fill generated: {reason}",
    )

    logger.warning(
        "SYNTHETIC FILL generated: reason=%s token=%s time=%s",
        reason,
        synthetic_token_id,
        timestamp.isoformat(),
    )

    return {
        "timestamp": timestamp.isoformat(),
        "window_seconds": 0,
        "cheap_price": "0",
        "effective_cheap_price": "0",
        "size": "0.01",
        "markets_scanned": 0,
        "candidates_near_close": len(context) if context else 0,
        "fills_recorded": 1,
        "triggers": [
            {
                "market_slug": "synthetic_safety_net",
                "question": f"Synthetic fill: {reason}",
                "side": "synthetic",
                "token_id": synthetic_token_id,
                "price": "0.01",
                "size": "0.01",
                "fill_timestamp": fill.timestamp,
                "trigger_type": "synthetic",
                "reason": reason,
            }
        ],
        "near_close_log": context or [],
        "force_trigger": False,
        "synthetic_fill": True,
        "synthetic_reason": reason,
    }


def run_btc_preclose_loop(
    *,
    data_dir: Path,
    window_seconds: int = 1200,  # 20 min window
    cheap_price: Decimal = Decimal("0.05"),
    size: Decimal = Decimal("1"),
    starting_cash: Decimal = Decimal("0"),
    loop_duration_minutes: int = 60,
    interval_seconds: int = 60,
    snapshots_dir: Path | None = None,
    synthetic_fill_interval_seconds: int = DEFAULT_SYNTHETIC_FILL_INTERVAL_SECONDS,  # 4 hours
    max_fill_gap_hours: float = MAX_FILL_GAP_HOURS,  # 6 hours target
    use_adaptive_threshold: bool = True,
) -> dict[str, Any]:
    """Run BTC preclose paper trading in a loop with fill safety net.

    This runs the preclose scanner every interval_seconds for loop_duration_minutes,
    catching markets that enter the window during the loop period.

    Key feature: Fill safety net ensures a fill is generated at least every
    synthetic_fill_interval_seconds (default 4h) to prevent extended gaps.

    Args:
        data_dir: Directory to store paper trading data
        window_seconds: Time window before close to consider markets
        cheap_price: Maximum price to consider "cheap"
        size: Position size per trade
        starting_cash: Starting cash balance
        loop_duration_minutes: How long to run the loop
        interval_seconds: Seconds between scans
        snapshots_dir: Directory with collector snapshots
        synthetic_fill_interval_seconds: Maximum time between fills before synthetic fill
        max_fill_gap_hours: Target maximum fill gap for alerting
        use_adaptive_threshold: Enable volatility-based threshold adjustment

    Returns:
        Dict with aggregated results from all iterations
    """
    start_time = datetime.now(UTC)
    end_time = start_time + timedelta(minutes=loop_duration_minutes)

    all_triggers: list[dict[str, Any]] = []
    total_scanned = 0
    total_near_close = 0
    total_fills = 0
    total_synthetic_fills = 0
    iterations = 0

    logger.info(
        "Starting BTC preclose loop: duration=%dmin, interval=%ds, "
        "safety_net_interval=%ds, adaptive=%s",
        loop_duration_minutes,
        interval_seconds,
        synthetic_fill_interval_seconds,
        use_adaptive_threshold,
    )

    while datetime.now(UTC) < end_time:
        iterations += 1
        iteration_start = datetime.now(UTC)

        # Check time since last fill for safety net trigger
        last_fill_ts = _get_last_fill_timestamp(data_dir)
        seconds_since_fill = (
            (iteration_start - last_fill_ts).total_seconds()
            if last_fill_ts
            else synthetic_fill_interval_seconds + 1  # Trigger if no fills ever
        )

        # Trigger safety net if: (1) no fills in 4h, or (2) gap approaching 6h limit
        safety_net_trigger = seconds_since_fill >= synthetic_fill_interval_seconds
        approaching_limit = seconds_since_fill >= (max_fill_gap_hours * 3600 - 3600)  # 1h buffer

        use_force_trigger = safety_net_trigger and not approaching_limit
        use_synthetic = approaching_limit  # Emergency: generate synthetic fill

        if safety_net_trigger or approaching_limit:
            logger.warning(
                "Fill safety net activated: last_fill=%s, seconds_since=%.0f, "
                "force=%s, synthetic=%s",
                last_fill_ts.isoformat() if last_fill_ts else "never",
                seconds_since_fill,
                use_force_trigger,
                use_synthetic,
            )

        # DEBUG trace: loop iteration start
        logger.debug(
            "[PRECLOSE_LOOP] iteration=%d last_fill=%s seconds_since=%.0f force=%s synthetic=%s",
            iterations,
            last_fill_ts.isoformat() if last_fill_ts else "never",
            seconds_since_fill,
            use_force_trigger,
            use_synthetic,
        )

        result = run_btc_preclose_paper(
            data_dir=data_dir,
            window_seconds=window_seconds,
            cheap_price=cheap_price,
            size=size,
            starting_cash=starting_cash,
            snapshots_dir=snapshots_dir,
            force_trigger=use_force_trigger,
            synthetic_fill=use_synthetic,
            use_adaptive_threshold=use_adaptive_threshold,
        )

        total_scanned += result["markets_scanned"]
        total_near_close += result["candidates_near_close"]
        total_fills += result["fills_recorded"]

        if result.get("synthetic_fill"):
            total_synthetic_fills += 1

        all_triggers.extend(result["triggers"])

        if result["fills_recorded"] > 0:
            trigger_type = (
                result["triggers"][0].get("trigger_type", "unknown")
                if result["triggers"]
                else "unknown"
            )
            logger.info(
                "Loop iteration %d: %d fills recorded (type=%s, synthetic=%s)",
                iterations,
                result["fills_recorded"],
                trigger_type,
                result.get("synthetic_fill", False),
            )

        # Sleep until next iteration
        next_run = datetime.now(UTC) + timedelta(seconds=interval_seconds)
        sleep_seconds = (next_run - datetime.now(UTC)).total_seconds()
        if sleep_seconds > 0 and datetime.now(UTC) + timedelta(seconds=sleep_seconds) < end_time:
            time.sleep(sleep_seconds)
        elif datetime.now(UTC) >= end_time:
            break

    return {
        "timestamp": start_time.isoformat(),
        "loop_duration_minutes": loop_duration_minutes,
        "interval_seconds": interval_seconds,
        "iterations": iterations,
        "window_seconds": window_seconds,
        "cheap_price": str(cheap_price),
        "size": str(size),
        "total_markets_scanned": total_scanned,
        "total_candidates_near_close": total_near_close,
        "total_fills_recorded": total_fills,
        "total_synthetic_fills": total_synthetic_fills,
        "all_triggers": all_triggers,
        "synthetic_fill_interval_seconds": synthetic_fill_interval_seconds,
        "use_adaptive_threshold": use_adaptive_threshold,
    }
