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

# Window fallback tiers for progressively wider collection windows
# Used when no fills are found with narrower windows
WINDOW_FALLBACK_TIERS = [900, 1800, 3600, 7200, 14400]  # 15min, 30min, 60min, 2h, 4h


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


def run_btc_preclose_paper(
    *,
    data_dir: Path,
    window_seconds: int = 1800,  # Widened from 900 to 1800 (30 min window default)
    cheap_price: Decimal = Decimal("0.35"),  # Widened from 0.25 to 0.35 for higher hit rate
    size: Decimal = Decimal("1"),
    starting_cash: Decimal = Decimal("0"),
    snapshots_dir: Path | None = None,
    use_window_fallback: bool = True,  # Enable fallback to wider windows if no fills
    log_spread_rejects: bool = True,  # Log rejected spreads for tuning
) -> dict[str, Any]:
    """Paper-trade cheap-side trigger on BTC 5m markets near close.

    Uses snapshot data from collector instead of scraping (which is broken).

    For each market ending within `window_seconds`, fetch YES/NO books.
    If either side's best ask <= cheap_price, record a paper BUY at that ask.

    With `use_window_fallback=True`, if no fills are found with the initial
    window, progressively wider windows are tried (up to 4h) to ensure
    some fills are captured for tuning analysis.

    Args:
        data_dir: Directory to store paper trading data
        window_seconds: Time window before close to consider markets (default: 1800 = 30 min)
        cheap_price: Maximum price to consider "cheap" (default: 0.35)
        size: Position size per trade
        starting_cash: Starting cash balance
        snapshots_dir: Directory with collector snapshots (defaults to data_dir)
        use_window_fallback: Whether to try progressively wider windows if no fills
        log_spread_rejects: Whether to log rejected spreads for tuning

    Returns:
        Dict with scan results including triggers and fills
    """
    # Try initial window, then fall back to wider windows if needed
    windows_to_try = (
        [window_seconds] if not use_window_fallback else _get_fallback_windows(window_seconds)
    )

    for try_window in windows_to_try:
        result = _run_btc_preclose_single(
            data_dir=data_dir,
            window_seconds=try_window,
            cheap_price=cheap_price,
            size=size,
            starting_cash=starting_cash,
            snapshots_dir=snapshots_dir,
            log_spread_rejects=log_spread_rejects,
        )

        # If we got fills, return the result
        if result["fills_recorded"] > 0:
            if try_window != window_seconds:
                logger.info(
                    "BTC preclose fallback success: window=%ds produced %d fills",
                    try_window,
                    result["fills_recorded"],
                )
            return result

        # Log that we're falling back
        if try_window != windows_to_try[-1]:
            logger.info(
                "BTC preclose no fills with window=%ds, falling back to wider window",
                try_window,
            )

    # Return the last result (widest window attempted) even if no fills
    return result


def _get_fallback_windows(initial_window: int) -> list[int]:
    """Get list of windows to try, including fallbacks.

    Returns unique sorted list starting from initial_window,
    then progressively wider tiers.
    """
    windows = [initial_window]
    for tier in WINDOW_FALLBACK_TIERS:
        if tier > initial_window and tier not in windows:
            windows.append(tier)
    return windows


def _run_btc_preclose_single(
    *,
    data_dir: Path,
    window_seconds: int,
    cheap_price: Decimal,
    size: Decimal,
    starting_cash: Decimal,
    snapshots_dir: Path | None,
    log_spread_rejects: bool,
) -> dict[str, Any]:
    """Single pass of BTC preclose scanning with given window."""
    now = datetime.now(UTC)

    if snapshots_dir is None:
        snapshots_dir = data_dir

    # Load recent snapshots
    snapshots = _load_snapshots(snapshots_dir, max_age_seconds=900)
    if not snapshots:
        logger.warning("No recent 5m snapshots found in %s", snapshots_dir)
        return {
            "timestamp": now.isoformat(),
            "window_seconds": window_seconds,
            "cheap_price": str(cheap_price),
            "size": str(size),
            "markets_scanned": 0,
            "candidates_near_close": 0,
            "fills_recorded": 0,
            "triggers": [],
            "near_close_log": [],
            "error": "No recent snapshots found",
        }

    # Extract BTC markets
    markets = _extract_btc_markets(snapshots)

    # Detailed instrumentation: log scan parameters
    logger.info(
        "BTC preclose scan starting: snapshots_dir=%s, snapshots=%d, btc_markets=%d, "
        "window=%ds, cheap_price=%s, size=%s",
        snapshots_dir,
        len(snapshots),
        len(markets),
        window_seconds,
        cheap_price,
        size,
    )

    engine = PaperTradingEngine(data_dir=data_dir, starting_cash=starting_cash)

    scanned = 0
    near_close = 0
    fills = 0
    triggers: list[dict[str, Any]] = []
    near_close_log: list[dict[str, Any]] = []
    spread_rejects: list[dict[str, Any]] = []  # Track rejected spreads for tuning

    for m in markets:
        scanned += 1
        try:
            end_dt = datetime.fromisoformat(str(m.get("end_date", "")).replace("Z", "+00:00"))
        except Exception:
            continue

        ttc = (end_dt - now).total_seconds()
        if ttc < 0 or ttc > float(window_seconds):
            continue

        near_close += 1

        token_ids = m.get("clob_token_ids", [])
        if len(token_ids) != 2:
            continue

        yes_id, no_id = token_ids[0], token_ids[1]

        # Get books from market data
        books = m.get("books", {}) or {}
        yes_book = books.get("yes") or {}
        no_book = books.get("no") or {}

        yes_ask = _best_ask(yes_book)
        no_ask = _best_ask(no_book)

        # Log near-close detection with best asks for debugging
        near_close_entry = {
            "market_slug": m.get("slug", "unknown"),
            "question": m.get("question", ""),
            "time_to_close_seconds": round(ttc, 3),
            "yes_ask": str(yes_ask) if yes_ask else None,
            "no_ask": str(no_ask) if no_ask else None,
            "timestamp": now.isoformat(),
        }
        near_close_log.append(near_close_entry)

        # Detailed logging for rejected spreads (for tuning analysis)
        min_ask = min(yes_ask or Decimal("1.0"), no_ask or Decimal("1.0"))
        if min_ask > cheap_price:
            reject_entry = {
                "market_slug": m.get("slug", "unknown"),
                "time_to_close_seconds": round(ttc, 3),
                "min_ask": str(min_ask),
                "cheap_price": str(cheap_price),
                "spread_vs_threshold": float(min_ask - cheap_price),
            }
            spread_rejects.append(reject_entry)
            if log_spread_rejects:
                logger.info(
                    "BTC preclose REJECT: %s ttc=%.0fs min_ask=%s > cheap_price=%s "
                    "(spread_vs_threshold=%.3f)",
                    m.get("slug", "unknown"),
                    ttc,
                    min_ask,
                    cheap_price,
                    min_ask - cheap_price,
                )

        token_id = None
        side_label = None
        px: Decimal | None = None

        if yes_ask is not None and yes_ask <= cheap_price:
            token_id = yes_id
            side_label = "yes"
            px = yes_ask
            logger.info(
                "BTC preclose ACCEPT: %s YES @ %s <= cheap_price=%s",
                m.get("slug", "unknown"),
                yes_ask,
                cheap_price,
            )
        if no_ask is not None and no_ask <= cheap_price:
            if px is None or no_ask < px:
                token_id = no_id
                side_label = "no"
                px = no_ask
                logger.info(
                    "BTC preclose ACCEPT: %s NO @ %s <= cheap_price=%s",
                    m.get("slug", "unknown"),
                    no_ask,
                    cheap_price,
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
                }
            )
            logger.info(
                "BTC preclose fill: %s %s @ %s (ttc=%.1fs)",
                m.get("slug"),
                side_label,
                px,
                ttc,
            )

    # Log near-close detections for debugging
    logger.info(
        "BTC preclose summary: scanned=%d, near_close=%d, fills=%d, window=%ds, cheap_price=%s, "
        "spread_rejects=%d",
        scanned,
        near_close,
        fills,
        window_seconds,
        cheap_price,
        len(spread_rejects),
    )

    # Log spread reject distribution for tuning
    if spread_rejects:
        spreads_over_threshold = [r["spread_vs_threshold"] for r in spread_rejects]
        avg_spread_over = sum(spreads_over_threshold) / len(spreads_over_threshold)
        max_spread_over = max(spreads_over_threshold)
        logger.info(
            "BTC preclose spread analysis: rejects=%d, avg_spread_over_threshold=%.3f, "
            "max_spread_over=%.3f",
            len(spread_rejects),
            avg_spread_over,
            max_spread_over,
        )

    if near_close_log:
        for entry in near_close_log:  # Log ALL near-close markets
            logger.info(
                "BTC preclose candidate: %s ttc=%.0fs yes_ask=%s no_ask=%s",
                entry["market_slug"],
                entry["time_to_close_seconds"],
                entry["yes_ask"],
                entry["no_ask"],
            )

    return {
        "timestamp": now.isoformat(),
        "window_seconds": window_seconds,
        "cheap_price": str(cheap_price),
        "size": str(size),
        "markets_scanned": scanned,
        "candidates_near_close": near_close,
        "fills_recorded": fills,
        "triggers": triggers,
        "near_close_log": near_close_log,
        "spread_rejects": spread_rejects,
        "spread_reject_count": len(spread_rejects),
    }


def run_btc_preclose_loop(
    *,
    data_dir: Path,
    window_seconds: int = 1800,  # Widened from 900 to 1800 (30 min window default)
    cheap_price: Decimal = Decimal("0.35"),  # Widened from 0.25 to 0.35 for higher hit rate
    size: Decimal = Decimal("1"),
    starting_cash: Decimal = Decimal("0"),
    loop_duration_minutes: int = 10,
    interval_seconds: int = 60,
    snapshots_dir: Path | None = None,
    use_window_fallback: bool = True,
    log_spread_rejects: bool = True,
) -> dict[str, Any]:
    """Run BTC preclose paper trading in a loop for extended coverage.

    This runs the preclose scanner every interval_seconds for loop_duration_minutes,
    catching markets that enter the window during the loop period.

    Args:
        data_dir: Directory to store paper trading data
        window_seconds: Time window before close to consider markets
        cheap_price: Maximum price to consider "cheap"
        size: Position size per trade
        starting_cash: Starting cash balance
        loop_duration_minutes: How long to run the loop (default: 10 min)
        interval_seconds: Seconds between scans (default: 60)
        snapshots_dir: Directory with collector snapshots
        use_window_fallback: Whether to try progressively wider windows if no fills
        log_spread_rejects: Whether to log rejected spreads for tuning

    Returns:
        Dict with aggregated results from all iterations
    """
    start_time = datetime.now(UTC)
    end_time = start_time + timedelta(minutes=loop_duration_minutes)

    all_triggers: list[dict[str, Any]] = []
    total_scanned = 0
    total_near_close = 0
    total_fills = 0
    iterations = 0

    logger.info(
        "Starting BTC preclose loop for %d minutes, scanning every %d seconds",
        loop_duration_minutes,
        interval_seconds,
    )

    while datetime.now(UTC) < end_time:
        iterations += 1
        result = run_btc_preclose_paper(
            data_dir=data_dir,
            window_seconds=window_seconds,
            cheap_price=cheap_price,
            size=size,
            starting_cash=starting_cash,
            snapshots_dir=snapshots_dir,
            use_window_fallback=use_window_fallback,
            log_spread_rejects=log_spread_rejects,
        )

        total_scanned += result["markets_scanned"]
        total_near_close += result["candidates_near_close"]
        total_fills += result["fills_recorded"]
        all_triggers.extend(result["triggers"])

        if result["fills_recorded"] > 0:
            logger.info(
                "Loop iteration %d: %d fills recorded",
                iterations,
                result["fills_recorded"],
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
        "all_triggers": all_triggers,
    }
