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
    files = list(snapshots_dir.glob("snapshot_5m_*.json")) + list(snapshots_dir.glob("snapshot_15m_*.json"))

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
    window_seconds: int = 1800,  # Widened to 1800s (30 min window) for more opportunities
    cheap_price: Decimal = Decimal("0.15"),  # Lowered from 0.25 to 0.15 for more fills
    size: Decimal = Decimal("1"),
    starting_cash: Decimal = Decimal("0"),
    snapshots_dir: Path | None = None,
    verbose_tick: bool = True,  # Log every tick with market data
) -> dict[str, Any]:
    """Paper-trade cheap-side trigger on BTC 5m markets near close.

    Uses snapshot data from collector instead of scraping (which is broken).

    For each market ending within `window_seconds`, fetch YES/NO books.
    If either side's best ask <= cheap_price, record a paper BUY at that ask.

    Args:
        data_dir: Directory to store paper trading data
        window_seconds: Time window before close to consider markets (default: 1800 = 30 min)
        cheap_price: Maximum price to consider "cheap" (default: 0.15)
        size: Position size per trade
        starting_cash: Starting cash balance
        snapshots_dir: Directory with collector snapshots (defaults to data_dir)
        verbose_tick: Log time_to_close and best bid/ask every tick

    Returns:
        Dict with scan results including triggers and fills
    """
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
            "near_misses": [],
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
    near_misses: list[dict[str, Any]] = []  # Track near-miss opportunities

    for m in markets:
        scanned += 1
        market_slug = m.get("slug", "unknown")
        
        try:
            end_dt = datetime.fromisoformat(str(m.get("end_date", "")).replace("Z", "+00:00"))
        except Exception as e:
            logger.debug("BTC preclose near-miss: %s - failed to parse end_date: %s", market_slug, e)
            near_misses.append({
                "market_slug": market_slug,
                "reason": "parse_error",
                "details": f"Failed to parse end_date: {e}",
                "timestamp": now.isoformat(),
            })
            continue

        ttc = (end_dt - now).total_seconds()
        
        # Debug: Log all markets and their time to close
        if ttc > 0:
            logger.debug(
                "BTC preclose evaluating: %s ttc=%.0fs (window=%ds)",
                market_slug, ttc, window_seconds
            )
        
        if ttc < 0:
            # Market already closed
            logger.debug(
                "BTC preclose near-miss: %s - market already closed (ttc=%.0fs)",
                market_slug, ttc
            )
            near_misses.append({
                "market_slug": market_slug,
                "reason": "already_closed",
                "time_to_close_seconds": round(ttc, 3),
                "timestamp": now.isoformat(),
            })
            continue
            
        if ttc > float(window_seconds):
            # Outside window
            logger.debug(
                "BTC preclose near-miss: %s - outside window (ttc=%.0fs > %ds)",
                market_slug, ttc, window_seconds
            )
            near_misses.append({
                "market_slug": market_slug,
                "reason": "outside_window",
                "time_to_close_seconds": round(ttc, 3),
                "window_seconds": window_seconds,
                "timestamp": now.isoformat(),
            })
            continue

        near_close += 1

        token_ids = m.get("clob_token_ids", [])
        if len(token_ids) != 2:
            logger.debug(
                "BTC preclose near-miss: %s - invalid token_ids count: %d",
                market_slug, len(token_ids)
            )
            near_misses.append({
                "market_slug": market_slug,
                "reason": "invalid_tokens",
                "token_count": len(token_ids),
                "timestamp": now.isoformat(),
            })
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
            "market_slug": market_slug,
            "question": m.get("question", ""),
            "time_to_close_seconds": round(ttc, 3),
            "yes_ask": str(yes_ask) if yes_ask else None,
            "no_ask": str(no_ask) if no_ask else None,
            "cheap_price": str(cheap_price),
            "timestamp": now.isoformat(),
        }
        near_close_log.append(near_close_entry)
        
        # Detailed logging for every tick (verbose instrumentation)
        min_ask = min(yes_ask or Decimal("1.0"), no_ask or Decimal("1.0"))
        if verbose_tick:
            logger.info(
                "BTC preclose tick: %s ttc=%.0fs yes_ask=%s no_ask=%s min_ask=%s cheap=%s fill=%s",
                market_slug,
                ttc,
                yes_ask,
                no_ask,
                min_ask,
                cheap_price,
                "YES" if min_ask <= cheap_price else "NO",
            )

        token_id = None
        side_label = None
        px: Decimal | None = None

        if yes_ask is not None and yes_ask <= cheap_price:
            token_id = yes_id
            side_label = "yes"
            px = yes_ask
        if no_ask is not None and no_ask <= cheap_price:
            if px is None or no_ask < px:
                token_id = no_id
                side_label = "no"
                px = no_ask

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
        else:
            # Near-miss: within window but price too high
            reason = "price_too_high"
            if yes_ask is None and no_ask is None:
                reason = "no_liquidity"
            elif yes_ask is None:
                reason = "no_yes_liquidity"
            elif no_ask is None:
                reason = "no_no_liquidity"
            
            logger.debug(
                "BTC preclose near-miss: %s - %s (yes_ask=%s, no_ask=%s, cheap=%s)",
                market_slug, reason, yes_ask, no_ask, cheap_price
            )
            near_misses.append({
                "market_slug": market_slug,
                "reason": reason,
                "time_to_close_seconds": round(ttc, 3),
                "yes_ask": str(yes_ask) if yes_ask else None,
                "no_ask": str(no_ask) if no_ask else None,
                "cheap_price": str(cheap_price),
                "min_ask": str(min_ask) if min_ask < Decimal("1.0") else None,
                "price_gap": str(min_ask - cheap_price) if min_ask < Decimal("1.0") else None,
                "timestamp": now.isoformat(),
            })

    # Log near-close detections for debugging
    logger.info(
        "BTC preclose summary: scanned=%d, near_close=%d, fills=%d, window=%ds, cheap_price=%s",
        scanned,
        near_close,
        fills,
        window_seconds,
        cheap_price,
    )
    
    # Log near-miss summary
    if near_misses:
        by_reason: dict[str, int] = {}
        for nm in near_misses:
            reason = nm["reason"]
            by_reason[reason] = by_reason.get(reason, 0) + 1
        
        logger.info(
            "BTC preclose near-miss summary: total=%d, by_reason=%s",
            len(near_misses), by_reason
        )
        
        # Log price_too_high misses with their gaps (for threshold tuning)
        price_misses = [nm for nm in near_misses if nm["reason"] == "price_too_high"]
        if price_misses:
            gaps = [Decimal(nm["price_gap"]) for nm in price_misses if nm.get("price_gap")]
            if gaps:
                avg_gap = sum(gaps) / len(gaps)
                min_gap = min(gaps)
                logger.info(
                    "BTC preclose price gaps: count=%d, avg=%s, min=%s - "
                    "consider lowering cheap_price threshold",
                    len(gaps), round(avg_gap, 4), round(min_gap, 4)
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

    # Update last run timestamp for fills_loop fail-safe tracking
    _update_btc_preclose_last_run(data_dir)

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
        "near_misses": near_misses,
        "near_miss_count": len(near_misses),
        "near_miss_by_reason": {
            reason: len([nm for nm in near_misses if nm["reason"] == reason])
            for reason in set(nm["reason"] for nm in near_misses)
        } if near_misses else {},
    }


def _update_btc_preclose_last_run(data_dir: Path) -> None:
    """Update timestamp file for fills_loop fail-safe tracking.

    Args:
        data_dir: Data directory to write state file
    """
    state_file = data_dir / "btc_preclose_last_run.txt"
    try:
        state_file.parent.mkdir(parents=True, exist_ok=True)
        state_file.write_text(datetime.now(UTC).isoformat())
        logger.debug("Updated btc_preclose_last_run state file")
    except OSError as e:
        logger.warning("Failed to write btc_preclose state file: %s", e)


def run_btc_preclose_loop(
    *,
    data_dir: Path,
    window_seconds: int = 1800,  # 30 min window (widened)
    cheap_price: Decimal = Decimal("0.15"),  # Lowered from 0.25 for more fills
    size: Decimal = Decimal("1"),
    starting_cash: Decimal = Decimal("0"),
    loop_duration_minutes: int = 30,  # Run for 30 min (final 30m before close)
    interval_seconds: int = 60,  # 60s cadence as specified
    snapshots_dir: Path | None = None,
    final_window_minutes: int = 30,  # Final window to focus on
) -> dict[str, Any]:
    """Run BTC preclose paper trading in a loop for extended coverage.

    This runs the preclose scanner every interval_seconds for loop_duration_minutes,
    catching markets that enter the window during the loop period. Designed to run
    at 60s cadence for the final 30m before market close.

    Args:
        data_dir: Directory to store paper trading data
        window_seconds: Time window before close to consider markets (default: 1800 = 30 min)
        cheap_price: Maximum price to consider "cheap" (default: 0.15)
        size: Position size per trade
        starting_cash: Starting cash balance
        loop_duration_minutes: How long to run the loop (default: 30 min)
        interval_seconds: Seconds between scans (default: 60)
        snapshots_dir: Directory with collector snapshots
        final_window_minutes: Final window to focus on (default: 30 min)

    Returns:
        Dict with aggregated results from all iterations
    """
    start_time = datetime.now(UTC)
    end_time = start_time + timedelta(minutes=loop_duration_minutes)
    
    all_triggers: list[dict[str, Any]] = []
    total_scanned = 0
    total_near_close = 0
    total_fills = 0
    total_near_misses = 0
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
        )
        
        total_scanned += result["markets_scanned"]
        total_near_close += result["candidates_near_close"]
        total_fills += result["fills_recorded"]
        total_near_misses += result.get("near_miss_count", 0)
        all_triggers.extend(result["triggers"])
        
        if result["fills_recorded"] > 0:
            logger.info(
                "Loop iteration %d: %d fills recorded",
                iterations,
                result["fills_recorded"],
            )
        
        # Log near-miss info for debugging
        if result.get("near_miss_count", 0) > 0:
            logger.debug(
                "Loop iteration %d: %d near-misses (by_reason: %s)",
                iterations,
                result["near_miss_count"],
                result.get("near_miss_by_reason", {}),
            )
        
        # Sleep until next iteration
        next_run = datetime.now(UTC) + timedelta(seconds=interval_seconds)
        sleep_seconds = (next_run - datetime.now(UTC)).total_seconds()
        if sleep_seconds > 0 and datetime.now(UTC) + timedelta(seconds=sleep_seconds) < end_time:
            time.sleep(sleep_seconds)
        elif datetime.now(UTC) >= end_time:
            break

    logger.info(
        "BTC preclose loop complete: iterations=%d, fills=%d, near_misses=%d",
        iterations, total_fills, total_near_misses
    )

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
        "total_near_misses": total_near_misses,
        "all_triggers": all_triggers,
    }
