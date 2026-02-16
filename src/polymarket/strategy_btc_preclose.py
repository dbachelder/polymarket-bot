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
    window_seconds: int = 1200,  # Widened from 600 to 1200 (20 min window) to catch smaller moves
    cheap_price: Decimal = Decimal("0.05"),  # Lowered from 0.08 to 0.05 for higher trigger sensitivity
    size: Decimal = Decimal("1"),
    starting_cash: Decimal = Decimal("0"),
    snapshots_dir: Path | None = None,
    force_trigger: bool = False,  # Time-based backup trigger (ignore price thresholds)
) -> dict[str, Any]:
    """Paper-trade cheap-side trigger on BTC 5m markets near close.

    Uses snapshot data from collector instead of scraping (which is broken).
    
    For each market ending within `window_seconds`, fetch YES/NO books.
    If either side's best ask <= cheap_price, record a paper BUY at that ask.

    Args:
        data_dir: Directory to store paper trading data
        window_seconds: Time window before close to consider markets (default: 600 = 10 min)
        cheap_price: Maximum price to consider "cheap" (default: 0.08)
        size: Position size per trade
        starting_cash: Starting cash balance
        snapshots_dir: Directory with collector snapshots (defaults to data_dir)

    Returns:
        Dict with scan results including triggers and fills
    """
    now = datetime.now(UTC)

    if snapshots_dir is None:
        snapshots_dir = data_dir

    # Log trigger conditions for debugging
    logger.debug(
        "BTC preclose scan starting: window=%ds, cheap_price=%s, force_trigger=%s",
        window_seconds,
        cheap_price,
        force_trigger,
    )

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
            "force_trigger": force_trigger,
        }

    # Extract BTC markets
    markets = _extract_btc_markets(snapshots)
    
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
            logger.debug("Skipping market %s: invalid end_date (%s)", market_slug, e)
            continue

        ttc = (end_dt - now).total_seconds()

        # Debug logging for window detection logic
        if ttc < 0:
            logger.debug(
                "Market %s: already closed (ttc=%.1fs)",
                market_slug, ttc
            )
            continue
        if ttc > float(window_seconds):
            logger.debug(
                "Market %s: outside window (ttc=%.1fs > window=%ds)",
                market_slug, ttc, window_seconds
            )
            continue

        logger.debug(
            "Market %s: INSIDE close window (ttc=%.1fs <= window=%ds)",
            market_slug, ttc, window_seconds
        )
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
        near_close_log.append({
            "market_slug": m.get("slug", "unknown"),
            "question": m.get("question", ""),
            "time_to_close_seconds": round(ttc, 3),
            "yes_ask": str(yes_ask) if yes_ask else None,
            "no_ask": str(no_ask) if no_ask else None,
            "cheap_threshold": str(cheap_price),
            "force_trigger": force_trigger,
            "timestamp": now.isoformat(),
        })

        token_id = None
        side_label = None
        px: Decimal | None = None

        # Check price thresholds (or force trigger for time-based backup)
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
            logger.debug(
                "FORCE TRIGGER for %s: selected %s @ %s (yes=%s, no=%s)",
                market_slug, side_label, px, yes_ask, no_ask
            )
        else:
            # Normal cheap-price trigger logic
            if yes_ask is not None and yes_ask <= cheap_price:
                token_id = yes_id
                side_label = "yes"
                px = yes_ask
                logger.debug(
                    "Cheap YES trigger for %s: ask=%s <= threshold=%s",
                    market_slug, yes_ask, cheap_price
                )
            if no_ask is not None and no_ask <= cheap_price:
                if px is None or no_ask < px:
                    token_id = no_id
                    side_label = "no"
                    px = no_ask
                    logger.debug(
                        "Cheap NO trigger for %s: ask=%s <= threshold=%s",
                        market_slug, no_ask, cheap_price
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
    if near_close_log:
        logger.info(
            "BTC preclose scanned %d markets, %d near close, %d fills",
            scanned,
            near_close,
            fills,
        )
        for entry in near_close_log[:5]:  # Log first 5 for brevity
            logger.debug("Near-close: %s", entry)
    else:
        logger.debug(
            "BTC preclose: no markets in close window (scanned=%d, window=%ds)",
            scanned, window_seconds
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
        "force_trigger": force_trigger,
    }


def run_btc_preclose_loop(
    *,
    data_dir: Path,
    window_seconds: int = 1200,  # Widened from 600 to 1200 (20 min window)
    cheap_price: Decimal = Decimal("0.05"),  # Lowered from 0.08 to 0.05 for higher sensitivity
    size: Decimal = Decimal("1"),
    starting_cash: Decimal = Decimal("0"),
    loop_duration_minutes: int = 10,
    interval_seconds: int = 60,
    snapshots_dir: Path | None = None,
    backup_trigger_interval_seconds: int = 14400,  # 4 hours: time-based backup trigger
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
        backup_trigger_interval_seconds: Time-based backup trigger interval (default: 4h)

    Returns:
        Dict with aggregated results from all iterations
    """
    start_time = datetime.now(UTC)
    end_time = start_time + timedelta(minutes=loop_duration_minutes)
    last_backup_trigger = start_time

    all_triggers: list[dict[str, Any]] = []
    total_scanned = 0
    total_near_close = 0
    total_fills = 0
    total_backup_fills = 0
    iterations = 0
    backup_triggers_fired = 0

    logger.info(
        "Starting BTC preclose loop for %d minutes, scanning every %d seconds, "
        "backup trigger every %d seconds",
        loop_duration_minutes,
        interval_seconds,
        backup_trigger_interval_seconds,
    )

    while datetime.now(UTC) < end_time:
        iterations += 1

        # Check if time-based backup trigger should fire
        now = datetime.now(UTC)
        time_since_backup = (now - last_backup_trigger).total_seconds()
        use_force_trigger = time_since_backup >= backup_trigger_interval_seconds

        if use_force_trigger:
            logger.info(
                "Time-based backup trigger activated (%.0fs since last trigger)",
                time_since_backup
            )
            backup_triggers_fired += 1
            last_backup_trigger = now

        result = run_btc_preclose_paper(
            data_dir=data_dir,
            window_seconds=window_seconds,
            cheap_price=cheap_price,
            size=size,
            starting_cash=starting_cash,
            snapshots_dir=snapshots_dir,
            force_trigger=use_force_trigger,
        )
        
        total_scanned += result["markets_scanned"]
        total_near_close += result["candidates_near_close"]
        total_fills += result["fills_recorded"]
        all_triggers.extend(result["triggers"])

        if use_force_trigger and result["fills_recorded"] > 0:
            total_backup_fills += result["fills_recorded"]

        if result["fills_recorded"] > 0:
            trigger_type = "backup" if use_force_trigger else "normal"
            logger.info(
                "Loop iteration %d: %d fills recorded (%s trigger)",
                iterations,
                result["fills_recorded"],
                trigger_type,
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
        "total_backup_fills": total_backup_fills,
        "backup_triggers_fired": backup_triggers_fired,
        "backup_trigger_interval_seconds": backup_trigger_interval_seconds,
        "all_triggers": all_triggers,
    }
