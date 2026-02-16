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


def _best_bid(book: dict) -> Decimal | None:
    bids = (book or {}).get("bids") or []
    if not bids:
        return None
    try:
        return max(Decimal(str(b["price"])) for b in bids)
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
    window_seconds: int = 600,  # Increased from 300 to 600 (10 min window)
    cheap_price: Decimal = Decimal("0.08"),  # Increased from 0.05 to 0.08
    fair_price_threshold: Decimal = Decimal("0.35"),
    wide_spread_threshold: Decimal = Decimal("0.10"),
    size: Decimal = Decimal("1"),
    starting_cash: Decimal = Decimal("0"),
    snapshots_dir: Path | None = None,
) -> dict[str, Any]:
    """Paper-trade cheap-side trigger on BTC 5m markets near close.

    Uses snapshot data from collector instead of scraping (which is broken).

    For each market ending within `window_seconds`, fetch YES/NO books.
    If either side's best ask <= cheap_price, record a paper BUY at that ask.

    Secondary entry: If spread > wide_spread_threshold and price <= fair_price_threshold,
    also trigger entry (captures wide spreads near close).

    Args:
        data_dir: Directory to store paper trading data
        window_seconds: Time window before close to consider markets (default: 600 = 10 min)
        cheap_price: Maximum price to consider "cheap" (default: 0.08)
        fair_price_threshold: Maximum price for wide-spread entry (default: 0.35)
        wide_spread_threshold: Spread > this triggers fair-price entry (default: 0.10)
        size: Position size per trade
        starting_cash: Starting cash balance
        snapshots_dir: Directory with collector snapshots (defaults to data_dir)

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
            "fair_price_threshold": str(fair_price_threshold),
            "wide_spread_threshold": str(wide_spread_threshold),
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
    
    engine = PaperTradingEngine(data_dir=data_dir, starting_cash=starting_cash)

    scanned = 0
    near_close = 0
    fills = 0
    triggers: list[dict[str, Any]] = []
    near_close_log: list[dict[str, Any]] = []

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
        yes_bid = _best_bid(yes_book)
        no_bid = _best_bid(no_book)

        # Calculate spreads for secondary entry logic
        yes_spread = (yes_ask - yes_bid) if yes_ask is not None and yes_bid is not None else None
        no_spread = (no_ask - no_bid) if no_ask is not None and no_bid is not None else None

        # Log near-close detection with best asks for debugging
        near_close_log.append({
            "market_slug": m.get("slug", "unknown"),
            "question": m.get("question", ""),
            "time_to_close_seconds": round(ttc, 3),
            "yes_ask": str(yes_ask) if yes_ask else None,
            "yes_bid": str(yes_bid) if yes_bid else None,
            "yes_spread": str(yes_spread) if yes_spread else None,
            "no_ask": str(no_ask) if no_ask else None,
            "no_bid": str(no_bid) if no_bid else None,
            "no_spread": str(no_spread) if no_spread else None,
            "timestamp": now.isoformat(),
        })

        token_id = None
        side_label = None
        px: Decimal | None = None
        trigger_reason = None

        # Primary entry: cheap price
        if yes_ask is not None and yes_ask <= cheap_price:
            token_id = yes_id
            side_label = "yes"
            px = yes_ask
            trigger_reason = "cheap"
        if no_ask is not None and no_ask <= cheap_price:
            if px is None or no_ask < px:
                token_id = no_id
                side_label = "no"
                px = no_ask
                trigger_reason = "cheap"

        # Secondary entry: wide spread + fair price
        if token_id is None:
            # Check YES side for wide spread entry
            if yes_ask is not None and yes_spread is not None:
                if yes_spread > wide_spread_threshold and yes_ask <= fair_price_threshold:
                    token_id = yes_id
                    side_label = "yes"
                    px = yes_ask
                    trigger_reason = "wide_spread"
                    logger.debug(
                        "Wide spread entry (YES): %s spread=%.3f > %.3f, ask=%.3f <= %.3f",
                        m.get("slug"), yes_spread, wide_spread_threshold, yes_ask, fair_price_threshold
                    )

            # Check NO side for wide spread entry
            if token_id is None and no_ask is not None and no_spread is not None:
                if no_spread > wide_spread_threshold and no_ask <= fair_price_threshold:
                    token_id = no_id
                    side_label = "no"
                    px = no_ask
                    trigger_reason = "wide_spread"
                    logger.debug(
                        "Wide spread entry (NO): %s spread=%.3f > %.3f, ask=%.3f <= %.3f",
                        m.get("slug"), no_spread, wide_spread_threshold, no_ask, fair_price_threshold
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
                "BTC preclose fill: %s %s @ %s (ttc=%.1fs, reason=%s)",
                m.get("slug"),
                side_label,
                px,
                ttc,
                trigger_reason,
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

    return {
        "timestamp": now.isoformat(),
        "window_seconds": window_seconds,
        "cheap_price": str(cheap_price),
        "fair_price_threshold": str(fair_price_threshold),
        "wide_spread_threshold": str(wide_spread_threshold),
        "size": str(size),
        "markets_scanned": scanned,
        "candidates_near_close": near_close,
        "fills_recorded": fills,
        "triggers": triggers,
        "near_close_log": near_close_log,
    }


def run_btc_preclose_loop(
    *,
    data_dir: Path,
    window_seconds: int = 600,
    cheap_price: Decimal = Decimal("0.08"),
    fair_price_threshold: Decimal = Decimal("0.35"),
    wide_spread_threshold: Decimal = Decimal("0.10"),
    size: Decimal = Decimal("1"),
    starting_cash: Decimal = Decimal("0"),
    loop_duration_minutes: int = 10,
    interval_seconds: int = 60,
    snapshots_dir: Path | None = None,
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
    logger.info(
        "Thresholds: cheap=%s, fair_price=%s, wide_spread=%s",
        cheap_price,
        fair_price_threshold,
        wide_spread_threshold,
    )

    while datetime.now(UTC) < end_time:
        iterations += 1
        result = run_btc_preclose_paper(
            data_dir=data_dir,
            window_seconds=window_seconds,
            cheap_price=cheap_price,
            fair_price_threshold=fair_price_threshold,
            wide_spread_threshold=wide_spread_threshold,
            size=size,
            starting_cash=starting_cash,
            snapshots_dir=snapshots_dir,
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
        "fair_price_threshold": str(fair_price_threshold),
        "wide_spread_threshold": str(wide_spread_threshold),
        "size": str(size),
        "total_markets_scanned": total_scanned,
        "total_candidates_near_close": total_near_close,
        "total_fills_recorded": total_fills,
        "all_triggers": all_triggers,
    }
