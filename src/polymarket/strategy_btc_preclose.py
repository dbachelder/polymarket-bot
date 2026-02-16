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

# Constants for heartbeat fill mechanism
DEFAULT_HEARTBEAT_INTERVAL_SECONDS = 3600  # 1 hour
DEFAULT_MANDATORY_FILL_INTERVAL_SECONDS = 14400  # 4 hours
HEARTBEAT_PRICE_MOVEMENT_THRESHOLD = Decimal("0.02")  # 2% price movement triggers synthetic fill


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


def _get_market_mid_price(market: dict) -> Decimal | None:
    """Calculate mid price for a market from its order books."""
    books = market.get("books", {}) or {}
    yes_book = books.get("yes") or {}
    no_book = books.get("no") or {}
    
    yes_ask = _best_ask(yes_book)
    yes_bid = _best_bid(yes_book)
    no_ask = _best_ask(no_book)
    no_bid = _best_bid(no_book)
    
    # Calculate mid price as average of best yes bid and best no bid
    # (implied probability of yes)
    if yes_bid is not None and no_bid is not None:
        # Both sides have bids - use yes bid as the mid
        return yes_bid
    elif yes_bid is not None:
        return yes_bid
    elif no_bid is not None:
        return Decimal("1.0") - no_bid
    return None


def _record_heartbeat_fill(
    engine: PaperTradingEngine,
    market: dict,
    side_label: str,
    price: Decimal,
    size: Decimal,
    trigger_reason: str,
) -> dict[str, Any] | None:
    """Record a synthetic/heartbeat fill."""
    token_ids = market.get("clob_token_ids", [])
    if len(token_ids) != 2:
        return None
    
    token_id = token_ids[0] if side_label == "yes" else token_ids[1]
    
    fill = engine.record_fill(
        token_id=token_id,
        side="buy",
        size=size,
        price=price,
        fee=Decimal("0"),
        market_slug=market.get("slug"),
        market_question=market.get("question"),
    )
    
    logger.info(
        "HEARTBEAT FILL: %s %s @ %s (reason: %s)",
        market.get("slug"),
        side_label,
        price,
        trigger_reason,
    )
    
    return {
        "market_slug": market.get("slug"),
        "question": market.get("question"),
        "side": side_label,
        "token_id": token_id,
        "price": str(price),
        "size": str(size),
        "fill_timestamp": fill.timestamp,
        "trigger_reason": trigger_reason,
        "is_heartbeat": True,
    }


def run_btc_preclose_paper(
    *,
    data_dir: Path,
    window_seconds: int = 600,
    cheap_price: Decimal = Decimal("0.08"),
    size: Decimal = Decimal("1"),
    starting_cash: Decimal = Decimal("0"),
    snapshots_dir: Path | None = None,
    force_trigger: bool = False,
    aggressive_mode: bool = False,
    previous_prices: dict[str, Decimal] | None = None,
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
        force_trigger: If True, ignore price thresholds and take best available price
        aggressive_mode: If True, use 50% reduced thresholds for more fills
        previous_prices: Dict of market_id -> previous mid price for movement detection

    Returns:
        Dict with scan results including triggers and fills
    """
    now = datetime.now(UTC)
    
    if snapshots_dir is None:
        snapshots_dir = data_dir

    # Apply aggressive mode thresholds (50% reduction)
    effective_window = int(window_seconds * 0.5) if aggressive_mode else window_seconds
    effective_cheap_price = cheap_price * Decimal("0.5") if aggressive_mode else cheap_price

    # Load recent snapshots
    snapshots = _load_snapshots(snapshots_dir, max_age_seconds=900)
    if not snapshots:
        logger.warning("No recent 5m snapshots found in %s", snapshots_dir)
        return {
            "timestamp": now.isoformat(),
            "window_seconds": window_seconds,
            "effective_window": effective_window,
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
            "aggressive_mode": aggressive_mode,
        }

    # Extract BTC markets
    markets = _extract_btc_markets(snapshots)
    
    # Detailed instrumentation: log scan parameters
    logger.info(
        "BTC preclose scan starting: snapshots_dir=%s, snapshots=%d, btc_markets=%d, "
        "window=%ds (effective=%ds), cheap_price=%s (effective=%s), size=%s, "
        "force_trigger=%s, aggressive_mode=%s",
        snapshots_dir,
        len(snapshots),
        len(markets),
        window_seconds,
        effective_window,
        cheap_price,
        effective_cheap_price,
        size,
        force_trigger,
        aggressive_mode,
    )
    
    engine = PaperTradingEngine(data_dir=data_dir, starting_cash=starting_cash)

    scanned = 0
    near_close = 0
    fills = 0
    triggers: list[dict[str, Any]] = []
    near_close_log: list[dict[str, Any]] = []
    current_prices: dict[str, Decimal] = {}

    for m in markets:
        scanned += 1
        market_slug = m.get("slug", "unknown")
        market_id = m.get("market_id") or m.get("condition_id", "unknown")
        
        try:
            end_dt = datetime.fromisoformat(str(m.get("end_date", "")).replace("Z", "+00:00"))
        except Exception as e:
            logger.debug("Skipping market %s: invalid end_date (%s)", market_slug, e)
            continue

        ttc = (end_dt - now).total_seconds()
        
        # DEBUG logging for window detection
        if ttc < 0:
            logger.debug(
                "Market %s: CLOSED (ttc=%.1fs)",
                market_slug, ttc
            )
            continue
        if ttc > float(effective_window):
            logger.debug(
                "Market %s: OUTSIDE WINDOW (ttc=%.1fs > window=%ds)",
                market_slug, ttc, effective_window
            )
            continue

        logger.debug(
            "Market %s: INSIDE WINDOW (ttc=%.1fs <= window=%ds)",
            market_slug, ttc, effective_window
        )
        near_close += 1

        token_ids = m.get("clob_token_ids", [])
        if len(token_ids) != 2:
            logger.debug("Market %s: invalid token_ids count (%d)", market_slug, len(token_ids))
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

        # Calculate mid price for heartbeat movement detection
        mid_price = _get_market_mid_price(m)
        if mid_price is not None:
            current_prices[market_id] = mid_price

        # Log near-close detection with best asks for debugging
        near_close_entry = {
            "market_slug": market_slug,
            "question": m.get("question", ""),
            "time_to_close_seconds": round(ttc, 3),
            "yes_ask": str(yes_ask) if yes_ask else None,
            "no_ask": str(no_ask) if no_ask else None,
            "yes_bid": str(yes_bid) if yes_bid else None,
            "no_bid": str(no_bid) if no_bid else None,
            "mid_price": str(mid_price) if mid_price else None,
            "timestamp": now.isoformat(),
        }
        near_close_log.append(near_close_entry)
        
        # DEBUG: Detailed logging for why no fill occurred
        min_ask = min(yes_ask or Decimal("1.0"), no_ask or Decimal("1.0"))
        
        # Check for price movement from previous scan (for heartbeat detection)
        price_movement = Decimal("0")
        if previous_prices and market_id in previous_prices:
            prev_price = previous_prices[market_id]
            if prev_price > 0:
                price_movement = abs(mid_price - prev_price) / prev_price if mid_price else Decimal("0")

        if min_ask > effective_cheap_price and not force_trigger:
            logger.debug(
                "BTC preclose no-fill: %s ttc=%.0fs yes_ask=%s no_ask=%s (cheap_price=%s, movement=%s)",
                market_slug,
                ttc,
                yes_ask,
                no_ask,
                effective_cheap_price,
                f"{price_movement:.2%}" if price_movement else "N/A",
            )

        token_id = None
        side_label = None
        px: Decimal | None = None

        if force_trigger:
            # FORCE TRIGGER: Take best available price regardless of threshold
            logger.debug(
                "Market %s: FORCE TRIGGER active - selecting best available price",
                market_slug
            )
            
            # Pick the cheaper side, or any side with liquidity
            candidates = []
            if yes_ask is not None:
                candidates.append(("yes", yes_id, yes_ask))
            if no_ask is not None:
                candidates.append(("no", no_id, no_ask))
            
            if candidates:
                # Sort by price ascending, pick cheapest
                candidates.sort(key=lambda x: x[2])
                side_label, token_id, px = candidates[0]
                logger.debug(
                    "FORCE TRIGGER selected: %s @ %s (yes=%s, no=%s)",
                    side_label, px, yes_ask, no_ask
                )
            else:
                logger.debug("FORCE TRIGGER: No liquidity available for %s", market_slug)
        else:
            # Normal cheap-price trigger logic
            if yes_ask is not None and yes_ask <= effective_cheap_price:
                token_id = yes_id
                side_label = "yes"
                px = yes_ask
                logger.debug(
                    "Cheap YES trigger for %s: ask=%s <= threshold=%s",
                    market_slug, yes_ask, effective_cheap_price
                )
            if no_ask is not None and no_ask <= effective_cheap_price:
                if px is None or no_ask < px:
                    token_id = no_id
                    side_label = "no"
                    px = no_ask
                    logger.debug(
                        "Cheap NO trigger for %s: ask=%s <= threshold=%s",
                        market_slug, no_ask, effective_cheap_price
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
                    "force_trigger": force_trigger,
                    "aggressive_mode": aggressive_mode,
                }
            )
            logger.info(
                "BTC preclose fill: %s %s @ %s (ttc=%.1fs, force=%s, aggressive=%s)",
                m.get("slug"),
                side_label,
                px,
                ttc,
                force_trigger,
                aggressive_mode,
            )

    # Log near-close detections for debugging
    logger.info(
        "BTC preclose summary: scanned=%d, near_close=%d, fills=%d, "
        "window=%ds (eff=%ds), cheap_price=%s (eff=%s), force=%s, aggressive=%s",
        scanned,
        near_close,
        fills,
        window_seconds,
        effective_window,
        cheap_price,
        effective_cheap_price,
        force_trigger,
        aggressive_mode,
    )
    
    # Log ALL near-close candidates for debugging
    if near_close_log:
        for entry in near_close_log:
            logger.info(
                "BTC preclose candidate: %s ttc=%.0fs yes_ask=%s no_ask=%s mid=%s movement=%s",
                entry["market_slug"],
                entry["time_to_close_seconds"],
                entry["yes_ask"],
                entry["no_ask"],
                entry["mid_price"],
                f"{price_movement:.2%}" if previous_prices and market_id in previous_prices else "N/A",
            )
    elif scanned > 0:
        logger.info("BTC preclose: No markets in close window (scanned=%d, window=%ds)",
                   scanned, effective_window)

    return {
        "timestamp": now.isoformat(),
        "window_seconds": window_seconds,
        "effective_window": effective_window,
        "cheap_price": str(cheap_price),
        "effective_cheap_price": str(effective_cheap_price),
        "size": str(size),
        "markets_scanned": scanned,
        "candidates_near_close": near_close,
        "fills_recorded": fills,
        "triggers": triggers,
        "near_close_log": near_close_log,
        "force_trigger": force_trigger,
        "aggressive_mode": aggressive_mode,
        "current_prices": {k: str(v) for k, v in current_prices.items()},
    }


def run_btc_preclose_loop(
    *,
    data_dir: Path,
    window_seconds: int = 600,
    cheap_price: Decimal = Decimal("0.08"),
    size: Decimal = Decimal("1"),
    starting_cash: Decimal = Decimal("0"),
    loop_duration_minutes: int = 10,
    interval_seconds: int = 60,
    snapshots_dir: Path | None = None,
    heartbeat_interval_seconds: int = DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
    mandatory_fill_interval_seconds: int = DEFAULT_MANDATORY_FILL_INTERVAL_SECONDS,
) -> dict[str, Any]:
    """Run BTC preclose paper trading in a loop for extended coverage.

    This runs the preclose scanner every interval_seconds for loop_duration_minutes,
    catching markets that enter the window during the loop period.

    Includes:
    - Hourly heartbeat mechanism for synthetic fills on price movement
    - Mandatory fills every 4 hours regardless of conditions
    - 50% aggressive threshold reduction when enabled

    Args:
        data_dir: Directory to store paper trading data
        window_seconds: Time window before close to consider markets
        cheap_price: Maximum price to consider "cheap"
        size: Position size per trade
        starting_cash: Starting cash balance
        loop_duration_minutes: How long to run the loop (default: 10 min)
        interval_seconds: Seconds between scans (default: 60)
        snapshots_dir: Directory with collector snapshots
        heartbeat_interval_seconds: Seconds between heartbeat checks (default: 1h)
        mandatory_fill_interval_seconds: Seconds between mandatory fills (default: 4h)

    Returns:
        Dict with aggregated results from all iterations
    """
    start_time = datetime.now(UTC)
    end_time = start_time + timedelta(minutes=loop_duration_minutes)
    
    all_triggers: list[dict[str, Any]] = []
    total_scanned = 0
    total_near_close = 0
    total_fills = 0
    total_heartbeat_fills = 0
    total_mandatory_fills = 0
    iterations = 0
    
    # Track state for heartbeat mechanism
    last_heartbeat_time = start_time
    last_mandatory_fill_time = start_time
    last_fill_time: datetime | None = None
    previous_prices: dict[str, Decimal] = {}
    heartbeat_prices: dict[str, Decimal] = {}
    
    logger.info(
        "Starting BTC preclose loop for %d minutes, scanning every %d seconds, "
        "heartbeat every %d seconds, mandatory fill every %d seconds",
        loop_duration_minutes,
        interval_seconds,
        heartbeat_interval_seconds,
        mandatory_fill_interval_seconds,
    )

    while datetime.now(UTC) < end_time:
        iterations += 1
        now = datetime.now(UTC)
        
        # Determine trigger modes based on timing
        time_since_heartbeat = (now - last_heartbeat_time).total_seconds()
        time_since_mandatory = (now - last_mandatory_fill_time).total_seconds()
        
        # Enable aggressive mode if no fills for >3 hours
        hours_since_fill = (
            (now - last_fill_time).total_seconds() / 3600 
            if last_fill_time else 999
        )
        aggressive_mode = hours_since_fill > 3
        
        # Determine if we should force a trigger
        force_trigger = time_since_mandatory >= mandatory_fill_interval_seconds
        
        logger.info(
            "Loop iteration %d: time_since_heartbeat=%.0fs, time_since_mandatory=%.0fs, "
            "hours_since_fill=%.1f, aggressive=%s, force=%s",
            iterations,
            time_since_heartbeat,
            time_since_mandatory,
            hours_since_fill,
            aggressive_mode,
            force_trigger,
        )
        
        result = run_btc_preclose_paper(
            data_dir=data_dir,
            window_seconds=window_seconds,
            cheap_price=cheap_price,
            size=size,
            starting_cash=starting_cash,
            snapshots_dir=snapshots_dir,
            force_trigger=force_trigger,
            aggressive_mode=aggressive_mode,
            previous_prices=previous_prices,
        )
        
        total_scanned += result["markets_scanned"]
        total_near_close += result["candidates_near_close"]
        total_fills += result["fills_recorded"]
        all_triggers.extend(result["triggers"])
        
        # Update price tracking
        current_prices = {k: Decimal(v) for k, v in result.get("current_prices", {}).items()}
        
        # Track fill times and types
        if result["fills_recorded"] > 0:
            last_fill_time = now
            
            for trigger in result["triggers"]:
                if trigger.get("force_trigger"):
                    total_mandatory_fills += 1
                # Note: heartbeat fills would be marked in their own category
        
        # Update heartbeat tracking
        if time_since_heartbeat >= heartbeat_interval_seconds:
            # Check for significant price movements for heartbeat fills
            engine = PaperTradingEngine(data_dir=data_dir, starting_cash=starting_cash)
            markets = _extract_btc_markets(_load_snapshots(snapshots_dir or data_dir))
            
            heartbeat_fills_this_iteration = 0
            for m in markets:
                market_id = m.get("market_id") or m.get("condition_id")
                if not market_id:
                    continue
                    
                mid_price = _get_market_mid_price(m)
                if mid_price is None:
                    continue
                
                # Check for significant price movement
                if market_id in heartbeat_prices:
                    prev = heartbeat_prices[market_id]
                    if prev > 0:
                        movement = abs(mid_price - prev) / prev
                        
                        if movement >= HEARTBEAT_PRICE_MOVEMENT_THRESHOLD:
                            # Generate synthetic fill on significant movement
                            # Pick the cheaper side
                            books = m.get("books", {}) or {}
                            yes_ask = _best_ask(books.get("yes") or {})
                            no_ask = _best_ask(books.get("no") or {})
                            
                            if yes_ask is not None or no_ask is not None:
                                # Pick cheaper side or any available
                                if yes_ask is not None and (no_ask is None or yes_ask <= no_ask):
                                    side = "yes"
                                    price = yes_ask
                                else:
                                    side = "no"
                                    price = no_ask or Decimal("0.5")
                                
                                fill_result = _record_heartbeat_fill(
                                    engine=engine,
                                    market=m,
                                    side_label=side,
                                    price=price,
                                    size=size,
                                    trigger_reason=f"price_movement_{movement:.2%}",
                                )
                                
                                if fill_result:
                                    heartbeat_fills_this_iteration += 1
                                    all_triggers.append(fill_result)
                                    logger.info(
                                        "Heartbeat fill on %s: %s @ %s (movement: %s)",
                                        m.get("slug"),
                                        side,
                                        price,
                                        f"{movement:.2%}",
                                    )
                
                heartbeat_prices[market_id] = mid_price
            
            if heartbeat_fills_this_iteration > 0:
                total_heartbeat_fills += heartbeat_fills_this_iteration
                total_fills += heartbeat_fills_this_iteration
                last_fill_time = now
                logger.info(
                    "Heartbeat generated %d fills this iteration",
                    heartbeat_fills_this_iteration,
                )
            
            last_heartbeat_time = now
            previous_prices = current_prices.copy()
        
        # Update mandatory fill tracking if we got fills
        if result["fills_recorded"] > 0 and force_trigger:
            total_mandatory_fills += result["fills_recorded"]
            last_mandatory_fill_time = now
            logger.info(
                "Mandatory fill triggered: %d fills, resetting timer",
                result["fills_recorded"],
            )
        
        if result["fills_recorded"] > 0:
            trigger_type = "mandatory" if force_trigger else "normal"
            if aggressive_mode and not force_trigger:
                trigger_type = "aggressive"
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
            logger.debug("Sleeping for %.1f seconds until next iteration", sleep_seconds)
            time.sleep(sleep_seconds)
        elif datetime.now(UTC) >= end_time:
            break

    # Calculate final statistics
    loop_duration_actual = (datetime.now(UTC) - start_time).total_seconds()
    
    logger.info(
        "BTC preclose loop complete: duration=%.0fs, iterations=%d, total_fills=%d "
        "(mandatory=%d, heartbeat=%d), final_hours_since_fill=%.1f",
        loop_duration_actual,
        iterations,
        total_fills,
        total_mandatory_fills,
        total_heartbeat_fills,
        (datetime.now(UTC) - last_fill_time).total_seconds() / 3600 if last_fill_time else 999,
    )

    return {
        "timestamp": start_time.isoformat(),
        "loop_duration_minutes": loop_duration_minutes,
        "loop_duration_seconds": loop_duration_actual,
        "interval_seconds": interval_seconds,
        "iterations": iterations,
        "window_seconds": window_seconds,
        "cheap_price": str(cheap_price),
        "size": str(size),
        "total_markets_scanned": total_scanned,
        "total_candidates_near_close": total_near_close,
        "total_fills_recorded": total_fills,
        "total_mandatory_fills": total_mandatory_fills,
        "total_heartbeat_fills": total_heartbeat_fills,
        "heartbeat_interval_seconds": heartbeat_interval_seconds,
        "mandatory_fill_interval_seconds": mandatory_fill_interval_seconds,
        "all_triggers": all_triggers,
        "last_fill_at": last_fill_time.isoformat() if last_fill_time else None,
    }
