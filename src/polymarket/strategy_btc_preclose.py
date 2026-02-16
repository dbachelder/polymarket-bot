from __future__ import annotations

import logging
import time
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

from .paper_trading import PaperTradingEngine
from .site import extract_5m_markets, fetch_predictions_page, parse_next_data

logger = logging.getLogger(__name__)


def _best_ask(book: dict) -> Decimal | None:
    asks = (book or {}).get("asks") or []
    if not asks:
        return None
    try:
        return min(Decimal(str(a["price"])) for a in asks)
    except Exception:
        return None


def run_btc_preclose_paper(
    *,
    data_dir: Path,
    window_seconds: int = 900,  # Wider window to increase hit-rate (15 min)
    cheap_price: Decimal = Decimal("0.10"),  # Loosen trigger to ensure daily paper fills
    size: Decimal = Decimal("1"),
    starting_cash: Decimal = Decimal("1000"),
) -> dict[str, Any]:
    """Paper-trade cheap-side trigger on BTC 5m markets near close.

    For each market on /predictions/5M ending within `window_seconds`, fetch YES/NO books.
    If either side's best ask <= cheap_price, record a paper BUY at that ask.

    This produces fills so we can verify whether near-close "cheap side" has edge.

    Args:
        data_dir: Directory to store paper trading data
        window_seconds: Time window before close to consider markets (default: 900 = 15 min)
        cheap_price: Maximum price to consider "cheap" (default: 0.10)
        size: Position size per trade
        starting_cash: Starting cash balance (default: 1000)

    Returns:
        Dict with scan results including triggers and fills
    """

    now = datetime.now(UTC)

    html = fetch_predictions_page("5M")
    nd = parse_next_data(html)
    markets = extract_5m_markets(nd)

    engine = PaperTradingEngine(data_dir=data_dir, starting_cash=starting_cash)

    scanned = 0
    near_close = 0
    fills = 0
    triggers: list[dict[str, Any]] = []
    near_close_log: list[dict[str, Any]] = []

    # Import here so the module stays pure / testable.
    from .clob import get_book

    for m in markets:
        scanned += 1
        try:
            end_dt = datetime.fromisoformat(str(m.end_date).replace("Z", "+00:00"))
        except Exception:
            continue

        ttc = (end_dt - now).total_seconds()
        if ttc < 0 or ttc > float(window_seconds):
            continue

        near_close += 1

        yes_id, no_id = m.clob_token_ids
        yes_book = get_book(yes_id)
        no_book = get_book(no_id)

        yes_ask = _best_ask(yes_book)
        no_ask = _best_ask(no_book)

        # Log near-close detection with best asks for debugging
        near_close_log.append({
            "market_slug": m.slug,
            "question": m.question,
            "time_to_close_seconds": round(ttc, 3),
            "yes_ask": str(yes_ask) if yes_ask else None,
            "no_ask": str(no_ask) if no_ask else None,
            "timestamp": now.isoformat(),
        })

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
                market_slug=m.slug,
                market_question=m.question,
            )
            fills += 1
            triggers.append(
                {
                    "market_slug": m.slug,
                    "question": m.question,
                    "ends_at": m.end_date,
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
                m.slug,
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
    }


def run_btc_preclose_loop(
    *,
    data_dir: Path,
    window_seconds: int = 900,
    cheap_price: Decimal = Decimal("0.10"),
    size: Decimal = Decimal("1"),
    starting_cash: Decimal = Decimal("1000"),
    loop_duration_minutes: int = 10,
    interval_seconds: int = 60,
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
