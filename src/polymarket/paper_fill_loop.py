"""Paper fill testbed loop - supervised continuous paper trading.

This module provides a supervised loop that:
1. Runs every 60 seconds
2. Attempts paper fills on markets nearing close (final 30 minutes)
3. Uses progressively relaxed thresholds (bounded) if no fills
4. Emits daily metrics for monitoring
"""

from __future__ import annotations

import json
import logging
import random
import time
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .fills_collector import get_fills_summary
from .strategy_btc_preclose import run_btc_preclose_paper

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path("data")
DEFAULT_PAPER_DIR = Path("data/paper_trading")
DEFAULT_SNAPSHOTS_DIR = Path("data")
DEFAULT_INTERVAL_SECONDS = 60  # Run every 60s as specified
DEFAULT_WINDOW_SECONDS = 1800  # 30 minutes before close
DEFAULT_FINAL_WINDOW_MINUTES = 30  # Final window to focus on

# Progressive threshold relaxation settings
DEFAULT_CHEAP_PRICE = Decimal("0.15")  # Starting threshold
DEFAULT_SIZE = Decimal("1")  # Bounded size cap
MIN_CHEAP_PRICE = Decimal("0.02")  # Hard floor (bounded)
MIN_WINDOW_SECONDS = 300  # Minimum 5 minute window
MAX_RELAXATION_STEPS = 5  # Max steps of relaxation
RELAXATION_FACTOR = Decimal("0.85")  # Reduce threshold by 15% each step
WINDOW_EXTENSION_FACTOR = 1.2  # Extend window by 20% each step

# Daily metric settings
DAILY_METRIC_HOUR = 0  # Emit daily metric at midnight UTC


def count_fills_last_24h(fills_path: Path) -> int:
    """Count fills appended in the last 24 hours.

    Args:
        fills_path: Path to fills.jsonl

    Returns:
        Number of fills in the last 24 hours
    """
    if not fills_path.exists():
        return 0

    cutoff = datetime.now(UTC) - timedelta(hours=24)
    count = 0

    with open(fills_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                ts_str = data.get("timestamp") or data.get("created_at")
                if ts_str:
                    try:
                        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                        if ts >= cutoff:
                            count += 1
                    except ValueError:
                        continue
            except json.JSONDecodeError:
                continue

    return count


def get_daily_metric(fills_path: Path) -> dict[str, Any]:
    """Get daily fill metric.

    Args:
        fills_path: Path to fills.jsonl

    Returns:
        Dict with daily metric data
    """
    fills_24h = count_fills_last_24h(fills_path)
    summary = get_fills_summary(fills_path)

    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "fills_appended_last_24h": fills_24h,
        "total_fills": summary.get("total_fills", 0),
        "last_fill_at": summary.get("last_fill_at"),
        "fills_path": str(fills_path),
        "alert": fills_24h == 0,  # Alert if 0 fills in last 24h
    }


def emit_daily_metric(
    fills_path: Path,
    on_alert: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Emit one-line daily metric and alert if 0 fills.

    Args:
        fills_path: Path to fills.jsonl
        on_alert: Optional callback for alerts

    Returns:
        Dict with metric data
    """
    metric = get_daily_metric(fills_path)

    # One-line metric format
    logger.info(
        "DAILY_METRIC: fills_appended_last_24h=%d total_fills=%d last_fill=%s",
        metric["fills_appended_last_24h"],
        metric["total_fills"],
        metric["last_fill_at"] or "N/A",
    )

    # Alert if 0 fills
    if metric["alert"]:
        alert_msg = (
            "🚨 ALERT: No fills recorded in last 24h. "
            "Paper fill testbed may need attention."
        )
        logger.warning(alert_msg)

        if on_alert:
            on_alert(alert_msg)
        else:
            # Try OpenClaw notification
            try:
                from openclaw.messaging import send_notification

                send_notification(alert_msg)
            except Exception:
                pass  # Best effort

    return metric


def calculate_relaxed_thresholds(
    base_cheap_price: Decimal,
    base_window_seconds: int,
    relaxation_step: int,
) -> tuple[Decimal, int]:
    """Calculate progressively relaxed thresholds.

    Args:
        base_cheap_price: Base cheap price threshold
        base_window_seconds: Base window in seconds
        relaxation_step: Current relaxation step (0 = no relaxation)

    Returns:
        Tuple of (relaxed_price, relaxed_window)
    """
    if relaxation_step <= 0:
        return base_cheap_price, base_window_seconds

    # Calculate relaxation (bounded)
    price_multiplier = RELAXATION_FACTOR ** relaxation_step
    window_multiplier = WINDOW_EXTENSION_FACTOR ** relaxation_step

    new_price = max(
        base_cheap_price * price_multiplier,
        MIN_CHEAP_PRICE,
    )
    new_window = min(
        int(base_window_seconds * window_multiplier),
        3600,  # Max 1 hour window
    )

    return new_price, new_window


def run_paper_fill_iteration(
    data_dir: Path,
    snapshots_dir: Path,
    cheap_price: Decimal,
    window_seconds: int,
    size: Decimal,
    relaxation_step: int = 0,
) -> dict[str, Any]:
    """Run a single paper fill iteration with relaxed thresholds.

    Args:
        data_dir: Directory for paper trading data
        snapshots_dir: Directory with snapshots
        cheap_price: Base cheap price threshold
        window_seconds: Base window in seconds
        size: Position size cap
        relaxation_step: Relaxation step (0 = base thresholds)

    Returns:
        Dict with iteration results
    """
    # Apply relaxation
    relaxed_price, relaxed_window = calculate_relaxed_thresholds(
        cheap_price, window_seconds, relaxation_step
    )

    logger.debug(
        "Paper fill iteration (relaxation=%d): price=%s window=%ds",
        relaxation_step,
        relaxed_price,
        relaxed_window,
    )

    # Run paper trading
    result = run_btc_preclose_paper(
        data_dir=data_dir,
        window_seconds=relaxed_window,
        cheap_price=relaxed_price,
        size=size,
        snapshots_dir=snapshots_dir,
        verbose_tick=False,  # Reduce noise in loop mode
    )

    result["relaxation_step"] = relaxation_step
    result["thresholds_used"] = {
        "cheap_price": str(relaxed_price),
        "window_seconds": relaxed_window,
    }

    return result


def run_paper_fill_testbed_loop(
    data_dir: Path | None = None,
    paper_dir: Path | None = None,
    snapshots_dir: Path | None = None,
    interval_seconds: float = DEFAULT_INTERVAL_SECONDS,
    window_seconds: int = DEFAULT_WINDOW_SECONDS,
    cheap_price: Decimal = DEFAULT_CHEAP_PRICE,
    size: Decimal = DEFAULT_SIZE,
    max_relaxation_steps: int = MAX_RELAXATION_STEPS,
    on_daily_metric: Callable[[dict], None] | None = None,
    on_alert: Callable[[str], None] | None = None,
) -> None:
    """Run supervised paper fill testbed loop.

    This loop runs continuously, attempting to generate paper fills every
    60 seconds. It progressively relaxes thresholds if no fills are generated,
    bounded by MIN_CHEAP_PRICE and max window limits.

    Daily metrics are emitted at midnight UTC.

    Args:
        data_dir: Base data directory
        paper_dir: Paper trading data directory
        snapshots_dir: Directory with collector snapshots
        interval_seconds: Seconds between iterations (default: 60)
        window_seconds: Time window before close (default: 1800 = 30min)
        cheap_price: Starting cheap price threshold (default: 0.15)
        size: Position size cap (default: 1)
        max_relaxation_steps: Max threshold relaxation steps
        on_daily_metric: Optional callback for daily metrics
        on_alert: Optional callback for alerts
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    if paper_dir is None:
        paper_dir = DEFAULT_PAPER_DIR
    if snapshots_dir is None:
        snapshots_dir = DEFAULT_SNAPSHOTS_DIR

    data_dir = Path(data_dir)
    paper_dir = Path(paper_dir)
    snapshots_dir = Path(snapshots_dir)
    fills_path = paper_dir / "fills.jsonl"

    paper_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Starting paper fill testbed loop: interval=%.0fs window=%ds price=%s size=%s",
        interval_seconds,
        window_seconds,
        cheap_price,
        size,
    )

    iteration = 0
    relaxation_step = 0
    consecutive_no_fills = 0
    last_daily_metric_date = None

    while True:
        started = time.time()
        iteration += 1

        try:
            # Check if it's time for daily metric (once per day at midnight UTC)
            now = datetime.now(UTC)
            today = now.date()

            if last_daily_metric_date != today and now.hour >= DAILY_METRIC_HOUR:
                metric = emit_daily_metric(fills_path, on_alert=on_alert)
                if on_daily_metric:
                    on_daily_metric(metric)
                last_daily_metric_date = today

            # Run paper fill iteration with relaxed thresholds
            result = run_paper_fill_iteration(
                data_dir=paper_dir,
                snapshots_dir=snapshots_dir,
                cheap_price=cheap_price,
                window_seconds=window_seconds,
                size=size,
                relaxation_step=relaxation_step,
            )

            fills_recorded = result.get("fills_recorded", 0)

            if fills_recorded > 0:
                logger.info(
                    "Paper fill success: %d fills (relaxation=%d)",
                    fills_recorded,
                    relaxation_step,
                )
                # Reset relaxation on success
                relaxation_step = 0
                consecutive_no_fills = 0
            else:
                consecutive_no_fills += 1

                # Progressively relax thresholds if no fills
                if consecutive_no_fills >= 3 and relaxation_step < max_relaxation_steps:
                    relaxation_step += 1
                    new_price, new_window = calculate_relaxed_thresholds(
                        cheap_price, window_seconds, relaxation_step
                    )
                    logger.info(
                        "Relaxing thresholds (step %d/%d): price=%s window=%ds",
                        relaxation_step,
                        max_relaxation_steps,
                        new_price,
                        new_window,
                    )

            # Log iteration summary
            logger.debug(
                "Iteration %d: scanned=%d near_close=%d fills=%d relaxation=%d",
                iteration,
                result.get("markets_scanned", 0),
                result.get("candidates_near_close", 0),
                fills_recorded,
                relaxation_step,
            )

        except Exception:
            logger.exception("Error in paper fill loop iteration %d", iteration)

        # Sleep until next iteration
        elapsed = time.time() - started
        sleep_for = max(0.0, interval_seconds - elapsed)
        sleep_for += random.uniform(0.0, min(5.0, interval_seconds * 0.1))  # Small jitter
        time.sleep(sleep_for)
