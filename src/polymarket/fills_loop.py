"""Dedicated fills collection loop service.

Runs collect-fills on a cadence, logging fill age and alerting on staleness.
This ensures fills.jsonl stays current even when pnl-loop has window logic issues.
"""

from __future__ import annotations

import logging
import random
import time
from pathlib import Path
from typing import TYPE_CHECKING

from .fills_collector import collect_fills, get_fills_summary

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path("data")
DEFAULT_FILLS_PATH = Path("data/fills.jsonl")
DEFAULT_PAPER_FILLS_PATH = Path("data/paper_trading/fills.jsonl")
DEFAULT_INTERVAL_SECONDS = 300.0  # 5 minutes
DEFAULT_STALE_ALERT_HOURS = 6.0


def _send_openclaw_notification(message: str) -> None:
    """Emit OpenClaw notification via gateway if available.

    This is a best-effort notification that won't fail the loop.
    """
    try:
        # Try to import and use OpenClaw messaging
        from openclaw.messaging import send_notification

        send_notification(message)
    except Exception:
        # Fallback: just log prominently
        logger.warning("[OpenClaw Notification] %s", message)


def check_fills_staleness(
    fills_path: Path,
    stale_alert_hours: float = DEFAULT_STALE_ALERT_HOURS,
) -> dict:
    """Check if fills are stale and return status info.

    Args:
        fills_path: Path to fills.jsonl
        stale_alert_hours: Hours before considering fills stale

    Returns:
        Dict with staleness status and metrics
    """
    summary = get_fills_summary(fills_path)

    result = {
        "fills_path": str(fills_path),
        "exists": summary["exists"],
        "total_fills": summary.get("total_fills", 0),
        "last_fill_at": summary.get("last_fill_at"),
        "age_seconds": summary.get("age_seconds"),
        "age_hours": None,
        "is_stale": False,
        "stale_threshold_hours": stale_alert_hours,
    }

    if summary.get("age_seconds") is not None:
        age_hours = summary["age_seconds"] / 3600
        result["age_hours"] = age_hours
        result["is_stale"] = age_hours > stale_alert_hours

    return result


def run_collect_fills_loop(
    data_dir: Path | None = None,
    fills_path: Path | None = None,
    paper_fills_path: Path | None = None,
    interval_seconds: float = DEFAULT_INTERVAL_SECONDS,
    include_account: bool = True,
    include_paper: bool = True,
    stale_alert_hours: float = DEFAULT_STALE_ALERT_HOURS,
    on_stale_alert: Callable[[str], None] | None = None,
) -> None:
    """Run continuous fills collection loop.

    Args:
        data_dir: Base data directory
        fills_path: Output path for fills.jsonl
        paper_fills_path: Path to paper trading fills.jsonl
        interval_seconds: Seconds between collection runs
        include_account: Whether to fetch real account fills
        include_paper: Whether to include paper trading fills
        stale_alert_hours: Hours before triggering stale alert
        on_stale_alert: Optional callback for stale alerts
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    if fills_path is None:
        fills_path = data_dir / "fills.jsonl"
    if paper_fills_path is None:
        paper_fills_path = data_dir / "paper_trading" / "fills.jsonl"

    data_dir = Path(data_dir)
    fills_path = Path(fills_path)
    paper_fills_path = Path(paper_fills_path)

    logger.info(
        "Starting collect-fills loop: interval=%.0fs account=%s paper=%s",
        interval_seconds,
        include_account,
        include_paper,
    )

    iteration = 0
    while True:
        started = time.time()
        iteration += 1

        try:
            # Collect fills
            logger.debug("Iteration %d: collecting fills...", iteration)
            collect_result = collect_fills(
                fills_path=fills_path,
                paper_fills_path=paper_fills_path,
                include_account=include_account,
                include_paper=include_paper,
            )

            # Log collection results
            logger.info(
                "Collected %d fills (%d account, %d paper, %d duplicates)",
                collect_result["total_appended"],
                collect_result["account_fills"],
                collect_result["paper_fills"],
                collect_result["duplicates_skipped"],
            )

            # Check staleness
            staleness = check_fills_staleness(fills_path, stale_alert_hours)

            if staleness["age_hours"] is not None:
                logger.info(
                    "Fill age: %.2fh (threshold: %.1fh) fills=%d",
                    staleness["age_hours"],
                    stale_alert_hours,
                    staleness["total_fills"],
                )
            else:
                logger.info("No fills found yet")

            # Alert if stale
            if staleness["is_stale"]:
                age_hours = staleness["age_hours"]
                warning_msg = (
                    f"FILLS STALE: Last fill was {age_hours:.1f}h ago "
                    f"(threshold: {stale_alert_hours:.1f}h). "
                    f"Total fills: {staleness['total_fills']}"
                )
                logger.warning(warning_msg)

                # When stale, do one best-effort backfill pass with a much larger window.
                # This mitigates API pagination edge cases and inclusive/exclusive timestamp behavior.
                try:
                    logger.warning("Stale backfill: re-syncing account fills with expanded limits...")
                    collect_fills(
                        fills_path=fills_path,
                        paper_fills_path=paper_fills_path,
                        include_account=include_account,
                        include_paper=include_paper,
                        since=None,
                        account_limit=500,
                        account_max_pages=50,
                    )
                except Exception:
                    logger.exception("Stale backfill attempt failed")

                # Emit OpenClaw notification
                notification = (
                    f"🚨 Polymarket fills stale: {age_hours:.1f}h since last fill. "
                    f"Triggered one-time backfill (limit=500, pages=50)."
                )
                if on_stale_alert:
                    on_stale_alert(notification)
                else:
                    _send_openclaw_notification(notification)

        except Exception:
            logger.exception("Error in collect-fills loop iteration %d", iteration)

        # Sleep until next iteration
        elapsed = time.time() - started
        sleep_for = max(0.0, interval_seconds - elapsed)
        sleep_for += random.uniform(0.0, min(10.0, interval_seconds * 0.05))  # Small jitter
        logger.debug("Sleeping for %.1fs", sleep_for)
        time.sleep(sleep_for)
