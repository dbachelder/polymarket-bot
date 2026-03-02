"""Dedicated fills collection loop service.

Runs collect-fills on a cadence, logging fill age and alerting on staleness.
This ensures fills.jsonl stays current even when pnl-loop has window logic issues.
"""

from __future__ import annotations

import logging
import os
import random
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from .fills_collector import AuthenticationError, collect_fills, get_fills_summary

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path("data")
DEFAULT_FILLS_PATH = Path("data/fills.jsonl")
DEFAULT_PAPER_FILLS_PATH = Path("data/paper_trading/fills.jsonl")
DEFAULT_INTERVAL_SECONDS = 300.0  # 5 minutes
DEFAULT_STALE_ALERT_HOURS = 6.0

# Auto-widening thresholds for stale fills
STALE_THRESHOLD_HOURS = 6.0  # Time without fills before considering stale
WIDEN_FACTOR = 1.15  # 15% increase per adjustment
WIDEN_JITTER = 0.05  # +/- 5% jitter
MAX_LOOKBACK_MULTIPLIER = 3.0  # Max 3x original lookback

# Fail-safe thresholds for forced collection when stale
FAILSAFE_MAX_LOOKBACK_MULTIPLIER = 5.0  # Widen to 5x during failsafe
EMPTY_CYCLE_ALERT_THRESHOLD = 2  # Alert after 2 consecutive empty cycles
BTC_PRECLOSE_STATE_FILE = "btc_preclose_last_run.txt"


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


def _log_env_verification() -> dict:
    """Log environment configuration verification for cron context debugging.

    Returns:
        Dict with env verification status
    """
    env_status = {
        "POLYMARKET_API_KEY": "set" if os.getenv("POLYMARKET_API_KEY") else "NOT SET",
        "POLYMARKET_API_SECRET": "set" if os.getenv("POLYMARKET_API_SECRET") else "NOT SET",
        "POLYMARKET_API_PASSPHRASE": "set" if os.getenv("POLYMARKET_API_PASSPHRASE") else "NOT SET",
        "POLYMARKET_DRY_RUN": os.getenv("POLYMARKET_DRY_RUN", "not set"),
        "PATH": os.getenv("PATH", "not set"),
        "VIRTUAL_ENV": os.getenv("VIRTUAL_ENV", "not set"),
        "PWD": os.getenv("PWD", "not set"),
    }

    logger.info("=" * 60)
    logger.info("ENVIRONMENT VERIFICATION (cron context check)")
    logger.info("=" * 60)
    for key, value in env_status.items():
        if key in ("POLYMARKET_API_KEY", "POLYMARKET_API_SECRET", "POLYMARKET_API_PASSPHRASE"):
            # Mask sensitive values
            masked = "****" + value[-4:] if len(value) > 4 and value != "NOT SET" else value
            logger.info("  %s: %s", key, masked)
        else:
            logger.info("  %s: %s", key, value)
    logger.info("=" * 60)

    # Check for common cron issues
    warnings = []
    if env_status["POLYMARKET_API_KEY"] == "NOT SET":
        warnings.append("POLYMARKET_API_KEY not set - .env file may not be loaded in cron context")
    if env_status["POLYMARKET_API_SECRET"] == "NOT SET":
        warnings.append("POLYMARKET_API_SECRET not set - .env file may not be loaded in cron context")
    if env_status["POLYMARKET_API_PASSPHRASE"] == "NOT SET":
        warnings.append("POLYMARKET_API_PASSPHRASE not set - .env file may not be loaded in cron context")

    if warnings:
        logger.warning("ENVIRONMENT WARNINGS:")
        for warning in warnings:
            logger.warning("  - %s", warning)

    return {"status": "verified", "warnings": warnings, "env": env_status}


def _get_btc_preclose_last_run(data_dir: Path) -> datetime | None:
    """Get timestamp of last btc-preclose run from state file.

    Args:
        data_dir: Data directory containing state file

    Returns:
        Timestamp of last run, or None if never run
    """
    state_file = data_dir / BTC_PRECLOSE_STATE_FILE
    if not state_file.exists():
        return None

    try:
        content = state_file.read_text().strip()
        return datetime.fromisoformat(content)
    except (ValueError, OSError) as e:
        logger.warning("Failed to read btc-preclose state file: %s", e)
        return None


def _update_btc_preclose_last_run(data_dir: Path) -> None:
    """Update timestamp of last btc-preclose run.

    Args:
        data_dir: Data directory to write state file
    """
    state_file = data_dir / BTC_PRECLOSE_STATE_FILE
    try:
        state_file.parent.mkdir(parents=True, exist_ok=True)
        state_file.write_text(datetime.now(UTC).isoformat())
    except OSError as e:
        logger.warning("Failed to write btc-preclose state file: %s", e)


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


def calculate_adjusted_lookback(
    last_fill_at: datetime | None,
    current_lookback_hours: float,
    original_lookback_hours: float,
) -> tuple[float, bool]:
    """Calculate adjusted lookback hours based on staleness.

    If no fills in STALE_THRESHOLD_HOURS, auto-widen lookback by WIDEN_FACTOR
    (with jitter). Bounded at MAX_LOOKBACK_MULTIPLIER * original.

    Args:
        last_fill_at: Timestamp of last fill, or None
        current_lookback_hours: Current lookback window in hours
        original_lookback_hours: Original/base lookback window in hours

    Returns:
        Tuple of (adjusted_lookback_hours, was_adjusted)
    """
    now = datetime.now(UTC)

    # Check if we have no fills or fills are stale
    if last_fill_at is None:
        # No fills yet, use current lookback
        return current_lookback_hours, False

    hours_since_last_fill = (now - last_fill_at).total_seconds() / 3600

    # If not stale, keep current lookback
    if hours_since_last_fill <= STALE_THRESHOLD_HOURS:
        return current_lookback_hours, False

    # Stale: calculate widened lookback with jitter
    jitter = random.uniform(-WIDEN_JITTER, WIDEN_JITTER)
    adjustment_factor = WIDEN_FACTOR + jitter
    new_lookback = current_lookback_hours * adjustment_factor

    # Apply upper bound
    max_lookback = original_lookback_hours * MAX_LOOKBACK_MULTIPLIER
    adjusted_lookback = min(new_lookback, max_lookback)

    was_adjusted = adjusted_lookback > current_lookback_hours
    return adjusted_lookback, was_adjusted


def run_collect_fills_loop(
    data_dir: Path | None = None,
    fills_path: Path | None = None,
    paper_fills_path: Path | None = None,
    interval_seconds: float = DEFAULT_INTERVAL_SECONDS,
    include_account: bool = True,
    include_paper: bool = True,
    stale_alert_hours: float = DEFAULT_STALE_ALERT_HOURS,
    lookback_hours: float = 72.0,
    on_stale_alert: Callable[[str], None] | None = None,
) -> None:
    """Run continuous fills collection loop.

    Automatically widens collection thresholds if no fills received for
    extended periods (stale detection with auto-adjust).

    Includes fail-safe: If fills are >6h stale AND btc-preclose hasn't fired,
    forces collection with relaxed thresholds (wider lookback).

    Args:
        data_dir: Base data directory
        fills_path: Output path for fills.jsonl
        paper_fills_path: Path to paper trading fills.jsonl
        interval_seconds: Seconds between collection runs
        include_account: Whether to fetch real account fills
        include_paper: Whether to include paper trading fills
        stale_alert_hours: Hours before triggering stale alert
        lookback_hours: Fixed lookback window in hours for fill queries
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

    # Track lookback for auto-adjustment
    original_lookback = float(lookback_hours)
    current_lookback = original_lookback
    last_fill_at: datetime | None = None

    # Track consecutive empty cycles for alerting
    consecutive_empty_cycles = 0

    logger.info(
        "Starting collect-fills loop: interval=%.0fs lookback=%.0fh account=%s paper=%s",
        interval_seconds,
        lookback_hours,
        include_account,
        include_paper,
    )

    # Verify environment configuration (important for cron context)
    env_verification = _log_env_verification()
    if env_verification.get("warnings"):
        logger.warning(
            "Environment verification found %d warnings - "
            ".env file may not be loaded in cron context",
            len(env_verification["warnings"]),
        )

    # Run startup diagnostic to verify credentials
    if include_account:
        from .fills_collector import startup_diagnostic
        startup_diagnostic()

    iteration = 0
    last_heartbeat_time: float | None = None

    while True:
        started = time.time()
        iteration += 1

        try:
            # Calculate adjusted lookback based on staleness
            current_lookback, was_adjusted = calculate_adjusted_lookback(
                last_fill_at=last_fill_at,
                current_lookback_hours=current_lookback,
                original_lookback_hours=original_lookback,
            )

            if was_adjusted:
                logger.warning(
                    "Auto-widening lookback: %.1fh (was %.1fh, original %.1fh)",
                    current_lookback,
                    current_lookback / WIDEN_FACTOR,
                    original_lookback,
                )

            # Check for fail-safe condition: fills stale AND btc-preclose hasn't fired
            # Use file's last fill time for failsafe check (more accurate than in-memory)
            summary_for_failsafe = get_fills_summary(fills_path)
            file_last_fill_at = None
            if summary_for_failsafe.get("last_fill_at"):
                file_last_fill_at = datetime.fromisoformat(summary_for_failsafe["last_fill_at"])

            btc_preclose_last_run = _get_btc_preclose_last_run(data_dir)
            failsafe_triggered = False
            failsafe_lookback = current_lookback

            if file_last_fill_at is not None:
                hours_since_last_fill = (datetime.now(UTC) - file_last_fill_at).total_seconds() / 3600
                hours_since_btc_preclose = (
                    (datetime.now(UTC) - btc_preclose_last_run).total_seconds() / 3600
                    if btc_preclose_last_run
                    else float("inf")
                )

                # Fail-safe: if fills >6h stale AND btc-preclose hasn't fired recently
                if hours_since_last_fill > STALE_THRESHOLD_HOURS and hours_since_btc_preclose > STALE_THRESHOLD_HOURS:
                    failsafe_lookback = original_lookback * FAILSAFE_MAX_LOOKBACK_MULTIPLIER
                    failsafe_triggered = True
                    logger.warning(
                        "FAILSAFE TRIGGERED: Last fill %.1fh ago, btc-preclose last ran %.1fh ago. "
                        "Forcing collection with relaxed lookback: %.1fh",
                        hours_since_last_fill,
                        hours_since_btc_preclose,
                        failsafe_lookback,
                    )

            # Collect fills (pass iteration for heartbeat logging)
            logger.debug("Iteration %d: collecting fills...", iteration)
            collect_result = collect_fills(
                fills_path=fills_path,
                paper_fills_path=paper_fills_path,
                include_account=include_account,
                include_paper=include_paper,
                lookback_hours=failsafe_lookback if failsafe_triggered else current_lookback,
                iteration=iteration,
                last_heartbeat_time=last_heartbeat_time,
            )

            # Update heartbeat time from result
            if collect_result.get("heartbeat_logged"):
                last_heartbeat_time = time.time()

            # Track consecutive empty cycles
            if collect_result["total_appended"] == 0:
                consecutive_empty_cycles += 1
                logger.info(
                    "Empty cycle #%d (threshold: %d)",
                    consecutive_empty_cycles,
                    EMPTY_CYCLE_ALERT_THRESHOLD,
                )

                # Alert if we've had >2 consecutive empty cycles
                if consecutive_empty_cycles > EMPTY_CYCLE_ALERT_THRESHOLD:
                    alert_msg = (
                        f"🚨 Polymarket fills collector returned empty "
                        f"for {consecutive_empty_cycles} consecutive cycles. "
                        f"Check API connectivity and market conditions."
                    )
                    logger.error(alert_msg)
                    if on_stale_alert:
                        on_stale_alert(alert_msg)
                    else:
                        _send_openclaw_notification(alert_msg)
            else:
                # Reset empty cycle counter on successful collection
                if consecutive_empty_cycles > 0:
                    logger.info(
                        "Resetting empty cycle counter after %d empty cycles",
                        consecutive_empty_cycles,
                    )
                consecutive_empty_cycles = 0

            # Log collection results
            logger.info(
                "Collected %d fills (%d account, %d paper, %d duplicates)%s",
                collect_result["total_appended"],
                collect_result["account_fills"],
                collect_result["paper_fills"],
                collect_result["duplicates_skipped"],
                " [FAILSAFE]" if failsafe_triggered else "",
            )

            # Reset lookback if we got new fills
            if collect_result["total_appended"] > 0:
                if current_lookback > original_lookback:
                    logger.info(
                        "Resetting lookback to %.1fh after successful fill",
                        original_lookback,
                    )
                current_lookback = original_lookback
                last_fill_at = datetime.now(UTC)
            else:
                # Update last_fill_at from file if available
                summary = get_fills_summary(fills_path)
                if summary.get("last_fill_at"):
                    last_fill_at = datetime.fromisoformat(summary["last_fill_at"])

            # Check staleness
            staleness = check_fills_staleness(fills_path, stale_alert_hours)

            if staleness["age_hours"] is not None:
                logger.info(
                    "Fill age: %.2fh (threshold: %.1fh) fills=%d lookback=%.1fh",
                    staleness["age_hours"],
                    stale_alert_hours,
                    staleness["total_fills"],
                    current_lookback,
                )
            else:
                logger.info("No fills found yet (lookback=%.1fh)", current_lookback)

            # Alert if stale
            if staleness["is_stale"]:
                age_hours = staleness["age_hours"]
                warning_msg = (
                    f"FILLS STALE: Last fill was {age_hours:.1f}h ago "
                    f"(threshold: {stale_alert_hours:.1f}h). "
                    f"Total fills: {staleness['total_fills']}"
                )
                logger.warning(warning_msg)

                # Emit OpenClaw notification
                notification = (
                    f"🚨 Polymarket fills stale: {age_hours:.1f}h since last fill. "
                    f"Check collect-fills-loop service."
                )
                if on_stale_alert:
                    on_stale_alert(notification)
                else:
                    _send_openclaw_notification(notification)

        except AuthenticationError as e:
            # Authentication errors are fatal - log critical and exit
            logger.critical(
                "AUTHENTICATION FAILED in iteration %d: %s. "
                "Fill collection cannot continue without valid API credentials. "
                "Set POLYMARKET_API_KEY, POLYMARKET_API_SECRET, and "
                "POLYMARKET_API_PASSPHRASE environment variables.",
                iteration,
                e,
            )
            # Send notification about auth failure
            try:
                _send_openclaw_notification(
                    "🚨 Polymarket fills loop STOPPED: Authentication failed. "
                    "Check API credentials (POLYMARKET_API_KEY, etc.)."
                )
            except Exception:
                pass  # Notification is best-effort
            raise  # Exit the loop

        except Exception:
            logger.exception("Error in collect-fills loop iteration %d", iteration)

        # Sleep until next iteration
        elapsed = time.time() - started
        sleep_for = max(0.0, interval_seconds - elapsed)
        sleep_for += random.uniform(0.0, min(10.0, interval_seconds * 0.05))  # Small jitter
        logger.debug("Sleeping for %.1fs", sleep_for)
        time.sleep(sleep_for)
