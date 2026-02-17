"""PnL verification loop for daily accounting.

Runs pnl-verify on a schedule, persists daily summaries,
and provides health check metrics.
"""

from __future__ import annotations

import json
import logging
import random
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from .fills_collector import (
    collect_fills,
    get_fills_summary,
)
from .pnl import (
    PnLVerifier,
    load_fills_from_file,
    load_orderbooks_from_snapshot,
    save_daily_summary,
)

if TYPE_CHECKING:
    from decimal import Decimal

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path("data")
DEFAULT_PNL_DIR = Path("data/pnl")
DEFAULT_SNAPSHOT_PATH = Path("data/latest_15m.json")


def get_latest_pnl_summary(pnl_dir: Path) -> dict:
    """Get the most recent PnL summary file.

    Args:
        pnl_dir: Directory containing pnl_*.json files

    Returns:
        Dict with latest summary info
    """
    result = {
        "pnl_dir": str(pnl_dir),
        "exists": pnl_dir.exists(),
        "latest_file": None,
        "latest_date": None,
        "age_seconds": None,
    }

    if not pnl_dir.exists():
        return result

    pnl_files = sorted(pnl_dir.glob("pnl_*.json"))
    if not pnl_files:
        return result

    latest = pnl_files[-1]
    result["latest_file"] = str(latest)

    # Extract date from filename (pnl_YYYY-MM-DD.json)
    try:
        date_str = latest.stem.replace("pnl_", "")
        date = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=UTC)
        result["latest_date"] = date.isoformat()
        result["age_seconds"] = (datetime.now(UTC) - date).total_seconds()
    except ValueError:
        # Fall back to file mtime
        mtime = datetime.fromtimestamp(latest.stat().st_mtime, tz=UTC)
        result["latest_date"] = mtime.isoformat()
        result["age_seconds"] = (datetime.now(UTC) - mtime).total_seconds()

    return result


def run_pnl_verification(
    data_dir: Path | None = None,
    snapshot_path: Path | None = None,
    pnl_dir: Path | None = None,
    starting_cash: Decimal | None = None,
) -> dict:
    """Run PnL verification and save daily summary.

    Args:
        data_dir: Base data directory
        snapshot_path: Path to snapshot file (or pointer)
        pnl_dir: Directory for PnL summaries
        starting_cash: Starting cash balance

    Returns:
        Dict with verification results
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    if snapshot_path is None:
        snapshot_path = DEFAULT_SNAPSHOT_PATH
    if pnl_dir is None:
        pnl_dir = DEFAULT_PNL_DIR

    data_dir = Path(data_dir)
    snapshot_path = Path(snapshot_path)
    pnl_dir = Path(pnl_dir)

    # Resolve snapshot pointer if needed
    if snapshot_path.exists():
        try:
            data = json.loads(snapshot_path.read_text())
            if isinstance(data, dict) and "path" in data:
                resolved = Path(data["path"])
                if resolved.exists():
                    snapshot_path = resolved
        except (json.JSONDecodeError, OSError):
            pass

    fills_path = data_dir / "fills.jsonl"

    result = {
        "timestamp": datetime.now(UTC).isoformat(),
        "fills_path": str(fills_path),
        "snapshot_path": str(snapshot_path),
        "pnl_dir": str(pnl_dir),
        "success": False,
        "error": None,
        "report": None,
        "summary_path": None,
    }

    try:
        # Check fills exist
        if not fills_path.exists():
            result["error"] = f"Fills file not found: {fills_path}"
            logger.error(result["error"])
            return result

        # Load fills
        fills = load_fills_from_file(fills_path)
        if not fills:
            result["error"] = "No fills found in file"
            logger.warning(result["error"])
            return result

        # Load orderbooks from snapshot
        orderbooks = None
        if snapshot_path.exists():
            orderbooks = load_orderbooks_from_snapshot(snapshot_path)
            logger.info("Loaded orderbooks from %s", snapshot_path)
        else:
            logger.warning("Snapshot not found: %s", snapshot_path)

        # Build verifier and compute PnL
        verifier = PnLVerifier(starting_cash=starting_cash or 0)
        verifier.add_fills(fills)

        report = verifier.compute_pnl(orderbooks=orderbooks)
        result["report"] = report.to_dict()
        result["success"] = True

        # Save daily summary
        pnl_dir.mkdir(parents=True, exist_ok=True)
        summary_path = save_daily_summary(report, out_dir=pnl_dir)
        result["summary_path"] = str(summary_path)

        logger.info(
            "PnL verification complete: realized=%.2f unrealized=%.2f net=%.2f",
            float(report.realized_pnl),
            float(report.unrealized_pnl),
            float(report.net_pnl),
        )

    except Exception as e:
        result["error"] = str(e)
        logger.exception("PnL verification failed: %s", e)

    return result


def pnl_health_check(
    data_dir: Path | None = None,
    max_fills_age_seconds: float = 86400.0,  # 24 hours
    max_pnl_age_seconds: float = 86400.0,  # 24 hours
) -> dict:
    """Check health of fills and PnL data.

    Args:
        data_dir: Base data directory
        max_fills_age_seconds: Maximum acceptable age of last fill
        max_pnl_age_seconds: Maximum acceptable age of last PnL summary

    Returns:
        Dict with health status
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    data_dir = Path(data_dir)

    fills_path = data_dir / "fills.jsonl"
    pnl_dir = data_dir / "pnl"

    result = {
        "healthy": True,
        "timestamp": datetime.now(UTC).isoformat(),
        "data_dir": str(data_dir),
        "fills": {},
        "pnl": {},
        "warnings": [],
    }

    # Check fills
    fills_summary = get_fills_summary(fills_path)

    # Count fills in last 24h for daily metric
    from .paper_fill_loop import count_fills_last_24h

    fills_24h = count_fills_last_24h(fills_path)

    result["fills"] = {
        "exists": fills_summary["exists"],
        "total_fills": fills_summary["total_fills"],
        "fills_appended_last_24h": fills_24h,
        "last_fill_at": fills_summary["last_fill_at"],
        "age_seconds": fills_summary["age_seconds"],
        "max_age_seconds": max_fills_age_seconds,
        "healthy": True,
    }

    if not fills_summary["exists"]:
        result["fills"]["healthy"] = False
        result["healthy"] = False
        result["warnings"].append("No fills file found")
    elif fills_summary["age_seconds"] is not None:
        if fills_summary["age_seconds"] > max_fills_age_seconds:
            result["fills"]["healthy"] = False
            result["healthy"] = False
            result["warnings"].append(
                f"Fills data is stale: {fills_summary['age_seconds']:.0f}s old"
            )

    # Alert if no fills in last 24h
    if fills_24h == 0 and fills_summary["exists"]:
        result["fills"]["healthy"] = False
        result["healthy"] = False
        result["warnings"].append("No fills appended in last 24h")

    # Check PnL summaries
    pnl_summary = get_latest_pnl_summary(pnl_dir)
    result["pnl"] = {
        "exists": pnl_summary["exists"],
        "latest_file": pnl_summary["latest_file"],
        "latest_date": pnl_summary["latest_date"],
        "age_seconds": pnl_summary["age_seconds"],
        "max_age_seconds": max_pnl_age_seconds,
        "healthy": True,
    }

    if not pnl_summary["exists"]:
        result["pnl"]["healthy"] = False
        result["healthy"] = False
        result["warnings"].append("No PnL summaries found")
    elif pnl_summary["age_seconds"] is not None:
        if pnl_summary["age_seconds"] > max_pnl_age_seconds:
            result["pnl"]["healthy"] = False
            result["healthy"] = False
            result["warnings"].append(
                f"PnL summary is stale: {pnl_summary['age_seconds']:.0f}s old"
            )

    return result


def collect_and_verify_loop(
    data_dir: Path | None = None,
    snapshot_path: Path | None = None,
    pnl_dir: Path | None = None,
    interval_seconds: float = 3600.0,  # 1 hour
    verify_time: str | None = None,  # e.g., "00:00" for midnight
    starting_cash: Decimal | None = None,
    include_account: bool = True,
    include_paper: bool = True,
) -> None:
    """Continuous loop to collect fills and run PnL verification.

    Args:
        data_dir: Base data directory
        snapshot_path: Path to snapshot file
        pnl_dir: Directory for PnL summaries
        interval_seconds: Seconds between collection runs
        verify_time: Time of day to run verification (HH:MM format)
        starting_cash: Starting cash balance
        include_account: Whether to fetch account fills
        include_paper: Whether to include paper trading fills
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    if snapshot_path is None:
        snapshot_path = DEFAULT_SNAPSHOT_PATH
    if pnl_dir is None:
        pnl_dir = DEFAULT_PNL_DIR

    data_dir = Path(data_dir)
    snapshot_path = Path(snapshot_path)
    pnl_dir = Path(pnl_dir)

    logger.info(
        "Starting PnL collection loop: interval=%.0fs verify_time=%s",
        interval_seconds,
        verify_time,
    )

    last_verify_date = None

    while True:
        started = time.time()

        try:
            # Collect fills
            logger.debug("Collecting fills...")
            collect_result = collect_fills(
                fills_path=data_dir / "fills.jsonl",
                include_account=include_account,
                include_paper=include_paper,
            )
            logger.info(
                "Collected %d fills (%d account, %d paper)",
                collect_result["total_appended"],
                collect_result["account_fills"],
                collect_result["paper_fills"],
            )

            # Check if we should run verification
            now = datetime.now(UTC)
            should_verify = False

            if verify_time:
                # Verify at specific time of day
                current_time = now.strftime("%H:%M")
                current_date = now.date()

                if current_time >= verify_time and last_verify_date != current_date:
                    should_verify = True
                    last_verify_date = current_date
            else:
                # Verify on interval (every N iterations approximately)
                should_verify = True

            if should_verify:
                logger.info("Running PnL verification...")
                verify_result = run_pnl_verification(
                    data_dir=data_dir,
                    snapshot_path=snapshot_path,
                    pnl_dir=pnl_dir,
                    starting_cash=starting_cash,
                )

                if verify_result["success"]:
                    logger.info(
                        "PnL verification saved to %s",
                        verify_result.get("summary_path", "unknown"),
                    )
                else:
                    logger.error("PnL verification failed: %s", verify_result.get("error"))

            # Health check logging
            health = pnl_health_check(data_dir=data_dir)
            if health["healthy"]:
                logger.debug(
                    "Health check OK: fills=%d pnl_age=%s",
                    health["fills"].get("total_fills", 0),
                    health["pnl"].get("age_seconds"),
                )
            else:
                logger.warning("Health check failed: %s", health["warnings"])

        except Exception:
            logger.exception("Error in PnL collection loop")

        # Sleep until next iteration
        elapsed = time.time() - started
        sleep_for = max(0.0, interval_seconds - elapsed)
        sleep_for += random.uniform(0.0, min(60.0, interval_seconds * 0.1))  # Small jitter
        time.sleep(sleep_for)
