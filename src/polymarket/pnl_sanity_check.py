"""NAV/PnL sanity check job for detecting impossible PnL jumps.

Computes realized and unrealized PnL from fills + current market mid prices,
then compares against historical PnL summaries to detect impossible jumps.

Usage:
    ./run.sh pnl-sanity-check           # Run check with defaults
    ./run.sh pnl-sanity-check --alert-threshold-usd 100  # Custom threshold
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING

from .pnl import (
    PnLVerifier,
    load_fills_from_file,
    load_orderbooks_from_snapshot,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path("data")
DEFAULT_PNL_DIR = Path("data/pnl")
DEFAULT_ALERT_THRESHOLD_USD = Decimal("100.0")  # Alert on $100+ jumps
DEFAULT_MAX_PNL_AGE_HOURS = 24.0


@dataclass
class SanityCheckResult:
    """Result of NAV/PnL sanity check."""

    # Status
    passed: bool = True
    alerts: list[str] = field(default_factory=list)

    # Current computed values
    computed_realized_pnl: Decimal = field(default_factory=lambda: Decimal("0"))
    computed_unrealized_pnl: Decimal = field(default_factory=lambda: Decimal("0"))
    computed_net_pnl: Decimal = field(default_factory=lambda: Decimal("0"))
    computed_cash_balance: Decimal = field(default_factory=lambda: Decimal("0"))
    computed_mark_to_mid: Decimal = field(default_factory=lambda: Decimal("0"))

    # Previous values from historical summary
    previous_realized_pnl: Decimal | None = None
    previous_unrealized_pnl: Decimal | None = None
    previous_net_pnl: Decimal | None = None
    previous_timestamp: str | None = None

    # Deltas
    realized_pnl_delta: Decimal = field(default_factory=lambda: Decimal("0"))
    unrealized_pnl_delta: Decimal = field(default_factory=lambda: Decimal("0"))
    net_pnl_delta: Decimal = field(default_factory=lambda: Decimal("0"))

    # Metadata
    check_timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    fills_count: int = 0
    time_since_previous_hours: float | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "status": "passed" if self.passed else "failed",
            "passed": self.passed,
            "alerts": self.alerts,
            "computed": {
                "realized_pnl": float(self.computed_realized_pnl),
                "unrealized_pnl": float(self.computed_unrealized_pnl),
                "net_pnl": float(self.computed_net_pnl),
                "cash_balance": float(self.computed_cash_balance),
                "mark_to_mid": float(self.computed_mark_to_mid),
            },
            "previous": {
                "realized_pnl": float(self.previous_realized_pnl)
                if self.previous_realized_pnl is not None
                else None,
                "unrealized_pnl": float(self.previous_unrealized_pnl)
                if self.previous_unrealized_pnl is not None
                else None,
                "net_pnl": float(self.previous_net_pnl)
                if self.previous_net_pnl is not None
                else None,
                "timestamp": self.previous_timestamp,
            },
            "deltas": {
                "realized_pnl": float(self.realized_pnl_delta),
                "unrealized_pnl": float(self.unrealized_pnl_delta),
                "net_pnl": float(self.net_pnl_delta),
            },
            "metadata": {
                "check_timestamp": self.check_timestamp,
                "fills_count": self.fills_count,
                "time_since_previous_hours": self.time_since_previous_hours,
            },
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


def load_latest_pnl_summary(pnl_dir: Path) -> dict | None:
    """Load the most recent PnL summary file.

    Args:
        pnl_dir: Directory containing pnl_*.json files

    Returns:
        Dict with PnL summary data or None if not found
    """
    if not pnl_dir.exists():
        return None

    pnl_files = sorted(pnl_dir.glob("pnl_*.json"))
    if not pnl_files:
        return None

    latest = pnl_files[-1]
    try:
        data = json.loads(latest.read_text())
        data["_source_file"] = str(latest)
        return data
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to load PnL summary %s: %s", latest, e)
        return None


def compute_pnl_from_fills(
    fills_path: Path,
    snapshot_path: Path | None = None,
    starting_cash: Decimal | None = None,
) -> dict:
    """Compute PnL from fills and current market prices.

    Args:
        fills_path: Path to fills.jsonl file
        snapshot_path: Optional path to snapshot for current prices
        starting_cash: Starting cash balance

    Returns:
        Dict with computed PnL values
    """
    result = {
        "success": False,
        "error": None,
        "realized_pnl": Decimal("0"),
        "unrealized_pnl": Decimal("0"),
        "net_pnl": Decimal("0"),
        "cash_balance": Decimal("0"),
        "mark_to_mid": Decimal("0"),
        "fills_count": 0,
    }

    try:
        if not fills_path.exists():
            result["error"] = f"Fills file not found: {fills_path}"
            return result

        # Load fills
        fills = load_fills_from_file(fills_path)
        result["fills_count"] = len(fills)

        if not fills:
            result["error"] = "No fills found"
            return result

        # Load orderbooks from snapshot if available
        orderbooks = None
        if snapshot_path and snapshot_path.exists():
            try:
                # Check if it's a pointer file
                data = json.loads(snapshot_path.read_text())
                if isinstance(data, dict) and "path" in data:
                    resolved = Path(data["path"])
                    if resolved.exists():
                        orderbooks = load_orderbooks_from_snapshot(resolved)
                else:
                    orderbooks = load_orderbooks_from_snapshot(snapshot_path)
            except (json.JSONDecodeError, OSError):
                pass

        # Build verifier and compute PnL
        verifier = PnLVerifier(starting_cash=starting_cash or Decimal("0"))
        verifier.add_fills(fills)

        report = verifier.compute_pnl(orderbooks=orderbooks)

        result["success"] = True
        result["realized_pnl"] = report.realized_pnl
        result["unrealized_pnl"] = report.unrealized_pnl
        result["net_pnl"] = report.net_pnl
        result["cash_balance"] = report.ending_cash
        result["mark_to_mid"] = report.mark_to_mid

    except Exception as e:
        result["error"] = str(e)
        logger.exception("Error computing PnL from fills: %s", e)

    return result


def check_pnl_sanity(
    data_dir: Path | None = None,
    snapshot_path: Path | None = None,
    pnl_dir: Path | None = None,
    alert_threshold_usd: Decimal | None = None,
    starting_cash: Decimal | None = None,
    max_pnl_age_hours: float = DEFAULT_MAX_PNL_AGE_HOURS,
) -> SanityCheckResult:
    """Run NAV/PnL sanity check.

    Computes PnL from fills + current mid prices and compares against
    historical PnL summary to detect impossible jumps.

    Args:
        data_dir: Base data directory
        snapshot_path: Path to snapshot file for current prices
        pnl_dir: Directory for PnL summaries
        alert_threshold_usd: Threshold for alerting on PnL jumps
        starting_cash: Starting cash balance
        max_pnl_age_hours: Maximum age of previous PnL summary

    Returns:
        SanityCheckResult with comparison and any alerts
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    if pnl_dir is None:
        pnl_dir = DEFAULT_PNL_DIR
    if alert_threshold_usd is None:
        alert_threshold_usd = DEFAULT_ALERT_THRESHOLD_USD

    data_dir = Path(data_dir)
    pnl_dir = Path(pnl_dir)

    result = SanityCheckResult()

    # Compute current PnL from fills
    fills_path = data_dir / "fills.jsonl"
    computed = compute_pnl_from_fills(
        fills_path=fills_path,
        snapshot_path=snapshot_path,
        starting_cash=starting_cash,
    )

    if not computed["success"]:
        result.passed = False
        result.alerts.append(f"Failed to compute PnL: {computed['error']}")
        return result

    result.computed_realized_pnl = computed["realized_pnl"]
    result.computed_unrealized_pnl = computed["unrealized_pnl"]
    result.computed_net_pnl = computed["net_pnl"]
    result.computed_cash_balance = computed["cash_balance"]
    result.computed_mark_to_mid = computed["mark_to_mid"]
    result.fills_count = computed["fills_count"]

    # Load previous PnL summary
    previous = load_latest_pnl_summary(pnl_dir)

    if previous is None:
        # No previous summary - this is OK for first run
        logger.info("No previous PnL summary found - skipping delta check")
        result.alerts.append("No previous PnL summary found (first run?)")
        return result

    # Extract previous values
    try:
        result.previous_realized_pnl = Decimal(str(previous.get("pnl", {}).get("realized_pnl", 0)))
        result.previous_unrealized_pnl = Decimal(
            str(previous.get("pnl", {}).get("unrealized_pnl", 0))
        )
        result.previous_net_pnl = Decimal(str(previous.get("pnl", {}).get("net_pnl", 0)))
        result.previous_timestamp = previous.get("metadata", {}).get("generated_at")

        # Calculate time since previous
        if result.previous_timestamp:
            try:
                prev_time = datetime.fromisoformat(result.previous_timestamp.replace("Z", "+00:00"))
                time_diff = datetime.now(UTC) - prev_time
                result.time_since_previous_hours = time_diff.total_seconds() / 3600
            except ValueError:
                pass
    except (KeyError, ValueError) as e:
        result.passed = False
        result.alerts.append(f"Failed to parse previous PnL summary: {e}")
        return result

    # Check if previous summary is too old
    if result.time_since_previous_hours and result.time_since_previous_hours > max_pnl_age_hours:
        result.alerts.append(
            f"Previous PnL summary is stale: {result.time_since_previous_hours:.1f}h old"
        )

    # Calculate deltas
    result.realized_pnl_delta = result.computed_realized_pnl - result.previous_realized_pnl
    result.unrealized_pnl_delta = result.computed_unrealized_pnl - result.previous_unrealized_pnl
    result.net_pnl_delta = result.computed_net_pnl - result.previous_net_pnl

    # Check for impossible jumps
    # Realized PnL should only change when positions are closed
    # A jump in realized PnL without corresponding fills is suspicious

    # Alert on large realized PnL jumps (indicates possible data inconsistency)
    if abs(result.realized_pnl_delta) > alert_threshold_usd:
        result.passed = False
        result.alerts.append(
            f"IMPOSSIBLE JUMP: Realized PnL changed by ${float(result.realized_pnl_delta):.2f} "
            f"(from ${float(result.previous_realized_pnl):.2f} to ${float(result.computed_realized_pnl):.2f}). "
            f"Realized PnL should only change when closing positions."
        )

    # Alert on large net PnL jumps (could indicate calculation error)
    if abs(result.net_pnl_delta) > alert_threshold_usd * 2:  # Higher threshold for net
        result.passed = False
        result.alerts.append(
            f"LARGE NET PnL JUMP: ${float(result.net_pnl_delta):.2f} change detected "
            f"(from ${float(result.previous_net_pnl):.2f} to ${float(result.computed_net_pnl):.2f})"
        )

    # Check for sign flips in realized PnL (very suspicious)
    if (result.previous_realized_pnl > 0 and result.computed_realized_pnl < 0) or (
        result.previous_realized_pnl < 0 and result.computed_realized_pnl > 0
    ):
        if (
            abs(result.realized_pnl_delta) > alert_threshold_usd / 10
        ):  # Lower threshold for sign flip
            result.passed = False
            result.alerts.append(
                f"SUSPICIOUS SIGN FLIP: Realized PnL flipped from "
                f"${float(result.previous_realized_pnl):.2f} to ${float(result.computed_realized_pnl):.2f}"
            )

    # Alert if cash balance seems inconsistent with fills
    # (This would require more detailed tracking, but we can do basic checks)
    if result.computed_cash_balance < -alert_threshold_usd:
        result.alerts.append(
            f"LARGE NEGATIVE CASH: Cash balance is ${float(result.computed_cash_balance):.2f}. "
            f"Check for missing deposits or starting cash configuration."
        )

    # Log results
    if result.passed:
        logger.info(
            "PnL sanity check passed: realized=%.2f unrealized=%.2f net=%.2f",
            float(result.computed_realized_pnl),
            float(result.computed_unrealized_pnl),
            float(result.computed_net_pnl),
        )
    else:
        logger.warning("PnL sanity check failed with %d alerts", len(result.alerts))
        for alert in result.alerts:
            logger.warning("ALERT: %s", alert)

    return result


def run_sanity_check_loop(
    data_dir: Path | None = None,
    snapshot_path: Path | None = None,
    pnl_dir: Path | None = None,
    interval_seconds: float = 3600.0,  # 1 hour
    alert_threshold_usd: Decimal | None = None,
    starting_cash: Decimal | None = None,
    max_pnl_age_hours: float = DEFAULT_MAX_PNL_AGE_HOURS,
) -> None:
    """Continuous loop to run PnL sanity checks.

    Args:
        data_dir: Base data directory
        snapshot_path: Path to snapshot file for current prices
        pnl_dir: Directory for PnL summaries
        interval_seconds: Seconds between checks
        alert_threshold_usd: Threshold for alerting on PnL jumps
        starting_cash: Starting cash balance
        max_pnl_age_hours: Maximum age of previous PnL summary
    """
    import time
    import random

    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    if pnl_dir is None:
        pnl_dir = DEFAULT_PNL_DIR

    data_dir = Path(data_dir)
    pnl_dir = Path(pnl_dir)

    logger.info(
        "Starting PnL sanity check loop: interval=%.0fs threshold=%s",
        interval_seconds,
        alert_threshold_usd,
    )

    while True:
        started = time.time()

        try:
            result = check_pnl_sanity(
                data_dir=data_dir,
                snapshot_path=snapshot_path,
                pnl_dir=pnl_dir,
                alert_threshold_usd=alert_threshold_usd,
                starting_cash=starting_cash,
                max_pnl_age_hours=max_pnl_age_hours,
            )

            if not result.passed:
                logger.error(
                    "PnL SANITY CHECK FAILED: %d alerts - %s",
                    len(result.alerts),
                    "; ".join(result.alerts),
                )
            else:
                logger.debug("PnL sanity check passed")

        except Exception:
            logger.exception("Error in PnL sanity check loop")

        # Sleep until next iteration
        elapsed = time.time() - started
        sleep_for = max(0.0, interval_seconds - elapsed)
        sleep_for += random.uniform(0.0, min(60.0, interval_seconds * 0.1))  # Small jitter
        time.sleep(sleep_for)
