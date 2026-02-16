"""Fills monitor - ensures continuous paper trading activity with auto-adjustment.

This module provides:
1. Monitoring of fills.jsonl for staleness (>6h without new fills)
2. Alert generation when fills stall
3. Auto-adjustment of strategy thresholds to increase hit rate
4. Bounded adjustments to prevent runaway threshold changes
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_FILLS_PATH = Path("data/fills.jsonl")
DEFAULT_STATE_PATH = Path("data/paper_trading/fills_monitor_state.json")

# Configurable thresholds
STALE_HOURS = 6  # Consider fills stale after 6 hours
MIN_CHEAP_PRICE = Decimal("0.01")  # Don't go below 1 cent
MAX_CHEAP_PRICE = Decimal("0.15")  # Don't go above 15 cents
MIN_WINDOW_SECONDS = 60  # Minimum 1 minute window
MAX_WINDOW_SECONDS = 600  # Maximum 10 minute window
ADJUSTMENT_FACTOR = Decimal("1.5")  # Multiply by this when adjusting


@dataclass
class FillsMonitorState:
    """Persistent state for the fills monitor."""

    last_fill_timestamp: str | None = None
    last_check_timestamp: str | None = None
    adjustment_count: int = 0
    current_cheap_price: Decimal = Decimal("0.05")
    current_window_seconds: int = 300
    alerts_triggered: int = 0
    total_fills_seen: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "last_fill_timestamp": self.last_fill_timestamp,
            "last_check_timestamp": self.last_check_timestamp,
            "adjustment_count": self.adjustment_count,
            "current_cheap_price": str(self.current_cheap_price),
            "current_window_seconds": self.current_window_seconds,
            "alerts_triggered": self.alerts_triggered,
            "total_fills_seen": self.total_fills_seen,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FillsMonitorState:
        return cls(
            last_fill_timestamp=data.get("last_fill_timestamp"),
            last_check_timestamp=data.get("last_check_timestamp"),
            adjustment_count=data.get("adjustment_count", 0),
            current_cheap_price=Decimal(str(data.get("current_cheap_price", "0.05"))),
            current_window_seconds=data.get("current_window_seconds", 300),
            alerts_triggered=data.get("alerts_triggered", 0),
            total_fills_seen=data.get("total_fills_seen", 0),
        )


def get_last_fill_timestamp(fills_path: Path) -> datetime | None:
    """Get the timestamp of the most recent fill.

    Args:
        fills_path: Path to fills.jsonl

    Returns:
        Timestamp of last fill, or None if no fills
    """
    if not fills_path.exists():
        return None

    last_ts = None
    try:
        with open(fills_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    ts_str = data.get("timestamp") or data.get("created_at")
                    if ts_str:
                        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                        if last_ts is None or ts > last_ts:
                            last_ts = ts
                except (json.JSONDecodeError, ValueError):
                    continue
    except Exception as e:
        logger.warning("Error reading fills file: %s", e)

    return last_ts


def count_fills(fills_path: Path) -> int:
    """Count total fills in the file.

    Args:
        fills_path: Path to fills.jsonl

    Returns:
        Number of fills
    """
    if not fills_path.exists():
        return 0

    count = 0
    try:
        with open(fills_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    count += 1
    except Exception as e:
        logger.warning("Error counting fills: %s", e)

    return count


def load_state(state_path: Path) -> FillsMonitorState:
    """Load monitor state from disk.

    Args:
        state_path: Path to state file

    Returns:
        FillsMonitorState
    """
    if not state_path.exists():
        return FillsMonitorState()

    try:
        data = json.loads(state_path.read_text())
        return FillsMonitorState.from_dict(data)
    except Exception as e:
        logger.warning("Error loading state: %s", e)
        return FillsMonitorState()


def save_state(state: FillsMonitorState, state_path: Path) -> None:
    """Save monitor state to disk.

    Args:
        state: State to save
        state_path: Path to state file
    """
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state.to_dict(), indent=2))


def check_fills_health(
    fills_path: Path = DEFAULT_FILLS_PATH,
    stale_hours: int = STALE_HOURS,
) -> dict[str, Any]:
    """Check if fills are stale (no new fills for > stale_hours).

    Args:
        fills_path: Path to fills.jsonl
        stale_hours: Hours to consider fills stale

    Returns:
        Dict with health status and metrics
    """
    now = datetime.now(UTC)
    last_fill = get_last_fill_timestamp(fills_path)
    total_fills = count_fills(fills_path)

    if last_fill is None:
        return {
            "healthy": False,
            "status": "no_fills",
            "message": "No fills found in file",
            "last_fill_at": None,
            "hours_since_last_fill": None,
            "total_fills": 0,
            "timestamp": now.isoformat(),
        }

    hours_since = (now - last_fill).total_seconds() / 3600
    is_healthy = hours_since < stale_hours

    return {
        "healthy": is_healthy,
        "status": "healthy" if is_healthy else "stale",
        "message": (
            f"Last fill {hours_since:.1f} hours ago"
            if is_healthy
            else f"No fills for {hours_since:.1f} hours (threshold: {stale_hours}h)"
        ),
        "last_fill_at": last_fill.isoformat(),
        "hours_since_last_fill": round(hours_since, 2),
        "total_fills": total_fills,
        "timestamp": now.isoformat(),
    }


def auto_adjust_thresholds(
    current_cheap_price: Decimal,
    current_window: int,
    adjustment_count: int,
) -> tuple[Decimal, int, bool]:
    """Auto-adjust thresholds to increase hit rate.

    Args:
        current_cheap_price: Current cheap price threshold
        current_window: Current window in seconds
        adjustment_count: Number of adjustments already made

    Returns:
        Tuple of (new_cheap_price, new_window, was_adjusted)
    """
    # Don't adjust if we've already adjusted too much
    if adjustment_count >= 3:
        logger.warning(
            "Max adjustments reached (%d), not adjusting further",
            adjustment_count,
        )
        return current_cheap_price, current_window, False

    # Calculate new values
    new_cheap_price = min(
        current_cheap_price * ADJUSTMENT_FACTOR,
        MAX_CHEAP_PRICE,
    )
    new_window = min(
        int(current_window * float(ADJUSTMENT_FACTOR)),
        MAX_WINDOW_SECONDS,
    )

    # Check if we're at bounds
    at_price_bound = new_cheap_price == current_cheap_price == MAX_CHEAP_PRICE
    at_window_bound = new_window == current_window == MAX_WINDOW_SECONDS

    if at_price_bound and at_window_bound:
        logger.warning("At maximum thresholds, cannot adjust further")
        return current_cheap_price, current_window, False

    logger.info(
        "Auto-adjusting thresholds: price %s -> %s, window %d -> %d (adj #%d)",
        current_cheap_price,
        new_cheap_price,
        current_window,
        new_window,
        adjustment_count + 1,
    )

    return new_cheap_price, new_window, True


def run_fills_monitor(
    fills_path: Path = DEFAULT_FILLS_PATH,
    state_path: Path = DEFAULT_STATE_PATH,
    stale_hours: int = STALE_HOURS,
    auto_adjust: bool = True,
) -> dict[str, Any]:
    """Run the fills monitor check and optionally auto-adjust.

    Args:
        fills_path: Path to fills.jsonl
        state_path: Path to monitor state file
        stale_hours: Hours to consider fills stale
        auto_adjust: Whether to auto-adjust thresholds

    Returns:
        Dict with check results and any adjustments made
    """
    now = datetime.now(UTC)
    state = load_state(state_path)

    # Check health
    health = check_fills_health(fills_path, stale_hours)

    # Update state
    state.last_check_timestamp = now.isoformat()
    if health["last_fill_at"]:
        state.last_fill_timestamp = health["last_fill_at"]
    state.total_fills_seen = health["total_fills"]

    result = {
        "timestamp": now.isoformat(),
        "healthy": health["healthy"],
        "status": health["status"],
        "message": health["message"],
        "hours_since_last_fill": health["hours_since_last_fill"],
        "total_fills": health["total_fills"],
        "adjustment_count": state.adjustment_count,
        "current_cheap_price": str(state.current_cheap_price),
        "current_window_seconds": state.current_window_seconds,
        "auto_adjusted": False,
        "alert_triggered": False,
    }

    # Handle stale fills
    if not health["healthy"]:
        state.alerts_triggered += 1
        result["alert_triggered"] = True
        logger.warning("FILLS STALL ALERT: %s", health["message"])

        if auto_adjust:
            new_price, new_window, was_adjusted = auto_adjust_thresholds(
                state.current_cheap_price,
                state.current_window_seconds,
                state.adjustment_count,
            )

            if was_adjusted:
                state.current_cheap_price = new_price
                state.current_window_seconds = new_window
                state.adjustment_count += 1
                result["auto_adjusted"] = True
                result["new_cheap_price"] = str(new_price)
                result["new_window_seconds"] = new_window

    # Save state
    save_state(state, state_path)

    return result


def get_current_thresholds(
    state_path: Path = DEFAULT_STATE_PATH,
    default_price: Decimal = Decimal("0.05"),
    default_window: int = 300,
) -> tuple[Decimal, int]:
    """Get current thresholds from monitor state.

    Args:
        state_path: Path to state file
        default_price: Default cheap price if no state
        default_window: Default window if no state

    Returns:
        Tuple of (cheap_price, window_seconds)
    """
    state = load_state(state_path)
    return state.current_cheap_price, state.current_window_seconds
