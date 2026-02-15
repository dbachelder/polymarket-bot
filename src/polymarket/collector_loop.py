from __future__ import annotations

import json
import logging
import random
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx

from .collector import collect_5m_snapshot, collect_15m_snapshot
from .microstructure import (
    DEFAULT_DEPTH_LEVELS,
    DEFAULT_EXTREME_PIN_THRESHOLD,
    DEFAULT_SPREAD_ALERT_THRESHOLD,
    generate_microstructure_summary,
    log_microstructure_alerts,
    write_microstructure_report,
)

logger = logging.getLogger(__name__)


def _prune_old_files(out_dir: Path, prefix: str, retention_hours: float) -> int:
    cutoff = datetime.now(UTC) - timedelta(hours=retention_hours)
    deleted = 0
    for p in out_dir.glob(f"{prefix}_*.json"):
        try:
            mtime = datetime.fromtimestamp(p.stat().st_mtime, tz=UTC)
        except FileNotFoundError:
            continue
        if mtime < cutoff:
            try:
                p.unlink()
                deleted += 1
            except FileNotFoundError:
                pass
    return deleted


def _prune_by_count(out_dir: Path, prefix: str, max_snapshots: int) -> int:
    """Prune oldest snapshots to keep only max_snapshots most recent.

    Args:
        out_dir: Directory containing snapshots
        prefix: File prefix to match (e.g., 'snapshot_15m')
        max_snapshots: Maximum number of snapshots to retain

    Returns:
        Number of files deleted
    """
    if max_snapshots <= 0:
        return 0

    snapshots: list[tuple[Path, datetime]] = []
    for p in out_dir.glob(f"{prefix}_*.json"):
        try:
            mtime = datetime.fromtimestamp(p.stat().st_mtime, tz=UTC)
            snapshots.append((p, mtime))
        except FileNotFoundError:
            continue

    if len(snapshots) <= max_snapshots:
        return 0

    # Sort by mtime, oldest first
    snapshots.sort(key=lambda x: x[1])

    deleted = 0
    to_delete = len(snapshots) - max_snapshots
    for p, _ in snapshots[:to_delete]:
        try:
            p.unlink()
            deleted += 1
        except FileNotFoundError:
            pass
    return deleted


def get_latest_snapshot_age_seconds(out_dir: Path, prefix: str = "snapshot_15m") -> float | None:
    """Get age of the most recent snapshot in seconds.

    Args:
        out_dir: Directory containing snapshots
        prefix: File prefix to match

    Returns:
        Age in seconds, or None if no snapshots found
    """
    latest_mtime: datetime | None = None

    for p in out_dir.glob(f"{prefix}_*.json"):
        try:
            mtime = datetime.fromtimestamp(p.stat().st_mtime, tz=UTC)
            if latest_mtime is None or mtime > latest_mtime:
                latest_mtime = mtime
        except FileNotFoundError:
            continue

    if latest_mtime is None:
        return None

    return (datetime.now(UTC) - latest_mtime).total_seconds()


def check_staleness_sla(
    out_dir: Path, max_age_seconds: float = 120.0, prefix: str = "snapshot_15m"
) -> dict:
    """Check if latest snapshot meets staleness SLA.

    Args:
        out_dir: Directory containing snapshots
        max_age_seconds: Maximum acceptable age in seconds (default: 120s = 2 min)
        prefix: File prefix to match

    Returns:
        Dict with health status:
        - healthy: bool
        - age_seconds: float | None
        - max_age_seconds: float
        - message: str
    """
    age_seconds = get_latest_snapshot_age_seconds(out_dir, prefix)

    if age_seconds is None:
        return {
            "healthy": False,
            "age_seconds": None,
            "max_age_seconds": max_age_seconds,
            "message": f"No snapshots found in {out_dir} with prefix {prefix}",
        }

    healthy = age_seconds <= max_age_seconds
    message = f"Snapshot age {age_seconds:.1f}s {'OK' if healthy else 'EXCEEDS'} SLA {max_age_seconds:.1f}s"

    return {
        "healthy": healthy,
        "age_seconds": age_seconds,
        "max_age_seconds": max_age_seconds,
        "message": message,
    }


def _run_microstructure_analysis(
    snapshot_path: Path,
    out_dir: Path,
    target_market_substring: str | None,
    spread_threshold: float,
    extreme_pin_threshold: float,
    depth_levels: int,
) -> Path:
    """Run microstructure analysis on a snapshot and write report.

    Returns the path to the generated report.
    """
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    report_path = out_dir / f"microstructure_15m_{ts}.json"

    write_microstructure_report(
        snapshot_path=snapshot_path,
        out_path=report_path,
        target_market_substring=target_market_substring,
        spread_threshold=spread_threshold,
        extreme_pin_threshold=extreme_pin_threshold,
        depth_levels=depth_levels,
    )

    # Also update a "latest" pointer for easy tailing
    latest = out_dir / "latest_microstructure_15m.json"
    summary = generate_microstructure_summary(
        snapshot_path=snapshot_path,
        target_market_substring=target_market_substring,
        spread_threshold=spread_threshold,
        extreme_pin_threshold=extreme_pin_threshold,
        depth_levels=depth_levels,
    )
    latest.write_text(json.dumps(summary, indent=2, sort_keys=True))

    # Log any alerts
    log_microstructure_alerts(summary)

    return report_path


def collect_15m_loop(
    out_dir: Path,
    interval_seconds: float = 60.0,
    max_backoff_seconds: float = 60.0,
    retention_hours: float | None = 24.0,
    max_snapshots: int | None = 1440,
    microstructure_interval_seconds: float = 60.0,
    microstructure_target: str | None = "bitcoin",
    spread_alert_threshold: float = DEFAULT_SPREAD_ALERT_THRESHOLD,
    extreme_pin_threshold: float = DEFAULT_EXTREME_PIN_THRESHOLD,
    depth_levels: int = DEFAULT_DEPTH_LEVELS,
) -> None:
    """Continuously snapshot /crypto/15M and CLOB books.

    On HTTP 429 or transient 5xx/network errors, exponentially back off up to max_backoff_seconds.
    Resets backoff after a successful snapshot.

    Args:
        out_dir: Directory to write snapshots
        interval_seconds: Seconds between snapshots (default: 60s for continuous polling)
        max_backoff_seconds: Maximum backoff on errors
        retention_hours: Hours to retain old snapshots (default: 24h)
        max_snapshots: Max number of snapshots to retain (default: 1440 ~= 24h at 60s intervals)
        microstructure_interval_seconds: Seconds between microstructure analyses
        microstructure_target: Substring to filter markets (e.g., 'bitcoin')
        spread_alert_threshold: Alert threshold for spread
        extreme_pin_threshold: Alert threshold for extreme price pinning
        depth_levels: Number of book levels to include in depth calculation
    """

    out_dir.mkdir(parents=True, exist_ok=True)

    base = max(0.5, float(interval_seconds))
    backoff = base
    last_microstructure_run = 0.0

    while True:
        started = time.time()
        snapshot_path = None

        try:
            snapshot_path = collect_15m_snapshot(out_dir)
            # Touch a small sidecar "latest" pointer (handy for tailing)
            latest = out_dir / "latest_15m.json"
            latest.write_text(
                json.dumps(
                    {"path": str(snapshot_path), "generated_at": datetime.now(UTC).isoformat()}
                )
            )

            backoff = base

            # Log snapshot age metric
            age_seconds = get_latest_snapshot_age_seconds(out_dir, prefix="snapshot_15m")
            if age_seconds is not None:
                logger.info("snapshot_age_seconds=%.1f path=%s", age_seconds, snapshot_path.name)

            # Prune old snapshots by age
            if retention_hours is not None:
                deleted = _prune_old_files(
                    out_dir, prefix="snapshot_15m", retention_hours=float(retention_hours)
                )
                if deleted > 0:
                    logger.info(
                        "pruned_old_snapshots count=%d retention_hours=%.1f",
                        deleted,
                        retention_hours,
                    )

            # Prune by count to prevent unbounded growth
            if max_snapshots is not None:
                deleted = _prune_by_count(
                    out_dir, prefix="snapshot_15m", max_snapshots=max_snapshots
                )
                if deleted > 0:
                    logger.info(
                        "pruned_excess_snapshots count=%d max_snapshots=%d", deleted, max_snapshots
                    )

            # Run microstructure analysis if enough time has passed
            if started - last_microstructure_run >= microstructure_interval_seconds:
                try:
                    _run_microstructure_analysis(
                        snapshot_path=snapshot_path,
                        out_dir=out_dir,
                        target_market_substring=microstructure_target,
                        spread_threshold=spread_alert_threshold,
                        extreme_pin_threshold=extreme_pin_threshold,
                        depth_levels=depth_levels,
                    )
                    last_microstructure_run = started
                except Exception:
                    logger.exception("Microstructure analysis failed")

        except (
            httpx.HTTPStatusError,
            httpx.ConnectError,
            httpx.ReadTimeout,
            httpx.RemoteProtocolError,
        ) as e:
            # httpx raises HTTPStatusError only if raise_for_status() was called.
            status = getattr(getattr(e, "response", None), "status_code", None)
            is_retryable = status in (429, 500, 502, 503, 504) or status is None
            if not is_retryable:
                raise

            backoff = min(max_backoff_seconds, max(backoff * 2, base))

        # sleep remaining time (with small jitter to desync from other bots)
        elapsed = time.time() - started
        sleep_for = max(0.0, backoff - elapsed)
        sleep_for += random.uniform(0.0, min(0.25 * base, 1.0))
        time.sleep(sleep_for)


def collect_5m_loop(
    out_dir: Path,
    interval_seconds: float = 5.0,
    max_backoff_seconds: float = 60.0,
    retention_hours: float | None = None,
) -> None:
    """Continuously snapshot /predictions/5M and CLOB orderbooks.

    On HTTP 429 or transient 5xx/network errors, exponentially back off up to max_backoff_seconds.
    Resets backoff after a successful snapshot.
    """

    out_dir.mkdir(parents=True, exist_ok=True)

    base = max(0.5, float(interval_seconds))
    backoff = base

    while True:
        started = time.time()
        try:
            out_path = collect_5m_snapshot(out_dir)
            # Touch a small sidecar "latest" pointer (handy for tailing)
            latest = out_dir / "latest_5m.json"
            latest.write_text(
                json.dumps({"path": str(out_path), "generated_at": datetime.now(UTC).isoformat()})
            )

            backoff = base

            if retention_hours is not None:
                _prune_old_files(
                    out_dir, prefix="snapshot_5m", retention_hours=float(retention_hours)
                )

        except (
            httpx.HTTPStatusError,
            httpx.ConnectError,
            httpx.ReadTimeout,
            httpx.RemoteProtocolError,
        ) as e:
            # httpx raises HTTPStatusError only if raise_for_status() was called.
            status = getattr(getattr(e, "response", None), "status_code", None)
            is_retryable = status in (429, 500, 502, 503, 504) or status is None
            if not is_retryable:
                raise

            backoff = min(max_backoff_seconds, max(backoff * 2, base))

        # sleep remaining time (with small jitter to desync from other bots)
        elapsed = time.time() - started
        sleep_for = max(0.0, backoff - elapsed)
        sleep_for += random.uniform(0.0, min(0.25 * base, 1.0))
        time.sleep(sleep_for)
