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
    interval_seconds: float = 5.0,
    max_backoff_seconds: float = 60.0,
    retention_hours: float | None = None,
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
        interval_seconds: Seconds between snapshots
        max_backoff_seconds: Maximum backoff on errors
        retention_hours: Hours to retain old snapshots (None = keep forever)
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
                json.dumps({"path": str(snapshot_path), "generated_at": datetime.now(UTC).isoformat()})
            )

            backoff = base

            if retention_hours is not None:
                _prune_old_files(out_dir, prefix="snapshot_15m", retention_hours=float(retention_hours))

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

        except (httpx.HTTPStatusError, httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError) as e:
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
                _prune_old_files(out_dir, prefix="snapshot_5m", retention_hours=float(retention_hours))

        except (httpx.HTTPStatusError, httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError) as e:
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
