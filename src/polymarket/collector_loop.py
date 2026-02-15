from __future__ import annotations

import json
import random
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx

from .collector import collect_5m_snapshot, collect_15m_snapshot


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


def collect_15m_loop(
    out_dir: Path,
    interval_seconds: float = 5.0,
    max_backoff_seconds: float = 60.0,
    retention_hours: float | None = None,
) -> None:
    """Continuously snapshot /crypto/15M and CLOB books.

    On HTTP 429 or transient 5xx/network errors, exponentially back off up to max_backoff_seconds.
    Resets backoff after a successful snapshot.
    """

    out_dir.mkdir(parents=True, exist_ok=True)

    base = max(0.5, float(interval_seconds))
    backoff = base

    while True:
        started = time.time()
        try:
            out_path = collect_15m_snapshot(out_dir)
            # Touch a small sidecar "latest" pointer (handy for tailing)
            latest = out_dir / "latest_15m.json"
            latest.write_text(json.dumps({"path": str(out_path), "generated_at": datetime.now(UTC).isoformat()}))

            backoff = base

            if retention_hours is not None:
                _prune_old_files(out_dir, prefix="snapshot_15m", retention_hours=float(retention_hours))

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
            latest.write_text(json.dumps({"path": str(out_path), "generated_at": datetime.now(UTC).isoformat()}))

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
