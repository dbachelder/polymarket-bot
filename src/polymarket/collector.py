from __future__ import annotations

import json
import random
import time
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

from .clob import get_book
from .site import (
    extract_5m_markets,
    extract_15m_markets,
    fetch_crypto_page,
    fetch_predictions_page,
    parse_next_data,
)


def collect_5m_snapshot(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    html = fetch_predictions_page("5M")
    data = parse_next_data(html)
    markets = extract_5m_markets(data)

    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"snapshot_5m_{ts}.json"

    payload: dict = {
        "generated_at": datetime.now(UTC).isoformat(),
        "count": len(markets),
        "markets": [],
    }

    for m in markets:
        yes_id, no_id = m.clob_token_ids
        payload["markets"].append(
            {
                **asdict(m),
                "books": {
                    "yes": get_book(yes_id),
                    "no": get_book(no_id),
                },
            }
        )

    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return out_path


def _get_book_with_backoff(
    token_id: str,
    *,
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
) -> dict:
    """Fetch orderbook with exponential backoff on 429/5xx errors.

    Args:
        token_id: CLOB token ID to fetch book for
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds

    Returns:
        Orderbook dict from CLOB API

    Raises:
        Exception: If all retries exhausted
    """
    for attempt in range(max_retries):
        try:
            return get_book(token_id)
        except Exception as e:
            # Check if it's a rate limit (429) or server error (5xx)
            is_rate_limit = "429" in str(e) or "Too Many Requests" in str(e)
            is_server_error = any(f"{code}" in str(e) for code in [500, 502, 503, 504])

            if not (is_rate_limit or is_server_error):
                raise

            if attempt == max_retries - 1:
                raise

            # Exponential backoff with jitter
            delay = min(base_delay * (2**attempt), max_delay)
            jitter = random.uniform(0, 0.1 * delay)
            time.sleep(delay + jitter)

    raise RuntimeError("Unreachable")


def collect_15m_snapshot(
    out_dir: Path,
    *,
    use_backoff: bool = True,
) -> Path:
    """Snapshot 15M crypto markets from /crypto/15M with CLOB orderbooks.

    Args:
        out_dir: Directory to write snapshot file
        use_backoff: If True, use exponential backoff on rate limits

    Returns:
        Path to written snapshot file
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    html = fetch_crypto_page("15M")
    data = parse_next_data(html)
    markets = extract_15m_markets(data)

    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"snapshot_15m_{ts}.json"

    payload: dict = {
        "generated_at": datetime.now(UTC).isoformat(),
        "count": len(markets),
        "markets": [],
    }

    book_fn = _get_book_with_backoff if use_backoff else get_book

    for m in markets:
        yes_id, no_id = m.clob_token_ids
        payload["markets"].append(
            {
                **asdict(m),
                "books": {
                    "yes": book_fn(yes_id),
                    "no": book_fn(no_id),
                },
            }
        )

    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return out_path


def prune_snapshots(
    out_dir: Path,
    *,
    pattern: str = "snapshot_15m_*.json",
    retention_hours: float = 24.0,
) -> int:
    """Prune old snapshot files based on retention policy.

    Args:
        out_dir: Directory containing snapshots
        pattern: Glob pattern to match files
        retention_hours: Keep files newer than this many hours

    Returns:
        Number of files deleted
    """
    cutoff = datetime.now(UTC) - timedelta(hours=retention_hours)
    deleted = 0

    for f in out_dir.glob(pattern):
        try:
            # Extract timestamp from filename: snapshot_15m_20260215T025545Z.json
            ts_str = f.stem.split("_")[-1]
            ts = datetime.strptime(ts_str, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
            if ts < cutoff:
                f.unlink()
                deleted += 1
        except (ValueError, OSError):
            continue

    return deleted
