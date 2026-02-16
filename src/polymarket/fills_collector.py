"""Fills collector for Polymarket account data.

Collects fills from:
1. Paper trading fills journal (data/paper_trading/fills.jsonl)
2. Real Polymarket account fills via API (when credentials available)

Writes append-only fills to data/fills.jsonl for PnL verification.
Handles deduplication via transaction hash.
"""

from __future__ import annotations

import json
import logging
import random
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import httpx

from .config import load_config
from .endpoints import CLOB_BASE
from .pnl import Fill

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

DEFAULT_FILLS_PATH = Path("data/fills.jsonl")
DEFAULT_PAPER_FILLS_PATH = Path("data/paper_trading/fills.jsonl")

# Health check thresholds
DEFAULT_STALE_HOURS = 6
DEFAULT_OVERLAP_HOURS = 2


def _client(timeout: float = 30.0) -> httpx.Client:
    """Create HTTP client for CLOB API."""
    return httpx.Client(
        base_url=CLOB_BASE,
        timeout=timeout,
        headers={"User-Agent": "polymarket-bot/0.1"},
    )


def _auth_headers(config) -> dict[str, str]:
    """Build authentication headers for CLOB API.

    Uses API key and passphrase for authenticated endpoints.
    Note: Full signature-based auth may be needed for some endpoints.
    """
    headers = {"User-Agent": "polymarket-bot/0.1"}
    if config.api_key:
        headers["POLYMARKET_API_KEY"] = config.api_key
    if config.api_passphrase:
        headers["POLYMARKET_PASSPHRASE"] = config.api_passphrase
    return headers


def fetch_account_fills(
    since: datetime | None = None,
    limit: int = 100,
    config=None,
    max_pages: int = 10,
) -> list[Fill]:
    """Fetch fills from Polymarket CLOB API for authenticated account.

    Args:
        since: Only fetch fills after this timestamp
        limit: Maximum fills to fetch per page
        config: Optional PolymarketConfig (loads from env if not provided)
        max_pages: Maximum pages to fetch (safety limit)

    Returns:
        List of Fill objects from account history
    """
    if config is None:
        config = load_config()

    if not config.has_credentials:
        logger.debug("No API credentials configured, skipping account fills")
        return []

    fills = []
    try:
        with _client() as client:
            headers = _auth_headers(config)

            # The CLOB API endpoint for fills/trades
            # Common patterns: /trades, /fills, /orders?status=FILLED
            params: dict[str, str | int] = {"limit": limit}
            if since:
                params["after"] = since.isoformat()

            # Try common endpoints with pagination
            endpoints_to_try = ["/trades", "/fills", "/orders"]

            for endpoint in endpoints_to_try:
                try:
                    page_fills = []
                    cursor = None
                    pages = 0

                    while pages < max_pages:
                        page_params = dict(params)
                        if cursor:
                            page_params["cursor"] = cursor

                        resp = client.get(endpoint, params=page_params, headers=headers)
                        if resp.status_code == 200:
                            data = resp.json()
                            # Handle different response formats
                            if isinstance(data, list):
                                fill_list = data
                                cursor = None  # No pagination for list responses
                            elif isinstance(data, dict):
                                fill_list = data.get(
                                    "trades", data.get("fills", data.get("data", []))
                                )
                                cursor = data.get("next_cursor") or data.get("cursor")
                            else:
                                fill_list = []
                                cursor = None

                            for fill_data in fill_list:
                                try:
                                    fill = Fill.from_dict(fill_data)
                                    page_fills.append(fill)
                                except (ValueError, KeyError, TypeError) as e:
                                    logger.warning("Failed to parse fill: %s", e)
                                    continue

                            pages += 1
                            # Stop if we got fewer than limit results (no more pages)
                            if len(fill_list) < limit:
                                break
                            # Stop if no cursor for pagination
                            if not cursor:
                                break
                        elif resp.status_code == 404:
                            break  # Try next endpoint
                        elif resp.status_code == 401:
                            logger.warning("Authentication failed for %s", endpoint)
                            return []
                        else:
                            logger.warning(
                                "Unexpected status %d from %s", resp.status_code, endpoint
                            )
                            break

                    if page_fills:
                        fills.extend(page_fills)
                        logger.info(
                            "Fetched %d fills from %s (%d pages)", len(page_fills), endpoint, pages
                        )
                        break  # Found working endpoint, stop trying others

                except httpx.HTTPError as e:
                    logger.warning("HTTP error fetching from %s: %s", endpoint, e)
                    continue

    except Exception as e:
        logger.exception("Error fetching account fills: %s", e)

    return fills


def load_paper_fills(paper_fills_path: Path | None = None) -> list[Fill]:
    """Load fills from paper trading journal.

    Args:
        paper_fills_path: Path to paper trading fills.jsonl

    Returns:
        List of Fill objects from paper trading
    """
    if paper_fills_path is None:
        paper_fills_path = DEFAULT_PAPER_FILLS_PATH

    if not paper_fills_path.exists():
        return []

    fills = []
    with open(paper_fills_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                fill = Fill.from_dict(data)
                fills.append(fill)
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning("Failed to parse paper fill: %s", e)
                continue

    return fills


def get_existing_tx_hashes(fills_path: Path) -> set[str]:
    """Get set of transaction hashes already in fills file.

    Args:
        fills_path: Path to fills.jsonl

    Returns:
        Set of transaction hashes for deduplication
    """
    if not fills_path.exists():
        return set()

    tx_hashes = set()
    with open(fills_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                tx_hash = data.get("transaction_hash") or data.get("tx_hash")
                if tx_hash:
                    tx_hashes.add(tx_hash)
            except json.JSONDecodeError:
                continue

    return tx_hashes


def get_last_fill_timestamp(fills_path: Path) -> datetime | None:
    """Get the timestamp of the most recent fill in the file.

    Args:
        fills_path: Path to fills.jsonl

    Returns:
        Timestamp of last fill, or None if no fills
    """
    if not fills_path.exists():
        return None

    last_ts = None
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

    return last_ts


def append_fills(fills: Sequence[Fill], fills_path: Path) -> int:
    """Append fills to the fills.jsonl file.

    Args:
        fills: List of fills to append
        fills_path: Path to fills.jsonl

    Returns:
        Number of fills appended
    """
    if not fills:
        return 0

    fills_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(fills_path, "a", encoding="utf-8") as f:
        for fill in fills:
            record = {
                "token_id": fill.token_id,
                "side": fill.side,
                "size": str(fill.size),
                "price": str(fill.price),
                "fee": str(fill.fee),
                "timestamp": fill.timestamp,
                "transaction_hash": fill.transaction_hash,
                "market_slug": fill.market_slug,
            }
            f.write(json.dumps(record, sort_keys=True) + "\n")
            count += 1

    return count


def _check_fills_age_health(
    fills_path: Path,
    stale_hours: float = DEFAULT_STALE_HOURS,
    last_seen_ts: datetime | None = None,
    query_since: datetime | None = None,
) -> dict:
    """Check if fills are stale and emit warning if needed.

    Args:
        fills_path: Path to fills.jsonl
        stale_hours: Hours to consider fills stale
        last_seen_ts: Last timestamp seen before query (for logging)
        query_since: The 'since' parameter used in the query (for logging)

    Returns:
        Dict with health status
    """
    now = datetime.now(UTC)
    last_fill = get_last_fill_timestamp(fills_path)

    result = {
        "healthy": True,
        "last_fill_at": last_fill.isoformat() if last_fill else None,
        "hours_since_last_fill": None,
        "stale_hours": stale_hours,
        "warning_emitted": False,
    }

    if last_fill is None:
        result["healthy"] = False
        logger.warning(
            "FILLS HEALTH: No fills found in file. last_seen_ts=%s, query_since=%s, fills_path=%s",
            last_seen_ts.isoformat() if last_seen_ts else None,
            query_since.isoformat() if query_since else None,
            fills_path,
        )
        return result

    hours_since = (now - last_fill).total_seconds() / 3600
    result["hours_since_last_fill"] = round(hours_since, 2)

    if hours_since > stale_hours:
        result["healthy"] = False
        result["warning_emitted"] = True
        logger.warning(
            "FILLS HEALTH: Fills are stale (%.1f hours > %.1f hours threshold). "
            "last_fill_at=%s, last_seen_ts=%s, query_since=%s, fills_path=%s",
            hours_since,
            stale_hours,
            last_fill.isoformat(),
            last_seen_ts.isoformat() if last_seen_ts else None,
            query_since.isoformat() if query_since else None,
            fills_path,
        )
    else:
        logger.debug(
            "FILLS HEALTH: Fills are fresh (%.1f hours < %.1f hours threshold). last_fill_at=%s",
            hours_since,
            stale_hours,
            last_fill.isoformat(),
        )

    return result


def collect_fills(
    fills_path: Path | None = None,
    paper_fills_path: Path | None = None,
    include_account: bool = True,
    include_paper: bool = True,
    since: datetime | None = None,
    overlap_hours: float = DEFAULT_OVERLAP_HOURS,
    check_health: bool = True,
    stale_hours: float = DEFAULT_STALE_HOURS,
) -> dict:
    """Collect fills from all sources and write to fills.jsonl.

    Args:
        fills_path: Output path for fills.jsonl
        paper_fills_path: Path to paper trading fills.jsonl
        include_account: Whether to fetch real account fills
        include_paper: Whether to include paper trading fills
        since: Only collect fills after this timestamp
        overlap_hours: Hours of overlap when querying from last_seen_ts
        check_health: Whether to check fills age health after collection
        stale_hours: Hours to consider fills stale for health check

    Returns:
        Dict with collection results summary
    """
    if fills_path is None:
        fills_path = DEFAULT_FILLS_PATH

    # Ensure directory exists
    fills_path.parent.mkdir(parents=True, exist_ok=True)

    # Get existing transaction hashes for deduplication
    existing_txs = get_existing_tx_hashes(fills_path)

    # Get last fill timestamp if not provided
    last_seen_ts = None
    query_since = since
    if query_since is None:
        last_seen_ts = get_last_fill_timestamp(fills_path)
        if last_seen_ts:
            # Apply overlap to catch late-arriving fills
            query_since = last_seen_ts - timedelta(hours=overlap_hours)

    results = {
        "fills_path": str(fills_path),
        "account_fills": 0,
        "paper_fills": 0,
        "duplicates_skipped": 0,
        "total_appended": 0,
        "last_seen_ts": last_seen_ts.isoformat() if last_seen_ts else None,
        "query_since": query_since.isoformat() if query_since else None,
        "overlap_hours": overlap_hours,
        "timestamp": datetime.now(UTC).isoformat(),
    }

    all_fills = []

    # Fetch account fills
    if include_account:
        try:
            account_fills = fetch_account_fills(since=query_since)
            for fill in account_fills:
                if fill.transaction_hash and fill.transaction_hash in existing_txs:
                    results["duplicates_skipped"] += 1
                    continue
                all_fills.append(fill)
            results["account_fills"] = len(account_fills)
        except Exception as e:
            logger.exception("Error fetching account fills: %s", e)

    # Load paper fills
    if include_paper:
        try:
            paper_fills = load_paper_fills(paper_fills_path)
            for fill in paper_fills:
                if fill.transaction_hash and fill.transaction_hash in existing_txs:
                    results["duplicates_skipped"] += 1
                    continue
                all_fills.append(fill)
            results["paper_fills"] = len(paper_fills)
        except Exception as e:
            logger.exception("Error loading paper fills: %s", e)

    # Sort by timestamp
    all_fills.sort(key=lambda f: f.timestamp)

    # Append to file
    appended = append_fills(all_fills, fills_path)
    results["total_appended"] = appended

    logger.info(
        "Collected %d fills (%d account, %d paper, %d duplicates skipped)",
        appended,
        results["account_fills"],
        results["paper_fills"],
        results["duplicates_skipped"],
    )

    # Check health after collection
    if check_health:
        health = _check_fills_age_health(
            fills_path=fills_path,
            stale_hours=stale_hours,
            last_seen_ts=last_seen_ts,
            query_since=query_since,
        )
        results["health"] = health

    return results


def collect_fills_loop(
    fills_path: Path | None = None,
    paper_fills_path: Path | None = None,
    include_account: bool = True,
    include_paper: bool = True,
    interval_seconds: float = 60.0,
    overlap_hours: float = DEFAULT_OVERLAP_HOURS,
    stale_hours: float = DEFAULT_STALE_HOURS,
    max_backoff_seconds: float = 300.0,
) -> None:
    """Continuously collect fills at regular intervals.

    This loop runs indefinitely, collecting fills from all sources
    and appending them to fills.jsonl. It handles backoff on errors
    and logs health status.

    Args:
        fills_path: Output path for fills.jsonl
        paper_fills_path: Path to paper trading fills.jsonl
        include_account: Whether to fetch real account fills
        include_paper: Whether to include paper trading fills
        interval_seconds: Seconds between collection runs (default: 60s)
        overlap_hours: Hours of overlap when querying from last_seen_ts
        stale_hours: Hours to consider fills stale for health check
        max_backoff_seconds: Maximum backoff on errors
    """
    if fills_path is None:
        fills_path = DEFAULT_FILLS_PATH

    fills_path = Path(fills_path)
    fills_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Starting fills collection loop: interval=%.0fs, overlap=%.1fh, stale_threshold=%.1fh",
        interval_seconds,
        overlap_hours,
        stale_hours,
    )

    base_interval = max(1.0, float(interval_seconds))
    backoff = base_interval

    while True:
        started = time.time()

        try:
            result = collect_fills(
                fills_path=fills_path,
                paper_fills_path=paper_fills_path,
                include_account=include_account,
                include_paper=include_paper,
                since=None,  # Always use last_seen_ts - overlap
                overlap_hours=overlap_hours,
                check_health=True,
                stale_hours=stale_hours,
            )

            # Log collection results
            logger.info(
                "Loop iteration complete: appended=%d, account=%d, paper=%d, health_healthy=%s",
                result["total_appended"],
                result["account_fills"],
                result["paper_fills"],
                result.get("health", {}).get("healthy", True),
            )

            # Reset backoff on success
            backoff = base_interval

        except Exception as e:
            logger.exception("Error in fills collection loop: %s", e)
            # Increase backoff on error
            backoff = min(max_backoff_seconds, backoff * 2)

        # Sleep until next iteration
        elapsed = time.time() - started
        sleep_for = max(0.0, backoff - elapsed)
        sleep_for += random.uniform(0.0, min(5.0, interval_seconds * 0.1))  # Small jitter
        time.sleep(sleep_for)


def get_fills_summary(fills_path: Path | None = None) -> dict:
    """Get summary of fills data.

    Args:
        fills_path: Path to fills.jsonl

    Returns:
        Dict with summary statistics
    """
    if fills_path is None:
        fills_path = DEFAULT_FILLS_PATH

    summary = {
        "fills_path": str(fills_path),
        "exists": fills_path.exists(),
        "total_fills": 0,
        "last_fill_at": None,
        "first_fill_at": None,
        "unique_tokens": 0,
        "unique_markets": 0,
        "age_seconds": None,
    }

    if not fills_path.exists():
        return summary

    tokens = set()
    markets = set()
    first_ts = None
    last_ts = None

    with open(fills_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                summary["total_fills"] += 1

                if data.get("token_id"):
                    tokens.add(data["token_id"])
                if data.get("market_slug"):
                    markets.add(data["market_slug"])

                ts_str = data.get("timestamp") or data.get("created_at")
                if ts_str:
                    try:
                        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                        if first_ts is None or ts < first_ts:
                            first_ts = ts
                        if last_ts is None or ts > last_ts:
                            last_ts = ts
                    except ValueError:
                        pass

            except json.JSONDecodeError:
                continue

    summary["unique_tokens"] = len(tokens)
    summary["unique_markets"] = len(markets)

    if first_ts:
        summary["first_fill_at"] = first_ts.isoformat()
    if last_ts:
        summary["last_fill_at"] = last_ts.isoformat()
        summary["age_seconds"] = (datetime.now(UTC) - last_ts).total_seconds()

    return summary
