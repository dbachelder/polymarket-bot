"""Fills collector for Polymarket account data.

Collects fills from:
1. Paper trading fills journal (data/paper_trading/fills.jsonl)
2. Real Polymarket account fills via API (when credentials available)

Writes append-only fills to data/fills.jsonl for PnL verification.
Handles deduplication via transaction hash.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
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

    from .config import PolymarketConfig

logger = logging.getLogger(__name__)

DEFAULT_FILLS_PATH = Path("data/fills.jsonl")
DEFAULT_PAPER_FILLS_PATH = Path("data/paper_trading/fills.jsonl")

# CLOB API endpoint for trade history (requires L2 authentication)
TRADES_ENDPOINT = "/data/trades"


def _generate_signature(
    *,
    secret: str,
    timestamp: str,
    method: str,
    request_path: str,
    body: str = "",
) -> str:
    """Generate HMAC-SHA256 signature for CLOB API authentication.

    Args:
        secret: API secret key
        timestamp: Unix timestamp in milliseconds as string
        method: HTTP method (GET, POST, etc.)
        request_path: API endpoint path
        body: Request body (for POST requests)

    Returns:
        Hex-encoded signature string.
    """
    message = timestamp + method.upper() + request_path + body
    signature = hmac.new(
        secret.encode("utf-8"),
        message.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return signature


def _build_auth_headers(
    config: PolymarketConfig,
    *,
    method: str,
    request_path: str,
    body: str = "",
) -> dict[str, str]:
    """Build L2 authentication headers for CLOB API request.

    The /data/trades endpoint requires L2 headers with signature.

    Args:
        config: Polymarket configuration with credentials
        method: HTTP method
        request_path: API endpoint path
        body: Request body

    Returns:
        Dictionary of HTTP headers.

    Raises:
        ValueError: If credentials are missing.
    """
    if not config.has_credentials:
        msg = "Cannot build auth headers: credentials missing"
        raise ValueError(msg)

    timestamp = str(int(time.time() * 1000))
    signature = _generate_signature(
        secret=config.api_secret or "",
        timestamp=timestamp,
        method=method,
        request_path=request_path,
        body=body,
    )

    return {
        "POLYMARKET-API-KEY": config.api_key or "",
        "POLYMARKET-SIGNATURE": signature,
        "POLYMARKET-TIMESTAMP": timestamp,
        "POLYMARKET-PASSPHRASE": config.api_passphrase or "",
        "Content-Type": "application/json",
    }


def _client(timeout: float = 30.0) -> httpx.Client:
    """Create HTTP client for CLOB API."""
    return httpx.Client(
        base_url=CLOB_BASE,
        timeout=timeout,
        headers={"User-Agent": "polymarket-bot/0.1"},
    )


def fetch_account_fills(
    since: datetime | None = None,
    limit: int = 100,
    config=None,
) -> list[Fill]:
    """Fetch fills from Polymarket CLOB API for authenticated account.

    Uses the /data/trades endpoint with L2 signature authentication.

    Args:
        since: Only fetch fills after this timestamp
        limit: Maximum fills to fetch
        config: Optional PolymarketConfig (loads from env if not provided)

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
            # Build L2 auth headers for /data/trades endpoint
            headers = _build_auth_headers(
                config,
                method="GET",
                request_path=TRADES_ENDPOINT,
            )

            # Build query params - use Unix timestamp for 'after' parameter
            params: dict[str, str | int] = {"limit": limit}
            if since:
                # CLOB API expects Unix timestamp (seconds) for 'after' parameter
                after_timestamp = int(since.timestamp())
                params["after"] = str(after_timestamp)
                logger.debug("Fetching trades after Unix timestamp: %s", after_timestamp)

            logger.debug(
                "Fetching account fills from %s with params: %s",
                TRADES_ENDPOINT,
                params,
            )

            resp = client.get(TRADES_ENDPOINT, params=params, headers=headers)

            # Log response status for diagnostics
            logger.debug(
                "CLOB API response: status=%s content-length=%s",
                resp.status_code,
                len(resp.content),
            )

            if resp.status_code == 200:
                data = resp.json()

                # Log raw response shape for diagnostics
                if isinstance(data, list):
                    logger.info("API returned %d trades (list format)", len(data))
                    trade_list = data
                elif isinstance(data, dict):
                    # Some endpoints wrap results in a data key
                    trade_list = data.get("trades", data.get("data", []))
                    logger.info(
                        "API returned %d trades (wrapped format, keys: %s)",
                        len(trade_list),
                        list(data.keys()),
                    )
                else:
                    logger.warning("Unexpected API response type: %s", type(data))
                    trade_list = []

                # Parse each trade into a Fill object
                for trade_data in trade_list:
                    try:
                        fill = Fill.from_dict(trade_data)
                        fills.append(fill)
                    except (ValueError, KeyError, TypeError) as e:
                        logger.warning("Failed to parse fill: %s. Data: %s", e, trade_data)
                        continue

                logger.info(
                    "Successfully fetched and parsed %d/%d fills from %s",
                    len(fills),
                    len(trade_list),
                    TRADES_ENDPOINT,
                )

            elif resp.status_code == 401:
                logger.warning(
                    "Authentication failed (401) for %s. "
                    "Check POLYMARKET_API_KEY, POLYMARKET_API_SECRET, "
                    "and POLYMARKET_API_PASSPHRASE are correct.",
                    TRADES_ENDPOINT,
                )
            elif resp.status_code == 403:
                logger.warning(
                    "Authorization failed (403) for %s. "
                    "API credentials may lack permission for this endpoint.",
                    TRADES_ENDPOINT,
                )
            else:
                logger.warning(
                    "Unexpected status %d from %s: %s",
                    resp.status_code,
                    TRADES_ENDPOINT,
                    resp.text[:500],
                )

    except httpx.HTTPError as e:
        logger.exception("HTTP error fetching account fills: %s", e)
    except Exception:
        logger.exception("Error fetching account fills")

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


def collect_fills(
    fills_path: Path | None = None,
    paper_fills_path: Path | None = None,
    include_account: bool = True,
    include_paper: bool = True,
    since: datetime | None = None,
    lookback_hours: float = 48.0,
) -> dict:
    """Collect fills from all sources and write to fills.jsonl.

    Args:
        fills_path: Output path for fills.jsonl
        paper_fills_path: Path to paper trading fills.jsonl
        include_account: Whether to fetch real account fills
        include_paper: Whether to include paper trading fills
        since: Only collect fills after this timestamp (deprecated: use lookback_hours)
        lookback_hours: Fixed lookback window in hours (default: 48h)

    Returns:
        Dict with collection results summary
    """
    if fills_path is None:
        fills_path = DEFAULT_FILLS_PATH

    # Ensure directory exists
    fills_path.parent.mkdir(parents=True, exist_ok=True)

    # Get existing transaction hashes for deduplication
    existing_txs = get_existing_tx_hashes(fills_path)

    # Use fixed lookback window instead of since=last_fill to avoid missing fills
    # when last_fill timestamp is stale or there are clock/sync issues
    if since is None:
        since = datetime.now(UTC) - timedelta(hours=lookback_hours)
        logger.debug("Using fixed lookback window: %.1fh (since=%s)", lookback_hours, since.isoformat())

    results = {
        "fills_path": str(fills_path),
        "account_fills": 0,
        "paper_fills": 0,
        "duplicates_skipped": 0,
        "total_appended": 0,
        "since": since.isoformat() if since else None,
        "lookback_hours": lookback_hours,
        "timestamp": datetime.now(UTC).isoformat(),
    }

    all_fills = []

    # Fetch account fills
    if include_account:
        try:
            account_fills = fetch_account_fills(since=since)
            for fill in account_fills:
                if fill.transaction_hash and fill.transaction_hash in existing_txs:
                    results["duplicates_skipped"] += 1
                    continue
                all_fills.append(fill)
            results["account_fills"] = len(account_fills)
        except Exception:
            logger.exception("Error fetching account fills")

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
        except Exception:
            logger.exception("Error loading paper fills")

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

    return results


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
