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

# API endpoints to try for fetching fills
# Order matters - most likely endpoints first
FILLS_ENDPOINTS = [
    "/trade-history",  # Polymarket CLOB API v2 endpoint for trade history
    "/trades",  # Alternative naming
    "/history",  # Generic history endpoint
    "/orders",  # Fallback to orders endpoint
]


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


def _mask_value(value: str | None, visible_chars: int = 4) -> str:
    """Mask a sensitive value for logging, showing only last N characters.

    Args:
        value: The value to mask
        visible_chars: Number of characters to show at the end

    Returns:
        Masked string like "****abcd" or "<not set>" if empty
    """
    if not value:
        return "<not set>"
    if len(value) <= visible_chars:
        return "****" + value[-visible_chars:] if len(value) > 0 else "<not set>"
    return "****" + value[-visible_chars:]


def validate_credentials(config) -> dict:
    """Validate API credentials and return diagnostic info.

    Args:
        config: PolymarketConfig instance

    Returns:
        Dict with validation results and diagnostic information
    """
    result = {
        "has_credentials": config.has_credentials,
        "can_trade": config.can_trade,
        "dry_run": config.dry_run,
        "api_key": _mask_value(config.api_key),
        "api_secret": _mask_value(config.api_secret),
        "api_passphrase": _mask_value(config.api_passphrase),
        "api_key_length": len(config.api_key) if config.api_key else 0,
        "api_secret_length": len(config.api_secret) if config.api_secret else 0,
        "api_passphrase_length": len(config.api_passphrase) if config.api_passphrase else 0,
        "warnings": [],
    }

    # Check for partial credentials
    has_any = bool(config.api_key or config.api_secret or config.api_passphrase)
    has_all = config.has_credentials

    if has_any and not has_all:
        if not config.api_key:
            result["warnings"].append("POLYMARKET_API_KEY is missing")
        if not config.api_secret:
            result["warnings"].append("POLYMARKET_API_SECRET is missing")
        if not config.api_passphrase:
            result["warnings"].append("POLYMARKET_API_PASSPHRASE is missing")

    if not has_any:
        result["warnings"].append(
            "No API credentials configured. Set POLYMARKET_API_KEY, "
            "POLYMARKET_API_SECRET, and POLYMARKET_API_PASSPHRASE environment variables."
        )

    return result


def test_api_auth(config) -> dict:
    """Test API authentication with a simple request.

    Args:
        config: PolymarketConfig instance

    Returns:
        Dict with test results including any errors
    """
    result = {
        "success": False,
        "status_code": None,
        "error": None,
        "endpoint": None,
        "response_preview": None,
    }

    if not config.has_credentials:
        result["error"] = "Cannot test auth: no credentials configured"
        return result

    try:
        with _client() as client:
            headers = _auth_headers(config)
            # Try a simple authenticated endpoint first
            result["endpoint"] = "/trade-history"
            resp = client.get("/trade-history", params={"limit": 1}, headers=headers)
            result["status_code"] = resp.status_code

            if resp.status_code == 200:
                result["success"] = True
                try:
                    data = resp.json()
                    if isinstance(data, list):
                        result["response_preview"] = f"List with {len(data)} items"
                    elif isinstance(data, dict):
                        result["response_preview"] = f"Dict with keys: {list(data.keys())}"
                    else:
                        result["response_preview"] = f"Type: {type(data).__name__}"
                except Exception as e:
                    result["response_preview"] = f"Failed to parse JSON: {e}"
            elif resp.status_code == 401:
                result["error"] = "Authentication failed: Invalid credentials"
            elif resp.status_code == 403:
                result["error"] = "Authorization failed: Insufficient permissions"
            else:
                result["error"] = f"Unexpected status code: {resp.status_code}"
                result["response_preview"] = resp.text[:200] if resp.text else "Empty response"

    except httpx.ConnectError as e:
        result["error"] = f"Connection error: {e}"
    except httpx.TimeoutException as e:
        result["error"] = f"Timeout error: {e}"
    except Exception as e:
        result["error"] = f"Error testing auth: {e}"

    return result


def fetch_account_fills(
    since: datetime | None = None,
    limit: int = 100,
    config=None,
) -> list[Fill]:
    """Fetch fills from Polymarket CLOB API for authenticated account.

    Args:
        since: Only fetch fills after this timestamp
        limit: Maximum fills to fetch
        config: Optional PolymarketConfig (loads from env if not provided)

    Returns:
        List of Fill objects from account history
    """
    if config is None:
        config = load_config()

    # Validate and log credentials status
    validation = validate_credentials(config)
    logger.info(
        "API Credentials status: has_credentials=%s, can_trade=%s, dry_run=%s",
        validation["has_credentials"],
        validation["can_trade"],
        validation["dry_run"],
    )
    logger.debug(
        "API Key: %s (length: %d)",
        validation["api_key"],
        validation["api_key_length"],
    )
    logger.debug(
        "API Secret: %s (length: %d)",
        validation["api_secret"],
        validation["api_secret_length"],
    )
    logger.debug(
        "API Passphrase: %s (length: %d)",
        validation["api_passphrase"],
        validation["api_passphrase_length"],
    )

    for warning in validation["warnings"]:
        logger.warning("Credential warning: %s", warning)

    if not config.has_credentials:
        logger.warning(
            "FILLS AUTH MISSING: No API credentials configured, skipping account fills. "
            "Set POLYMARKET_API_KEY, POLYMARKET_API_SECRET, and POLYMARKET_API_PASSPHRASE."
        )
        return []

    # Test auth before attempting to fetch fills
    auth_test = test_api_auth(config)
    if not auth_test["success"]:
        logger.error(
            "FILLS AUTH FAILED: API auth test failed - %s (status: %s, endpoint: %s)",
            auth_test["error"],
            auth_test["status_code"],
            auth_test["endpoint"],
        )
        if auth_test.get("response_preview"):
            logger.error("Response preview: %s", auth_test["response_preview"])
        return []

    logger.info(
        "API auth test passed (%s), fetching fills from account",
        auth_test.get("response_preview", "OK"),
    )

    fills = []
    endpoint_results = []

    try:
        with _client() as client:
            headers = _auth_headers(config)

            # Build params - use timestamp in milliseconds for some APIs
            params: dict[str, str | int] = {"limit": limit}
            if since:
                # Try both ISO format and timestamp formats
                # Some APIs expect milliseconds since epoch
                params["after"] = since.isoformat()

            logger.debug("Fetching fills with params: %s", params)

            for endpoint in FILLS_ENDPOINTS:
                try:
                    logger.debug("Trying endpoint: %s", endpoint)
                    resp = client.get(endpoint, params=params, headers=headers)

                    endpoint_results.append({
                        "endpoint": endpoint,
                        "status": resp.status_code,
                        "content_length": len(resp.content),
                    })

                    if resp.status_code == 200:
                        content_type = resp.headers.get("content-type", "")
                        logger.debug(
                            "Endpoint %s returned 200 (content-type: %s, length: %d)",
                            endpoint,
                            content_type,
                            len(resp.content),
                        )

                        try:
                            data = resp.json()
                        except json.JSONDecodeError as e:
                            logger.warning(
                                "Endpoint %s returned invalid JSON: %s",
                                endpoint,
                                e,
                            )
                            continue

                        # Handle different response formats
                        fill_list: list[dict] = []
                        if isinstance(data, list):
                            fill_list = data
                            logger.debug(
                                "Endpoint %s returned list with %d items",
                                endpoint,
                                len(fill_list),
                            )
                        elif isinstance(data, dict):
                            # Try various common keys
                            for key in ["trades", "fills", "data", "history", "results"]:
                                if key in data:
                                    fill_list = data[key]
                                    logger.debug(
                                        "Endpoint %s returned dict with '%s' key containing %d items",
                                        endpoint,
                                        key,
                                        len(fill_list),
                                    )
                                    break
                            else:
                                # If no recognized key, log the available keys for debugging
                                logger.debug(
                                    "Endpoint %s returned dict with keys: %s",
                                    endpoint,
                                    list(data.keys()),
                                )
                        else:
                            logger.warning(
                                "Endpoint %s returned unexpected type: %s",
                                endpoint,
                                type(data).__name__,
                            )
                            continue

                        # Parse fills from the list
                        parsed_count = 0
                        error_count = 0
                        for fill_data in fill_list:
                            try:
                                fill = Fill.from_dict(fill_data)
                                fills.append(fill)
                                parsed_count += 1
                            except (ValueError, KeyError, TypeError) as e:
                                logger.debug(
                                    "Failed to parse fill from %s: %s (data: %s)",
                                    endpoint,
                                    e,
                                    str(fill_data)[:200],
                                )
                                error_count += 1
                                continue

                        logger.info(
                            "Fetched %d fills from %s (parsed: %d, errors: %d)",
                            len(fill_list),
                            endpoint,
                            parsed_count,
                            error_count,
                        )

                        # If we got fills from this endpoint, don't try others
                        if fills:
                            break

                    elif resp.status_code == 404:
                        logger.debug("Endpoint %s not found (404), trying next", endpoint)
                        continue
                    elif resp.status_code == 401:
                        logger.warning(
                            "Authentication failed for %s (401) - check API credentials",
                            endpoint,
                        )
                        break
                    else:
                        logger.warning(
                            "Unexpected status %d from %s: %s",
                            resp.status_code,
                            endpoint,
                            resp.text[:200],
                        )

                except httpx.HTTPError as e:
                    logger.warning("HTTP error fetching from %s: %s", endpoint, e)
                    continue

        # Log summary of endpoint attempts
        logger.debug("Endpoint results: %s", endpoint_results)

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
        logger.debug("Paper fills file not found: %s", paper_fills_path)
        return []

    fills = []
    line_count = 0
    with open(paper_fills_path, encoding="utf-8") as f:
        for line in f:
            line_count += 1
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                fill = Fill.from_dict(data)
                fills.append(fill)
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(
                    "Failed to parse paper fill at line %d: %s",
                    line_count,
                    e,
                )
                continue

    logger.debug("Loaded %d paper fills from %d lines", len(fills), line_count)
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
    line_count = 0
    with open(fills_path, encoding="utf-8") as f:
        for line in f:
            line_count += 1
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

    logger.debug("Found %d unique tx hashes in %d lines", len(tx_hashes), line_count)
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
    lookback_hours: float = 72.0,
) -> dict:
    """Collect fills from all sources and write to fills.jsonl.

    Args:
        fills_path: Output path for fills.jsonl
        paper_fills_path: Path to paper trading fills.jsonl
        include_account: Whether to fetch real account fills
        include_paper: Whether to include paper trading fills
        since: Only collect fills after this timestamp (deprecated: use lookback_hours)
        lookback_hours: Fixed lookback window in hours (default: 72h)

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
        logger.debug(
            "Using fixed lookback window: %.1fh (since=%s)",
            lookback_hours,
            since.isoformat(),
        )

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
