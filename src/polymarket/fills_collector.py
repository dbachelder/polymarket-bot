"""Fills collector for Polymarket account data.

Collects fills from:
1. Paper trading fills journal (data/paper_trading/fills.jsonl)
2. Real Polymarket account fills via API (when credentials available)

Writes append-only fills to data/fills.jsonl for PnL verification.
Handles deduplication via transaction hash.

Includes:
- Retry logic with exponential backoff for transient errors
- Circuit breaker pattern for auth failures
- Heartbeat logging to distinguish silence from failure
"""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

from .config import load_config
from .endpoints import CLOB_BASE
from .pnl import Fill
from .trading import _build_auth_headers

if TYPE_CHECKING:
    from collections.abc import Sequence


class AuthenticationError(Exception):
    """Raised when API authentication fails or credentials are missing."""

logger = logging.getLogger(__name__)

DEFAULT_FILLS_PATH = Path("data/fills.jsonl")
DEFAULT_PAPER_FILLS_PATH = Path("data/paper_trading/fills.jsonl")

# Retry configuration
DEFAULT_MAX_RETRIES = 3
DEFAULT_BASE_DELAY = 1.0
DEFAULT_MAX_DELAY = 30.0
DEFAULT_BACKOFF_FACTOR = 2.0

# Circuit breaker configuration
DEFAULT_CB_FAILURE_THRESHOLD = 5
DEFAULT_CB_RESET_TIMEOUT_SECONDS = 300  # 5 minutes

# Heartbeat configuration
DEFAULT_HEARTBEAT_INTERVAL_SECONDS = 1800  # 30 minutes


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if recovered


@dataclass
class CircuitBreaker:
    """Circuit breaker for API calls.

    Prevents repeated calls to a failing service by opening the circuit
    after a threshold of failures. After a timeout, enters half-open state
    to test if the service has recovered.

    Args:
        failure_threshold: Number of failures before opening circuit
        reset_timeout_seconds: Seconds before attempting recovery
        name: Circuit breaker name for logging
    """

    failure_threshold: int = DEFAULT_CB_FAILURE_THRESHOLD
    reset_timeout_seconds: float = DEFAULT_CB_RESET_TIMEOUT_SECONDS
    name: str = "default"

    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _failure_count: int = field(default=0, init=False)
    _last_failure_time: float | None = field(default=None, init=False)
    _last_success_time: float | None = field(default=None, init=False)

    @property
    def state(self) -> CircuitState:
        """Get current circuit state, transitioning if needed."""
        if self._state == CircuitState.OPEN:
            # Check if we should try half-open
            if self._last_failure_time is not None:
                elapsed = time.time() - self._last_failure_time
                if elapsed >= self.reset_timeout_seconds:
                    logger.info(
                        "Circuit breaker '%s' entering HALF_OPEN after %.0fs timeout",
                        self.name,
                        elapsed,
                    )
                    self._state = CircuitState.HALF_OPEN
        return self._state

    def can_execute(self) -> bool:
        """Check if execution is allowed."""
        return self.state in (CircuitState.CLOSED, CircuitState.HALF_OPEN)

    def record_success(self) -> None:
        """Record a successful call."""
        if self._state == CircuitState.HALF_OPEN:
            logger.info("Circuit breaker '%s' closing (recovered)", self.name)
            self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_success_time = time.time()

    def record_failure(self, is_auth_failure: bool = False) -> None:
        """Record a failed call.

        Args:
            is_auth_failure: If True, open circuit immediately (auth failures
                are unlikely to resolve without intervention)
        """
        self._failure_count += 1
        self._last_failure_time = time.time()

        if is_auth_failure:
            # Auth failures open circuit immediately
            if self._state != CircuitState.OPEN:
                logger.error(
                    "Circuit breaker '%s' OPEN due to auth failure (credentials invalid)",
                    self.name,
                )
                self._state = CircuitState.OPEN
        elif self._failure_count >= self.failure_threshold:
            if self._state != CircuitState.OPEN:
                logger.error(
                    "Circuit breaker '%s' OPEN after %d failures",
                    self.name,
                    self._failure_count,
                )
                self._state = CircuitState.OPEN

    def get_status(self) -> dict[str, Any]:
        """Get circuit breaker status for diagnostics."""
        now = time.time()
        return {
            "name": self.name,
            "state": self._state.value,
            "failure_count": self._failure_count,
            "failure_threshold": self.failure_threshold,
            "seconds_since_last_failure": (
                now - self._last_failure_time if self._last_failure_time else None
            ),
            "seconds_since_last_success": (
                now - self._last_success_time if self._last_success_time else None
            ),
            "reset_timeout_seconds": self.reset_timeout_seconds,
        }


# Global circuit breaker instance for fills API
_fills_circuit_breaker = CircuitBreaker(
    failure_threshold=DEFAULT_CB_FAILURE_THRESHOLD,
    reset_timeout_seconds=DEFAULT_CB_RESET_TIMEOUT_SECONDS,
    name="fills_api",
)


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


def _is_retryable_error(status_code: int | None, error: Exception | None = None) -> bool:
    """Check if an error is retryable.

    Args:
        status_code: HTTP status code if available
        error: Exception if available

    Returns:
        True if the error is likely transient and retryable
    """
    # Never retry auth failures
    if status_code in (401, 403):
        return False

    # Retry rate limiting with backoff
    if status_code == 429:
        return True

    # Retry server errors
    if status_code in (500, 502, 503, 504):
        return True

    # Retry on network/timeout errors
    if isinstance(error, (httpx.ConnectError, httpx.TimeoutException, httpx.ReadTimeout)):
        return True

    # Don't retry other 4xx errors (client errors)
    if status_code and 400 <= status_code < 500:
        return False

    # Default: retry unknown errors
    return True


def _calculate_retry_delay(
    attempt: int,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
) -> float:
    """Calculate retry delay with exponential backoff and jitter.

    Args:
        attempt: Current retry attempt (0-indexed)
        base_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        backoff_factor: Multiplier for each retry

    Returns:
        Delay in seconds with jitter applied
    """
    delay = min(base_delay * (backoff_factor**attempt), max_delay)
    # Add jitter (±25%)
    jitter = delay * random.uniform(-0.25, 0.25)
    return delay + jitter


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


def check_api_auth(config) -> dict:
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
    }

    if not config.has_credentials:
        result["error"] = "Cannot test auth: no credentials configured"
        return result

    # Check circuit breaker first
    if not _fills_circuit_breaker.can_execute():
        result["error"] = "Circuit breaker is OPEN - too many recent failures"
        result["circuit_breaker"] = _fills_circuit_breaker.get_status()
        return result

    try:
        with _client() as client:
            headers = _auth_headers(config)
            # Try a simple authenticated endpoint first
            result["endpoint"] = "/orders"
            resp = client.get("/orders", params={"limit": 1}, headers=headers)
            result["status_code"] = resp.status_code

            if resp.status_code == 200:
                result["success"] = True
                _fills_circuit_breaker.record_success()
            elif resp.status_code == 401:
                result["error"] = "Authentication failed: Invalid credentials"
                _fills_circuit_breaker.record_failure(is_auth_failure=True)
            elif resp.status_code == 403:
                result["error"] = "Authorization failed: Insufficient permissions"
                _fills_circuit_breaker.record_failure(is_auth_failure=True)
            else:
                result["error"] = f"Unexpected status code: {resp.status_code}"
                _fills_circuit_breaker.record_failure()

    except httpx.ConnectError as e:
        result["error"] = f"Connection error: {e}"
        _fills_circuit_breaker.record_failure()
    except httpx.TimeoutException as e:
        result["error"] = f"Timeout error: {e}"
        _fills_circuit_breaker.record_failure()
    except Exception as e:
        result["error"] = f"Error testing auth: {e}"
        _fills_circuit_breaker.record_failure()

    return result


def fetch_account_fills_with_retry(
    since: datetime | None = None,
    limit: int = 100,
    config=None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
) -> list[Fill]:
    """Fetch fills from Polymarket CLOB API with retry logic.

    Args:
        since: Only fetch fills after this timestamp
        limit: Maximum fills to fetch
        config: Optional PolymarketConfig (loads from env if not provided)
        max_retries: Maximum number of retry attempts
        base_delay: Initial retry delay in seconds
        max_delay: Maximum retry delay in seconds

    Returns:
        List of Fill objects from account history

    Raises:
        AuthenticationError: If API credentials are missing or invalid.
    """
    if config is None:
        config = load_config()

    if not config.has_credentials:
        msg = (
            f"FILLS AUTH FAILED: No API credentials configured. "
            f"Set POLYMARKET_API_KEY, POLYMARKET_API_SECRET, and POLYMARKET_API_PASSPHRASE. "
            f"Key present: {bool(config.api_key)}, "
            f"Secret present: {bool(config.api_secret)}, "
            f"Passphrase present: {bool(config.api_passphrase)}"
        )
        logger.error(msg)
        raise AuthenticationError(msg)

    # Check circuit breaker
    if not _fills_circuit_breaker.can_execute():
        cb_status = _fills_circuit_breaker.get_status()
        logger.error(
            "FILLS CIRCUIT OPEN: Circuit breaker is %s after %d failures. "
            "Last failure %.0fs ago. Waiting for timeout (%.0fs) before retry.",
            cb_status["state"],
            cb_status["failure_count"],
            cb_status["seconds_since_last_failure"] or 0,
            cb_status["reset_timeout_seconds"],
        )
        return []

    fills = []
    last_error = None

    for attempt in range(max_retries + 1):
        if attempt > 0:
            delay = _calculate_retry_delay(attempt - 1, base_delay, max_delay)
            logger.info("Retry attempt %d/%d after %.1fs delay", attempt, max_retries, delay)
            time.sleep(delay)

        try:
            with _client() as client:

                # The CLOB API endpoint for fills/trades
                params: dict[str, str | int] = {"limit": limit}
                if since:
                    params["after"] = since.isoformat()

                # Try common endpoints
                endpoints_to_try = ["/trades", "/fills", "/orders"]
                endpoint_success = False

                for endpoint in endpoints_to_try:
                    try:
                        # Use signature-based auth headers (required for account endpoints)
                        headers = _build_auth_headers(config, method="GET", request_path=endpoint)
                        headers["User-Agent"] = "polymarket-bot/0.1"
                        resp = client.get(endpoint, params=params, headers=headers)

                        if resp.status_code == 200:
                            data = resp.json()
                            # Handle different response formats
                            if isinstance(data, list):
                                fill_list = data
                            elif isinstance(data, dict):
                                fill_list = data.get("trades", data.get("fills", data.get("data", [])))
                            else:
                                fill_list = []

                            for fill_data in fill_list:
                                try:
                                    fill = Fill.from_dict(fill_data)
                                    fills.append(fill)
                                except (ValueError, KeyError, TypeError) as e:
                                    logger.warning("Failed to parse fill: %s", e)
                                    continue

                            logger.info("Fetched %d fills from %s", len(fills), endpoint)
                            _fills_circuit_breaker.record_success()
                            endpoint_success = True
                            break

                        elif resp.status_code == 404:
                            continue  # Try next endpoint

                        elif resp.status_code == 401:
                            logger.error(
                                "FILLS AUTH FAILED: HTTP 401 from %s - invalid credentials. "
                                "Check POLYMARKET_API_KEY, POLYMARKET_API_SECRET, POLYMARKET_API_PASSPHRASE",
                                endpoint,
                            )
                            _fills_circuit_breaker.record_failure(is_auth_failure=True)
                            # Don't retry auth failures
                            return fills

                        elif resp.status_code == 429:
                            logger.warning("Rate limited (429) on %s, will retry", endpoint)
                            # Let retry logic handle this
                            last_error = httpx.HTTPStatusError(
                                f"Rate limited: {resp.status_code}",
                                request=resp.request,
                                response=resp,
                            )
                            break  # Break endpoint loop, outer retry will handle

                        else:
                            logger.warning("Unexpected status %d from %s", resp.status_code, endpoint)
                            last_error = httpx.HTTPStatusError(
                                f"Unexpected status: {resp.status_code}",
                                request=resp.request,
                                response=resp,
                            )

                    except httpx.HTTPError as e:
                        logger.warning("HTTP error fetching from %s: %s", endpoint, e)
                        last_error = e
                        continue

                if endpoint_success:
                    return fills

                # If we got here without success, check if we should retry
                if last_error is not None:
                    status_code = None
                    if isinstance(last_error, httpx.HTTPStatusError):
                        status_code = last_error.response.status_code

                    if not _is_retryable_error(status_code, last_error):
                        logger.error("Non-retryable error, giving up: %s", last_error)
                        _fills_circuit_breaker.record_failure(is_auth_failure=(status_code in (401, 403)))
                        return fills

                    if attempt >= max_retries:
                        logger.error("Max retries (%d) exceeded, giving up: %s", max_retries, last_error)
                        _fills_circuit_breaker.record_failure()
                        return fills

        except (
            httpx.ConnectError,
            httpx.TimeoutException,
            httpx.ReadTimeout,
            httpx.RemoteProtocolError,
        ) as e:
            logger.warning("Network error on attempt %d: %s", attempt + 1, e)
            last_error = e

            if attempt >= max_retries:
                logger.error("Max retries (%d) exceeded for network error: %s", max_retries, e)
                _fills_circuit_breaker.record_failure()
                return fills

        except Exception as e:
            logger.exception("Unexpected error fetching fills: %s", e)
            _fills_circuit_breaker.record_failure()
            return fills

    return fills


def fetch_account_fills(
    since: datetime | None = None,
    limit: int = 100,
    config=None,
) -> list[Fill]:
    """Fetch fills from Polymarket CLOB API for authenticated account.

    This is a convenience wrapper that calls fetch_account_fills_with_retry
    with default retry settings.

    Args:
        since: Only fetch fills after this timestamp
        limit: Maximum fills to fetch
        config: Optional PolymarketConfig (loads from env if not provided)

    Returns:
        List of Fill objects from account history
    """
    return fetch_account_fills_with_retry(since=since, limit=limit, config=config)


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


def _log_heartbeat(
    iteration: int,
    fills_path: Path,
    last_heartbeat_time: float | None,
    circuit_breaker: CircuitBreaker,
) -> float:
    """Log heartbeat to distinguish silence from failure.

    Args:
        iteration: Current loop iteration
        fills_path: Path to fills file
        last_heartbeat_time: Timestamp of last heartbeat
        circuit_breaker: Circuit breaker instance to check status

    Returns:
        Current timestamp (for tracking next heartbeat)
    """
    now = time.time()

    # Only log every 30 minutes
    if last_heartbeat_time is not None:
        elapsed = now - last_heartbeat_time
        if elapsed < DEFAULT_HEARTBEAT_INTERVAL_SECONDS:
            return last_heartbeat_time

    # Get fills summary
    summary = get_fills_summary(fills_path)
    cb_status = circuit_breaker.get_status()

    # Calculate age
    age_str = "N/A"
    if summary.get("age_seconds") is not None:
        age_hours = summary["age_seconds"] / 3600
        age_str = f"{age_hours:.1f}h"

    logger.info(
        "HEARTBEAT: iteration=%d fills=%d age=%s circuit=%s failures=%d",
        iteration,
        summary.get("total_fills", 0),
        age_str,
        cb_status["state"],
        cb_status["failure_count"],
    )

    return now


def collect_fills(
    fills_path: Path | None = None,
    paper_fills_path: Path | None = None,
    include_account: bool = True,
    include_paper: bool = True,
    since: datetime | None = None,
    lookback_hours: float = 72.0,
    iteration: int = 0,
    last_heartbeat_time: float | None = None,
) -> dict:
    """Collect fills from all sources and write to fills.jsonl.

    Args:
        fills_path: Output path for fills.jsonl
        paper_fills_path: Path to paper trading fills.jsonl
        include_account: Whether to fetch real account fills
        include_paper: Whether to include paper trading fills
        since: Only collect fills after this timestamp (deprecated: use lookback_hours)
        lookback_hours: Fixed lookback window in hours (default: 72h)
        iteration: Loop iteration number (for heartbeat logging)
        last_heartbeat_time: Timestamp of last heartbeat

    Returns:
        Dict with collection results summary
    """
    if fills_path is None:
        fills_path = DEFAULT_FILLS_PATH

    # Ensure directory exists
    fills_path.parent.mkdir(parents=True, exist_ok=True)

    # Log heartbeat (distinguishes silence from failure)
    new_heartbeat_time = _log_heartbeat(
        iteration=iteration,
        fills_path=fills_path,
        last_heartbeat_time=last_heartbeat_time,
        circuit_breaker=_fills_circuit_breaker,
    )

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
        "heartbeat_logged": new_heartbeat_time != last_heartbeat_time,
        "circuit_breaker_state": _fills_circuit_breaker.state.value,
    }

    all_fills = []

    # Validate credentials before attempting to fetch account fills
    if include_account:
        config = load_config()
        if not config.has_credentials:
            msg = (
                f"COLLECT FILLS AUTH FAILED: Missing API credentials - "
                f"KEY:{'yes' if config.api_key else 'NO'}, "
                f"SECRET:{'yes' if config.api_secret else 'NO'}, "
                f"PASSPHRASE:{'yes' if config.api_passphrase else 'NO'}. "
                f"Set POLYMARKET_API_KEY, POLYMARKET_API_SECRET, and "
                f"POLYMARKET_API_PASSPHRASE environment variables."
            )
            logger.error(msg)
            raise AuthenticationError(msg)
        logger.debug("COLLECT FILLS: API credentials present")

    # Fetch account fills with retry logic
    if include_account:
        try:
            account_fills = fetch_account_fills_with_retry(since=since)
            for fill in account_fills:
                if fill.transaction_hash and fill.transaction_hash in existing_txs:
                    results["duplicates_skipped"] += 1
                    continue
                all_fills.append(fill)
            results["account_fills"] = len(account_fills)
        except AuthenticationError:
            raise  # Re-raise auth errors to fail fast
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


def startup_diagnostic() -> dict:
    """Run startup diagnostic to verify environment and credentials.

    This should be called at service startup to provide clear error messages
    if credentials are missing or API auth is failing.

    Returns:
        Dict with diagnostic results including actionable error messages.
    """
    from .config import validate_credentials_diagnostic

    logger.info("=" * 60)
    logger.info("POLYMARKET FILLS COLLECTOR - STARTUP DIAGNOSTIC")
    logger.info("=" * 60)

    results = {
        "timestamp": datetime.now(UTC).isoformat(),
        "working_directory": str(Path.cwd()),
        "credentials_ok": False,
        "api_auth_ok": False,
        "circuit_breaker": _fills_circuit_breaker.get_status(),
        "errors": [],
        "warnings": [],
        "actions_required": [],
    }

    # Step 1: Check credentials
    logger.info("STEP 1: Checking API credentials...")
    cred_diag = validate_credentials_diagnostic()
    results["credentials"] = cred_diag

    if not cred_diag["has_credentials"]:
        results["errors"].append("API credentials are missing")
        results["actions_required"].extend(cred_diag["recommendations"])
        logger.error("CREDENTIALS MISSING: No API credentials configured")
        for rec in cred_diag["recommendations"]:
            logger.error("  ACTION: %s", rec)
    else:
        results["credentials_ok"] = True
        logger.info("CREDENTIALS OK: All required credentials are present")

    # Step 2: Test API authentication
    if results["credentials_ok"]:
        logger.info("STEP 2: Testing API authentication...")
        config = load_config()
        auth_test = check_api_auth(config)
        results["api_auth_test"] = auth_test
        results["circuit_breaker"] = _fills_circuit_breaker.get_status()

        if auth_test["success"]:
            results["api_auth_ok"] = True
            logger.info("API AUTH OK: Authentication test passed")
        else:
            results["errors"].append(f"API authentication failed: {auth_test['error']}")
            results["actions_required"].append(
                "Check that credentials are correct and not expired"
            )
            results["actions_required"].append(
                "Verify API key has necessary permissions"
            )
            logger.error(
                "API AUTH FAILED: %s (status: %s, endpoint: %s)",
                auth_test["error"],
                auth_test["status_code"],
                auth_test["endpoint"],
            )
    else:
        logger.warning("STEP 2: Skipping API auth test (no credentials)")
        results["api_auth_test"] = {"skipped": True, "reason": "no_credentials"}

    # Step 3: Check data directory
    logger.info("STEP 3: Checking data directory...")
    data_dir = Path("data")
    results["data_dir"] = {
        "path": str(data_dir),
        "exists": data_dir.exists(),
        "writable": False,
    }
    if data_dir.exists():
        try:
            test_file = data_dir / ".write_test"
            test_file.write_text("test")
            test_file.unlink()
            results["data_dir"]["writable"] = True
            logger.info("DATA DIR OK: %s exists and is writable", data_dir)
        except Exception as e:
            results["warnings"].append(f"Data directory not writable: {e}")
            logger.warning("DATA DIR WARNING: %s not writable: %s", data_dir, e)
    else:
        logger.info("DATA DIR: %s does not exist (will be created)", data_dir)

    # Step 4: Check circuit breaker state
    logger.info("STEP 4: Checking circuit breaker state...")
    cb_status = _fills_circuit_breaker.get_status()
    if cb_status["state"] != "closed":
        logger.warning(
            "CIRCUIT BREAKER: State is %s with %d failures",
            cb_status["state"],
            cb_status["failure_count"],
        )
    else:
        logger.info("CIRCUIT BREAKER: CLOSED (normal operation)")

    logger.info("=" * 60)
    logger.info(
        "DIAGNOSTIC COMPLETE - Status: %s",
        "OK" if results["api_auth_ok"] else "FAILED"
    )
    logger.info("=" * 60)

    return results


def test_credentials_detailed() -> dict:
    """Test credentials with detailed output for CLI use.

    Returns:
        Dict with full diagnostic information.
    """
    import os

    result = startup_diagnostic()

    # Also return raw env var status for debugging
    result["raw_env"] = {
        "POLYMARKET_API_KEY": "set" if os.getenv("POLYMARKET_API_KEY") else "not set",
        "POLYMARKET_API_SECRET": "set" if os.getenv("POLYMARKET_API_SECRET") else "not set",
        "POLYMARKET_API_PASSPHRASE": "set" if os.getenv("POLYMARKET_API_PASSPHRASE") else "not set",
        "POLYMARKET_DRY_RUN": os.getenv("POLYMARKET_DRY_RUN", "not set"),
    }

    return result


def get_circuit_breaker_status() -> dict:
    """Get current circuit breaker status.

    Returns:
        Dict with circuit breaker status information
    """
    return _fills_circuit_breaker.get_status()


def reset_circuit_breaker() -> dict:
    """Reset the circuit breaker to closed state.

    This can be called after fixing auth issues to resume API calls.

    Returns:
        Dict with new circuit breaker status
    """
    global _fills_circuit_breaker
    old_status = _fills_circuit_breaker.get_status()

    # Create new circuit breaker in closed state
    _fills_circuit_breaker = CircuitBreaker(
        failure_threshold=DEFAULT_CB_FAILURE_THRESHOLD,
        reset_timeout_seconds=DEFAULT_CB_RESET_TIMEOUT_SECONDS,
        name="fills_api",
    )

    logger.info(
        "Circuit breaker reset: %s -> closed (was %s with %d failures)",
        old_status["state"],
        old_status["state"],
        old_status["failure_count"],
    )

    return _fills_circuit_breaker.get_status()
