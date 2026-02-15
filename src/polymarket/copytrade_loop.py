"""Copytrade accounting loop for Polymarket.

Collects fills from Polymarket API and runs daily PnL verification.
Designed for tracking copy-traded positions from other wallets.

Features:
- Fills collection from Polymarket Gamma API
- Persistent fill journal (append-only JSONL)
- Daily PnL verification with cash tracking
- Configurable collection intervals
"""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import httpx

from .pnl import Fill, PnLVerifier, load_orderbooks_from_snapshot, save_daily_summary

logger = logging.getLogger(__name__)

# Constants
DEFAULT_DATA_DIR = Path("data/copytrade")
DEFAULT_FILLS_FILE = "fills.jsonl"
DEFAULT_DAILY_PNL_DIR = "data/copytrade/pnl"
GAMMA_API_BASE = "https://gamma-api.polymarket.com"


@dataclass
class CopytradeConfig:
    """Configuration for copytrade accounting."""

    wallet_address: str | None = None
    data_dir: Path = field(default_factory=lambda: Path(DEFAULT_DATA_DIR))
    daily_pnl_dir: Path = field(default_factory=lambda: Path(DEFAULT_DAILY_PNL_DIR))
    starting_cash: Decimal = field(default_factory=lambda: Decimal("10000"))

    def __post_init__(self) -> None:
        """Ensure paths are Path objects."""
        self.data_dir = Path(self.data_dir)
        self.daily_pnl_dir = Path(self.daily_pnl_dir)


@dataclass
class FillsCollectionResult:
    """Result of a fills collection run."""

    new_fills: int = 0
    total_fills: int = 0
    last_fill_timestamp: str | None = None
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "new_fills": self.new_fills,
            "total_fills": self.total_fills,
            "last_fill_timestamp": self.last_fill_timestamp,
            "errors": self.errors,
            "collected_at": datetime.now(UTC).isoformat(),
        }


def _gamma_client(timeout: float = 30.0) -> httpx.Client:
    """Create an HTTP client for Gamma API."""
    return httpx.Client(
        base_url=GAMMA_API_BASE,
        timeout=timeout,
        headers={"User-Agent": "polymarket-bot/0.1"},
    )


def fetch_fills_for_wallet(
    wallet_address: str,
    since: datetime | None = None,
    limit: int = 100,
) -> list[Fill]:
    """Fetch fills/trades for a specific wallet from Polymarket Gamma API.

    Args:
        wallet_address: The wallet address to fetch fills for
        since: Only fetch fills after this timestamp
        limit: Maximum number of fills to fetch

    Returns:
        List of Fill objects
    """
    fills = []

    try:
        with _gamma_client() as client:
            # Build query params
            params: dict[str, str | int] = {
                "user": wallet_address,
                "limit": limit,
            }
            if since:
                params["startDate"] = since.isoformat()

            # Fetch trades from Gamma API
            response = client.get("/trades", params=params)
            response.raise_for_status()
            data = response.json()

            # Handle different response formats
            trades = data if isinstance(data, list) else data.get("trades", [])

            for trade in trades:
                try:
                    fill = _parse_gamma_trade(trade)
                    if fill:
                        fills.append(fill)
                except (ValueError, KeyError) as e:
                    logger.warning("Failed to parse trade: %s", e)
                    continue

    except httpx.HTTPStatusError as e:
        logger.error("HTTP error fetching fills: %s", e)
        raise
    except Exception as e:
        logger.error("Error fetching fills: %s", e)
        raise

    return fills


def _parse_gamma_trade(trade: dict) -> Fill | None:
    """Parse a Gamma API trade into a Fill object.

    Args:
        trade: Trade dict from Gamma API

    Returns:
        Fill object or None if parsing fails
    """
    # Gamma API trade format
    token_id = trade.get("asset_id") or trade.get("token_id") or ""
    if not token_id:
        return None

    side = (trade.get("side") or trade.get("trade_side") or "buy").lower()

    # Size might be in different fields
    size_val = trade.get("size") or trade.get("amount") or trade.get("takerAmount") or "0"

    # Price might be in different fields
    price_val = (
        trade.get("price")
        or trade.get("execution_price")
        or trade.get("priceInNative")
        or "0"
    )

    # Fee
    fee_val = trade.get("fee") or trade.get("trade_fee") or trade.get("gas_fee") or "0"

    # Timestamp
    timestamp = trade.get("timestamp") or trade.get("created_at") or trade.get("transaction_time")
    if not timestamp:
        timestamp = datetime.now(UTC).isoformat()

    # Transaction hash
    tx_hash = trade.get("transaction_hash") or trade.get("tx_hash") or trade.get("transactionHash")

    # Market info
    market_slug = trade.get("market_slug") or trade.get("slug") or trade.get("market")
    if not market_slug:
        # Try to extract from condition_id or market id
        market_slug = trade.get("condition_id") or trade.get("market_id", "unknown")

    return Fill(
        token_id=str(token_id),
        side=side,
        size=Decimal(str(size_val)),
        price=Decimal(str(price_val)),
        fee=Decimal(str(fee_val)),
        timestamp=str(timestamp),
        transaction_hash=tx_hash,
        market_slug=market_slug,
    )


def load_existing_fills(fills_path: Path) -> list[Fill]:
    """Load existing fills from journal file.

    Args:
        fills_path: Path to fills JSONL file

    Returns:
        List of existing Fill objects
    """
    if not fills_path.exists():
        return []

    fills = []
    with open(fills_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                fill = Fill.from_dict(data)
                fills.append(fill)
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning("Failed to parse fill line: %s", e)
                continue

    return fills


def save_fill(fills_path: Path, fill: Fill) -> None:
    """Append a fill to the journal file.

    Args:
        fills_path: Path to fills JSONL file
        fill: Fill to save
    """
    fills_path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "token_id": fill.token_id,
        "side": fill.side,
        "size": str(fill.size),
        "price": str(fill.price),
        "fee": str(fill.fee),
        "timestamp": fill.timestamp,
        "market_slug": fill.market_slug,
        "transaction_hash": fill.transaction_hash,
    }

    with open(fills_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def deduplicate_fills(existing: list[Fill], new: list[Fill]) -> list[Fill]:
    """Deduplicate fills based on transaction hash and token_id.

    Args:
        existing: List of existing fills
        new: List of new fills to check

    Returns:
        List of truly new fills
    """
    # Build set of existing identifiers
    existing_ids = set()
    for fill in existing:
        key = (fill.transaction_hash, fill.token_id, str(fill.size), str(fill.price))
        existing_ids.add(key)

    truly_new = []
    for fill in new:
        key = (fill.transaction_hash, fill.token_id, str(fill.size), str(fill.price))
        if key not in existing_ids:
            truly_new.append(fill)
            existing_ids.add(key)

    return truly_new


def collect_fills(
    config: CopytradeConfig,
    since: datetime | None = None,
) -> FillsCollectionResult:
    """Collect fills for a wallet and save to journal.

    Args:
        config: Copytrade configuration
        since: Only fetch fills after this timestamp

    Returns:
        FillsCollectionResult with collection stats
    """
    result = FillsCollectionResult()

    if not config.wallet_address:
        result.errors.append("No wallet address configured")
        return result

    fills_path = config.data_dir / DEFAULT_FILLS_FILE

    try:
        # Fetch new fills from API
        new_fills = fetch_fills_for_wallet(config.wallet_address, since=since)

        # Load existing fills
        existing_fills = load_existing_fills(fills_path)

        # Deduplicate
        truly_new = deduplicate_fills(existing_fills, new_fills)

        # Save new fills
        for fill in truly_new:
            save_fill(fills_path, fill)

        result.new_fills = len(truly_new)
        result.total_fills = len(existing_fills) + len(truly_new)

        if truly_new:
            # Get timestamp of most recent fill
            latest = max(truly_new, key=lambda f: f.datetime_utc)
            result.last_fill_timestamp = latest.timestamp

        logger.info(
            "Collected %d new fills (total: %d) for wallet %s",
            result.new_fills,
            result.total_fills,
            config.wallet_address,
        )

    except Exception as e:
        error_msg = f"Failed to collect fills: {e}"
        logger.exception(error_msg)
        result.errors.append(error_msg)

    return result


def run_pnl_verification(
    config: CopytradeConfig,
    snapshot_path: Path | None = None,
) -> Path | None:
    """Run PnL verification and save daily summary.

    Args:
        config: Copytrade configuration
        snapshot_path: Optional path to collector snapshot for prices

    Returns:
        Path to saved daily summary or None if failed
    """
    fills_path = config.data_dir / DEFAULT_FILLS_FILE

    if not fills_path.exists():
        logger.warning("No fills file found at %s", fills_path)
        return None

    try:
        # Load all fills
        fills = load_existing_fills(fills_path)

        if not fills:
            logger.warning("No fills to verify")
            return None

        # Build verifier
        verifier = PnLVerifier(starting_cash=config.starting_cash)
        verifier.add_fills(fills)

        # Load orderbooks from snapshot if provided
        orderbooks = None
        if snapshot_path and snapshot_path.exists():
            orderbooks = load_orderbooks_from_snapshot(snapshot_path)

        # Compute PnL report
        report = verifier.compute_pnl(orderbooks=orderbooks)

        # Save daily summary
        summary_path = save_daily_summary(
            report,
            out_dir=config.daily_pnl_dir,
            date=datetime.now(UTC),
        )

        logger.info(
            "PnL verification complete: realized=%.2f, unrealized=%.2f, net=%.2f",
            float(report.realized_pnl),
            float(report.unrealized_pnl),
            float(report.net_pnl),
        )

        return summary_path

    except Exception as e:
        logger.exception("PnL verification failed: %s", e)
        return None


def copytrade_loop(
    config: CopytradeConfig,
    fill_collection_interval_seconds: float = 300.0,  # 5 minutes default
    pnl_verification_time_utc: str = "00:00",  # Daily at midnight UTC
    snapshot_dir: Path | None = None,
    max_backoff_seconds: float = 300.0,
) -> None:
    """Run the copytrade accounting loop.

    Continuously collects fills and runs daily PnL verification.

    Args:
        config: Copytrade configuration
        fill_collection_interval_seconds: Seconds between fill collections
        pnl_verification_time_utc: Time to run daily PnL (HH:MM format)
        snapshot_dir: Directory to find latest snapshot for prices
        max_backoff_seconds: Max backoff on errors
    """
    config.data_dir.mkdir(parents=True, exist_ok=True)
    config.daily_pnl_dir.mkdir(parents=True, exist_ok=True)

    base_interval = max(60.0, float(fill_collection_interval_seconds))
    backoff = base_interval
    last_pnl_date: datetime | None = None
    last_fill_collection = 0.0

    logger.info(
        "Starting copytrade loop for wallet %s (interval: %.0fs, PnL at %s UTC)",
        config.wallet_address or "(none)",
        base_interval,
        pnl_verification_time_utc,
    )

    while True:
        started = time.time()

        try:
            # Collect fills if enough time has passed
            if started - last_fill_collection >= base_interval:
                # Calculate "since" from last known fill
                fills_path = config.data_dir / DEFAULT_FILLS_FILE
                existing = load_existing_fills(fills_path)
                since = None
                if existing:
                    latest = max(existing, key=lambda f: f.datetime_utc)
                    since = latest.datetime_utc

                result = collect_fills(config, since=since)
                last_fill_collection = started

                if result.errors:
                    logger.warning("Fill collection had errors: %s", result.errors)
                else:
                    # Reset backoff on success
                    backoff = base_interval

            # Check if it's time for daily PnL verification
            current_time = datetime.now(UTC)
            pnl_hour, pnl_minute = map(int, pnl_verification_time_utc.split(":"))

            should_run_pnl = False
            if last_pnl_date is None:
                should_run_pnl = (
                    current_time.hour == pnl_hour and current_time.minute >= pnl_minute
                )
            elif current_time.date() > last_pnl_date.date():
                should_run_pnl = (
                    current_time.hour == pnl_hour and current_time.minute >= pnl_minute
                )

            if should_run_pnl:
                # Find latest snapshot if snapshot_dir provided
                snapshot_path = None
                if snapshot_dir:
                    snapshots = sorted(snapshot_dir.glob("snapshot_*.json"))
                    if snapshots:
                        snapshot_path = snapshots[-1]

                summary_path = run_pnl_verification(config, snapshot_path)
                if summary_path:
                    last_pnl_date = current_time
                    logger.info("Daily PnL saved to %s", summary_path)

        except (
            httpx.HTTPStatusError,
            httpx.ConnectError,
            httpx.ReadTimeout,
            httpx.RemoteProtocolError,
        ) as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            is_retryable = status in (429, 500, 502, 503, 504) or status is None
            if not is_retryable:
                raise

            backoff = min(max_backoff_seconds, max(backoff * 2, base_interval))
            logger.warning("Transient error, backing off to %.0fs: %s", backoff, e)

        except Exception as e:
            logger.exception("Error in copytrade loop: %s", e)
            backoff = min(max_backoff_seconds, max(backoff * 2, base_interval))

        # Sleep remaining time
        elapsed = time.time() - started
        sleep_for = max(0.0, backoff - elapsed)
        sleep_for += random.uniform(0.0, min(0.25 * base_interval, 10.0))
        time.sleep(sleep_for)


def run_single_pnl_verify(
    data_dir: Path,
    output_dir: Path | None = None,
    snapshot_path: Path | None = None,
    starting_cash: Decimal | None = None,
) -> Path | None:
    """Run a single PnL verification (non-loop mode).

    Args:
        data_dir: Directory containing fills.jsonl
        output_dir: Directory for PnL output (default: data_dir/pnl)
        snapshot_path: Optional path to snapshot for prices
        starting_cash: Starting cash balance

    Returns:
        Path to saved summary or None
    """
    config = CopytradeConfig(
        data_dir=data_dir,
        daily_pnl_dir=output_dir or data_dir / "pnl",
        starting_cash=starting_cash or Decimal("0"),
    )

    return run_pnl_verification(config, snapshot_path)
