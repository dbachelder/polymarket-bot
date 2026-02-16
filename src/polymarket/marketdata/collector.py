"""Market data collector using the provider abstraction.

Replaces binance_collector.py with a provider-agnostic implementation.
"""

from __future__ import annotations

import csv
import json
import logging
import random
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

from polymarket.marketdata import MarketDataProvider, Snapshot
from polymarket.marketdata.auto import create_provider

logger = logging.getLogger(__name__)

DEFAULT_SYMBOL = "BTCUSDT"


class MarketDataCollector:
    """Collector for market data using any provider.

    Supports both REST-based collection (one-shot or loop) and
    can be extended for WebSocket collection.
    """

    def __init__(
        self,
        provider: MarketDataProvider,
        symbol: str = DEFAULT_SYMBOL,
    ):
        """Initialize collector.

        Args:
            provider: Market data provider instance
            symbol: Trading pair symbol
        """
        self.provider = provider
        self.symbol = symbol

    def collect_snapshot(
        self,
        kline_intervals: list[str] | None = None,
        trade_lookback_seconds: int = 60,
    ) -> Snapshot:
        """Collect a single snapshot.

        Args:
            kline_intervals: List of kline intervals to fetch
            trade_lookback_seconds: How far back to fetch trades

        Returns:
            Snapshot with all requested data
        """
        return self.provider.get_snapshot(
            symbol=self.symbol,
            kline_intervals=kline_intervals,
            trade_lookback_seconds=trade_lookback_seconds,
        )

    def save_snapshot(
        self,
        snapshot: Snapshot,
        out_dir: Path,
        write_csv: bool = True,
    ) -> Path:
        """Save snapshot to disk.

        Args:
            snapshot: Snapshot to save
            out_dir: Output directory
            write_csv: Also write trades to CSV

        Returns:
            Path to the saved JSON file
        """
        out_dir.mkdir(parents=True, exist_ok=True)

        ts_str = datetime.fromtimestamp(snapshot.timestamp_ms / 1000, tz=UTC).strftime(
            "%Y%m%dT%H%M%SZ"
        )

        # Write JSON snapshot
        out_path = out_dir / f"{snapshot.provider}_{self.symbol.lower()}_{ts_str}.json"
        out_path.write_text(json.dumps(snapshot.to_dict(), indent=2, sort_keys=True))

        # Write trades to CSV for easy analysis
        if write_csv and snapshot.trades:
            csv_path = out_dir / f"{snapshot.provider}_{self.symbol.lower()}_trades_{ts_str}.csv"
            with csv_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=snapshot.trades[0].to_dict().keys())
                writer.writeheader()
                for trade in snapshot.trades:
                    writer.writerow(trade.to_dict())

        # Write latest pointer
        latest = out_dir / f"latest_{self.symbol.lower()}.json"
        latest.write_text(
            json.dumps(
                {
                    "path": str(out_path),
                    "generated_at": datetime.now(UTC).isoformat(),
                    "timestamp_ms": snapshot.timestamp_ms,
                    "provider": snapshot.provider,
                }
            )
        )

        return out_path


def collect_snapshot(
    out_dir: Path,
    provider_name: str = "auto",
    symbol: str = DEFAULT_SYMBOL,
    kline_intervals: list[str] | None = None,
    binance_base_url: str | None = None,
    timeout: float = 30.0,
) -> Path:
    """Collect a single snapshot using specified provider.

    Args:
        out_dir: Directory to write output files
        provider_name: Provider name (binance, coinbase, kraken, auto)
        symbol: Trading pair symbol
        kline_intervals: Kline intervals to fetch
        binance_base_url: Optional override for Binance base URL
        timeout: Request timeout

    Returns:
        Path to the written JSON file
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    kline_intervals = kline_intervals or ["1m", "5m"]

    provider = create_provider(provider_name, binance_base_url, timeout)

    try:
        collector = MarketDataCollector(provider, symbol)
        snapshot = collector.collect_snapshot(kline_intervals=kline_intervals)
        return collector.save_snapshot(snapshot, out_dir)
    finally:
        provider.close()


def collect_loop(
    out_dir: Path,
    provider_name: str = "auto",
    symbol: str = DEFAULT_SYMBOL,
    kline_intervals: list[str] | None = None,
    snapshot_interval_seconds: float = 5.0,
    max_reconnect_delay: float = 60.0,
    retention_hours: float | None = None,
    binance_base_url: str | None = None,
    timeout: float = 30.0,
) -> None:
    """Continuously collect market data via REST polling.

    Args:
        out_dir: Directory to write output files
        provider_name: Provider name (binance, coinbase, kraken, auto)
        symbol: Trading pair symbol
        kline_intervals: Kline intervals to fetch
        snapshot_interval_seconds: How often to collect snapshots
        max_reconnect_delay: Maximum delay between retries on error
        retention_hours: If set, delete files older than this many hours
        binance_base_url: Optional override for Binance base URL
        timeout: Request timeout
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    kline_intervals = kline_intervals or ["1m", "5m"]

    reconnect_delay = 1.0

    while True:
        provider = create_provider(provider_name, binance_base_url, timeout)
        collector = MarketDataCollector(provider, symbol)

        try:
            while True:
                start_time = time.time()

                try:
                    snapshot = collector.collect_snapshot(kline_intervals=kline_intervals)
                    collector.save_snapshot(snapshot, out_dir)

                    # Reset reconnect delay on success
                    reconnect_delay = 1.0

                    # Prune old files if retention is set
                    if retention_hours is not None:
                        _prune_old_files(out_dir, retention_hours)

                except Exception as e:
                    logger.error("Error collecting snapshot: %s", e)
                    # Will retry on next iteration

                # Sleep for remaining interval time
                elapsed = time.time() - start_time
                sleep_time = max(0.1, snapshot_interval_seconds - elapsed)
                time.sleep(sleep_time)

        except KeyboardInterrupt:
            logger.info("Stopping collector")
            provider.close()
            break
        except Exception as e:
            logger.error("Collector loop error: %s", e)
        finally:
            provider.close()

        # Exponential backoff with jitter before reconnecting
        delay = reconnect_delay + random.uniform(0, 1.0)
        logger.info("Reconnecting in %.1f seconds...", delay)
        time.sleep(delay)
        reconnect_delay = min(max_reconnect_delay, reconnect_delay * 1.5)


def _prune_old_files(out_dir: Path, retention_hours: float) -> int:
    """Delete files older than retention_hours.

    Args:
        out_dir: Directory to clean
        retention_hours: Delete files older than this

    Returns:
        Number of files deleted
    """
    cutoff = datetime.now(UTC) - timedelta(hours=retention_hours)
    deleted = 0

    for p in out_dir.glob("*.json"):
        # Skip "latest" files
        if "latest" in p.name:
            continue
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
