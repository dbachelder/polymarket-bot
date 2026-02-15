"""Feature builder for Binance market data.

Computes returns over multiple horizons, realized volatility proxy, and signed volume metrics.
Designed for lead/lag analysis with Polymarket snapshots.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .binance_collector import AggTrade

logger = logging.getLogger(__name__)

# Default return horizons in seconds
DEFAULT_HORIZONS = [5, 30, 60, 300]  # 5s, 30s, 60s, 5m


@dataclass(frozen=True)
class Returns:
    """Returns over multiple horizons."""

    horizon_seconds: int
    simple_return: float
    log_return: float
    start_price: float
    end_price: float
    start_time: str
    end_time: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "horizon_seconds": self.horizon_seconds,
            "simple_return": self.simple_return,
            "log_return": self.log_return,
            "start_price": self.start_price,
            "end_price": self.end_price,
            "start_time": self.start_time,
            "end_time": self.end_time,
        }


@dataclass(frozen=True)
class RealizedVol:
    """Realized volatility proxy from trade-by-trade returns."""

    window_seconds: int
    annualized_vol: float
    mean_return: float
    variance: float
    trade_count: int
    start_time: str
    end_time: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_seconds": self.window_seconds,
            "annualized_vol": self.annualized_vol,
            "mean_return": self.mean_return,
            "variance": self.variance,
            "trade_count": self.trade_count,
            "start_time": self.start_time,
            "end_time": self.end_time,
        }


@dataclass(frozen=True)
class VolumeMetrics:
    """Signed volume metrics from aggregated trades."""

    window_seconds: int
    total_volume: float
    signed_volume: float
    buy_volume: float
    sell_volume: float
    buy_count: int
    sell_count: int
    vwap: float
    start_time: str
    end_time: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_seconds": self.window_seconds,
            "total_volume": self.total_volume,
            "signed_volume": self.signed_volume,
            "buy_volume": self.buy_volume,
            "sell_volume": self.sell_volume,
            "buy_count": self.buy_count,
            "sell_count": self.sell_count,
            "vwap": self.vwap,
            "start_time": self.start_time,
            "end_time": self.end_time,
        }


@dataclass
class FeatureVector:
    """Complete feature vector for a point in time."""

    timestamp: str
    timestamp_ms: int
    symbol: str
    reference_price: float
    returns: list[Returns] = field(default_factory=list)
    realized_vols: list[RealizedVol] = field(default_factory=list)
    volume_metrics: list[VolumeMetrics] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "timestamp_ms": self.timestamp_ms,
            "symbol": self.symbol,
            "reference_price": self.reference_price,
            "returns": [r.to_dict() for r in self.returns],
            "realized_vols": [v.to_dict() for v in self.realized_vols],
            "volume_metrics": [v.to_dict() for v in self.volume_metrics],
        }


class FeatureBuilder:
    """Build features from Binance market data."""

    def __init__(
        self,
        horizons: list[int] | None = None,
        vol_windows: list[int] | None = None,
    ):
        """Initialize feature builder.

        Args:
            horizons: Return horizons in seconds (default: [5, 30, 60, 300])
            vol_windows: Volatility calculation windows in seconds (default: same as horizons)
        """
        self.horizons = horizons or DEFAULT_HORIZONS
        self.vol_windows = vol_windows or self.horizons

    def _trades_to_dataframe(self, trades: list[AggTrade]) -> pd.DataFrame:
        """Convert list of trades to pandas DataFrame."""
        if not trades:
            return pd.DataFrame(
                columns=["timestamp_ms", "price", "quantity", "is_buyer_maker", "signed_volume"]
            )

        data = [
            {
                "timestamp_ms": t.timestamp_ms,
                "price": t.price,
                "quantity": t.quantity,
                "is_buyer_maker": t.is_buyer_maker,
                "signed_volume": t.signed_volume,
            }
            for t in trades
        ]
        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
        return df.sort_values("timestamp")

    def compute_returns(
        self,
        trades: list[AggTrade],
        reference_time_ms: int | None = None,
    ) -> list[Returns]:
        """Compute returns over configured horizons.

        Args:
            trades: List of aggregated trades
            reference_time_ms: Reference timestamp (default: latest trade)

        Returns:
            List of returns for each horizon
        """
        if not trades:
            return []

        df = self._trades_to_dataframe(trades)
        if df.empty:
            return []

        if reference_time_ms is None:
            reference_time_ms = int(df["timestamp_ms"].max())

        reference_price = self._get_price_at_time(df, reference_time_ms)

        results = []
        for horizon_sec in self.horizons:
            start_time_ms = reference_time_ms - (horizon_sec * 1000)
            start_price = self._get_price_at_time(df, start_time_ms)

            if start_price is None or reference_price is None or start_price <= 0:
                continue

            simple_return = (reference_price - start_price) / start_price
            log_return = pd.np.log(reference_price / start_price) if hasattr(pd, 'np') else __import__('numpy').log(reference_price / start_price)

            results.append(
                Returns(
                    horizon_seconds=horizon_sec,
                    simple_return=simple_return,
                    log_return=log_return,
                    start_price=start_price,
                    end_price=reference_price,
                    start_time=datetime.fromtimestamp(start_time_ms / 1000, tz=UTC).isoformat(),
                    end_time=datetime.fromtimestamp(reference_time_ms / 1000, tz=UTC).isoformat(),
                )
            )

        return results

    def _get_price_at_time(self, df: pd.DataFrame, timestamp_ms: int) -> float | None:
        """Get interpolated price at a specific timestamp."""
        if df.empty:
            return None

        # Find closest trade
        closest_idx = (df["timestamp_ms"] - timestamp_ms).abs().idxmin()
        return df.loc[closest_idx, "price"]

    def compute_realized_vol(
        self,
        trades: list[AggTrade],
        reference_time_ms: int | None = None,
    ) -> list[RealizedVol]:
        """Compute realized volatility proxy from trade returns.

        Uses log returns between consecutive trades, annualized.

        Args:
            trades: List of aggregated trades
            reference_time_ms: Reference timestamp (default: latest trade)

        Returns:
            List of realized volatility for each window
        """
        if len(trades) < 2:
            return []

        df = self._trades_to_dataframe(trades)
        if df.empty or len(df) < 2:
            return []

        if reference_time_ms is None:
            reference_time_ms = int(df["timestamp_ms"].max())

        # Calculate log returns between consecutive trades
        df = df.sort_values("timestamp_ms").reset_index(drop=True)
        df["log_return"] = df["price"].apply(lambda x: pd.np.log(x) if hasattr(pd, 'np') else __import__('numpy').log(x)).diff()

        results = []
        for window_sec in self.vol_windows:
            start_time_ms = reference_time_ms - (window_sec * 1000)

            mask = df["timestamp_ms"] >= start_time_ms
            window_df = df[mask]

            if len(window_df) < 2:
                continue

            returns = window_df["log_return"].dropna()
            if len(returns) < 2:
                continue

            mean_return = float(returns.mean())
            variance = float(returns.var())

            # Annualize (assuming 365 days * 24 hours * 60 minutes * 60 seconds)
            seconds_per_year = 365 * 24 * 60 * 60
            annualized_vol = (variance * (seconds_per_year / window_sec)) ** 0.5

            results.append(
                RealizedVol(
                    window_seconds=window_sec,
                    annualized_vol=annualized_vol,
                    mean_return=mean_return,
                    variance=variance,
                    trade_count=len(window_df),
                    start_time=datetime.fromtimestamp(start_time_ms / 1000, tz=UTC).isoformat(),
                    end_time=datetime.fromtimestamp(reference_time_ms / 1000, tz=UTC).isoformat(),
                )
            )

        return results

    def compute_volume_metrics(
        self,
        trades: list[AggTrade],
        reference_time_ms: int | None = None,
    ) -> list[VolumeMetrics]:
        """Compute signed volume metrics.

        Args:
            trades: List of aggregated trades
            reference_time_ms: Reference timestamp (default: latest trade)

        Returns:
            List of volume metrics for each window
        """
        if not trades:
            return []

        df = self._trades_to_dataframe(trades)
        if df.empty:
            return []

        if reference_time_ms is None:
            reference_time_ms = int(df["timestamp_ms"].max())

        results = []
        for window_sec in self.horizons:
            start_time_ms = reference_time_ms - (window_sec * 1000)

            mask = df["timestamp_ms"] >= start_time_ms
            window_df = df[mask]

            if window_df.empty:
                continue

            total_volume = float(window_df["quantity"].sum())
            signed_volume = float(window_df["signed_volume"].sum())

            buy_mask = ~window_df["is_buyer_maker"]
            sell_mask = window_df["is_buyer_maker"]

            buy_volume = float(window_df.loc[buy_mask, "quantity"].sum())
            sell_volume = float(window_df.loc[sell_mask, "quantity"].sum())
            buy_count = int(buy_mask.sum())
            sell_count = int(sell_mask.sum())

            # VWAP calculation
            notional = float((window_df["price"] * window_df["quantity"]).sum())
            vwap = notional / total_volume if total_volume > 0 else 0.0

            results.append(
                VolumeMetrics(
                    window_seconds=window_sec,
                    total_volume=total_volume,
                    signed_volume=signed_volume,
                    buy_volume=buy_volume,
                    sell_volume=sell_volume,
                    buy_count=buy_count,
                    sell_count=sell_count,
                    vwap=vwap,
                    start_time=datetime.fromtimestamp(start_time_ms / 1000, tz=UTC).isoformat(),
                    end_time=datetime.fromtimestamp(reference_time_ms / 1000, tz=UTC).isoformat(),
                )
            )

        return results

    def build_features(
        self,
        trades: list[AggTrade],
        reference_time_ms: int | None = None,
    ) -> FeatureVector:
        """Build complete feature vector from trades.

        Args:
            trades: List of aggregated trades
            reference_time_ms: Reference timestamp (default: latest trade)

        Returns:
            FeatureVector with all computed features
        """
        if reference_time_ms is None and trades:
            reference_time_ms = max(t.timestamp_ms for t in trades)

        reference_price = None
        if trades and reference_time_ms:
            df = self._trades_to_dataframe(trades)
            if not df.empty:
                reference_price = self._get_price_at_time(df, reference_time_ms)

        timestamp = (
            datetime.fromtimestamp(reference_time_ms / 1000, tz=UTC).isoformat()
            if reference_time_ms
            else datetime.now(UTC).isoformat()
        )

        # Extract symbol from trades if available
        symbol = "BTCUSDT"
        if trades:
            # Try to infer from context or use default
            pass

        return FeatureVector(
            timestamp=timestamp,
            timestamp_ms=reference_time_ms or int(datetime.now(UTC).timestamp() * 1000),
            symbol=symbol,
            reference_price=reference_price or 0.0,
            returns=self.compute_returns(trades, reference_time_ms),
            realized_vols=self.compute_realized_vol(trades, reference_time_ms),
            volume_metrics=self.compute_volume_metrics(trades, reference_time_ms),
        )


def align_to_polymarket_snapshots(
    binance_data_dir: Path,
    polymarket_data_dir: Path,
    tolerance_seconds: float = 1.0,
) -> list[dict[str, Any]]:
    """Align Binance features to Polymarket snapshot timestamps.

    Args:
        binance_data_dir: Directory containing Binance snapshot files
        polymarket_data_dir: Directory containing Polymarket snapshot files
        tolerance_seconds: Maximum time difference for alignment

    Returns:
        List of aligned (polymarket_timestamp, binance_features) records
    """
    # Load Binance snapshots
    binance_snapshots = []
    for f in binance_data_dir.glob("binance_*.json"):
        if "latest" in f.name:
            continue
        try:
            data = json.loads(f.read_text())
            binance_snapshots.append(data)
        except (json.JSONDecodeError, FileNotFoundError):
            continue

    # Load Polymarket snapshots
    pm_snapshots = []
    for f in polymarket_data_dir.glob("snapshot_*.json"):
        if "latest" in f.name:
            continue
        try:
            data = json.loads(f.read_text())
            pm_snapshots.append(data)
        except (json.JSONDecodeError, FileNotFoundError):
            continue

    # Sort by timestamp
    binance_snapshots.sort(key=lambda x: x.get("timestamp_ms", 0))
    pm_snapshots.sort(key=lambda x: x.get("generated_at", ""))

    aligned = []
    builder = FeatureBuilder()

    for pm_snap in pm_snapshots:
        pm_time_str = pm_snap.get("generated_at", "")
        if not pm_time_str:
            continue

        pm_time = datetime.fromisoformat(pm_time_str.replace("Z", "+00:00"))
        pm_time_ms = int(pm_time.timestamp() * 1000)

        # Find closest Binance snapshot
        closest = None
        min_diff = float("inf")

        for bn_snap in binance_snapshots:
            bn_time_ms = bn_snap.get("timestamp_ms", 0)
            diff = abs(bn_time_ms - pm_time_ms) / 1000.0
            if diff < min_diff:
                min_diff = diff
                closest = bn_snap

        if closest and min_diff <= tolerance_seconds:
            # Build features from Binance data
            trades = [
                AggTrade(
                    timestamp_ms=t["timestamp_ms"],
                    price=t["price"],
                    quantity=t["quantity"],
                    is_buyer_maker=t["is_buyer_maker"],
                    trade_id=t["trade_id"],
                )
                for t in closest.get("trades", [])
            ]

            features = builder.build_features(trades, pm_time_ms)

            aligned.append({
                "polymarket_timestamp": pm_time_str,
                "binance_timestamp": closest.get("timestamp"),
                "time_diff_seconds": min_diff,
                "polymarket_data": pm_snap,
                "binance_features": features.to_dict(),
            })

    return aligned


def save_aligned_features(
    aligned: list[dict[str, Any]],
    out_path: Path,
) -> None:
    """Save aligned features to JSON file."""
    out_path.write_text(json.dumps(aligned, indent=2, sort_keys=True))
    logger.info("Saved %d aligned records to %s", len(aligned), out_path)
