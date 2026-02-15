"""Dataset join: align Polymarket 15m snapshots with Binance features.

Provides lead/lag correlation analysis between BTC microstructure and
Polymarket implied probability changes.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from .binance_collector import AggTrade
from .binance_features import FeatureBuilder

logger = logging.getLogger(__name__)

# Default return horizons for lead/lag analysis (in seconds)
DEFAULT_HORIZONS = [5, 15, 30, 60, 300, 900]  # 5s, 15s, 30s, 1m, 5m, 15m


def _parse_timestamp(ts_str: str) -> datetime:
    """Parse ISO timestamp string to datetime."""
    return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))


def _load_polymarket_snapshots(
    data_dir: Path,
    hours: float | None = None,
) -> list[dict[str, Any]]:
    """Load Polymarket 15m snapshots."""
    snapshots = []
    cutoff = datetime.now(UTC) - timedelta(hours=hours) if hours else None

    for f in sorted(data_dir.glob("snapshot_15m_*.json")):
        if "latest" in f.name:
            continue
        try:
            data = json.loads(f.read_text())
            ts_str = data.get("generated_at", "")
            if not ts_str:
                continue
            ts = _parse_timestamp(ts_str)
            if cutoff and ts < cutoff:
                continue
            snapshots.append(data)
        except (json.JSONDecodeError, FileNotFoundError, ValueError):
            continue

    snapshots.sort(key=lambda x: x.get("generated_at", ""))
    return snapshots


def _load_binance_snapshots(
    data_dir: Path,
    hours: float | None = None,
) -> list[dict[str, Any]]:
    """Load Binance market data snapshots."""
    snapshots = []
    cutoff = datetime.now(UTC) - timedelta(hours=hours) if hours else None

    for f in sorted(data_dir.glob("binance_*.json")):
        if "latest" in f.name:
            continue
        try:
            data = json.loads(f.read_text())
            ts_str = data.get("timestamp", "")
            if not ts_str:
                continue
            ts = _parse_timestamp(ts_str)
            if cutoff and ts < cutoff:
                continue
            snapshots.append(data)
        except (json.JSONDecodeError, FileNotFoundError, ValueError):
            continue

    snapshots.sort(key=lambda x: x.get("timestamp_ms", 0))
    return snapshots


def _extract_btc_market_probabilities(
    pm_snapshot: dict[str, Any],
) -> dict[str, Any] | None:
    """Extract BTC-related market probabilities from Polymarket snapshot."""
    markets = pm_snapshot.get("markets", [])

    for market in markets:
        title = market.get("title", "").lower()
        if "bitcoin" not in title and "btc" not in title:
            continue

        books = market.get("books", {})
        yes_book = books.get("yes", {})
        bids = yes_book.get("bids", [])
        asks = yes_book.get("asks", [])

        # Need at least one side of the book
        if not bids and not asks:
            continue

        has_bid = len(bids) > 0
        has_ask = len(asks) > 0
        best_bid = float(bids[0]["price"]) if has_bid else 0.0
        best_ask = float(asks[0]["price"]) if has_ask else 1.0

        if has_bid and has_ask:
            mid_price = (best_bid + best_ask) / 2.0
        elif has_bid:
            mid_price = best_bid
        elif has_ask:
            mid_price = best_ask
        else:
            continue

        spread = best_ask - best_bid if best_ask > best_bid else 0.0

        return {
            "market_title": market.get("title", ""),
            "market_slug": market.get("slug", ""),
            "yes_token_id": market.get("clob_token_ids", [None, None])[0],
            "best_bid": best_bid,
            "best_ask": best_ask,
            "mid_price": mid_price,
            "spread": spread,
            "bid_size": float(bids[0]["size"]) if bids else 0.0,
            "ask_size": float(asks[0]["size"]) if asks else 0.0,
        }

    return None


@dataclass
class LeadLagCorrelation:
    """Lead/lag correlation result."""

    horizon_seconds: int
    btc_lead_corr: float | None
    btc_lag_corr: float | None
    btc_lead_pvalue: float | None
    btc_lag_pvalue: float | None
    sample_size: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "horizon_seconds": self.horizon_seconds,
            "horizon_label": self._format_horizon(),
            "btc_lead_corr": self.btc_lead_corr,
            "btc_lag_corr": self.btc_lag_corr,
            "btc_lead_pvalue": self.btc_lead_pvalue,
            "btc_lag_pvalue": self.btc_lag_pvalue,
            "sample_size": self.sample_size,
        }

    def _format_horizon(self) -> str:
        s = self.horizon_seconds
        if s < 60:
            return f"{s}s"
        elif s < 3600:
            return f"{s // 60}m"
        else:
            return f"{s // 3600}h"


@dataclass
class SanityMetrics:
    """Sanity check metrics for the aligned dataset."""

    total_pm_snapshots: int
    total_bn_snapshots: int
    aligned_pairs: int
    pm_with_btc_market: int
    missingness_pct: float
    mean_clock_drift_seconds: float
    max_clock_drift_seconds: float
    btc_market_titles: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_pm_snapshots": self.total_pm_snapshots,
            "total_bn_snapshots": self.total_bn_snapshots,
            "aligned_pairs": self.aligned_pairs,
            "pm_with_btc_market": self.pm_with_btc_market,
            "missingness_pct": round(self.missingness_pct, 2),
            "mean_clock_drift_seconds": round(self.mean_clock_drift_seconds, 3),
            "max_clock_drift_seconds": round(self.max_clock_drift_seconds, 3),
            "btc_market_titles": self.btc_market_titles,
        }


@dataclass
class JoinReport:
    """Complete dataset join report."""

    generated_at: str
    hours_analyzed: float
    correlation_results: list[LeadLagCorrelation]
    sanity_metrics: SanityMetrics

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "hours_analyzed": self.hours_analyzed,
            "correlations": [c.to_dict() for c in self.correlation_results],
            "sanity_metrics": self.sanity_metrics.to_dict(),
        }

    def to_text(self) -> str:
        """Generate human-readable text report."""
        lines = [
            "=" * 70,
            "DATASET JOIN REPORT: Polymarket 15m + Binance BTC",
            "=" * 70,
            f"Generated: {self.generated_at}",
            f"Hours analyzed: {self.hours_analyzed:.1f}",
            "",
            "--- Sanity Metrics ---",
            f"Polymarket snapshots:   {self.sanity_metrics.total_pm_snapshots}",
            f"Binance snapshots:      {self.sanity_metrics.total_bn_snapshots}",
            f"Aligned pairs:          {self.sanity_metrics.aligned_pairs}",
            f"PM with BTC market:     {self.sanity_metrics.pm_with_btc_market}",
            f"Missingness:            {self.sanity_metrics.missingness_pct:.1f}%",
            f"Mean clock drift:       {self.sanity_metrics.mean_clock_drift_seconds:.3f}s",
            f"Max clock drift:        {self.sanity_metrics.max_clock_drift_seconds:.3f}s",
        ]

        if self.sanity_metrics.btc_market_titles:
            lines.append("\nBTC Markets found:")
            for title in set(self.sanity_metrics.btc_market_titles):
                lines.append(f"  - {title}")

        lines.extend(
            [
                "",
                "--- Lead/Lag Correlations ---",
                f"{'Horizon':<10} {'BTC Lead':<12} {'PM Lead':<12} {'Samples':<10}",
                "-" * 50,
            ]
        )

        for corr in self.correlation_results:
            lead_str = f"{corr.btc_lead_corr:.3f}" if corr.btc_lead_corr is not None else "N/A"
            lag_str = f"{corr.btc_lag_corr:.3f}" if corr.btc_lag_corr is not None else "N/A"
            lines.append(
                f"{corr._format_horizon():<10} {lead_str:<12} {lag_str:<12} {corr.sample_size:<10}"
            )

        lines.extend(
            [
                "",
                "Interpretation:",
                "  BTC Lead: Correlation of BTC returns leading PM prob changes",
                "  PM Lead:  Correlation of BTC returns lagging PM prob changes",
                "  Positive = BTC up -> PM prob up (or BTC down -> PM prob down)",
                "",
                "=" * 70,
            ]
        )

        return "\n".join(lines)


def _align_snapshots(
    pm_snapshots: list[dict[str, Any]],
    bn_snapshots: list[dict[str, Any]],
    tolerance_seconds: float = 5.0,
) -> list[tuple[dict[str, Any], dict[str, Any], float]]:
    """Align Polymarket and Binance snapshots by timestamp."""
    aligned = []

    for pm_snap in pm_snapshots:
        pm_ts_str = pm_snap.get("generated_at", "")
        if not pm_ts_str:
            continue

        try:
            pm_ts = _parse_timestamp(pm_ts_str)
            pm_ms = int(pm_ts.timestamp() * 1000)
        except ValueError:
            continue

        closest = None
        min_diff = float("inf")

        for bn_snap in bn_snapshots:
            bn_ms = bn_snap.get("timestamp_ms", 0)
            if not bn_ms:
                continue
            diff = abs(bn_ms - pm_ms) / 1000.0
            if diff < min_diff:
                min_diff = diff
                closest = bn_snap

        if closest and min_diff <= tolerance_seconds:
            aligned.append((pm_snap, closest, min_diff))

    return aligned


def compute_lead_lag_correlations(
    btc_returns: dict[int, list[float | None]],
    pm_changes: dict[int, list[float | None]],
    horizons: list[int],
) -> list[LeadLagCorrelation]:
    """Compute lead/lag correlations for each horizon."""
    results = []

    for h in horizons:
        btc_rets = btc_returns.get(h, [])
        pm_chgs = pm_changes.get(h, [])

        if len(btc_rets) != len(pm_chgs):
            continue

        valid_pairs = [(b, p) for b, p in zip(btc_rets, pm_chgs) if b is not None and p is not None]

        if len(valid_pairs) < 10:
            results.append(
                LeadLagCorrelation(
                    horizon_seconds=h,
                    btc_lead_corr=None,
                    btc_lag_corr=None,
                    btc_lead_pvalue=None,
                    btc_lag_pvalue=None,
                    sample_size=len(valid_pairs),
                )
            )
            continue

        btc_arr = np.array([p[0] for p in valid_pairs])
        pm_arr = np.array([p[1] for p in valid_pairs])

        try:
            if len(btc_arr) > 2:
                btc_lead_corr = np.corrcoef(btc_arr[:-1], pm_arr[1:])[0, 1]
                btc_lag_corr = np.corrcoef(btc_arr[1:], pm_arr[:-1])[0, 1]
                btc_lead_corr = None if np.isnan(btc_lead_corr) else float(btc_lead_corr)
                btc_lag_corr = None if np.isnan(btc_lag_corr) else float(btc_lag_corr)
            else:
                btc_lead_corr = btc_lag_corr = None

            results.append(
                LeadLagCorrelation(
                    horizon_seconds=h,
                    btc_lead_corr=btc_lead_corr,
                    btc_lag_corr=btc_lag_corr,
                    btc_lead_pvalue=None,
                    btc_lag_pvalue=None,
                    sample_size=len(valid_pairs),
                )
            )
        except Exception:
            results.append(
                LeadLagCorrelation(
                    horizon_seconds=h,
                    btc_lead_corr=None,
                    btc_lag_corr=None,
                    btc_lead_pvalue=None,
                    btc_lag_pvalue=None,
                    sample_size=len(valid_pairs),
                )
            )

    return results


def build_aligned_dataset(
    polymarket_data_dir: Path,
    binance_data_dir: Path,
    hours: float = 24.0,
    tolerance_seconds: float = 5.0,
    horizons: list[int] | None = None,
) -> JoinReport:
    """Build aligned dataset and compute lead/lag correlations."""
    horizons = horizons or DEFAULT_HORIZONS

    pm_snapshots = _load_polymarket_snapshots(polymarket_data_dir, hours)
    bn_snapshots = _load_binance_snapshots(binance_data_dir, hours)

    logger.info(
        "Loaded %d Polymarket and %d Binance snapshots", len(pm_snapshots), len(bn_snapshots)
    )

    pm_with_btc = sum(1 for s in pm_snapshots if _extract_btc_market_probabilities(s) is not None)

    aligned = _align_snapshots(pm_snapshots, bn_snapshots, tolerance_seconds)

    drifts = [d for _, _, d in aligned]
    mean_drift = sum(drifts) / len(drifts) if drifts else 0.0
    max_drift = max(drifts) if drifts else 0.0

    btc_titles = []
    for pm_snap, _, _ in aligned:
        prob_data = _extract_btc_market_probabilities(pm_snap)
        if prob_data:
            btc_titles.append(prob_data["market_title"])

    pm_timestamps = []
    pm_probs = []
    bn_returns = {h: [] for h in horizons}

    for pm_snap, bn_snap, _ in aligned:
        prob_data = _extract_btc_market_probabilities(pm_snap)
        if prob_data is None:
            continue

        ts_str = pm_snap.get("generated_at", "")
        try:
            ts = _parse_timestamp(ts_str)
            pm_timestamps.append(ts)
            pm_probs.append(prob_data["mid_price"])
        except ValueError:
            continue

        trades = [
            AggTrade(
                timestamp_ms=t["timestamp_ms"],
                price=t["price"],
                quantity=t["quantity"],
                is_buyer_maker=t["is_buyer_maker"],
                trade_id=t.get("trade_id", 0),
            )
            for t in bn_snap.get("trades", [])
        ]

        ts_ms = int(ts.timestamp() * 1000)
        builder = FeatureBuilder(horizons=horizons)
        features = builder.build_features(trades, ts_ms)

        horizon_returns = {r.horizon_seconds: r.log_return for r in features.returns}
        for h in horizons:
            bn_returns[h].append(horizon_returns.get(h))

    pm_changes = {h: [] for h in horizons}
    for i, ts in enumerate(pm_timestamps):
        for h in horizons:
            target_time = ts - timedelta(seconds=h)
            prob_h_ago = None

            for j in range(i - 1, -1, -1):
                if pm_timestamps[j] <= target_time:
                    prob_h_ago = pm_probs[j]
                    break

            if prob_h_ago is not None and prob_h_ago > 0:
                pm_changes[h].append(np.log(pm_probs[i] / prob_h_ago))
            else:
                pm_changes[h].append(None)

    correlations = compute_lead_lag_correlations(bn_returns, pm_changes, horizons)

    total_points = len(pm_timestamps) * len(horizons)
    missing_points = sum(1 for h in horizons for r in bn_returns[h] if r is None)
    missingness_pct = (missing_points / total_points * 100) if total_points > 0 else 0.0

    sanity = SanityMetrics(
        total_pm_snapshots=len(pm_snapshots),
        total_bn_snapshots=len(bn_snapshots),
        aligned_pairs=len(aligned),
        pm_with_btc_market=pm_with_btc,
        missingness_pct=missingness_pct,
        mean_clock_drift_seconds=mean_drift,
        max_clock_drift_seconds=max_drift,
        btc_market_titles=btc_titles[:5],
    )

    return JoinReport(
        generated_at=datetime.now(UTC).isoformat(),
        hours_analyzed=hours,
        correlation_results=correlations,
        sanity_metrics=sanity,
    )


def save_report(
    report: JoinReport,
    out_path: Path,
    text_path: Path | None = None,
) -> tuple[Path, Path | None]:
    """Save report to JSON and optional text file."""
    out_path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    logger.info("Saved JSON report to %s", out_path)

    text_out = None
    if text_path:
        text_path.write_text(report.to_text())
        logger.info("Saved text report to %s", text_path)
        text_out = text_path

    return out_path, text_out
