"""Tests for dataset join module."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from polymarket.dataset_join import (
    JoinReport,
    LeadLagCorrelation,
    SanityMetrics,
    _align_snapshots,
    _extract_btc_market_probabilities,
    _load_binance_snapshots,
    _load_polymarket_snapshots,
    _parse_timestamp,
    build_aligned_dataset,
    compute_lead_lag_correlations,
    save_report,
)


class TestParseTimestamp:
    """Test timestamp parsing."""

    def test_parse_iso_with_z(self) -> None:
        """Parse ISO timestamp with Z suffix."""
        ts = _parse_timestamp("2026-02-15T12:00:00Z")
        assert ts.year == 2026
        assert ts.month == 2
        assert ts.day == 15
        assert ts.hour == 12
        assert ts.minute == 0

    def test_parse_iso_with_offset(self) -> None:
        """Parse ISO timestamp with timezone offset."""
        ts = _parse_timestamp("2026-02-15T12:00:00+00:00")
        assert ts.hour == 12


class TestExtractBtcMarketProbabilities:
    """Test BTC market extraction from Polymarket snapshot."""

    def test_extracts_btc_market(self) -> None:
        """Extract BTC market data from snapshot."""
        snapshot = {
            "markets": [
                {
                    "title": "Bitcoin above $100k at 3pm?",
                    "slug": "bitcoin-above-100k",
                    "clob_token_ids": ["token_yes", "token_no"],
                    "books": {
                        "yes": {
                            "bids": [{"price": "0.65", "size": "100"}],
                            "asks": [{"price": "0.67", "size": "50"}],
                        }
                    },
                }
            ]
        }
        result = _extract_btc_market_probabilities(snapshot)
        assert result is not None
        assert result["market_title"] == "Bitcoin above $100k at 3pm?"
        assert result["mid_price"] == 0.66  # (0.65 + 0.67) / 2
        assert result["spread"] == pytest.approx(0.02, abs=1e-10)

    def test_skips_non_btc_markets(self) -> None:
        """Skip markets without BTC in title."""
        snapshot = {
            "markets": [
                {
                    "title": "Ethereum price prediction",
                    "books": {"yes": {"bids": [], "asks": []}},
                }
            ]
        }
        result = _extract_btc_market_probabilities(snapshot)
        assert result is None

    def test_handles_missing_books(self) -> None:
        """Handle markets with missing order books."""
        snapshot = {
            "markets": [
                {
                    "title": "Bitcoin prediction",
                    "books": {"yes": {"bids": [], "asks": []}},
                }
            ]
        }
        result = _extract_btc_market_probabilities(snapshot)
        assert result is None


class TestAlignSnapshots:
    """Test snapshot alignment by timestamp."""

    def test_aligns_by_closest_timestamp(self) -> None:
        """Align PM and Binance snapshots by closest timestamp."""
        base_ms = int(datetime(2026, 2, 15, 12, 0, 0, tzinfo=UTC).timestamp() * 1000)
        pm_snaps = [
            {"generated_at": "2026-02-15T12:00:00Z"},
            {"generated_at": "2026-02-15T12:01:00Z"},
        ]
        bn_snaps = [
            {"timestamp_ms": base_ms},  # 12:00:00
            {"timestamp_ms": base_ms + 60000},  # 12:01:00
        ]
        aligned = _align_snapshots(pm_snaps, bn_snaps, tolerance_seconds=5.0)
        assert len(aligned) == 2
        assert aligned[0][2] <= 5.0  # drift within tolerance

    def test_exceeds_tolerance(self) -> None:
        """Exclude pairs exceeding tolerance."""
        pm_snaps = [{"generated_at": "2026-02-15T12:00:00Z"}]
        bn_snaps = [{"timestamp_ms": 1_709_600_500_000}]  # 12:01:40 (100s diff)
        aligned = _align_snapshots(pm_snaps, bn_snaps, tolerance_seconds=5.0)
        assert len(aligned) == 0

    def test_empty_inputs(self) -> None:
        """Handle empty input lists."""
        aligned = _align_snapshots([], [], tolerance_seconds=5.0)
        assert aligned == []


class TestComputeLeadLagCorrelations:
    """Test lead/lag correlation computation."""

    def test_computes_correlations(self) -> None:
        """Compute correlations for valid data."""
        # Need at least 10 samples for correlations to be computed
        btc_returns = {
            60: [0.01, 0.02, -0.01, 0.005, 0.015, -0.005, 0.008, 0.012, -0.003, 0.007],
            300: [0.05, 0.03, -0.02, 0.01, 0.04, -0.01, 0.02, 0.015, -0.005, 0.025],
        }
        pm_changes = {
            60: [0.005, 0.015, -0.005, 0.002, 0.01, -0.003, 0.006, 0.009, -0.002, 0.005],
            300: [0.03, 0.02, -0.01, 0.005, 0.025, -0.008, 0.015, 0.012, -0.003, 0.018],
        }
        results = compute_lead_lag_correlations(btc_returns, pm_changes, [60, 300])
        assert len(results) == 2
        assert all(r.btc_lead_corr is not None for r in results)
        assert all(r.btc_lag_corr is not None for r in results)

    def test_insufficient_samples(self) -> None:
        """Handle insufficient sample size."""
        btc_returns = {60: [0.01, 0.02]}
        pm_changes = {60: [0.005, 0.015]}
        results = compute_lead_lag_correlations(btc_returns, pm_changes, [60])
        assert len(results) == 1
        assert results[0].btc_lead_corr is None  # Need at least 10 samples

    def test_with_nones(self) -> None:
        """Handle data with None values."""
        btc_returns = {60: [0.01, None, 0.02, 0.03]}
        pm_changes = {60: [0.005, 0.015, None, 0.02]}
        results = compute_lead_lag_correlations(btc_returns, pm_changes, [60])
        assert len(results) == 1
        assert results[0].sample_size == 2  # Only valid pairs


class TestLeadLagCorrelation:
    """Test LeadLagCorrelation dataclass."""

    def test_to_dict(self) -> None:
        """Convert to dictionary."""
        corr = LeadLagCorrelation(
            horizon_seconds=60,
            btc_lead_corr=0.5,
            btc_lag_corr=0.3,
            btc_lead_pvalue=None,
            btc_lag_pvalue=None,
            sample_size=100,
        )
        d = corr.to_dict()
        assert d["horizon_seconds"] == 60
        assert d["horizon_label"] == "1m"
        assert d["btc_lead_corr"] == 0.5

    def test_format_horizon(self) -> None:
        """Format horizon for display."""
        assert LeadLagCorrelation(30, None, None, None, None, 0)._format_horizon() == "30s"
        assert LeadLagCorrelation(60, None, None, None, None, 0)._format_horizon() == "1m"
        assert LeadLagCorrelation(3600, None, None, None, None, 0)._format_horizon() == "1h"


class TestSanityMetrics:
    """Test SanityMetrics dataclass."""

    def test_to_dict(self) -> None:
        """Convert to dictionary."""
        metrics = SanityMetrics(
            total_pm_snapshots=100,
            total_bn_snapshots=95,
            aligned_pairs=90,
            pm_with_btc_market=80,
            missingness_pct=5.5,
            mean_clock_drift_seconds=2.5,
            max_clock_drift_seconds=5.0,
            btc_market_titles=["BTC Market 1"],
        )
        d = metrics.to_dict()
        assert d["total_pm_snapshots"] == 100
        assert d["missingness_pct"] == 5.5


class TestJoinReport:
    """Test JoinReport dataclass."""

    def test_to_dict(self) -> None:
        """Convert to dictionary."""
        report = JoinReport(
            generated_at="2026-02-15T12:00:00Z",
            hours_analyzed=24.0,
            correlation_results=[],
            sanity_metrics=SanityMetrics(0, 0, 0, 0, 0.0, 0.0, 0.0),
        )
        d = report.to_dict()
        assert d["hours_analyzed"] == 24.0
        assert "sanity_metrics" in d

    def test_to_text(self) -> None:
        """Generate human-readable text."""
        corr = LeadLagCorrelation(
            horizon_seconds=60,
            btc_lead_corr=0.5,
            btc_lag_corr=0.3,
            btc_lead_pvalue=None,
            btc_lag_pvalue=None,
            sample_size=100,
        )
        report = JoinReport(
            generated_at="2026-02-15T12:00:00Z",
            hours_analyzed=24.0,
            correlation_results=[corr],
            sanity_metrics=SanityMetrics(
                total_pm_snapshots=100,
                total_bn_snapshots=95,
                aligned_pairs=90,
                pm_with_btc_market=80,
                missingness_pct=5.0,
                mean_clock_drift_seconds=2.5,
                max_clock_drift_seconds=5.0,
                btc_market_titles=["BTC Market"],
            ),
        )
        text = report.to_text()
        assert "Polymarket 15m + Binance BTC" in text
        assert "100" in text  # PM snapshots
        assert "1m" in text  # Horizon label


class TestSaveReport:
    """Test report saving."""

    def test_saves_json(self, tmp_path: Path) -> None:
        """Save JSON report."""
        report = JoinReport(
            generated_at="2026-02-15T12:00:00Z",
            hours_analyzed=24.0,
            correlation_results=[],
            sanity_metrics=SanityMetrics(0, 0, 0, 0, 0.0, 0.0, 0.0),
        )
        out_path = tmp_path / "report.json"
        save_report(report, out_path)
        assert out_path.exists()
        data = json.loads(out_path.read_text())
        assert data["hours_analyzed"] == 24.0

    def test_saves_text(self, tmp_path: Path) -> None:
        """Save both JSON and text reports."""
        report = JoinReport(
            generated_at="2026-02-15T12:00:00Z",
            hours_analyzed=24.0,
            correlation_results=[],
            sanity_metrics=SanityMetrics(0, 0, 0, 0, 0.0, 0.0, 0.0),
        )
        json_path = tmp_path / "report.json"
        text_path = tmp_path / "report.txt"
        save_report(report, json_path, text_path)
        assert json_path.exists()
        assert text_path.exists()
        assert "Polymarket" in text_path.read_text()


class TestLoadPolymarketSnapshots:
    """Test loading Polymarket snapshots."""

    def test_loads_and_filters_by_hours(self, tmp_path: Path) -> None:
        """Load snapshots filtered by hours."""
        now = datetime.now(UTC)
        old_time = now - timedelta(hours=25)
        recent_time = now - timedelta(hours=1)

        old_snap = {
            "generated_at": old_time.isoformat(),
            "markets": [],
        }
        recent_snap = {
            "generated_at": recent_time.isoformat(),
            "markets": [],
        }

        (tmp_path / "snapshot_15m_old.json").write_text(json.dumps(old_snap))
        (tmp_path / "snapshot_15m_recent.json").write_text(json.dumps(recent_snap))

        results = _load_polymarket_snapshots(tmp_path, hours=24)
        assert len(results) == 1
        assert results[0]["generated_at"] == recent_time.isoformat()

    def test_skips_latest_files(self, tmp_path: Path) -> None:
        """Skip files with 'latest' in name."""
        (tmp_path / "snapshot_15m_latest.json").write_text(
            json.dumps({"generated_at": datetime.now(UTC).isoformat()})
        )
        results = _load_polymarket_snapshots(tmp_path)
        assert len(results) == 0


class TestLoadBinanceSnapshots:
    """Test loading Binance snapshots."""

    def test_loads_snapshots(self, tmp_path: Path) -> None:
        """Load Binance snapshots."""
        snap = {
            "timestamp": datetime.now(UTC).isoformat(),
            "timestamp_ms": int(datetime.now(UTC).timestamp() * 1000),
            "trades": [],
        }
        (tmp_path / "binance_btcusdt_20260215T120000Z.json").write_text(json.dumps(snap))
        results = _load_binance_snapshots(tmp_path)
        assert len(results) == 1


class TestBuildAlignedDataset:
    """Test full dataset building."""

    def test_builds_with_no_data(self, tmp_path: Path) -> None:
        """Handle empty data directories."""
        pm_dir = tmp_path / "pm"
        bn_dir = tmp_path / "bn"
        pm_dir.mkdir()
        bn_dir.mkdir()

        report = build_aligned_dataset(pm_dir, bn_dir, hours=24.0)
        assert report.hours_analyzed == 24.0
        assert report.sanity_metrics.total_pm_snapshots == 0
        assert report.sanity_metrics.total_bn_snapshots == 0

    def test_no_lookahead(self, tmp_path: Path) -> None:
        """Ensure no lookahead bias in feature computation."""
        now = datetime.now(UTC)
        pm_time = now - timedelta(hours=1)

        pm_snap = {
            "generated_at": pm_time.isoformat(),
            "markets": [
                {
                    "title": "Bitcoin prediction",
                    "slug": "btc-pred",
                    "clob_token_ids": ["yes", "no"],
                    "books": {
                        "yes": {
                            "bids": [{"price": "0.6", "size": "100"}],
                            "asks": [{"price": "0.62", "size": "100"}],
                        }
                    },
                }
            ],
        }

        bn_snap = {
            "timestamp": pm_time.isoformat(),
            "timestamp_ms": int(pm_time.timestamp() * 1000),
            "trades": [
                {
                    "timestamp_ms": int((pm_time - timedelta(minutes=5)).timestamp() * 1000),
                    "price": 50000.0,
                    "quantity": 1.0,
                    "is_buyer_maker": False,
                    "trade_id": 1,
                }
            ],
        }

        pm_dir = tmp_path / "pm"
        bn_dir = tmp_path / "bn"
        pm_dir.mkdir()
        bn_dir.mkdir()

        (pm_dir / "snapshot_15m_test.json").write_text(json.dumps(pm_snap))
        (bn_dir / "binance_btcusdt_test.json").write_text(json.dumps(bn_snap))

        report = build_aligned_dataset(pm_dir, bn_dir, hours=24.0)
        assert report.sanity_metrics.aligned_pairs == 1
        assert report.sanity_metrics.pm_with_btc_market == 1
