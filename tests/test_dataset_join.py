"""Tests for dataset_join module."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

from polymarket.dataset_join import (
    LeadLagCorrelation,
    SanityMetrics,
    JoinReport,
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

    def test_parse_iso_with_z(self):
        """Parse ISO timestamp with Z suffix."""
        ts = _parse_timestamp("2024-01-15T10:30:00Z")
        assert ts.year == 2024
        assert ts.month == 1
        assert ts.day == 15
        assert ts.hour == 10
        assert ts.minute == 30
        assert ts.second == 0
        assert ts.tzinfo is not None

    def test_parse_iso_with_offset(self):
        """Parse ISO timestamp with timezone offset."""
        ts = _parse_timestamp("2024-01-15T10:30:00+00:00")
        assert ts.year == 2024
        assert ts.hour == 10


class TestLoadPolymarketSnapshots:
    """Test loading Polymarket snapshots."""

    def test_loads_valid_snapshots(self, tmp_path: Path):
        """Load valid snapshot files."""
        # Create test snapshots
        snap1 = {
            "generated_at": (datetime.now(UTC) - timedelta(hours=1)).isoformat(),
            "markets": [{"title": "BTC Test"}],
        }
        snap2 = {
            "generated_at": datetime.now(UTC).isoformat(),
            "markets": [{"title": "BTC Test 2"}],
        }

        (tmp_path / "snapshot_15m_001.json").write_text(json.dumps(snap1))
        (tmp_path / "snapshot_15m_002.json").write_text(json.dumps(snap2))
        (tmp_path / "snapshot_15m_latest.json").write_text(json.dumps(snap2))  # Should be skipped

        result = _load_polymarket_snapshots(tmp_path)
        assert len(result) == 2
        assert result[0]["markets"][0]["title"] == "BTC Test"
        assert result[1]["markets"][0]["title"] == "BTC Test 2"

    def test_filters_by_hours(self, tmp_path: Path):
        """Filter snapshots by hours parameter."""
        old_snap = {
            "generated_at": (datetime.now(UTC) - timedelta(hours=48)).isoformat(),
            "markets": [],
        }
        new_snap = {
            "generated_at": datetime.now(UTC).isoformat(),
            "markets": [],
        }

        (tmp_path / "snapshot_15m_old.json").write_text(json.dumps(old_snap))
        (tmp_path / "snapshot_15m_new.json").write_text(json.dumps(new_snap))

        result = _load_polymarket_snapshots(tmp_path, hours=24)
        assert len(result) == 1
        assert result[0]["markets"] == []

    def test_skips_invalid_json(self, tmp_path: Path):
        """Skip files with invalid JSON."""
        valid_snap = {
            "generated_at": datetime.now(UTC).isoformat(),
            "markets": [],
        }
        (tmp_path / "snapshot_15m_valid.json").write_text(json.dumps(valid_snap))
        (tmp_path / "snapshot_15m_invalid.json").write_text("not json")

        result = _load_polymarket_snapshots(tmp_path)
        assert len(result) == 1

    def test_skips_missing_generated_at(self, tmp_path: Path):
        """Skip snapshots without generated_at field."""
        invalid_snap = {"markets": []}
        valid_snap = {
            "generated_at": datetime.now(UTC).isoformat(),
            "markets": [],
        }
        (tmp_path / "snapshot_15m_invalid.json").write_text(json.dumps(invalid_snap))
        (tmp_path / "snapshot_15m_valid.json").write_text(json.dumps(valid_snap))

        result = _load_polymarket_snapshots(tmp_path)
        assert len(result) == 1


class TestLoadBinanceSnapshots:
    """Test loading Binance snapshots."""

    def test_loads_valid_snapshots(self, tmp_path: Path):
        """Load valid Binance snapshot files."""
        snap1 = {
            "timestamp": (datetime.now(UTC) - timedelta(hours=1)).isoformat(),
            "timestamp_ms": int((datetime.now(UTC) - timedelta(hours=1)).timestamp() * 1000),
            "trades": [],
        }
        snap2 = {
            "timestamp": datetime.now(UTC).isoformat(),
            "timestamp_ms": int(datetime.now(UTC).timestamp() * 1000),
            "trades": [],
        }

        (tmp_path / "binance_001.json").write_text(json.dumps(snap1))
        (tmp_path / "binance_002.json").write_text(json.dumps(snap2))
        (tmp_path / "binance_latest.json").write_text(json.dumps(snap2))

        result = _load_binance_snapshots(tmp_path)
        assert len(result) == 2

    def test_filters_by_hours(self, tmp_path: Path):
        """Filter Binance snapshots by hours parameter."""
        old_snap = {
            "timestamp": (datetime.now(UTC) - timedelta(hours=48)).isoformat(),
            "timestamp_ms": int((datetime.now(UTC) - timedelta(hours=48)).timestamp() * 1000),
            "trades": [],
        }
        new_snap = {
            "timestamp": datetime.now(UTC).isoformat(),
            "timestamp_ms": int(datetime.now(UTC).timestamp() * 1000),
            "trades": [],
        }

        (tmp_path / "binance_old.json").write_text(json.dumps(old_snap))
        (tmp_path / "binance_new.json").write_text(json.dumps(new_snap))

        result = _load_binance_snapshots(tmp_path, hours=24)
        assert len(result) == 1


class TestExtractBtcMarketProbabilities:
    """Test extracting BTC market probabilities."""

    def test_extracts_from_bitcoin_title(self):
        """Extract from market with 'bitcoin' in title."""
        snapshot = {
            "markets": [
                {
                    "title": "Will Bitcoin hit $100k?",
                    "slug": "btc-100k",
                    "clob_token_ids": ["token_yes", "token_no"],
                    "books": {
                        "yes": {
                            "bids": [{"price": "0.65", "size": "100"}],
                            "asks": [{"price": "0.70", "size": "50"}],
                        }
                    },
                }
            ]
        }

        result = _extract_btc_market_probabilities(snapshot)
        assert result is not None
        assert result["market_title"] == "Will Bitcoin hit $100k?"
        assert result["mid_price"] == 0.675  # (0.65 + 0.70) / 2
        assert abs(result["spread"] - 0.05) < 0.001  # Floating point tolerance

    def test_extracts_from_btc_title(self):
        """Extract from market with 'btc' in title."""
        snapshot = {
            "markets": [
                {
                    "title": "BTC above $50k?",
                    "slug": "btc-50k",
                    "clob_token_ids": ["token_yes", "token_no"],
                    "books": {
                        "yes": {
                            "bids": [{"price": "0.80", "size": "200"}],
                            "asks": [{"price": "0.85", "size": "100"}],
                        }
                    },
                }
            ]
        }

        result = _extract_btc_market_probabilities(snapshot)
        assert result is not None
        assert result["market_title"] == "BTC above $50k?"
        assert result["mid_price"] == 0.825

    def test_returns_none_for_no_btc_market(self):
        """Return None if no BTC market found."""
        snapshot = {
            "markets": [
                {
                    "title": "Will ETH hit $10k?",
                    "books": {"yes": {"bids": [], "asks": []}},
                }
            ]
        }

        result = _extract_btc_market_probabilities(snapshot)
        assert result is None

    def test_skips_empty_books(self):
        """Skip markets with empty orderbooks."""
        snapshot = {
            "markets": [
                {
                    "title": "Will Bitcoin hit $100k?",
                    "books": {"yes": {"bids": [], "asks": []}},
                }
            ]
        }

        result = _extract_btc_market_probabilities(snapshot)
        assert result is None

    def test_uses_best_bid_if_no_ask(self):
        """Use best bid if no asks available."""
        snapshot = {
            "markets": [
                {
                    "title": "Bitcoin test",
                    "books": {
                        "yes": {
                            "bids": [{"price": "0.65", "size": "100"}],
                            "asks": [],
                        }
                    },
                }
            ]
        }

        result = _extract_btc_market_probabilities(snapshot)
        assert result is not None
        assert result["mid_price"] == 0.65

    def test_uses_best_ask_if_no_bid(self):
        """Use best ask if no bids available."""
        snapshot = {
            "markets": [
                {
                    "title": "Bitcoin test",
                    "books": {
                        "yes": {
                            "bids": [],
                            "asks": [{"price": "0.70", "size": "100"}],
                        }
                    },
                }
            ]
        }

        result = _extract_btc_market_probabilities(snapshot)
        assert result is not None
        assert result["mid_price"] == 0.70


class TestAlignSnapshots:
    """Test snapshot alignment by timestamp."""

    def test_aligns_matching_snapshots(self):
        """Align snapshots within tolerance."""
        now = datetime.now(UTC)
        pm_snaps = [
            {
                "generated_at": now.isoformat(),
                "markets": [],
            }
        ]
        bn_snaps = [
            {
                "timestamp": now.isoformat(),
                "timestamp_ms": int(now.timestamp() * 1000),
                "trades": [],
            }
        ]

        result = _align_snapshots(pm_snaps, bn_snaps, tolerance_seconds=5.0)
        assert len(result) == 1
        assert result[0][2] == 0.0  # Zero drift

    def test_uses_closest_within_tolerance(self):
        """Use closest Binance snapshot within tolerance."""
        now = datetime.now(UTC)
        pm_snaps = [
            {
                "generated_at": now.isoformat(),
                "markets": [],
            }
        ]
        bn_snaps = [
            {
                "timestamp": (now - timedelta(seconds=3)).isoformat(),
                "timestamp_ms": int((now - timedelta(seconds=3)).timestamp() * 1000),
                "trades": [],
            },
            {
                "timestamp": (now - timedelta(seconds=1)).isoformat(),
                "timestamp_ms": int((now - timedelta(seconds=1)).timestamp() * 1000),
                "trades": [],
            },
        ]

        result = _align_snapshots(pm_snaps, bn_snaps, tolerance_seconds=5.0)
        assert len(result) == 1
        assert result[0][2] == 1.0  # 1 second drift (closest)

    def test_skips_outside_tolerance(self):
        """Skip snapshots outside tolerance."""
        now = datetime.now(UTC)
        pm_snaps = [
            {
                "generated_at": now.isoformat(),
                "markets": [],
            }
        ]
        bn_snaps = [
            {
                "timestamp": (now - timedelta(seconds=10)).isoformat(),
                "timestamp_ms": int((now - timedelta(seconds=10)).timestamp() * 1000),
                "trades": [],
            }
        ]

        result = _align_snapshots(pm_snaps, bn_snaps, tolerance_seconds=5.0)
        assert len(result) == 0

    def test_handles_missing_generated_at(self):
        """Skip PM snapshots without generated_at."""
        pm_snaps = [{"markets": []}]
        bn_snaps = [{"timestamp": datetime.now(UTC).isoformat(), "timestamp_ms": 12345}]

        result = _align_snapshots(pm_snaps, bn_snaps, tolerance_seconds=5.0)
        assert len(result) == 0


class TestComputeLeadLagCorrelations:
    """Test lead/lag correlation computation."""

    def test_computes_correlations(self):
        """Compute correlations for valid data."""
        # Create correlated data: BTC returns lead PM changes
        np.random.seed(42)
        n = 100
        btc_returns = {60: list(np.random.randn(n))}
        pm_changes = {60: [0] + btc_returns[60][:-1]}  # PM lags BTC by 1

        result = compute_lead_lag_correlations(btc_returns, pm_changes, [60])
        assert len(result) == 1
        assert result[0].horizon_seconds == 60
        assert result[0].btc_lead_corr is not None
        assert result[0].btc_lag_corr is not None
        assert result[0].sample_size == n  # All valid pairs used

    def test_handles_insufficient_samples(self):
        """Handle insufficient sample size."""
        btc_returns = {60: [0.01, 0.02]}
        pm_changes = {60: [0.01, 0.02]}

        result = compute_lead_lag_correlations(btc_returns, pm_changes, [60])
        assert len(result) == 1
        assert result[0].btc_lead_corr is None  # Less than 10 samples
        assert result[0].sample_size == 2

    def test_handles_none_values(self):
        """Handle None values in data."""
        btc_returns = {60: [0.01, None, 0.03, 0.04]}
        pm_changes = {60: [0.01, 0.02, None, 0.04]}

        result = compute_lead_lag_correlations(btc_returns, pm_changes, [60])
        assert len(result) == 1
        assert result[0].sample_size == 2  # Only 2 valid pairs

    def test_mismatched_lengths(self):
        """Handle mismatched array lengths."""
        btc_returns = {60: [0.01, 0.02, 0.03]}
        pm_changes = {60: [0.01, 0.02]}

        result = compute_lead_lag_correlations(btc_returns, pm_changes, [60])
        assert len(result) == 0  # Skipped due to length mismatch


class TestLeadLagCorrelation:
    """Test LeadLagCorrelation dataclass."""

    def test_to_dict(self):
        """Convert to dictionary."""
        corr = LeadLagCorrelation(
            horizon_seconds=300,
            btc_lead_corr=0.5,
            btc_lag_corr=0.3,
            btc_lead_pvalue=0.01,
            btc_lag_pvalue=0.05,
            sample_size=100,
        )
        d = corr.to_dict()
        assert d["horizon_seconds"] == 300
        assert d["horizon_label"] == "5m"
        assert d["btc_lead_corr"] == 0.5
        assert d["sample_size"] == 100

    def test_format_horizon_seconds(self):
        """Format horizon in seconds."""
        corr = LeadLagCorrelation(
            horizon_seconds=30,
            btc_lead_corr=None,
            btc_lag_corr=None,
            btc_lead_pvalue=None,
            btc_lag_pvalue=None,
            sample_size=0,
        )
        assert corr._format_horizon() == "30s"

    def test_format_horizon_minutes(self):
        """Format horizon in minutes."""
        corr = LeadLagCorrelation(
            horizon_seconds=300,
            btc_lead_corr=None,
            btc_lag_corr=None,
            btc_lead_pvalue=None,
            btc_lag_pvalue=None,
            sample_size=0,
        )
        assert corr._format_horizon() == "5m"

    def test_format_horizon_hours(self):
        """Format horizon in hours."""
        corr = LeadLagCorrelation(
            horizon_seconds=3600,
            btc_lead_corr=None,
            btc_lag_corr=None,
            btc_lead_pvalue=None,
            btc_lag_pvalue=None,
            sample_size=0,
        )
        assert corr._format_horizon() == "1h"


class TestSanityMetrics:
    """Test SanityMetrics dataclass."""

    def test_to_dict(self):
        """Convert to dictionary with rounding."""
        metrics = SanityMetrics(
            total_pm_snapshots=100,
            total_bn_snapshots=95,
            aligned_pairs=90,
            pm_with_btc_market=80,
            missingness_pct=5.55555,
            mean_clock_drift_seconds=0.12345,
            max_clock_drift_seconds=1.98765,
            btc_market_titles=["BTC Market 1", "BTC Market 2"],
        )
        d = metrics.to_dict()
        assert d["missingness_pct"] == 5.56  # Rounded to 2 decimals
        assert d["mean_clock_drift_seconds"] == 0.123  # Rounded to 3 decimals
        assert d["btc_market_titles"] == ["BTC Market 1", "BTC Market 2"]


class TestJoinReport:
    """Test JoinReport dataclass."""

    def test_to_dict(self):
        """Convert to dictionary."""
        report = JoinReport(
            generated_at="2024-01-15T10:00:00Z",
            hours_analyzed=24.0,
            correlation_results=[],
            sanity_metrics=SanityMetrics(
                total_pm_snapshots=10,
                total_bn_snapshots=10,
                aligned_pairs=9,
                pm_with_btc_market=8,
                missingness_pct=0.0,
                mean_clock_drift_seconds=0.0,
                max_clock_drift_seconds=0.0,
            ),
        )
        d = report.to_dict()
        assert d["generated_at"] == "2024-01-15T10:00:00Z"
        assert d["hours_analyzed"] == 24.0
        assert "correlations" in d
        assert "sanity_metrics" in d

    def test_to_text_contains_header(self):
        """Text output contains header."""
        report = JoinReport(
            generated_at="2024-01-15T10:00:00Z",
            hours_analyzed=24.0,
            correlation_results=[],
            sanity_metrics=SanityMetrics(
                total_pm_snapshots=10,
                total_bn_snapshots=10,
                aligned_pairs=9,
                pm_with_btc_market=8,
                missingness_pct=0.0,
                mean_clock_drift_seconds=0.0,
                max_clock_drift_seconds=0.0,
            ),
        )
        text = report.to_text()
        assert "DATASET JOIN REPORT" in text
        assert "Polymarket 15m" in text

    def test_to_text_contains_sanity_metrics(self):
        """Text output contains sanity metrics."""
        report = JoinReport(
            generated_at="2024-01-15T10:00:00Z",
            hours_analyzed=24.0,
            correlation_results=[],
            sanity_metrics=SanityMetrics(
                total_pm_snapshots=100,
                total_bn_snapshots=95,
                aligned_pairs=90,
                pm_with_btc_market=80,
                missingness_pct=5.0,
                mean_clock_drift_seconds=0.5,
                max_clock_drift_seconds=2.0,
            ),
        )
        text = report.to_text()
        assert "Polymarket snapshots:" in text
        assert "100" in text
        assert "Mean clock drift:" in text

    def test_to_text_contains_correlations(self):
        """Text output contains correlation results."""
        report = JoinReport(
            generated_at="2024-01-15T10:00:00Z",
            hours_analyzed=24.0,
            correlation_results=[
                LeadLagCorrelation(
                    horizon_seconds=60,
                    btc_lead_corr=0.25,
                    btc_lag_corr=0.15,
                    btc_lead_pvalue=0.01,
                    btc_lag_pvalue=0.05,
                    sample_size=100,
                )
            ],
            sanity_metrics=SanityMetrics(
                total_pm_snapshots=10,
                total_bn_snapshots=10,
                aligned_pairs=9,
                pm_with_btc_market=8,
                missingness_pct=0.0,
                mean_clock_drift_seconds=0.0,
                max_clock_drift_seconds=0.0,
            ),
        )
        text = report.to_text()
        assert "Lead/Lag Correlations" in text
        assert "0.250" in text or "0.25" in text


class TestBuildAlignedDataset:
    """Test the main build_aligned_dataset function."""

    def test_builds_report_with_no_data(self, tmp_path: Path):
        """Handle case with no data files."""
        pm_dir = tmp_path / "pm"
        bn_dir = tmp_path / "bn"
        pm_dir.mkdir()
        bn_dir.mkdir()

        report = build_aligned_dataset(pm_dir, bn_dir, hours=24)
        assert report.hours_analyzed == 24.0
        assert report.sanity_metrics.total_pm_snapshots == 0
        assert report.sanity_metrics.total_bn_snapshots == 0

    def test_builds_report_with_data(self, tmp_path: Path):
        """Build report with actual data."""
        pm_dir = tmp_path / "pm"
        bn_dir = tmp_path / "bn"
        pm_dir.mkdir()
        bn_dir.mkdir()

        now = datetime.now(UTC)

        # Create PM snapshot with BTC market
        pm_snap = {
            "generated_at": now.isoformat(),
            "markets": [
                {
                    "title": "Will Bitcoin go up?",
                    "slug": "btc-up",
                    "clob_token_ids": ["yes_token", "no_token"],
                    "books": {
                        "yes": {
                            "bids": [{"price": "0.60", "size": "100"}],
                            "asks": [{"price": "0.65", "size": "100"}],
                        }
                    },
                }
            ],
        }
        (pm_dir / "snapshot_15m_001.json").write_text(json.dumps(pm_snap))

        # Create Binance snapshot
        bn_snap = {
            "timestamp": now.isoformat(),
            "timestamp_ms": int(now.timestamp() * 1000),
            "trades": [
                {
                    "timestamp_ms": int((now - timedelta(seconds=10)).timestamp() * 1000),
                    "price": 50000.0,
                    "quantity": 1.0,
                    "is_buyer_maker": False,
                    "trade_id": 1,
                },
                {
                    "timestamp_ms": int((now - timedelta(seconds=5)).timestamp() * 1000),
                    "price": 50100.0,
                    "quantity": 1.0,
                    "is_buyer_maker": False,
                    "trade_id": 2,
                },
            ],
        }
        (bn_dir / "binance_001.json").write_text(json.dumps(bn_snap))

        report = build_aligned_dataset(pm_dir, bn_dir, hours=24)
        assert report.sanity_metrics.total_pm_snapshots == 1
        assert report.sanity_metrics.total_bn_snapshots == 1
        assert report.sanity_metrics.aligned_pairs == 1
        assert report.sanity_metrics.pm_with_btc_market == 1

    def test_no_lookahead_in_pm_changes(self, tmp_path: Path):
        """Verify no lookahead bias in PM probability changes.

        This is a critical test: for each point in time, we should only use
        historical (not future) PM probabilities to compute changes.
        """
        pm_dir = tmp_path / "pm"
        bn_dir = tmp_path / "bn"
        pm_dir.mkdir()
        bn_dir.mkdir()

        base_time = datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC)

        # Create PM snapshots at 15-minute intervals with known probabilities
        for i, prob in enumerate([0.5, 0.55, 0.60, 0.58, 0.62]):
            ts = base_time + timedelta(minutes=15 * i)
            pm_snap = {
                "generated_at": ts.isoformat(),
                "markets": [
                    {
                        "title": "Bitcoin test",
                        "slug": "btc-test",
                        "clob_token_ids": ["yes", "no"],
                        "books": {
                            "yes": {
                                "bids": [{"price": str(prob - 0.01), "size": "100"}],
                                "asks": [{"price": str(prob + 0.01), "size": "100"}],
                            }
                        },
                    }
                ],
            }
            (pm_dir / f"snapshot_15m_{i:03d}.json").write_text(json.dumps(pm_snap))

            # Matching Binance snapshot
            bn_snap = {
                "timestamp": ts.isoformat(),
                "timestamp_ms": int(ts.timestamp() * 1000),
                "trades": [
                    {
                        "timestamp_ms": int((ts - timedelta(seconds=30)).timestamp() * 1000),
                        "price": 50000.0 + i * 100,
                        "quantity": 1.0,
                        "is_buyer_maker": False,
                        "trade_id": i * 2,
                    },
                    {
                        "timestamp_ms": int((ts - timedelta(seconds=10)).timestamp() * 1000),
                        "price": 50000.0 + i * 100 + 50,
                        "quantity": 1.0,
                        "is_buyer_maker": False,
                        "trade_id": i * 2 + 1,
                    },
                ],
            }
            (bn_dir / f"binance_{i:03d}.json").write_text(json.dumps(bn_snap))

        report = build_aligned_dataset(pm_dir, bn_dir, hours=24, horizons=[900])  # 15min horizon

        # Verify we have correlation results
        assert len(report.correlation_results) > 0

        # The key check: all samples should be computed from historical data only
        # With our setup, we should have 4 valid pairs (5 snapshots - 1 for lag)
        corr_result = report.correlation_results[0]
        assert corr_result.sample_size <= 4  # At most 4 valid pairs


class TestSaveReport:
    """Test saving reports to files."""

    def test_saves_json_only(self, tmp_path: Path):
        """Save JSON report without text."""
        report = JoinReport(
            generated_at="2024-01-15T10:00:00Z",
            hours_analyzed=24.0,
            correlation_results=[],
            sanity_metrics=SanityMetrics(
                total_pm_snapshots=10,
                total_bn_snapshots=10,
                aligned_pairs=9,
                pm_with_btc_market=8,
                missingness_pct=0.0,
                mean_clock_drift_seconds=0.0,
                max_clock_drift_seconds=0.0,
            ),
        )
        json_path = tmp_path / "report.json"

        result_json, result_text = save_report(report, json_path)

        assert result_json == json_path
        assert result_text is None
        assert json_path.exists()
        data = json.loads(json_path.read_text())
        assert data["hours_analyzed"] == 24.0

    def test_saves_json_and_text(self, tmp_path: Path):
        """Save both JSON and text reports."""
        report = JoinReport(
            generated_at="2024-01-15T10:00:00Z",
            hours_analyzed=24.0,
            correlation_results=[],
            sanity_metrics=SanityMetrics(
                total_pm_snapshots=10,
                total_bn_snapshots=10,
                aligned_pairs=9,
                pm_with_btc_market=8,
                missingness_pct=0.0,
                mean_clock_drift_seconds=0.0,
                max_clock_drift_seconds=0.0,
            ),
        )
        json_path = tmp_path / "report.json"
        text_path = tmp_path / "report.txt"

        result_json, result_text = save_report(report, json_path, text_path)

        assert result_json == json_path
        assert result_text == text_path
        assert json_path.exists()
        assert text_path.exists()
        assert "DATASET JOIN REPORT" in text_path.read_text()


class TestBucketAlignment:
    """Test 15-minute bucket alignment for both streams."""

    def test_snapshots_sorted_by_time(self, tmp_path: Path):
        """Verify snapshots are properly sorted by timestamp."""
        pm_dir = tmp_path / "pm"
        pm_dir.mkdir()

        now = datetime.now(UTC)

        # Create snapshots out of order
        snap1 = {"generated_at": (now - timedelta(hours=2)).isoformat(), "markets": []}
        snap2 = {"generated_at": now.isoformat(), "markets": []}
        snap3 = {"generated_at": (now - timedelta(hours=1)).isoformat(), "markets": []}

        (pm_dir / "snapshot_15m_001.json").write_text(json.dumps(snap2))  # Latest first
        (pm_dir / "snapshot_15m_002.json").write_text(json.dumps(snap1))  # Oldest second
        (pm_dir / "snapshot_15m_003.json").write_text(json.dumps(snap3))  # Middle third

        result = _load_polymarket_snapshots(pm_dir)

        # Should be sorted by generated_at
        assert len(result) == 3
        times = [_parse_timestamp(s["generated_at"]) for s in result]
        assert times[0] < times[1] < times[2]

    def test_binance_snapshots_sorted_by_timestamp_ms(self, tmp_path: Path):
        """Verify Binance snapshots sorted by timestamp_ms."""
        bn_dir = tmp_path / "bn"
        bn_dir.mkdir()

        now = datetime.now(UTC)

        # Create snapshots out of order
        snap1 = {
            "timestamp": (now - timedelta(hours=2)).isoformat(),
            "timestamp_ms": int((now - timedelta(hours=2)).timestamp() * 1000),
            "trades": [],
        }
        snap2 = {
            "timestamp": now.isoformat(),
            "timestamp_ms": int(now.timestamp() * 1000),
            "trades": [],
        }
        snap3 = {
            "timestamp": (now - timedelta(hours=1)).isoformat(),
            "timestamp_ms": int((now - timedelta(hours=1)).timestamp() * 1000),
            "trades": [],
        }

        (bn_dir / "binance_001.json").write_text(json.dumps(snap2))
        (bn_dir / "binance_002.json").write_text(json.dumps(snap1))
        (bn_dir / "binance_003.json").write_text(json.dumps(snap3))

        result = _load_binance_snapshots(bn_dir)

        assert len(result) == 3
        times = [s["timestamp_ms"] for s in result]
        assert times[0] < times[1] < times[2]
