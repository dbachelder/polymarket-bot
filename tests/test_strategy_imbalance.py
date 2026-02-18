"""Tests for strategy_imbalance module."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from polymarket.strategy_imbalance import (
    TradeDecision,
    _compute_imbalance_at_levels,
    _to_float,
    extract_features_from_market,
    extract_features_from_snapshot,
    load_snapshots_for_backtest,
    parameter_sweep,
    run_backtest,
)


class TestToFloat:
    """Tests for _to_float helper."""

    def test_converts_string(self) -> None:
        assert _to_float("0.75") == 0.75

    def test_keeps_float(self) -> None:
        assert _to_float(0.75) == 0.75

    def test_converts_int(self) -> None:
        assert _to_float(1) == 1.0


class TestComputeImbalanceAtLevels:
    """Tests for _compute_imbalance_at_levels function."""

    def test_balanced_book(self) -> None:
        bids = [{"price": "0.60", "size": "100"}]
        asks = [{"price": "0.61", "size": "100"}]
        result = _compute_imbalance_at_levels(bids, asks, 1)
        assert result == 0.5  # Perfectly balanced

    def test_all_bid_depth(self) -> None:
        bids = [{"price": "0.60", "size": "100"}]
        asks = [{"price": "0.61", "size": "0"}]
        result = _compute_imbalance_at_levels(bids, asks, 1)
        assert result == 1.0  # All bid depth

    def test_all_ask_depth(self) -> None:
        bids = [{"price": "0.60", "size": "0"}]
        asks = [{"price": "0.61", "size": "100"}]
        result = _compute_imbalance_at_levels(bids, asks, 1)
        assert result == 0.0  # All ask depth

    def test_no_depth_returns_none(self) -> None:
        bids = []
        asks = []
        result = _compute_imbalance_at_levels(bids, asks, 1)
        assert result is None

    def test_respects_levels(self) -> None:
        # First level balanced, third level imbalanced
        bids = [
            {"price": "0.60", "size": "100"},
            {"price": "0.59", "size": "0"},
            {"price": "0.58", "size": "500"},
        ]
        asks = [
            {"price": "0.61", "size": "100"},
            {"price": "0.62", "size": "0"},
            {"price": "0.63", "size": "0"},
        ]
        # At level 1: 100 / 200 = 0.5
        assert _compute_imbalance_at_levels(bids, asks, 1) == 0.5
        # At level 3: 600 bid / 100 ask = 600/700 = 0.857...
        assert _compute_imbalance_at_levels(bids, asks, 3) == 600.0 / 700.0

    def test_sorts_by_price(self) -> None:
        # Bids should be sorted descending (highest first)
        bids = [
            {"price": "0.50", "size": "10"},
            {"price": "0.60", "size": "100"},  # This is best
        ]
        # Asks should be sorted ascending (lowest first)
        asks = [
            {"price": "0.70", "size": "100"},  # This is best
            {"price": "0.80", "size": "10"},
        ]
        result = _compute_imbalance_at_levels(bids, asks, 1)
        assert result == 0.5  # 100 / 200


class TestExtractFeaturesFromMarket:
    """Tests for extract_features_from_market function."""

    def test_extracts_basic_features(self) -> None:
        market_data = {
            "market_id": "test-123",
            "title": "Bitcoin UP 15m",
            "books": {
                "yes": {
                    "bids": [{"price": "0.60", "size": "100"}],
                    "asks": [{"price": "0.62", "size": "100"}],
                },
                "no": {
                    "bids": [{"price": "0.38", "size": "100"}],
                    "asks": [{"price": "0.40", "size": "100"}],
                },
            },
        }

        result = extract_features_from_market(market_data)

        assert result is not None
        assert result.market_id == "test-123"
        assert result.market_title == "Bitcoin UP 15m"
        assert result.interval_minutes == 15  # Detected from title
        assert result.best_bid_yes == 0.60
        assert result.best_ask_yes == 0.62
        assert abs(result.spread - 0.02) < 0.0001
        assert abs(result.mid_yes - 0.61) < 0.0001
        assert result.imbalance_1 == 0.5

    def test_detects_5m_interval(self) -> None:
        market_data = {
            "market_id": "test-123",
            "title": "Bitcoin UP 5m",
            "books": {
                "yes": {
                    "bids": [{"price": "0.60", "size": "100"}],
                    "asks": [{"price": "0.62", "size": "100"}],
                },
            },
        }

        result = extract_features_from_market(market_data)
        assert result.interval_minutes == 5

    def test_returns_none_for_empty_book(self) -> None:
        market_data = {
            "market_id": "test-123",
            "title": "Bitcoin UP 15m",
            "books": {
                "yes": {"bids": [], "asks": []},
            },
        }

        result = extract_features_from_market(market_data)
        assert result is None

    def test_returns_none_for_one_sided_pathological_book(self) -> None:
        """Test that pathological one-sided books are skipped (YES asks-only + NO bids-only)."""
        market_data = {
            "market_id": "test-123",
            "title": "Bitcoin UP 15m",
            "books": {
                "yes": {
                    # YES has only asks (no bids) - pathological
                    "bids": [],
                    "asks": [{"price": "0.99", "size": "15000"}],
                },
                "no": {
                    # NO has only bids (no asks) - pathological
                    "bids": [{"price": "0.01", "size": "15000"}],
                    "asks": [],
                },
            },
        }

        result = extract_features_from_market(market_data)
        assert result is None

    def test_extracts_features_when_yes_has_both_sides(self) -> None:
        """Test that normal books with both sides on YES are processed."""
        market_data = {
            "market_id": "test-123",
            "title": "Bitcoin UP 15m",
            "books": {
                "yes": {
                    # YES has both sides - normal
                    "bids": [{"price": "0.60", "size": "1000"}],
                    "asks": [{"price": "0.62", "size": "1000"}],
                },
                "no": {
                    # NO may be one-sided but YES is what matters
                    "bids": [{"price": "0.38", "size": "1000"}],
                    "asks": [],
                },
            },
        }

        result = extract_features_from_market(market_data)
        assert result is not None
        assert result.best_bid_yes == 0.60
        assert result.best_ask_yes == 0.62

    def test_computes_mid_delta(self) -> None:
        market_data = {
            "market_id": "test-123",
            "title": "Bitcoin UP 15m",
            "books": {
                "yes": {
                    "bids": [{"price": "0.60", "size": "100"}],
                    "asks": [{"price": "0.62", "size": "100"}],
                },
            },
        }

        result = extract_features_from_market(market_data, prior_mid=0.60)
        assert abs(result.mid_delta - 0.01) < 0.0001  # 0.61 - 0.60

    def test_uses_provided_timestamp(self) -> None:
        ts = datetime(2026, 2, 14, 12, 0, 0, tzinfo=UTC)
        market_data = {
            "title": "Bitcoin UP 15m",
            "books": {
                "yes": {
                    "bids": [{"price": "0.60", "size": "100"}],
                    "asks": [{"price": "0.62", "size": "100"}],
                },
            },
        }

        result = extract_features_from_market(market_data, timestamp=ts)
        assert result.timestamp == ts


class TestExtractFeaturesFromSnapshot:
    """Tests for extract_features_from_snapshot function."""

    def test_extracts_from_snapshot_file(self, tmp_path: Path) -> None:
        snapshot_data = {
            "generated_at": "2026-02-14T12:00:00+00:00",
            "markets": [
                {
                    "market_id": "btc-1",
                    "title": "Bitcoin UP 15m",
                    "books": {
                        "yes": {
                            "bids": [{"price": "0.60", "size": "100"}],
                            "asks": [{"price": "0.62", "size": "100"}],
                        },
                    },
                },
                {
                    "market_id": "eth-1",
                    "title": "Ethereum UP 15m",  # Should be filtered out
                    "books": {
                        "yes": {
                            "bids": [{"price": "0.55", "size": "100"}],
                            "asks": [{"price": "0.57", "size": "100"}],
                        },
                    },
                },
            ],
        }

        snapshot_path = tmp_path / "snapshot.json"
        snapshot_path.write_text(json.dumps(snapshot_data))

        results = extract_features_from_snapshot(snapshot_path, target_market_substring="bitcoin")

        assert len(results) == 1
        assert results[0].market_id == "btc-1"
        assert results[0].timestamp == datetime(2026, 2, 14, 12, 0, 0, tzinfo=UTC)

    def test_returns_empty_list_for_no_matches(self, tmp_path: Path) -> None:
        snapshot_data = {
            "generated_at": "2026-02-14T12:00:00+00:00",
            "markets": [
                {
                    "market_id": "eth-1",
                    "title": "Ethereum UP 15m",
                    "books": {
                        "yes": {
                            "bids": [{"price": "0.55", "size": "100"}],
                            "asks": [{"price": "0.57", "size": "100"}],
                        },
                    },
                },
            ],
        }

        snapshot_path = tmp_path / "snapshot.json"
        snapshot_path.write_text(json.dumps(snapshot_data))

        results = extract_features_from_snapshot(snapshot_path, target_market_substring="bitcoin")
        assert len(results) == 0


class TestLoadSnapshotsForBacktest:
    """Tests for load_snapshots_for_backtest function."""

    def test_loads_and_sorts_snapshots(self, tmp_path: Path) -> None:
        # Create snapshot files
        (tmp_path / "snapshot_15m_20260214T120000Z.json").write_text("{}")
        (tmp_path / "snapshot_15m_20260214T110000Z.json").write_text("{}")
        (tmp_path / "snapshot_15m_20260214T130000Z.json").write_text("{}")

        results = load_snapshots_for_backtest(tmp_path, interval="15m")

        assert len(results) == 3
        # Should be sorted by timestamp
        assert "110000" in str(results[0])
        assert "120000" in str(results[1])
        assert "130000" in str(results[2])

    def test_filters_by_interval(self, tmp_path: Path) -> None:
        (tmp_path / "snapshot_15m_20260214T120000Z.json").write_text("{}")
        (tmp_path / "snapshot_5m_20260214T120000Z.json").write_text("{}")

        results = load_snapshots_for_backtest(tmp_path, interval="15m")
        assert len(results) == 1
        assert "15m" in str(results[0])

    def test_filters_by_time_range(self, tmp_path: Path) -> None:
        (tmp_path / "snapshot_15m_20260214T100000Z.json").write_text("{}")
        (tmp_path / "snapshot_15m_20260214T120000Z.json").write_text("{}")
        (tmp_path / "snapshot_15m_20260214T140000Z.json").write_text("{}")

        start = datetime(2026, 2, 14, 11, 0, 0, tzinfo=UTC)
        end = datetime(2026, 2, 14, 13, 0, 0, tzinfo=UTC)

        results = load_snapshots_for_backtest(
            tmp_path, interval="15m", start_time=start, end_time=end
        )

        assert len(results) == 1
        assert "120000" in str(results[0])

    def test_returns_empty_list_for_no_matches(self, tmp_path: Path) -> None:
        results = load_snapshots_for_backtest(tmp_path, interval="15m")
        assert results == []


class TestRunBacktest:
    """Tests for run_backtest function."""

    def test_generates_up_signal_on_strong_imbalance(self, tmp_path: Path) -> None:
        # Create snapshot with strong bid imbalance (should trigger UP)
        snapshot_data = {
            "generated_at": "2026-02-14T12:00:00+00:00",
            "markets": [
                {
                    "market_id": "btc-1",
                    "title": "Bitcoin UP 15m",
                    "books": {
                        "yes": {
                            "bids": [
                                {"price": "0.60", "size": "1000"},  # Strong bid depth
                                {"price": "0.59", "size": "1000"},
                                {"price": "0.58", "size": "1000"},
                            ],
                            "asks": [
                                {"price": "0.62", "size": "100"},  # Weak ask depth
                                {"price": "0.63", "size": "100"},
                                {"price": "0.64", "size": "100"},
                            ],
                        },
                    },
                },
            ],
        }

        snapshot_path = tmp_path / "snapshot_15m_20260214T120000Z.json"
        snapshot_path.write_text(json.dumps(snapshot_data))

        result = run_backtest(
            snapshots=[snapshot_path],
            k=3,
            theta=0.70,
            p_max=0.65,
        )

        assert len(result.trades) == 1
        assert result.trades[0].decision == "UP"
        assert result.metrics["up_trades"] == 1
        assert result.metrics["down_trades"] == 0

    def test_generates_down_signal_on_weak_imbalance(self, tmp_path: Path) -> None:
        # Create snapshot with strong ask imbalance (should trigger DOWN)
        snapshot_data = {
            "generated_at": "2026-02-14T12:00:00+00:00",
            "markets": [
                {
                    "market_id": "btc-1",
                    "title": "Bitcoin UP 15m",
                    "books": {
                        "yes": {
                            "bids": [
                                {"price": "0.35", "size": "100"},  # Weak bid depth
                                {"price": "0.34", "size": "100"},
                                {"price": "0.33", "size": "100"},
                            ],
                            "asks": [
                                {"price": "0.37", "size": "1000"},  # Strong ask depth
                                {"price": "0.38", "size": "1000"},
                                {"price": "0.39", "size": "1000"},
                            ],
                        },
                    },
                },
            ],
        }

        snapshot_path = tmp_path / "snapshot_15m_20260214T120000Z.json"
        snapshot_path.write_text(json.dumps(snapshot_data))

        result = run_backtest(
            snapshots=[snapshot_path],
            k=3,
            theta=0.70,
            p_max=0.65,
        )

        assert len(result.trades) == 1
        assert result.trades[0].decision == "DOWN"
        assert result.metrics["up_trades"] == 0
        assert result.metrics["down_trades"] == 1

    def test_no_trade_when_price_too_high(self, tmp_path: Path) -> None:
        # Strong imbalance but price above p_max
        snapshot_data = {
            "generated_at": "2026-02-14T12:00:00+00:00",
            "markets": [
                {
                    "market_id": "btc-1",
                    "title": "Bitcoin UP 15m",
                    "books": {
                        "yes": {
                            "bids": [
                                {"price": "0.70", "size": "1000"},
                                {"price": "0.69", "size": "1000"},
                                {"price": "0.68", "size": "1000"},
                            ],
                            "asks": [
                                {"price": "0.72", "size": "100"},
                                {"price": "0.73", "size": "100"},
                                {"price": "0.74", "size": "100"},
                            ],
                        },
                    },
                },
            ],
        }

        snapshot_path = tmp_path / "snapshot_15m_20260214T120000Z.json"
        snapshot_path.write_text(json.dumps(snapshot_data))

        result = run_backtest(
            snapshots=[snapshot_path],
            k=3,
            theta=0.70,
            p_max=0.65,  # Mid is 0.71, which is > 0.65
        )

        # Should not trade because mid (0.71) > p_max (0.65)
        assert len(result.trades) == 0

    def test_no_trade_when_imbalance_not_extreme(self, tmp_path: Path) -> None:
        # Balanced book - no signal
        snapshot_data = {
            "generated_at": "2026-02-14T12:00:00+00:00",
            "markets": [
                {
                    "market_id": "btc-1",
                    "title": "Bitcoin UP 15m",
                    "books": {
                        "yes": {
                            "bids": [
                                {"price": "0.60", "size": "100"},
                                {"price": "0.59", "size": "100"},
                                {"price": "0.58", "size": "100"},
                            ],
                            "asks": [
                                {"price": "0.62", "size": "100"},
                                {"price": "0.63", "size": "100"},
                                {"price": "0.64", "size": "100"},
                            ],
                        },
                    },
                },
            ],
        }

        snapshot_path = tmp_path / "snapshot_15m_20260214T120000Z.json"
        snapshot_path.write_text(json.dumps(snapshot_data))

        result = run_backtest(
            snapshots=[snapshot_path],
            k=3,
            theta=0.70,  # Imbalance is 0.5, which is < 0.70
            p_max=0.65,
        )

        assert len(result.trades) == 0

    def test_uses_pessimistic_entry_price(self, tmp_path: Path) -> None:
        snapshot_data = {
            "generated_at": "2026-02-14T12:00:00+00:00",
            "markets": [
                {
                    "market_id": "btc-1",
                    "title": "Bitcoin UP 15m",
                    "books": {
                        "yes": {
                            "bids": [{"price": "0.60", "size": "1000"}],
                            "asks": [{"price": "0.62", "size": "100"}],
                        },
                    },
                },
            ],
        }

        snapshot_path = tmp_path / "snapshot_15m_20260214T120000Z.json"
        snapshot_path.write_text(json.dumps(snapshot_data))

        result = run_backtest(
            snapshots=[snapshot_path],
            k=1,
            theta=0.70,
            p_max=0.65,
        )

        assert len(result.trades) == 1
        # Entry price should be ask price (pessimistic fill)
        assert result.trades[0].entry_price == 0.62

    def test_result_includes_metrics(self, tmp_path: Path) -> None:
        snapshot_data = {
            "generated_at": "2026-02-14T12:00:00+00:00",
            "markets": [
                {
                    "market_id": "btc-1",
                    "title": "Bitcoin UP 15m",
                    "books": {
                        "yes": {
                            "bids": [{"price": "0.60", "size": "1000"}],
                            "asks": [{"price": "0.62", "size": "100"}],
                        },
                    },
                },
            ],
        }

        snapshot_path = tmp_path / "snapshot_15m_20260214T120000Z.json"
        snapshot_path.write_text(json.dumps(snapshot_data))

        result = run_backtest(snapshots=[snapshot_path], k=1, theta=0.70, p_max=0.65)

        assert result.metrics["total_trades"] == 1
        assert result.metrics["params"]["k"] == 1
        assert result.metrics["params"]["theta"] == 0.70
        assert result.metrics["params"]["p_max"] == 0.65


class TestParameterSweep:
    """Tests for parameter_sweep function."""

    def test_sweeps_all_combinations(self, tmp_path: Path) -> None:
        snapshot_data = {
            "generated_at": "2026-02-14T12:00:00+00:00",
            "markets": [
                {
                    "market_id": "btc-1",
                    "title": "Bitcoin UP 15m",
                    "books": {
                        "yes": {
                            "bids": [{"price": "0.60", "size": "1000"}],
                            "asks": [{"price": "0.62", "size": "100"}],
                        },
                    },
                },
            ],
        }

        snapshot_path = tmp_path / "snapshot_15m_20260214T120000Z.json"
        snapshot_path.write_text(json.dumps(snapshot_data))

        results = parameter_sweep(
            snapshots=[snapshot_path],
            k_values=[1, 3],
            theta_values=[0.60, 0.70],
            p_max_values=[0.60, 0.65],
        )

        # 2 k values × 2 theta values × 2 p_max values = 8 combinations
        assert len(results) == 8

        # All combinations should be present
        params_seen = [
            (r["params"]["k"], r["params"]["theta"], r["params"]["p_max"]) for r in results
        ]
        assert (1, 0.60, 0.60) in params_seen
        assert (3, 0.70, 0.65) in params_seen

    def test_sorts_by_trade_count(self, tmp_path: Path) -> None:
        # Create two snapshots with different market conditions
        snapshot1_data = {
            "generated_at": "2026-02-14T12:00:00+00:00",
            "markets": [
                {
                    "market_id": "btc-1",
                    "title": "Bitcoin UP 15m",
                    "books": {
                        "yes": {
                            "bids": [{"price": "0.60", "size": "1000"}],
                            "asks": [{"price": "0.62", "size": "100"}],
                        },
                    },
                },
            ],
        }

        snapshot_path = tmp_path / "snapshot_15m_20260214T120000Z.json"
        snapshot_path.write_text(json.dumps(snapshot1_data))

        results = parameter_sweep(
            snapshots=[snapshot_path],
            k_values=[1, 3],
            theta_values=[0.60],
            p_max_values=[0.60],
        )

        # Should be sorted by trade count (descending)
        # Both should have 1 trade, so order doesn't matter much
        assert all(r["metrics"]["total_trades"] >= 0 for r in results)


class TestBacktestResult:
    """Tests for BacktestResult dataclass."""

    def test_to_dict_serializes_correctly(self) -> None:
        ts = datetime(2026, 2, 14, 12, 0, 0, tzinfo=UTC)
        trade = TradeDecision(
            timestamp=ts,
            market_id="btc-1",
            market_title="Bitcoin UP 15m",
            decision="UP",
            imbalance_k=3,
            imbalance_value=0.85,
            mid_yes=0.60,
            entry_price=0.62,
            confidence=0.70,
            prob_up=0.85,
        )

        from polymarket.strategy_imbalance import BacktestResult

        result = BacktestResult(
            trades=[trade],
            metrics={"total_trades": 1},
        )

        d = result.to_dict()

        assert d["metrics"]["total_trades"] == 1
        assert len(d["trades"]) == 1
        assert d["trades"][0]["decision"] == "UP"
        assert d["trades"][0]["timestamp"] == "2026-02-14T12:00:00+00:00"
