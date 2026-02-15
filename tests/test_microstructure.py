from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from polymarket.microstructure import (
    DEFAULT_DEPTH_LEVELS,
    DEFAULT_EXTREME_PIN_THRESHOLD,
    DEFAULT_SPREAD_ALERT_THRESHOLD,
    _compute_book_metrics,
    _compute_implied_probabilities,
    analyze_market_microstructure,
    analyze_snapshot_microstructure,
    check_alerts,
    generate_microstructure_summary,
    write_microstructure_report,
)


def test_compute_book_metrics_basic():
    """Test basic book metrics computation."""
    bids = [
        {"price": "0.50", "size": "100"},
        {"price": "0.49", "size": "200"},
    ]
    asks = [
        {"price": "0.52", "size": "150"},
        {"price": "0.53", "size": "250"},
    ]

    result = _compute_book_metrics(bids, asks, depth_levels=10)

    assert result["best_bid"] == 0.50
    assert result["best_ask"] == 0.52
    assert result["spread"] == pytest.approx(0.02)
    assert result["bid_depth"] == 300  # 100 + 200
    assert result["ask_depth"] == 400  # 150 + 250
    assert result["imbalance"] == pytest.approx(-0.142857, rel=1e-3)  # (300-400)/(300+400)


def test_compute_book_metrics_empty_book():
    """Test handling of empty books."""
    result = _compute_book_metrics([], [])

    assert result["best_bid"] is None
    assert result["best_ask"] is None
    assert result["spread"] is None


def test_compute_book_metrics_only_bids():
    """Test handling of book with only bids."""
    bids = [{"price": "0.50", "size": "100"}]
    result = _compute_book_metrics(bids, [])

    assert result["best_bid"] == 0.50
    assert result["best_ask"] is None
    assert result["spread"] is None
    assert result["bid_depth"] == 100.0
    assert result["ask_depth"] is None
    assert result["imbalance"] is None


def test_compute_book_metrics_depth_limiting():
    """Test that depth_levels limits the calculation."""
    bids = [{"price": f"{0.50 - i*0.01:.2f}", "size": "100"} for i in range(20)]
    asks = [{"price": f"{0.52 + i*0.01:.2f}", "size": "100"} for i in range(20)]

    result = _compute_book_metrics(bids, asks, depth_levels=5)

    # Should only sum top 5 levels
    assert result["bid_depth"] == 500
    assert result["ask_depth"] == 500
    assert result["imbalance"] == 0.0  # Perfectly balanced


def test_compute_implied_probabilities():
    """Test implied probability computation."""
    yes_metrics = {
        "best_bid": 0.60,
        "best_ask": 0.62,
    }
    no_metrics = {
        "best_bid": 0.38,
        "best_ask": 0.40,
    }

    result = _compute_implied_probabilities(yes_metrics, no_metrics)

    assert result is not None
    assert result["yes_mid"] == pytest.approx(0.61)
    assert result["no_mid"] == pytest.approx(0.39)
    assert result["mid_sum"] == pytest.approx(1.00)
    assert result["yes_implied"] == pytest.approx(0.61)
    assert result["yes_implied_from_no"] == pytest.approx(0.61)  # 1 - 0.39
    assert result["consistency_diff"] == pytest.approx(0.0)


def test_compute_implied_probabilities_incomplete():
    """Test handling of incomplete metrics."""
    yes_metrics = {"best_bid": 0.60, "best_ask": None}
    no_metrics = {"best_bid": 0.38, "best_ask": 0.40}

    result = _compute_implied_probabilities(yes_metrics, no_metrics)

    assert result is None


def test_check_alerts_spread():
    """Test spread alerts."""
    metrics = {
        "market_title": "Test Market",
        "yes": {"best_bid": 0.10, "best_ask": 0.80, "spread": 0.70},
        "no": {"best_bid": 0.15, "best_ask": 0.85, "spread": 0.70},
    }

    alerts = check_alerts(metrics, spread_threshold=0.50, extreme_pin_threshold=0.05)

    # Should have 2 spread alerts (yes and no)
    assert len(alerts) == 2
    assert "YES spread alert" in alerts[0]
    assert "NO spread alert" in alerts[1]


def test_check_alerts_extreme_pin():
    """Test extreme price pinning alerts."""
    metrics = {
        "market_title": "Test Market",
        "yes": {"best_bid": 0.01, "best_ask": 0.99, "spread": 0.98},
        "no": {"best_bid": 0.01, "best_ask": 0.99, "spread": 0.98},
    }

    alerts = check_alerts(metrics, spread_threshold=0.99, extreme_pin_threshold=0.05)

    # Should have 4 extreme pin alerts (yes bid, yes ask, no bid, no ask)
    assert len(alerts) == 4
    assert any("YES best bid pinned" in a for a in alerts)
    assert any("YES best ask pinned" in a for a in alerts)
    assert any("NO best bid pinned" in a for a in alerts)
    assert any("NO best ask pinned" in a for a in alerts)


def test_check_alerts_healthy():
    """Test that healthy markets produce no alerts."""
    metrics = {
        "market_title": "Test Market",
        "yes": {"best_bid": 0.48, "best_ask": 0.52, "spread": 0.04},
        "no": {"best_bid": 0.47, "best_ask": 0.51, "spread": 0.04},
    }

    alerts = check_alerts(metrics, spread_threshold=0.50, extreme_pin_threshold=0.05)

    assert len(alerts) == 0


def test_analyze_market_microstructure():
    """Test full market microstructure analysis."""
    market_data = {
        "title": "Bitcoin Up or Down - 15 min",
        "market_id": "123",
        "event_id": "456",
        "books": {
            "yes": {
                "bids": [{"price": "0.60", "size": "1000"}],
                "asks": [{"price": "0.62", "size": "1500"}],
            },
            "no": {
                "bids": [{"price": "0.38", "size": "800"}],
                "asks": [{"price": "0.40", "size": "1200"}],
            },
        },
    }

    result = analyze_market_microstructure(market_data)

    assert result["market_title"] == "Bitcoin Up or Down - 15 min"
    assert result["market_id"] == "123"
    assert result["event_id"] == "456"
    assert "timestamp" in result

    # Check YES metrics
    yes = result["yes"]
    assert yes["best_bid"] == 0.60
    assert yes["best_ask"] == 0.62
    assert yes["spread"] == pytest.approx(0.02)

    # Check NO metrics
    no = result["no"]
    assert no["best_bid"] == 0.38
    assert no["best_ask"] == 0.40
    assert no["spread"] == pytest.approx(0.02)

    # Check implied probabilities
    implied = result["implied_probabilities"]
    assert implied is not None
    assert implied["yes_mid"] == pytest.approx(0.61)
    assert implied["no_mid"] == pytest.approx(0.39)


def test_analyze_market_microstructure_no_books():
    """Test analysis when books are missing."""
    market_data = {
        "title": "Test Market",
        "books": {},
    }

    result = analyze_market_microstructure(market_data)

    assert result["market_title"] == "Test Market"
    assert result["yes"]["best_bid"] is None
    assert result["no"]["best_bid"] is None


def test_analyze_snapshot_microstructure(tmp_path: Path):
    """Test analyzing a full snapshot file."""
    snapshot = {
        "markets": [
            {
                "title": "Bitcoin Market",
                "books": {
                    "yes": {"bids": [{"price": "0.60", "size": "100"}], "asks": [{"price": "0.62", "size": "100"}]},
                    "no": {"bids": [{"price": "0.38", "size": "100"}], "asks": [{"price": "0.40", "size": "100"}]},
                },
            },
            {
                "title": "Ethereum Market",
                "books": {
                    "yes": {"bids": [{"price": "0.55", "size": "100"}], "asks": [{"price": "0.57", "size": "100"}]},
                    "no": {"bids": [{"price": "0.43", "size": "100"}], "asks": [{"price": "0.45", "size": "100"}]},
                },
            },
        ],
    }

    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot))

    results = analyze_snapshot_microstructure(snapshot_path, target_market_substring="bitcoin")

    assert len(results) == 1
    assert results[0]["market_title"] == "Bitcoin Market"


def test_generate_microstructure_summary(tmp_path: Path):
    """Test generating a full summary report."""
    snapshot = {
        "markets": [
            {
                "title": "Bitcoin Up or Down - 15 min",
                "market_id": "123",
                "event_id": "456",
                "books": {
                    "yes": {
                        "bids": [{"price": "0.01", "size": "30000"}],
                        "asks": [{"price": "0.99", "size": "30000"}],
                    },
                    "no": {
                        "bids": [{"price": "0.01", "size": "30000"}],
                        "asks": [{"price": "0.99", "size": "30000"}],
                    },
                },
            },
        ],
    }

    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot))

    summary = generate_microstructure_summary(
        snapshot_path=snapshot_path,
        target_market_substring="bitcoin",
        spread_threshold=0.50,
        extreme_pin_threshold=0.05,
        depth_levels=10,
    )

    assert summary["markets_analyzed"] == 1
    assert summary["snapshot_path"] == str(snapshot_path)
    assert "generated_at" in summary

    # Should have alerts for extreme pinning AND wide spreads
    # 2 spread alerts (YES + NO) + 4 extreme pin alerts (bid/ask for each) = 6 total
    assert summary["alert_count"] == 6
    assert len(summary["alerts"]) == 6

    # Check market summary
    ms = summary["market_summaries"][0]
    assert ms["market_title"] == "Bitcoin Up or Down - 15 min"
    assert ms["yes_spread"] == pytest.approx(0.98)
    assert ms["no_spread"] == pytest.approx(0.98)


def test_write_microstructure_report(tmp_path: Path):
    """Test writing a microstructure report to disk."""
    snapshot = {
        "markets": [
            {
                "title": "Bitcoin Market",
                "books": {
                    "yes": {"bids": [{"price": "0.60", "size": "100"}], "asks": [{"price": "0.62", "size": "100"}]},
                    "no": {"bids": [{"price": "0.38", "size": "100"}], "asks": [{"price": "0.40", "size": "100"}]},
                },
            },
        ],
    }

    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot))

    out_path = tmp_path / "report.json"
    result_path = write_microstructure_report(
        snapshot_path=snapshot_path,
        out_path=out_path,
        target_market_substring=None,
    )

    assert result_path == out_path
    assert out_path.exists()

    # Verify the report is valid JSON
    report = json.loads(out_path.read_text())
    assert report["markets_analyzed"] == 1
    assert len(report["market_summaries"]) == 1


def test_btc_15m_extreme_pinning_scenario():
    """Test the specific scenario mentioned in the ticket: BTC 15m pinned at 0.01/0.99."""
    # Note: bids sorted descending means highest first, asks sorted ascending means lowest first
    # For extreme pinning scenario: best_bid=0.01 (highest bid), best_ask=0.99 (lowest ask)
    market_data = {
        "title": "Bitcoin Up or Down - 15 min",
        "market_id": "1375837",
        "event_id": "208612",
        "books": {
            "yes": {
                # Bids: sorted descending -> 0.03, 0.02, 0.01 (best_bid = 0.03)
                # But in real pinned scenario, bids only exist at low prices like 0.01
                "bids": [{"price": "0.01", "size": "29808.1"}, {"price": "0.02", "size": "13077.66"}],
                # Asks: sorted ascending -> 0.98, 0.99 (best_ask = 0.98)
                "asks": [{"price": "0.99", "size": "29685.93"}, {"price": "0.98", "size": "13739.72"}],
            },
            "no": {
                "bids": [{"price": "0.01", "size": "29685.93"}, {"price": "0.02", "size": "13739.72"}],
                "asks": [{"price": "0.99", "size": "29808.1"}, {"price": "0.98", "size": "13077.66"}],
            },
        },
    }

    result = analyze_market_microstructure(market_data, depth_levels=10)

    # Verify YES metrics - best_bid is HIGHEST (0.02), best_ask is LOWEST (0.98)
    yes = result["yes"]
    assert yes["best_bid"] == 0.02  # Highest bid
    assert yes["best_ask"] == 0.98  # Lowest ask
    assert yes["spread"] == pytest.approx(0.96)
    # Top 10 depth (only have 2 levels in test data)
    assert yes["bid_depth"] == pytest.approx(29808.1 + 13077.66)
    assert yes["ask_depth"] == pytest.approx(29685.93 + 13739.72)

    # Verify NO metrics
    no = result["no"]
    assert no["best_bid"] == 0.02  # Highest bid
    assert no["best_ask"] == 0.98  # Lowest ask
    assert no["spread"] == pytest.approx(0.96)

    # Implied probabilities
    implied = result["implied_probabilities"]
    assert implied is not None
    assert implied["yes_mid"] == pytest.approx(0.50)  # (0.02 + 0.98) / 2
    assert implied["no_mid"] == pytest.approx(0.50)  # (0.02 + 0.98) / 2
    assert implied["mid_sum"] == pytest.approx(1.00)  # Should sum to ~1.0

    # Check alerts - note: with best_bid=0.02, extreme pin threshold of 0.05
    # means only bids <= 0.05 trigger. 0.02 <= 0.05, so it should trigger.
    alerts = check_alerts(result, spread_threshold=0.50, extreme_pin_threshold=0.05)

    # Should have spread alerts (spread=0.96 > 0.50) and extreme pin alerts (0.02 <= 0.05)
    assert any("YES spread alert" in a for a in alerts)
    assert any("NO spread alert" in a for a in alerts)
    assert any("YES best bid pinned at extreme: 0.02" in a for a in alerts)
    assert any("YES best ask pinned at extreme: 0.98" in a for a in alerts)
    assert any("NO best bid pinned at extreme: 0.02" in a for a in alerts)
    assert any("NO best ask pinned at extreme: 0.98" in a for a in alerts)
