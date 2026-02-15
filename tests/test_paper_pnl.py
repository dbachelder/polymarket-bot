"""Tests for paper_pnl module."""

from __future__ import annotations

import json

import pytest

from polymarket.paper_pnl import (
    PaperPnLResult,
    Position,
    TradeResult,
    evaluate_simple_exit_rule,
    generate_report,
)


class TestPosition:
    """Test Position dataclass."""

    def test_position_creation(self):
        """Test creating a Position."""
        pos = Position(
            market_slug="bitcoin-up-or-down",
            token_id="token123",
            side="yes",
            entry_price=0.55,
            entry_time="2024-01-01T00:00:00+00:00",
            size=1.0,
        )

        assert pos.market_slug == "bitcoin-up-or-down"
        assert pos.side == "yes"
        assert pos.entry_price == 0.55
        assert pos.pnl == 0.0  # Not closed yet

    def test_position_close(self):
        """Test closing a position."""
        pos = Position(
            market_slug="bitcoin-up-or-down",
            token_id="token123",
            side="yes",
            entry_price=0.55,
            entry_time="2024-01-01T00:00:00+00:00",
            size=1.0,
        )

        pos.close(exit_price=0.65, exit_time="2024-01-01T00:15:00+00:00", reason="timebox_15m")

        assert pos.exit_price == 0.65
        assert pos.exit_reason == "timebox_15m"
        assert pos.pnl == pytest.approx(0.10, rel=1e-3)  # (0.65 - 0.55) * 1.0
        assert pos.pnl_pct == pytest.approx(0.1818, rel=1e-3)  # 0.10 / 0.55

    def test_position_close_loss(self):
        """Test closing a position at a loss."""
        pos = Position(
            market_slug="bitcoin-up-or-down",
            token_id="token123",
            side="yes",
            entry_price=0.55,
            entry_time="2024-01-01T00:00:00+00:00",
            size=1.0,
        )

        pos.close(exit_price=0.45, exit_time="2024-01-01T00:15:00+00:00", reason="stop_loss")

        assert pos.pnl == pytest.approx(-0.10, rel=1e-3)
        assert pos.pnl_pct < 0


class TestTradeResult:
    """Test TradeResult dataclass."""

    def test_trade_result_creation(self):
        """Test creating a TradeResult."""
        trade = TradeResult(
            market_slug="bitcoin-up-or-down",
            decision="up",
            entry_price=0.55,
            exit_price=0.65,
            entry_time="2024-01-01T00:00:00+00:00",
            exit_time="2024-01-01T00:15:00+00:00",
            confidence=0.75,
            pnl=0.10,
            pnl_pct=0.1818,
            exit_reason="timebox_15m",
        )

        assert trade.market_slug == "bitcoin-up-or-down"
        assert trade.decision == "up"
        assert trade.pnl == 0.10
        assert trade.exit_reason == "timebox_15m"


class TestEvaluateSimpleExitRule:
    """Test paper PnL evaluation."""

    def create_aligned_data(self, returns_value: float = 0.01) -> list[dict]:
        """Helper to create aligned data for testing."""
        return [
            {
                "polymarket_timestamp": "2024-01-01T00:00:00+00:00",
                "pricefeed_timestamp": "2024-01-01T00:00:00+00:00",
                "time_diff_seconds": 0.0,
                "venue": "coinbase",
                "polymarket_data": {
                    "markets": [
                        {
                            "market_slug": "bitcoin-up-or-down-15m",
                            "title": "Will Bitcoin go up or down?",
                            "clob_token_ids": ["token_yes", "token_no"],
                            "books": {
                                "yes": {
                                    "bids": [{"price": "0.55", "size": "100"}],
                                    "asks": [{"price": "0.57", "size": "100"}],
                                },
                                "no": {
                                    "bids": [{"price": "0.43", "size": "100"}],
                                    "asks": [{"price": "0.45", "size": "100"}],
                                },
                            },
                        }
                    ]
                },
                "pricefeed_features": {
                    "timestamp": "2024-01-01T00:00:00+00:00",
                    "timestamp_ms": 1700000000000,
                    "symbol": "BTC-USD",
                    "venue": "coinbase",
                    "reference_price": 50000.0,
                    "returns": [
                        {
                            "horizon_seconds": 60,
                            "simple_return": returns_value,
                            "log_return": returns_value * 0.99,
                            "start_price": 50000.0,
                            "end_price": 50000.0 * (1 + returns_value),
                            "start_time": "2024-01-01T00:00:00+00:00",
                            "end_time": "2024-01-01T00:01:00+00:00",
                        }
                    ],
                },
            }
        ]

    def test_evaluate_up_signal(self):
        """Test evaluation with UP signal."""
        aligned = self.create_aligned_data(returns_value=0.01)  # Positive return = UP

        result = evaluate_simple_exit_rule(
            aligned_data=aligned,
            exit_rule="mark_at_end",
            confidence_threshold=0.5,
        )

        assert result.total_trades == 1
        assert result.trades[0].decision == "up"

    def test_evaluate_down_signal(self):
        """Test evaluation with DOWN signal."""
        aligned = self.create_aligned_data(returns_value=-0.01)  # Negative return = DOWN

        result = evaluate_simple_exit_rule(
            aligned_data=aligned,
            exit_rule="mark_at_end",
            confidence_threshold=0.5,
        )

        assert result.total_trades == 1
        assert result.trades[0].decision == "down"

    def test_evaluate_no_signal_below_threshold(self):
        """Test evaluation when confidence is below threshold."""
        aligned = self.create_aligned_data(returns_value=0.001)  # Small return

        result = evaluate_simple_exit_rule(
            aligned_data=aligned,
            exit_rule="mark_at_end",
            confidence_threshold=0.9,  # High threshold
        )

        # Should not trade with low confidence
        assert result.total_trades == 0

    def test_evaluate_no_btc_market(self):
        """Test evaluation when no BTC market is found."""
        aligned = [
            {
                "polymarket_timestamp": "2024-01-01T00:00:00+00:00",
                "polymarket_data": {
                    "markets": [
                        {
                            "market_slug": "ethereum-market",
                            "title": "Ethereum market",
                        }
                    ]
                },
                "pricefeed_features": {
                    "returns": [{"horizon_seconds": 60, "simple_return": 0.01}]
                },
            }
        ]

        result = evaluate_simple_exit_rule(
            aligned_data=aligned,
            exit_rule="mark_at_end",
            confidence_threshold=0.5,
            target_substring="bitcoin",  # Looking for bitcoin
        )

        assert result.total_trades == 0

    def test_pnl_calculation(self):
        """Test PnL calculation is correct."""
        # Create two data points: entry and exit
        # Entry: mid = 0.56, Exit: mid = 0.61, PnL = 0.05
        aligned = [
            {
                "polymarket_timestamp": "2024-01-01T00:00:00+00:00",
                "polymarket_data": {
                    "markets": [
                        {
                            "market_slug": "bitcoin-up-or-down",
                            "title": "Will Bitcoin go up or down?",
                            "clob_token_ids": ["token_yes", "token_no"],
                            "books": {
                                "yes": {
                                    "bids": [{"price": "0.55", "size": "100"}],
                                    "asks": [{"price": "0.57", "size": "100"}],
                                },
                                "no": {
                                    "bids": [{"price": "0.43", "size": "100"}],
                                    "asks": [{"price": "0.45", "size": "100"}],
                                },
                            },
                        }
                    ]
                },
                "pricefeed_features": {
                    "returns": [{"horizon_seconds": 60, "simple_return": 0.01}]
                },
            },
            {
                "polymarket_timestamp": "2024-01-01T00:15:00+00:00",
                "polymarket_data": {
                    "markets": [
                        {
                            "market_slug": "bitcoin-up-or-down",
                            "title": "Will Bitcoin go up or down?",
                            "clob_token_ids": ["token_yes", "token_no"],
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
                    ]
                },
                "pricefeed_features": {
                    "returns": [{"horizon_seconds": 60, "simple_return": 0.01}]
                },
            },
        ]

        result = evaluate_simple_exit_rule(
            aligned_data=aligned,
            exit_rule="mark_at_end",
            confidence_threshold=0.5,
        )

        # Entry at mid of 0.55/0.57 = 0.56, exit at mid of 0.60/0.62 = 0.61
        # PnL = 0.61 - 0.56 = 0.05
        assert result.total_trades == 1
        # Allow for small floating point differences
        assert result.trades[0].pnl == pytest.approx(0.05, abs=0.01)


class TestPaperPnLResult:
    """Test PaperPnLResult dataclass."""

    def test_result_creation(self):
        """Test creating a result."""
        trade = TradeResult(
            market_slug="test",
            decision="up",
            entry_price=0.55,
            exit_price=0.65,
            entry_time="2024-01-01T00:00:00+00:00",
            exit_time="2024-01-01T00:15:00+00:00",
            confidence=0.7,
            pnl=0.10,
            pnl_pct=0.18,
            exit_reason="timebox",
        )

        result = PaperPnLResult(
            total_trades=1,
            winning_trades=1,
            losing_trades=0,
            total_pnl=0.10,
            avg_pnl_per_trade=0.10,
            win_rate=1.0,
            sharpe_ratio=None,
            max_drawdown=0.0,
            trades=[trade],
            parameters={"exit_rule": "timebox"},
        )

        assert result.total_trades == 1
        assert result.winning_trades == 1
        assert result.total_pnl == 0.10
        assert result.win_rate == 1.0

    def test_result_to_dict(self):
        """Test converting result to dict."""
        result = PaperPnLResult(
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            total_pnl=0.0,
            avg_pnl_per_trade=0.0,
            win_rate=0.0,
            sharpe_ratio=None,
            max_drawdown=0.0,
            trades=[],
            parameters={},
        )

        d = result.to_dict()
        assert d["total_trades"] == 0
        assert d["win_rate"] == 0.0
        assert "trades" in d


class TestGenerateReport:
    """Test report generation."""

    def test_human_format(self):
        """Test human-readable report format."""
        trade = TradeResult(
            market_slug="bitcoin-up-or-down",
            decision="up",
            entry_price=0.55,
            exit_price=0.65,
            entry_time="2024-01-01T00:00:00+00:00",
            exit_time="2024-01-01T00:15:00+00:00",
            confidence=0.75,
            pnl=0.10,
            pnl_pct=0.18,
            exit_reason="timebox_15m",
        )

        result = PaperPnLResult(
            total_trades=1,
            winning_trades=1,
            losing_trades=0,
            total_pnl=0.10,
            avg_pnl_per_trade=0.10,
            win_rate=1.0,
            sharpe_ratio=1.5,
            max_drawdown=0.0,
            trades=[trade],
            parameters={
                "exit_rule": "timebox",
                "timebox_minutes": 15.0,
                "confidence_threshold": 0.6,
            },
        )

        report = generate_report(result, format="human")

        assert "PAPER PnL EVALUATION REPORT" in report
        assert "Total Trades:   1" in report
        assert "Win Rate:       100.0%" in report
        assert "Total PnL:" in report
        assert "Sharpe Ratio:   1.50" in report

    def test_json_format(self):
        """Test JSON report format."""
        result = PaperPnLResult(
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            total_pnl=0.0,
            avg_pnl_per_trade=0.0,
            win_rate=0.0,
            sharpe_ratio=None,
            max_drawdown=0.0,
            trades=[],
            parameters={},
        )

        report = generate_report(result, format="json")
        data = json.loads(report)

        assert data["total_trades"] == 0
        assert "generated_at" in data
