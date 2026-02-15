"""Tests for paper trading engine."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

from polymarket.paper_trading import (
    EquitySnapshot,
    PaperTradingEngine,
    PositionState,
    ReconciliationResult,
    generate_pnl_attribution_report,
    run_equity_calculation_against_snapshots,
)
from polymarket.pnl import BookLevel, OrderBook


class TestPositionState:
    """Test PositionState dataclass."""

    def test_position_state_creation(self):
        pos = PositionState(
            token_id="token123",
            market_slug="will-bitcoin-hit-100k",
            net_size=Decimal("100"),
            avg_cost_basis=Decimal("0.55"),
        )
        assert pos.token_id == "token123"
        assert pos.market_slug == "will-bitcoin-hit-100k"
        assert pos.net_size == Decimal("100")
        assert pos.avg_cost_basis == Decimal("0.55")

    def test_position_state_to_dict(self):
        pos = PositionState(
            token_id="token123",
            market_slug="test-market",
            net_size=Decimal("100.5"),
            avg_cost_basis=Decimal("0.55"),
            realized_pnl=Decimal("10.25"),
        )
        d = pos.to_dict()
        assert d["token_id"] == "token123"
        assert d["market_slug"] == "test-market"
        assert d["net_size"] == 100.5
        assert d["avg_cost_basis"] == 0.55
        assert d["realized_pnl"] == 10.25

    def test_position_state_from_dict(self):
        data = {
            "token_id": "token456",
            "market_slug": "another-market",
            "net_size": "200",
            "avg_cost_basis": "0.60",
            "realized_pnl": "25.50",
            "total_fees": "2.00",
        }
        pos = PositionState.from_dict(data)
        assert pos.token_id == "token456"
        assert pos.net_size == Decimal("200")
        assert pos.avg_cost_basis == Decimal("0.60")
        assert pos.realized_pnl == Decimal("25.50")


class TestEquitySnapshot:
    """Test EquitySnapshot dataclass."""

    def test_equity_snapshot_creation(self):
        snap = EquitySnapshot(
            timestamp="2024-01-15T12:00:00Z",
            cash_balance=Decimal("5000"),
            mark_to_market=Decimal("5500"),
            liquidation_value=Decimal("5400"),
            realized_pnl=Decimal("500"),
            unrealized_pnl=Decimal("500"),
            total_fees=Decimal("50"),
            net_equity=Decimal("10500"),
            position_count=5,
            open_position_count=3,
        )
        assert snap.timestamp == "2024-01-15T12:00:00Z"
        assert snap.cash_balance == Decimal("5000")
        assert snap.net_equity == Decimal("10500")

    def test_equity_snapshot_roundtrip(self):
        snap = EquitySnapshot(
            timestamp="2024-01-15T12:00:00Z",
            cash_balance=Decimal("5000.50"),
            mark_to_market=Decimal("5500.25"),
            liquidation_value=Decimal("5400"),
            realized_pnl=Decimal("500"),
            unrealized_pnl=Decimal("500.25"),
            total_fees=Decimal("50"),
            net_equity=Decimal("10500.75"),
            position_count=5,
            open_position_count=3,
        )
        d = snap.to_dict()
        restored = EquitySnapshot.from_dict(d)
        assert restored.cash_balance == snap.cash_balance
        assert restored.net_equity == snap.net_equity
        assert restored.position_count == snap.position_count


class TestPaperTradingEngine:
    """Test PaperTradingEngine class."""

    def test_engine_initialization(self, tmp_path: Path):
        engine = PaperTradingEngine(data_dir=tmp_path, starting_cash=Decimal("5000"))
        assert engine.starting_cash == Decimal("5000")
        assert engine.data_dir == tmp_path
        assert engine.fills_path == tmp_path / "fills.jsonl"

    def test_record_fill(self, tmp_path: Path):
        engine = PaperTradingEngine(data_dir=tmp_path, starting_cash=Decimal("10000"))
        fill = engine.record_fill(
            token_id="token123",
            side="buy",
            size=Decimal("100"),
            price=Decimal("0.50"),
            fee=Decimal("1.00"),
            market_slug="test-market",
        )

        assert fill.token_id == "token123"
        assert fill.side == "buy"
        assert fill.size == Decimal("100")
        assert engine.fills_path.exists()

    def test_load_fills(self, tmp_path: Path):
        engine = PaperTradingEngine(data_dir=tmp_path, starting_cash=Decimal("10000"))

        # Record some fills
        engine.record_fill("token1", "buy", Decimal("100"), Decimal("0.50"))
        engine.record_fill("token1", "sell", Decimal("50"), Decimal("0.60"))

        # Load and verify
        fills = engine.load_fills()
        assert len(fills) == 2
        assert fills[0].token_id == "token1"
        assert fills[1].side == "sell"

    def test_get_positions(self, tmp_path: Path):
        engine = PaperTradingEngine(data_dir=tmp_path, starting_cash=Decimal("10000"))

        # Record fills to build position
        engine.record_fill("token1", "buy", Decimal("100"), Decimal("0.50"), market_slug="market-a")
        engine.record_fill("token1", "sell", Decimal("30"), Decimal("0.60"), market_slug="market-a")
        engine.record_fill("token2", "buy", Decimal("50"), Decimal("0.40"), market_slug="market-b")

        positions = engine.get_positions()
        assert len(positions) == 2
        assert positions["token1"].net_size == Decimal("70")
        assert positions["token1"].market_slug == "market-a"
        assert positions["token2"].net_size == Decimal("50")

    def test_compute_equity(self, tmp_path: Path):
        engine = PaperTradingEngine(data_dir=tmp_path, starting_cash=Decimal("10000"))

        # Record a fill
        engine.record_fill("token1", "buy", Decimal("100"), Decimal("0.50"), fee=Decimal("1.00"))

        # Create orderbook for pricing
        orderbooks = {
            "token1": OrderBook(
                token_id="token1",
                bids=[BookLevel(price=Decimal("0.58"), size=Decimal("1000"))],
                asks=[BookLevel(price=Decimal("0.62"), size=Decimal("1000"))],
            )
        }

        equity = engine.compute_equity(orderbooks=orderbooks)

        # Cash = 10000 - (100 * 0.50 + 1) = 10000 - 51 = 9949
        assert equity.cash_balance == Decimal("9949")
        # Mark to market = 100 * 0.60 (mid of 0.58 and 0.62)
        assert equity.mark_to_market == Decimal("60")
        # Net equity
        assert equity.net_equity == Decimal("10009")
        assert equity.open_position_count == 1

    def test_record_and_load_equity(self, tmp_path: Path):
        engine = PaperTradingEngine(data_dir=tmp_path, starting_cash=Decimal("10000"))

        # Record some equity snapshots
        engine.record_equity()
        engine.record_equity()

        # Load equity curve
        curve = engine.load_equity_curve()
        assert len(curve) == 2
        assert curve[0].cash_balance == Decimal("10000")

    def test_equity_curve_filtering(self, tmp_path: Path):
        engine = PaperTradingEngine(data_dir=tmp_path, starting_cash=Decimal("10000"))

        # Record equity with specific timestamps
        snap1 = EquitySnapshot(
            timestamp="2024-01-01T00:00:00Z",
            cash_balance=Decimal("10000"),
            mark_to_market=Decimal("0"),
            liquidation_value=Decimal("0"),
            realized_pnl=Decimal("0"),
            unrealized_pnl=Decimal("0"),
            total_fees=Decimal("0"),
            net_equity=Decimal("10000"),
            position_count=0,
            open_position_count=0,
        )
        snap2 = EquitySnapshot(
            timestamp="2024-02-01T00:00:00Z",
            cash_balance=Decimal("11000"),
            mark_to_market=Decimal("0"),
            liquidation_value=Decimal("0"),
            realized_pnl=Decimal("1000"),
            unrealized_pnl=Decimal("0"),
            total_fees=Decimal("0"),
            net_equity=Decimal("11000"),
            position_count=0,
            open_position_count=0,
        )

        with open(engine.equity_path, "w") as f:
            f.write(json.dumps(snap1.to_dict()) + "\n")
            f.write(json.dumps(snap2.to_dict()) + "\n")

        curve = engine.load_equity_curve(since="2024-01-15T00:00:00Z")
        assert len(curve) == 1
        assert curve[0].timestamp == "2024-02-01T00:00:00Z"

    def test_get_equity_curve_summary(self, tmp_path: Path):
        engine = PaperTradingEngine(data_dir=tmp_path, starting_cash=Decimal("10000"))

        # Create equity curve
        for i, equity_val in enumerate([10000, 10500, 10200, 11000]):
            snap = EquitySnapshot(
                timestamp=f"2024-01-{i + 1:02d}T00:00:00Z",
                cash_balance=Decimal(str(equity_val)),
                mark_to_market=Decimal("0"),
                liquidation_value=Decimal("0"),
                realized_pnl=Decimal(str(equity_val - 10000)),
                unrealized_pnl=Decimal("0"),
                total_fees=Decimal("0"),
                net_equity=Decimal(str(equity_val)),
                position_count=0,
                open_position_count=0,
            )
            with open(engine.equity_path, "a") as f:
                f.write(json.dumps(snap.to_dict()) + "\n")

        summary = engine.get_equity_curve_summary()

        assert summary["data_points"] == 4
        assert summary["starting_equity"] == 10000.0
        assert summary["current_equity"] == 11000.0
        assert summary["total_return"] == 1000.0
        assert summary["total_return_pct"] == 10.0
        # Max drawdown = 10500 -> 10200 = 300
        assert summary["max_drawdown"] == 300.0

    def test_save_positions(self, tmp_path: Path):
        engine = PaperTradingEngine(data_dir=tmp_path, starting_cash=Decimal("10000"))

        # Record fills
        engine.record_fill("token1", "buy", Decimal("100"), Decimal("0.50"), market_slug="market-a")

        path = engine.save_positions()
        assert path.exists()

        data = json.loads(path.read_text())
        assert "saved_at" in data
        assert "positions" in data
        assert "token1" in data["positions"]


class TestReconciliation:
    """Test reconciliation against snapshots."""

    def test_reconcile_no_positions(self, tmp_path: Path):
        engine = PaperTradingEngine(data_dir=tmp_path, starting_cash=Decimal("10000"))

        # Create a mock snapshot file
        snapshot_data = {
            "generated_at": "2024-01-15T12:00:00Z",
            "markets": [
                {
                    "condition_id": "market123",
                    "books": {
                        "yes": {
                            "bids": [{"price": "0.60", "size": "100"}],
                            "asks": [{"price": "0.65", "size": "100"}],
                        },
                    },
                }
            ],
        }
        snapshot_path = tmp_path / "snapshot_15m_20240115T120000Z.json"
        snapshot_path.write_text(json.dumps(snapshot_data))

        result = engine.reconcile_against_snapshot(snapshot_path)

        assert result.positions_reconciled == 0
        assert result.positions_with_drift == 0
        assert result.total_drift_usd == Decimal("0")

    def test_reconcile_with_positions(self, tmp_path: Path):
        engine = PaperTradingEngine(data_dir=tmp_path, starting_cash=Decimal("10000"))

        # Record a fill
        engine.record_fill(
            "market123_yes", "buy", Decimal("100"), Decimal("0.50"), market_slug="test-market"
        )

        # Create snapshot with different price
        snapshot_data = {
            "generated_at": "2024-01-15T12:00:00Z",
            "markets": [
                {
                    "condition_id": "market123",
                    "books": {
                        "yes": {
                            "bids": [{"price": "0.58", "size": "100"}],
                            "asks": [{"price": "0.62", "size": "100"}],
                        },
                    },
                }
            ],
        }
        snapshot_path = tmp_path / "snapshot_15m_20240115T120000Z.json"
        snapshot_path.write_text(json.dumps(snapshot_data))

        result = engine.reconcile_against_snapshot(
            snapshot_path, drift_threshold_usd=Decimal("0.01")
        )

        assert result.positions_reconciled == 1
        # Drift = 100 * 0.60 (mid) - 100 * 0.50 (cost) = 10
        assert result.positions_with_drift == 1
        assert result.total_drift_usd == Decimal("10")

    def test_reconciliation_result_to_dict(self):
        result = ReconciliationResult(
            snapshot_timestamp="2024-01-15T12:00:00Z",
            positions_reconciled=5,
            positions_with_drift=2,
            total_drift_usd=Decimal("15.50"),
            max_drift_pct=Decimal("5.25"),
            warnings=["Test warning"],
            position_drifts=[{"token_id": "token1", "drift_usd": 10.0}],
        )
        d = result.to_dict()
        assert d["positions_reconciled"] == 5
        assert d["total_drift_usd"] == 15.50
        assert len(d["warnings"]) == 1


class TestBacktestAgainstSnapshots:
    """Test running equity calculation against snapshots."""

    def test_run_equity_calculation(self, tmp_path: Path):
        # Create snapshot directory with mock snapshots
        snapshot_dir = tmp_path / "snapshots"
        snapshot_dir.mkdir()

        for i in range(3):
            snapshot_data = {
                "generated_at": f"2024-01-{i + 1:02d}T12:00:00Z",
                "markets": [
                    {
                        "condition_id": f"market{i}",
                        "books": {
                            "yes": {
                                "bids": [{"price": "0.55", "size": "100"}],
                                "asks": [{"price": "0.60", "size": "100"}],
                            },
                        },
                    }
                ],
            }
            snap_path = snapshot_dir / f"snapshot_15m_202401{i + 1:02d}T120000Z.json"
            snap_path.write_text(json.dumps(snapshot_data))

        # Create fills
        data_dir = tmp_path / "paper_trading"
        engine = PaperTradingEngine(data_dir=data_dir, starting_cash=Decimal("10000"))
        engine.record_fill("market0_yes", "buy", Decimal("100"), Decimal("0.50"))

        summary = run_equity_calculation_against_snapshots(
            data_dir=data_dir,
            snapshot_dir=snapshot_dir,
            output_file=None,
        )

        assert summary["snapshots_processed"] == 3
        assert summary["data_points"] == 3
        assert "starting_equity" in summary

    def test_run_equity_calculation_no_snapshots(self, tmp_path: Path):
        data_dir = tmp_path / "paper_trading"
        snapshot_dir = tmp_path / "snapshots"
        snapshot_dir.mkdir()

        summary = run_equity_calculation_against_snapshots(
            data_dir=data_dir,
            snapshot_dir=snapshot_dir,
        )

        assert "error" in summary
        assert "No 15m snapshots found" in summary["error"]


class TestPnlAttribution:
    """Test PnL attribution report."""

    def test_generate_pnl_attribution(self, tmp_path: Path):
        engine = PaperTradingEngine(data_dir=tmp_path, starting_cash=Decimal("10000"))

        # Record fills for multiple markets
        engine.record_fill(
            "token1", "buy", Decimal("100"), Decimal("0.50"), market_slug="market-winning"
        )
        engine.record_fill(
            "token1", "sell", Decimal("100"), Decimal("0.60"), market_slug="market-winning"
        )
        engine.record_fill(
            "token2", "buy", Decimal("100"), Decimal("0.50"), market_slug="market-losing"
        )
        engine.record_fill(
            "token2", "sell", Decimal("100"), Decimal("0.40"), market_slug="market-losing"
        )

        report = generate_pnl_attribution_report(engine)

        assert report["summary"]["total_markets"] == 2
        assert report["summary"]["winning_markets"] == 1
        assert report["summary"]["losing_markets"] == 1
        assert len(report["markets"]) == 2

        # Sorting should put winner first
        assert report["markets"][0]["market_slug"] == "market-winning"
        assert report["markets"][0]["realized_pnl"] > 0
        assert report["markets"][1]["realized_pnl"] < 0

    def test_pnl_attribution_empty(self, tmp_path: Path):
        engine = PaperTradingEngine(data_dir=tmp_path, starting_cash=Decimal("10000"))

        report = generate_pnl_attribution_report(engine)

        assert report["summary"]["total_markets"] == 0
        assert len(report["markets"]) == 0


class TestPaperTradingIntegration:
    """Integration tests for paper trading workflow."""

    def test_full_workflow(self, tmp_path: Path):
        """Test complete fills -> positions -> equity workflow."""
        engine = PaperTradingEngine(data_dir=tmp_path, starting_cash=Decimal("10000"))

        # Step 1: Record initial equity (before any trades)
        initial_equity = engine.record_equity()
        assert initial_equity.net_equity == Decimal("10000")

        # Step 2: Record fills
        engine.record_fill(
            token_id="btc_yes_123",
            side="buy",
            size=Decimal("200"),
            price=Decimal("0.52"),
            fee=Decimal("2.00"),
            market_slug="will-btc-hit-100k",
            market_question="Will BTC hit $100k?",
        )
        engine.record_fill(
            token_id="btc_yes_123",
            side="sell",
            size=Decimal("100"),
            price=Decimal("0.58"),
            fee=Decimal("1.00"),
            market_slug="will-btc-hit-100k",
        )

        # Step 3: Verify positions
        positions = engine.get_positions()
        assert "btc_yes_123" in positions
        pos = positions["btc_yes_123"]
        assert pos.net_size == Decimal("100")
        assert pos.realized_pnl > 0  # Profitable sale

        # Step 4: Compute equity
        orderbooks = {
            "btc_yes_123": OrderBook(
                token_id="btc_yes_123",
                bids=[BookLevel(price=Decimal("0.60"), size=Decimal("500"))],
                asks=[BookLevel(price=Decimal("0.62"), size=Decimal("500"))],
            )
        }
        equity = engine.compute_equity(orderbooks=orderbooks)

        # Cash: 10000 - (200*0.52 + 2) + (100*0.58 - 1) = 10000 - 106 + 57 = 9951
        assert equity.cash_balance == Decimal("9951")
        # Position value: 100 * 0.61 (mid) = 61
        assert equity.mark_to_market == Decimal("61")
        assert equity.open_position_count == 1

        # Step 5: Record equity after trades
        recorded = engine.record_equity(orderbooks=orderbooks)
        assert recorded.net_equity == equity.net_equity

        # Step 6: Load and verify equity curve
        curve = engine.load_equity_curve()
        assert len(curve) == 2  # Initial + after trades
        assert curve[0].net_equity == Decimal("10000")  # Starting equity
        assert curve[1].net_equity == equity.net_equity

        # Step 7: Get summary
        summary = engine.get_equity_curve_summary()
        assert summary["data_points"] == 2
        assert summary["starting_equity"] == 10000.0
        assert summary["current_equity"] == float(equity.net_equity)

    def test_rebuild_from_fills(self, tmp_path: Path):
        """Test state reconstruction from fill journal."""
        engine = PaperTradingEngine(data_dir=tmp_path, starting_cash=Decimal("10000"))

        # Record fills
        engine.record_fill("token1", "buy", Decimal("100"), Decimal("0.50"))
        engine.record_fill("token1", "sell", Decimal("50"), Decimal("0.60"))

        # Create new engine instance (simulating restart)
        engine2 = PaperTradingEngine(data_dir=tmp_path, starting_cash=Decimal("10000"))
        assert not engine2._fills_loaded

        # Rebuild state
        engine2.rebuild_state_from_fills()
        assert engine2._fills_loaded

        # Verify positions match
        positions = engine2.get_positions()
        assert positions["token1"].net_size == Decimal("50")

    def test_multiple_markets(self, tmp_path: Path):
        """Test tracking positions across multiple markets."""
        engine = PaperTradingEngine(data_dir=tmp_path, starting_cash=Decimal("10000"))

        # Trade in multiple markets
        markets = [
            ("btc_yes", "btc-market", Decimal("100"), Decimal("0.55")),
            ("eth_yes", "eth-market", Decimal("200"), Decimal("0.45")),
            ("sol_yes", "sol-market", Decimal("50"), Decimal("0.60")),
        ]

        for token, slug, size, price in markets:
            engine.record_fill(token, "buy", size, price, market_slug=slug)

        positions = engine.get_positions()
        assert len(positions) == 3

        # Create orderbooks for all positions
        orderbooks = {}
        for token, _, _, _ in markets:
            orderbooks[token] = OrderBook(
                token_id=token,
                bids=[BookLevel(price=Decimal("0.58"), size=Decimal("1000"))],
                asks=[BookLevel(price=Decimal("0.62"), size=Decimal("1000"))],
            )

        equity = engine.compute_equity(orderbooks=orderbooks)
        assert equity.position_count == 3
        assert equity.open_position_count == 3

    def test_cash_balance_tracking(self, tmp_path: Path):
        """Test that cash balance is correctly tracked through fills."""
        engine = PaperTradingEngine(data_dir=tmp_path, starting_cash=Decimal("1000"))

        # Buy: cash decreases
        engine.record_fill("token1", "buy", Decimal("100"), Decimal("0.50"), fee=Decimal("1.00"))
        # Cost = 100 * 0.50 + 1 = 51
        assert engine._verifier.cash_balance == Decimal("949")

        # Sell: cash increases
        engine.record_fill("token1", "sell", Decimal("50"), Decimal("0.60"), fee=Decimal("0.50"))
        # Proceeds = 50 * 0.60 - 0.50 = 29.50
        assert engine._verifier.cash_balance == Decimal("978.50")
