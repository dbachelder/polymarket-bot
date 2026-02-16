"""Tests for the accounting module."""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from polymarket.accounting import (
    AccountingDB,
    AccountingFill,
    init_accounting_db,
)


class TestAccountingFill:
    """Tests for AccountingFill dataclass."""

    def test_fill_creation(self) -> None:
        """Test creating an AccountingFill."""
        fill = AccountingFill(
            fill_id="test-001",
            timestamp=datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC),
            token_id="token-123",
            side="buy",
            size=Decimal("100"),
            price=Decimal("0.55"),
            fee=Decimal("0.11"),
            cash_flow=Decimal("-55.11"),
            market_slug="test-market",
            trader_source="0xabc",
        )

        assert fill.fill_id == "test-001"
        assert fill.token_id == "token-123"
        assert fill.side == "buy"
        assert fill.size == Decimal("100")
        assert fill.cash_flow == Decimal("-55.11")

    def test_fill_to_dict(self) -> None:
        """Test fill serialization to dict."""
        fill = AccountingFill(
            fill_id="test-002",
            timestamp=datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC),
            token_id="token-456",
            side="sell",
            size=Decimal("50"),
            price=Decimal("0.60"),
            fee=Decimal("0.06"),
            cash_flow=Decimal("29.94"),
        )

        data = fill.to_dict()
        assert data["fill_id"] == "test-002"
        assert data["side"] == "sell"
        assert data["size"] == "50"
        assert data["cash_flow"] == "29.94"

    def test_fill_from_dict(self) -> None:
        """Test fill deserialization from dict."""
        data = {
            "fill_id": "test-003",
            "timestamp": "2024-01-01T12:00:00+00:00",
            "token_id": "token-789",
            "side": "buy",
            "size": "75",
            "price": "0.45",
            "fee": "0.0675",
            "cash_flow": "-33.8175",
            "market_slug": "another-market",
            "trader_source": "0xdef",
            "original_tx_hash": "0xtxhash",
        }

        fill = AccountingFill.from_dict(data)
        assert fill.fill_id == "test-003"
        assert fill.token_id == "token-789"
        assert fill.size == Decimal("75")
        assert fill.market_slug == "another-market"
        assert fill.original_tx_hash == "0xtxhash"


class TestAccountingDB:
    """Tests for AccountingDB."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = AccountingDB(db_path)
            db.init_schema()
            yield db
            db.close()

    def test_init_schema(self, temp_db: AccountingDB) -> None:
        """Test database schema initialization."""
        conn = temp_db._get_connection()

        # Check tables exist
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = {t["name"] for t in tables}

        assert "fills" in table_names
        assert "positions" in table_names
        assert "cash_ledger" in table_names
        assert "portfolio_snapshots" in table_names
        assert "snapshot_valuations" in table_names

    def test_set_initial_cash(self, temp_db: AccountingDB) -> None:
        """Test setting initial cash balance."""
        temp_db.set_initial_cash(Decimal("10000"))

        balance = temp_db.get_cash_balance()
        assert balance == Decimal("10000")

    def test_record_fill_buy(self, temp_db: AccountingDB) -> None:
        """Test recording a buy fill."""
        temp_db.set_initial_cash(Decimal("10000"))

        fill = AccountingFill(
            fill_id="fill-001",
            timestamp=datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC),
            token_id="token-yes-123",
            side="buy",
            size=Decimal("100"),
            price=Decimal("0.55"),
            fee=Decimal("0.11"),
            cash_flow=Decimal("-55.11"),
        )

        temp_db.record_fill(fill)

        # Check cash balance decreased
        assert temp_db.get_cash_balance() == Decimal("10000") - Decimal("55.11")

        # Check position was created
        positions = temp_db.get_positions(open_only=True)
        assert len(positions) == 1
        assert positions[0].token_id == "token-yes-123"
        assert positions[0].net_size == Decimal("100")
        assert positions[0].avg_cost_basis == Decimal("0.55")

    def test_record_fill_sell(self, temp_db: AccountingDB) -> None:
        """Test recording a sell fill with realized PnL."""
        temp_db.set_initial_cash(Decimal("10000"))

        # First buy
        buy_fill = AccountingFill(
            fill_id="fill-002",
            timestamp=datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC),
            token_id="token-yes-456",
            side="buy",
            size=Decimal("100"),
            price=Decimal("0.50"),
            fee=Decimal("0.10"),
            cash_flow=Decimal("-50.10"),
        )
        temp_db.record_fill(buy_fill)

        # Then sell at profit
        sell_fill = AccountingFill(
            fill_id="fill-003",
            timestamp=datetime(2024, 1, 2, 12, 0, 0, tzinfo=UTC),
            token_id="token-yes-456",
            side="sell",
            size=Decimal("100"),
            price=Decimal("0.60"),
            fee=Decimal("0.12"),
            cash_flow=Decimal("59.88"),
        )
        temp_db.record_fill(sell_fill)

        # Check realized PnL
        positions = temp_db.get_positions(open_only=False)
        assert len(positions) == 1
        assert positions[0].net_size == Decimal("0")
        # Realized PnL = (0.60 - 0.50) * 100 - 0.12 = 10 - 0.12 = 9.88
        assert positions[0].realized_pnl == Decimal("9.88")

    def test_record_partial_sell(self, temp_db: AccountingDB) -> None:
        """Test recording a partial sell (closing part of position)."""
        temp_db.set_initial_cash(Decimal("10000"))

        # Buy 100 shares
        buy_fill = AccountingFill(
            fill_id="fill-004",
            timestamp=datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC),
            token_id="token-yes-789",
            side="buy",
            size=Decimal("100"),
            price=Decimal("0.50"),
            fee=Decimal("0.10"),
            cash_flow=Decimal("-50.10"),
        )
        temp_db.record_fill(buy_fill)

        # Sell 60 shares at profit
        sell_fill = AccountingFill(
            fill_id="fill-005",
            timestamp=datetime(2024, 1, 2, 12, 0, 0, tzinfo=UTC),
            token_id="token-yes-789",
            side="sell",
            size=Decimal("60"),
            price=Decimal("0.60"),
            fee=Decimal("0.072"),
            cash_flow=Decimal("35.928"),
        )
        temp_db.record_fill(sell_fill)

        # Check position
        positions = temp_db.get_positions(open_only=True)
        assert len(positions) == 1
        assert positions[0].net_size == Decimal("40")  # 100 - 60
        # Realized PnL on 60 shares = (0.60 - 0.50) * 60 - 0.072 = 6 - 0.072 = 5.928
        assert positions[0].realized_pnl == Decimal("5.928")

    def test_get_fills_filtering(self, temp_db: AccountingDB) -> None:
        """Test fill retrieval with filters."""
        temp_db.set_initial_cash(Decimal("10000"))

        base_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)

        # Record fills for different tokens
        for i in range(5):
            fill = AccountingFill(
                fill_id=f"fill-{i}",
                timestamp=base_time + timedelta(hours=i),
                token_id=f"token-{i % 2}",  # Alternates between token-0 and token-1
                side="buy",
                size=Decimal("10"),
                price=Decimal("0.5"),
                fee=Decimal("0.01"),
                cash_flow=Decimal("-5.01"),
            )
            temp_db.record_fill(fill)

        # Test since filter
        fills = temp_db.get_fills(since=base_time + timedelta(hours=2))
        assert len(fills) == 3  # fills at hours 2, 3, 4

        # Test token filter
        fills = temp_db.get_fills(token_id="token-0")
        assert len(fills) == 3  # fills 0, 2, 4

        # Test limit
        fills = temp_db.get_fills(limit=2)
        assert len(fills) == 2

    def test_cash_ledger(self, temp_db: AccountingDB) -> None:
        """Test cash ledger tracking."""
        # Use consistent timestamps
        base_time = datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC)
        temp_db.set_initial_cash(Decimal("10000"), timestamp=base_time)

        fill_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        fill = AccountingFill(
            fill_id="fill-ledger",
            timestamp=fill_time,
            token_id="token-abc",
            side="buy",
            size=Decimal("100"),
            price=Decimal("0.50"),
            fee=Decimal("0.10"),
            cash_flow=Decimal("-50.10"),
        )
        temp_db.record_fill(fill)

        # Check ledger entries (in descending order by timestamp)
        entries = temp_db.get_cash_ledger()
        assert len(entries) == 2  # initial + fill

        # Fill entry is first (most recent - 12:00 > 10:00)
        assert entries[0].entry_type == "fill"
        assert entries[0].amount == Decimal("-50.10")
        assert entries[0].balance_after == Decimal("9949.90")

        # Initial entry is second (older - 10:00)
        assert entries[1].entry_type == "initial"
        assert entries[1].balance_after == Decimal("10000")

    def test_record_portfolio_snapshot(self, temp_db: AccountingDB) -> None:
        """Test portfolio snapshot recording."""
        temp_db.set_initial_cash(Decimal("10000"))

        # Create a position
        fill = AccountingFill(
            fill_id="fill-snap",
            timestamp=datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC),
            token_id="token-yes-snap",
            side="buy",
            size=Decimal("100"),
            price=Decimal("0.50"),
            fee=Decimal("0.10"),
            cash_flow=Decimal("-50.10"),
        )
        temp_db.record_fill(fill)

        # Record snapshot with price at 0.60
        prices = {"token-yes-snap": Decimal("0.60")}
        snapshot = temp_db.record_portfolio_snapshot(
            prices=prices,
            timestamp=datetime(2024, 1, 1, 13, 0, 0, tzinfo=UTC),
        )

        assert snapshot.cash_balance == Decimal("9949.90")
        assert snapshot.positions_value == Decimal("60")  # 100 shares @ 0.60
        assert snapshot.unrealized_pnl == Decimal("10")  # (0.60 - 0.50) * 100
        assert snapshot.nav == Decimal("10009.90")

    def test_get_latest_snapshot(self, temp_db: AccountingDB) -> None:
        """Test retrieving latest snapshot."""
        temp_db.set_initial_cash(Decimal("10000"))

        # Record two snapshots
        for hour in [12, 13]:
            fill = AccountingFill(
                fill_id=f"fill-snap-{hour}",
                timestamp=datetime(2024, 1, 1, hour, 0, 0, tzinfo=UTC),
                token_id="token-yes",
                side="buy",
                size=Decimal("10"),
                price=Decimal("0.50"),
                fee=Decimal("0.01"),
                cash_flow=Decimal("-5.01"),
            )
            temp_db.record_fill(fill)

            prices = {"token-yes": Decimal(f"0.{50 + hour}")}  # 0.62, 0.63
            temp_db.record_portfolio_snapshot(
                prices=prices,
                timestamp=datetime(2024, 1, 1, hour, 30, 0, tzinfo=UTC),
            )

        latest = temp_db.get_latest_snapshot()
        assert latest is not None
        assert latest.snapshot_time.hour == 13

    def test_get_account_summary(self, temp_db: AccountingDB) -> None:
        """Test account summary calculation."""
        temp_db.set_initial_cash(Decimal("10000"))

        # Record some activity
        for i in range(3):
            fill = AccountingFill(
                fill_id=f"fill-sum-{i}",
                timestamp=datetime.now(UTC) - timedelta(hours=i),
                token_id="token-yes",
                side="buy",
                size=Decimal("10"),
                price=Decimal("0.50"),
                fee=Decimal("0.01"),
                cash_flow=Decimal("-5.01"),
            )
            temp_db.record_fill(fill)

        summary = temp_db.get_account_summary(days=7)

        assert summary["period_days"] == 7
        assert summary["total_fills"] == 3
        assert summary["open_positions"] == 1
        assert summary["cash_balance"] < 10000  # Spent on fills

    def test_get_exposures(self, temp_db: AccountingDB) -> None:
        """Test exposure breakdown calculation."""
        temp_db.set_initial_cash(Decimal("10000"))

        # Create multiple positions
        for i, size in enumerate([100, 200]):
            fill = AccountingFill(
                fill_id=f"fill-exp-{i}",
                timestamp=datetime(2024, 1, 1, 12, i, 0, tzinfo=UTC),
                token_id=f"token-yes-{i}",
                side="buy",
                size=Decimal(str(size)),
                price=Decimal("0.50"),
                fee=Decimal("0.1"),
                cash_flow=Decimal(f"-{size * 0.5 + 0.1:.2f}"),
            )
            temp_db.record_fill(fill)

        # Record snapshot to get NAV
        prices = {
            "token-yes-0": Decimal("0.55"),
            "token-yes-1": Decimal("0.52"),
        }
        temp_db.record_portfolio_snapshot(prices=prices)

        exposures = temp_db.get_exposures()

        assert len(exposures) == 2
        # First position should have higher NAV % due to larger size
        assert exposures[0]["nav_pct"] > exposures[1]["nav_pct"]


class TestPositionCalculations:
    """Tests for position-level calculations."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = AccountingDB(db_path)
            db.init_schema()
            yield db
            db.close()

    def test_avg_cost_basis_calculation(self, temp_db: AccountingDB) -> None:
        """Test average cost basis with multiple buys."""
        temp_db.set_initial_cash(Decimal("10000"))

        # Buy 100 @ 0.50
        fill1 = AccountingFill(
            fill_id="fill-cost-1",
            timestamp=datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC),
            token_id="token-yes",
            side="buy",
            size=Decimal("100"),
            price=Decimal("0.50"),
            fee=Decimal("0.1"),
            cash_flow=Decimal("-50.10"),
        )
        temp_db.record_fill(fill1)

        # Buy 100 @ 0.60
        fill2 = AccountingFill(
            fill_id="fill-cost-2",
            timestamp=datetime(2024, 1, 2, 12, 0, 0, tzinfo=UTC),
            token_id="token-yes",
            side="buy",
            size=Decimal("100"),
            price=Decimal("0.60"),
            fee=Decimal("0.12"),
            cash_flow=Decimal("-60.12"),
        )
        temp_db.record_fill(fill2)

        pos = temp_db.get_position("token-yes")
        assert pos is not None
        assert pos.net_size == Decimal("200")
        # Avg cost = (100*0.50 + 100*0.60) / 200 = 0.55
        assert pos.avg_cost_basis == Decimal("0.55")

    def test_realized_pnl_with_multiple_sells(self, temp_db: AccountingDB) -> None:
        """Test realized PnL with multiple partial sells."""
        temp_db.set_initial_cash(Decimal("10000"))

        # Buy 100 @ 0.50
        temp_db.record_fill(AccountingFill(
            fill_id="fill-multi-1",
            timestamp=datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC),
            token_id="token-yes",
            side="buy",
            size=Decimal("100"),
            price=Decimal("0.50"),
            fee=Decimal("0.1"),
            cash_flow=Decimal("-50.10"),
        ))

        # Sell 50 @ 0.60 (profit $5 on 50 shares, minus fee)
        temp_db.record_fill(AccountingFill(
            fill_id="fill-multi-2",
            timestamp=datetime(2024, 1, 2, 12, 0, 0, tzinfo=UTC),
            token_id="token-yes",
            side="sell",
            size=Decimal("50"),
            price=Decimal("0.60"),
            fee=Decimal("0.06"),
            cash_flow=Decimal("29.94"),
        ))

        # Sell remaining 50 @ 0.55 (profit $2.50 on 50 shares, minus fee)
        temp_db.record_fill(AccountingFill(
            fill_id="fill-multi-3",
            timestamp=datetime(2024, 1, 3, 12, 0, 0, tzinfo=UTC),
            token_id="token-yes",
            side="sell",
            size=Decimal("50"),
            price=Decimal("0.55"),
            fee=Decimal("0.055"),
            cash_flow=Decimal("27.445"),
        ))

        pos = temp_db.get_position("token-yes")
        assert pos is not None
        assert pos.net_size == Decimal("0")
        # Realized PnL = (0.60 - 0.50) * 50 - 0.06 + (0.55 - 0.50) * 50 - 0.055
        #              = 5 - 0.06 + 2.5 - 0.055 = 7.385
        assert pos.realized_pnl == Decimal("7.385")


class TestCannedFills:
    """Tests with predefined canned fill scenarios."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = AccountingDB(db_path)
            db.init_schema()
            yield db
            db.close()

    def test_scenario_simple_profit(self, temp_db: AccountingDB) -> None:
        """Simple profit scenario: buy low, sell high."""
        temp_db.set_initial_cash(Decimal("1000"))

        fills = [
            AccountingFill(
                fill_id="profit-1",
                timestamp=datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC),
                token_id="btc-up-yes",
                side="buy",
                size=Decimal("100"),
                price=Decimal("0.40"),
                fee=Decimal("0.08"),
                cash_flow=Decimal("-40.08"),
                market_slug="btc-up",
            ),
            AccountingFill(
                fill_id="profit-2",
                timestamp=datetime(2024, 1, 2, 12, 0, 0, tzinfo=UTC),
                token_id="btc-up-yes",
                side="sell",
                size=Decimal("100"),
                price=Decimal("0.60"),
                fee=Decimal("0.12"),
                cash_flow=Decimal("59.88"),
                market_slug="btc-up",
            ),
        ]

        for fill in fills:
            temp_db.record_fill(fill)

        # Verify final state
        assert temp_db.get_cash_balance() == Decimal("1019.80")  # 1000 - 40.08 + 59.88

        positions = temp_db.get_positions(open_only=False)
        assert len(positions) == 1
        # Realized PnL = (0.60-0.40)*100 - 0.12 (sell fee only, buy fee tracked separately)
        assert positions[0].realized_pnl == Decimal("19.88")

    def test_scenario_loss_cut(self, temp_db: AccountingDB) -> None:
        """Loss cutting scenario: buy high, sell low."""
        temp_db.set_initial_cash(Decimal("1000"))

        fills = [
            AccountingFill(
                fill_id="loss-1",
                timestamp=datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC),
                token_id="eth-up-yes",
                side="buy",
                size=Decimal("100"),
                price=Decimal("0.70"),
                fee=Decimal("0.14"),
                cash_flow=Decimal("-70.14"),
                market_slug="eth-up",
            ),
            AccountingFill(
                fill_id="loss-2",
                timestamp=datetime(2024, 1, 2, 12, 0, 0, tzinfo=UTC),
                token_id="eth-up-yes",
                side="sell",
                size=Decimal("100"),
                price=Decimal("0.55"),
                fee=Decimal("0.11"),
                cash_flow=Decimal("54.89"),
                market_slug="eth-up",
            ),
        ]

        for fill in fills:
            temp_db.record_fill(fill)

        positions = temp_db.get_positions(open_only=False)
        # Realized PnL = (0.55 - 0.70) * 100 - 0.11 = -15 - 0.11 = -15.11 (sell fee only)
        assert positions[0].realized_pnl == Decimal("-15.11")

    def test_scenario_multiple_markets(self, temp_db: AccountingDB) -> None:
        """Multiple concurrent positions in different markets."""
        temp_db.set_initial_cash(Decimal("10000"))

        fills = [
            # Market 1: BTC up
            AccountingFill(
                fill_id="multi-1",
                timestamp=datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC),
                token_id="btc-up-yes",
                side="buy",
                size=Decimal("100"),
                price=Decimal("0.50"),
                fee=Decimal("0.10"),
                cash_flow=Decimal("-50.10"),
                market_slug="btc-up",
            ),
            # Market 2: ETH up
            AccountingFill(
                fill_id="multi-2",
                timestamp=datetime(2024, 1, 1, 12, 1, 0, tzinfo=UTC),
                token_id="eth-up-yes",
                side="buy",
                size=Decimal("200"),
                price=Decimal("0.30"),
                fee=Decimal("0.12"),
                cash_flow=Decimal("-60.12"),
                market_slug="eth-up",
            ),
            # Close BTC at profit
            AccountingFill(
                fill_id="multi-3",
                timestamp=datetime(2024, 1, 2, 12, 0, 0, tzinfo=UTC),
                token_id="btc-up-yes",
                side="sell",
                size=Decimal("100"),
                price=Decimal("0.65"),
                fee=Decimal("0.13"),
                cash_flow=Decimal("64.87"),
                market_slug="btc-up",
            ),
            # Close ETH at loss
            AccountingFill(
                fill_id="multi-4",
                timestamp=datetime(2024, 1, 2, 12, 1, 0, tzinfo=UTC),
                token_id="eth-up-yes",
                side="sell",
                size=Decimal("200"),
                price=Decimal("0.25"),
                fee=Decimal("0.10"),
                cash_flow=Decimal("49.90"),
                market_slug="eth-up",
            ),
        ]

        for fill in fills:
            temp_db.record_fill(fill)

        # Check positions
        btc_pos = temp_db.get_position("btc-up-yes")
        eth_pos = temp_db.get_position("eth-up-yes")

        # BTC: bought @ 0.50, sold @ 0.65 = $15 profit - $0.13 sell fee = $14.87
        assert btc_pos.realized_pnl == Decimal("14.87")

        # ETH: bought @ 0.30, sold @ 0.25 = -$10 loss - $0.10 sell fee = -$10.10
        assert eth_pos.realized_pnl == Decimal("-10.10")

        # Net PnL = $4.77
        summary = temp_db.get_account_summary(days=7)
        assert summary["realized_pnl"] == pytest.approx(4.77, rel=0.01)

    def test_scenario_copy_trading(self, temp_db: AccountingDB) -> None:
        """Copy trading scenario with trader attribution."""
        temp_db.set_initial_cash(Decimal("10000"))

        fills = [
            AccountingFill(
                fill_id="copy-1",
                timestamp=datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC),
                token_id="market-a-yes",
                side="buy",
                size=Decimal("50"),
                price=Decimal("0.45"),
                fee=Decimal("0.045"),
                cash_flow=Decimal("-22.545"),
                trader_source="0xtrader1",
                original_tx_hash="0xabc123",
            ),
            AccountingFill(
                fill_id="copy-2",
                timestamp=datetime(2024, 1, 1, 12, 5, 0, tzinfo=UTC),
                token_id="market-b-yes",
                side="buy",
                size=Decimal("30"),
                price=Decimal("0.55"),
                fee=Decimal("0.033"),
                cash_flow=Decimal("-16.533"),
                trader_source="0xtrader2",
                original_tx_hash="0xdef456",
            ),
        ]

        for fill in fills:
            temp_db.record_fill(fill)

        # Check trader filtering
        trader1_fills = temp_db.get_fills(trader_source="0xtrader1")
        assert len(trader1_fills) == 1
        assert trader1_fills[0].trader_source == "0xtrader1"
        assert trader1_fills[0].original_tx_hash == "0xabc123"


class TestInitFunction:
    """Tests for the init_accounting_db convenience function."""

    def test_init_accounting_db(self) -> None:
        """Test the convenience init function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "accounting.db"
            db = init_accounting_db(db_path)

            # Verify it created the schema
            conn = db._get_connection()
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            table_names = {t["name"] for t in tables}

            assert "fills" in table_names
            assert "positions" in table_names

            db.close()

    def test_init_accounting_db_default_path(self) -> None:
        """Test init with default path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            import os
            original_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                db = init_accounting_db()
                assert db.db_path.name == "accounting.db"
                db.close()
            finally:
                os.chdir(original_cwd)
