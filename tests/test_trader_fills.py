"""Tests for trader fills module."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from polymarket.trader_fills import (
    TraderAccounting,
    TraderFill,
    TraderFillTracker,
    TraderNAVSnapshot,
    TraderPosition,
)


class TestTraderFill:
    """Tests for TraderFill dataclass."""

    def test_creation(self) -> None:
        """Test creating a TraderFill."""
        fill = TraderFill(
            token_id="token123",
            side="buy",
            size=Decimal("100"),
            price=Decimal("0.55"),
            fee=Decimal("0.11"),
            timestamp="2024-01-01T00:00:00+00:00",
            transaction_hash="0xabc",
            market_slug="test-market",
        )

        assert fill.token_id == "token123"
        assert fill.side == "buy"
        assert fill.size == Decimal("100")
        assert fill.transaction_hash == "0xabc"

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        fill = TraderFill(
            token_id="token123",
            side="buy",
            size=Decimal("100"),
            price=Decimal("0.55"),
            fee=Decimal("0.11"),
            timestamp="2024-01-01T00:00:00+00:00",
        )

        data = fill.to_dict()

        assert data["token_id"] == "token123"
        assert data["side"] == "buy"
        assert data["size"] == "100"
        assert data["price"] == "0.55"

    def test_from_dict(self) -> None:
        """Test creation from dictionary."""
        data = {
            "token_id": "token123",
            "side": "buy",
            "size": "100",
            "price": "0.55",
            "fee": "0.11",
            "timestamp": "2024-01-01T00:00:00+00:00",
            "transaction_hash": "0xabc",
            "market_slug": "test-market",
        }

        fill = TraderFill.from_dict(data)

        assert fill.token_id == "token123"
        assert fill.size == Decimal("100")
        assert fill.price == Decimal("0.55")

    def test_cash_flow_buy(self) -> None:
        """Test cash flow calculation for buy."""
        fill = TraderFill(
            token_id="token123",
            side="buy",
            size=Decimal("100"),
            price=Decimal("0.55"),
            fee=Decimal("0.11"),
            timestamp="2024-01-01T00:00:00+00:00",
        )

        # Buy: -(100 * 0.55 + 0.11) = -55.11
        assert fill.cash_flow == Decimal("-55.11")

    def test_cash_flow_sell(self) -> None:
        """Test cash flow calculation for sell."""
        fill = TraderFill(
            token_id="token123",
            side="sell",
            size=Decimal("100"),
            price=Decimal("0.60"),
            fee=Decimal("0.12"),
            timestamp="2024-01-01T00:00:00+00:00",
        )

        # Sell: 100 * 0.60 - 0.12 = 59.88
        assert fill.cash_flow == Decimal("59.88")

    def test_to_pnl_fill(self) -> None:
        """Test conversion to PnL Fill."""
        fill = TraderFill(
            token_id="token123",
            side="buy",
            size=Decimal("100"),
            price=Decimal("0.55"),
            fee=Decimal("0.11"),
            timestamp="2024-01-01T00:00:00+00:00",
        )

        pnl_fill = fill.to_pnl_fill()

        assert pnl_fill.token_id == "token123"
        assert pnl_fill.side == "buy"
        assert pnl_fill.size == Decimal("100")


class TestTraderPosition:
    """Tests for TraderPosition dataclass."""

    def test_creation(self) -> None:
        """Test creating a TraderPosition."""
        pos = TraderPosition(
            token_id="token123",
            trader_address="0xabc",
            market_slug="test-market",
        )

        assert pos.token_id == "token123"
        assert pos.trader_address == "0xabc"
        assert pos.net_size == Decimal("0")

    def test_add_buy(self) -> None:
        """Test adding a buy fill."""
        pos = TraderPosition(token_id="token123", trader_address="0xabc")

        pos.add_buy(
            size=Decimal("100"),
            price=Decimal("0.55"),
            fee=Decimal("0.11"),
            timestamp="2024-01-01T00:00:00+00:00",
        )

        assert pos.net_size == Decimal("100")
        assert pos.total_cost == Decimal("55")  # 100 * 0.55
        assert pos.buy_count == 1

    def test_add_sell_realized_pnl(self) -> None:
        """Test realized PnL calculation on sell."""
        pos = TraderPosition(token_id="token123", trader_address="0xabc")

        # Buy at 0.55
        pos.add_buy(
            size=Decimal("100"),
            price=Decimal("0.55"),
            fee=Decimal("0.11"),
            timestamp="2024-01-01T00:00:00+00:00",
        )

        # Sell at 0.60
        realized = pos.add_sell(
            size=Decimal("100"),
            price=Decimal("0.60"),
            fee=Decimal("0.12"),
            timestamp="2024-01-01T01:00:00+00:00",
        )

        # Realized: proceeds - cost_basis - fee
        # = 60 - 55 - 0.12 = 4.88
        assert realized == Decimal("4.88")
        assert pos.net_size == Decimal("0")
        assert pos.realized_pnl == Decimal("4.88")

    def test_avg_cost_basis(self) -> None:
        """Test average cost basis calculation."""
        pos = TraderPosition(token_id="token123", trader_address="0xabc")

        pos.add_buy(
            size=Decimal("50"),
            price=Decimal("0.50"),
            fee=Decimal("0.05"),
            timestamp="2024-01-01T00:00:00+00:00",
        )
        pos.add_buy(
            size=Decimal("50"),
            price=Decimal("0.60"),
            fee=Decimal("0.06"),
            timestamp="2024-01-01T01:00:00+00:00",
        )

        # Avg cost: (25 + 30) / 100 = 0.55
        assert pos.avg_cost_basis == Decimal("0.55")


class TestTraderAccounting:
    """Tests for TraderAccounting class."""

    def test_creation(self) -> None:
        """Test creating TraderAccounting."""
        acc = TraderAccounting(trader_address="0xabc")

        assert acc.trader_address == "0xabc"
        assert acc.cash_balance == Decimal("0")
        assert len(acc.positions) == 0

    def test_add_fill_updates_cash(self) -> None:
        """Test that add_fill updates cash balance."""
        acc = TraderAccounting(trader_address="0xabc")

        fill = TraderFill(
            token_id="token123",
            side="buy",
            size=Decimal("100"),
            price=Decimal("0.55"),
            fee=Decimal("0.11"),
            timestamp="2024-01-01T00:00:00+00:00",
        )

        acc.add_fill(fill)

        assert acc.cash_balance == Decimal("-55.11")

    def test_add_fill_creates_position(self) -> None:
        """Test that add_fill creates positions."""
        acc = TraderAccounting(trader_address="0xabc")

        fill = TraderFill(
            token_id="token123",
            side="buy",
            size=Decimal("100"),
            price=Decimal("0.55"),
            fee=Decimal("0.11"),
            timestamp="2024-01-01T00:00:00+00:00",
        )

        acc.add_fill(fill)

        assert "token123" in acc.positions
        assert acc.positions["token123"].net_size == Decimal("100")

    def test_compute_nav(self) -> None:
        """Test NAV computation."""
        acc = TraderAccounting(trader_address="0xabc")

        # Add a buy
        acc.add_fill(TraderFill(
            token_id="token123",
            side="buy",
            size=Decimal("100"),
            price=Decimal("0.55"),
            fee=Decimal("0.11"),
            timestamp="2024-01-01T00:00:00+00:00",
        ))

        nav = acc.compute_nav(current_prices={"token123": Decimal("0.60")})

        assert nav.trader_address == "0xabc"
        assert nav.cash_balance == Decimal("-55.11")
        assert nav.positions_value == Decimal("60")  # 100 * 0.60
        assert nav.nav == Decimal("4.89")  # -55.11 + 60
        assert nav.unrealized_pnl == Decimal("5")  # (0.60 - 0.55) * 100


class TestTraderFillTracker:
    """Tests for TraderFillTracker class."""

    def test_init_creates_directories(self, tmp_path: Path) -> None:
        """Test initialization creates directories."""
        _tracker = TraderFillTracker(data_dir=tmp_path)

        assert tmp_path.exists()
        assert (tmp_path / "fills").exists()
        assert (tmp_path / "nav").exists()

    def test_save_and_load_fills(self, tmp_path: Path) -> None:
        """Test saving and loading fills."""
        tracker = TraderFillTracker(data_dir=tmp_path)
        address = "0xabc"

        fill = TraderFill(
            token_id="token123",
            side="buy",
            size=Decimal("100"),
            price=Decimal("0.55"),
            fee=Decimal("0.11"),
            timestamp="2024-01-01T00:00:00+00:00",
        )

        tracker.save_fill(address, fill)
        loaded = tracker.load_fills(address)

        assert len(loaded) == 1
        assert loaded[0].token_id == "token123"
        assert loaded[0].size == Decimal("100")

    def test_load_accounting(self, tmp_path: Path) -> None:
        """Test loading accounting state."""
        tracker = TraderFillTracker(data_dir=tmp_path)
        address = "0xabc"

        # Save some fills
        fill1 = TraderFill(
            token_id="token123",
            side="buy",
            size=Decimal("100"),
            price=Decimal("0.55"),
            fee=Decimal("0.11"),
            timestamp="2024-01-01T00:00:00+00:00",
        )
        fill2 = TraderFill(
            token_id="token123",
            side="sell",
            size=Decimal("50"),
            price=Decimal("0.60"),
            fee=Decimal("0.06"),
            timestamp="2024-01-01T01:00:00+00:00",
        )

        tracker.save_fill(address, fill1)
        tracker.save_fill(address, fill2)

        acc = tracker.load_accounting(address)

        assert acc.trader_address == address
        assert acc.cash_balance == Decimal("-25.17")  # -55.11 + 29.94
        assert acc.positions["token123"].net_size == Decimal("50")

    def test_record_and_load_nav(self, tmp_path: Path) -> None:
        """Test recording and loading NAV snapshots."""
        tracker = TraderFillTracker(data_dir=tmp_path)
        address = "0xabc"

        # Add a fill first
        tracker.save_fill(address, TraderFill(
            token_id="token123",
            side="buy",
            size=Decimal("100"),
            price=Decimal("0.55"),
            fee=Decimal("0.11"),
            timestamp="2024-01-01T00:00:00+00:00",
        ))

        # Record NAV
        _snapshot = tracker.record_nav_snapshot(address, {"token123": Decimal("0.60")})

        # Load history
        history = tracker.get_nav_history(address)

        assert len(history) == 1
        assert history[0].trader_address == address

    def test_get_trader_summary(self, tmp_path: Path) -> None:
        """Test getting trader summary."""
        tracker = TraderFillTracker(data_dir=tmp_path)
        address = "0xabc"

        tracker.save_fill(address, TraderFill(
            token_id="token123",
            side="buy",
            size=Decimal("100"),
            price=Decimal("0.55"),
            fee=Decimal("0.11"),
            timestamp="2024-01-01T00:00:00+00:00",
        ))

        summary = tracker.get_trader_summary(address)

        assert summary["address"] == address
        assert summary["total_fills"] == 1
        assert summary["open_positions"] == 1


class TestTraderNAVSnapshot:
    """Tests for TraderNAVSnapshot dataclass."""

    def test_creation(self) -> None:
        """Test creating a snapshot."""
        snapshot = TraderNAVSnapshot(
            trader_address="0xabc",
            timestamp="2024-01-01T00:00:00+00:00",
            cash_balance=Decimal("-55.11"),
            positions_value=Decimal("60"),
            realized_pnl=Decimal("0"),
            unrealized_pnl=Decimal("5"),
            total_fees=Decimal("0.11"),
            nav=Decimal("4.89"),
            position_count=1,
            open_position_count=1,
        )

        assert snapshot.trader_address == "0xabc"
        assert snapshot.nav == Decimal("4.89")

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        snapshot = TraderNAVSnapshot(
            trader_address="0xabc",
            timestamp="2024-01-01T00:00:00+00:00",
            cash_balance=Decimal("-55.11"),
            positions_value=Decimal("60"),
            realized_pnl=Decimal("0"),
            unrealized_pnl=Decimal("5"),
            total_fees=Decimal("0.11"),
            nav=Decimal("4.89"),
            position_count=1,
            open_position_count=1,
        )

        data = snapshot.to_dict()

        assert data["trader_address"] == "0xabc"
        assert data["nav"] == "4.89"

    def test_from_dict(self) -> None:
        """Test creation from dictionary."""
        data = {
            "trader_address": "0xabc",
            "timestamp": "2024-01-01T00:00:00+00:00",
            "cash_balance": "-55.11",
            "positions_value": "60",
            "realized_pnl": "0",
            "unrealized_pnl": "5",
            "total_fees": "0.11",
            "nav": "4.89",
            "position_count": 1,
            "open_position_count": 1,
        }

        snapshot = TraderNAVSnapshot.from_dict(data)

        assert snapshot.trader_address == "0xabc"
        assert snapshot.nav == Decimal("4.89")
