"""Tests for copy trading module."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from polymarket.copy_trading import (
    CopyEquitySnapshot,
    CopyFill,
    CopyPosition,
    CopyTradeConfig,
    PaperCopyEngine,
    SlippageModel,
)
from polymarket.trader_fills import TraderFill


class TestSlippageModel:
    """Tests for SlippageModel class."""

    def test_creation(self) -> None:
        """Test creating slippage model."""
        model = SlippageModel(
            base_slippage_bps=10,
            spread_impact_bps=50,
            size_impact_factor=Decimal("0.0001"),
        )

        assert model.base_slippage_bps == 10
        assert model.spread_impact_bps == 50

    def test_calculate_slippage_base_only(self) -> None:
        """Test slippage calculation with base only."""
        model = SlippageModel(base_slippage_bps=10)

        slippage = model.calculate_slippage(
            original_price=Decimal("0.55"),
            size=Decimal("100"),
        )

        # 10 bps of 0.55 = 0.00055
        expected = Decimal("0.55") * Decimal("0.001")
        assert abs(slippage - expected) < Decimal("0.00001")

    def test_calculate_slippage_with_spread(self) -> None:
        """Test slippage with spread impact."""
        model = SlippageModel(
            base_slippage_bps=10,
            spread_impact_bps=50,
        )

        # 2% spread = 0.02 at price 0.55
        spread = Decimal("0.02")
        slippage = model.calculate_slippage(
            original_price=Decimal("0.55"),
            size=Decimal("100"),
            spread=spread,
        )

        # Should be more than base only
        base_only = Decimal("0.55") * Decimal("0.001")
        assert slippage > base_only

    def test_adjust_price_buy(self) -> None:
        """Test price adjustment for buy."""
        model = SlippageModel(base_slippage_bps=10)

        price = model.adjust_price_for_copy(
            side="buy",
            original_price=Decimal("0.55"),
            size=Decimal("100"),
        )

        # Buy price should be higher (worse)
        assert price > Decimal("0.55")
        assert price < Decimal("0.99")  # Capped

    def test_adjust_price_sell(self) -> None:
        """Test price adjustment for sell."""
        model = SlippageModel(base_slippage_bps=10)

        price = model.adjust_price_for_copy(
            side="sell",
            original_price=Decimal("0.55"),
            size=Decimal("100"),
        )

        # Sell price should be lower (worse)
        assert price < Decimal("0.55")
        assert price > Decimal("0.01")  # Capped

    def test_max_slippage_cap(self) -> None:
        """Test that slippage is capped."""
        model = SlippageModel(
            base_slippage_bps=10,
            max_slippage_bps=100,  # 1% max
        )

        # Very large size should hit cap
        slippage = model.calculate_slippage(
            original_price=Decimal("0.55"),
            size=Decimal("1000000"),
        )

        max_slippage = Decimal("0.55") * Decimal("0.01")  # 1% of price
        assert slippage <= max_slippage


class TestCopyFill:
    """Tests for CopyFill dataclass."""

    def test_creation(self) -> None:
        """Test creating a CopyFill."""
        fill = CopyFill(
            copy_fill_id="copy_001",
            timestamp="2024-01-01T00:00:00+00:00",
            token_id="token123",
            side="buy",
            size=Decimal("100"),
            copy_price=Decimal("0.56"),
            fee=Decimal("0.112"),
            original_trader="0xabc",
            original_tx_hash="0xoriginal",
            original_price=Decimal("0.55"),
            original_timestamp="2024-01-01T00:00:00+00:00",
        )

        assert fill.copy_fill_id == "copy_001"
        assert fill.original_trader == "0xabc"
        assert fill.copy_price == Decimal("0.56")

    def test_cash_flow_buy(self) -> None:
        """Test cash flow for buy."""
        fill = CopyFill(
            copy_fill_id="copy_001",
            timestamp="2024-01-01T00:00:00+00:00",
            token_id="token123",
            side="buy",
            size=Decimal("100"),
            copy_price=Decimal("0.56"),
            fee=Decimal("0.112"),
            original_trader="0xabc",
            original_tx_hash="0xoriginal",
            original_price=Decimal("0.55"),
            original_timestamp="2024-01-01T00:00:00+00:00",
        )

        # Buy: -(100 * 0.56 + 0.112) = -56.112
        assert fill.cash_flow == Decimal("-56.112")


class TestCopyPosition:
    """Tests for CopyPosition class."""

    def test_creation(self) -> None:
        """Test creating a position."""
        pos = CopyPosition(token_id="token123")

        assert pos.token_id == "token123"
        assert pos.net_size == Decimal("0")

    def test_add_buy_fill(self) -> None:
        """Test adding a buy fill."""
        pos = CopyPosition(token_id="token123")

        fill = CopyFill(
            copy_fill_id="copy_001",
            timestamp="2024-01-01T00:00:00+00:00",
            token_id="token123",
            side="buy",
            size=Decimal("100"),
            copy_price=Decimal("0.56"),
            fee=Decimal("0.112"),
            original_trader="0xabc",
            original_tx_hash="0xoriginal",
            original_price=Decimal("0.55"),
            original_timestamp="2024-01-01T00:00:00+00:00",
        )

        pos.add_fill(fill)

        assert pos.net_size == Decimal("100")
        assert pos.copy_count == 1

    def test_add_sell_fill_realized_pnl(self) -> None:
        """Test realized PnL on sell."""
        pos = CopyPosition(token_id="token123")

        # Buy at 0.55
        pos.add_fill(CopyFill(
            copy_fill_id="copy_001",
            timestamp="2024-01-01T00:00:00+00:00",
            token_id="token123",
            side="buy",
            size=Decimal("100"),
            copy_price=Decimal("0.55"),
            fee=Decimal("0.11"),
            original_trader="0xabc",
            original_tx_hash="0xoriginal",
            original_price=Decimal("0.55"),
            original_timestamp="2024-01-01T00:00:00+00:00",
        ))

        # Sell at 0.60
        pos.add_fill(CopyFill(
            copy_fill_id="copy_002",
            timestamp="2024-01-01T01:00:00+00:00",
            token_id="token123",
            side="sell",
            size=Decimal("100"),
            copy_price=Decimal("0.60"),
            fee=Decimal("0.12"),
            original_trader="0xabc",
            original_tx_hash="0xoriginal2",
            original_price=Decimal("0.60"),
            original_timestamp="2024-01-01T01:00:00+00:00",
        ))

        # Realized PnL: 60 - 55 - 0.12 = 4.88
        assert pos.realized_pnl == Decimal("4.88")
        assert pos.net_size == Decimal("0")


class TestCopyTradeConfig:
    """Tests for CopyTradeConfig class."""

    def test_creation(self) -> None:
        """Test creating config."""
        config = CopyTradeConfig(
            top_k=10,
            position_size_usd=Decimal("200"),
            max_open_positions=30,
        )

        assert config.top_k == 10
        assert config.position_size_usd == Decimal("200")
        assert config.max_open_positions == 30

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        config = CopyTradeConfig(top_k=10)

        data = config.to_dict()

        assert data["top_k"] == 10
        assert "slippage" in data
        assert data["slippage"]["base_slippage_bps"] == 10

    def test_from_dict(self) -> None:
        """Test creation from dictionary."""
        data = {
            "top_k": 10,
            "min_trader_score": 40.0,
            "position_size_usd": "200",
            "slippage": {
                "base_slippage_bps": 20,
            },
        }

        config = CopyTradeConfig.from_dict(data)

        assert config.top_k == 10
        assert config.min_trader_score == 40.0
        assert config.position_size_usd == Decimal("200")
        assert config.slippage.base_slippage_bps == 20


class TestPaperCopyEngine:
    """Tests for PaperCopyEngine class."""

    def test_init_creates_directories(self, tmp_path: Path) -> None:
        """Test initialization creates directories."""
        engine = PaperCopyEngine(data_dir=tmp_path)

        assert tmp_path.exists()
        assert engine.config is not None
        assert engine.cash_balance == engine.config.starting_cash

    def test_init_creates_default_config(self, tmp_path: Path) -> None:
        """Test that default config is created."""
        engine = PaperCopyEngine(data_dir=tmp_path)

        assert (tmp_path / "config.json").exists()
        assert engine.config.top_k == 5  # Default

    def test_copy_trade_buy(self, tmp_path: Path) -> None:
        """Test copying a buy trade."""
        engine = PaperCopyEngine(data_dir=tmp_path)

        original = TraderFill(
            token_id="token123",
            side="buy",
            size=Decimal("100"),
            price=Decimal("0.55"),
            fee=Decimal("0.11"),
            timestamp="2024-01-01T00:00:00+00:00",
            transaction_hash="0xoriginal",
            market_slug="test-market",
        )

        copy = engine.copy_trade(original)

        assert copy is not None
        assert copy.side == "buy"
        assert copy.token_id == "token123"
        assert copy.original_tx_hash == "0xoriginal"
        # Copy price should be higher due to slippage
        assert copy.copy_price > Decimal("0.55")

    def test_copy_trade_updates_state(self, tmp_path: Path) -> None:
        """Test that copy_trade updates engine state."""
        engine = PaperCopyEngine(data_dir=tmp_path)
        starting_cash = engine.cash_balance

        original = TraderFill(
            token_id="token123",
            side="buy",
            size=Decimal("100"),
            price=Decimal("0.55"),
            fee=Decimal("0.11"),
            timestamp="2024-01-01T00:00:00+00:00",
            transaction_hash="0xoriginal",
        )

        engine.copy_trade(original)

        # Cash should decrease
        assert engine.cash_balance < starting_cash
        # Position should be created
        assert "token123" in engine.positions
        # Size is based on position_size_usd / price = 100 / 0.55 = ~181.82
        assert engine.positions["token123"].net_size > Decimal("180")

    def test_can_copy_trade_respects_limits(self, tmp_path: Path) -> None:
        """Test that copy respects position limits."""
        config = CopyTradeConfig(
            max_open_positions=1,
            position_size_usd=Decimal("100"),
        )
        engine = PaperCopyEngine(data_dir=tmp_path, config=config)

        # First trade should work
        fill1 = TraderFill(
            token_id="token1",
            side="buy",
            size=Decimal("100"),
            price=Decimal("0.50"),
            fee=Decimal("0.10"),
            timestamp="2024-01-01T00:00:00+00:00",
            transaction_hash="0x1",
        )
        assert engine._can_copy_trade(fill1)
        engine.copy_trade(fill1)

        # Second trade should fail (max positions reached)
        fill2 = TraderFill(
            token_id="token2",
            side="buy",
            size=Decimal("100"),
            price=Decimal("0.50"),
            fee=Decimal("0.10"),
            timestamp="2024-01-01T00:00:00+00:00",
            transaction_hash="0x2",
        )
        assert not engine._can_copy_trade(fill2)

    def test_compute_equity(self, tmp_path: Path) -> None:
        """Test equity computation."""
        engine = PaperCopyEngine(data_dir=tmp_path)

        # Add a position
        engine.copy_trade(TraderFill(
            token_id="token123",
            side="buy",
            size=Decimal("100"),
            price=Decimal("0.55"),
            fee=Decimal("0.11"),
            timestamp="2024-01-01T00:00:00+00:00",
            transaction_hash="0x1",
        ))

        equity = engine.compute_equity(current_prices={"token123": Decimal("0.60")})

        assert equity.position_count == 1
        assert equity.open_position_count == 1
        # Position value depends on copied size (position_size_usd / entry_price)
        # With default position_size_usd=100 and price=0.55, size ~= 181.82
        # At price 0.60: value ~= 181.82 * 0.60 ~= 109.09
        assert equity.positions_value > Decimal("100")

    def test_get_performance_summary(self, tmp_path: Path) -> None:
        """Test performance summary."""
        engine = PaperCopyEngine(data_dir=tmp_path)

        # Add a trade
        engine.copy_trade(TraderFill(
            token_id="token123",
            side="buy",
            size=Decimal("100"),
            price=Decimal("0.55"),
            fee=Decimal("0.11"),
            timestamp="2024-01-01T00:00:00+00:00",
            transaction_hash="0x1",
        ))

        summary = engine.get_performance_summary()

        assert "starting_cash" in summary
        assert "current_equity" in summary
        assert summary["total_copy_trades"] == 1
        assert summary["copied_traders"] >= 0


class TestCopyEquitySnapshot:
    """Tests for CopyEquitySnapshot dataclass."""

    def test_creation(self) -> None:
        """Test creating snapshot."""
        snapshot = CopyEquitySnapshot(
            timestamp="2024-01-01T00:00:00+00:00",
            cash_balance=Decimal("9944.89"),
            positions_value=Decimal("60"),
            realized_pnl=Decimal("0"),
            unrealized_pnl=Decimal("5"),
            total_fees=Decimal("0.11"),
            net_equity=Decimal("10004.89"),
            position_count=1,
            open_position_count=1,
            copied_traders=["0xabc"],
        )

        assert snapshot.net_equity == Decimal("10004.89")
        assert snapshot.copied_traders == ["0xabc"]

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        snapshot = CopyEquitySnapshot(
            timestamp="2024-01-01T00:00:00+00:00",
            cash_balance=Decimal("10000"),
            positions_value=Decimal("0"),
            realized_pnl=Decimal("0"),
            unrealized_pnl=Decimal("0"),
            total_fees=Decimal("0"),
            net_equity=Decimal("10000"),
            position_count=0,
            open_position_count=0,
        )

        data = snapshot.to_dict()

        assert data["net_equity"] == "10000"
        assert "copied_traders" in data
