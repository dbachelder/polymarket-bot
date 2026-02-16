"""Tests for btc_both_sides_arb module."""

from decimal import Decimal

from polymarket.btc_both_sides_arb import (
    BothSidesArbitrageStrategy,
    BothSidesOpportunity,
    BothSidesTrade,
    calculate_confidence,
    calculate_spread,
    calculate_spread_after_fees,
    check_mispricing,
    close_trade,
    execute_paper_trade,
)


class TestCalculateSpread:
    """Tests for calculate_spread function."""

    def test_spread_normal(self) -> None:
        """Test spread calculation with normal prices."""
        spread = calculate_spread(Decimal("0.48"), Decimal("0.46"))
        # 1.0 - 0.48 - 0.46 = 0.06
        assert spread == Decimal("0.06")

    def test_spread_zero(self) -> None:
        """Test spread when prices sum to 1.0."""
        spread = calculate_spread(Decimal("0.50"), Decimal("0.50"))
        assert spread == Decimal("0")

    def test_spread_negative(self) -> None:
        """Test spread when prices sum to > 1.0."""
        spread = calculate_spread(Decimal("0.55"), Decimal("0.50"))
        # 1.0 - 0.55 - 0.50 = -0.05
        assert spread == Decimal("-0.05")


class TestCalculateSpreadAfterFees:
    """Tests for calculate_spread_after_fees function."""

    def test_spread_after_fees_typical(self) -> None:
        """Test spread calculation after fees."""
        spread = calculate_spread_after_fees(Decimal("0.48"), Decimal("0.46"))
        # Total cost = 0.94
        # Gross profit = 1.0 - 0.94 = 0.06
        # Fees = 0.94 * 0.04 = 0.0376
        # Net = 0.06 - 0.0376 = 0.0224
        assert spread == Decimal("0.0224")

    def test_spread_after_fees_no_profit(self) -> None:
        """Test when spread is too small to profit after fees."""
        spread = calculate_spread_after_fees(Decimal("0.495"), Decimal("0.495"))
        # Total cost = 0.99
        # Gross profit = 0.01
        # Fees = 0.99 * 0.04 = 0.0396
        # Net = 0.01 - 0.0396 = -0.0296
        assert spread < Decimal("0")


class TestCheckMispricing:
    """Tests for check_mispricing function."""

    def test_mispriced_true(self) -> None:
        """Test when market is mispriced."""
        is_mispriced, spread = check_mispricing(
            Decimal("0.47"), Decimal("0.47"), min_spread=Decimal("0.02")
        )
        # Sum = 0.94, gross = 0.06
        # Fees = 0.94 * 0.04 = 0.0376
        # Net = 0.06 - 0.0376 = 0.0224 > 0.02
        assert is_mispriced is True
        assert spread == Decimal("0.0224")

    def test_mispriced_false_sum_too_high(self) -> None:
        """Test when sum is too high for mispricing."""
        is_mispriced, spread = check_mispricing(
            Decimal("0.50"), Decimal("0.49"), min_spread=Decimal("0.02")
        )
        # Sum = 0.99 > 1.0 - 0.04 = 0.96
        assert is_mispriced is False

    def test_mispriced_false_spread_too_small(self) -> None:
        """Test when spread after fees is too small."""
        is_mispriced, spread = check_mispricing(
            Decimal("0.49"), Decimal("0.48"), min_spread=Decimal("0.03")
        )
        # Sum = 0.97, gross = 0.03
        # Fees = 0.97 * 0.04 = 0.0388
        # Net = 0.03 - 0.0388 = -0.0088 < 0.03
        assert is_mispriced is False


class TestCalculateConfidence:
    """Tests for calculate_confidence function."""

    def test_confidence_base(self) -> None:
        """Test base confidence calculation."""
        confidence = calculate_confidence(Decimal("0.48"), Decimal("0.46"))
        # Base = 0.5
        # Price sanity (+0.1)
        # Neither side too cheap (+0.1)
        # Volume = None (no bonus)
        # Aligned = None (no bonus)
        assert confidence == Decimal("0.7")

    def test_confidence_with_volume(self) -> None:
        """Test confidence with high volume."""
        confidence = calculate_confidence(
            Decimal("0.48"), Decimal("0.46"), volume=Decimal("100000")
        )
        # Base 0.5 + sanity 0.1 + cheap 0.1 + volume 0.15 = 0.85
        assert confidence == Decimal("0.85")

    def test_confidence_aligned(self) -> None:
        """Test confidence with alignment bonus."""
        confidence = calculate_confidence(Decimal("0.48"), Decimal("0.46"), aligned_15m=True)
        # Base 0.5 + sanity 0.1 + cheap 0.1 + aligned 0.15 = 0.85
        assert confidence == Decimal("0.85")

    def test_confidence_extreme_price(self) -> None:
        """Test confidence when one side is very cheap."""
        confidence = calculate_confidence(Decimal("0.95"), Decimal("0.02"))
        # Base 0.5 + sanity 0.1 (both valid) + no cheap bonus (0.02 < 0.05)
        assert confidence == Decimal("0.6")


class TestBothSidesOpportunity:
    """Tests for BothSidesOpportunity dataclass."""

    def test_is_valid_true(self) -> None:
        """Test valid opportunity."""
        opp = BothSidesOpportunity(
            market_id="m1",
            event_id="e1",
            interval="5m",
            up_token_id="up1",
            down_token_id="down1",
            up_price=Decimal("0.47"),
            down_price=Decimal("0.47"),
            price_sum=Decimal("0.94"),
            fee_buffer=Decimal("0.04"),
            spread=Decimal("0.06"),
            spread_after_fees=Decimal("0.0224"),
            aligned_15m=None,
            confidence=Decimal("0.7"),
            timestamp="2024-01-01T00:00:00Z",
            market_metadata={"title": "BTC 5m"},
        )
        assert opp.is_valid is True

    def test_is_valid_false_spread(self) -> None:
        """Test invalid due to small spread."""
        opp = BothSidesOpportunity(
            market_id="m1",
            event_id="e1",
            interval="5m",
            up_token_id="up1",
            down_token_id="down1",
            up_price=Decimal("0.49"),
            down_price=Decimal("0.49"),
            price_sum=Decimal("0.98"),
            fee_buffer=Decimal("0.04"),
            spread=Decimal("0.02"),
            spread_after_fees=Decimal("-0.0196"),  # Negative after fees
            aligned_15m=None,
            confidence=Decimal("0.7"),
            timestamp="2024-01-01T00:00:00Z",
            market_metadata={"title": "BTC 5m"},
        )
        assert opp.is_valid is False

    def test_is_valid_false_confidence(self) -> None:
        """Test invalid due to low confidence."""
        opp = BothSidesOpportunity(
            market_id="m1",
            event_id="e1",
            interval="5m",
            up_token_id="up1",
            down_token_id="down1",
            up_price=Decimal("0.47"),
            down_price=Decimal("0.47"),
            price_sum=Decimal("0.94"),
            fee_buffer=Decimal("0.04"),
            spread=Decimal("0.06"),
            spread_after_fees=Decimal("0.0224"),
            aligned_15m=None,
            confidence=Decimal("0.3"),  # Below 0.5 threshold
            timestamp="2024-01-01T00:00:00Z",
            market_metadata={"title": "BTC 5m"},
        )
        assert opp.is_valid is False

    def test_profit_calculation(self) -> None:
        """Test profit calculations."""
        opp = BothSidesOpportunity(
            market_id="m1",
            event_id="e1",
            interval="5m",
            up_token_id="up1",
            down_token_id="down1",
            up_price=Decimal("0.48"),
            down_price=Decimal("0.46"),
            price_sum=Decimal("0.94"),
            fee_buffer=Decimal("0.04"),
            spread=Decimal("0.06"),
            spread_after_fees=Decimal("0.0224"),
            aligned_15m=None,
            confidence=Decimal("0.7"),
            timestamp="2024-01-01T00:00:00Z",
            market_metadata={"title": "BTC 5m"},
        )
        assert opp.total_position_cost == Decimal("0.94")
        assert opp.guaranteed_payout == Decimal("1.0")
        assert opp.gross_profit == Decimal("0.06")
        assert opp.fees == Decimal("0.0376")  # 0.94 * 0.04
        assert opp.net_profit == Decimal("0.0224")

    def test_to_dict(self) -> None:
        """Test serialization to dict."""
        opp = BothSidesOpportunity(
            market_id="m1",
            event_id="e1",
            interval="5m",
            up_token_id="up1",
            down_token_id="down1",
            up_price=Decimal("0.48"),
            down_price=Decimal("0.46"),
            price_sum=Decimal("0.94"),
            fee_buffer=Decimal("0.04"),
            spread=Decimal("0.06"),
            spread_after_fees=Decimal("0.0224"),
            aligned_15m=True,
            confidence=Decimal("0.7"),
            timestamp="2024-01-01T00:00:00Z",
            market_metadata={"title": "BTC 5m"},
        )
        d = opp.to_dict()
        assert d["market_id"] == "m1"
        assert d["interval"] == "5m"
        assert d["up_price"] == "0.48"
        assert d["aligned_15m"] is True


class TestExecutePaperTrade:
    """Tests for execute_paper_trade function."""

    def test_trade_creation(self) -> None:
        """Test paper trade creation."""
        opp = BothSidesOpportunity(
            market_id="m1",
            event_id="e1",
            interval="5m",
            up_token_id="up1",
            down_token_id="down1",
            up_price=Decimal("0.48"),
            down_price=Decimal("0.46"),
            price_sum=Decimal("0.94"),
            fee_buffer=Decimal("0.04"),
            spread=Decimal("0.06"),
            spread_after_fees=Decimal("0.0224"),
            aligned_15m=True,
            confidence=Decimal("0.7"),
            timestamp="2024-01-01T00:00:00Z",
            market_metadata={"title": "BTC 5m"},
        )

        trade = execute_paper_trade(opp, position_size=Decimal("100"), trade_id="test_001")

        assert trade.trade_id == "test_001"
        assert trade.market_id == "m1"
        assert trade.interval == "5m"
        assert trade.up_entry_price == Decimal("0.48")
        assert trade.down_entry_price == Decimal("0.46")
        assert trade.position_size == Decimal("100")
        assert trade.total_cost == Decimal("0.94")
        assert trade.spread_at_entry == Decimal("0.0224")
        assert trade.aligned_15m is True
        assert trade.status == "open"

    def test_trade_auto_id(self) -> None:
        """Test auto-generated trade ID."""
        opp = BothSidesOpportunity(
            market_id="m1",
            event_id="e1",
            interval="5m",
            up_token_id="up1",
            down_token_id="down1",
            up_price=Decimal("0.48"),
            down_price=Decimal("0.46"),
            price_sum=Decimal("0.94"),
            fee_buffer=Decimal("0.04"),
            spread=Decimal("0.06"),
            spread_after_fees=Decimal("0.0224"),
            aligned_15m=None,
            confidence=Decimal("0.7"),
            timestamp="2024-01-01T00:00:00Z",
            market_metadata={"title": "BTC 5m"},
        )

        trade = execute_paper_trade(opp, position_size=Decimal("100"))

        assert trade.trade_id.startswith("bsa_5m_")


class TestCloseTrade:
    """Tests for close_trade function."""

    def test_close_up_wins(self) -> None:
        """Test closing trade when UP side wins."""
        trade = BothSidesTrade(
            trade_id="test_001",
            timestamp="2024-01-01T00:00:00Z",
            market_id="m1",
            event_id="e1",
            interval="5m",
            up_token_id="up1",
            down_token_id="down1",
            up_entry_price=Decimal("0.48"),
            down_entry_price=Decimal("0.46"),
            position_size=Decimal("100"),
            total_cost=Decimal("0.94"),
            spread_at_entry=Decimal("0.0224"),
            aligned_15m=True,
            status="open",
        )

        closed = close_trade(trade, winning_side="up", close_reason="resolution")

        assert closed.status == "closed"
        assert closed.winning_side == "up"
        assert closed.payout == Decimal("100")  # 100 * 1.0
        # PnL = 100 - 0.94 - (0.94 * 0.04) = 100 - 0.94 - 0.0376 = 99.0224
        assert closed.net_pnl == Decimal("99.0224")
        assert closed.close_reason == "resolution"

    def test_close_down_wins(self) -> None:
        """Test closing trade when DOWN side wins."""
        trade = BothSidesTrade(
            trade_id="test_001",
            timestamp="2024-01-01T00:00:00Z",
            market_id="m1",
            event_id="e1",
            interval="5m",
            up_token_id="up1",
            down_token_id="down1",
            up_entry_price=Decimal("0.48"),
            down_entry_price=Decimal("0.46"),
            position_size=Decimal("50"),
            total_cost=Decimal("0.94"),
            spread_at_entry=Decimal("0.0224"),
            aligned_15m=False,
            status="open",
        )

        closed = close_trade(trade, winning_side="down")

        assert closed.status == "closed"
        assert closed.winning_side == "down"
        assert closed.payout == Decimal("50")


class TestBothSidesTrade:
    """Tests for BothSidesTrade dataclass."""

    def test_to_dict(self) -> None:
        """Test serialization to dict."""
        trade = BothSidesTrade(
            trade_id="test_001",
            timestamp="2024-01-01T00:00:00Z",
            market_id="m1",
            event_id="e1",
            interval="5m",
            up_token_id="up1",
            down_token_id="down1",
            up_entry_price=Decimal("0.48"),
            down_entry_price=Decimal("0.46"),
            position_size=Decimal("100"),
            total_cost=Decimal("0.94"),
            spread_at_entry=Decimal("0.0224"),
            aligned_15m=True,
            status="open",
        )

        d = trade.to_dict()
        assert d["trade_id"] == "test_001"
        assert d["market_id"] == "m1"
        assert d["up_entry_price"] == "0.48"
        assert d["aligned_15m"] is True
        assert d["status"] == "open"
        assert d["payout"] is None


class TestBothSidesArbitrageStrategy:
    """Tests for BothSidesArbitrageStrategy class."""

    def test_strategy_init(self, tmp_path) -> None:
        """Test strategy initialization."""
        strategy = BothSidesArbitrageStrategy(
            position_size=Decimal("200"),
            check_alignment=True,
            data_dir=tmp_path,
        )
        assert strategy.position_size == Decimal("200")
        assert strategy.check_alignment is True
        assert strategy.data_dir == tmp_path

    def test_get_stats_empty(self, tmp_path) -> None:
        """Test stats with no trades."""
        strategy = BothSidesArbitrageStrategy(data_dir=tmp_path)
        stats = strategy.get_stats()
        assert stats["total_opportunities"] == 0
        assert stats["total_trades"] == 0
        assert stats["total_pnl"] == Decimal("0")
