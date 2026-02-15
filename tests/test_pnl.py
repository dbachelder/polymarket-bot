"""Tests for PnL verification engine."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

from polymarket.pnl import (
    BookLevel,
    Fill,
    OrderBook,
    Position,
    compute_pnl,
    load_fills_from_file,
    load_orderbooks_from_file,
)


class TestFill:
    """Test Fill dataclass and parsing."""

    def test_fill_from_dict_standard(self):
        data = {
            "token_id": "123",
            "side": "buy",
            "size": "100",
            "price": "0.55",
            "fee": "0.50",
            "timestamp": "2024-01-01T00:00:00Z",
            "transaction_hash": "0xabc",
        }
        fill = Fill.from_dict(data)
        assert fill.token_id == "123"
        assert fill.side == "buy"
        assert fill.size == Decimal("100")
        assert fill.price == Decimal("0.55")
        assert fill.fee == Decimal("0.50")
        assert fill.timestamp == "2024-01-01T00:00:00Z"
        assert fill.transaction_hash == "0xabc"

    def test_fill_from_dict_api_variants(self):
        # Test alternative field names (Polymarket API variants)
        data = {
            "asset_id": "456",
            "trade_side": "SELL",
            "takerAmount": "50",
            "execution_price": "0.75",
            "trade_fee": "0.25",
            "created_at": "2024-02-01T12:00:00Z",
            "tx_hash": "0xdef",
        }
        fill = Fill.from_dict(data)
        assert fill.token_id == "456"
        assert fill.side == "sell"
        assert fill.size == Decimal("50")
        assert fill.price == Decimal("0.75")

    def test_fill_from_dict_missing_optional(self):
        data = {
            "token_id": "789",
            "side": "buy",
            "size": "10",
            "price": "0.50",
        }
        fill = Fill.from_dict(data)
        assert fill.fee == Decimal("0")
        assert fill.timestamp == ""
        assert fill.transaction_hash is None


class TestPosition:
    """Test Position tracking and cost basis."""

    def test_position_add_buy(self):
        pos = Position(token_id="123")
        pos.add_buy(size=Decimal("100"), price=Decimal("0.50"), fee=Decimal("1.00"))
        
        assert pos.net_size == Decimal("100")
        assert pos.avg_cost_basis == Decimal("0.50")
        assert pos.total_fees == Decimal("1.00")
        assert pos.buy_count == 1

    def test_position_multiple_buys_averaging(self):
        pos = Position(token_id="123")
        pos.add_buy(size=Decimal("100"), price=Decimal("0.50"), fee=Decimal("1.00"))
        pos.add_buy(size=Decimal("100"), price=Decimal("0.60"), fee=Decimal("1.00"))
        
        # Average cost = (100*0.50 + 100*0.60) / 200 = 0.55
        assert pos.net_size == Decimal("200")
        assert pos.avg_cost_basis == Decimal("0.55")
        assert pos.total_fees == Decimal("2.00")

    def test_position_sell_realized_pnl(self):
        pos = Position(token_id="123")
        pos.add_buy(size=Decimal("100"), price=Decimal("0.50"), fee=Decimal("1.00"))
        
        # Sell half at higher price
        realized = pos.add_sell(size=Decimal("50"), price=Decimal("0.60"), fee=Decimal("0.50"))
        
        # Realized = proceeds - cost_basis - fee
        # = 50*0.60 - 50*0.50 - 0.50 = 30 - 25 - 0.50 = 4.50
        assert realized == Decimal("4.50")
        assert pos.net_size == Decimal("50")
        assert pos.realized_pnl == Decimal("4.50")

    def test_position_full_sell(self):
        pos = Position(token_id="123")
        pos.add_buy(size=Decimal("100"), price=Decimal("0.50"), fee=Decimal("1.00"))
        realized = pos.add_sell(size=Decimal("100"), price=Decimal("0.60"), fee=Decimal("1.00"))
        
        # Realized = 100*0.60 - 100*0.50 - 1.00 = 60 - 50 - 1 = 9
        assert realized == Decimal("9")
        assert pos.net_size == Decimal("0")

    def test_position_empty_cost_basis(self):
        pos = Position(token_id="123")
        assert pos.avg_cost_basis == Decimal("0")


class TestOrderBook:
    """Test OrderBook and liquidation calculations."""

    def test_orderbook_from_dict(self):
        data = {
            "bids": [
                {"price": "0.60", "size": "100"},
                {"price": "0.55", "size": "200"},
            ],
            "asks": [
                {"price": "0.65", "size": "150"},
                {"price": "0.70", "size": "100"},
            ],
        }
        book = OrderBook.from_dict("token123", data)
        assert book.token_id == "token123"
        assert len(book.bids) == 2
        assert book.bids[0].price == Decimal("0.60")  # Sorted high to low
        assert len(book.asks) == 2
        assert book.asks[0].price == Decimal("0.65")  # Sorted low to high

    def test_walk_liquidation_value_simple(self):
        book = OrderBook(
            token_id="123",
            bids=[
                BookLevel(price=Decimal("0.60"), size=Decimal("100")),
                BookLevel(price=Decimal("0.55"), size=Decimal("200")),
            ],
            asks=[],
        )
        
        # Sell 50 shares at 0.60
        value = book.get_walk_liquidation_value(Decimal("50"), is_yes=True)
        assert value == Decimal("30")  # 50 * 0.60

    def test_walk_liquidation_value_multi_level(self):
        book = OrderBook(
            token_id="123",
            bids=[
                BookLevel(price=Decimal("0.60"), size=Decimal("50")),
                BookLevel(price=Decimal("0.55"), size=Decimal("100")),
            ],
            asks=[],
        )
        
        # Sell 100 shares - 50 @ 0.60, 50 @ 0.55
        value = book.get_walk_liquidation_value(Decimal("100"), is_yes=True)
        assert value == Decimal("57.5")  # 50*0.60 + 50*0.55

    def test_walk_liquidation_value_exceeds_book(self):
        book = OrderBook(
            token_id="123",
            bids=[
                BookLevel(price=Decimal("0.60"), size=Decimal("50")),
            ],
            asks=[],
        )
        
        # Try to sell more than book depth - remainder gets 0
        value = book.get_walk_liquidation_value(Decimal("100"), is_yes=True)
        assert value == Decimal("30")  # 50 * 0.60, rest is 0

    def test_walk_liquidation_value_no_position(self):
        book = OrderBook(
            token_id="123",
            bids=[BookLevel(price=Decimal("0.60"), size=Decimal("100"))],
            asks=[],
        )
        value = book.get_walk_liquidation_value(Decimal("0"), is_yes=True)
        assert value == Decimal("0")


class TestComputePnL:
    """Test full PnL computation."""

    def test_simple_buy_sell(self):
        fills = [
            Fill("token1", "buy", Decimal("100"), Decimal("0.50"), Decimal("1"), "2024-01-01T00:00:00Z"),
            Fill("token1", "sell", Decimal("100"), Decimal("0.60"), Decimal("1"), "2024-01-01T01:00:00Z"),
        ]
        
        report = compute_pnl(fills)
        
        assert report.total_fills == 2
        assert report.realized_pnl == Decimal("9")  # (100 * 0.60) - (100 * 0.50) - 1 - 1
        assert report.net_pnl == Decimal("7")  # realized - fees

    def test_unrealized_pnl(self):
        fills = [
            Fill("token1", "buy", Decimal("100"), Decimal("0.50"), Decimal("1"), "2024-01-01T00:00:00Z"),
        ]
        
        current_prices = {"token1": Decimal("0.60")}
        report = compute_pnl(fills, current_prices=current_prices)
        
        # Unrealized = (0.60 - 0.50) * 100 = 10
        assert report.unrealized_pnl == Decimal("10")
        assert len(report.positions) == 1
        assert report.positions[0]["net_size"] == 100.0

    def test_multiple_tokens(self):
        fills = [
            Fill("token1", "buy", Decimal("100"), Decimal("0.50"), Decimal("1"), "2024-01-01T00:00:00Z"),
            Fill("token2", "buy", Decimal("50"), Decimal("0.30"), Decimal("0.5"), "2024-01-01T00:00:00Z"),
        ]
        
        report = compute_pnl(fills)
        
        assert report.unique_tokens == 2
        assert len(report.positions) == 2

    def test_empty_fills(self):
        report = compute_pnl([])
        assert report.total_fills == 0
        assert report.unique_tokens == 0
        assert report.net_pnl == Decimal("0")

    def test_with_liquidation_value(self):
        fills = [
            Fill("token1", "buy", Decimal("100"), Decimal("0.50"), Decimal("1"), "2024-01-01T00:00:00Z"),
        ]
        
        orderbooks = {
            "token1": OrderBook(
                token_id="token1",
                bids=[BookLevel(price=Decimal("0.58"), size=Decimal("200"))],
                asks=[],
            )
        }
        
        report = compute_pnl(fills, orderbooks=orderbooks)
        
        # Mark to market = 100 * 0.58 = 58
        assert report.mark_to_market == Decimal("58")
        # Liquidation = 100 * 0.58 = 58 (full fill at best bid)
        assert report.liquidation_value == Decimal("58")


class TestLoadFillsFromFile:
    """Test loading fills from JSON files."""

    def test_load_array_format(self, tmp_path: Path):
        data = [
            {"token_id": "123", "side": "buy", "size": "100", "price": "0.50"},
            {"token_id": "123", "side": "sell", "size": "50", "price": "0.60"},
        ]
        file_path = tmp_path / "fills.json"
        file_path.write_text(json.dumps(data))
        
        fills = load_fills_from_file(file_path)
        assert len(fills) == 2
        assert fills[0].size == Decimal("100")

    def test_load_fills_key_format(self, tmp_path: Path):
        data = {
            "fills": [
                {"token_id": "123", "side": "buy", "size": "100", "price": "0.50"},
            ]
        }
        file_path = tmp_path / "fills.json"
        file_path.write_text(json.dumps(data))
        
        fills = load_fills_from_file(file_path)
        assert len(fills) == 1

    def test_load_data_key_format(self, tmp_path: Path):
        data = {
            "data": [
                {"token_id": "123", "side": "buy", "size": "100", "price": "0.50"},
            ]
        }
        file_path = tmp_path / "fills.json"
        file_path.write_text(json.dumps(data))
        
        fills = load_fills_from_file(file_path)
        assert len(fills) == 1


class TestLoadOrderbooksFromFile:
    """Test loading orderbooks from JSON files."""

    def test_load_orderbooks(self, tmp_path: Path):
        data = {
            "token1": {
                "bids": [{"price": "0.60", "size": "100"}],
                "asks": [{"price": "0.65", "size": "150"}],
            },
            "token2": {
                "bids": [{"price": "0.40", "size": "200"}],
                "asks": [{"price": "0.45", "size": "100"}],
            },
        }
        file_path = tmp_path / "books.json"
        file_path.write_text(json.dumps(data))
        
        books = load_orderbooks_from_file(file_path)
        assert len(books) == 2
        assert "token1" in books
        assert books["token1"].bids[0].price == Decimal("0.60")
