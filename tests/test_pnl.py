"""Tests for PnL verification engine."""

from __future__ import annotations

import json
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import pytest

from polymarket.pnl import (
    DAILY_SUMMARY_DIR,
    BookLevel,
    Fill,
    OrderBook,
    PnLReport,
    PnLVerifier,
    Position,
    compute_pnl,
    load_fills_from_file,
    load_orderbooks_from_file,
    load_orderbooks_from_snapshot,
    save_daily_summary,
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

    def test_fill_cash_flow_buy(self):
        fill = Fill(
            token_id="123",
            side="buy",
            size=Decimal("100"),
            price=Decimal("0.50"),
            fee=Decimal("1.00"),
            timestamp="2024-01-01T00:00:00Z",
        )
        # Buy: -(size * price + fee) = -(50 + 1) = -51
        assert fill.cash_flow == Decimal("-51")

    def test_fill_cash_flow_sell(self):
        fill = Fill(
            token_id="123",
            side="sell",
            size=Decimal("100"),
            price=Decimal("0.60"),
            fee=Decimal("1.00"),
            timestamp="2024-01-01T00:00:00Z",
        )
        # Sell: size * price - fee = 60 - 1 = 59
        assert fill.cash_flow == Decimal("59")

    def test_fill_datetime_utc(self):
        fill = Fill(
            token_id="123",
            side="buy",
            size=Decimal("100"),
            price=Decimal("0.50"),
            fee=Decimal("1"),
            timestamp="2024-01-15T12:30:45Z",
        )
        dt = fill.datetime_utc
        assert dt.year == 2024
        assert dt.month == 1
        assert dt.day == 15
        assert dt.hour == 12
        assert dt.minute == 30

    def test_fill_with_market_slug(self):
        data = {
            "token_id": "123",
            "side": "buy",
            "size": "100",
            "price": "0.50",
            "market_slug": "will-bitcoin-hit-100k",
        }
        fill = Fill.from_dict(data)
        assert fill.market_slug == "will-bitcoin-hit-100k"


class TestPosition:
    """Test Position tracking and cost basis."""

    def test_position_add_buy(self):
        pos = Position(token_id="123")
        pos.add_buy(size=Decimal("100"), price=Decimal("0.50"), fee=Decimal("1.00"))

        assert pos.net_size == Decimal("100")
        assert pos.avg_cost_basis == Decimal("0.50")
        assert pos.total_fees == Decimal("1.00")
        assert pos.buy_count == 1
        assert pos.total_bought == Decimal("100")

    def test_position_multiple_buys_averaging(self):
        pos = Position(token_id="123")
        pos.add_buy(size=Decimal("100"), price=Decimal("0.50"), fee=Decimal("1.00"))
        pos.add_buy(size=Decimal("100"), price=Decimal("0.60"), fee=Decimal("1.00"))

        # Average cost = (100*0.50 + 100*0.60) / 200 = 0.55
        assert pos.net_size == Decimal("200")
        assert pos.avg_cost_basis == Decimal("0.55")
        assert pos.total_fees == Decimal("2.00")
        assert pos.total_bought == Decimal("200")

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
        assert pos.total_sold == Decimal("50")

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

    def test_position_verify_ok(self):
        pos = Position(token_id="123")
        pos.add_buy(size=Decimal("100"), price=Decimal("0.50"), fee=Decimal("1.00"))
        pos.add_sell(size=Decimal("30"), price=Decimal("0.60"), fee=Decimal("0.50"))

        warnings = pos.verify()
        assert len(warnings) == 0

    def test_position_verify_mismatch(self):
        pos = Position(token_id="123")
        pos.net_size = Decimal("50")
        pos.total_bought = Decimal("100")
        pos.total_sold = Decimal("40")  # Should be 50

        warnings = pos.verify()
        assert len(warnings) == 1
        assert "size mismatch" in warnings[0]


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

    def test_orderbook_mid_price(self):
        book = OrderBook(
            token_id="123",
            bids=[BookLevel(price=Decimal("0.58"), size=Decimal("100"))],
            asks=[BookLevel(price=Decimal("0.62"), size=Decimal("100"))],
        )
        # Mid = (0.58 + 0.62) / 2 = 0.60
        assert book.mid_price == Decimal("0.60")

    def test_orderbook_spread(self):
        book = OrderBook(
            token_id="123",
            bids=[BookLevel(price=Decimal("0.58"), size=Decimal("100"))],
            asks=[BookLevel(price=Decimal("0.62"), size=Decimal("100"))],
        )
        # Spread = 0.62 - 0.58 = 0.04
        assert book.spread == Decimal("0.04")

    def test_orderbook_mid_price_empty(self):
        book = OrderBook(token_id="123", bids=[], asks=[])
        assert book.mid_price is None

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


class TestPnLVerifier:
    """Test PnLVerifier class with cash tracking."""

    def test_verifier_initialization(self):
        verifier = PnLVerifier(starting_cash=Decimal("1000"))
        assert verifier.starting_cash == Decimal("1000")
        assert verifier.cash_balance == Decimal("1000")

    def test_verifier_add_fill_buy(self):
        verifier = PnLVerifier(starting_cash=Decimal("1000"))
        fill = Fill(
            token_id="123",
            side="buy",
            size=Decimal("100"),
            price=Decimal("0.50"),
            fee=Decimal("1"),
            timestamp="2024-01-01T00:00:00Z",
        )
        verifier.add_fill(fill)

        # Cash flow: -(100 * 0.50 + 1) = -51
        assert verifier.cash_balance == Decimal("949")
        assert len(verifier.positions) == 1
        assert verifier.positions["123"].net_size == Decimal("100")

    def test_verifier_add_fill_sell(self):
        verifier = PnLVerifier(starting_cash=Decimal("1000"))
        # First buy
        verifier.add_fill(
            Fill(
                token_id="123",
                side="buy",
                size=Decimal("100"),
                price=Decimal("0.50"),
                fee=Decimal("1"),
                timestamp="2024-01-01T00:00:00Z",
            )
        )
        # Then sell
        verifier.add_fill(
            Fill(
                token_id="123",
                side="sell",
                size=Decimal("100"),
                price=Decimal("0.60"),
                fee=Decimal("1"),
                timestamp="2024-01-01T01:00:00Z",
            )
        )

        # Cash: 1000 - 51 + 59 = 1008
        assert verifier.cash_balance == Decimal("1008")
        assert verifier.positions["123"].net_size == Decimal("0")

    def test_verifier_cashflow_conservation_ok(self):
        verifier = PnLVerifier(starting_cash=Decimal("1000"))
        verifier.add_fill(
            Fill(
                token_id="123",
                side="buy",
                size=Decimal("100"),
                price=Decimal("0.50"),
                fee=Decimal("1"),
                timestamp="2024-01-01T00:00:00Z",
            )
        )

        is_conserved, warnings = verifier.verify_cashflow()
        assert is_conserved is True
        assert len(warnings) == 0

    def test_verifier_cashflow_conservation_fail(self):
        verifier = PnLVerifier(starting_cash=Decimal("1000"))
        verifier.add_fill(
            Fill(
                token_id="123",
                side="buy",
                size=Decimal("100"),
                price=Decimal("0.50"),
                fee=Decimal("1"),
                timestamp="2024-01-01T00:00:00Z",
            )
        )
        # Manually corrupt cash balance
        verifier.cash_balance = Decimal("500")

        is_conserved, warnings = verifier.verify_cashflow()
        assert is_conserved is False
        assert len(warnings) == 1
        assert "Cashflow not conserved" in warnings[0]

    def test_verifier_position_verification(self):
        verifier = PnLVerifier()
        verifier.add_fill(
            Fill(
                token_id="123",
                side="buy",
                size=Decimal("100"),
                price=Decimal("0.50"),
                fee=Decimal("1"),
                timestamp="2024-01-01T00:00:00Z",
            )
        )
        verifier.add_fill(
            Fill(
                token_id="123",
                side="sell",
                size=Decimal("30"),
                price=Decimal("0.60"),
                fee=Decimal("0.50"),
                timestamp="2024-01-01T01:00:00Z",
            )
        )

        is_verified, warnings = verifier.verify_positions()
        assert is_verified is True
        assert len(warnings) == 0

    def test_verifier_compute_pnl_report(self):
        verifier = PnLVerifier(starting_cash=Decimal("1000"))
        verifier.add_fill(
            Fill(
                token_id="123",
                side="buy",
                size=Decimal("100"),
                price=Decimal("0.50"),
                fee=Decimal("1"),
                timestamp="2024-01-01T00:00:00Z",
            )
        )
        verifier.add_fill(
            Fill(
                token_id="123",
                side="sell",
                size=Decimal("50"),
                price=Decimal("0.60"),
                fee=Decimal("0.50"),
                timestamp="2024-01-01T01:00:00Z",
            )
        )

        report = verifier.compute_pnl()

        assert report.total_fills == 2
        assert report.unique_tokens == 1
        assert report.starting_cash == Decimal("1000")
        assert report.realized_pnl == Decimal("4.5")  # From TestPosition

    def test_verifier_compute_pnl_with_orderbooks(self):
        verifier = PnLVerifier()
        verifier.add_fill(
            Fill(
                token_id="123",
                side="buy",
                size=Decimal("100"),
                price=Decimal("0.50"),
                fee=Decimal("1"),
                timestamp="2024-01-01T00:00:00Z",
            )
        )

        orderbooks = {
            "123": OrderBook(
                token_id="123",
                bids=[
                    BookLevel(price=Decimal("0.58"), size=Decimal("50")),
                    BookLevel(price=Decimal("0.55"), size=Decimal("100")),
                ],
                asks=[BookLevel(price=Decimal("0.62"), size=Decimal("100"))],
            )
        }

        report = verifier.compute_pnl(orderbooks=orderbooks)

        # Mark to mid: 100 * 0.60 = 60 (mid of 0.58 and 0.62)
        assert report.mark_to_mid == Decimal("60")
        # Liquidation: 50*0.58 + 50*0.55 = 29 + 27.5 = 56.5
        assert report.liquidation_value == Decimal("56.5")
        # Unrealized: (0.60 - 0.50) * 100 = 10
        assert report.unrealized_pnl == Decimal("10")

    def test_verifier_filter_by_since(self):
        verifier = PnLVerifier()
        verifier.add_fill(
            Fill(
                token_id="123",
                side="buy",
                size=Decimal("100"),
                price=Decimal("0.50"),
                fee=Decimal("1"),
                timestamp="2024-01-01T00:00:00Z",
            )
        )
        verifier.add_fill(
            Fill(
                token_id="123",
                side="sell",
                size=Decimal("50"),
                price=Decimal("0.60"),
                fee=Decimal("0.50"),
                timestamp="2024-01-15T00:00:00Z",
            )
        )

        report = verifier.compute_pnl(since="2024-01-10T00:00:00Z")
        assert report.total_fills == 1  # Only the sell on Jan 15

    def test_verifier_filter_by_market(self):
        verifier = PnLVerifier()
        verifier.add_fill(
            Fill(
                token_id="123",
                side="buy",
                size=Decimal("100"),
                price=Decimal("0.50"),
                fee=Decimal("1"),
                timestamp="2024-01-01T00:00:00Z",
                market_slug="will-bitcoin-hit-100k",
            )
        )
        verifier.add_fill(
            Fill(
                token_id="456",
                side="buy",
                size=Decimal("50"),
                price=Decimal("0.30"),
                fee=Decimal("0.50"),
                timestamp="2024-01-01T00:00:00Z",
                market_slug="will-ethereum-merge",
            )
        )

        report = verifier.compute_pnl(market_filter="bitcoin")
        assert report.total_fills == 1
        assert report.unique_tokens == 1


class TestPnLReport:
    """Test PnLReport serialization."""

    def test_report_to_dict(self):
        report = PnLReport(
            total_fills=10,
            unique_tokens=2,
            realized_pnl=Decimal("100"),
            unrealized_pnl=Decimal("50"),
        )
        d = report.to_dict()

        assert d["summary"]["total_fills"] == 10
        assert d["summary"]["unique_tokens"] == 2
        assert d["pnl"]["realized_pnl"] == 100.0
        assert d["pnl"]["unrealized_pnl"] == 50.0

    def test_report_to_json(self):
        report = PnLReport(total_fills=5, unique_tokens=1)
        json_str = report.to_json()
        data = json.loads(json_str)
        assert data["summary"]["total_fills"] == 5

    def test_report_save(self, tmp_path: Path):
        report = PnLReport(total_fills=5, unique_tokens=1)
        out_path = tmp_path / "test_report.json"
        report.save(out_path)

        assert out_path.exists()
        data = json.loads(out_path.read_text())
        assert data["summary"]["total_fills"] == 5


class TestComputePnL:
    """Test full PnL computation (legacy function)."""

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
                asks=[BookLevel(price=Decimal("0.58"), size=Decimal("200"))],  # Same price for mid
            )
        }

        report = compute_pnl(fills, orderbooks=orderbooks)

        # Mark to mid = 100 * 0.58 = 58
        assert report.mark_to_mid == Decimal("58")
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

    def test_load_jsonl_format(self, tmp_path: Path):
        file_path = tmp_path / "fills.jsonl"
        file_path.write_text(
            '{"token_id": "123", "side": "buy", "size": "100", "price": "0.50"}\n'
            '{"token_id": "123", "side": "sell", "size": "50", "price": "0.60"}\n'
        )

        fills = load_fills_from_file(file_path)
        assert len(fills) == 2
        assert fills[0].size == Decimal("100")


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


class TestLoadOrderbooksFromSnapshot:
    """Test loading orderbooks from collector snapshots."""

    def test_load_from_snapshot(self, tmp_path: Path):
        data = {
            "count": 1,
            "markets": [
                {
                    "condition_id": "market123",
                    "books": {
                        "yes": {
                            "bids": [{"price": "0.60", "size": "100"}],
                            "asks": [{"price": "0.65", "size": "150"}],
                        },
                        "no": {
                            "bids": [{"price": "0.35", "size": "200"}],
                            "asks": [{"price": "0.40", "size": "100"}],
                        },
                    },
                }
            ],
        }
        file_path = tmp_path / "snapshot.json"
        file_path.write_text(json.dumps(data))

        books = load_orderbooks_from_snapshot(file_path)
        assert len(books) == 2
        assert "market123_yes" in books
        assert "market123_no" in books
        assert books["market123_yes"].bids[0].price == Decimal("0.60")

    def test_load_empty_snapshot(self, tmp_path: Path):
        data = {"count": 0, "markets": []}
        file_path = tmp_path / "snapshot.json"
        file_path.write_text(json.dumps(data))

        books = load_orderbooks_from_snapshot(file_path)
        assert len(books) == 0


class TestSaveDailySummary:
    """Test daily summary persistence."""

    def test_save_daily_summary(self, tmp_path: Path):
        report = PnLReport(
            total_fills=10,
            unique_tokens=2,
            realized_pnl=Decimal("100"),
        )

        out_path = save_daily_summary(report, out_dir=tmp_path, date=datetime(2024, 1, 15))

        assert out_path.exists()
        assert out_path.name == "pnl_2024-01-15.json"

        data = json.loads(out_path.read_text())
        assert data["summary"]["total_fills"] == 10

    def test_save_daily_summary_creates_dir(self, tmp_path: Path):
        report = PnLReport(total_fills=5, unique_tokens=1)
        nested_dir = tmp_path / "nested" / "pnl"

        out_path = save_daily_summary(report, out_dir=nested_dir)

        assert nested_dir.exists()
        assert out_path.exists()

    def test_save_daily_summary_default_date(self, tmp_path: Path):
        from datetime import timezone

        report = PnLReport(total_fills=5, unique_tokens=1)

        out_path = save_daily_summary(report, out_dir=tmp_path)

        assert out_path.exists()
        # Filename should be today's date (UTC, since that's what the implementation uses)
        today_str = datetime.now(timezone.utc).strftime("pnl_%Y-%m-%d.json")
        assert out_path.name == today_str
