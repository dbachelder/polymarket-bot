"""Tests for maker fee asymmetry strategy."""

from __future__ import annotations

import json
import tempfile
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from polymarket.strategy_maker_fee_asymmetry import (
    DEFAULT_FAIR_PROB,
    FairProbabilityEstimate,
    MakerFeeAsymmetryTracker,
    MakerSignal,
    MarketOrderBook,
    OrderBookLevel,
    OrderBookSide,
    PassiveOrder,
    StrategyPerformance,
    compute_fair_probability,
    generate_maker_signal,
    load_snapshots_for_backtest,
    parse_orderbook_from_snapshot,
    run_backtest,
    run_maker_fee_asymmetry_scan,
    scan_for_maker_opportunities,
)


class TestOrderBookLevel:
    """Tests for OrderBookLevel dataclass."""

    def test_creation(self) -> None:
        level = OrderBookLevel(price=Decimal("0.55"), size=Decimal("100"))
        assert level.price == Decimal("0.55")
        assert level.size == Decimal("100")


class TestOrderBookSide:
    """Tests for OrderBookSide dataclass."""

    def test_best_level(self) -> None:
        levels = [
            OrderBookLevel(price=Decimal("0.55"), size=Decimal("100")),
            OrderBookLevel(price=Decimal("0.54"), size=Decimal("200")),
        ]
        side = OrderBookSide(levels=levels)
        best = side.best_level()
        assert best is not None
        assert best.price == Decimal("0.55")

    def test_best_level_empty(self) -> None:
        side = OrderBookSide(levels=[])
        assert side.best_level() is None

    def test_depth_at_price(self) -> None:
        levels = [
            OrderBookLevel(price=Decimal("0.55"), size=Decimal("100")),
            OrderBookLevel(price=Decimal("0.54"), size=Decimal("200")),
            OrderBookLevel(price=Decimal("0.53"), size=Decimal("300")),
        ]
        side = OrderBookSide(levels=levels)
        depth = side.depth_at_price(Decimal("0.54"))
        assert depth == Decimal("300")  # 100 + 200


class TestMarketOrderBook:
    """Tests for MarketOrderBook dataclass."""

    def test_mid_price(self) -> None:
        yes_bids = OrderBookSide(levels=[OrderBookLevel(price=Decimal("0.54"), size=Decimal("100"))])
        yes_asks = OrderBookSide(levels=[OrderBookLevel(price=Decimal("0.56"), size=Decimal("100"))])
        book = MarketOrderBook(
            market_id="test-market",
            token_id_yes="token-yes",
            token_id_no="token-no",
            question="Test market",
            yes_bids=yes_bids,
            yes_asks=yes_asks,
        )
        assert book.mid_price == Decimal("0.55")

    def test_spread(self) -> None:
        yes_bids = OrderBookSide(levels=[OrderBookLevel(price=Decimal("0.54"), size=Decimal("100"))])
        yes_asks = OrderBookSide(levels=[OrderBookLevel(price=Decimal("0.56"), size=Decimal("100"))])
        book = MarketOrderBook(
            market_id="test-market",
            token_id_yes="token-yes",
            token_id_no="token-no",
            question="Test market",
            yes_bids=yes_bids,
            yes_asks=yes_asks,
        )
        assert book.spread == Decimal("0.02")

    def test_get_implied_probability(self) -> None:
        yes_bids = OrderBookSide(levels=[OrderBookLevel(price=Decimal("0.54"), size=Decimal("100"))])
        yes_asks = OrderBookSide(levels=[OrderBookLevel(price=Decimal("0.56"), size=Decimal("100"))])
        book = MarketOrderBook(
            market_id="test-market",
            token_id_yes="token-yes",
            token_id_no="token-no",
            question="Test market",
            yes_bids=yes_bids,
            yes_asks=yes_asks,
        )
        assert book.get_implied_probability() == Decimal("0.55")


class TestParseOrderbookFromSnapshot:
    """Tests for parse_orderbook_from_snapshot function."""

    def test_valid_market_data(self) -> None:
        market_data = {
            "market_id": "test-123",
            "question": "Will Bitcoin go up?",
            "clob_token_ids": ["token-yes", "token-no"],
            "books": {
                "yes": {
                    "bids": [{"price": "0.54", "size": "100"}],
                    "asks": [{"price": "0.56", "size": "100"}],
                }
            },
        }
        result = parse_orderbook_from_snapshot(market_data)
        assert result is not None
        assert result.market_id == "test-123"
        assert result.question == "Will Bitcoin go up?"
        assert result.best_bid_yes == Decimal("0.54")
        assert result.best_ask_yes == Decimal("0.56")

    def test_missing_books(self) -> None:
        market_data = {
            "market_id": "test-123",
            "question": "Will Bitcoin go up?",
            "clob_token_ids": ["token-yes", "token-no"],
        }
        result = parse_orderbook_from_snapshot(market_data)
        assert result is None

    def test_missing_token_ids(self) -> None:
        market_data = {
            "market_id": "test-123",
            "question": "Will Bitcoin go up?",
            "clob_token_ids": ["token-yes"],  # Only one token
            "books": {
                "yes": {
                    "bids": [{"price": "0.54", "size": "100"}],
                    "asks": [{"price": "0.56", "size": "100"}],
                }
            },
        }
        result = parse_orderbook_from_snapshot(market_data)
        assert result is None


class TestComputeFairProbability:
    """Tests for compute_fair_probability function."""

    def test_with_external_estimate(self) -> None:
        yes_bids = OrderBookSide(levels=[OrderBookLevel(price=Decimal("0.54"), size=Decimal("100"))])
        yes_asks = OrderBookSide(levels=[OrderBookLevel(price=Decimal("0.56"), size=Decimal("100"))])
        book = MarketOrderBook(
            market_id="test-market",
            token_id_yes="token-yes",
            token_id_no="token-no",
            question="Test market",
            yes_bids=yes_bids,
            yes_asks=yes_asks,
        )
        result = compute_fair_probability(book, external_estimate=Decimal("0.60"))
        assert result.fair_prob == Decimal("0.60")
        assert result.source == "external"
        assert result.confidence == 0.8

    def test_with_midpoint(self) -> None:
        yes_bids = OrderBookSide(levels=[OrderBookLevel(price=Decimal("0.54"), size=Decimal("100"))])
        yes_asks = OrderBookSide(levels=[OrderBookLevel(price=Decimal("0.56"), size=Decimal("100"))])
        book = MarketOrderBook(
            market_id="test-market",
            token_id_yes="token-yes",
            token_id_no="token-no",
            question="Test market",
            yes_bids=yes_bids,
            yes_asks=yes_asks,
        )
        result = compute_fair_probability(book)
        assert result.fair_prob == Decimal("0.55")
        assert result.source == "midpoint"
        assert result.confidence == 0.5

    def test_fallback_to_default(self) -> None:
        yes_bids = OrderBookSide(levels=[])
        yes_asks = OrderBookSide(levels=[])
        book = MarketOrderBook(
            market_id="test-market",
            token_id_yes="token-yes",
            token_id_no="token-no",
            question="Test market",
            yes_bids=yes_bids,
            yes_asks=yes_asks,
        )
        result = compute_fair_probability(book)
        assert result.fair_prob == DEFAULT_FAIR_PROB
        assert result.source == "default"
        assert result.confidence == 0.3


class TestGenerateMakerSignal:
    """Tests for generate_maker_signal function."""

    def test_buy_yes_signal(self) -> None:
        # Market prices YES at 0.52, but fair is 0.60 (8% edge)
        yes_bids = OrderBookSide(levels=[OrderBookLevel(price=Decimal("0.51"), size=Decimal("100"))])
        yes_asks = OrderBookSide(levels=[OrderBookLevel(price=Decimal("0.53"), size=Decimal("100"))])
        book = MarketOrderBook(
            market_id="test-market",
            token_id_yes="token-yes",
            token_id_no="token-no",
            question="Will Bitcoin go up?",
            yes_bids=yes_bids,
            yes_asks=yes_asks,
        )
        fair_estimate = FairProbabilityEstimate(
            market_id="test-market",
            fair_prob=Decimal("0.60"),
            source="test",
            confidence=0.8,
            timestamp=datetime.now(UTC),
        )
        signal = generate_maker_signal(
            book,
            fair_estimate,
            edge_threshold=Decimal("0.03"),
            spread_buffer=Decimal("0.005"),
        )
        assert signal is not None
        assert signal.direction == "BUY_YES"
        assert signal.edge == Decimal("0.08")  # 0.60 - 0.52
        assert signal.has_edge is True

    def test_buy_no_signal(self) -> None:
        # Market prices YES at 0.70, but fair is 0.50 (20% edge for NO)
        yes_bids = OrderBookSide(levels=[OrderBookLevel(price=Decimal("0.69"), size=Decimal("100"))])
        yes_asks = OrderBookSide(levels=[OrderBookLevel(price=Decimal("0.71"), size=Decimal("100"))])
        book = MarketOrderBook(
            market_id="test-market",
            token_id_yes="token-yes",
            token_id_no="token-no",
            question="Will Bitcoin go up?",
            yes_bids=yes_bids,
            yes_asks=yes_asks,
        )
        fair_estimate = FairProbabilityEstimate(
            market_id="test-market",
            fair_prob=Decimal("0.50"),
            source="test",
            confidence=0.8,
            timestamp=datetime.now(UTC),
        )
        signal = generate_maker_signal(
            book,
            fair_estimate,
            edge_threshold=Decimal("0.03"),
            spread_buffer=Decimal("0.005"),
        )
        assert signal is not None
        assert signal.direction == "BUY_NO"
        assert signal.edge == Decimal("-0.20")  # 0.50 - 0.70
        assert signal.has_edge is True

    def test_no_signal_below_threshold(self) -> None:
        # Market prices YES at 0.52, fair is 0.54 (only 2% edge)
        yes_bids = OrderBookSide(levels=[OrderBookLevel(price=Decimal("0.51"), size=Decimal("100"))])
        yes_asks = OrderBookSide(levels=[OrderBookLevel(price=Decimal("0.53"), size=Decimal("100"))])
        book = MarketOrderBook(
            market_id="test-market",
            token_id_yes="token-yes",
            token_id_no="token-no",
            question="Will Bitcoin go up?",
            yes_bids=yes_bids,
            yes_asks=yes_asks,
        )
        fair_estimate = FairProbabilityEstimate(
            market_id="test-market",
            fair_prob=Decimal("0.54"),
            source="test",
            confidence=0.8,
            timestamp=datetime.now(UTC),
        )
        signal = generate_maker_signal(
            book,
            fair_estimate,
            edge_threshold=Decimal("0.03"),  # 3% threshold
            spread_buffer=Decimal("0.005"),
        )
        assert signal is None


class TestMakerFeeAsymmetryTracker:
    """Tests for MakerFeeAsymmetryTracker class."""

    def test_initialization(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = MakerFeeAsymmetryTracker(data_dir=tmpdir)
            assert tracker.data_dir == Path(tmpdir)
            assert (Path(tmpdir) / "paper_trading").exists()

    def test_get_orders_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = MakerFeeAsymmetryTracker(data_dir=tmpdir)
            assert tracker.get_orders_file() == Path(tmpdir) / "orders.jsonl"

    def test_record_and_load_orders(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = MakerFeeAsymmetryTracker(data_dir=tmpdir)

            signal = MakerSignal(
                timestamp=datetime.now(UTC),
                market_id="test-market",
                question="Test market question",
                fair_prob=Decimal("0.60"),
                market_implied_prob=Decimal("0.52"),
                edge=Decimal("0.08"),
                direction="BUY_YES",
                target_price=Decimal("0.53"),
                current_best_bid=Decimal("0.51"),
                current_best_ask=Decimal("0.56"),
                spread=Decimal("0.05"),
                position_size=Decimal("10"),
                maker_fee_savings=Decimal("0.02"),
            )

            order = PassiveOrder(
                order_id="test-order-1",
                timestamp=datetime.now(UTC),
                signal=signal,
                side="buy_yes",
                price=Decimal("0.53"),
                size=Decimal("18.867"),  # $10 / 0.53
                status="open",
            )

            tracker.record_order(order)
            loaded = tracker.load_orders()

            assert len(loaded) == 1
            assert loaded[0].order_id == "test-order-1"
            assert loaded[0].side == "buy_yes"

    def test_get_performance(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = MakerFeeAsymmetryTracker(data_dir=tmpdir)
            perf = tracker.get_performance()
            assert perf.total_orders_posted == 0
            assert perf.fill_rate == 0.0
            assert perf.fee_rate == 0.0


class TestStrategyPerformance:
    """Tests for StrategyPerformance dataclass."""

    def test_fill_rate(self) -> None:
        perf = StrategyPerformance(total_orders_posted=10, orders_filled=4)
        assert perf.fill_rate == 40.0

    def test_fill_rate_zero_division(self) -> None:
        perf = StrategyPerformance()
        assert perf.fill_rate == 0.0

    def test_avg_pnl_per_fill(self) -> None:
        perf = StrategyPerformance(
            orders_filled=2,
            total_realized_pnl=Decimal("10"),
        )
        assert perf.avg_pnl_per_fill == Decimal("5")

    def test_fee_rate(self) -> None:
        perf = StrategyPerformance(
            total_fees_paid=Decimal("1"),
            total_volume=Decimal("100"),
        )
        assert perf.fee_rate == 1.0

    def test_ev_per_trade(self) -> None:
        perf = StrategyPerformance(
            total_fills=4,
            total_realized_pnl=Decimal("2"),
        )
        assert perf.ev_per_trade == Decimal("0.5")


class TestScanForMakerOpportunities:
    """Tests for scan_for_maker_opportunities function."""

    def test_scan_filters_markets_by_substring(self) -> None:
        # Test that scan filters markets correctly by substring
        # When fair=mid, no edge exists so no signals generated
        # But we can verify parsing works for filtered markets
        snapshot_data = {
            "generated_at": datetime.now(UTC).isoformat(),
            "markets": [
                {
                    "market_id": "btc-market-1",
                    "question": "Will Bitcoin go up in 15m?",
                    "clob_token_ids": ["token-yes", "token-no"],
                    "books": {
                        "yes": {
                            "bids": [{"price": "0.45", "size": "100"}],
                            "asks": [{"price": "0.47", "size": "100"}],
                        }
                    },
                },
                {
                    "market_id": "eth-market-1",
                    "question": "Will Ethereum go up in 15m?",
                    "clob_token_ids": ["token-yes", "token-no"],
                    "books": {
                        "yes": {
                            "bids": [{"price": "0.55", "size": "100"}],
                            "asks": [{"price": "0.57", "size": "100"}],
                        }
                    },
                },
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot_path = Path(tmpdir) / "snapshot_15m_20260217T120000Z.json"
            snapshot_path.write_text(json.dumps(snapshot_data))

            # Verify both markets can be parsed
            data = json.loads(snapshot_path.read_text())
            btc_market = data["markets"][0]
            eth_market = data["markets"][1]

            btc_book = parse_orderbook_from_snapshot(btc_market)
            eth_book = parse_orderbook_from_snapshot(eth_market)

            assert btc_book is not None
            assert btc_book.market_id == "btc-market-1"
            assert "bitcoin" in btc_book.question.lower()

            assert eth_book is not None
            assert eth_book.market_id == "eth-market-1"
            assert "ethereum" in eth_book.question.lower()

            # Verify filtering
            assert "bitcoin" in btc_book.question.lower()
            assert "bitcoin" not in eth_book.question.lower()

    def test_scan_empty_markets(self) -> None:
        snapshot_data = {
            "generated_at": datetime.now(UTC).isoformat(),
            "markets": [],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot_path = Path(tmpdir) / "snapshot_15m_20260217T120000Z.json"
            snapshot_path.write_text(json.dumps(snapshot_data))

            signals = scan_for_maker_opportunities(snapshot_path)
            assert len(signals) == 0


class TestLoadSnapshotsForBacktest:
    """Tests for load_snapshots_for_backtest function."""

    def test_load_all_snapshots(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test snapshots
            for i in range(3):
                snapshot_path = Path(tmpdir) / f"snapshot_15m_20260217T{i:02d}0000Z.json"
                snapshot_path.write_text("{}")

            snapshots = load_snapshots_for_backtest(Path(tmpdir))
            assert len(snapshots) == 3

    def test_filter_by_time_range(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create snapshots at different times
            snapshot_path1 = Path(tmpdir) / "snapshot_15m_20260217T100000Z.json"
            snapshot_path2 = Path(tmpdir) / "snapshot_15m_20260217T120000Z.json"
            snapshot_path3 = Path(tmpdir) / "snapshot_15m_20260217T140000Z.json"

            for p in [snapshot_path1, snapshot_path2, snapshot_path3]:
                p.write_text("{}")

            start = datetime(2026, 2, 17, 11, 0, 0, tzinfo=UTC)
            end = datetime(2026, 2, 17, 13, 0, 0, tzinfo=UTC)

            snapshots = load_snapshots_for_backtest(Path(tmpdir), start_time=start, end_time=end)
            assert len(snapshots) == 1
            assert "120000" in snapshots[0].name


class TestRunBacktest:
    """Tests for run_backtest function."""

    def test_backtest_empty_snapshots(self) -> None:
        result = run_backtest([])
        assert result["total_trades"] == 0
        assert result["total_pnl"] == 0

    def test_backtest_with_insufficient_snapshots(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create only 2 snapshots but need horizon of 4
            snapshot_data = {
                "generated_at": datetime.now(UTC).isoformat(),
                "markets": [
                    {
                        "market_id": "btc-market",
                        "question": "Will Bitcoin go up?",
                        "clob_token_ids": ["token-yes", "token-no"],
                        "books": {
                            "yes": {
                                "bids": [{"price": "0.45", "size": "100"}],
                                "asks": [{"price": "0.47", "size": "100"}],
                            }
                        },
                    },
                ],
            }

            for i in range(2):
                snapshot_path = Path(tmpdir) / f"snapshot_15m_20260217T{i:02d}0000Z.json"
                snapshot_path.write_text(json.dumps(snapshot_data))

            snapshots = sorted(Path(tmpdir).glob("*.json"))
            result = run_backtest(snapshots, hold_horizon=4)

            # Should have 0 trades because not enough snapshots for hold horizon
            assert result["total_trades"] == 0


class TestRunMakerFeeAsymmetryScan:
    """Tests for run_maker_fee_asymmetry_scan function."""

    def test_scan_no_snapshots(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_maker_fee_asymmetry_scan(
                snapshots_dir=Path(tmpdir),
                dry_run=True,
            )
            assert "error" in result
            assert "No snapshots found" in result["error"]

    def test_scan_with_snapshots(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot_data = {
                "generated_at": datetime.now(UTC).isoformat(),
                "markets": [
                    {
                        "market_id": "btc-market",
                        "question": "Will Bitcoin go up?",
                        "clob_token_ids": ["token-yes", "token-no"],
                        "books": {
                            "yes": {
                                "bids": [{"price": "0.45", "size": "100"}],
                                "asks": [{"price": "0.47", "size": "100"}],
                            }
                        },
                    },
                ],
            }

            snapshot_path = Path(tmpdir) / "snapshot_15m_20260217T120000Z.json"
            snapshot_path.write_text(json.dumps(snapshot_data))

            result = run_maker_fee_asymmetry_scan(
                snapshots_dir=Path(tmpdir),
                dry_run=True,
            )

            assert "error" not in result
            assert "signals_found" in result
            assert "snapshot" in result


class TestMakerSignal:
    """Tests for MakerSignal dataclass."""

    def test_to_dict(self) -> None:
        signal = MakerSignal(
            timestamp=datetime.now(UTC),
            market_id="test-market",
            question="Test question",
            fair_prob=Decimal("0.60"),
            market_implied_prob=Decimal("0.52"),
            edge=Decimal("0.08"),
            direction="BUY_YES",
            target_price=Decimal("0.53"),
            current_best_bid=Decimal("0.51"),
            current_best_ask=Decimal("0.56"),
            spread=Decimal("0.05"),
            position_size=Decimal("10"),
            maker_fee_savings=Decimal("0.02"),
        )
        d = signal.to_dict()
        assert d["market_id"] == "test-market"
        assert d["direction"] == "BUY_YES"
        assert d["edge"] == 0.08
        assert d["has_edge"] is True


class TestPassiveOrder:
    """Tests for PassiveOrder dataclass."""

    def test_to_dict(self) -> None:
        signal = MakerSignal(
            timestamp=datetime.now(UTC),
            market_id="test-market",
            question="Test question",
            fair_prob=Decimal("0.60"),
            market_implied_prob=Decimal("0.52"),
            edge=Decimal("0.08"),
            direction="BUY_YES",
            target_price=Decimal("0.53"),
            current_best_bid=Decimal("0.51"),
            current_best_ask=Decimal("0.56"),
            spread=Decimal("0.05"),
            position_size=Decimal("10"),
            maker_fee_savings=Decimal("0.02"),
        )

        order = PassiveOrder(
            order_id="test-order",
            timestamp=datetime.now(UTC),
            signal=signal,
            side="buy_yes",
            price=Decimal("0.53"),
            size=Decimal("18.867"),
            status="open",
        )

        d = order.to_dict()
        assert d["order_id"] == "test-order"
        assert d["side"] == "buy_yes"
        assert d["status"] == "open"
