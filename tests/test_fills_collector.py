"""Tests for fills collector module."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from polymarket.fills_collector import (
    append_fills,
    collect_fills,
    get_existing_tx_hashes,
    get_fills_summary,
    get_last_fill_timestamp,
    load_paper_fills,
)
from polymarket.pnl import Fill


class TestFillPersistence:
    """Test fill loading and persistence."""

    def test_load_paper_fills_empty(self, tmp_path: Path) -> None:
        """Loading from non-existent file returns empty list."""
        fills_path = tmp_path / "nonexistent.jsonl"
        fills = load_paper_fills(fills_path)
        assert fills == []

    def test_load_paper_fills(self, tmp_path: Path) -> None:
        """Load fills from JSONL file."""
        fills_path = tmp_path / "fills.jsonl"
        fills_path.write_text(
            json.dumps(
                {
                    "token_id": "abc123",
                    "side": "buy",
                    "size": "100",
                    "price": "0.55",
                    "fee": "0.50",
                    "timestamp": "2024-01-15T12:00:00+00:00",
                    "transaction_hash": "0xabc",
                    "market_slug": "test-market",
                }
            )
            + "\n"
            + json.dumps(
                {
                    "token_id": "def456",
                    "side": "sell",
                    "size": "50",
                    "price": "0.60",
                    "fee": "0.25",
                    "timestamp": "2024-01-15T13:00:00+00:00",
                    "transaction_hash": "0xdef",
                    "market_slug": "test-market-2",
                }
            )
            + "\n"
        )

        fills = load_paper_fills(fills_path)
        assert len(fills) == 2

        assert fills[0].token_id == "abc123"
        assert fills[0].side == "buy"
        assert fills[0].size == Decimal("100")
        assert fills[0].price == Decimal("0.55")

        assert fills[1].token_id == "def456"
        assert fills[1].side == "sell"

    def test_get_existing_tx_hashes(self, tmp_path: Path) -> None:
        """Extract transaction hashes from fills file."""
        fills_path = tmp_path / "fills.jsonl"
        fills_path.write_text(
            json.dumps(
                {
                    "token_id": "abc",
                    "side": "buy",
                    "size": "100",
                    "price": "0.5",
                    "fee": "0",
                    "timestamp": "2024-01-15T12:00:00+00:00",
                    "transaction_hash": "0xabc",
                }
            )
            + "\n"
            + json.dumps(
                {
                    "token_id": "def",
                    "side": "sell",
                    "size": "50",
                    "price": "0.6",
                    "fee": "0",
                    "timestamp": "2024-01-15T13:00:00+00:00",
                    "transaction_hash": "0xdef",
                }
            )
            + "\n"
            + json.dumps(
                {
                    "token_id": "ghi",
                    "side": "buy",
                    "size": "25",
                    "price": "0.7",
                    "fee": "0",
                    "timestamp": "2024-01-15T14:00:00+00:00",
                    # No transaction_hash
                }
            )
            + "\n"
        )

        hashes = get_existing_tx_hashes(fills_path)
        assert hashes == {"0xabc", "0xdef"}

    def test_get_existing_tx_hashes_empty(self, tmp_path: Path) -> None:
        """Empty file returns empty set."""
        fills_path = tmp_path / "fills.jsonl"
        hashes = get_existing_tx_hashes(fills_path)
        assert hashes == set()

    def test_append_fills(self, tmp_path: Path) -> None:
        """Append fills to file."""
        fills_path = tmp_path / "fills.jsonl"

        fill1 = Fill(
            token_id="abc",
            side="buy",
            size=Decimal("100"),
            price=Decimal("0.5"),
            fee=Decimal("0.5"),
            timestamp="2024-01-15T12:00:00+00:00",
            transaction_hash="0xabc",
            market_slug="test-market",
        )
        fill2 = Fill(
            token_id="def",
            side="sell",
            size=Decimal("50"),
            price=Decimal("0.6"),
            fee=Decimal("0.25"),
            timestamp="2024-01-15T13:00:00+00:00",
            transaction_hash="0xdef",
            market_slug="test-market",
        )

        count = append_fills([fill1, fill2], fills_path)
        assert count == 2

        # Verify file contents
        lines = fills_path.read_text().strip().split("\n")
        assert len(lines) == 2

        data1 = json.loads(lines[0])
        assert data1["token_id"] == "abc"
        assert data1["side"] == "buy"
        assert data1["size"] == "100"

        data2 = json.loads(lines[1])
        assert data2["token_id"] == "def"
        assert data2["side"] == "sell"

    def test_append_fills_empty(self, tmp_path: Path) -> None:
        """Appending empty list returns 0."""
        fills_path = tmp_path / "fills.jsonl"
        count = append_fills([], fills_path)
        assert count == 0
        assert not fills_path.exists()


class TestFillsSummary:
    """Test fills summary functions."""

    def test_get_fills_summary_empty(self, tmp_path: Path) -> None:
        """Summary for non-existent file."""
        fills_path = tmp_path / "fills.jsonl"
        summary = get_fills_summary(fills_path)

        assert summary["exists"] is False
        assert summary["total_fills"] == 0

    def test_get_fills_summary(self, tmp_path: Path) -> None:
        """Get summary of fills file."""
        fills_path = tmp_path / "fills.jsonl"
        now = datetime.now(UTC)

        fills_path.write_text(
            json.dumps(
                {
                    "token_id": "abc123",
                    "side": "buy",
                    "size": "100",
                    "price": "0.55",
                    "fee": "0.50",
                    "timestamp": (now - timedelta(hours=2)).isoformat(),
                    "transaction_hash": "0xabc",
                    "market_slug": "market-1",
                }
            )
            + "\n"
            + json.dumps(
                {
                    "token_id": "def456",
                    "side": "sell",
                    "size": "50",
                    "price": "0.60",
                    "fee": "0.25",
                    "timestamp": (now - timedelta(hours=1)).isoformat(),
                    "transaction_hash": "0xdef",
                    "market_slug": "market-2",
                }
            )
            + "\n"
            + json.dumps(
                {
                    "token_id": "abc123",  # Same token
                    "side": "buy",
                    "size": "25",
                    "price": "0.65",
                    "fee": "0.10",
                    "timestamp": now.isoformat(),
                    "transaction_hash": "0xghi",
                    "market_slug": "market-1",  # Same market
                }
            )
            + "\n"
        )

        summary = get_fills_summary(fills_path)

        assert summary["exists"] is True
        assert summary["total_fills"] == 3
        assert summary["unique_tokens"] == 2  # abc123, def456
        assert summary["unique_markets"] == 2  # market-1, market-2
        assert summary["last_fill_at"] is not None
        assert summary["age_seconds"] is not None

    def test_get_last_fill_timestamp(self, tmp_path: Path) -> None:
        """Extract last fill timestamp."""
        fills_path = tmp_path / "fills.jsonl"
        now = datetime.now(UTC)

        fills_path.write_text(
            json.dumps(
                {
                    "token_id": "abc",
                    "timestamp": (now - timedelta(hours=2)).isoformat(),
                }
            )
            + "\n"
            + json.dumps(
                {
                    "token_id": "def",
                    "timestamp": now.isoformat(),
                }
            )
            + "\n"
        )

        last_ts = get_last_fill_timestamp(fills_path)
        assert last_ts is not None
        # Allow small difference due to rounding
        assert abs((last_ts - now).total_seconds()) < 1

    def test_get_last_fill_timestamp_empty(self, tmp_path: Path) -> None:
        """Empty file returns None."""
        fills_path = tmp_path / "fills.jsonl"
        assert get_last_fill_timestamp(fills_path) is None


class TestCollectFills:
    """Test the collect_fills function."""

    def test_collect_fills_from_paper(self, tmp_path: Path) -> None:
        """Collect fills from paper trading."""
        paper_dir = tmp_path / "paper_trading"
        paper_dir.mkdir()
        paper_fills_path = paper_dir / "fills.jsonl"

        paper_fills_path.write_text(
            json.dumps(
                {
                    "token_id": "paper1",
                    "side": "buy",
                    "size": "100",
                    "price": "0.55",
                    "fee": "0.50",
                    "timestamp": "2024-01-15T12:00:00+00:00",
                    "transaction_hash": "0xpaper1",
                    "market_slug": "paper-market",
                }
            )
            + "\n"
        )

        fills_path = tmp_path / "fills.jsonl"

        result = collect_fills(
            fills_path=fills_path,
            paper_fills_path=paper_fills_path,
            include_account=False,
            include_paper=True,
        )

        assert result["paper_fills"] == 1
        assert result["account_fills"] == 0
        assert result["total_appended"] == 1
        assert fills_path.exists()

    def test_collect_fills_deduplication(self, tmp_path: Path) -> None:
        """Duplicate fills are skipped."""
        fills_path = tmp_path / "fills.jsonl"

        # Pre-populate with existing fill
        fills_path.write_text(
            json.dumps(
                {
                    "token_id": "existing",
                    "side": "buy",
                    "size": "100",
                    "price": "0.5",
                    "fee": "0",
                    "timestamp": "2024-01-15T12:00:00+00:00",
                    "transaction_hash": "0xexisting",
                }
            )
            + "\n"
        )

        paper_dir = tmp_path / "paper_trading"
        paper_dir.mkdir()
        paper_fills_path = paper_dir / "fills.jsonl"

        paper_fills_path.write_text(
            json.dumps(
                {
                    "token_id": "existing",
                    "side": "buy",
                    "size": "100",
                    "price": "0.5",
                    "fee": "0",
                    "timestamp": "2024-01-15T12:00:00+00:00",
                    "transaction_hash": "0xexisting",  # Same tx hash
                }
            )
            + "\n"
            + json.dumps(
                {
                    "token_id": "new",
                    "side": "sell",
                    "size": "50",
                    "price": "0.6",
                    "fee": "0",
                    "timestamp": "2024-01-15T13:00:00+00:00",
                    "transaction_hash": "0xnew",
                }
            )
            + "\n"
        )

        result = collect_fills(
            fills_path=fills_path,
            paper_fills_path=paper_fills_path,
            include_account=False,
            include_paper=True,
        )

        assert result["paper_fills"] == 2
        assert result["duplicates_skipped"] == 1
        assert result["total_appended"] == 1

        # Verify only new fill was added
        lines = fills_path.read_text().strip().split("\n")
        assert len(lines) == 2


class TestCollectFillsWithOverlap:
    """Test collect_fills with overlap_hours parameter."""

    def test_uses_overlap_when_no_since_provided(self, tmp_path: Path) -> None:
        """When no since is provided, query from last_seen_ts - overlap."""
        fills_path = tmp_path / "fills.jsonl"
        # Create an existing fill from 5 hours ago
        last_ts = datetime.now(UTC) - timedelta(hours=5)
        fills_path.write_text(
            json.dumps({
                "token_id": "t1",
                "side": "buy",
                "size": "100",
                "price": "0.5",
                "fee": "0.1",
                "timestamp": last_ts.isoformat(),
                "transaction_hash": "tx1",
            })
            + "\n"
        )

        # Collect with 2h overlap - should query since last_ts - 2h
        result = collect_fills(
            fills_path=fills_path,
            include_account=False,
            include_paper=False,  # Skip both to just test overlap logic
            overlap_hours=2.0,
            check_health=True,
        )

        # Verify the query used overlap
        assert result["last_seen_ts"] == last_ts.isoformat()
        expected_since = last_ts - timedelta(hours=2)
        assert result["query_since"] == expected_since.isoformat()
        assert result["overlap_hours"] == 2.0

    def test_respects_explicit_since_parameter(self, tmp_path: Path) -> None:
        """When since is provided explicitly, use it instead of overlap."""
        fills_path = tmp_path / "fills.jsonl"
        explicit_since = datetime.now(UTC) - timedelta(hours=10)

        result = collect_fills(
            fills_path=fills_path,
            include_account=False,
            include_paper=False,
            since=explicit_since,
            overlap_hours=2.0,
        )

        # When since is provided, last_seen_ts should be None and query_since should be since
        assert result["last_seen_ts"] is None
        assert result["query_since"] == explicit_since.isoformat()

    def test_overlap_hours_in_result(self, tmp_path: Path) -> None:
        """overlap_hours is included in the result."""
        fills_path = tmp_path / "fills.jsonl"

        result = collect_fills(
            fills_path=fills_path,
            include_account=False,
            include_paper=False,
            overlap_hours=3.5,
        )

        assert result["overlap_hours"] == 3.5


class TestFillsAgeHealth:
    """Test fills age health check functionality."""

    def test_health_healthy_for_fresh_fills(self, tmp_path: Path) -> None:
        """Fresh fills (< 6h) should be healthy."""
        fills_path = tmp_path / "fills.jsonl"
        # Fill from 1 hour ago
        ts = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
        fills_path.write_text(
            json.dumps({"token_id": "t1", "side": "buy", "size": "100", "price": "0.5", "fee": "0.1", "timestamp": ts, "transaction_hash": "tx1"})
            + "\n"
        )

        from polymarket.fills_collector import _check_fills_age_health

        result = _check_fills_age_health(fills_path, stale_hours=6)
        assert result["healthy"] is True
        assert result["hours_since_last_fill"] < 6
        assert result["warning_emitted"] is False

    def test_health_stale_for_old_fills(self, tmp_path: Path) -> None:
        """Old fills (> 6h) should be stale."""
        fills_path = tmp_path / "fills.jsonl"
        # Fill from 10 hours ago
        ts = (datetime.now(UTC) - timedelta(hours=10)).isoformat()
        fills_path.write_text(
            json.dumps({"token_id": "t1", "side": "buy", "size": "100", "price": "0.5", "fee": "0.1", "timestamp": ts, "transaction_hash": "tx1"})
            + "\n"
        )

        from polymarket.fills_collector import _check_fills_age_health

        result = _check_fills_age_health(fills_path, stale_hours=6)
        assert result["healthy"] is False
        assert result["hours_since_last_fill"] > 6
        assert result["warning_emitted"] is True

    def test_health_unhealthy_for_missing_file(self, tmp_path: Path) -> None:
        """Missing file should be unhealthy."""
        fills_path = tmp_path / "fills.jsonl"

        from polymarket.fills_collector import _check_fills_age_health

        result = _check_fills_age_health(fills_path, stale_hours=6)
        assert result["healthy"] is False
        assert result["last_fill_at"] is None

    def test_health_in_collect_fills_result(self, tmp_path: Path) -> None:
        """Health check result is included in collect_fills output."""
        fills_path = tmp_path / "fills.jsonl"
        ts = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
        fills_path.write_text(
            json.dumps({"token_id": "t1", "side": "buy", "size": "100", "price": "0.5", "fee": "0.1", "timestamp": ts, "transaction_hash": "tx1"})
            + "\n"
        )

        result = collect_fills(
            fills_path=fills_path,
            include_account=False,
            include_paper=False,
            check_health=True,
            stale_hours=6,
        )

        assert "health" in result
        assert result["health"]["healthy"] is True

    def test_health_check_can_be_disabled(self, tmp_path: Path) -> None:
        """Health check can be disabled with check_health=False."""
        fills_path = tmp_path / "fills.jsonl"

        result = collect_fills(
            fills_path=fills_path,
            include_account=False,
            include_paper=False,
            check_health=False,
        )

        assert "health" not in result
