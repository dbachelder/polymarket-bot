"""Tests for PnL loop module."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from polymarket.pnl_loop import (
    get_latest_pnl_summary,
    pnl_health_check,
    run_pnl_verification,
)


class TestGetLatestPnlSummary:
    """Test getting latest PnL summary."""

    def test_empty_directory(self, tmp_path: Path) -> None:
        """Empty directory returns not found."""
        pnl_dir = tmp_path / "pnl"
        result = get_latest_pnl_summary(pnl_dir)

        assert result["exists"] is False
        assert result["latest_file"] is None

    def test_no_pnl_files(self, tmp_path: Path) -> None:
        """Directory exists but has no pnl files."""
        pnl_dir = tmp_path / "pnl"
        pnl_dir.mkdir()
        (pnl_dir / "other.txt").write_text("not a pnl file")

        result = get_latest_pnl_summary(pnl_dir)
        assert result["exists"] is True
        assert result["latest_file"] is None

    def test_finds_latest_file(self, tmp_path: Path) -> None:
        """Find the most recent PnL file."""
        pnl_dir = tmp_path / "pnl"
        pnl_dir.mkdir()

        # Create files with different dates
        old_file = pnl_dir / "pnl_2024-01-14.json"
        old_file.write_text("{}")
        new_file = pnl_dir / "pnl_2024-01-15.json"
        new_file.write_text("{}")

        result = get_latest_pnl_summary(pnl_dir)

        assert result["exists"] is True
        assert result["latest_file"] == str(new_file)
        assert result["latest_date"] is not None


class TestPnlHealthCheck:
    """Test PnL health check."""

    def test_no_data(self, tmp_path: Path) -> None:
        """Health check with no data."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        result = pnl_health_check(data_dir, max_fills_age_seconds=3600, max_pnl_age_seconds=3600)

        assert result["healthy"] is False
        assert any("No fills file" in w for w in result["warnings"])
        assert any("No PnL summaries" in w for w in result["warnings"])

    def test_stale_fills(self, tmp_path: Path) -> None:
        """Health check with stale fills."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        fills_path = data_dir / "fills.jsonl"
        old_time = datetime.now(UTC) - timedelta(hours=48)
        fills_path.write_text(
            json.dumps(
                {
                    "token_id": "abc",
                    "side": "buy",
                    "size": "100",
                    "price": "0.5",
                    "fee": "0",
                    "timestamp": old_time.isoformat(),
                }
            )
            + "\n"
        )

        result = pnl_health_check(data_dir, max_fills_age_seconds=3600, max_pnl_age_seconds=3600)

        assert result["healthy"] is False
        assert result["fills"]["healthy"] is False
        assert any("stale" in w.lower() for w in result["warnings"])

    def test_fresh_data(self, tmp_path: Path) -> None:
        """Health check with fresh data."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        fills_path = data_dir / "fills.jsonl"
        now = datetime.now(UTC)
        fills_path.write_text(
            json.dumps(
                {
                    "token_id": "abc",
                    "side": "buy",
                    "size": "100",
                    "price": "0.5",
                    "fee": "0",
                    "timestamp": now.isoformat(),
                }
            )
            + "\n"
        )

        pnl_dir = data_dir / "pnl"
        pnl_dir.mkdir()
        today = now.strftime("%Y-%m-%d")
        (pnl_dir / f"pnl_{today}.json").write_text("{}")

        result = pnl_health_check(data_dir, max_fills_age_seconds=86400, max_pnl_age_seconds=86400)

        assert result["healthy"] is True
        assert result["fills"]["healthy"] is True
        assert result["pnl"]["healthy"] is True


class TestRunPnlVerification:
    """Test PnL verification run."""

    def test_no_fills_file(self, tmp_path: Path) -> None:
        """Verification fails if no fills file."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        result = run_pnl_verification(data_dir)

        assert result["success"] is False
        assert "not found" in result["error"].lower()

    def test_empty_fills(self, tmp_path: Path) -> None:
        """Verification handles empty fills."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        fills_path = data_dir / "fills.jsonl"
        fills_path.write_text("")

        result = run_pnl_verification(data_dir)

        assert result["success"] is False
        assert "no fills" in result["error"].lower()

    def test_successful_verification(self, tmp_path: Path) -> None:
        """Successful verification with fills."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # Create fills file
        fills_path = data_dir / "fills.jsonl"
        fills_path.write_text(
            json.dumps(
                {
                    "token_id": "abc_yes",
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
        )

        # Create a snapshot
        snapshot_path = data_dir / "snapshot_20240115_120000.json"
        snapshot_path.write_text(
            json.dumps(
                {
                    "generated_at": "2024-01-15T12:00:00+00:00",
                    "markets": [
                        {
                            "condition_id": "abc",
                            "books": {
                                "yes": {
                                    "bids": [{"price": "0.56", "size": "1000"}],
                                    "asks": [{"price": "0.57", "size": "1000"}],
                                }
                            },
                        }
                    ],
                }
            )
        )

        result = run_pnl_verification(
            data_dir=data_dir,
            snapshot_path=snapshot_path,
            starting_cash=Decimal("10000"),
        )

        assert result["success"] is True
        assert result["report"] is not None
        assert result["summary_path"] is not None
        assert Path(result["summary_path"]).exists()

    def test_uses_latest_pointer(self, tmp_path: Path) -> None:
        """Verification resolves snapshot pointer files."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # Create fills
        fills_path = data_dir / "fills.jsonl"
        fills_path.write_text(
            json.dumps(
                {
                    "token_id": "abc_yes",
                    "side": "buy",
                    "size": "100",
                    "price": "0.55",
                    "fee": "0.50",
                    "timestamp": "2024-01-15T12:00:00+00:00",
                }
            )
            + "\n"
        )

        # Create actual snapshot
        snapshot_path = data_dir / "snapshot_20240115_120000.json"
        snapshot_path.write_text(
            json.dumps(
                {
                    "generated_at": "2024-01-15T12:00:00+00:00",
                    "markets": [
                        {
                            "condition_id": "abc",
                            "books": {
                                "yes": {
                                    "bids": [{"price": "0.56", "size": "1000"}],
                                    "asks": [{"price": "0.57", "size": "1000"}],
                                }
                            },
                        }
                    ],
                }
            )
        )

        # Create pointer file
        latest_path = data_dir / "latest_15m.json"
        latest_path.write_text(json.dumps({"path": str(snapshot_path), "generated_at": ""}))

        result = run_pnl_verification(
            data_dir=data_dir,
            snapshot_path=latest_path,  # Pass pointer, should resolve
            starting_cash=Decimal("10000"),
        )

        assert result["success"] is True
