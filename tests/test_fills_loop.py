"""Tests for fills_loop module."""

from __future__ import annotations

import json
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from polymarket.fills_loop import (
    check_fills_staleness,
    run_collect_fills_loop,
)


class TestCheckFillsStaleness:
    """Tests for check_fills_staleness function."""

    def test_no_fills_file(self):
        """Test staleness check when fills file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fills_path = Path(tmpdir) / "fills.jsonl"
            result = check_fills_staleness(fills_path, stale_alert_hours=6.0)

            assert result["exists"] is False
            assert result["total_fills"] == 0
            assert result["age_hours"] is None
            assert result["is_stale"] is False

    def test_fresh_fills(self):
        """Test staleness check when fills are fresh."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fills_path = Path(tmpdir) / "fills.jsonl"

            # Create a fill from 1 hour ago
            fill = {
                "token_id": "test-token",
                "side": "buy",
                "size": "100",
                "price": "0.5",
                "timestamp": (datetime.now(UTC) - timedelta(hours=1)).isoformat(),
                "transaction_hash": "tx1",
                "market_slug": "test-market",
            }
            fills_path.write_text(json.dumps(fill) + "\n")

            result = check_fills_staleness(fills_path, stale_alert_hours=6.0)

            assert result["exists"] is True
            assert result["total_fills"] == 1
            assert result["age_hours"] is not None
            assert result["age_hours"] < 6.0
            assert result["is_stale"] is False

    def test_stale_fills(self):
        """Test staleness check when fills are stale."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fills_path = Path(tmpdir) / "fills.jsonl"

            # Create a fill from 10 hours ago
            fill = {
                "token_id": "test-token",
                "side": "buy",
                "size": "100",
                "price": "0.5",
                "timestamp": (datetime.now(UTC) - timedelta(hours=10)).isoformat(),
                "transaction_hash": "tx1",
                "market_slug": "test-market",
            }
            fills_path.write_text(json.dumps(fill) + "\n")

            result = check_fills_staleness(fills_path, stale_alert_hours=6.0)

            assert result["exists"] is True
            assert result["is_stale"] is True
            assert result["age_hours"] > 6.0

    def test_custom_stale_threshold(self):
        """Test staleness check with custom threshold."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fills_path = Path(tmpdir) / "fills.jsonl"

            # Create a fill from 3 hours ago
            fill = {
                "token_id": "test-token",
                "side": "buy",
                "size": "100",
                "price": "0.5",
                "timestamp": (datetime.now(UTC) - timedelta(hours=3)).isoformat(),
                "transaction_hash": "tx1",
                "market_slug": "test-market",
            }
            fills_path.write_text(json.dumps(fill) + "\n")

            # With 6h threshold, should not be stale
            result = check_fills_staleness(fills_path, stale_alert_hours=6.0)
            assert result["is_stale"] is False

            # With 2h threshold, should be stale
            result = check_fills_staleness(fills_path, stale_alert_hours=2.0)
            assert result["is_stale"] is True


class TestRunCollectFillsLoop:
    """Tests for run_collect_fills_loop function."""

    @patch("polymarket.fills_loop.collect_fills")
    @patch("polymarket.fills_loop.time.sleep")
    @patch("polymarket.fills_loop.time.time")
    def test_single_iteration(self, mock_time, mock_sleep, mock_collect_fills):
        """Test that loop runs at least one iteration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            fills_path = data_dir / "fills.jsonl"

            # Mock time to control the loop timing
            mock_time.side_effect = [0.0, 1.0]  # started, elapsed

            # Mock collect_fills to return a result
            mock_collect_fills.return_value = {
                "total_appended": 5,
                "account_fills": 3,
                "paper_fills": 2,
                "duplicates_skipped": 0,
            }

            # Run loop with a timeout to prevent infinite loop
            with pytest.raises(StopIteration):
                # Use a side effect to break out of the infinite loop
                mock_sleep.side_effect = StopIteration()
                run_collect_fills_loop(
                    data_dir=data_dir,
                    fills_path=fills_path,
                    interval_seconds=300.0,
                )

            # Verify collect_fills was called
            mock_collect_fills.assert_called_once()

    @patch("polymarket.fills_loop.collect_fills")
    @patch("polymarket.fills_loop.time.sleep")
    @patch("polymarket.fills_loop.time.time")
    def test_respects_interval(self, mock_time, mock_sleep, mock_collect_fills):
        """Test that loop respects interval timing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)

            # Mock time: started at 0, ended at 10 (10s elapsed)
            mock_time.side_effect = [0.0, 10.0]

            mock_collect_fills.return_value = {
                "total_appended": 0,
                "account_fills": 0,
                "paper_fills": 0,
                "duplicates_skipped": 0,
            }

            # Should sleep for interval - elapsed = 300 - 10 = 290s
            # Plus some jitter (0-15s for 300s interval at 5%)
            with pytest.raises(StopIteration):
                mock_sleep.side_effect = StopIteration()
                run_collect_fills_loop(
                    data_dir=data_dir,
                    interval_seconds=300.0,
                )

            # Verify sleep was called with approximately the right value
            mock_sleep.assert_called_once()
            sleep_arg = mock_sleep.call_args[0][0]
            assert sleep_arg >= 290.0  # At least interval - elapsed
            assert sleep_arg < 310.0  # Interval - elapsed + max jitter

    @patch("polymarket.fills_loop.collect_fills")
    @patch("polymarket.fills_loop.time.sleep")
    @patch("polymarket.fills_loop.time.time")
    def test_passes_correct_flags(self, mock_time, mock_sleep, mock_collect_fills):
        """Test that account/paper flags are passed correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)

            mock_time.side_effect = [0.0, 1.0]
            mock_collect_fills.return_value = {
                "total_appended": 0,
                "account_fills": 0,
                "paper_fills": 0,
                "duplicates_skipped": 0,
            }

            with pytest.raises(StopIteration):
                mock_sleep.side_effect = StopIteration()
                run_collect_fills_loop(
                    data_dir=data_dir,
                    include_account=True,
                    include_paper=False,
                )

            # Verify collect_fills was called with correct flags
            call_kwargs = mock_collect_fills.call_args[1]
            assert call_kwargs["include_account"] is True
            assert call_kwargs["include_paper"] is False

    @patch("polymarket.fills_loop.collect_fills")
    @patch("polymarket.fills_loop.time.sleep")
    @patch("polymarket.fills_loop.time.time")
    @patch("polymarket.fills_loop._send_openclaw_notification")
    def test_stale_alert_triggered(
        self, mock_notify, mock_time, mock_sleep, mock_collect_fills
    ):
        """Test that stale alert is triggered when fills are old."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            fills_path = data_dir / "fills.jsonl"

            # Create a stale fill (10 hours ago)
            fill = {
                "token_id": "test-token",
                "side": "buy",
                "size": "100",
                "price": "0.5",
                "timestamp": (datetime.now(UTC) - timedelta(hours=10)).isoformat(),
                "transaction_hash": "tx1",
                "market_slug": "test-market",
            }
            fills_path.parent.mkdir(parents=True, exist_ok=True)
            fills_path.write_text(json.dumps(fill) + "\n")

            mock_time.side_effect = [0.0, 1.0]
            mock_collect_fills.return_value = {
                "total_appended": 0,
                "account_fills": 0,
                "paper_fills": 0,
                "duplicates_skipped": 0,
            }

            with pytest.raises(StopIteration):
                mock_sleep.side_effect = StopIteration()
                run_collect_fills_loop(
                    data_dir=data_dir,
                    fills_path=fills_path,
                    stale_alert_hours=6.0,
                )

            # Verify notification was sent
            mock_notify.assert_called_once()
            notification_msg = mock_notify.call_args[0][0]
            assert "stale" in notification_msg.lower() or "FILLS STALE" in notification_msg

    @patch("polymarket.fills_loop.collect_fills")
    @patch("polymarket.fills_loop.time.sleep")
    @patch("polymarket.fills_loop.time.time")
    def test_handles_errors_gracefully(self, mock_time, mock_sleep, mock_collect_fills):
        """Test that loop continues after errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)

            mock_time.side_effect = [0.0, 1.0]
            mock_collect_fills.side_effect = Exception("API error")

            # Should not raise - should catch and log the error
            with pytest.raises(StopIteration):
                mock_sleep.side_effect = StopIteration()
                run_collect_fills_loop(
                    data_dir=data_dir,
                    interval_seconds=300.0,
                )

            # Verify error was caught and loop attempted to continue
            mock_collect_fills.assert_called_once()

    @patch("polymarket.fills_loop.collect_fills")
    @patch("polymarket.fills_loop.time.sleep")
    @patch("polymarket.fills_loop.time.time")
    def test_custom_callback_for_stale_alert(self, mock_time, mock_sleep, mock_collect_fills):
        """Test that custom stale alert callback is used."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            fills_path = data_dir / "fills.jsonl"

            # Create a stale fill
            fill = {
                "token_id": "test-token",
                "side": "buy",
                "size": "100",
                "price": "0.5",
                "timestamp": (datetime.now(UTC) - timedelta(hours=10)).isoformat(),
                "transaction_hash": "tx1",
                "market_slug": "test-market",
            }
            fills_path.parent.mkdir(parents=True, exist_ok=True)
            fills_path.write_text(json.dumps(fill) + "\n")

            mock_time.side_effect = [0.0, 1.0]
            mock_collect_fills.return_value = {
                "total_appended": 0,
                "account_fills": 0,
                "paper_fills": 0,
                "duplicates_skipped": 0,
            }

            custom_callback = MagicMock()

            with pytest.raises(StopIteration):
                mock_sleep.side_effect = StopIteration()
                run_collect_fills_loop(
                    data_dir=data_dir,
                    fills_path=fills_path,
                    stale_alert_hours=6.0,
                    on_stale_alert=custom_callback,
                )

            # Verify custom callback was called instead of default
            custom_callback.assert_called_once()
